import itertools

import numpy as np
import torch
import torch.nn.functional as F
from ray.rllib.policy import Policy
from ray.rllib.policy.rnn_sequencing import chop_into_sequences
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.modelv2 import _unpack_obs
from ray.rllib.models.torch.torch_action_dist import TorchCategorical


from model import AgentNet, VDN, QMIX


def _unpack_obs_batch(obs_batch, space, unpack_state=False):
    unpacked = _unpack_obs(
        torch.tensor(obs_batch, dtype=torch.float32),
        space,
        tensorlib=torch
    )

    assert isinstance(unpacked[0], dict)
    assert 'obs' in unpacked[0]
    assert 'action_mask' in unpacked[0]

    observations = torch.stack(list(map(lambda x: x['obs'], unpacked)), 1)
    action_masks = torch.stack(list(map(lambda x: x['action_mask'], unpacked)), 1)

    if unpack_state:
        assert 'state' in unpacked[0]
        states = torch.stack(list(map(lambda x: x['state'], unpacked)), 1)
        return observations, action_masks, states[:,0,:]

    return observations, action_masks


def _append_agent_ids(obs, n_agents):
    return torch.cat((
        obs,
        F.one_hot(torch.arange(obs.size(1), device=obs.device), num_classes=n_agents).unsqueeze(0).repeat((obs.size(0),1,1)),
    ), -1)


class QmixPolicy(Policy):

    def __init__(self, observation_space, action_space, config):
        self.framework = "torch"
        super().__init__(observation_space, action_space, config)

        self._n_agents = len(observation_space.original_space.spaces)
        self._n_actions = action_space.spaces[0].n

        self.exploration = self._create_exploration()

        obs_dim = observation_space.original_space.spaces[0]['obs'].shape[0] + self._n_agents
        state_dim = observation_space.original_space.spaces[0]['state'].shape[0]

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.agent_net = AgentNet(obs_dim, self._n_actions, config['agent_hidden_size']).to(self.device)
        self.target_agent_net = AgentNet(obs_dim, self._n_actions, config['agent_hidden_size']).to(self.device)

        if config['mix_type'] == 'VDN':
            self.mix_net = VDN()
            self.target_mix_net = VDN()
        if config['mix_type'] == 'QMIX':
            self.mix_net, self.target_mix_net = [
                QMIX(
                    state_dim=state_dim,
                    n_agents=self._n_agents,
                    hidden_dim=config['mix_hidden_size'],
                    hypernet_hidden_dim=config['mix_hypernet_hidden_size']
                ).to(self.device)
                for _ in range(2)
            ]
        else:
            raise ValueError('Invalid mix net type: %s' % config['mix_type'])

        self.update_target() # Sync target networks

        net_params = itertools.chain(self.agent_net.parameters(), self.mix_net.parameters())
        if config['optimizer']['type'] == 'adam':
            self.opt = torch.optim.Adam(net_params, lr=config['optimizer']['lr'])
        elif config['optimizer']['type'] == 'rms_prop':
            self.opt = torch.optim.RMSprop(net_params, lr=config['optimizer']['lr'])
        else:
            raise ValueError('Invalid optimizer type')

    def get_initial_state(self):
        return [np.zeros((self._n_agents, self.agent_net._hidden_dim), dtype=np.float32)]

    def compute_actions(
        self,
        obs_batch,
        state_batches,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        explore=None,
        timestep=None,
        **kwargs):
        '''
        return:
            actions - Tuple(n_agents,)
        '''
        with torch.no_grad():
            observations, action_mask = _unpack_obs_batch(obs_batch, self.observation_space.original_space)
            observations, action_mask = observations.to(self.device), action_mask.to(self.device)
            state_batches = torch.tensor(state_batches[0], dtype=observations.dtype, device=self.device)

            q, rnn_states = self.agent_net(_append_agent_ids(observations, self._n_agents), state_batches)

            masked_q = q.clone()
            masked_q[action_mask < 1] = -np.inf
            masked_q = masked_q.view((q.size(0) * q.size(1), q.size(2)))

            actions, _ = self.exploration.get_exploration_action(
                action_distribution=TorchCategorical(masked_q),
                explore=explore,
                timestep=timestep,
            )

            actions = actions.view(q.shape[:-1]).cpu().numpy()
            rnn_states = rnn_states.cpu().numpy()

        return tuple(actions.T), [rnn_states], {}

    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        if sample_batch['dones'][-1]:
            info = sample_batch['infos'][-1]['_group_info']
            won = list(map(lambda x: x['battle_won'], filter(lambda x: 'battle_won' in x, info)))
            if len(won) > 0:
                episode.custom_metrics["win"] = int(any(won))
        else:
            print("Episode not done!")

        return sample_batch

    def _compute_loss(self, seq_lens, obs, action_mask, states, actions, next_obs, next_action_mask, next_states, dones, rewards):
        B, T = obs.size(0), obs.size(1)

        # Unroll & compute target Q values
        with torch.no_grad():
            rnn_state = torch.zeros((B, self._n_agents, self.agent_net._hidden_dim), dtype=obs.dtype, device=obs.device)
            _, rnn_state = self.target_agent_net(_append_agent_ids(obs[:,0].contiguous(), self._n_agents), rnn_state)

            q_target = []
            for t in range(T):
                _q, rnn_state = self.target_agent_net(_append_agent_ids(next_obs[:,t].contiguous(), self._n_agents), rnn_state)
                q_target.append(_q)

            q_target = torch.stack(q_target, 1)
            q_target[next_action_mask < 1] = -np.inf
            q_target = torch.max(q_target, -1).values
            q_target_tot = self.target_mix_net(next_states.view((B*T, next_states.size(2))), q_target.view((B*T, self._n_agents))).view((B, T))
            y = rewards + self.config['gamma'] * (1 - dones) * q_target_tot

            loss_mask = (torch.arange(T, device=self.device).unsqueeze(0).repeat((B, 1)) < torch.as_tensor(seq_lens, device=self.device).unsqueeze(-1)).float()

        # Unroll & compute Q values
        rnn_state = torch.zeros((B, self._n_agents, self.agent_net._hidden_dim), dtype=obs.dtype, device=obs.device)
        q = []
        for t in range(T):
            _q, rnn_state = self.agent_net(_append_agent_ids(obs[:,t].contiguous(), self._n_agents), rnn_state)
            q.append(_q)

        q = torch.stack(q, 1)
        action_q = torch.gather(q, 3, actions.unsqueeze(-1)).squeeze(-1)
        q_tot = self.mix_net(states.view((B*T, states.size(2))), action_q.view((B*T, self._n_agents))).view((B, T))

        return torch.sum(torch.square(q_tot - y) * loss_mask) / loss_mask.sum()

    def learn_on_batch(self, samples):
        observations, action_mask, states = _unpack_obs_batch(samples['obs'], self.observation_space.original_space, unpack_state=True)
        next_observations, next_action_mask, next_states = _unpack_obs_batch(samples['new_obs'], self.observation_space.original_space, unpack_state=True)

        chopped, _, seq_lens = chop_into_sequences(
            episode_ids=samples[SampleBatch.EPS_ID],
            unroll_ids=samples[SampleBatch.UNROLL_ID],
            agent_indices=samples[SampleBatch.AGENT_INDEX],
            feature_columns=[
                observations, action_mask, states, samples['actions'],
                next_observations, next_action_mask, next_states,
                samples['dones'], samples['rewards']
            ],
            state_columns=[],
            max_seq_len=9999
        )

        B, T = len(seq_lens), max(seq_lens)
        print('Learn - B=%d, T=%d' % (B, T))
        for i, t in enumerate(chopped):
            chopped[i] = torch.as_tensor(t, device=self.device).reshape(tuple([B, T] + list(t.shape[1:])))

        self.opt.zero_grad()
        loss = self._compute_loss(seq_lens, *chopped)
        loss.backward()
        self.opt.step()

        return {'loss': float(loss.item())}

    def get_weights(self):
        return {
            'agent': {k: v.cpu() for k,v in self.agent_net.state_dict().items()},
            'mix': {k: v.cpu() for k,v in self.mix_net.state_dict().items()}
        }

    def set_weights(self, weights):
        self.agent_net.load_state_dict(weights['agent'])
        self.mix_net.load_state_dict(weights['mix'])

    def update_target(self):
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mix_net.load_state_dict(self.mix_net.state_dict())

    def is_recurrent(self) -> bool:
        return True

    def num_state_tensors(self) -> int:
        return 1