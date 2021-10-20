import torch
import torch.nn.functional as F


class AgentNet(torch.nn.Module):

    def __init__(self, obs_dim, n_actions, hidden_dim, **kwargs):
        super().__init__(**kwargs)

        self._obs_dim = obs_dim
        self._hidden_dim = hidden_dim
        self._n_actions = n_actions

        self.l1 = torch.nn.Linear(obs_dim, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, n_actions)
        self.rnn = torch.nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, obs, rnn_state):
        bs, n_agents, obs_dim = obs.shape
        obs = obs.view((bs * n_agents, obs_dim))
        rnn_state = rnn_state.view(bs * n_agents, self._hidden_dim)

        y = self.l1(obs)
        rnn_state = self.rnn(y, rnn_state)
        q = self.l2(rnn_state)

        return (
            q.view((bs, n_agents, self._n_actions)),
            rnn_state.view((bs, n_agents, self._hidden_dim))
        )


class VDN(torch.nn.Module):

    def forward(self, states, agent_qs):
        return torch.sum(agent_qs, -1)


class MonotonicHyperNetLayer(torch.nn.Module):

    def __init__(self, in_dim, state_dim, out_dim, hyper_hidden_dim, final_bias=False, **kwargs):
        super().__init__(**kwargs)

        self._in_dim = in_dim
        self._out_dim = out_dim

        self.W = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hyper_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hyper_hidden_dim, in_dim * out_dim)
        )

        if final_bias:
            self.b = torch.nn.Sequential(
                torch.nn.Linear(state_dim, in_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(in_dim, out_dim)
            )
        else:
            self.b = torch.nn.Linear(state_dim, out_dim)


    def forward(self, input, hyper_input):
        w = torch.abs(self.W(hyper_input).reshape((hyper_input.size(0), self._out_dim, self._in_dim)))
        b = self.b(hyper_input)

        return torch.einsum('ikj,ij->ik', w, input) + b


class QMIX(torch.nn.Module):

    def __init__(self, state_dim=3, n_agents=2, hidden_dim=8, hypernet_hidden_dim=8, **kwargs):
        super().__init__(**kwargs)

        self._n_agents = n_agents
        self._hidden_dim = hidden_dim

        self.l1 = MonotonicHyperNetLayer(n_agents, state_dim, hidden_dim, hypernet_hidden_dim)
        self.l2 = MonotonicHyperNetLayer(hidden_dim, state_dim, 1, hypernet_hidden_dim, final_bias=True)

    def forward(self, states, agent_qs):
        y = F.elu(self.l1(agent_qs, states))
        y = self.l2(y, states)

        return y.reshape((states.size(0),))