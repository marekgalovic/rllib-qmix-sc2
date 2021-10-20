from typing import List
import random
import time

from ray.rllib.agents import with_common_config
from ray.rllib.agents.trainer_template import build_trainer

from ray.rllib.evaluation.worker_set import WorkerSet

from ray.rllib.execution.common import _check_sample_batch_type, _get_shared_metrics, SAMPLE_TIMER
from ray.rllib.execution.replay_ops import Replay, StoreToReplayBuffer
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.execution.train_ops import TrainOneStep, UpdateTargetNetwork
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.concurrency_ops import Concurrently

from ray.rllib.policy.sample_batch import SampleBatch

from ray.rllib.utils.typing import SampleBatchType, TrainerConfigDict
from ray.util.iter import LocalIterator

from qmix_policy import QmixPolicy


class EpisodicReplayBuffer:

    def __init__(self, num_slots:int):
        self.num_slots = num_slots
        self.replay_batches = []
        self.replay_index = 0

    def add_batch(self, sample_batch: SampleBatchType) -> None:
        episodes = sample_batch.split_by_episode()
        print("Store episodes:", len(episodes))
        for episode in episodes:
            self._add_episode(episode)

    def _add_episode(self, episode: SampleBatchType) -> None:
        if len(self.replay_batches) < self.num_slots:
            self.replay_batches.append(episode)
        else:
            self.replay_batches[self.replay_index] = episode
            self.replay_index = (self.replay_index + 1) % self.num_slots

    def replay(self) -> SampleBatchType:
        return random.choice(self.replay_batches)


class ConcatEpisodes:

    def __init__(self, episodes_count:int):
        self.episodes_count = episodes_count
        self.buffer = []
        self.count = 0
        self.last_batch_time = time.perf_counter()

    def __call__(self, batch: SampleBatchType) -> List[SampleBatchType]:
        _check_sample_batch_type(batch)

        if batch.count == 0:
            return []

        self.buffer.append(batch)
        self.count += batch.count

        if len(self.buffer) >= self.episodes_count:
            out = SampleBatch.concat_samples(self.buffer)

            perf_counter = time.perf_counter()
            timer = _get_shared_metrics().timers[SAMPLE_TIMER]
            timer.push(perf_counter - self.last_batch_time)
            timer.push_units_processed(self.count)

            self.last_batch_time = perf_counter
            self.buffer = []
            self.count = 0
            return [out]

        return []


class UpdateTargetNetworkEpisodes(UpdateTargetNetwork):

    def __init__(self, workers, update_freq):
        super().__init__(workers, 1)
        self._optim_steps = 0
        self._update_freq = update_freq

    def __call__(self, arg) -> None:
        self._optim_steps += 1
        if self._optim_steps >= self._update_freq:
            self._optim_steps = 0
            return super().__call__(arg)


def prefill_replay_buffer(store_op, replay_buffer, config):
    i = 0
    while len(replay_buffer.replay_batches) < config['buffer_size']:
        if i % 10 == 0:
            print('Prefill replay buffer:', len(replay_buffer.replay_batches))
        next(store_op)
        i += 1


def execution_plan(workers: WorkerSet, config: TrainerConfigDict) -> LocalIterator[dict]:
    print('Get execution plan')
    rollouts = ParallelRollouts(workers, mode='bulk_sync')
    replay_buffer = EpisodicReplayBuffer(config['buffer_size'])

    store_op = rollouts\
        .for_each(StoreToReplayBuffer(local_buffer=replay_buffer))

    prefill_replay_buffer(store_op, replay_buffer, config)

    train_op = Replay(local_buffer=replay_buffer)\
        .combine(ConcatEpisodes(config['train_batch_size']))\
        .for_each(TrainOneStep(workers))\
        .for_each(UpdateTargetNetworkEpisodes(workers, config['target_network_update_freq']))

    exec_op = Concurrently(
        [store_op, train_op],
        mode='round_robin',
        output_indexes=[1],
        round_robin_weights=[1, config['training_intensity']]
    )
    
    return StandardMetricsReporting(exec_op, workers, config)


DEFAULT_CONFIG = with_common_config({
    'agent_hidden_size': 64,
    'mix_type': 'QMIX',
    'mix_hidden_size': 32,
    'mix_hypernet_hidden_size': 64,
    'buffer_size': 5000,
    'batch_mode': 'complete_episodes',
    'training_intensity': 100,
    'multiagent': {
        'count_steps_by': 'env_steps',
    },
    'train_batch_size': 32,
    'target_network_update_freq': 200,
    'gamma': 0.99,
    'optimizer': {
        'type': 'adam',
        'lr': 5e-4
    },
    # Exploration
    "explore": True,
    "exploration_config": {
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.0,
        "final_epsilon": 0.05,
        "epsilon_timesteps": 100000,
    },
    # Evaluation
    "evaluation_interval": 100,
    "evaluation_num_episodes": 10,
    "evaluation_config": {
        "explore": False
    }
})


QmixTrainer = build_trainer(
    name="CustomQMIX", # Cannot use QMIX because Tune will resolve RLLib's implementation https://github.com/ray-project/ray/issues/19455
    default_config=DEFAULT_CONFIG,
    default_policy=QmixPolicy,
    get_policy_class=None,
    execution_plan=execution_plan
)