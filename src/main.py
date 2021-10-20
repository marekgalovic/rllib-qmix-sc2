import os
import time
from argparse import ArgumentParser

import ray
import ray.tune as tune

from env import env_creator
from qmix_trainer import QmixTrainer


class CleanupSC2Processes(tune.Callback):

    def on_trial_start(self, *args, **kwargs):
        os.system('pkill -f StarCraftII')
        time.sleep(1)


def main(args):
    ray.init()
    tune.register_env('sc2_grouped', env_creator)

    tune.run(
        QmixTrainer,
        name='qmix_sc2',
        stop={
            'training_iteration': args.num_iters,
        },
        config={
            'env': 'sc2_grouped',
            "framework": "torch",
            "num_gpus": 1,
            "num_workers": args.num_workers,
            "env_config": {
                "map_name": args.map_name,
            },
            'buffer_size': args.buffer_size,
            'train_batch_size': args.train_batch_size,
            'training_intensity': args.training_intensity,
            'optimizer': {
                'type': args.optimizer_type,
                'lr': args.optimizer_lr
            },
            "exploration_config": {
                "epsilon_timesteps": args.exploration_timesteps,
            },
        },
        callbacks=[
            CleanupSC2Processes()
        ],
        num_samples=args.num_samples
    )


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--num-iters', type=int, default=30000)
    parser.add_argument('--num-workers', type=int, default=30)
    parser.add_argument('--buffer-size', type=int, default=5000)
    parser.add_argument('--train-batch-size', type=int, default=32)
    parser.add_argument('--training-intensity', type=int, default=100)
    parser.add_argument('--optimizer-type', type=str, default='adam')
    parser.add_argument('--optimizer-lr', type=float, default=5e-4)
    parser.add_argument('--exploration-timesteps', type=int, default=100000)
    parser.add_argument('--map-name', type=str, default='1c3s5z')
    parser.add_argument('--num-samples', type=int, default=1)

    main(parser.parse_args())