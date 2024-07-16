# © 2024 Bill Chow. All rights reserved.
# Unauthorized use, modification, or distribution of this code is strictly prohibited.

from typing import NamedTuple

import optax
import pgx


class Config(NamedTuple):
    # Self-play
    self_play_batch_size: int = 512
    simulations: int = 32
    max_actions: int = 42

    # Training
    training_batch_size: int = 4096
    # ~30 games per training step. Formula assuming training_batch_size=4096. Calculated from the AlphaZero paper
    train_to_self_play_ratio: int = self_play_batch_size // 30
    max_training_steps: int = 25_000

    # Optimizer hyperparameters
    learning_rate_schedule: optax.Schedule = optax.linear_onecycle_schedule(max_training_steps, 2e-1)
    momentum_schedule: optax.Schedule = optax.linear_onecycle_schedule(max_training_steps, 0.8, 0.3, 0.85, 0.8 / 0.95,
                                                                       1.)
    # weight_decay: float = 1e-4
    # momentum: float = 0.9
    optimizer: optax.GradientTransformation = optax.inject_hyperparams(optax.sgd)(learning_rate=learning_rate_schedule,
                                                                                  momentum=momentum_schedule)

    # Evaluation
    evaluation_interval: int = 1000  # max_training_steps // 700
    save_svg_interval: int = 5000  # max_training_steps // 14
    checkpoint_interval: int = evaluation_interval  # Needed for restoring checkpoint for evaluation
    svg_directory: str = 'animations'
    evaluation_batch_size: int = 32_000

    # Replay buffer
    game_buffer_size: int = 1_000_000 // 16  # Changing this will affect self-play and training batch sizes above
    min_buffer_size: int = training_batch_size
    run_name: str = 'run2_armageddon'

    # PGX configuration
    env_id: pgx.EnvId = 'connect_four'
    seed: int = 0
    policy_shape = (7,)
    action_count = 7
