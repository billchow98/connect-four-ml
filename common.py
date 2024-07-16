# © 2024 Bill Chow. All rights reserved.
# Unauthorized use, modification, or distribution of this code is strictly prohibited.

import os
from typing import Any

import clu.metrics
import flax.struct
import flax.training.train_state
import jax
import jax.nn
import jax.numpy as jnp
import logging
import orbax.checkpoint as ocp
from flax.typing import FrozenDict

from config import *

config = Config()

# Logging ##############################################################################################################

logger = logging.getLogger('__main__')

logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s [%(levelname)-8.8s] %(name)s: %(message)s')

for handler in logging.getLogger().handlers:  # The main logger's handlers
    handler.setFormatter(log_formatter)  # Set other libraries' loggers to a standardized format

file_handler = logging.FileHandler(f'{config.run_name}.log')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Devices ##############################################################################################################

devices = jax.local_devices()
logger.info(f'jax.local_devices(): {devices}')

# Select devices by ID
if 'DEVICES' in os.environ:
    DEVICE_IDS = [int(x) for x in os.environ['DEVICES'].split(',')]
    devices = [devices[i] for i in DEVICE_IDS]

logger.info(f'selecting devices: {devices}')
num_devices = len(devices)
logger.info(f'JAX found {num_devices} devices')


# Classes ##############################################################################################################

@flax.struct.dataclass
class Metrics(clu.metrics.Collection):
    loss: clu.metrics.Average.from_output('loss')
    policy_loss: clu.metrics.Average.from_output('policy_loss')
    value_loss: clu.metrics.Average.from_output('value_loss')


class TrainState(flax.training.train_state.TrainState):
    batch_stats: FrozenDict[str, Any]
    metrics: Metrics


class SelfPlayOutput(NamedTuple):
    rewards: jax.Array
    terminated: jax.Array
    observation: jax.Array
    action_weights: jax.Array


class TrainBatch(NamedTuple):
    observation: jax.Array
    target_policy: jax.Array
    target_value: jax.Array
    terminated_index: jax.Array

    def __len__(self) -> int:
        return self.observation.shape[0]

    # Concatenates TrainBatch `other` with the existing data in self
    def concat(self, other: 'TrainBatch') -> 'TrainBatch':
        # Concatenate each jax.Array along axis=0 (the batch axis)
        new_train_batch = jax.tree.map(lambda x, y: jnp.concatenate((x, y)), self, other)

        # Remove old data if past the maximum buffer size
        buffer_size = new_train_batch.observation.shape[0]
        if buffer_size > config.game_buffer_size:
            # Simple sliding window strategy
            new_train_batch = jax.tree.map(lambda x: x[-config.game_buffer_size:], new_train_batch)
        return new_train_batch

    def as_flipped(self, flip: jax.Array, game_stage: jax.Array, train_step: int) -> 'TrainBatch':
        p = (0 - 0.9) * (train_step / (config.max_training_steps // 4)) ** 1 + 0.9
        q = 16 * (game_stage - p)
        loss_weight = jax.nn.sigmoid(q)
        return TrainBatch(observation=jax.lax.select(flip, jnp.flip(self.observation, axis=1), self.observation),
                          target_policy=jax.lax.select(flip, jnp.flip(self.target_policy), self.target_policy),
                          target_value=self.target_value,
                          terminated_index=loss_weight)  # HACK: Store loss weight here to save memory

    # Randomly samples one position from `config.training_batch_size` randomly-selected games for training
    def sample_batch(self, rng_key: jax.Array, train_step: int) -> 'TrainBatch':
        buffer_size = len(self)
        rng_key, subkey = jax.random.split(rng_key)
        sample_indices = jax.random.choice(key=subkey,
                                           a=buffer_size,
                                           shape=(config.training_batch_size // num_devices * num_devices,),
                                           replace=False)
        del subkey
        batch = jax.tree.map(lambda x: x[sample_indices, ...], self)  # [BUF_SZ, MAX_L, ...] -> [B, MAX_L, ...]

        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, len(sample_indices))
        del subkey

        # Randomly pick one position per game
        def body_function(game: 'TrainBatch', rng_key: jax.Array) -> 'TrainBatch':
            # Get terminated index and sample a position in [0, terminated_index)
            rng_key, subkey = jax.random.split(rng_key)
            sample_index = jax.random.randint(key=subkey, shape=(), minval=0, maxval=game.terminated_index[0])
            del subkey
            sample = jax.tree.map(lambda x: x[sample_index] if x.ndim > 0 else x, game)  # [MAX_L, ...] -> [...]
            flip = jax.random.choice(rng_key, jnp.array([True, False]))
            return sample.as_flipped(flip, sample_index / sample.terminated_index, train_step)

        samples = jax.vmap(body_function)(batch, keys)
        del keys

        batch = jax.tree.map(lambda x: x.reshape(num_devices, config.training_batch_size // num_devices, *x.shape[1:]),
                             samples)
        return batch


# Global variables #####################################################################################################

env = pgx.make(config.env_id)

# Initialize checkpoint manager
ocp_options = ocp.CheckpointManagerOptions(max_to_keep=3,
                                           keep_period=config.checkpoint_interval,
                                           best_fn=lambda x: x['validation_elo'],  # For saving previous run's ELO
                                           step_format_fixed_length=5,
                                           create=True)

checkpoint_manager = ocp.CheckpointManager(os.path.realpath(f'checkpoints/{config.run_name}'), options=ocp_options)

# For BayesElo
pgn_file_path = '/tmp/results.pgn'
