# © 2024 Bill Chow. All rights reserved.
# Unauthorized use, modification, or distribution of this code is strictly prohibited.

# Imports #############################################################################################################

import functools
import math
import os
import pathlib
import pickle
import sys
from typing import Any

import flax.jax_utils
import flax.linen as nn
import flax.metrics.tensorboard
import jax
import jax.nn
import jax.numpy as jnp
import mctx
import numpy as np
import optax
import orbax.checkpoint as ocp
import pgx
from flax.typing import FrozenVariableDict

import evaluate
from common import *
from network import model


# Training #############################################################################################################

def create_train_state(module: nn.Module, rng: jax.Array, optimizer: optax.GradientTransformation) -> TrainState:
    variables = module.init(rngs=rng, x=jnp.empty((1, 6, 7, 2)), train=True)
    params, batch_stats = variables['params'], variables['batch_stats']

    return TrainState.create(apply_fn=module.apply,
                             params=params,
                             tx=optimizer,
                             batch_stats=batch_stats,
                             metrics=Metrics.empty())


def recurrent_function(train_state: TrainState, rng_key: jax.Array, action: jax.Array,
                       embedding: pgx.State) -> (mctx.RecurrentFnOutput, pgx.State):
    del rng_key  # Unused

    params, batch_stats = train_state.params, train_state.batch_stats
    state = embedding

    # Update state (embedding)
    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)

    # Get prior_logits (shape [B, ...]) and value (shape [B]) for new state
    logits, value = train_state.apply_fn({'params': params, 'batch_stats': batch_stats}, state.observation, train=False)

    # We have to mask invalid actions here as MuZero allows illegal actions within its search space
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    # Reference formula: leaf_value = reward + discount * value

    # Set reward to the game outcome if state.terminated, if not set it to zero (rely on value estimate)
    # Already done by PGX API
    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]  # [B, 2] -> [B]

    # If state.terminated, set discount to 0 (rely on reward)
    # If not, set it to -1 (the calculated value is for the opposing player)
    discount = jnp.where(state.terminated, 0., -1.) * jnp.ones_like(value)  # [B]

    return mctx.RecurrentFnOutput(reward=reward, discount=discount, prior_logits=logits, value=value), state


# We simply format the inputs for the search, run it, and dump the outputs needed for to_training_batch()
@jax.pmap
def self_play(train_state: TrainState, rng_key: jax.Array) -> SelfPlayOutput:
    # Function executed at each step in a self_play episode
    def step_function(carry: pgx.State, x: jax.Array) -> tuple[pgx.State, SelfPlayOutput]:
        state = carry
        rng_key = x
        params, batch_stats = train_state.params, train_state.batch_stats
        logits, value = train_state.apply_fn({'params': params, 'batch_stats': batch_stats}, state.observation,
                                             train=False)

        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        policy_output = mctx.gumbel_muzero_policy(
            params=train_state,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_function,
            num_simulations=config.simulations,
            invalid_actions=~state.legal_action_mask)  # Take note of the negation symbol

        # Backup observation and current_player for SelfPlayOutput before applying action to state
        observation = state.observation  # [B, ...]
        current_player = state.current_player

        # Update state (carry)
        # For now, we don't do the auto-reset thing to keep things simple
        state = jax.vmap(env.step)(state, policy_output.action)  # [B]

        return state, SelfPlayOutput(rewards=state.rewards[jnp.arange(state.rewards.shape[0]), current_player],
                                     # [B, 2]
                                     terminated=state.terminated,  # [B]
                                     observation=observation,
                                     action_weights=policy_output.action_weights)

    # Initialize batch_size // num_devices states
    rng_key, subkey = jax.random.split(rng_key)
    keys = jax.random.split(subkey, config.self_play_batch_size // num_devices)  # [B]
    del subkey
    state = jax.vmap(env.init)(keys)  # [B]
    del keys

    # Generate PRNG keys for each step
    keys = jax.random.split(rng_key, config.max_actions)  # [MAX_L]
    del rng_key

    # Get output for each step in batches
    _, output = jax.lax.scan(f=step_function, init=state, xs=keys)  # [B], [MAX_L] -> [B], [MAX_L, B, ...]

    # Swap axes
    output = jax.tree.map(f=lambda x: x.swapaxes(0, 1), tree=output)  # [MAX_L, B, ...] -> [B, MAX_L, ...]

    return output


# We simply format the self-play outputs for train_step()
#     self_play_outputs: batched SelfPlayOutputs (shape [B, ...])
@jax.pmap
def to_training_batch(self_play_outputs: SelfPlayOutput, rng_key: jax.Array) -> TrainBatch:
    del rng_key  # Unused

    # Backpropagate the game outcome to preceding states to create target_value

    # Swap axes first for jax.lax.scan()
    target_value = self_play_outputs.rewards.swapaxes(0, 1)  # [B, MAX_L] -> [MAX_L, B]

    # jax.lax.scan() in reverse direction. Output preserves the original order.
    # Body function for jax.lax.scan(): [B], [B] -> [B]
    #     carry: jax.Array (shape [B]) of the reward value to be backpropagated
    #         x: jax.Array (shape [B]) of the current reward value at each index
    # Starting with init=jnp.zeros((B)), we can flip the reward value's
    # sign every turn without needing to first find it explicitly
    _, target_value = jax.lax.scan(f=lambda carry, x: (x - carry, x - carry),
                                   init=jnp.zeros(config.self_play_batch_size // num_devices),
                                   xs=target_value,
                                   reverse=True)  # [B], [MAX_L, B] -> [B], [MAX_L, B]

    # Implement Armageddon mode. A draw is a loss for the first player, and a win for the second
    armageddon_value = jnp.empty_like(target_value)  # [MAX_L, B]
    armageddon_value = armageddon_value.at[::2].set(-1.)
    armageddon_value = armageddon_value.at[1::2].set(1.)
    target_value = jax.lax.select(target_value != 0, target_value, armageddon_value)

    # Swap axes back
    target_value = target_value.swapaxes(0, 1)  # [MAX_L, B] -> [B, MAX_L]

    return TrainBatch(observation=self_play_outputs.observation,
                      target_policy=self_play_outputs.action_weights,
                      target_value=target_value,
                      terminated_index=(
                          jnp.repeat(jnp.argmax(self_play_outputs.terminated, axis=1), config.max_actions)
                          .reshape(-1, config.max_actions)))  # [B, MAX_L] -> [B]


# Train for a single step
@functools.partial(jax.pmap, axis_name='ensemble')
def train_step(state: TrainState, batch: TrainBatch) -> TrainState:
    # Loss function
    def loss_fn(params: FrozenVariableDict, batch_stats: FrozenDict[str, Any]) \
            -> (jax.Array, (FrozenDict[str, Any], jax.Array, jax.Array, jax.Array)):
        (logits, value), updates = state.apply_fn({'params': params, 'batch_stats': batch_stats},
                                                  batch.observation,
                                                  train=True,
                                                  mutable=['batch_stats'])

        policy_loss = optax.softmax_cross_entropy(logits=logits, labels=batch.target_policy)
        value_loss = optax.squared_error(predictions=value, targets=batch.target_value)
        loss_weights = batch.terminated_index  # HACK: Store loss weight here to save memory
        weighted_loss = jnp.mean(loss_weights * (policy_loss + 2 * value_loss))
        return weighted_loss, (updates, policy_loss.mean(), value_loss.mean())

    # Get gradients and metrics
    grad_fn = jax.value_and_grad(fun=loss_fn, has_aux=True)
    (loss, (updates, policy_loss, value_loss)), grads = grad_fn(state.params, state.batch_stats)

    # Average out gradients across devices
    grads = jax.lax.pmean(grads, axis_name='ensemble')

    # Apply gradients to state
    state = state.apply_gradients(grads=grads)

    # Gather metrics across devices
    metric_updates = Metrics.gather_from_model_output(axis_name='ensemble',
                                                      loss=loss,
                                                      policy_loss=policy_loss,
                                                      value_loss=value_loss)
    metrics = state.metrics.merge(metric_updates)

    state = state.replace(batch_stats=updates['batch_stats'], metrics=metrics)
    return state


if __name__ == '__main__':
    # Generate root rng_key
    # - If you call jax.random.split() on the same key, you will always get the same split keys
    # - Never use the same key twice
    rng_key = jax.random.key(config.seed)

    # Initialize model state
    rng_key, subkey = jax.random.split(rng_key)
    state = create_train_state(model, subkey, config.optimizer)
    del subkey

    # Restore state checkpoint if available
    latest_step = checkpoint_manager.latest_step()
    if latest_step is not None:
        state = checkpoint_manager.restore(latest_step, args=ocp.args.StandardRestore(state))
        logger.info(f'restored training checkpoint at step {latest_step}')
    else:
        logger.warning(f'no checkpoint found in \'{os.path.realpath(f"checkpoints/{config.run_name}")}\'')
        latest_step = -1  # The first step should be 0, so we can evaluate when step % interval == 0

    # Replicate state across all devices
    state = jax.device_put_replicated(state, devices)

    metrics_history = {'train_loss': math.inf,
                       'train_policy_loss': math.inf,
                       'train_value_loss': math.inf,
                       'validation_elo': 0.,
                       'validation_elo_delta': 0.}
    summary_writer = flax.metrics.tensorboard.SummaryWriter(os.path.realpath(f'tensorboard/{config.run_name}'))
    replay_buffer = None

    # Restore last saved replay_buffer if it exists
    if os.path.isfile(f'{config.run_name}_replay_buffer.pkl'):
        with open(f'{config.run_name}_replay_buffer.pkl', 'rb') as f:
            replay_buffer = pickle.load(f)
        logger.info('loaded replay_buffer from storage')

    # Training loop
    for step in range(latest_step + 1, config.max_training_steps + 1, config.train_to_self_play_ratio):
        # Self-play ####################################################################################################

        # Self-play games until there is enough to at least train one batch
        self_played_once = False
        while replay_buffer is None or len(replay_buffer) < config.min_buffer_size or not self_played_once:
            self_played_once = True
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            del subkey
            self_play_output = self_play(state, keys)
            del keys

            # Process training data
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            del subkey
            batch = to_training_batch(self_play_output, keys)
            del keys

            # Combine batches across devices
            batch = jax.tree.map(lambda x: x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]), batch)

            # Merge new data into buffer first
            replay_buffer = batch if replay_buffer is None else replay_buffer.concat(batch)

            # Update log with size of replay_buffer
            logger.info(f'replay_buffer size = {replay_buffer.observation.shape[0]}')

            # Backup replay_buffer in case we stop the run halfway
            with open(f'{config.run_name}_replay_buffer.pkl', 'wb') as f:
                pickle.dump(replay_buffer, f)
            logger.info('saved replay_buffer to storage')

        # Training #####################################################################################################

        # Train for config.train_to_self_play_ratio steps before self-playing again

        for _ in range(config.train_to_self_play_ratio):

            # Sample a batch from the data
            rng_key, subkey = jax.random.split(rng_key)
            batch = replay_buffer.sample_batch(subkey, step)
            del subkey

            state = train_step(state, batch)

            # Print metrics
            training_info = f'train step {step}: '

            for metric, value in state.metrics.unreplicate().compute().items():
                # Convert to Python float to make it JSON-serializable
                metrics_history[f'train_{metric}'] = value.item()
                summary_writer.scalar(f'train/{metric}', value, step)
                training_info += f'{metric}: {value.item():.4f}, '

            training_info = training_info[:-2]  # Remove trailing ', '
            logger.info(training_info)

            # Validation ###############################################################################################
            if step % config.evaluation_interval == 0 and step != 0:
                previous_step = step - config.evaluation_interval
                rng_key, subkey = jax.random.split(rng_key)
                keys = jax.random.split(subkey, num_devices)
                del subkey

                # Create baseline_train_state from saved weights
                checkpoint_manager.wait_until_finished()
                baseline_train_state = checkpoint_manager.restore(previous_step, args=ocp.args.StandardRestore(state))
                logger.info(f'restored step {previous_step} for evaluation')

                # Calculate ELO delta
                elo_delta = evaluate.calculate_elo_delta(baseline_train_state, state, keys)
                metrics_history['validation_elo_delta'] = elo_delta
                metrics_history['validation_elo'] = (checkpoint_manager.metrics(previous_step)['validation_elo'] +
                                                     elo_delta)

                # Print ELO metrics
                elo_delta = metrics_history['validation_elo_delta']
                elo = metrics_history['validation_elo']
                summary_writer.scalar('validation/elo_delta', elo_delta, step)
                summary_writer.scalar('validation/elo', elo, step)
                logger.info(f'validation_elo_delta: {int(elo_delta)}, validation_elo: {int(elo)}')

                # Print model weights scalar and histogram metrics on Tensorboard
                for key, value in flax.traverse_util.flatten_dict(state.params, sep='/').items():
                    value = np.array(value)
                    if value.size == 1:
                        summary_writer.scalar(key, value, step)
                    else:
                        summary_writer.histogram(key, value, step)

                # Print softmax-ed logits and value to Tensorboard every
                # evaluation to ensure that we are on the right track
                rng_key, subkey = jax.random.split(rng_key)
                keys = jax.random.split(subkey, 1)
                del subkey
                pgx_state = jax.vmap(env.init)(keys)
                del keys
                single_state = jax.tree.map(lambda x: x[0], state)
                logits, value = single_state.apply_fn({'params': single_state.params,
                                                       'batch_stats': single_state.batch_stats},
                                                      pgx_state.observation, train=False)
                logits = jax.nn.softmax(logits[0])

                for i in range(7):
                    summary_writer.scalar(f'validation/starting_logit{i + 1}', logits[i], step)

                summary_writer.scalar('validation/starting_value', value, step)

            # Save checkpoint sample self-play SVG every interval ######################################################
            if step % config.save_svg_interval == 0 and step != 0:
                logger.info(f'running self-play game for SVG generation')
                rng_key, subkey = jax.random.split(rng_key)
                keys = jax.random.split(subkey, 1)
                del subkey
                pgx_state = jax.vmap(env.init)(keys)
                del keys
                single_state = jax.tree.map(lambda x: x[0], state)
                states = []

                while not (pgx_state.terminated or pgx_state.truncated):
                    rng_key, subkey = jax.random.split(rng_key)

                    params, batch_stats = single_state.params, single_state.batch_stats
                    logits, value = state.apply_fn({'params': params, 'batch_stats': batch_stats},
                                                   pgx_state.observation, train=False)
                    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=pgx_state)

                    policy_output = mctx.gumbel_muzero_policy(
                        params=single_state,
                        rng_key=subkey,
                        root=root,
                        recurrent_fn=recurrent_function,
                        num_simulations=800,
                        invalid_actions=~pgx_state.legal_action_mask,  # Take note of the negation symbol
                        gumbel_scale=0.)  # It's OK to play deterministically since we are only generating one game
                    del subkey

                    action = policy_output.action

                    pgx_state = jax.vmap(env.step)(pgx_state, action)
                    states.append(pgx_state)

                pathlib.Path(f'{config.svg_directory}/{config.run_name}').mkdir(parents=True, exist_ok=True)
                pgx.save_svg_animation(states, f'{config.svg_directory}/{config.run_name}/{step:05}.svg',
                                       frame_duration_seconds=1.)
                logger.info(f'saved self-play game in {config.svg_directory}/{config.run_name}/{step:05}.svg')

            # Save checkpoint
            checkpoint_manager.save(step=step,
                                    metrics=metrics_history,
                                    args=ocp.args.StandardSave(flax.jax_utils.unreplicate(state)))

            step += 1

        summary_writer.flush()

    checkpoint_manager.wait_until_finished()
    logger.info('training has completed!')
