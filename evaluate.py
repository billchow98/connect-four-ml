# © 2024 Bill Chow. All rights reserved.
# Unauthorized use, modification, or distribution of this code is strictly prohibited.

import argparse
import functools
import importlib
import os.path
import subprocess
from typing import Any

import jax
import jax.numpy as jnp
import mctx
import pgx

import main
from common import *


@functools.partial(jax.pmap, axis_name='ensemble')
def calculate_elo_delta_base(baseline_train_state: TrainState, test_train_state: TrainState,
                             rng_key: jax.Array) -> tuple[jax.Array, jax.Array]:
    # Function executed at each step in a self_play episode
    # We need two separate functions for the different models to prevent JIT different shape errors
    def test_step_function(state: pgx.State, rng_key: jax.Array) -> pgx.State:
        params, batch_stats = test_train_state.params, test_train_state.batch_stats
        logits, value = test_train_state.apply_fn({'params': params, 'batch_stats': batch_stats}, state.observation,
                                                  train=False)

        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        policy_output = mctx.gumbel_muzero_policy(
            params=test_train_state,
            rng_key=rng_key,
            root=root,
            recurrent_fn=main.recurrent_function,
            num_simulations=2,
            invalid_actions=~state.legal_action_mask)  # Take note of the negation symbol
        del rng_key

        state = jax.vmap(env.step)(state, policy_output.action)  # [B]
        return state

    def baseline_step_function(state: pgx.State, rng_key: jax.Array) -> pgx.State:
        params, batch_stats = baseline_train_state.params, baseline_train_state.batch_stats
        logits, value = baseline_train_state.apply_fn({'params': params, 'batch_stats': batch_stats}, state.observation,
                                                      train=False)

        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        policy_output = mctx.gumbel_muzero_policy(
            params=baseline_train_state,
            rng_key=rng_key,
            root=root,
            recurrent_fn=main.recurrent_function,
            num_simulations=2,
            invalid_actions=~state.legal_action_mask)  # Take note of the negation symbol
        del rng_key

        state = jax.vmap(env.step)(state, policy_output.action)  # [B]
        return state

    # test_side_index: Side index of test network
    def play_function(state: pgx.State, rng_key: jax.Array, test_side_index: int) -> jax.Array:
        shape = (config.evaluation_batch_size // 2 // num_devices, config.max_actions)  # [B, MAX_L]
        keys = jax.random.split(rng_key, shape[1])  # [MAX_L]. We only need one key per step (based on MCTX API)
        del rng_key

        def body_function(carry: tuple[pgx.State, jax.Array],
                          i: jax.Array) -> tuple[tuple[pgx.State, jax.Array], jax.Array]:
            state, previous_terminated = carry
            state = jax.lax.cond((i & 1) == test_side_index, test_step_function, baseline_step_function, state, keys[i])
            return (state, state.terminated), (1 * state.terminated - previous_terminated) * (state.rewards[:, 0] + 3)

        _, rewards = jax.lax.scan(body_function, (state, jnp.full(shape[0], False)), jnp.arange(shape[1]))  # [MAX_L, B]
        del keys
        rewards = jnp.max(rewards, axis=0)  # [B] (win=4, draw=3, loss=2)
        rewards -= 3  # [B] (win=1, draw=0, loss=-1)
        return rewards  # Return game result from Black (first player)'s perspective

    # Initialize evaluation_batch_size // 2 // num_devices states for games where test network plays first
    rng_key, subkey = jax.random.split(rng_key)
    keys = jax.random.split(subkey, config.evaluation_batch_size // 2 // num_devices)  # [B]
    assert len(keys) != 0
    del subkey
    state = jax.vmap(env.init)(keys)  # [B]
    del keys

    # PGX library randomizes the starting player without documenting it
    # No wonder they require a key for env.init
    # Without this, state.rewards can give 1 or -1 even if the same player wins
    state = state.replace(current_player=jnp.zeros_like(state.current_player))

    # Generate PRNG key
    rng_key, subkey = jax.random.split(rng_key)

    # Get batch_size // 2 // num_devices results (test network plays first)
    test_first_results = play_function(state, subkey, test_side_index=0)  # [B], [], [] -> [B]
    del subkey

    # Re-initialize states
    rng_key, subkey = jax.random.split(rng_key)
    keys = jax.random.split(subkey, config.evaluation_batch_size // 2 // num_devices)  # [B]
    del subkey
    state = jax.vmap(env.init)(keys)  # [B]
    del keys

    # Explained above
    state = state.replace(current_player=jnp.zeros_like(state.current_player))

    # Switch sides (baseline network plays first). Consume rng_key (its final use)
    baseline_first_results = play_function(state, rng_key, test_side_index=1)  # [B], [], [] -> [B]
    del rng_key

    return test_first_results, baseline_first_results


def calculate_elo_delta(baseline_train_state: TrainState, test_train_state: TrainState, rng_key: jax.Array,
                        baseline_name: str = 'baseline', test_name: str = 'test') -> int:
    # Get game results
    test_first_results, baseline_first_results = calculate_elo_delta_base(baseline_train_state, test_train_state,
                                                                          rng_key)

    # Flatten results from devices
    test_first_results = test_first_results.reshape(-1)
    baseline_first_results = baseline_first_results.reshape(-1)

    assert (len(test_first_results) == len(baseline_first_results) ==
            config.evaluation_batch_size // 2 // num_devices * num_devices)

    # Will overwrite previous data
    with open(pgn_file_path, 'w') as pgn_file:
        def print_results(game_results: jax.Array, p1_name: str, p2_name: str) -> None:
            for _ in range((game_results == 1.).sum()):
                pgn_file.write(f'[White "{p1_name}"][Black "{p2_name}"][Result "1-0"] 1. e4 e5\n')
            for _ in range((game_results == 0.).sum()):
                pgn_file.write(f'[White "{p1_name}"][Black "{p2_name}"][Result "1/2-1/2"] 1. e4 e5\n')
            for _ in range((game_results == -1.).sum()):
                pgn_file.write(f'[White "{p1_name}"][Black "{p2_name}"][Result "0-1"] 1. e4 e5\n')

        print_results(test_first_results, p1_name=test_name, p2_name=baseline_name)
        print_results(baseline_first_results, p1_name=baseline_name, p2_name=test_name)

    # Raw ELO delta, positive uncertainty in ELO, negative uncertainty in ELO
    raw_output = subprocess.run(['./calc_elo_delta.sh', test_name], stdout=subprocess.PIPE).stdout

    # Get raw BayesElo stats
    # For two-player ELO ratings, BayesElo gives x for one player and -x for the other
    # The elo delta is thus _twice_ of the raw output (x)
    # The uncertainties also have to be added up
    elo_delta, plus_uncertainty, minus_uncertainty = [2 * int(x) for x in raw_output.split()]

    logger.info('test network ELO difference from baseline network: '
                # Double the uncertainty because the ELO difference involves two ELO deltas with uncertainties
                # E.g. 10 +- 1 vs -10 +- 1. The delta could range from +22 to +20
                f'{elo_delta:+} (+{plus_uncertainty}/-{minus_uncertainty}) ELO '
                f'from {len(test_first_results) + len(baseline_first_results)} self-played games')

    # Collect stats
    test_win_count = (test_first_results == 1.).sum() + (baseline_first_results == -1.).sum()
    test_draw_count = (test_first_results == 0.).sum() + (baseline_first_results == 0.).sum()
    test_loss_count = (test_first_results == -1.).sum() + (baseline_first_results == 1.).sum()
    logger.info(f'game outcomes: W: {test_win_count}, D: {test_draw_count}, L: {test_loss_count}')

    return elo_delta


if __name__ == '__main__':
    def get_train_state(run_name: str, run_type: str) -> Any:
        # Initialize state
        model = importlib.import_module(f'networks.{run_name}.network').model
        config = importlib.import_module(f'networks.{run_name}.config').Config()
        optimizer = config.optimizer
        rng_key = jax.random.key(config.seed)
        rng_key, subkey = jax.random.split(rng_key)
        state = main.create_train_state(model, subkey, optimizer)
        del subkey
        state = jax.device_put_replicated(state, devices)

        # Create train_state from saved weights
        checkpoint_manager = ocp.CheckpointManager(os.path.realpath(f'checkpoints/{run_name}'),
                                                   options=ocp_options)
        latest_step = checkpoint_manager.latest_step()
        train_state = checkpoint_manager.restore(latest_step, args=ocp.args.StandardRestore(state))
        logger.info(f'restored step {latest_step} of {run_name} as {run_type}')
        return train_state


    # Parse run names
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline_run_name', nargs=1)
    parser.add_argument('test_run_name', nargs=1)
    args = parser.parse_args()

    baseline_run_name = args.baseline_run_name[0]
    test_run_name = args.test_run_name[0]

    # Create TrainStates from saved weights
    baseline_train_state = get_train_state(baseline_run_name, 'baseline')
    test_train_state = get_train_state(test_run_name, 'test')

    # Calculate ELO delta and generate PGN
    config = config._replace(evaluation_batch_size=32_000)  # Large enough without causing an OOM error on my GPU
    keys = jax.random.split(jax.random.key(0), num_devices)
    elo_delta = calculate_elo_delta(baseline_train_state, test_train_state, keys, baseline_run_name, test_run_name)
    del keys

    raw_output = subprocess.run(['./calc_bayeselo.sh', pgn_file_path], stdout=subprocess.PIPE).stdout
    logger.info(f'bayeselo raw output:\n\n{raw_output.decode("utf-8")}')
