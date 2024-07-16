# © 2024 Bill Chow. All rights reserved.
# Unauthorized use, modification, or distribution of this code is strictly prohibited.

import logging

import jax.random
from flax import linen as nn
from jax import numpy as jnp

from common import *


class _ConvBlock(nn.Module):
    hidden_planes: int

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(features=self.hidden_planes, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.gelu(x)
        return x


class _ResidualBlock(nn.Module):
    hidden_planes: int

    @nn.compact
    def __call__(self, x, train: bool):
        shortcut = x  # Identity shortcut
        x = nn.Conv(features=self.hidden_planes, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=self.hidden_planes, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x += shortcut
        x = nn.gelu(x)
        return x


class _PolicyHead(nn.Module):
    hidden_planes: int

    @nn.compact
    def __call__(self, x, train: bool):
        # Source: arXiv [2111.09259]
        # Switch to a linear layer as in AlphaGo Zero / AlphaZero (for Go) for Connect Four
        x = nn.Conv(features=self.hidden_planes, kernel_size=(1, 1))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.gelu(x)
        x = x.reshape(x.shape[0], -1)  # [B, 6, 7, F] -> [B, ...] (flatten but preserve batch dimension)
        logits = nn.Dense(features=config.action_count)(x)  # [B, ...] -> [B, 7]
        logits = logits.reshape(-1, *config.policy_shape)
        return logits


class _ValueHead(nn.Module):
    hidden_planes: int

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(features=1, kernel_size=(1, 1))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.gelu(x)
        x = x.reshape(x.shape[0], -1)  # Flatten but preserve batch dimension
        x = nn.Dense(features=self.hidden_planes)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=1)(x)
        x = nn.tanh(x)
        value = x.reshape(x.shape[0])  # [B, 1] -> [B]
        return value


class AlphaZeroNet(nn.Module):
    blocks: int
    hidden_planes: int

    @nn.compact
    def __call__(self, x, train: bool):
        x = _ConvBlock(hidden_planes=self.hidden_planes)(x, train)

        for _ in range(self.blocks):
            x = _ResidualBlock(hidden_planes=self.hidden_planes)(x, train)

        logits, value = (_PolicyHead(hidden_planes=self.hidden_planes)(x, train),
                         _ValueHead(hidden_planes=self.hidden_planes)(x, train))

        return logits, value


model = AlphaZeroNet(blocks=5, hidden_planes=64)


def summarize_model() -> None:
    # Currently takes quite a while, even for small models
    if logger.level != logging.DEBUG:
        return

    # Disable color for logging to text file
    table_fn = nn.tabulate(model, jax.random.key(0), compute_flops=True, compute_vjp_flops=False,
                           console_kwargs={'force_terminal': False, 'width': 240})
    logger.debug('\n%s' % table_fn(jnp.empty((1, 6, 7, 2)), train=False))
