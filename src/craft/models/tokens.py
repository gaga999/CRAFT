"""NNX token-wise shared FFNs with periodic residual MLP exchange."""

import jax.numpy as jnp
from flax import nnx

from craft.models.mlp import MLP


class MixerBlock(nnx.Module):
    def __init__(self, width, tokens, exchange, *, rngs):
        self.ffn_norm = nnx.LayerNorm(width, rngs=rngs)
        self.ffn = MLP(width, (4 * width,), width, rngs=rngs)
        self.exchange = exchange
        if exchange:
            self.token_norm = nnx.LayerNorm(width, rngs=rngs)
            self.token_mlp = MLP(tokens, (2 * tokens,), tokens, rngs=rngs)

    def local(self, x):
        return x + self.ffn(self.ffn_norm(x))

    def __call__(self, x):
        x = self.local(x)
        if self.exchange:
            y = jnp.swapaxes(self.token_norm(x), -1, -2)
            x = x + jnp.swapaxes(self.token_mlp(y), -1, -2)
        return x


class TokenBackbone(nnx.Module):
    def __init__(
        self,
        observation_dim,
        action_dim,
        chunk_length,
        *,
        width=128,
        depth=4,
        exchange_every=2,
        condition_dim=0,
        rngs,
    ):
        if any(
            type(value) is not int or value < 1
            for value in (
                observation_dim,
                action_dim,
                chunk_length,
                width,
                depth,
                exchange_every,
            )
        ):
            raise ValueError(
                "Model dimensions, depth and exchange interval must be positive integers"
            )
        if type(condition_dim) is not int or condition_dim < 0:
            raise ValueError("Condition dimension must be a nonnegative integer")
        self.state_projection = nnx.Linear(observation_dim, width, rngs=rngs)
        self.action_projection = nnx.Linear(action_dim, width, rngs=rngs)
        self.position = nnx.Param(
            nnx.initializers.normal(0.02)(rngs.params(), (1, chunk_length + 1, width))
        )
        self.condition_projection = (
            nnx.Linear(condition_dim, width, rngs=rngs) if condition_dim else None
        )
        self.blocks = nnx.List(
            [
                MixerBlock(
                    width,
                    chunk_length + 1,
                    (i + 1) % exchange_every == 0 or i == depth - 1,
                    rngs=rngs,
                )
                for i in range(depth)
            ]
        )
        self.output_norm = nnx.LayerNorm(width, rngs=rngs)

    def tokenize(self, states, actions, condition=None):
        if (
            states.ndim != 2
            or actions.ndim != 3
            or states.shape[0] != actions.shape[0]
            or actions.shape[1] != self.position.shape[1] - 1
        ):
            raise ValueError(
                "Expected states [B,O] and configured action chunks [B,H,A]"
            )
        if condition is not None and self.condition_projection is None:
            raise ValueError("This backbone does not accept a condition")
        x = jnp.concatenate(
            [self.state_projection(states)[:, None], self.action_projection(actions)], 1
        )
        x = x + self.position[...]
        if self.condition_projection is not None:
            if condition is None:
                raise ValueError("Actor condition is required")
            x = x + self.condition_projection(condition)[:, None]
        return x

    def __call__(self, states, actions, condition=None):
        x = self.tokenize(states, actions, condition)
        for block in self.blocks:
            x = block(x)
        return self.output_norm(x)


class GoalEncoder(nnx.Module):
    def __init__(self, observation_dim, width, latent_dim, *, rngs):
        self.features = MLP(observation_dim, (width,), width, rngs=rngs)
        self.norm = nnx.LayerNorm(width, rngs=rngs)
        self.projection = MLP(width, (width,), latent_dim, rngs=rngs)

    def encode(self, states):
        return self.norm(self.features(states))

    def __call__(self, states):
        h = self.encode(states)
        return h, self.projection(h)


class TokenCritic(nnx.Module):
    def __init__(
        self,
        observation_dim,
        action_dim,
        chunk_length,
        *,
        width=128,
        depth=4,
        exchange_every=2,
        latent_dim=128,
        batch_size=256,
        rngs,
    ):
        self.goal_encoder = GoalEncoder(observation_dim, width, latent_dim, rngs=rngs)
        self.backbone = TokenBackbone(
            observation_dim,
            action_dim,
            chunk_length,
            width=width,
            depth=depth,
            exchange_every=exchange_every,
            rngs=rngs,
        )
        self.phi_head = MLP(width, (width,), latent_dim, rngs=rngs)
        self.log_scale = nnx.Param(jnp.log(jnp.array(10.0, jnp.float32)))
        self.bias = nnx.Param(-jnp.log(jnp.array(batch_size - 1.0, jnp.float32)))

    def encode(self, states):
        return self.goal_encoder.encode(states)

    def features(self, states, actions):
        return self.backbone(states, actions)

    def __call__(self, states, actions, goals):
        phi = self.phi_head(self.features(states, actions).mean(axis=1))
        _, psi = self.goal_encoder(goals)

        def unit(x):
            return x / jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + 1e-8)

        return unit(phi), unit(psi)

    def score(self, states, actions, goals):
        phi, psi = self(states, actions, goals)
        return (phi * psi).sum(-1)
