"""Shared token-wise FFNs with periodic MLP-based information exchange."""

import jax.numpy as jnp
from flax import linen as nn


class TokenBackbone(nn.Module):
    width: int = 128
    depth: int = 2
    exchange_every: int = 2

    @nn.compact
    def __call__(self, state, actions, condition=None):
        s = nn.Dense(self.width, name="state_projection")(state)[:, None]
        a = nn.Dense(self.width, name="action_projection")(actions)
        x = jnp.concatenate([s, a], axis=1)
        x += self.param(
            "position", nn.initializers.normal(0.02), (1, x.shape[1], self.width)
        )
        if condition is not None:
            x += nn.Dense(self.width, name="condition_projection")(condition)[:, None]
        if self.depth < 1 or self.exchange_every < 1:
            raise ValueError("Depth and exchange_every must be positive")
        for i in range(self.depth):
            # Dense acts only on channels: one FFN shared across all tokens.
            # Layers have independent parameters; sharing is across positions.
            y = nn.LayerNorm(name=f"ffn_norm_{i}")(x)
            y = nn.gelu(nn.Dense(4 * self.width, name=f"expand_{i}")(y))
            x += nn.Dense(self.width, name=f"contract_{i}")(y)
            if (i + 1) % self.exchange_every == 0 or i == self.depth - 1:
                # [B,T,C] -> [B,C,T]: the same token MLP is used per channel.
                y = nn.LayerNorm(name=f"exchange_norm_{i}")(x)
                y = jnp.swapaxes(y, -1, -2)
                y = nn.gelu(nn.Dense(2 * x.shape[1], name=f"token_expand_{i}")(y))
                y = nn.Dense(x.shape[1], name=f"token_contract_{i}")(y)
                x += jnp.swapaxes(y, -1, -2)
        return nn.LayerNorm(name="output_norm")(x)


class GoalEncoder(nn.Module):
    width: int
    latent_dim: int

    @nn.compact
    def __call__(self, state):
        # Actor-facing features precede the separate contrastive projection head.
        h = nn.gelu(nn.Dense(self.width)(state))
        h = nn.LayerNorm()(nn.Dense(self.width)(h))
        z = nn.Dense(self.latent_dim)(nn.gelu(nn.Dense(self.width)(h)))
        return h, z


class TokenCritic(nn.Module):
    width: int = 128
    depth: int = 2
    exchange_every: int = 2
    latent_dim: int = 128

    def setup(self):
        self.goal_encoder = GoalEncoder(self.width, self.latent_dim)
        self.backbones = [
            TokenBackbone(
                self.width, self.depth, self.exchange_every, name=f"tokens_{i}"
            )
            for i in range(2)
        ]
        self.phi_heads = [nn.Dense(self.latent_dim, name=f"phi_{i}") for i in range(2)]
        self.psi_heads = [nn.Dense(self.latent_dim, name=f"psi_{i}") for i in range(2)]

    def encode(self, states):
        return self.goal_encoder(states)[0]

    def __call__(self, states, actions, goals):
        _, z = self.goal_encoder(goals)
        phi = jnp.stack(
            [
                head(nn.gelu(net(states, actions)[:, 0]))
                for net, head in zip(self.backbones, self.phi_heads)
            ]
        )
        psi = jnp.stack([head(z) for head in self.psi_heads])
        return phi, psi


class TokenActor(nn.Module):
    action_dim: int
    width: int = 128
    depth: int = 2
    exchange_every: int = 2

    @nn.compact
    def __call__(self, states, goals, embeddings, actions, time):
        # Raw goals, representation features, and time condition all token positions.
        t = time[:, None]
        condition = jnp.concatenate(
            [goals, embeddings, t, jnp.sin(jnp.pi * t), jnp.cos(jnp.pi * t)], axis=-1
        )
        x = TokenBackbone(self.width, self.depth, self.exchange_every)(
            states, actions, condition
        )
        return nn.Dense(self.action_dim, kernel_init=nn.initializers.normal(0.01))(
            x[:, 1:]
        )
