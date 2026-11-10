"""NNX action-token policy, shared by flow teacher and one-step student."""

import jax
import jax.numpy as jnp
from flax import nnx

from craft.models.tokens import TokenBackbone


def encode_condition(critic, observations, goals):
    return jax.lax.stop_gradient(
        jnp.concatenate([critic.encode(observations), critic.encode(goals)], -1)
    )


class ChunkPolicy(nnx.Module):
    def __init__(
        self,
        observation_dim,
        action_dim,
        chunk_length,
        *,
        width=128,
        depth=4,
        exchange_every=2,
        rngs,
    ):
        self.chunk_length = chunk_length
        self.action_dim = action_dim
        self.backbone = TokenBackbone(
            observation_dim,
            action_dim,
            chunk_length,
            width=width,
            depth=depth,
            exchange_every=exchange_every,
            condition_dim=observation_dim + 2 * width + 3,
            rngs=rngs,
        )
        self.output = nnx.Linear(
            width, action_dim, kernel_init=nnx.initializers.normal(0.01), rngs=rngs
        )

    def __call__(self, states, goals, embeddings, actions, time):
        t = time[:, None]
        condition = jnp.concatenate(
            [goals, embeddings, t, jnp.sin(jnp.pi * t), jnp.cos(jnp.pi * t)], -1
        )
        return self.output(self.backbone(states, actions, condition)[:, 1:])

    def sample(self, critic, observations, goals, rng):
        single = observations.ndim == 1
        if single:
            observations, goals = observations[None], goals[None]
        noise = jax.random.normal(
            rng, (observations.shape[0], self.chunk_length, self.action_dim)
        )
        actions = jnp.clip(
            self(
                observations,
                goals,
                encode_condition(critic, observations, goals),
                noise,
                jnp.zeros(observations.shape[0]),
            ),
            -1,
            1,
        )
        return actions[0] if single else actions


def integrate_flow(teacher, observations, goals, embeddings, noise, steps):
    def step(x, i):
        t = jnp.full((x.shape[0],), i / steps)
        return x + teacher(observations, goals, embeddings, x, t) / steps, None

    return jnp.clip(jax.lax.scan(step, noise, jnp.arange(steps))[0], -1, 1)
