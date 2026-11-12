"""Token-wise conditional VAE; Gaussian latents exist at every action position."""

import jax
import jax.numpy as jnp
from flax import nnx

from craft.models.mlp import MLP
from craft.models.tokens import MixerBlock


class TokenMixer(nnx.Module):
    def __init__(self, tokens, width, depth, every, *, rngs):
        self.blocks = nnx.List(
            [
                MixerBlock(
                    width, tokens, (i + 1) % every == 0 or i == depth - 1, rngs=rngs
                )
                for i in range(depth)
            ]
        )
        self.norm = nnx.LayerNorm(width, rngs=rngs)

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class GaussianHead(nnx.Module):
    def __init__(self, width, latent, low, high, *, rngs):
        self.output = nnx.Linear(
            width, 2 * latent, kernel_init=nnx.initializers.normal(0.01), rngs=rngs
        )
        self.low, self.high = low, high
        self.offset = float(jnp.log(-low / high))

    def __call__(self, x):
        mean, raw = jnp.split(self.output(x), 2, axis=-1)
        log_std = self.low + (self.high - self.low) * jax.nn.sigmoid(raw + self.offset)
        return mean, log_std


def sample_gaussian(parameters, key):
    mean, log_std = parameters
    return mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)


class Posterior(nnx.Module):
    def __init__(self, config, *, rngs):
        w, h = config["width"], config["chunk_length"]
        self.action_projection = nnx.Linear(w, w, rngs=rngs)
        self.goal_projection = nnx.Linear(w, w, rngs=rngs)
        self.mixer = TokenMixer(
            h + 2, w, config["fusion_depth"], config["exchange_every"], rngs=rngs
        )
        self.head = GaussianHead(
            w,
            config["vae_latent_dim"],
            config["log_std_min"],
            config["log_std_max"],
            rngs=rngs,
        )

    def __call__(self, tokens, goal):
        x = jnp.concatenate(
            [self.action_projection(tokens), self.goal_projection(goal)[:, None]],
            axis=1,
        )
        return self.head(self.mixer(x)[:, 1:-1])


class Prior(nnx.Module):
    def __init__(self, observation_dim, config, *, rngs):
        self.h, self.z = config["chunk_length"], config["vae_latent_dim"]
        w = config["width"]
        self.body = MLP(2 * observation_dim + 2 * w, (w, w), self.h * w, rngs=rngs)
        self.head = GaussianHead(
            w, self.z, config["log_std_min"], config["log_std_max"], rngs=rngs
        )
        self.width = w

    def __call__(self, states, goals, state_features, goal_features):
        features = jax.lax.stop_gradient(
            jnp.concatenate([state_features, goal_features], -1)
        )
        condition = jnp.concatenate([states, goals, features], -1)
        return self.head(
            self.body(condition).reshape(states.shape[0], self.h, self.width)
        )


class Decoder(nnx.Module):
    def __init__(self, action_dim, config, *, rngs):
        w, h = config["width"], config["chunk_length"]
        self.state_condition = nnx.Linear(w, w, rngs=rngs)
        self.goal_condition = nnx.Linear(w, w, rngs=rngs)
        self.condition = nnx.Linear(w, w, rngs=rngs)
        self.latent = nnx.Linear(config["vae_latent_dim"], w, rngs=rngs)
        self.position = nnx.Param(
            nnx.initializers.normal(0.02)(rngs.params(), (1, h, w))
        )
        self.log_gate = nnx.Param(jnp.log(jnp.array(0.1, jnp.float32)))
        self.mixer = TokenMixer(
            h, w, config["depth"], config["exchange_every"], rngs=rngs
        )
        self.output = nnx.Linear(
            w, action_dim, kernel_init=nnx.initializers.normal(0.01), rngs=rngs
        )

    @property
    def gate(self):
        return jnp.exp(self.log_gate[...])

    def __call__(self, z, state_features, goal_features, mask):
        c = self.state_condition(state_features) + self.goal_condition(goal_features)
        # Mask AFTER every condition projection, including biases; no bypass tokens.
        injected = mask * self.gate * self.condition(c)[:, None]
        x = self.latent(z) + self.position[...] + injected
        return jnp.tanh(self.output(self.mixer(x)))
