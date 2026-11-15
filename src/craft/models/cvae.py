"""Token-wise conditional VAE; Gaussian latents exist at every action position.

`cvae_representation` selects where CVAE features come from. "critic": callers pass
critic features. "own": posterior/decoder encode raw inputs with their own copies of
the critic architecture they replace, so only the representation's training signal
differs; prior uses only raw state/goal.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from craft.models.mlp import MLP
from craft.models.tokens import MixerBlock, TokenBackbone


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


class StateEncoder(nnx.Module):
    """Same function class as the critic's psi feature path (`critic.encode`)."""

    def __init__(self, observation_dim, width, *, rngs):
        self.features = MLP(observation_dim, (width,), width, rngs=rngs)
        self.norm = nnx.LayerNorm(width, rngs=rngs)

    def __call__(self, states):
        return self.norm(self.features(states))


def critic_condition(critic, representation, states, goals):
    """Detached critic state/goal features for "critic" CVAEs; None for "own"."""
    if representation == "own":
        return None
    return jax.tree.map(
        jax.lax.stop_gradient, (critic.encode(states), critic.encode(goals))
    )


def _check_features(module, features):
    if (features is None) != (module.representation == "own"):
        raise ValueError(
            f"{type(module).__name__} with cvae_representation="
            f"{module.representation!r} {'rejects' if features is not None else 'requires'}"
            " critic features"
        )


class Posterior(nnx.Module):
    def __init__(self, observation_dim, action_dim, config, *, rngs):
        w, h = config["width"], config["chunk_length"]
        self.representation = config["cvae_representation"]
        if self.representation == "own":
            # Mirrors critic.features (phi backbone) and critic.encode (psi).
            self.backbone = TokenBackbone(
                observation_dim,
                action_dim,
                h,
                width=w,
                depth=config["depth"],
                exchange_every=config["exchange_every"],
                rngs=rngs,
            )
            self.goal_encoder = StateEncoder(observation_dim, w, rngs=rngs)
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

    def __call__(self, states, actions, goals, features=None):
        """features: (critic.features(s, a) [B,H+1,W], critic.encode(g) [B,W])."""
        _check_features(self, features)
        if features is None:
            features = self.backbone(states, actions), self.goal_encoder(goals)
        tokens, goal = features
        x = jnp.concatenate(
            [self.action_projection(tokens), self.goal_projection(goal)[:, None]],
            axis=1,
        )
        return self.head(self.mixer(x)[:, 1:-1])


class Prior(nnx.Module):
    def __init__(self, observation_dim, config, *, rngs):
        self.h, self.z = config["chunk_length"], config["vae_latent_dim"]
        w = config["width"]
        self.representation = config["cvae_representation"]
        features = 2 * w if self.representation == "critic" else 0
        self.body = MLP(2 * observation_dim + features, (w, w), self.h * w, rngs=rngs)
        self.head = GaussianHead(
            w, self.z, config["log_std_min"], config["log_std_max"], rngs=rngs
        )
        self.width = w

    def __call__(self, states, goals, features=None):
        """features: (critic.encode(s), critic.encode(g)), always detached here."""
        _check_features(self, features)
        condition = [states, goals]
        if features is not None:
            condition += [jax.lax.stop_gradient(x) for x in features]
        condition = jnp.concatenate(condition, -1)
        return self.head(
            self.body(condition).reshape(states.shape[0], self.h, self.width)
        )


class Decoder(nnx.Module):
    def __init__(self, observation_dim, action_dim, config, *, rngs):
        w, h = config["width"], config["chunk_length"]
        self.representation = config["cvae_representation"]
        if self.representation == "own":
            # Shared for state and goal, like critic.encode.
            self.encoder = StateEncoder(observation_dim, w, rngs=rngs)
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

    def __call__(self, z, states, goals, mask, features=None):
        """features: (critic.encode(s), critic.encode(g)) for "critic" decoders."""
        _check_features(self, features)
        if features is None:
            features = self.encoder(states), self.encoder(goals)
        state_features, goal_features = features
        c = self.state_condition(state_features) + self.goal_condition(goal_features)
        # Mask AFTER every condition projection, including biases; no bypass tokens.
        injected = mask * self.gate * self.condition(c)[:, None]
        x = self.latent(z) + self.position[...] + injected
        return jnp.tanh(self.output(self.mixer(x)))
