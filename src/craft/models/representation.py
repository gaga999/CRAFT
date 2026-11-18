"""CRL representation whose action latent is also the decoder/actor space.

(s, A) -> token backbone -> per-action latent z [B,H,Z] (parameter-free LayerNorm)
s -> state encoder e_s;  phi = unit(head([flatten(z); e_s]));  psi = unit(psi(g))
SigLIP logit = exp(log_scale) * phi.psi + bias;  decoder(s, z) -> A.
"""

import jax.numpy as jnp
from flax import nnx

from craft.models.mlp import MLP
from craft.models.similarity import siglip_logits
from craft.models.tokens import TokenBackbone, TokenMixer


def unit(x):
    return x / jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + 1e-8)


def normalize_latent(x):
    """Parameter-free LayerNorm per action token: mean 0, RMS 1 over Z.

    Shared by encoder and actor, so generated latents lie on the encoder's
    output set (a (Z-2)-sphere of radius sqrt(Z) per token).
    """
    x = x - x.mean(-1, keepdims=True)
    return x / jnp.sqrt(jnp.square(x).mean(-1, keepdims=True) + 1e-6)


class ChunkDecoder(nnx.Module):
    """(s, z) -> bounded chunk. State enters as a token AND as an additive skip."""

    def __init__(self, observation_dim, action_dim, config, *, rngs):
        w, h = config["width"], config["chunk_length"]
        self.state_token = nnx.Linear(observation_dim, w, rngs=rngs)
        self.state_skip = nnx.Linear(observation_dim, w, rngs=rngs)
        self.latent_projection = nnx.Linear(config["latent_dim"], w, rngs=rngs)
        self.position = nnx.Param(
            nnx.initializers.normal(0.02)(rngs.params(), (1, h + 1, w))
        )
        self.mixer = TokenMixer(
            h + 1, w, config["decoder_depth"], config["exchange_every"], rngs=rngs
        )
        self.output = nnx.Linear(
            w, action_dim, kernel_init=nnx.initializers.normal(0.01), rngs=rngs
        )

    def __call__(self, states, z):
        actions = self.latent_projection(z) + self.state_skip(states)[:, None]
        x = jnp.concatenate([self.state_token(states)[:, None], actions], 1)
        return jnp.tanh(self.output(self.mixer(x + self.position[...])[:, 1:]))


class Representation(nnx.Module):
    def __init__(self, observation_dim, action_dim, config, *, rngs, batch_size):
        w, h, z, d = (
            config[k] for k in ("width", "chunk_length", "latent_dim", "embedding_dim")
        )
        self.backbone = TokenBackbone(
            observation_dim,
            action_dim,
            h,
            width=w,
            depth=config["depth"],
            exchange_every=config["exchange_every"],
            rngs=rngs,
        )
        self.latent_projection = nnx.Linear(w, z, rngs=rngs)
        self.state_encoder = MLP(observation_dim, (w,), w, layer_norm=True, rngs=rngs)
        # Simplified nonlinear fuse: one hidden layer over [flatten(z); e_s].
        self.head = MLP(h * z + w, (w,), d, rngs=rngs)
        self.goal_encoder = MLP(observation_dim, (w, w), d, layer_norm=True, rngs=rngs)
        self.decoder = ChunkDecoder(observation_dim, action_dim, config, rngs=rngs)
        self.log_scale = nnx.Param(jnp.log(jnp.array(10.0, jnp.float32)))
        # Optimal binary-NCE logit offset for 1 positive : B-1 negatives.
        self.bias = nnx.Param(-jnp.log(jnp.array(batch_size - 1.0, jnp.float32)))

    def latent(self, states, actions):
        tokens = self.backbone(states, actions)[:, 1:]
        return normalize_latent(self.latent_projection(tokens))

    def state_features(self, states):
        return self.state_encoder(states)

    def embed(self, z, state_features):
        flat = z.reshape(z.shape[0], -1)
        return unit(self.head(jnp.concatenate([flat, state_features], -1)))

    def goal(self, goals):
        return unit(self.goal_encoder(goals))

    def phi(self, states, actions):
        return self.embed(self.latent(states, actions), self.state_features(states))

    def logits(self, phi, psi, *, pairwise=False):
        return siglip_logits(
            phi, psi, self.log_scale[...], self.bias[...], pairwise=pairwise
        )

    def score(self, states, z, goals):
        """Actor objective: scaled similarity of latent z under the TRUE state.

        The bias is a constant (no gradient) and is omitted so that score
        normalization reflects similarity, not the batch-size offset.
        """
        phi = self.embed(z, self.state_features(states))
        return jnp.exp(self.log_scale[...]) * jnp.sum(phi * self.goal(goals), -1)

    def decode(self, states, z):
        return self.decoder(states, z)


class LatentActor(nnx.Module):
    """Deterministic (s, g) -> z on the encoder's normalized latent set."""

    def __init__(self, observation_dim, config, *, rngs):
        self.shape = (config["chunk_length"], config["latent_dim"])
        self.network = MLP(
            2 * observation_dim,
            (config["actor_width"],) * config["actor_depth"],
            self.shape[0] * self.shape[1],
            layer_norm=True,
            rngs=rngs,
        )

    def __call__(self, states, goals):
        out = self.network(jnp.concatenate([states, goals], -1))
        return normalize_latent(out.reshape(states.shape[0], *self.shape))
