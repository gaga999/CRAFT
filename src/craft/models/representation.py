"""CRL representation whose action latent is also the decoder/actor space.

(s, A) -> token backbone -> per-action latent z [B,H,Z] (parameter-free LayerNorm)
s -> state encoder e_s;  phi = head([flatten(z); e_s]);  psi = goal_encoder(g)
similarity "cosine": logit = t * unit(phi).unit(psi) + b, t = min(exp(log_scale), max_scale)
similarity "dot":    logit = phi.psi / sqrt(D) + b (confidence lives in the norms)
decoder(s, z) -> A.
"""

import math

import jax.numpy as jnp
from flax import nnx

from craft.models.mlp import MLP
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
        w, h = config["decoder_width"], config["chunk_length"]
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
        h, z, d = (config[k] for k in ("chunk_length", "latent_dim", "embedding_dim"))
        w = config["action_encoder_width"]
        self.backbone = TokenBackbone(
            observation_dim,
            action_dim,
            h,
            width=w,
            depth=config["action_encoder_depth"],
            exchange_every=config["exchange_every"],
            rngs=rngs,
        )
        self.latent_projection = nnx.Linear(w, z, rngs=rngs)
        sw = config["state_encoder_width"]
        self.state_encoder = MLP(
            observation_dim,
            (sw,) * config["state_encoder_depth"],
            sw,
            layer_norm=True,
            rngs=rngs,
        )
        # Nonlinear fuse over [flatten(z); e_s].
        self.head = MLP(
            h * z + sw, (config["head_width"],) * config["head_depth"], d, rngs=rngs
        )
        self.goal_encoder = MLP(
            observation_dim,
            (config["goal_encoder_width"],) * config["goal_encoder_depth"],
            d,
            layer_norm=True,
            rngs=rngs,
        )
        self.decoder = ChunkDecoder(observation_dim, action_dim, config, rngs=rngs)
        self.similarity = config["similarity"]
        if self.similarity == "cosine":
            self.log_scale = nnx.Param(jnp.log(jnp.array(10.0, jnp.float32)))
            # Upper bound on the temperature; above it log_scale gets no gradient.
            self.max_log_scale = math.log(config["max_scale"])
        else:
            # Reference CRL scaling: the norms carry confidence; no temperature.
            self.dot_scale = 1 / math.sqrt(d)
        # Optimal binary-NCE logit offset for 1 positive : B-1 negatives.
        self.bias = nnx.Param(-jnp.log(jnp.array(batch_size - 1.0, jnp.float32)))

    def _embedding(self, raw):
        return unit(raw) if self.similarity == "cosine" else raw

    def latent(self, states, actions):
        tokens = self.backbone(states, actions)[:, 1:]
        return normalize_latent(self.latent_projection(tokens))

    def state_features(self, states):
        return self.state_encoder(states)

    def head_output(self, z, state_features):
        """Unnormalized phi (its norm is a diagnostic in both similarity modes)."""
        return self.head(
            jnp.concatenate([z.reshape(z.shape[0], -1), state_features], -1)
        )

    def embed(self, z, state_features):
        return self._embedding(self.head_output(z, state_features))

    def goal(self, goals):
        return self._embedding(self.goal_encoder(goals))

    def phi(self, states, actions):
        return self.embed(self.latent(states, actions), self.state_features(states))

    def temperature(self):
        if self.similarity == "cosine":
            return jnp.exp(jnp.minimum(self.log_scale[...], self.max_log_scale))
        return self.dot_scale

    def logits(self, phi, psi, *, pairwise=False):
        similarity = (
            jnp.einsum("id,jd->ij", phi, psi) if pairwise else jnp.sum(phi * psi, -1)
        )
        return self.temperature() * similarity + self.bias[...]

    def score(self, states, z, goals):
        """Actor objective: the critic logit of latent z under the TRUE state.

        The bias is a constant (no gradient) and is omitted so that score
        normalization reflects similarity, not the batch-size offset.
        """
        phi = self.embed(z, self.state_features(states))
        return self.temperature() * jnp.sum(phi * self.goal(goals), -1)

    def decode(self, states, z):
        return self.decoder(states, z)


ACTOR_INPUTS = ("goal_embedding", "state_embedding")


def actor_condition_dim(config):
    sizes = {
        "goal_embedding": config["embedding_dim"],
        "state_embedding": config["state_encoder_width"],
    }
    return sum(sizes[name] for name in config["actor_inputs"])


class LatentActor(nnx.Module):
    """Deterministic (s, g[, condition]) -> z on the encoder's latent set.

    `condition` is the detached representation features named by
    `actor_inputs`; the actor never updates the representation.
    """

    def __init__(self, observation_dim, config, *, rngs):
        self.shape = (config["chunk_length"], config["latent_dim"])
        self.network = MLP(
            2 * observation_dim + actor_condition_dim(config),
            (config["actor_width"],) * config["actor_depth"],
            self.shape[0] * self.shape[1],
            layer_norm=True,
            rngs=rngs,
        )

    def __call__(self, states, goals, condition=None):
        parts = [states, goals] + ([] if condition is None else [condition])
        out = self.network(jnp.concatenate(parts, -1))
        return normalize_latent(out.reshape(states.shape[0], *self.shape))
