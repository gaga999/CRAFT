"""Explicit construction of role-compatible, initialized NNX components."""

from flax import nnx

from craft.models.tokens import TokenCritic
from craft.policies.chunk import ChunkPolicy


def make_crl_models(config, observation_dim, action_dim, seed):
    common = {key: config[key] for key in ("width", "depth", "exchange_every")}
    dims = (observation_dim, action_dim, config["chunk_length"])
    critic = TokenCritic(
        *dims, **common, latent_dim=config["latent_dim"], rngs=nnx.Rngs(seed)
    )
    teacher = ChunkPolicy(*dims, **common, rngs=nnx.Rngs(seed + 1))
    actor = ChunkPolicy(*dims, **common, rngs=nnx.Rngs(seed + 2))
    return critic, teacher, actor
