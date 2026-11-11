"""Explicit construction of role-compatible, initialized NNX components."""

from craft.models.tokens import TokenCritic
from craft.policies.chunk import ChunkPolicy


def make_crl_models(config, observation_dim, action_dim, *, rngs):
    common = {key: config[key] for key in ("width", "depth", "exchange_every")}
    dims = (observation_dim, action_dim, config["chunk_length"])
    critic = TokenCritic(*dims, **common, latent_dim=config["latent_dim"], rngs=rngs)
    teacher = ChunkPolicy(*dims, **common, rngs=rngs)
    actor = ChunkPolicy(*dims, **common, rngs=rngs)
    return critic, teacher, actor
