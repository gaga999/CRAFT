"""Explicit initialized NNX components for the CVAE POC."""

from craft.models.cvae import Decoder, Posterior, Prior
from craft.models.tokens import TokenCritic


def make_poc_models(config, observation_dim, action_dim, *, rngs, batch_size):
    common = {k: config[k] for k in ("width", "depth", "exchange_every", "latent_dim")}
    critic = TokenCritic(
        observation_dim,
        action_dim,
        config["chunk_length"],
        **common,
        batch_size=batch_size,
        rngs=rngs,
    )
    return (
        critic,
        Posterior(config, rngs=rngs),
        Decoder(action_dim, config, rngs=rngs),
        Prior(observation_dim, config, rngs=rngs),
    )
