"""Prior-to-action composition, independent of training objectives."""

import jax.numpy as jnp

from craft.models.cvae import sample_gaussian


def sample_action_chunk(critic, prior, decoder, observations, goals, rng):
    single = observations.ndim == 1
    if single:
        observations, goals = observations[None], goals[None]
    hs, hg = critic.encode(observations), critic.encode(goals)
    parameters = prior(observations, goals, hs, hg)
    actions = decoder(
        sample_gaussian(parameters, rng),
        hs,
        hg,
        jnp.ones((observations.shape[0], 1, 1)),
    )
    return actions[0] if single else actions
