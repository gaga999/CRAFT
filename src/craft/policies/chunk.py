"""Prior-to-action composition, independent of training objectives."""

import jax.numpy as jnp

from craft.models.cvae import critic_condition, sample_gaussian


def sample_action_chunk(
    critic, prior, decoder, observations, goals, rng, *, representation
):
    single = observations.ndim == 1
    if single:
        observations, goals = observations[None], goals[None]
    condition = critic_condition(critic, representation, observations, goals)
    parameters = prior(observations, goals, condition)
    actions = decoder(
        sample_gaussian(parameters, rng),
        observations,
        goals,
        jnp.ones((observations.shape[0], 1, 1)),
        condition,
    )
    return actions[0] if single else actions
