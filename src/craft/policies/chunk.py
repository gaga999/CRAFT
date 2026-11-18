"""Inference composition: actor latent -> decoder chunk. No dataset actions."""


def act(representation, actor, observations, goals):
    single = observations.ndim == 1
    if single:
        observations, goals = observations[None], goals[None]
    actions = representation.decode(observations, actor(observations, goals))
    return actions[0] if single else actions
