"""Inference composition: actor latent -> decoder chunk. No dataset actions."""

import jax
import jax.numpy as jnp


def actor_condition(representation, states, goals, inputs):
    """Detached representation features the actor reads (`agent.actor_inputs`)."""
    parts = []
    if "goal_embedding" in inputs:
        parts.append(representation.goal(goals))
    if "state_embedding" in inputs:
        parts.append(representation.state_features(states))
    return jax.lax.stop_gradient(jnp.concatenate(parts, -1)) if parts else None


def generate(representation, actor, states, goals, inputs):
    return actor(states, goals, actor_condition(representation, states, goals, inputs))


def act(representation, actor, observations, goals, inputs):
    single = observations.ndim == 1
    if single:
        observations, goals = observations[None], goals[None]
    z = generate(representation, actor, observations, goals, inputs)
    actions = representation.decode(observations, z)
    return actions[0] if single else actions
