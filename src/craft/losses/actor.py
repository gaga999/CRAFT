"""Latent actor objective: normalized critic score plus (weighted) latent BC.

L = -score / sg(mean|score|) + bc_weight * mean_i(w_i * mean((z_hat_i - sg(z_i))^2))

The score normalization follows the reference CRL+AC actor (scale-invariant Q
term). w_i = 1, or advantage weights exp(beta * A_i) clipped at max_weight,
with A_i = (score(s, z_data, g) - baseline) / sg(mean|score(s, z_data, g)|).
"""

import jax
import jax.numpy as jnp


def advantage_weights(data_score, baseline, beta, max_weight):
    scale = jax.lax.stop_gradient(jnp.abs(data_score).mean() + 1e-6)
    advantage = jax.lax.stop_gradient((data_score - baseline) / scale)
    weights = jnp.minimum(jnp.exp(beta * advantage), max_weight)
    return weights, advantage


def actor_loss(score, z_hat, z_data, bc_weight, weights=None):
    scale = jax.lax.stop_gradient(jnp.abs(score).mean() + 1e-6)
    q_loss = -score.mean() / scale
    per_sample = jnp.square(z_hat - jax.lax.stop_gradient(z_data)).mean((1, 2))
    if weights is not None:
        per_sample = jax.lax.stop_gradient(weights) * per_sample
    bc = per_sample.mean()
    return q_loss + bc_weight * bc, (q_loss, bc)
