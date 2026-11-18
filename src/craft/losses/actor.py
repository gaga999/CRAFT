"""Latent actor objective: normalized critic score plus latent behavior cloning.

L = -score / sg(mean|score|) + bc_weight * mean((z_hat - sg(z_data))^2)

The score normalization follows the reference CRL+AC actor (scale-invariant Q
term), so bc_weight does not depend on the learned SigLIP temperature.
"""

import jax
import jax.numpy as jnp


def actor_loss(score, z_hat, z_data, bc_weight):
    scale = jax.lax.stop_gradient(jnp.abs(score).mean() + 1e-6)
    q_loss = -score.mean() / scale
    bc = jnp.square(z_hat - jax.lax.stop_gradient(z_data)).mean()
    return q_loss + bc_weight * bc, (q_loss, bc)
