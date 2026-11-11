"""Diagonal Gaussian KL: sum over time/latent dimensions, mean over batch."""

import jax.numpy as jnp


def gaussian_kl(q, p):
    qm, ql = q
    pm, pl = p
    element = (
        pl
        - ql
        + 0.5 * (jnp.exp(2 * (ql - pl)) + jnp.square(qm - pm) * jnp.exp(-2 * pl) - 1)
    )
    return element.sum(axis=(-1, -2)).mean()
