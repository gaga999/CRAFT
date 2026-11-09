"""Standalone objective example, not the CRAFT training objective."""

import jax.numpy as jnp
from jax import Array


def reconstruction_mse(prediction: Array, target: Array) -> Array:
    """Compute MSE for equal-shaped, fully valid action chunks."""
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")
    return jnp.mean(jnp.square(prediction - target))
