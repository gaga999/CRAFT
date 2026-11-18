"""Action-chunk reconstruction error; mean over batch, time and action."""

import jax.numpy as jnp


def reconstruction_mse(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")
    return jnp.mean(jnp.square(prediction - target))
