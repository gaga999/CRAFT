"""SigLIP-style BxB logits; sum pairs per anchor, then average anchors."""

import jax
import jax.numpy as jnp


def contrastive_loss(phi, psi, log_scale, bias):
    logits = jnp.exp(log_scale) * (phi @ psi.T) + bias
    labels = 2 * jnp.eye(phi.shape[0]) - 1
    loss = -jax.nn.log_sigmoid(labels * logits).sum(-1).mean()
    return loss, {
        "critic/loss": loss,
        "critic/scale": jnp.exp(log_scale),
        "critic/bias": bias,
        "critic/accuracy": (logits.argmax(-1) == jnp.arange(phi.shape[0])).mean(),
        "critic/positive": jnp.diag(logits).mean(),
    }
