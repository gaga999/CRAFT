"""SigLIP-style BxB logits; sum pairs per anchor, then average anchors."""

import jax
import jax.numpy as jnp

from craft.models.similarity import siglip_logits


def contrastive_loss(phi, psi, log_scale, bias):
    logits = siglip_logits(phi, psi, log_scale, bias, pairwise=True)
    n = phi.shape[0]
    eye = jnp.eye(n)
    labels = 2 * eye - 1
    loss = -jax.nn.log_sigmoid(labels * logits).sum(-1).mean()
    return loss, {
        "crl/loss": loss,
        "crl/scale": jnp.exp(log_scale),
        "crl/bias": bias,
        "crl/accuracy": (logits.argmax(-1) == jnp.arange(n)).mean(),
        "crl/binary_accuracy": ((logits > 0) == (eye > 0)).mean(),
        "crl/logit_positive": jnp.diag(logits).mean(),
        "crl/logit_negative": (logits * (1 - eye)).sum() / (n * (n - 1)),
    }
