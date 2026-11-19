"""SigLIP-style BxB logits; sum pairs per anchor, then average anchors."""

import jax
import jax.numpy as jnp

from craft.models.similarity import siglip_logits


def contrastive_loss(phi, psi, log_scale, bias):
    logits = siglip_logits(phi, psi, log_scale, bias, pairwise=True)
    labels = 2 * jnp.eye(phi.shape[0]) - 1
    loss = -jax.nn.log_sigmoid(labels * logits).sum(-1).mean()
    return loss, {"crl/loss": loss, "crl/scale": jnp.exp(log_scale)}
