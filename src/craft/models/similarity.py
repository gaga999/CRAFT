"""Shared SigLIP logits for aligned pairs and all-pairs contrastive learning."""

import jax.numpy as jnp


def siglip_logits(phi, psi, log_scale, bias, *, pairwise=False):
    similarity = (
        jnp.einsum("id,jd->ij", phi, psi) if pairwise else jnp.sum(phi * psi, axis=-1)
    )
    return jnp.exp(log_scale) * similarity + bias
