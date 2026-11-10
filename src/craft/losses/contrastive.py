"""Binary NCE as in the reference action-chunked CRL implementation."""

import jax.numpy as jnp
import optax


def contrastive_loss(phi, psi):
    logits = jnp.einsum("ebd,ecd->ebc", phi, psi) / jnp.sqrt(phi.shape[-1])
    labels = jnp.eye(phi.shape[1])[None]
    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    return loss, {
        "critic/loss": loss,
        "critic/phi_norm": jnp.linalg.norm(phi, axis=-1).mean(),
        "critic/psi_norm": jnp.linalg.norm(psi, axis=-1).mean(),
        "critic/psi_std": jnp.std(psi, axis=1).mean(),
        "critic/accuracy": (logits.argmax(-1) == jnp.arange(phi.shape[1])).mean(),
        "critic/positive": jnp.diagonal(logits, axis1=1, axis2=2).mean(),
    }


def q_values(phi, psi):
    return (phi * psi).sum(-1) / jnp.sqrt(phi.shape[-1])
