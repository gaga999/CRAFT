"""Binary NCE over BxB pairwise logits; sum pairs per anchor, then average anchors."""

import jax
import jax.numpy as jnp


def contrastive_loss(logits):
    labels = 2 * jnp.eye(logits.shape[0]) - 1
    return -jax.nn.log_sigmoid(labels * logits).sum(-1).mean()


def pairwise_auc(logits):
    """P(true goal outscores another sample's goal) over all off-diagonal pairs."""
    n = logits.shape[0]
    wins = (jnp.diag(logits)[:, None] > logits) & ~jnp.eye(n, dtype=bool)
    return wins.sum() / (n * (n - 1))
