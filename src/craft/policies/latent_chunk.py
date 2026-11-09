"""Illustrate a policy that composes multiple models to generate chunks."""

from dataclasses import dataclass
from typing import Any

from jax import Array


@dataclass(frozen=True)
class LatentChunkPolicy:
    """Inference logic only; the agent owns parameters and training state."""

    encoder: Any
    generator: Any
    decoder: Any

    def sample_action_chunk(
        self, params: Any, observations: Array, goals: Array, rng: Array
    ) -> Array:
        """Return [B, H, A]; supply a fresh RNG key for each call."""
        raise NotImplementedError(
            "Implement observation/goal encoding, latent generation, and decoding."
        )
