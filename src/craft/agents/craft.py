"""Research scaffold: algorithm-specific state and stage updates belong here."""

from dataclasses import dataclass
from typing import Any

from craft.policies.latent_chunk import LatentChunkPolicy
from craft.types import Batch, Metrics


@dataclass(frozen=True)
class CRAFTAgent:
    """Conceptual container; choose a JAX PyTree state before JIT compilation."""

    state: Any  # Parameters, optimizer states, target states, and RNG.
    policy: LatentChunkPolicy

    def update(self, batch: Batch, *, stage: str) -> tuple["CRAFTAgent", Metrics]:
        """Dispatch outside JIT; each stage defines its own update function."""
        raise NotImplementedError(
            f"Implement stage {stage!r}, including losses and gradient ownership."
        )
