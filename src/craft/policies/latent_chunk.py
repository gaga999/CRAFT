"""Bind the current agent to a parameter-consistent inference interface."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LatentChunkPolicy:
    """Compatibility facade; this baseline generates actions, not action latents."""

    agent: Any

    def sample_action_chunk(self, observations, goals, rng):
        return self.agent.sample_action_chunk(observations, goals, rng)
