"""Checkpoint boundary; serialization format is not selected yet."""

from pathlib import Path
from typing import Any


def save_checkpoint(path: Path, *, agent: Any, run_state: Any) -> None:
    """Save agent plus stage/step, sampler RNG, and resolved config metadata."""
    raise NotImplementedError("Checkpoint persistence and restore are pending")
