"""Minimal stage orchestration example; runtime services are added separately."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from craft.types import Metrics


@dataclass(frozen=True)
class Stage:
    name: str
    steps: int

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("Stage steps must be positive")


def train_offline(
    agent: Any,
    dataset: Any,
    stages: Sequence[Stage],
    *,
    batch_size: int,
    start_step: int = 0,
    on_step: Callable[[int, str, Any, Metrics], None] | None = None,
) -> Any:
    """Keep update math in the agent; callbacks receive global step and stage."""
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    step = 0
    if not 0 <= start_step <= sum(stage.steps for stage in stages):
        raise ValueError("Resume step outside stage schedule")
    for stage in stages:
        for _ in range(stage.steps):
            step += 1
            if step <= start_step:
                continue
            batch = dataset.sample(batch_size)
            agent, metrics = agent.update(batch, stage=stage.name)
            if on_step is not None:
                on_step(step, stage.name, agent, metrics)
    return agent
