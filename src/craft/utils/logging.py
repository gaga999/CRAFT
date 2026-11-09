"""Small logging example; a W&B adapter can expose the same log method."""

from craft.types import Metrics


class ConsoleLogger:
    def log(self, metrics: Metrics, *, step: int) -> None:
        print(f"step={step} {metrics}")
