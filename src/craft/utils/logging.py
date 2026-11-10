"""Local JSONL metrics with an optional W&B backend."""

import json
from pathlib import Path


class Logger:
    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run = None
        if config["logging"]["backend"] == "wandb":
            import wandb

            self.run = wandb.init(
                project=config["logging"]["project"],
                config=config,
                dir=str(self.directory),
            )
        elif config["logging"]["backend"] != "console":
            raise ValueError("Unknown logging backend")
        self.stream = (self.directory / "metrics.jsonl").open("a")

    def log(self, metrics, *, step):
        metrics = {key: float(value) for key, value in metrics.items()}
        self.stream.write(json.dumps({"step": step, **metrics}, allow_nan=False) + "\n")
        self.stream.flush()
        if self.run is not None:
            self.run.log(metrics, step=step)
        print(f"step={step} {metrics}")

    def close(self):
        self.stream.close()
        if self.run is not None:
            self.run.finish()
