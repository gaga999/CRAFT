"""One metrics stream per session; no accidental append or cloud-run reuse."""

import json

import jax


class Logger:
    def __init__(self, directory, config):
        self.stream = (directory / "metrics.jsonl").open("x")
        self.last_step = -1
        self.run = None
        try:
            if config["logging"]["backend"] == "wandb":
                import wandb

                self.run = wandb.init(
                    project=config["logging"]["project"],
                    config=config,
                    dir=str(directory),
                    name=config["logging"]["name"] or directory.name,
                    resume="never",
                )
            elif config["logging"]["backend"] != "console":
                raise ValueError("Unknown logging backend")
        except BaseException:
            self.stream.close()
            raise

    def log(self, metrics, *, step):
        if step <= self.last_step:
            raise ValueError(
                "Metrics must have strictly increasing steps within a session"
            )
        metrics = {key: float(value) for key, value in jax.device_get(metrics).items()}
        line = json.dumps({"step": step, **metrics}, allow_nan=False)
        self.stream.write(line + "\n")
        self.stream.flush()
        if self.run is not None:
            self.run.log(metrics, step=step)
        self.last_step = step
        print(f"step={step} {metrics}", flush=True)

    def close(self):
        try:
            if self.run is not None:
                self.run.finish()
        finally:
            self.stream.close()
