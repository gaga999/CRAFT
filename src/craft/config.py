"""Strict experiment schema and validation before side effects."""

import math
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class AgentConfig:
    chunk_length: int = 3
    width: int = 128
    depth: int = 4
    exchange_every: int = 2
    latent_dim: int = 128
    learning_rate: float = 0.0003
    flow_steps: int = 10
    distillation_weight: float = 0.1
    max_grad_norm: float = 10.0


@dataclass
class DataConfig:
    name: str = "puzzle-3x3-noisy-v0"
    directory: str = "data/ogbench"
    path: str | None = None
    synthetic: bool = False
    batch_size: int = 256
    discount: float = 0.95


@dataclass
class StageConfig:
    name: str = "joint"
    steps: int = 1000000


@dataclass
class TrainConfig:
    stages: list[StageConfig] = field(default_factory=lambda: [StageConfig()])
    log_interval: int = 1000
    eval_interval: int = 50000
    save_interval: int = 10000


@dataclass
class EvaluationConfig:
    episodes: int = 10
    execution_length: int = 1
    task_ids: list[int] = field(default_factory=list)


@dataclass
class LoggingConfig:
    backend: str = "console"
    project: str = "craft"


@dataclass
class ExperimentConfig:
    seed: int = 0
    output_dir: str = "runs/puzzle"
    agent: AgentConfig = field(default_factory=AgentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def finite_float(value, name, minimum=0, strict=True):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (value <= minimum if strict else value < minimum)
    ):
        raise ValueError(
            f"{name} must be finite and {'>' if strict else '>='} {minimum}"
        )


def validate_agent(config):
    expected = set(OmegaConf.to_container(OmegaConf.structured(AgentConfig)))
    if set(config) != expected:
        raise ValueError(
            f"Agent configuration keys differ: {set(config) ^ expected}. Use distillation_weight, not alpha."
        )
    for key in (
        "chunk_length",
        "width",
        "depth",
        "exchange_every",
        "latent_dim",
        "flow_steps",
    ):
        positive_int(config[key], f"agent.{key}")
    for key in ("learning_rate", "max_grad_norm"):
        finite_float(config[key], f"agent.{key}")
    finite_float(
        config["distillation_weight"], "agent.distillation_weight", strict=False
    )


def load_config(path, overrides=()):
    # Merge first without a schema to retain original numeric types for validation.
    supplied = OmegaConf.merge(
        OmegaConf.load(path), OmegaConf.from_dotlist(list(overrides))
    )
    # Structured merge rejects unknown keys. Disable conversion during validation.
    merged = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), supplied)
    config = OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)
    raw = OmegaConf.to_container(supplied, resolve=True, throw_on_missing=True)

    def verify_types(source, defaults):
        if isinstance(source, dict):
            for key, value in source.items():
                verify_types(value, defaults[key])
        elif isinstance(source, list):
            for i, value in enumerate(source):
                verify_types(value, defaults[i])
        elif defaults is not None:
            if type(defaults) in (bool, int, str) and type(source) is not type(
                defaults
            ):
                raise ValueError(f"Configuration type mismatch: {source!r}")
            if type(defaults) is float and type(source) not in (float, int):
                raise ValueError(f"Expected numeric configuration, got {source!r}")

    verify_types(raw, config)
    validate_config(config)
    for key in ("output_dir",):
        config[key] = str(Path(config[key]).expanduser().resolve())
    config["data"]["directory"] = str(
        Path(config["data"]["directory"]).expanduser().resolve()
    )
    if config["data"]["path"] is not None:
        config["data"]["path"] = str(
            Path(config["data"]["path"]).expanduser().resolve()
        )
    return config


def validate_config(config):
    validate_agent(config["agent"])
    positive_int(config["seed"], "seed", 0)
    if config["seed"] >= 2**32:
        raise ValueError("seed must be less than 2**32")
    positive_int(config["data"]["batch_size"], "data.batch_size", 2)
    finite_float(config["data"]["discount"], "data.discount")
    if config["data"]["discount"] >= 1:
        raise ValueError("data.discount must be < 1")
    if not config["train"]["stages"]:
        raise ValueError("At least one training stage is required")
    for stage in config["train"]["stages"]:
        if stage["name"] not in ("joint", "representation", "action"):
            raise ValueError(f"Unknown stage: {stage['name']}")
        positive_int(stage["steps"], "stage.steps")
    if sum(s["steps"] for s in config["train"]["stages"]) >= 2**31:
        raise ValueError("Training schedule exceeds int32 step capacity")
    for key in ("log_interval", "eval_interval", "save_interval"):
        positive_int(config["train"][key], f"train.{key}")
    for key in ("episodes", "execution_length"):
        positive_int(config["evaluation"][key], f"evaluation.{key}")
    if config["evaluation"]["execution_length"] > config["agent"]["chunk_length"]:
        raise ValueError("execution_length must not exceed chunk_length")
    tasks = config["evaluation"]["task_ids"]
    for task in tasks:
        positive_int(task, "task_id")
    if len(set(tasks)) != len(tasks):
        raise ValueError("task_ids must be unique")
    if config["logging"]["backend"] not in ("console", "wandb"):
        raise ValueError("logging.backend must be console or wandb")
    for value in (
        config["output_dir"],
        config["logging"]["project"],
        config["data"]["directory"],
    ):
        if not value.strip():
            raise ValueError("Paths and logging project must not be empty")
    if config["data"]["synthetic"]:
        if config["agent"]["chunk_length"] > 32:
            raise ValueError("Synthetic episodes have only 32 transitions")
    else:
        name = config["data"]["name"]
        if not name.endswith("-v0") or any(
            mode in name.split("-")
            for mode in ("visual", "powderworld", "oraclerep", "singletask")
        ):
            raise ValueError(
                "Expected a standard state-based continuous-action OGBench dataset"
            )
        if config["data"]["path"] is not None:
            path = config["data"]["path"]
            if not isinstance(path, str) or Path(path).suffix != ".npz":
                raise ValueError("Custom dataset path must be a .npz filename")
