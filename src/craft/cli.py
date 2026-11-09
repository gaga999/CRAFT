"""Configuration inspection entry points; training assembly is still pending."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def inspect_config(command: str) -> None:
    parser = argparse.ArgumentParser(description=f"CRAFT {command} scaffold")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    OmegaConf.resolve(config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    print(OmegaConf.to_yaml(config, resolve=True))
    if not args.dry_run:
        parser.error(
            "Research components and runtime assembly are not implemented yet. "
            "Use --dry-run to inspect the resolved configuration."
        )
    print("Configuration resolved. No training or evaluation was performed.")


def train_main() -> None:
    inspect_config("training")


def evaluate_main() -> None:
    inspect_config("evaluation")
