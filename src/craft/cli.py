"""Thin configuration/command entry points."""

import argparse

from omegaconf import OmegaConf

from craft.config import load_config, validate_config
from craft.experiments import run_experiment


def run(evaluate=False):
    parser = argparse.ArgumentParser(description="CRAFT NNX offline CRL")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restore")
    parser.add_argument(
        "--steps", type=int, help="Maximum additional updates for this invocation"
    )
    parser.add_argument(
        "--output-dir", help="Fresh output directory for standalone evaluation"
    )
    parser.add_argument("--allow-code-change", action="store_true")
    parser.add_argument(
        "--name", help="W&B run display name (same as logging.name=...)"
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    if args.name is not None:
        # Set after parsing so names like "0.1" or "a=b" stay literal strings.
        config["logging"]["name"] = args.name
        validate_config(config)
    if args.dry_run:
        print(OmegaConf.to_yaml(config))
        return
    session = run_experiment(
        config,
        checkpoint=args.restore,
        evaluate=evaluate,
        max_steps=args.steps,
        destination=args.output_dir,
        allow_code_change=args.allow_code_change,
    )
    print(f"Session saved to {session}")


def train_main():
    run()


def evaluate_main():
    run(evaluate=True)
