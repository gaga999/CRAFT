"""Resolved-config training/evaluation entry points."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf

from craft.agents.craft import CRAFTAgent
from craft.data.offline import OfflineDataset, synthetic_dataset
from craft.envs.factory import make_env_and_datasets
from craft.evaluators.rollout import evaluate_rollouts
from craft.losses.contrastive import contrastive_loss
from craft.trainers.offline import Stage, train_offline
from craft.utils.checkpoint import restore_checkpoint, save_checkpoint
from craft.utils.logging import Logger


def run(evaluate=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restore", type=Path)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = OmegaConf.merge(
        OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides)
    )
    config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if args.dry_run:
        print(OmegaConf.to_yaml(config))
        return
    if evaluate and args.restore is None:
        parser.error("Evaluation requires --restore")
    stages = config["train"]["stages"]
    if any(
        s["name"] not in ("joint", "representation", "action") or s["steps"] <= 0
        for s in stages
    ):
        raise ValueError("Invalid training stage")
    if (
        config["data"]["batch_size"] < 2
        or not 1
        <= config["evaluation"]["execution_length"]
        <= config["agent"]["chunk_length"]
    ):
        raise ValueError("Invalid batch or execution length")
    if any(
        config["train"][k] <= 0
        for k in ("log_interval", "eval_interval", "save_interval")
    ):
        raise ValueError("Intervals must be positive")
    env = None
    logger = None
    try:
        if config["data"]["synthetic"]:
            data = OfflineDataset(
                synthetic_dataset(config["seed"]),
                config["agent"]["chunk_length"],
                config["seed"],
                config["data"]["discount"],
            )
            val = None
        else:
            env, data, val = make_env_and_datasets(
                config["data"]["name"],
                chunk_length=config["agent"]["chunk_length"],
                seed=config["seed"],
                discount=config["data"]["discount"],
                dataset_dir=config["data"]["directory"],
                dataset_path=config["data"]["path"],
            )
        agent = CRAFTAgent.create(
            config["seed"],
            data.data["observations"].shape[-1],
            data.data["actions"].shape[-1],
            config["agent"],
        )
        step = 0
        if args.restore:
            agent, saved = restore_checkpoint(args.restore, agent=agent)
            if saved["config"] != config:
                raise ValueError(
                    "Resume/evaluation requires the saved resolved configuration"
                )
            step = saved["step"]
            data.rng.bit_generator.state = saved["sampler_rng"]
            if val is not None and saved["validation_rng"] is not None:
                val.rng.bit_generator.state = saved["validation_rng"]
        directory = Path(config["output_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, directory / "config.yaml")
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        (directory / "revision.txt").write_text(revision or "uncommitted repository")
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        metadata = {
            "revision": revision,
            "dirty": bool(dirty),
            "python": platform.python_version(),
            "packages": {
                dist.metadata["Name"]: dist.version
                for dist in importlib.metadata.distributions()
            },
        }
        (directory / "environment.json").write_text(json.dumps(metadata, indent=2))
        logger = Logger(directory, config)

        def evaluation(current_agent):
            if env is None:
                raise ValueError(
                    "Environment evaluation is unavailable with synthetic data"
                )
            return evaluate_rollouts(
                current_agent.sample_action_chunk,
                env,
                episodes=config["evaluation"]["episodes"],
                seed=config["seed"],
                execution_length=config["evaluation"]["execution_length"],
            )

        if evaluate:
            logger.log(evaluation(agent), step=step)
            return

        def on_step(current_step, stage, current_agent, metrics):
            nonlocal step
            step = current_step
            if step % config["train"]["log_interval"] == 0:
                if val is not None:
                    vb = val.sample(config["data"]["batch_size"])
                    phi, psi = current_agent.critic.apply_fn(
                        {"params": current_agent.critic.params},
                        vb["observations"],
                        vb["action_chunks"],
                        vb["value_goals"],
                    )
                    _, vm = contrastive_loss(phi, psi)
                    metrics = {
                        **metrics,
                        **{f"validation/{k}": v for k, v in vm.items()},
                    }
                logger.log(metrics, step=step)
            if env is not None and step % config["train"]["eval_interval"] == 0:
                logger.log(evaluation(current_agent), step=step)
            if step % config["train"]["save_interval"] == 0:
                save_checkpoint(
                    directory / "checkpoint.msgpack",
                    agent=current_agent,
                    run_state={
                        "step": step,
                        "config": config,
                        "sampler_rng": data.rng.bit_generator.state,
                        "validation_rng": None
                        if val is None
                        else val.rng.bit_generator.state,
                    },
                )

        agent = train_offline(
            agent,
            data,
            [Stage(**stage) for stage in stages],
            batch_size=config["data"]["batch_size"],
            start_step=step,
            on_step=on_step,
        )
        save_checkpoint(
            directory / "checkpoint.msgpack",
            agent=agent,
            run_state={
                "step": step,
                "config": config,
                "sampler_rng": data.rng.bit_generator.state,
                "validation_rng": None if val is None else val.rng.bit_generator.state,
            },
        )
        # Block asynchronous device work before reporting successful completion.
        jax.block_until_ready(agent)
        if not all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree.leaves(agent)):
            raise FloatingPointError("Nonfinite agent state")
    finally:
        if logger is not None:
            logger.close()
        if env is not None:
            env.close()


def train_main():
    run()


def evaluate_main():
    run(evaluate=True)
