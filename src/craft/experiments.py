"""Experiment assembly; session-local services keep source runs immutable."""

import json
from math import gcd
from pathlib import Path

from craft.agents.craft import CRAFTAgent
from craft.config import validate_config
from craft.data.offline import OfflineDataset, synthetic_dataset
from craft.envs.factory import make_env, make_env_and_datasets
from craft.evaluators.rollout import evaluate_rollouts
from craft.trainers.offline import train_offline
from craft.utils.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from craft.utils.logging import Logger
from craft.utils.random import make_random_engines
from craft.utils.runs import allocate_session, provenance, write_json


def checkpoint_metadata(agent, config, training_engine, identity, environment):
    return {
        "step": int(agent.step[...]),
        "config": config,
        "dimensions": [agent.observation_dim, agent.action_dim],
        "dataset_identity": identity,
        "training_engine": training_engine.numpy_state(),
        "source_sha256": environment["source_sha256"],
    }


def check_restore(config, saved, *, evaluate):
    fields = (
        ("seed", "agent", "data", "train", "output_dir") if evaluate else tuple(config)
    )

    def contract(value, key):
        if key == "agent":
            # Checkpoints predating these options used critic features and
            # sent guidance gradients through the prior std.
            return {"cvae_representation": "critic", "guide_through_std": True, **value}
        if key == "data":
            return {
                k: v
                for k, v in value.items()
                if k not in ("workers", "prefetch_batches", "device_prefetch")
            }
        return value

    if any(
        contract(config[key], key) != contract(saved["config"][key], key)
        for key in fields
    ):
        raise ValueError(
            "Checkpoint/config mismatch; resume needs its original configuration except data prefetch settings, evaluation may also change evaluation/logging settings"
        )
    total = sum(s["steps"] for s in saved["config"]["train"]["stages"])
    if type(saved["step"]) is not int or not 0 <= saved["step"] <= total:
        raise ValueError("Checkpoint step is outside its training schedule")


def run_experiment(
    config,
    *,
    checkpoint=None,
    evaluate=False,
    max_steps=None,
    destination=None,
    allow_code_change=False,
):
    validate_config(config)
    if not evaluate and config["data"]["path"] is not None:
        path = Path(config["data"]["path"])
        if not path.is_file() or not path.with_name(path.stem + "-val.npz").is_file():
            raise ValueError(
                "Custom dataset requires training and matching -val.npz files"
            )
    if evaluate and (checkpoint is None or config["data"]["synthetic"]):
        raise ValueError("Evaluation requires a checkpoint and real environment")
    if max_steps is not None and (
        type(max_steps) is not int or max_steps < 1 or evaluate
    ):
        raise ValueError("--steps must be positive and is training-only")
    environment = provenance()
    saved = None
    if checkpoint is not None:
        payload = read_checkpoint(checkpoint)
        saved = json.loads(payload["run_state"])
        check_restore(config, saved, evaluate=evaluate)
        if (
            not evaluate
            and saved["source_sha256"] != environment["source_sha256"]
            and not allow_code_change
        ):
            raise ValueError(
                "Training code changed since checkpoint; inspect changes and explicitly pass --allow-code-change to resume"
            )
    elif allow_code_change:
        raise ValueError("--allow-code-change requires a resume checkpoint")
    if not evaluate and destination is not None:
        raise ValueError(
            "--output-dir is evaluation-only; use output_dir=... for new training"
        )
    training_engine, evaluation_engine = make_random_engines(config["seed"])
    env = logger = session = agent = data = val = None
    identity = None
    try:
        if saved is None:
            config = dict(config)
            session = allocate_session(config, mode="train")
            config["output_dir"] = str(session)
        if evaluate:
            env = make_env(config["data"]["name"])
            observation_dim, action_dim = saved["dimensions"]
            if env.observation_space.shape != (
                observation_dim,
            ) or env.action_space.shape != (action_dim,):
                raise ValueError("Checkpoint/environment dimensions differ")
        else:
            if config["data"]["synthetic"]:
                data = OfflineDataset(
                    synthetic_dataset(rng=training_engine.numpy("synthetic")),
                    config["agent"]["chunk_length"],
                    config["data"]["discount"],
                    rng=training_engine.numpy("data"),
                )
            else:
                env, data, val = make_env_and_datasets(
                    config["data"]["name"],
                    chunk_length=config["agent"]["chunk_length"],
                    rng=training_engine.numpy("data"),
                    validation_rng=training_engine.numpy("validation"),
                    discount=config["data"]["discount"],
                    dataset_dir=config["data"]["directory"],
                    dataset_path=config["data"]["path"],
                )
            observation_dim = data.data["observations"].shape[-1]
            action_dim = data.data["actions"].shape[-1]
            identity = {
                "train": data.fingerprint(),
                "validation": None if val is None else val.fingerprint(),
            }
            if saved is not None and (
                identity != saved["dataset_identity"]
                or [observation_dim, action_dim] != saved["dimensions"]
            ):
                raise ValueError("Dataset contents/dimensions changed since checkpoint")
        if env is not None:
            available = len(env.unwrapped.task_infos)
            if any(task > available for task in config["evaluation"]["task_ids"]):
                raise ValueError("evaluation.task_ids exceeds environment task count")
        agent = CRAFTAgent.create(
            observation_dim,
            action_dim,
            config["agent"],
            rngs=training_engine.jax,
            batch_size=config["data"]["batch_size"],
        )
        if saved is not None:
            restored = restore_checkpoint(checkpoint, agent=agent, payload=payload)
            if restored != saved or int(agent.step[...]) != saved["step"]:
                raise ValueError("Checkpoint metadata and agent step disagree")
            if not evaluate:
                training_engine.restore_numpy(saved["training_engine"])
            session = allocate_session(
                config,
                mode="evaluate" if evaluate else "resume",
                checkpoint=checkpoint,
                destination=destination,
                allow_code_change=allow_code_change,
                checkpoint_sha256=payload["checkpoint_sha256"],
            )
        write_json(
            session / "dataset.json",
            identity if identity is not None else saved["dataset_identity"],
        )
        logger = Logger(session, config)

        def evaluation_metrics():
            return evaluate_rollouts(
                agent.sample_action_chunk,
                env,
                engine=evaluation_engine,
                **config["evaluation"],
            )

        if evaluate:
            logger.log(evaluation_metrics(), step=saved["step"])
            write_json(
                session / "status.json", {"status": "completed", "step": saved["step"]}
            )
            return session

        def save():
            save_checkpoint(
                session / "checkpoint.msgpack",
                agent=agent,
                run_state=checkpoint_metadata(
                    agent, config, training_engine, identity, environment
                ),
            )

        def on_step(step, stage, metrics):
            # One merged log per step avoids W&B dropping same-step evaluation.
            metrics = dict(metrics)
            log_due = step % config["train"]["log_interval"] == 0
            eval_due = env is not None and step % config["train"]["eval_interval"] == 0
            if log_due and val is not None:
                metrics.update(
                    agent.validation_metrics(
                        val.sample(
                            config["data"]["batch_size"], include_actor_goals=False
                        ),
                        evaluation_engine.task_key(0),
                    )
                )
            if eval_due:
                metrics.update(evaluation_metrics())
            if log_due or eval_due:
                logger.log(metrics, step=step)
            if step % config["train"]["save_interval"] == 0:
                save()

        try:
            train_offline(
                agent,
                data,
                config["train"]["stages"],
                batch_size=config["data"]["batch_size"],
                max_steps=max_steps,
                on_step=on_step,
                workers=config["data"]["workers"],
                prefetch_batches=config["data"]["prefetch_batches"],
                device_prefetch=config["data"]["device_prefetch"],
                # Every log/eval/save step is a chunk end, so callbacks are unchanged.
                steps_per_call=min(
                    100,
                    gcd(
                        config["train"]["log_interval"],
                        config["train"]["eval_interval"],
                        config["train"]["save_interval"],
                    ),
                ),
            )
        except KeyboardInterrupt:
            save()
            write_json(
                session / "status.json",
                {"status": "interrupted", "step": int(agent.step[...])},
            )
            raise
        save()
        status = (
            "completed"
            if int(agent.step[...])
            == sum(stage["steps"] for stage in config["train"]["stages"])
            else "stopped"
        )
        write_json(
            session / "status.json", {"status": status, "step": int(agent.step[...])}
        )
        return session
    except KeyboardInterrupt:
        if session is not None:
            write_json(
                session / "status.json",
                {
                    "status": "interrupted",
                    "step": 0 if agent is None else int(agent.step[...]),
                },
            )
        raise
    except Exception as exc:
        if session is not None:
            write_json(
                session / "status.json",
                {
                    "status": "failed",
                    "step": 0 if agent is None else int(agent.step[...]),
                    "error": str(exc),
                },
            )
        raise
    finally:
        try:
            if logger is not None:
                logger.close()
        finally:
            if env is not None:
                env.close()
