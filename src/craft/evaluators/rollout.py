"""Task-local evaluation RNG and bounded episode-local chunk execution."""

import jax
import numpy as np


def evaluate_rollouts(
    act, env, *, engine, episodes=10, execution_length=1, task_ids=()
):
    if episodes <= 0 or execution_length <= 0:
        raise ValueError("Episodes and execution length must be positive")
    tasks = env.unwrapped.task_infos
    selected = list(task_ids) if task_ids else list(range(1, len(tasks) + 1))
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(task < 1 or task > len(tasks) for task in selected)
    ):
        raise ValueError("Invalid evaluation task IDs")
    metrics = {}
    limit = getattr(getattr(env, "spec", None), "max_episode_steps", None)
    if limit is None:
        raise ValueError("Evaluation requires an environment time limit")
    for task_id in selected:
        successes = []
        rng = engine.task_key(task_id)
        for episode in range(episodes):
            observation, info = env.reset(
                seed=engine.episode_seed(task_id, episode), options={"task_id": task_id}
            )
            goal = info["goal"]
            if np.shape(goal) != np.shape(observation) or not np.isfinite(goal).all():
                raise ValueError("Goal must be finite and share the observation space")
            success = 0.0
            chunk = None
            position = 0
            for _ in range(limit):
                if chunk is None or position == execution_length:
                    rng, key = jax.random.split(rng)
                    chunk = np.asarray(act(observation, goal, key))
                    if (
                        chunk.ndim != 2
                        or chunk.shape[1:] != env.action_space.shape
                        or len(chunk) < execution_length
                        or not np.isfinite(chunk).all()
                        or np.any(np.abs(chunk) > 1.0001)
                    ):
                        raise ValueError(
                            "Policy returned an invalid/nonfinite/out-of-bounds action chunk"
                        )
                    position = 0
                observation, _, terminated, truncated, info = env.step(chunk[position])
                position += 1
                if "success" not in info or not np.isfinite(info["success"]):
                    raise ValueError("Environment must expose finite success metrics")
                success = max(success, float(info["success"]))
                if terminated or truncated:
                    break
            else:
                raise RuntimeError(
                    "Environment failed to terminate/truncate within its time limit"
                )
            successes.append(success)
        metrics[f"evaluation/{tasks[task_id - 1]['task_name']}/success"] = float(
            np.mean(successes)
        )
    metrics["evaluation/overall_success"] = float(np.mean(list(metrics.values())))
    return metrics
