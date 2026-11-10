"""OGBench task evaluation with episode-local chunk caches."""

import jax
import numpy as np


def evaluate_rollouts(act, env, *, episodes=10, seed=0, execution_length=1):
    if episodes <= 0 or execution_length <= 0:
        raise ValueError("Episodes and execution length must be positive")
    tasks = env.unwrapped.task_infos
    metrics = {}
    rng = jax.random.PRNGKey(seed)
    for task_id, task in enumerate(tasks, 1):
        successes = []
        for episode in range(episodes):
            observation, info = env.reset(
                seed=seed + episode, options={"task_id": task_id}
            )
            goal = info["goal"]
            done = False
            success = 0.0
            chunk = None
            position = 0
            while not done:
                if chunk is None or position == min(execution_length, len(chunk)):
                    rng, key = jax.random.split(rng)
                    chunk = np.asarray(act(observation, goal, key))
                    position = 0
                observation, _, terminated, truncated, info = env.step(chunk[position])
                position += 1
                done = terminated or truncated
                success = max(success, float(info.get("success", 0)))
            successes.append(success)
        metrics[f"evaluation/{task['task_name']}/success"] = float(np.mean(successes))
    metrics["evaluation/overall_success"] = float(np.mean(list(metrics.values())))
    return metrics
