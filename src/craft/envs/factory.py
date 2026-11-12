"""Official OGBench adapters with resource-safe construction and shape checks."""

import numpy as np

from craft.data.loading import load_ogbench_datasets


def make_env(name):
    if any(
        mode in name.split("-")
        for mode in ("visual", "powderworld", "oraclerep", "singletask")
    ):
        raise ValueError(
            "Only standard state-based goal-conditioned continuous datasets are supported"
        )
    import ogbench
    from gymnasium.spaces import Box

    env = ogbench.make_env_and_datasets(name, env_only=True)
    try:
        if not all(
            isinstance(space, Box) and len(space.shape) == 1
            for space in (env.action_space, env.observation_space)
        ):
            raise ValueError(
                "Expected vector observations and continuous vector actions"
            )
        if not np.allclose(env.action_space.low, -1) or not np.allclose(
            env.action_space.high, 1
        ):
            raise ValueError("Expected action space bounds [-1,1]")
    except BaseException:
        env.close()
        raise
    return env


def make_env_and_datasets(
    name,
    *,
    chunk_length,
    rng,
    validation_rng,
    discount=0.95,
    dataset_dir="data/ogbench",
    dataset_path=None,
):
    env = make_env(name)
    try:
        train, val = load_ogbench_datasets(
            name,
            directory=dataset_dir,
            path=dataset_path,
            chunk_length=chunk_length,
            discount=discount,
            rng=rng,
            validation_rng=validation_rng,
        )
        for source in (train, val):
            if source is not None and (
                source.data["observations"].shape[1:] != env.observation_space.shape
                or source.data["actions"].shape[1:] != env.action_space.shape
            ):
                raise ValueError("Dataset dimensions disagree with environment spaces")
    except BaseException:
        env.close()
        raise
    return env, train, val
