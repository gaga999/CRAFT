"""OGBench integration uses its official loader and independent train/val splits."""

from craft.data.offline import OfflineDataset


def make_env_and_datasets(
    name,
    *,
    chunk_length,
    seed,
    discount=0.99,
    dataset_dir="~/.ogbench/data",
    dataset_path=None,
):
    if any(
        mode in name.split("-")
        for mode in ("visual", "powderworld", "oraclerep", "singletask")
    ):
        raise ValueError(
            "Only standard state-based goal-conditioned continuous datasets are supported"
        )
    try:
        import ogbench
    except ImportError as exc:
        raise ImportError("Install OGBench with: uv sync --extra ogbench") from exc
    env, train, val = ogbench.make_env_and_datasets(
        name, dataset_dir=dataset_dir, dataset_path=dataset_path, compact_dataset=False
    )
    try:
        if len(env.action_space.shape) != 1 or len(env.observation_space.shape) != 1:
            raise ValueError(
                "Only state-based continuous-action OGBench datasets are supported"
            )
        train = OfflineDataset(train, chunk_length, seed, discount)
        val = (
            None
            if val is None
            else OfflineDataset(val, chunk_length, seed + 1, discount)
        )
    except Exception:
        env.close()
        raise
    return env, train, val
