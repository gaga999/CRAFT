"""Load standard OGBench NPZ splits without materializing next observations."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from craft.data.offline import OfflineDataset


def load_split(path, chunk_length, discount, rng):
    # NPZ handles close immediately; only the three needed arrays are decompressed.
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("observations", "actions", "terminals")}
    return OfflineDataset(data, chunk_length, discount, rng=rng, compact=True)


def load_ogbench_datasets(
    name, *, directory, path, chunk_length, discount, rng, validation_rng
):
    if path is None:
        from ogbench.utils import download_datasets

        download_datasets([name], str(directory))
        train_path = Path(directory) / f"{name}.npz"
    else:
        train_path = Path(path)
    validation_path = train_path.with_name(train_path.stem + "-val.npz")
    # Independent splits can decompress and validate concurrently, sharing no RNG.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="craft-load") as pool:
        train = pool.submit(load_split, train_path, chunk_length, discount, rng)
        validation = pool.submit(
            load_split, validation_path, chunk_length, discount, validation_rng
        )
        return train.result(), validation.result()
