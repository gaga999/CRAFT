"""Episode-safe sampling from resident regular or compact OGBench arrays."""

import hashlib

import numpy as np


class OfflineDataset:
    def __init__(self, data, chunk_length=1, discount=0.95, *, rng, compact=False):
        if type(chunk_length) is not int or chunk_length < 1 or not 0 < discount < 1:
            raise ValueError("Require positive chunk length and discount in (0, 1)")
        self.data = dict(data)
        self.compact = compact
        self.chunk_length = chunk_length
        self.discount = discount
        self.rng = rng
        required = {"observations", "actions", "terminals"}
        if not compact:
            required.add("next_observations")
        if not required <= set(data):
            raise ValueError("Incomplete transition dataset")
        s, a = data["observations"], data["actions"]
        n = len(s)
        if s.ndim != 2 or a.ndim != 2 or not np.issubdtype(a.dtype, np.floating):
            raise ValueError(
                "Only vector observations and continuous vector actions are supported"
            )
        if any(len(data[k]) != n for k in required):
            raise ValueError("Dataset arrays must have matching lengths")
        if data["terminals"].shape != (n,) or (
            not compact and data["next_observations"].shape != s.shape
        ):
            raise ValueError("Expected next_observations [N,O] and terminals [N]")
        # Cast once, not after every advanced-index gather. Float32 inputs alias.
        for key in required:
            dtype = None if key == "terminals" and compact else np.float32
            value = self.data[key] = np.asarray(data[key], dtype=dtype, order="C")
            for start in range(0, n, 65536):
                block = value[start : start + 65536]
                if key == "actions" and compact:
                    block = block[data["terminals"][start : start + 65536] == 0]
                if not np.isfinite(block).all():
                    raise ValueError(f"Nonfinite dataset values: {key}")
                if key == "actions" and np.any(np.abs(block) > 1.0001):
                    raise ValueError("Expected OGBench actions normalized to [-1, 1]")
        if not np.isin(data["terminals"], [0, 1]).all():
            raise ValueError("terminals must be binary episode boundaries")
        ends = np.flatnonzero(data["terminals"] > 0)
        if n == 0 or len(ends) == 0 or ends[-1] != n - 1:
            raise ValueError("Dataset must end at an episode boundary")
        starts = self._episode_starts = np.concatenate(([0], ends[:-1] + 1))
        self.episode_ends = ends - int(compact)
        if np.any(self.episode_ends < starts):
            raise ValueError("Episodes must contain at least one transition")
        # Avoid a dense N-element episode-end table and N searchsorted calls.
        self.valid = (
            np.concatenate(
                [
                    np.arange(begin, end - chunk_length + 2, dtype=np.int64)
                    for begin, end in zip(starts, self.episode_ends)
                    if end - begin + 1 >= chunk_length
                ]
            )
            if np.any(self.episode_ends - starts + 1 >= chunk_length)
            else np.empty(0, np.int64)
        )
        if not len(self.valid):
            raise ValueError("No episode is long enough for the requested chunk")
        self._chunk_offsets = np.arange(chunk_length)
        self._transitions = n - len(ends) if compact else n

    def fingerprint(self):
        h = hashlib.sha256()
        for key in ("observations", "actions", "next_observations", "terminals"):
            # Hash the same logical transitions in either storage layout.
            value = self.data[
                "observations" if key == "next_observations" and self.compact else key
            ]
            shape = (self._transitions, *value.shape[1:])
            h.update(f"{key}:{shape}:float32".encode())
            ranges = (
                zip(self._episode_starts, self.episode_ends + 1)
                if self.compact
                else ((0, len(value)),)
            )
            shift = int(self.compact and key in ("next_observations", "terminals"))
            for begin, end in ranges:
                block = value[begin + shift : end + shift]
                # memoryview avoids another complete chunk-sized bytes copy.
                h.update(
                    memoryview(np.ascontiguousarray(block, dtype=np.float32)).cast("B")
                )
        return h.hexdigest()

    def sample_indices(self, batch_size, *, rng=None):
        """Plan RNG draws serially; workers only gather immutable index plans."""
        if type(batch_size) is not int or batch_size < 2:
            raise ValueError("Contrastive learning needs batch_size >= 2")
        rng = self.rng if rng is None else rng
        i = rng.choice(self.valid, batch_size)
        # Future offsets are at least one. The last future state comes from next_observations.
        remaining = self.episode_ends[np.searchsorted(self.episode_ends, i)] - i + 1
        value_offset = np.minimum(
            rng.geometric(1 - self.discount, batch_size), remaining
        )
        # Retain this draw even when actor goals are omitted from model input.
        actor_offset = rng.integers(1, remaining + 1)
        return i, i + value_offset - 1, i + actor_offset - 1

    def gather(self, indices, *, include_actor_goals=True):
        i, value_indices, actor_indices = indices
        future = (
            self.data["observations"]
            if self.compact
            else self.data["next_observations"]
        )
        shift = int(self.compact)

        batch = {
            "observations": self.data["observations"][i],
            "action_chunks": self.data["actions"][i[:, None] + self._chunk_offsets],
            "value_goals": future[value_indices + shift],
        }
        if include_actor_goals:
            batch["actor_goals"] = future[actor_indices + shift]
        return batch

    def sample(self, batch_size, *, include_actor_goals=True):
        return self.gather(
            self.sample_indices(batch_size), include_actor_goals=include_actor_goals
        )


def synthetic_dataset(*, rng, episodes=8, length=32, observation_dim=6, action_dim=2):
    """Small continuous trajectories for plumbing checks, not performance claims."""
    s = rng.normal(size=(episodes, length + 1, observation_dim)).astype(np.float32)
    a = rng.uniform(-1, 1, (episodes * length, action_dim)).astype(np.float32)
    terminals = np.zeros((episodes, length), np.float32)
    terminals[:, -1] = 1
    return {
        "observations": s[:, :-1].reshape(-1, observation_dim),
        "next_observations": s[:, 1:].reshape(-1, observation_dim),
        "actions": a,
        "terminals": terminals.ravel(),
    }
