"""Episode-safe chunks and future goals from regular (noncompact) OGBench data."""

import hashlib

import numpy as np


class OfflineDataset:
    def __init__(self, data, chunk_length=1, discount=0.95, *, rng):
        if chunk_length < 1 or not 0 < discount < 1:
            raise ValueError("Require positive chunk length and discount in (0, 1)")
        self.data = data
        self.chunk_length = chunk_length
        self.discount = discount
        self.rng = rng
        if not {"observations", "actions", "next_observations", "terminals"} <= set(
            data
        ):
            raise ValueError("Incomplete transition dataset")
        s, a = data["observations"], data["actions"]
        n = len(s)
        if s.ndim != 2 or a.ndim != 2 or not np.issubdtype(a.dtype, np.floating):
            raise ValueError(
                "Only vector observations and continuous vector actions are supported"
            )
        if any(
            len(data[k]) != n for k in ("actions", "terminals", "next_observations")
        ):
            raise ValueError("Dataset arrays must have matching lengths")
        if data["next_observations"].shape != s.shape or data["terminals"].shape != (
            n,
        ):
            raise ValueError("Expected next_observations [N,O] and terminals [N]")
        for key in ("observations", "actions", "next_observations", "terminals"):
            value = data[key]
            for start in range(0, n, 65536):
                if not np.isfinite(value[start : start + 65536]).all():
                    raise ValueError(f"Nonfinite dataset values: {key}")
        if not np.isin(data["terminals"], [0, 1]).all():
            raise ValueError("terminals must be binary episode boundaries")
        ends = np.flatnonzero(data["terminals"] > 0)
        if n == 0 or len(ends) == 0 or ends[-1] != n - 1:
            raise ValueError("Dataset must end at an episode boundary")
        self.ends = ends[np.searchsorted(ends, np.arange(n))]
        self.valid = np.flatnonzero(self.ends - np.arange(n) + 1 >= chunk_length)
        if not len(self.valid):
            raise ValueError("No episode is long enough for the requested chunk")
        if np.any(np.abs(a) > 1.0001):
            raise ValueError("Expected OGBench actions normalized to [-1, 1]")

    def fingerprint(self):
        h = hashlib.sha256()
        for key in ("observations", "actions", "next_observations", "terminals"):
            value = self.data[key]
            h.update(f"{key}:{value.shape}:{value.dtype}".encode())
            for start in range(0, len(value), 65536):
                h.update(np.ascontiguousarray(value[start : start + 65536]).tobytes())
        return h.hexdigest()

    def sample(self, batch_size):
        if batch_size < 2:
            raise ValueError("Contrastive learning needs batch_size >= 2")
        i = self.rng.choice(self.valid, batch_size)
        # Future offsets are at least one. The last future state comes from next_observations.
        remaining = self.ends[i] - i + 1
        value_offset = np.minimum(
            self.rng.geometric(1 - self.discount, batch_size), remaining
        )
        actor_offset = self.rng.integers(1, remaining + 1)

        def goals(offset):
            return self.data["next_observations"][i + offset - 1].astype(np.float32)

        return {
            "observations": self.data["observations"][i].astype(np.float32),
            "action_chunks": self.data["actions"][
                i[:, None] + np.arange(self.chunk_length)
            ].astype(np.float32),
            "value_goals": goals(value_offset),
            "actor_goals": goals(actor_offset),
        }


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
