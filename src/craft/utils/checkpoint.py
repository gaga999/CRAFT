"""Versioned, atomic NNX checkpoints. Restore validates before mutating a model."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import jax
import numpy as np
from flax import nnx, serialization

FORMAT_VERSION = 2


def pure_state(agent):
    return nnx.to_pure_dict(nnx.state(agent))


def require_finite(tree):
    if not all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree.leaves(tree)):
        raise FloatingPointError("Nonfinite model/optimizer/checkpoint state")


def save_checkpoint(path, *, agent, run_state):
    state = pure_state(agent)
    require_finite(state)
    payload = {
        "format_version": FORMAT_VERSION,
        "agent": jax.device_get(state),
        "run_state": json.dumps(run_state, allow_nan=False),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".checkpoint-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(serialization.msgpack_serialize(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_checkpoint(path):
    raw = Path(path).read_bytes()
    payload = serialization.msgpack_restore(raw)
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            "Unsupported checkpoint format; Linen checkpoints cannot resume NNX training"
        )
    if not isinstance(payload.get("agent"), dict) or not isinstance(
        payload.get("run_state"), str
    ):
        raise TypeError("Malformed NNX checkpoint payload")
    payload["checkpoint_sha256"] = hashlib.sha256(raw).hexdigest()
    require_finite(payload["agent"])
    return payload


def restore_checkpoint(path, *, agent, payload=None):
    if payload is None:
        payload = read_checkpoint(path)
    metadata = json.loads(payload["run_state"])
    if not isinstance(metadata, dict):
        raise TypeError("Checkpoint run metadata must be a JSON object")
    require_finite(payload["agent"])
    expected, got = (
        jax.tree.flatten_with_path(pure_state(agent))[0],
        jax.tree.flatten_with_path(payload["agent"])[0],
    )
    if len(expected) != len(got) or any(
        p != q
        or np.shape(a) != np.shape(b)
        or np.asarray(a).dtype != np.asarray(b).dtype
        for (p, a), (q, b) in zip(expected, got)
    ):
        raise ValueError(
            "Checkpoint state paths, shapes, or dtypes do not match this NNX agent"
        )
    nnx.update(agent, jax.tree.map(jax.numpy.asarray, payload["agent"]))
    return metadata
