"""Versioned, atomic NNX checkpoints. Restore validates before mutating a model."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import jax
import numpy as np
from flax import nnx, serialization

FORMAT_VERSION = 6


def _is_key(value):
    return jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)


def _serialize_array(value):
    return jax.random.key_data(value) if _is_key(value) else value


def pure_state(agent):
    return jax.tree.map(_serialize_array, nnx.to_pure_dict(nnx.state(agent)))


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
            "Unsupported checkpoint format; expected v6 latent-actor agent (representation, actor and their optimizers). Start a new run for older checkpoints."
        )
    if not isinstance(payload.get("agent"), dict) or not isinstance(
        payload.get("run_state"), str
    ):
        raise TypeError("Malformed NNX checkpoint payload")
    payload["checkpoint_sha256"] = hashlib.sha256(raw).hexdigest()
    require_finite(payload["agent"])
    return payload


def _matching_arrays(template, values, what):
    expected = jax.tree.flatten_with_path(template)[0]
    got = jax.tree.flatten_with_path(values)[0]
    if len(expected) != len(got) or any(
        p != q
        or np.shape(a) != np.shape(b)
        or np.asarray(a).dtype != np.asarray(b).dtype
        for (p, a), (q, b) in zip(expected, got)
    ):
        raise ValueError(f"Checkpoint {what} paths, shapes, or dtypes do not match")


def _restore_array(expected, value):
    if _is_key(expected):
        return jax.random.wrap_key_data(
            jax.numpy.asarray(value), impl=jax.random.key_impl(expected)
        )
    return jax.numpy.asarray(value)


def restore_checkpoint(path, *, agent, payload=None):
    """Restore the COMPLETE agent state (models, optimizers, RNG, step)."""
    if payload is None:
        payload = read_checkpoint(path)
    metadata = json.loads(payload["run_state"])
    if not isinstance(metadata, dict):
        raise TypeError("Checkpoint run metadata must be a JSON object")
    require_finite(payload["agent"])
    _matching_arrays(pure_state(agent), payload["agent"], "agent state")
    template = nnx.to_pure_dict(nnx.state(agent))
    nnx.update(agent, jax.tree.map(_restore_array, template, payload["agent"]))
    return metadata


def load_modules(payload, *, agent, modules):
    """Copy only the parameters of named agent modules (no optimizer/step/RNG).

    Every module is validated before any is written, so failure leaves the
    agent unchanged.
    """
    source = payload["agent"]
    for name in modules:
        if name not in source:
            raise ValueError(f"Checkpoint has no module {name!r}")
        _matching_arrays(
            pure_state(getattr(agent, name)), source[name], f"module {name!r}"
        )
    for name in modules:
        module = getattr(agent, name)
        template = nnx.to_pure_dict(nnx.state(module))
        nnx.update(module, jax.tree.map(_restore_array, template, source[name]))
