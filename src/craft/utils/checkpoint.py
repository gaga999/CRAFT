"""Atomic Flax checkpoint with optimizer, agent RNG, sampler RNG, and run metadata."""

import json
import os
from pathlib import Path

from flax import serialization


def save_checkpoint(path, *, agent, run_state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "agent": serialization.to_state_dict(agent),
        "run_state": json.dumps(run_state),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(serialization.msgpack_serialize(payload))
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def restore_checkpoint(path, *, agent):
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    return serialization.from_state_dict(agent, payload["agent"]), json.loads(
        payload["run_state"]
    )
