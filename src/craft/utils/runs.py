"""Exclusive run/session allocation and immutable experiment provenance."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import jax
from omegaconf import OmegaConf


def write_json(path, value):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".metadata-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance():
    root = Path(__file__).resolve().parents[3]

    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    # Hash implementation and locked dependencies, including untracked source files.
    h = hashlib.sha256()
    for item in sorted(
        [*root.glob("src/**/*.py"), root / "pyproject.toml", root / "uv.lock"]
    ):
        h.update(str(item.relative_to(root)).encode())
        h.update(item.read_bytes())
    status = git("status", "--porcelain")
    return {
        "revision": git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "source_sha256": h.hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "devices": [str(device) for device in jax.devices()],
        "packages": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
    }


def allocate_session(
    config,
    *,
    mode,
    checkpoint=None,
    destination=None,
    allow_code_change=False,
    checkpoint_sha256=None,
):
    root = Path(config["output_dir"]).expanduser().resolve()
    token = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    if mode == "train":
        path = root.with_name(f"{root.name}-{token}")
        config = {**config, "output_dir": str(path)}
    elif mode == "resume":
        path = root / "sessions" / f"resume-{token}"
    elif mode == "evaluate":
        path = (
            Path(destination)
            .expanduser()
            .resolve()
            .with_name(f"{Path(destination).name}-{token}")
            if destination
            else root.parent / "evaluations" / f"{root.name}-{token}"
        )
        source = Path(checkpoint).resolve().parent
        if (
            path == root
            or root in path.parents
            or path == source
            or source in path.parents
        ):
            raise ValueError(
                "Evaluation output must be outside the source training run"
            )
    else:
        raise ValueError(f"Unknown session mode: {mode}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    metadata = {
        "mode": mode,
        "allow_code_change": allow_code_change,
        "started_at": datetime.now(UTC).isoformat(),
        "environment": provenance(),
        "source_checkpoint": None
        if checkpoint is None
        else {
            "path": str(Path(checkpoint).resolve()),
            "sha256": checkpoint_sha256 or file_hash(checkpoint),
        },
    }
    source_root = Path(__file__).resolve().parents[3]
    with zipfile.ZipFile(
        path / "source.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for item in sorted(
            [
                *source_root.glob("src/**/*.py"),
                source_root / "pyproject.toml",
                source_root / "uv.lock",
            ]
        ):
            archive.write(item, item.relative_to(source_root))
    OmegaConf.save(config, path / "config.yaml")
    write_json(path / "metadata.json", metadata)
    write_json(path / "status.json", {"status": "initializing", "step": 0})
    return path
