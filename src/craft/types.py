"""Shared vocabulary; semantic shapes are documented in docs/interfaces.md."""

from typing import Any

# Array leaves may be NumPy arrays on the host or JAX arrays on device.
Batch = dict[str, Any]
Metrics = dict[str, Any]
