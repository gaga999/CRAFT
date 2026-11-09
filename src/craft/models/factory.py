"""Explicit model construction; no automatic imports or base class required."""

from collections.abc import Mapping
from typing import Any

from craft.models.mlp import MLP


def make_encoder(config: Mapping[str, Any]) -> MLP:
    """Construct an encoder module; initialize its variables in the agent."""
    if config["name"] != "mlp":
        raise ValueError(f"Unknown encoder: {config['name']!r}")
    hidden_dims = tuple(config["hidden_dims"])
    output_dim = config["output_dim"]
    if any(width <= 0 for width in (*hidden_dims, output_dim)):
        raise ValueError("Model dimensions must be positive")
    return MLP(hidden_dims=hidden_dims, output_dim=output_dim)
