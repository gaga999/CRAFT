"""Minimal Linen example; the project-wide Linen/NNX choice remains open."""

from flax import linen as nn
from jax import Array


class MLP(nn.Module):
    """Map inputs [..., input_dim] to features [..., output_dim]."""

    hidden_dims: tuple[int, ...]
    output_dim: int

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        for width in self.hidden_dims:
            inputs = nn.relu(nn.Dense(width)(inputs))
        return nn.Dense(self.output_dim)(inputs)
