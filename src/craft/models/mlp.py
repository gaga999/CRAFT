"""Small NNX MLP with explicit dimensions and eager initialization."""

from itertools import pairwise

from flax import nnx


class MLP(nnx.Module):
    """Linear layers with GELU; optional LayerNorm after each hidden activation."""

    def __init__(self, input_dim, hidden_dims, output_dim, *, layer_norm=False, rngs):
        dims = (input_dim, *hidden_dims, output_dim)
        if any(type(dim) is not int or dim <= 0 for dim in dims):
            raise ValueError("MLP dimensions must be positive integers")
        self.layers = nnx.List([nnx.Linear(a, b, rngs=rngs) for a, b in pairwise(dims)])
        self.norms = nnx.List(
            [nnx.LayerNorm(d, rngs=rngs) for d in hidden_dims] if layer_norm else []
        )

    def __call__(self, inputs):
        for i, layer in enumerate(self.layers[:-1]):
            inputs = nnx.gelu(layer(inputs))
            if len(self.norms):
                inputs = self.norms[i](inputs)
        return self.layers[-1](inputs)
