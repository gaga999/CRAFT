"""Small NNX MLP with explicit dimensions and eager initialization."""

from itertools import pairwise

from flax import nnx


class MLP(nnx.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, *, rngs):
        dims = (input_dim, *hidden_dims, output_dim)
        if any(type(dim) is not int or dim <= 0 for dim in dims):
            raise ValueError("MLP dimensions must be positive integers")
        self.layers = nnx.List([nnx.Linear(a, b, rngs=rngs) for a, b in pairwise(dims)])

    def __call__(self, inputs):
        for layer in self.layers[:-1]:
            inputs = nnx.gelu(layer(inputs))
        return self.layers[-1](inputs)
