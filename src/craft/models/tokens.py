"""NNX token-wise shared FFNs with periodic residual token-MLP exchange."""

import jax
import jax.numpy as jnp
from flax import nnx

from craft.models.mlp import MLP


class TokenPointwiseLinear(nnx.Module):
    """Mix token channels in [batch, tokens, width] without swapping axes."""

    def __init__(self, in_tokens, out_tokens, *, rngs):
        self.kernel = nnx.Param(
            nnx.initializers.lecun_normal()(
                rngs.params(), (1, in_tokens, out_tokens), jnp.float32
            )
        )
        self.bias = nnx.Param(
            nnx.initializers.zeros(rngs.params(), (out_tokens,), jnp.float32)
        )

    def __call__(self, x):
        return (
            jax.lax.conv_general_dilated(
                x,
                self.kernel[...],
                window_strides=(1,),
                padding="VALID",
                dimension_numbers=("NCW", "WIO", "NCW"),
            )
            + self.bias[...][None, :, None]
        )


class TokenPointwiseMLP(nnx.Module):
    def __init__(self, tokens, *, rngs):
        self.layers = nnx.List(
            [
                TokenPointwiseLinear(tokens, 2 * tokens, rngs=rngs),
                TokenPointwiseLinear(2 * tokens, tokens, rngs=rngs),
            ]
        )

    def __call__(self, x):
        return self.layers[1](nnx.gelu(self.layers[0](x)))


class MixerBlock(nnx.Module):
    def __init__(self, width, tokens, exchange, *, rngs):
        self.ffn_norm = nnx.LayerNorm(width, rngs=rngs)
        self.ffn = MLP(width, (4 * width,), width, rngs=rngs)
        self.exchange = exchange
        if exchange:
            self.token_norm = nnx.LayerNorm(width, rngs=rngs)
            self.token_mlp = TokenPointwiseMLP(tokens, rngs=rngs)

    def local(self, x):
        return x + self.ffn(self.ffn_norm(x))

    def __call__(self, x):
        x = self.local(x)
        if self.exchange:
            x = x + self.token_mlp(self.token_norm(x))
        return x


class TokenMixer(nnx.Module):
    """Mixer blocks over a fixed token count, followed by LayerNorm.

    Token exchange runs every `exchange_every` blocks and always in the last one.
    """

    def __init__(self, tokens, width, depth, exchange_every, *, rngs):
        if any(
            type(v) is not int or v < 1 for v in (tokens, width, depth, exchange_every)
        ):
            raise ValueError("Mixer sizes must be positive integers")
        self.blocks = nnx.List(
            [
                MixerBlock(
                    width,
                    tokens,
                    (i + 1) % exchange_every == 0 or i == depth - 1,
                    rngs=rngs,
                )
                for i in range(depth)
            ]
        )
        self.norm = nnx.LayerNorm(width, rngs=rngs)

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class TokenBackbone(nnx.Module):
    """[state, a_1..a_H] tokens with learned positions -> mixed tokens [B,H+1,W]."""

    def __init__(
        self,
        observation_dim,
        action_dim,
        chunk_length,
        *,
        width,
        depth,
        exchange_every,
        rngs,
    ):
        self.chunk_length = chunk_length
        self.state_projection = nnx.Linear(observation_dim, width, rngs=rngs)
        self.action_projection = nnx.Linear(action_dim, width, rngs=rngs)
        self.position = nnx.Param(
            nnx.initializers.normal(0.02)(rngs.params(), (1, chunk_length + 1, width))
        )
        self.mixer = TokenMixer(
            chunk_length + 1, width, depth, exchange_every, rngs=rngs
        )

    def __call__(self, states, actions):
        if (
            states.ndim != 2
            or actions.ndim != 3
            or states.shape[0] != actions.shape[0]
            or actions.shape[1] != self.chunk_length
        ):
            raise ValueError(
                "Expected states [B,O] and configured action chunks [B,H,A]"
            )
        x = jnp.concatenate(
            [self.state_projection(states)[:, None], self.action_projection(actions)], 1
        )
        return self.mixer(x + self.position[...])
