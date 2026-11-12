# Models (`src/craft/models/`)

All modules are Flax NNX and are created eagerly by `make_poc_models`
(`models/factory.py`). Shapes use the notation in [../README.md](../README.md).

## Building blocks

### `MLP` (`mlp.py`)
`Linear → GELU → … → Linear`, no activation on the output. It validates that all
dimensions are positive ints.

### `TokenPointwiseLinear` / `TokenPointwiseMLP` (`tokens.py:11-47`)
These mix information **across tokens** for each width channel independently. A
`[B,T,W]` tensor is treated as NCW (tokens = channels) and convolved with a
kernel-size-1 kernel `(1, T_in, T_out)`. This equals `x^T @ K` on the token axis
without explicit transposes. `tests/test_model_contracts.py` checks it against a
dense reference and asserts that the jaxpr contains no `transpose`. The MLP is
`T → 2T → T` with GELU.

### `MixerBlock` (`tokens.py:50-66`)
```text
x = x + FFN(LayerNorm(x))                 # per-token channel MLP, W → 4W → W
if exchange: x = x + TokenMLP(LayerNorm(x))   # cross-token mixing
```
A block exchanges tokens when `(i+1) % exchange_every == 0` or when it is the last
block. With depth 4 and `exchange_every` 2, blocks 2 and 4 exchange. The same rule
is used in `TokenBackbone` and `TokenMixer`.

### `TokenMixer` (`cvae.py:11-26`)
A stack of `MixerBlock`s followed by a final LayerNorm. Used by the posterior and
the decoder.

### `GaussianHead` (`cvae.py:29-40`)
`Linear(W → 2Z)` (kernel init N(0, 0.01)) splits into `mean` and `raw`, then
```text
log_std = low + (high - low) · sigmoid(raw + log(-low/high))
```
With `raw ≈ 0` at init, `log_std ≈ 0` (σ ≈ 1). It is smoothly bounded to
`[log_std_min, log_std_max]` = `[-4, 1]`.

`sample_gaussian((mean, log_std), key)` is the reparameterized sample.

## Critic: `TokenCritic` (`tokens.py:162-207`)

```text
TokenBackbone(s [B,O], A [B,H,A]):
  tokens = concat(Linear_s(s)[:,None], Linear_a(A))  → [B,H+1,W]
  tokens += position [1,H+1,W]
  depth × MixerBlock → LayerNorm                       → features [B,H+1,W]

phi = unit(phi_head(features.mean(axis=1)))           [B,D]  (phi_head: MLP W→W→D)

GoalEncoder(x [B,O]):
  encode(x) = LayerNorm(MLP(x; O→W→W))                [B,W]   ← used by CVAE as "psi feature"
  psi       = unit(MLP(encode(x); W→W→D))             [B,D]

score(s,A,g) = exp(log_scale)·<phi,psi> + bias        [B]
```

- `log_scale` starts at `log 10`; `bias` starts at `-log(B-1)`, so it depends on
  `data.batch_size`.
- `unit(x) = x / sqrt(|x|² + 1e-8)`.
- `encode` is applied to both goals and **states**, which requires goal space =
  observation space. The OGBench adapter and the evaluator both check this.
- `TokenBackbone` has an optional `condition_dim` path that the critic never
  uses (see code-review M2).

## Posterior: `q(Z | s, A, g)` (`cvae.py:48-69`)

```text
x = concat(Linear(features) [B,H+1,W], Linear(encode(g))[:,None] [B,1,W])  → [B,H+2,W]
    token order: [state, a_1 … a_H, goal]
TokenMixer(fusion_depth=2) → take positions 1..H (the action tokens) → GaussianHead
→ (mean, log_std) each [B,H,Z]
```

`features` and `encode(g)` come from the critic. They are detached unless
`agent.cvae_updates_crl=true`.

## Prior: `p(Z | s, g)` (`cvae.py:72-89`)

```text
condition = concat(s, g, sg(encode(s)), sg(encode(g)))   [B, 2O+2W]
MLP(2O+2W → W → W → H·W) → reshape [B,H,W] → GaussianHead → [B,H,Z] each
```

The prior always applies stop-gradient to the critic features, inside the module
itself, so gradients from the prior never reach the critic.

## Decoder: `D(Z, encode(s), encode(g), mask)` (`cvae.py:92-119`)

```text
c        = Linear_s(encode(s)) + Linear_g(encode(g))         [B,W]
injected = mask · exp(log_gate) · Linear_c(c)[:,None]        [B,1,W]   (mask ∈ {0,1}, [B,1,1])
x        = Linear_z(Z) + position [1,H,W] + injected          [B,H,W]
Â        = tanh(Linear(TokenMixer(depth)(x)))                 [B,H,A]  (output kernel init N(0,0.01))
```

- `log_gate` starts at `log 0.1` and is learned (the gate is always positive).
- The mask is applied after **all** condition projections and biases, so a dropped
  sample sees no condition information at all. Training drops each sample's
  condition with probability `condition_drop_prob` (0.2). Validation, guidance and
  inference use mask = 1.
- `tanh` bounds the actions; no hard clip is applied anywhere.

## Inference policy (`policies/chunk.py`)

```text
hs, hg = critic.encode(s), critic.encode(g)
Z ~ prior(s, g, hs, hg)
Â = decoder(Z, hs, hg, mask=1)      # [B,H,A], or [H,A] for a single unbatched observation
```

The agent wraps this with `nnx.jit` as `CRAFTAgent.sample_action_chunk`.

## Parameter groups

`CRAFTAgent.optimizers[i]` holds one `nnx.Optimizer(wrt=nnx.Param)` per module in
the order `(critic, posterior, decoder, prior)`. Scalars such as
`critic.log_scale`, `critic.bias` and `decoder.log_gate` are `nnx.Param`, so their
module's optimizer trains them.
