# CRAFT latent-actor POC

**Contrastive Representation for Action Flow and Temporal Abstraction**

A Flax NNX proof of concept that generates action chunks in a CRL-shaped
latent space. The CRL state-action encoder produces a per-action latent that
both feeds the contrastive head and decodes back to the chunk; an actor
generates that latent and is guided by the critic score. Standard vector-state,
continuous-action OGBench is supported. This is an experimental method, not a
validated learning result.

## Run

```bash
uv sync --locked
# Each invocation appends a UTC timestamp and UUID to output_dir automatically.
CUDA_VISIBLE_DEVICES=0 uv run craft --config configs/experiment/puzzle_3x3_noisy.yaml
uv run craft --config configs/experiment/smoke.yaml   # plumbing check
uv run python -m unittest discover -s tests -v
uv run ruff check .
```

`--steps N` limits additional updates without changing the saved schedule.
`--dry-run` validates/prints settings. OGBench downloads data on first use.
Shipped real-task configs log to W&B (`WANDB_MODE=offline` for local runs,
`logging.backend=console` to disable).

### GPU (NVIDIA Linux)

```bash
uv sync --locked --extra cuda
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
  uv run --extra cuda craft --config configs/experiment/cube_single_noisy.yaml
```

`CUDA_VISIBLE_DEVICES` selects the GPU (one run uses one device). JAX
preallocates 75% of GPU memory at startup; to share a GPU between runs, prefix
one of these [memory options](https://docs.jax.dev/en/latest/gpu_memory_allocation.html):

| Variable | Effect |
| --- | --- |
| `XLA_PYTHON_CLIENT_MEM_FRACTION=.25` | preallocate this fraction instead of `.75` |
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` | no preallocation; grow on demand (may fragment) |
| `XLA_PYTHON_CLIENT_ALLOCATOR=platform` | allocate/free exactly as needed; slow, for debugging |

```bash
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=.25 \
  uv run --extra cuda craft --config configs/experiment/puzzle_3x3_noisy.yaml
```

| Config | Dataset | Role | Steps |
| --- | --- | --- | --- |
| `smoke.yaml` | synthetic | plumbing only | 3 |
| `pointmaze_medium.yaml` | `pointmaze-medium-navigate-v0` | easy sanity task | 300k |
| `cube_single_noisy.yaml` | `cube-single-noisy-v0` | main task | 1M |
| `puzzle_3x3_noisy.yaml` | `puzzle-3x3-noisy-v0` | puzzle task | 1M |
| `cube_double_play.yaml` | `cube-double-play-v0` | hard task | 1M |

Settings and their evidence are in [method details](docs/method.md).

## Architecture

```text
(s, a_1..a_H) -> token Mixer -> action tokens -> Linear -> LayerNorm -> z [H, Z]
s -> state encoder -> e_s;   phi = unit(MLP([flatten(z); e_s]));   psi = unit(MLP(g))
z~ = LN(z + sigma*eps)   (one sample, read by BOTH the head and the decoder)
CRL: SigLIP  t * phi(z~).psi + b,  t <= max_scale     reconstruction: decoder(s, z~) -> A
actor: (s, g) -> MLP -> LayerNorm -> z_hat -> decoder(s, z_hat) -> executed chunk A_hat
actor loss: -score(s, E(s, A_hat), g) / sg(mean|score|) + bc_weight * |z_hat - sg(z)|^2
```

`score` is `t * phi(z, e_s(s)) . psi(g)` with the true state. With the default
`guidance: reencoded` the actor is scored on the latent of the chunk it would
execute, `E(s, decoder(s, z_hat))`; `guidance: latent` scores `z_hat` directly
(exploitable, kept as an ablation). Representation weights (CRL +
`reconstruction_weight` x reconstruction, AdamW with `weight_decay` on weight
matrices) and actor weights (actor loss, `actor_weight_decay`) have separate
optimizers; the actor differentiates through fixed representation weights. See
[contracts](docs/interfaces.md) and [method decisions](docs/method.md).

## Stages and initialization

`train.stages` is a list of `representation` (encoder, heads, decoder),
`actor` (actor only, representation fixed) and `joint` (both, each with its own
objective from pre-update parameters). Task configs use one `joint` stage;
`smoke.yaml` runs all three in order as an example.

`init.checkpoint` starts a NEW run from another run's parameters:

```bash
uv run craft --config configs/experiment/cube_single_noisy.yaml \
  init.checkpoint=runs/<source-run>/checkpoint.msgpack "init.modules=[representation]" \
  "train.stages=[{name: actor, steps: 300000}]"
```

Only the listed modules' parameters are copied; optimizers, step and the data
sampler start fresh. The source must come from the same dataset and dimensions,
and its path, SHA-256, modules and step are recorded in `metadata.json`.

## Metrics

| Prefix | Meaning |
| --- | --- |
| `crl/` | loss, effective (capped) scale `t`, bias `b`, top-1 and binary accuracy, mean positive/negative logits; training reads noisy latents, validation clean ones |
| `rec/` | chunk and first-action MSE (training: noisy decoder input); validation: clean, noisy and batch-shuffled latents |
| `latent/` | per-dimension batch std; validation: `phi` change when actions vs states are swapped, and their ratio |
| `actor/` | q/BC terms, guided score vs dataset score; validation: `score_direct`, `score_reencoded`, `score_data`, `score_gain` (re-encoded minus data: real improvement), `score_exploit_gap` (direct minus re-encoded), latent/action/cycle MSE |
| `grad/` | pre-clip norms; validation: CRL vs reconstruction on shared encoder, q vs BC on actor |
| `evaluation/` | per-task and overall success |

`validation/*` is computed on the official validation split at each log step.

## Throughput

Each NNX transform call costs milliseconds of host-side graph traversal, which
can exceed this model's device time. Training therefore scans up to
gcd(log, eval, save intervals) ≤ 100 updates per dispatch as a pure function
of the split state; a nonfinite step rejects the whole chunk on the host, so
no per-step state copy is kept. Evaluation snapshots the policy once into a
`jax.jit` of explicit state. `time/data_wait_seconds` and
`time/update_seconds` are per-step averages over the chunk.

Datasets stay in host RAM in OGBench's compact layout; `data.workers` threads
prepare up to `data.prefetch_batches` CPU batches and `data.device_prefetch`
device batches. Set `data.workers=0` for synchronous preparation.

## Outputs and resume

`output_dir` is a **prefix**, e.g. `runs/cube_single_noisy-<UTC>-<UUID>`. The
resolved path is printed and saved in config/checkpoint. Use the printed
directory to resume or evaluate:

```bash
RUN_DIR="runs/your-printed-run-directory"
uv run craft --config "$RUN_DIR/config.yaml" --restore "$RUN_DIR/checkpoint.msgpack"
uv run python scripts/evaluate.py --config "$RUN_DIR/config.yaml" \
  --restore "$RUN_DIR/checkpoint.msgpack" evaluation.episodes=20
```

Resume creates a new session under the original run; evaluation writes outside
it. Original metadata/metrics remain immutable. Checkpoint **v6** stores both
models, both optimizers, RNG, sampler and schedule state; older checkpoints are
rejected. Code changes on resume require `--allow-code-change`. Data prefetch
settings may change on resume. Ctrl-C saves the latest accepted update.

| Directory/file (under `src/craft/`) | Responsibility |
| --- | --- |
| `models/tokens.py`, `models/representation.py` | Token Mixer, representation, decoder, actor |
| `agents/craft.py`, `losses/` | Gradient ownership, stages, optimizers, diagnostics |
| `policies/`, `evaluators/` | Inference composition and task rollouts |
| `config.py`, `experiments.py`, `cli.py` | Strict settings, assembly, initialization, entry points |
| `data/`, `envs/`, `trainers/`, `utils/` | Sampling, OGBench, scheduling, RNG/session/checkpoint services |
