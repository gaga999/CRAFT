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
The W&B display name defaults to the session directory name; set it with
`--name` or `logging.name` to find ablations by name:

```bash
uv run craft --config configs/experiment/cube_double_play.yaml --name awbc-actor agent.bc_weighting=advantage
```

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
s -> state encoder -> e_s;   phi = head([flatten(z); e_s]);   psi = goal_encoder(g)
z~ = LN(z + sigma*eps)   (one sample, read by BOTH the head and the decoder)
critic logit: cosine  t * unit(phi).unit(psi) + b, t <= max_scale
              dot     phi.psi / sqrt(D) + b      (no normalization, no temperature)
reconstruction: decoder(s, z~) -> A
actor: (s, g[, sg psi(g), sg e_s(s)]) -> MLP -> LayerNorm -> z_hat -> decoder -> A_hat
actor loss: -score(s, E(s, A_hat), g) / sg(mean|score|) + bc_weight * mean_i w_i |z_hat_i - sg(z_i)|^2
```

`score` is the logit without `b`, computed with the true state. With the default
`guidance: reencoded` the actor is scored on the latent of the chunk it would
execute; `guidance: latent` scores `z_hat` directly (exploitable, kept as an
ablation). Representation weights (CRL + `reconstruction_weight` x
reconstruction, AdamW with `weight_decay` on weight matrices) and actor weights
(`actor_weight_decay`) have separate optimizers; the actor differentiates
through fixed representation weights and reads detached representation
features. See [contracts](docs/interfaces.md) and [method decisions](docs/method.md).

### Options

| Key | Values | Effect |
| --- | --- | --- |
| `<component>_width`, `<component>_depth` | component: `action_encoder`, `state_encoder`, `goal_encoder`, `head`, `decoder`, `actor` | Capacity per component (`decoder_width`; `action_encoder_depth` is Mixer blocks; others are MLP hidden layers). Defaults reproduce the earlier shared-width model |
| `similarity` | `cosine` (default), `dot` | Critic logit form; `max_scale` applies to `cosine` only |
| `actor_inputs` | subset of `goal_embedding`, `state_embedding` | Extra detached actor inputs `psi(g)` and `e_s(s)` |
| `bc_weighting` | `none` (default), `advantage` | Per-sample BC weight `w = min(exp(advantage_beta * A), advantage_max_weight)`, `A = (score(s, z_data, g) - baseline) / sg(mean abs score)` |
| `advantage_baseline` | `actor` (default), `batch` | `actor`: the actor's own guided score per sample. `batch`: the batch-mean dataset score, which mixes states and goals and so also favours easy goals |

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

Names are `<component>/<name>_<split>`: each W&B section is one component, and
`_train` (training batch, noisy latents) sits next to `_val` (official
validation split, clean latents, computed at each log step).

| Component | Metric | Diagnoses | Problem when |
| --- | --- | --- | --- |
| critic | `loss_train` / `loss_val` | Generalization | gap widens: overfitting |
| | `scale_train` (cosine) | Temperature | pinned at `max_scale` |
| | `top1_val`, `logit_positive_val`, `logit_negative_val` | Ranking and calibration on the training goal distribution | positives drift down while train improves |
| | `auc_<k>_val` (k = 5, 20, 50, 100, 200) | Effective horizon: true state k steps ahead vs other samples' goals | near 0.5 at the distances tasks need |
| | `action_effect_val`, `state_effect_val`, `action_to_state_effect_val` | Change of `phi` when actions or states are swapped | action effect near 0: critic ignores actions |
| | `phi_norm_val`, `psi_norm_val` | Unnormalized embedding norms (confidence under `dot`) | growing without bound |
| decoder | `rec_train` / `rec_val`, `rec_first_val` | Reconstruction (whole chunk, executed first step) | high |
| | `rec_noisy_val` | Robustness at the training noise level | far above `rec_val` |
| | `rec_shuffled_val` | Decoder uses `z` | close to `rec_val` |
| latent | `batch_std_val` | Spread of `z` across samples | near 0: collapse |
| actor | `q_loss_train`, `bc_train` | Loss terms (BC after weighting) | |
| | `score_gain_train`, `score_gain_val` | Executed chunk vs dataset chunk under the critic | <= 0: guidance gives nothing |
| | `score_data_val` | Level of dataset scores | reference for the two above |
| | `exploit_gap_val` | Latent scored above its executed chunk | large: latent exploitation |
| | `action_mse_rel_val` | Actor-vs-data action MSE / data action variance | > 1: out of distribution |
| | `latent_mse_val`, `cycle_mse_val` | Distance from data latents; off the encoder's image | cycle large: decoder input unlike training |
| | `goal_sensitivity_val` | Action change when goals are swapped / data variance | near 0: actor ignores the goal |
| | `advantage_train`, `weight_mean_*`, `weight_clipped_*` (advantage BC) | Weight distribution | most weights clipped or all near 1 |
| balance | `grad_norm_representation_train`, `grad_norm_actor_train` | Pre-clip gradient norms | spikes or clipping |
| | `crl_to_rec_val`, `q_to_bc_val` | Gradient ratio of each optimizer's two terms | one term dominates |
| evaluation | `<task>/success`, `overall_success` | Task success | |

`time/` holds per-step data wait and update time.

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
