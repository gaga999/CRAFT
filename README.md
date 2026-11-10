# CRAFT

**Contrastive Representation for Action Flow and Temporal Abstraction**

Offline goal-conditioned CRL with token-based action chunks and a representation-
conditioned actor. This is an initial research baseline, not a reproduction claim.
Online collection, image encoders, and discrete actions are not implemented.

## Quickstart

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are required. From this directory:

```bash
uv sync --locked
uv run python scripts/train.py --config configs/experiment/smoke.yaml
uv run python -m unittest discover -s tests -v
uv run ruff check .

# OGBench: downloads train/validation data on first use.
uv sync --locked --extra ogbench
uv run craft --config configs/experiment/offline.yaml
# Override dataset, seed, or other settings with OmegaConf dotlist syntax:
uv run craft --config configs/experiment/offline.yaml \
  data.name=cube-single-play-v0 seed=1 output_dir=runs/cube-seed1
```

Use `--dry-run` to print settings without initializing components. The smoke config
uses synthetic trajectories for three updates. It does not test control performance.
Dataset/model dimensions are inferred from data; chunk length and network capacity
are configurable. Supported inputs are OGBench's standard **vector-state,
continuous-action, goal-conditioned** datasets (navigation and manipulation),
with actions in `[-1, 1]`. Visual, discrete, oracle-representation, and single-task
variants are outside this baseline's evaluation contract.

```bash
# Resume using the exact saved configuration and optimizer/sampler states:
uv run craft --config runs/crl_tokens/config.yaml \
  --restore runs/crl_tokens/checkpoint.msgpack
uv run python scripts/evaluate.py --config runs/crl_tokens/config.yaml \
  --restore runs/crl_tokens/checkpoint.msgpack
# Optional W&B logging (set WANDB_MODE=offline for local-only use):
uv run craft --config configs/experiment/offline.yaml logging.backend=wandb
```

Checkpoints contain agent/optimizer/RNG states and sampler states; resolved config,
revision/dirty status, package versions, and JSONL metrics are saved in `output_dir`.
Use separate output directories per experiment. Resume currently requires an
identical resolved config, including the original stage schedule. CUDA-enabled
Linux installations can add `--extra cuda` to `uv sync`.

## Packages

| Package | Role |
| --- | --- |
| JAX / Flax Linen | Autodiff, JIT, PRNG, models, and functional training state |
| Optax | Adam, gradient clipping, and sigmoid contrastive loss |
| NumPy | Episode-safe sampling and host-side dataset processing |
| OmegaConf | YAML configuration and command-line overrides |
| W&B | Optional tracking; console and JSONL work independently |
| OGBench (extra) | Dataset loading, simulation, and goal-conditioned tasks |
| uv / Ruff / unittest | Dependencies, linting, and numerical/contract checks |

## Architecture

```text
scripts + config -> trainer -> dataset.sample -> agent.update
                                                 | critic + teacher + student
raw state + action tokens -> MLP mixer -> phi --+--> contrastive score
future goal -> encoder -> projection -> psi ------+
raw state/goal + detached encoder features
  + noisy action tokens + time -> MLP mixer teacher / one-step actor -> chunk
agent's current inference -> evaluator -> OGBench environment
```

Each critic sees one state token and one token per action with learned positions.
Each layer applies an FFN shared across token positions. Every `exchange_every`
layers (and at the final layer), a residual token-mixing MLP exchanges information
across positions, with weights shared across channels. The chunk remains tokenized.
Two critics use dot-product scores scaled by `sqrt(latent_dim)` and
binary NCE. Their goal encoder is shared; interaction backbones and score heads
are separate. The actor consumes raw state/goal and detached state/goal features
from this learned encoder, before its contrastive projection head.

Following the reference's FQL branch, a flow teacher learns behavior matching, and
a one-step stochastic actor maximizes the minimum critic score with teacher
matching regularization. Both use the same token architecture with independent
parameters. This baseline generates actions directly; latent action generation
remains a research extension. See [method details and references](docs/baseline.md).

## Structure and responsibilities

| Path | Responsibility |
| --- | --- |
| `configs/experiment/` | Offline defaults and synthetic smoke settings |
| `scripts/train.py`, `scripts/evaluate.py` | Thin command launchers |
| `src/craft/cli.py` | Config, component assembly, scheduled runtime services |
| `models/tokens.py` | Token interaction, critic encoder/heads, actor networks |
| `models/mlp.py`, `models/factory.py` | Standalone MLP example and model construction |
| `losses/contrastive.py` | Binary NCE and paired critic scores |
| `agents/craft.py` | Initialization, stage-specific updates, train state, inference |
| `policies/latent_chunk.py` | Bound inference facade (legacy name; direct-action baseline) |
| `data/offline.py` | Full-chunk and future-goal sampling without crossing episodes |
| `envs/factory.py` | Official OGBench loader and shape validation |
| `trainers/offline.py` | Multi-stage iteration, resume position, callbacks |
| `evaluators/rollout.py` | Per-task success, episode-local chunk execution/replanning |
| `utils/` | Console/W&B logging and atomic Flax checkpoint persistence |
| `tests/test_baseline.py` | Gradients, token order, stages, resume, sampling, rollout |
| `docs/` | Research plan, method decisions, and interface contracts |

Source paths in the table are relative to `src/craft/` where applicable. Modules
follow documented role contracts without mandatory base classes. Stages are
`joint` (default), `representation` (critic only), or `action` (teacher/actor only).
The same agent retains optimizer states across stage changes. Future online work
can reuse agent updates and inference while adding interaction and replay insertion.

Mixer defaults: `depth=4`, `exchange_every=2`. FFNs share weights across tokens,
not across layers. Token mixing depends on the configured chunk length. Previous
attention-model checkpoints are incompatible; start a new run/output directory.
