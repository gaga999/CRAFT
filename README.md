# CRAFT

**Contrastive Representation for Action Flow and Temporal Abstraction**

Offline goal-conditioned CRL with MLP action tokens and an embedding-conditioned
FQL actor. Standard vector-state, continuous-action OGBench datasets are supported;
online collection, images, and discrete actions are not implemented.

## Quickstart

Python 3.12+ and [uv](https://docs.astral.sh/uv/); OGBench is installed automatically.

```bash
uv sync --locked
uv run craft --config configs/experiment/smoke.yaml output_dir=runs/smoke-new
uv run python -m unittest discover -s tests -v
uv run ruff check .

# Cube single noisy; official train/validation data download on first use.
JAX_PLATFORMS=cpu uv run craft --config configs/experiment/cube_single_noisy.yaml
# Short plumbing check, preserving the full schedule for later resume:
JAX_PLATFORMS=cpu uv run craft --config configs/experiment/cube_single_noisy.yaml \
  --steps 3 output_dir=runs/cube-check \
  train.log_interval=1 train.eval_interval=3 evaluation.episodes=1 \
  'evaluation.task_ids=[1]'
```

`--dry-run` resolves and validates settings without side effects. Dotlist overrides
are strict: unknown keys, invalid types, and nonfinite values fail early. CPU configs
are capacity/budget choices, not tuned or paper-equivalent results. `--steps N`
limits additional updates in this invocation; it does not change the saved schedule.

## Runs, resume, evaluation

New training **requires a nonexistent output directory**. Old files are never
appended or overwritten. Resume validates config, dataset hashes, and checkpoint
structure; code changes require an explicit `--allow-code-change` after inspection.

```bash
uv run craft --config runs/cube_single_noisy/config.yaml \
  --restore runs/cube_single_noisy/checkpoint.msgpack
uv run python scripts/evaluate.py --config runs/cube_single_noisy/config.yaml \
  --restore runs/cube_single_noisy/checkpoint.msgpack evaluation.episodes=20
```

Resume writes a fresh `sessions/resume-*/` under the original run; its checkpoint
path is printed on exit. Evaluation loads only the environment and writes to a fresh
sibling `evaluations/` directory; `--output-dir PATH` can select another fresh location
outside the source run. It may override evaluation/logging settings. Neither mode
rewrites the original config, metadata, metrics, or checkpoint.

Each session stores resolved config, code/dependency snapshot, environment metadata,
JSONL metrics, status, and dataset hashes. Training checkpoints include parameters,
optimizers, step, and training/validation RNG. `Ctrl-C` saves the latest accepted
update; hard termination relies on the last periodic checkpoint. NNX checkpoint
format v3 records the new RNG layout and rejects older checkpoints; start a new run.
To enable W&B, set `logging.backend=wandb`; `WANDB_MODE=offline` keeps tracking local. Each session has
its own W&B run and one merged metrics record per logged step.

## Architecture

**Flax NNX** owns model graphs, parameters, optimizers, and in-place JIT updates;
there is no Linen/TrainState layer. All gradients are computed from pre-update
weights, and nonfinite updates are rejected atomically.

```text
CLI/config -> experiment services -> trainer -> data.sample -> agent.update
state + ordered action tokens -> shared FFNs / periodic MLP exchange -> phi
future goal -> encoder / projection -> psi -> binary contrastive score
state + goal + detached encoder features + noise -> chunk policy -> actions
checkpoint + environment -> evaluator -> independent evaluation session
```

FFN weights are shared across tokens, with independent parameters per layer.
`exchange_every` controls token mixing; the final layer also exchanges information.
Two critic heads share the goal encoder. Flow teacher and one-step actor are separate
NNX policies. `joint`, `representation`, and `action` stages retain separate optimizer
states. Entry points create training/evaluation engines through `utils/random.py`;
modules receive their streams without reseeding. Model initialization, updates,
data, and validation are isolated; evaluation task keys are repeatable and never
advance training RNG. Checkpoints save JAX keys/counters and NumPy stream states.
[Method/config rationale](docs/baseline.md) and [contracts](docs/interfaces.md).

| Location | Responsibility |
| --- | --- |
| `config.py`, `cli.py` | Strict schema and thin entry points |
| `experiments.py`, `utils/` | Assembly, isolated sessions, logging, checkpoints |
| `models/`, `policies/` | NNX encoders/mixers and action-generation composition |
| `agents/`, `losses/` | Gradient ownership, optimizer updates, objectives |
| `data/`, `envs/` | Episode-safe sampling and official OGBench adapters |
| `trainers/`, `evaluators/` | Scheduling, resume position, task rollouts |
| `configs/experiment/`, `tests/` | Experiment settings and regression/numerical checks |

Source locations are relative to `src/craft/`. Core packages: JAX, Flax NNX, Optax,
NumPy, OmegaConf, OGBench, and W&B; uv/Ruff/unittest support development.

## Devices

```bash
# Supported NVIDIA Linux machine; keep the extra on subsequent uv run commands.
uv sync --locked --extra cuda
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda uv run --extra cuda craft \
  --config configs/experiment/cube_single_noisy.yaml output_dir=runs/gpu-cube
uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
```

The CUDA 13 extra needs a compatible NVIDIA driver/GPU; see [JAX requirements](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu).
It does not enable Apple GPU acceleration. Training uses one device; select CPU
with `JAX_PLATFORMS=cpu` before Python starts.
