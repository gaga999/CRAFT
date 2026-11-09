# CRAFT: Contrastive Representation for Action Flow and Temporal Abstraction

**Contrastive Representation for Action Flow and Temporal Abstraction**

Research scaffold for offline goal-conditioned RL, with future online support in
mind. See [the research plan](docs/project-plan.md) for proposed methods. Algorithm
updates, dataset/environment adapters, rollout evaluation, and checkpointing are
not implemented yet.

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --locked
uv run python scripts/train.py --config configs/experiment/offline.yaml --dry-run
uv run python scripts/evaluate.py --config configs/experiment/offline.yaml --dry-run
# Equivalent training entry point:
uv run craft --config configs/experiment/offline.yaml --dry-run
uv run ruff check .
```

`--dry-run` resolves and prints the OmegaConf configuration; it does not initialize
models, read data, contact W&B, train, or evaluate. Running without it reports the
unfinished runtime assembly. For a supported Linux NVIDIA setup, the existing
CUDA extra is installed with `uv sync --locked --extra cuda`.

## Packages

| Package | Purpose |
| --- | --- |
| JAX | Array computation, autodiff, JIT, and explicit PRNG keys |
| Flax | Neural networks; the MLP example uses Linen, full-method API choice pending |
| Optax | Optimizers and parameter updates |
| NumPy | Host-side data processing |
| OmegaConf | YAML loading, interpolation, and resolved experiment settings |
| W&B (`wandb`) | Planned experiment tracking backend |
| Ruff / uv | Linting and dependency/environment management |

## Architecture

```text
scripts + config -> trainer -> data.sample() -> agent.update(batch, stage)
                                               | models + losses + optimizer
                    evaluator <- bound policy <- agent parameters
                         |          | encoder -> generator -> decoder -> chunk
                        env
```

The **agent** owns training state and algorithm-specific updates. The **policy**
composes models into inference behavior using current parameters. The **trainer**
controls stages and schedules logging, evaluation, and saving. One agent can run
representation and action stages with different update rules; separate agents are
useful when training independently reusable systems with checkpoint handoffs.

Interfaces are documented conventions supported by focused checks, without a
mandatory base class. Factories explicitly map configuration names to role-compatible
implementations. See [interface contracts and extension steps](docs/interfaces.md).
The stage names and budgets in the example config are illustrative.

## Project structure and responsibilities

```text
configs/experiment/offline.yaml    Illustrative offline settings and stage sequence
scripts/train.py                   Training launcher
scripts/evaluate.py                Evaluation launcher
docs/project-plan.md              Research scope, hypotheses, and comparisons
docs/interfaces.md                Shapes, state ownership, and extension workflow
src/craft/
  __init__.py                     Package and craft command entry point
  cli.py                          Configuration inspection; runtime assembly pending
  types.py                        Batch and metrics vocabulary
  models/mlp.py                   Executable minimal encoder example
  models/factory.py               Explicit encoder construction from configuration
  losses/reconstruction.py        Standalone MSE example; not the CRAFT objective
  policies/latent_chunk.py        Multi-model inference interface scaffold
  agents/craft.py                 State and stage-update interface scaffold
  data/offline.py                 Dataset sampling interface scaffold
  envs/factory.py                 Environment construction scaffold
  trainers/offline.py             Minimal reusable multi-stage training loop
  evaluators/rollout.py           Environment evaluation scaffold
  utils/logging.py                Console logger example
  utils/checkpoint.py             Persistence interface scaffold
tests/                            Future component/contract checks
```

Add a model by following its role contract, implementing it under `models/`, and
adding its name to the factory. Add a learning method through an agent's updates
and a policy's inference composition. Future online work adds collection and replay
insertion around compatible sampling/update interfaces.
