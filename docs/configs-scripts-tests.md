# Configs, scripts and tests

## Shipped configs (`configs/experiment/`)

All share `chunk_length=5`, `width=128`, `depth=4`, `exchange_every=2`,
`latent_dim=128`, `vae_latent_dim=16`, `fusion_depth=2`, LR 3e-4 (prior 1e-4),
`condition_drop_prob=0.2`, `batch_size=256`, `execution_length=1`, one `joint`
stage. Only `smoke.yaml` uses a tiny model. The table lists only the differences.

| File | Dataset | β (kl) | λ (guide) | γ | Steps | log / eval / save | Eval eps | Backend | Purpose |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `smoke.yaml` | synthetic | 0.01 | 0.1 | 0.95 | 3 | 1 / 100 / 3 | 1 | console | plumbing check; width 16, depth 2, Z=3, B=4, `workers=0` |
| `cube_single_noisy.yaml` | cube-single-noisy-v0 | 0.01 | 0.1 | 0.95 | 1M | 100 / 10k / 10k | 10 | console | main README example |
| `cpu_cube.yaml` | cube-double-play-v0 | 0.01 | 0.1 | 0.95 | 100k | 100 / 5k / 5k | 2 | wandb | short run on harder cube task |
| `cpu_puzzle.yaml` | puzzle-3x3-noisy-v0 | **0.03** | **0.4** | **0.99** | 1M | 100 / 5k / 5k | 2 | wandb | puzzle variant with stronger guidance |
| `offline.yaml` | pointmaze-medium-navigate-v0 | 0.01 | 0.1 | 0.95 | 1M | 1k / 50k / 50k | 20 | console | navigation; `output_dir: runs/crl_tokens` (legacy name) |

Schema defaults in `config.py` (used for any key a YAML omits): dataset
`puzzle-3x3-noisy-v0`, `output_dir=runs/puzzle`, log/eval/save `1000/50000/10000`,
`evaluation.episodes=10`, `task_ids=[]` (all tasks). None of the shipped configs
sets `evaluation.task_ids`.

Every config supports dotlist overrides, e.g.
`craft --config configs/experiment/cube_single_noisy.yaml agent.guide_weight=0.3 seed=1`.

## Scripts (`scripts/`)

| File | Equivalent | Notes |
| --- | --- | --- |
| `train.py` | `craft` console script | duplicate entry point for `uv run python scripts/train.py` |
| `evaluate.py` | `cli.evaluate_main` | no console script exists for evaluation; requires `--restore` and a real env |

## Tests (`tests/`, stdlib `unittest`)

Run them from the repository root: the tests use relative paths such as
`configs/experiment/smoke.yaml` and `src/craft`.

```bash
uv run python -m unittest discover -s tests -v
```

| File | Covers |
| --- | --- |
| `test_baseline.py` | shapes of all CVAE tensors; every module updates under `joint`; update reproducibility; the `cvae_updates_crl` switch; dropout blocks the whole condition; guidance gradients reach the prior through a frozen decoder and none when disabled; guidance changes only the prior; validation is read-only and repeatable; KL formula; non-finite batches are rejected atomically; stage freezing; checkpoint round-trip continues identically; phi depends on action order. Defines the shared `FakeEnv`, `seeded_agent`, `seeded_source` helpers. |
| `test_model_contracts.py` | conv token mixing equals a dense reference (outputs and gradients, no transposes); SigLIP score/loss equations; guidance gradient scales with `exp(log_scale)` and is invariant to the bias; the gate stays positive and has a gradient. |
| `test_data_pipeline.py` | compact = regular layout (batches, fingerprint, memory); float32 aliasing; invalid-data rejection; adapter loads both splits and closes env on failure; prefetch ordering and live-RNG equality for workers 0/1/3; lookahead regeneration after a JSON RNG restore; real concurrency; worker failure propagation; no deadlock on a full queue; trainer stage order and resume batches; reject/interrupt RNG handling; prefetch-key restore contract. |
| `test_random_and_clipping.py` | engine stream independence and determinism; seed bounds; atomic numpy restore; init uses the live engine; **no RNG construction outside `utils/random.py`** (AST scan); evaluation cadence does not change training; v<5 checkpoints are rejected. (Its docstring mentions "saturated student outputs", a leftover from the removed FQL student.) |
| `test_runtime.py` | all shipped configs load; invalid overrides fail; run collision → new directory; resume equals an uninterrupted run (model + optimizer + metrics steps); evaluation never touches the source run and loads no dataset; resume rejects config/dataset changes without writing; interrupt saves a checkpoint and sets `status=interrupted`; Logger rejects duplicate steps, NaN and append. |

Known issue (from reading, see [code-review.md](code-review.md) R1):
`test_runtime.ConfigTests.test_all_shipped_configs_and_coefficient` asserts that
every shipped config has `guide_weight == 0.1` and `discount == 0.95`, but
`cpu_puzzle.yaml` uses 0.4 and 0.99, so this test fails as written.
