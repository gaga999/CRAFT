# Runtime: CLI, config, experiment, trainer, evaluator, run services

## CLI (`cli.py`)

```bash
craft --config CFG.yaml [key=value ...]          # = python scripts/train.py
python scripts/evaluate.py --config RUN/config.yaml --restore RUN/checkpoint.msgpack [evaluation.episodes=20]
```

| Flag | Effect |
| --- | --- |
| `--dry-run` | print the resolved config and exit (it is still validated) |
| `--restore CKPT` | resume training (train entry) or choose the checkpoint (evaluate entry) |
| `--steps N` | at most N more updates in this invocation; the saved schedule is unchanged; training only |
| `--output-dir DIR` | evaluation only: prefix for the evaluation session (must be outside the source run) |
| `--allow-code-change` | permit resume when `source_sha256` differs from the checkpoint's |
| `key=value` | OmegaConf dotlist overrides, e.g. `logging.backend=wandb` |

## Config (`config.py`)

The schema is a set of dataclasses: `agent`, `data`, `train` (`stages[]`,
intervals), `evaluation` (`episodes`, `execution_length`, `task_ids`), `logging`,
plus `seed` and `output_dir`. `load_config`:

1. Merges the YAML with the overrides **without** a schema, to keep the raw
   types.
2. Merges again into the structured schema; unknown keys raise.
3. `verify_types` compares the raw values with the schema types. An int is
   accepted for a float field, but a bool is not accepted for an int field.
4. `validate_config` runs the range checks: the agent keys must equal the schema
   exactly; positive ints and finite floats; `0 < discount < 1`;
   `-10 ≤ log_std_min < 0 < log_std_max ≤ 3`; stage names come from
   `{joint, representation, action}`; total steps `< 2³¹`;
   `execution_length ≤ chunk_length`; unique positive `task_ids`; dataset names
   must be `*-v0` and not visual etc.; `data.path` must be `.npz`.
5. Resolves `output_dir`, `data.directory` and `data.path` to **absolute** paths.

## Experiment assembly (`experiments.run_experiment`)

Validation runs before any side effect, in this order: config → custom dataset
files exist → evaluate needs a checkpoint and a real env → `--steps` is
training-only → read the checkpoint and run `check_restore` (config equality) →
code hash unless `--allow-code-change` → `--output-dir` is evaluation-only.

```text
make_random_engines(seed) → (training_engine, evaluation_engine)
session = allocate_session(mode=train)            (fresh run only; config.output_dir := session)
train/resume:  datasets (+ env) built from training_engine numpy streams "data"/"validation"
               fingerprint + dims must match the checkpoint on resume
evaluate:      env only; dims must match the checkpoint
agent = CRAFTAgent.create(..., rngs=training_engine.jax)
resume/eval:   restore_checkpoint → (train) restore numpy streams → allocate_session(resume|evaluate)
write dataset.json, Logger(session)
evaluate:      log evaluate_rollouts once at saved step → status completed
train:         train_offline(on_step) → save → status completed|stopped
```

`check_restore` allows these differences:

- **resume**: every top-level key must equal the saved config, except
  `data.workers`, `data.prefetch_batches` and `data.device_prefetch`.
- **evaluate**: only `seed`, `agent`, `data` (minus the prefetch keys), `train` and
  `output_dir` must match, so `evaluation.*` and `logging.*` may change.

`on_step(step, stage, metrics)` merges everything into one log row per step,
because W&B drops repeated same-step logs:

- every `log_interval` steps: training metrics plus validation metrics, if a
  validation split exists
- every `eval_interval` steps (real env only): `evaluate_rollouts` metrics
- every `save_interval` steps: checkpoint

Status handling: on `KeyboardInterrupt` it saves a checkpoint, writes
`status=interrupted` and re-raises. Any other exception writes `status=failed`
with the error and **does not save**.

## Trainer (`trainers/offline.train_offline`)

It walks `stages` cumulatively from `agent.step` up to
`min(total, start + max_steps)`. For each step: `next(batches)` → `agent.update` →
`bool(numerics/finite)`, which is a device sync and acts as the per-step barrier
→ `accept()`. If an exception arrives after the agent has already committed the
step (for example Ctrl-C during the update), the batch is accepted first so that
the sampler and the model stay consistent.

## Evaluator (`evaluators/rollout.evaluate_rollouts`)

For each selected task (default: all tasks `1..len(task_infos)`):

- the JAX key is `evaluation_engine.task_key(task_id)`, so it is independent of
  task order and evaluation cadence
- for each episode, `env.reset(seed=episode_seed(task, ep), options={task_id})`
- it re-plans a chunk every `execution_length` steps with
  `agent.sample_action_chunk(obs, goal, key)` and checks shape, finiteness and
  bounds
- success = max of `info["success"]` over the episode; the episode must
  terminate or truncate within `spec.max_episode_steps`

It returns `evaluation/<task_name>/success` and `evaluation/overall_success`.
Rollouts are sequential: one env, one observation per call.

## Random engines (`utils/random.py`)

```text
root = key(seed, threefry)
engine(role): role 0 = training, role 1 = evaluation
  jax nnx.Rngs: params=fold(root,role,0)  updates=fold(..,1)  evaluation=fold(..,2)
  numpy:        data(100) validation(101) synthetic(102)  via SeedSequence([seed, role, domain])
  episode_seed(task, ep) = SeedSequence([seed, role, 200, task, ep])
```

| Stream | Used for | Persisted? |
| --- | --- | --- |
| training.jax.params | model init | yes, in agent state |
| training.jax.updates | update keys (q/p samples, dropout mask) | yes, in agent state (the counter rolls back on reject) |
| training.numpy data / validation / synthetic | train sampler / val sampler / synthetic data | yes, `run_state.training_engine` |
| evaluation.jax.evaluation | `task_key(t)` for rollouts; `task_key(0)` for validation-metric noise | derived (stateless) |
| evaluation episode seeds | env reset seeds | derived (stateless) |

A test enforces that no other module constructs keys or generators
(`test_key_and_generator_construction_are_centralized`).

## Checkpoints (`utils/checkpoint.py`)

The payload is msgpack:
`{format_version: 5, agent: <pure NNX state dict>, run_state: JSON string}`.

- `agent` holds every `nnx.Variable`: params, all four optimizer states, the RNG
  keys (stored as raw key data) and `step`.
- `run_state` = `{step, config, dimensions, dataset_identity, training_engine
  (numpy states), source_sha256}`.
- Save: it checks that all leaves are finite, writes to a temp file in the same
  directory, fsyncs, then calls `os.replace`, so the write is atomic.
- Restore: it checks the format version, finiteness, and that the flattened
  paths, shapes and dtypes match the live agent **before** calling `nnx.update`.
  PRNG keys are re-wrapped with their original implementation.

## Run directories (`utils/runs.py`)

```text
runs/<prefix>-<YYYYmmddTHHMMSSZ>-<uuid8>/        fresh training session (mkdir exist_ok=False)
├── config.yaml        resolved config (output_dir = this directory)
├── metadata.json      mode, start time, provenance (git rev, dirty, source_sha256, python, platform, devices, all package versions), source checkpoint + sha256
├── source.zip         src/**/*.py + pyproject.toml + uv.lock
├── dataset.json       train/validation fingerprints
├── status.json        initializing → completed | stopped | interrupted | failed
├── metrics.jsonl      one JSON row per logged step (opened with mode "x": never appended)
├── checkpoint.msgpack
└── sessions/resume-<token>/     one per resume; same files; ITS checkpoint.msgpack holds the newer state
runs/evaluations/<run-name>-<token>/   one per evaluation (or <--output-dir>-<token>)
```

Note: a resumed session writes its checkpoint in its **own** session directory,
and the root `checkpoint.msgpack` keeps the old step. To continue a second time,
resume from `RUN/sessions/resume-…/checkpoint.msgpack`, using that session's
`config.yaml` or the root one (they are equivalent). See code-review R2.

`provenance()` hashes `src/**/*.py`, `pyproject.toml` and `uv.lock` under the repo
root, located as `utils/runs.py`'s `parents[3]`. The resume guard compares this
hash (`source_sha256`).

## Logging (`utils/logging.py`)

`Logger` opens `metrics.jsonl` exclusively. It requires strictly increasing steps
within a session, converts every metric to `float` and serializes with
`allow_nan=False`, so a NaN metric raises. It mirrors each row to W&B
(`resume="never"`, run name = session directory name) and prints it to stdout.
