# Data pipeline (`src/craft/data/`, `envs/factory.py`)

## Sources

| Source | Built by | Layout |
| --- | --- | --- |
| OGBench `<name>.npz` + `<name>-val.npz` | `load_ogbench_datasets` (downloads via `ogbench.utils.download_datasets` into `data.directory` unless `data.path` is set) | **compact** |
| custom `data.path=/x/foo.npz` (needs `/x/foo-val.npz`) | same | compact |
| `data.synthetic=true` | `synthetic_dataset` (8 episodes × 32 steps, O=6, A=2) | regular, no validation split, no env |

Both splits are decompressed concurrently in two threads (`loading.py`). Only
`observations`, `actions` and `terminals` are read.

### Regular vs compact layout (`OfflineDataset.__init__`)

- **Regular**: `observations[N,O]`, `next_observations[N,O]`, `actions[N,A]`,
  `terminals[N]`, where the last transition of each episode has `terminals=1`.
- **Compact** (OGBench storage): each episode stores **one extra observation
  slot** (the final state) whose `terminals=1` and whose action is a dummy.
  `next_observations` is implied by `observations[i+1]`. Dummy actions are excluded
  from the finite and bounds checks.

`episode_ends` = last real transition index of each episode (compact: terminal
slot − 1). Both layouts produce identical batches and identical
`fingerprint()`s. `tests/test_data_pipeline.py` checks this against
`ogbench.utils.load_dataset`.

Validation at load time: vector observations and float actions; matching lengths;
finite values; `|action| ≤ 1.0001`; binary terminals; the data must end on a
boundary; every episode has at least one transition; at least one episode is at
least `H` long. Arrays are cast to float32 once, and float32 input is used without
a copy.

## Sampling (`sample_indices` + `gather`)

```text
i        ~ uniform over valid starts   (start s.t. i..i+H-1 lie in one episode)
remaining = episode_end(i) − i + 1      (transitions left, including i)
k_value  = min(Geometric(1−γ), remaining)            ≥ 1
k_actor  = Uniform{1..remaining}                      (drawn but unused — kept for RNG sequence)
batch:
  observations  = obs[i]                              [B,O]
  action_chunks = actions[i : i+H]                    [B,H,A]
  value_goals   = state at time i + k_value           [B,O]   (compact: obs[i+k]; regular: next_obs[i+k−1])
  actor_goals   = state at time i + k_actor           only if include_actor_goals (CRAFT never uses it)
```

- The goal offset is measured from the **chunk start**, so it can fall inside the
  chunk (`k < H`). Truncating at `remaining` puts the leftover mass on the
  episode's final state. This mirrors OGBench's `GCDataset`.
- Sampling is with replacement, and `rng.choice` uses the dataset's NumPy
  `Generator`.
- `fingerprint()` is a SHA-256 over the logical transitions in regular layout,
  hashed per episode for compact data. It is stored as the dataset identity in
  checkpoints and in `dataset.json`.

## Prefetch and the RNG commit protocol (`prefetch.py`)

```text
             private RNG copy (_rng)                     dataset.rng (live, checkpointed)
producer thread:  plan(indices, rng_state_after) ─┐
                  ThreadPool(workers).gather ─────┤  up to prefetch_batches in flight
                  jax.device_put(..., device) ────┘→ Queue(maxsize=device_prefetch)
consumer (trainer):  next() → batch                 ...update...   accept() ─► dataset.rng.state = rng_state_after
```

- Only the producer thread plans indices, serially, so the sample order does not
  depend on the worker count. Workers only gather, by advanced indexing into the
  shared arrays, and never copy the dataset.
- `accept()` copies the planned RNG state into the live generator. Rejected or
  interrupted batches are therefore never "consumed". After a resume the
  lookahead is regenerated identically from the checkpointed state. This is why
  `data.workers`, `prefetch_batches` and `device_prefetch` may change on resume.
- `next()` refuses to hand out a new batch until the previous one is accepted.
- A worker exception is wrapped in `WorkerFailure` and re-raised in the consumer.
  `close()` stops and joins the producer and drains the queued device buffers.
- `workers=0` is fully synchronous: plan, gather and `device_put` all happen
  inside `next()`.
- The arrays must not be mutated while a prefetcher is active.

## Environment adapter (`envs/factory.py`)

`make_env(name)` rejects `visual`, `powderworld`, `oraclerep` and `singletask`
datasets. It builds `ogbench.make_env_and_datasets(name, env_only=True)` and
requires 1-D `Box` spaces with action bounds `[-1,1]`. `make_env_and_datasets`
also loads both splits and checks their dimensions against the env. On any
failure it closes the env before re-raising.
