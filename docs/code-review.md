# Code review: issues and suggestions

Scope: every file under `src/`, `scripts/`, `tests/`, `configs/`, plus README and
the existing docs, at commit `0fe6b26`. Findings come from reading the code; the
test suite was **not** executed for this review. Severity:
**R** = affects correctness, reproducibility or research validity,
**M** = maintainability, efficiency or usability. Each item gives the location,
the problem, and a suggested fix.

The engineering baseline is strong overall: strict config validation, atomic
versioned checkpoints, deterministic prefetch with an RNG commit protocol,
isolated RNG streams, immutable run directories, and tests for gradient ownership.
Most findings are edge cases in that lifecycle and a few modeling questions.

## R — correctness / reproducibility / research validity

### R1. A shipped config contradicts a test → the suite fails
`tests/test_runtime.py:28-32` asserts that every `configs/experiment/*.yaml` has
`guide_weight == 0.1` and `discount == 0.95`. `cpu_puzzle.yaml` has
`guide_weight: 0.4` and `discount: 0.99` (and `kl_weight: 0.03`).
**Fix:** decide which one is intended. If the puzzle settings are deliberate,
rewrite the test to check what actually matters (e.g. that the configs load and
`smoke.yaml` matches the schema defaults), or keep a per-file expected table.
Record why puzzle uses different coefficients in `baseline.md`; per AGENTS.md
these are untuned and must not be copied across tasks silently.

### R2. Resumed checkpoints live in the session directory; the root checkpoint goes stale
`experiments.py:187-194` saves to `session / "checkpoint.msgpack"`. For a resume
the session is `RUN/sessions/resume-<token>/` (`utils/runs.py:84-85`). README only
documents `--restore "$RUN_DIR/checkpoint.msgpack"`, so a second resume, or an
evaluation that follows the README, silently starts from or evaluates the **old**
step. This creates divergent branches of one run that are easy to mistake for one
another.
**Fix (pick one):** (a) write a `RUN/latest.json` pointer (`{session, step,
sha256}`) after each save and have the CLI accept `--restore RUN` to follow it;
(b) at minimum print the new checkpoint path at exit and document the chain in
README. Keep the root checkpoint immutable, as the provenance rules require.

### R3. A non-finite update aborts training without saving the last good state
`trainers/offline.py:45-48` raises `FloatingPointError` after a rejected update.
`experiments.py:257-267` marks the run `failed` but never calls `save()`, so up to
`save_interval` updates (10k on `cube_single_noisy`) are lost, even though the
agent is guaranteed to be unchanged and finite. The same applies to any other
exception (for example R8).
**Fix:** in the `except Exception` path of training, call `save()` when
`agent is not None`, guarded by its own try so the original error is still
raised. Optionally add a config option to skip N rejected batches instead of
aborting; that is a methodological choice and should be logged.

### R4. Guidance can exploit the critic off the data manifold
`agents/craft.py:80-96`: the prior maximizes `score(s, D(z_p), g)` through a
frozen decoder. The only thing that keeps `z_p` close to the posterior is
β·KL with β = 0.01, against λ = 0.1, and the critic has only seen dataset chunks
(it never trains on decoded chunks). The logit is bounded (unit vectors), so it
cannot diverge, but the prior can move to latents whose decoded chunks the critic
over-scores. Nothing logged would reveal this: `validation/vae/prior_sample_mse`
measures distance to the dataset chunk, not critic reliability.
**Suggestions:** log on validation (i) the critic score of prior-decoded chunks
vs dataset chunks vs posterior-decoded chunks, and (ii) the prior-vs-posterior KL
of the latent means. Run a guidance-off control (`agent.guidance_enabled=false`)
with matched seeds before attributing success changes to guidance.

### R5. Goal offsets can fall inside the action chunk
`data/offline.py:102-108`: `k ~ Geometric(1−γ)` is measured from the chunk start,
so with γ = 0.95, `P(k ≤ 4) ≈ 18.5%` of positives lie inside the chunk the critic
is conditioned on. These are near-trivial positives for an action-chunk critic,
and episode-end truncation adds mass at the final state. This matches OGBench's
`GCDataset`, but it has not been checked against the CRL+action-chunking
reference ("Three Steps at a Time"), which `project-plan.md` §4 says must be
preserved.
**Fix:** verify the reference's goal-offset convention. If it samples from
`t+H`, add a documented option (e.g. `data.goal_offset_start: chunk_start |
chunk_end`) and keep the current behavior only if it matches.

### R6. Validation metrics are not checked for finiteness, and a NaN kills the run
`agents/craft.py:186-209` returns validation metrics without a finite check.
`utils/logging.py:35-36` serializes with `allow_nan=False`, so a NaN validation
value raises `ValueError` inside `on_step`. Combined with R3, the run is marked
`failed` and loses its unsaved progress.
**Fix:** decide on a policy. Either fail fast but save first (R3), or log
non-finite validation values as an explicit counter or `null` so that a single
bad validation batch does not end a 1M-step run.

## M — maintainability / efficiency / usability

### M1. Stage-frozen groups still compute full gradients
`agents/craft.py:111-139` always computes the CRL, CVAE and prior (two VJP pulls)
gradients; `commit` then discards those of frozen groups. In a `representation`
stage this costs roughly 2-3× the necessary compute. Because `stage` is static
under `jit`, a Python `if stage == ...` can skip the unused branches. Keep the
metrics that are still meaningful.

### M2. Dead code and stale names
- `losses/reconstruction.py`: unused "standalone example". Delete it.
- `models/tokens.py:79-139`: the `condition_dim` / `condition_projection` path of
  `TokenBackbone` is never used (an actor-conditioning scaffold). Remove it, or
  move it to the branch that needs it.
- `data/offline.py:106-107, 124-125`: the `actor_goals` RNG draw exists only to
  keep the RNG sequence; nothing consumes it. Checkpoint v5 is already a hard
  compatibility break, so remove the draw the next time the format version is
  bumped.
- `types.py`: `Batch` / `Metrics` are never imported. Use them in signatures or
  delete them.
- `config.py:102` error message mentions FQL; `tests/test_random_and_clipping.py:1`
  docstring mentions a "saturated student". Both refer to removed designs.
- `models/cvae.py:51`: `Posterior.action_projection` also projects the state
  token. A name like `token_projection` would be accurate.
- `configs/experiment/offline.yaml`: `output_dir: runs/crl_tokens` is a legacy
  name and the file name `offline.yaml` doesn't say "pointmaze". Rename it to
  `pointmaze_medium.yaml`.

### M3. Decoder condition path is three stacked affine maps
`models/cvae.py:115-117`: `Linear_c(Linear_s(hs) + Linear_g(hg))` with no
nonlinearity in between collapses to one affine map of `[hs, hg]`. The extra
layer adds parameters but no expressiveness. Either insert a nonlinearity
(e.g. GELU) or drop `Linear_c`; the masking semantics hold either way. Changing it
alters the state layout, so it needs a checkpoint version bump.

### M4. Config defaults vs shipped configs; misleading names
- The `config.py` defaults (`runs/puzzle`, puzzle dataset, intervals
  1000/50000/10000) match no shipped YAML. AGENTS.md asks that defaults and
  shipped configs be consistent. Align the defaults with `cube_single_noisy.yaml`
  (the README's main example), or make them obviously minimal.
- `cpu_cube.yaml` / `cpu_puzzle.yaml` use the full 128-wide model and `wandb`
  logging, so the `cpu_` prefix suggests a CPU compromise that isn't there.
  `cpu_cube` also runs 100k steps where the others run 1M. Rename them, or add a
  header comment saying what each config is for.
- README runs `JAX_PLATFORMS=cpu` on a 1M-step config. Either point the CPU
  example at a short budget (`--steps`) or say that it is slow.

### M5. Evaluation throughput and per-call `nnx.jit` overhead
`evaluators/rollout.py`: one env, one observation per call, synchronous, inside
the training loop (training stops while it runs). Each call goes through
`nnx.jit` on a bound method, which re-traverses the module graph every call; the
same overhead hits `agent.update` once per training step.
**Suggestions:** measure `time/update_seconds` first. If the overhead matters,
use `nnx.split` once + `jax.jit` over `(graphdef, state)` (or
`nnx.cached_partial`) for the update and the actor. Batch evaluation episodes
with vectorized envs.

### M6. Runs are not relocatable
`config.py:158-166` stores absolute `output_dir` / `data.directory`, and
`check_restore` (`experiments.py:29-52`) requires exact equality, including for
evaluation. Copying a run to another machine or path (as AGENTS.md describes for
the ws* hosts) blocks resume and evaluation unless the absolute path is
identical. **Fix:** exclude `output_dir` and `data.directory` from the equality
check (dataset identity is already verified by fingerprint), or compare them
relative to the run root.

### M7. Resume cannot change logging or evaluation settings, and errors don't name the key
Resume compares every top-level key (`experiments.py:30-31`), so switching
`logging.backend` to wandb or changing `evaluation.episodes` on resume is
rejected. That may be intentional, but the error message does not say which key
differs. **Fix:** compute and print the list of differing dotted keys. Consider
whitelisting `logging.*` (not part of the experiment's science) for resume.

### M8. `provenance()` assumes an editable install from the repo layout
`utils/runs.py:41` and `:121` locate the repo with `Path(__file__).parents[3]`.
Under a non-editable install (e.g. `pip install .` or a wheel) that resolves
inside `site-packages/`. The `src/**/*.py` glob is then empty and
`(root / "pyproject.toml").read_bytes()` raises `FileNotFoundError`, so every run
fails at startup with an unhelpful error. **Fix:** detect the missing
`pyproject.toml` and raise an actionable message ("run from a uv editable
checkout"), or hash the installed package's files via `importlib.resources`. Also note that `provenance()` runs twice per session
(`experiments.py:77`, `runs.py:113`), each time calling `git status` and hashing.
Compute it once and pass it in.

### M9. Lifecycle details
- No evaluation runs at the end of training unless the final step is a multiple
  of `eval_interval` (e.g. a `--steps` partial run). Consider a final evaluation
  when the full schedule completes.
- When the final step is a multiple of `save_interval`, the checkpoint is written
  twice (`on_step` and then `save()` at `experiments.py:236`). Skip the second
  write if the step was just saved.
- Every log row prints the full metrics dict to stdout (`logging.py:42`), which
  is noisy at `log_interval=100`. Print a short summary instead.
- `validation_metrics` reuses `evaluation_engine.task_key(0)`, sharing a
  namespace with task ids. A dedicated `validation` key stream would make the
  intent explicit.
- `config.py:117` error names `kl_weight` without the `agent.` prefix that the
  other messages use.

### M10. Smaller modeling notes (not bugs; worth recording in baseline.md)
- `critic.bias` is initialized from `batch_size` (`tokens.py:188`), so the model
  depends on a data setting. That is fine, since the batch size is locked on
  resume, but it should be documented.
- Sampling is with replacement, and goals from the same episode act as negatives,
  so the batch contains some false negatives and duplicate anchors. This is
  standard for CRL, but it inflates and deflates `critic/accuracy` slightly.
- Only one sample of `z` is used for `prior_sample_mse`. Multiple samples, or
  best-of-k, would separate "the prior is wrong" from "the prior is
  multimodal".
- The decoder's state/goal condition uses `encode` (pre-projection psi features).
  The posterior and the prior see the raw state, but the decoder sees the state
  only through these features, so its information about the state is limited to
  what psi keeps.

## Operational note (environment, not repository code)

On `ws2`, `python -m unittest` crashed while XLA was creating its thread pool
(`pthread_create failed`, `ulimit -u` = 2048 shared by every thread of the
user). Pinning the process to fewer cores (`taskset -c 0-7 …`) or running on a
less loaded host is the likely workaround; this has not been verified.
