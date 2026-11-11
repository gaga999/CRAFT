# CRAFT CVAE POC

**Contrastive Representation for Action Flow and Temporal Abstraction**

A Flax NNX proof of concept combining CRL representations with a conditional VAE
and a stochastic latent prior. Standard vector-state, continuous-action OGBench
is supported. This is an experimental method, not a validated learning result.

## Run

```bash
uv sync --locked
# Each invocation appends a UTC timestamp and UUID to output_dir automatically.
JAX_PLATFORMS=cpu uv run craft --config configs/experiment/cube_single_noisy.yaml
# Short check using the same model; printed output names are unique.
uv run craft --config configs/experiment/smoke.yaml
uv run python -m unittest discover -s tests -v
uv run ruff check .
```

`--steps N` limits additional updates without changing the saved schedule.
`--dry-run` validates/prints settings. OGBench downloads data on first use.
Enable tracking with `logging.backend=wandb` (`WANDB_MODE=offline` for local runs).
For NVIDIA Linux, install `uv sync --locked --extra cuda` and keep `--extra cuda`
on `uv run`; select a device using `CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda`.
See [JAX device requirements](https://docs.jax.dev/en/latest/installation.html).

## Architecture

```text
(s, ground-truth A) -> phi Mixer tokens -> mean pool -> projection -> normalized phi
                  |                                          | BxB sigmoid loss
                  |       g -> independent psi MLP -> normalized psi
                  + cat psi goal feature -> posterior Mixer -> Gaussian Zq per action
s,g + detached psi(s),psi(g) -> prior MLP -> Gaussian Zp per action
Zq or Zp + position + gated state/goal condition -> decoder Mixer -> tanh actions
```

Chunk length is 5. Posterior fuses 6 phi tokens and one psi goal token, then reads
5 action positions. Decoder has 5 latent tokens; conditions have separate state/goal
projections, a learned scalar gate, and per-sample branch dropout (default 0.2).
Inference/guidance keep the condition on. Shared FFNs exchange information every
2 layers. Prior/decoder share no weights with critic; they use actual detached critic
features. There is no FQL teacher or student.

Losses: sigmoid CRL with learned positive scale/bias; posterior reconstruction plus
Gaussian KL; prior KL plus cosine critic guidance (enabled by default). Guidance
updates only prior weights, differentiating THROUGH fixed decoder/critic weights.
No hard action clip is used in training or guidance; tanh bounds decoder outputs.
CVAE-to-CRL gradients are optional (`agent.cvae_updates_crl`, default false).
See [method details](docs/baseline.md) and [contracts](docs/interfaces.md).

## Outputs and resume

`output_dir` is a **prefix**, e.g. `runs/cube_single_noisy-<UTC>-<UUID>`. The resolved
path is printed and saved in config/checkpoint. Reusing the prefix creates a new
run without mixing metrics. Use the actual printed directory to resume:

```bash
RUN_DIR="runs/your-printed-run-directory"
uv run craft --config "$RUN_DIR/config.yaml" --restore "$RUN_DIR/checkpoint.msgpack"
uv run python scripts/evaluate.py --config "$RUN_DIR/config.yaml" \
  --restore "$RUN_DIR/checkpoint.msgpack" evaluation.episodes=20
```

Resume creates a new session under the original run; evaluation writes outside it.
Original metadata/metrics remain immutable. Checkpoint **v4** stores all NNX model,
optimizer, RNG, sampler and schedule state; older FQL checkpoints are rejected.
Code changes on resume require explicit `--allow-code-change`. Sessions archive
resolved config, code/dependencies, environment metadata, dataset hashes and JSONL
metrics. Ctrl-C saves the latest accepted update; hard termination relies on the
last periodic save.

| Directory/file (under `src/craft/`) | Responsibility |
| --- | --- |
| `models/tokens.py`, `models/cvae.py` | Critic, shared token FFNs, posterior, prior and decoder |
| `agents/craft.py`, `losses/` | Gradient ownership, staged optimizers, CRL and CVAE losses |
| `policies/`, `evaluators/` | Inference composition and task rollouts |
| `config.py`, `experiments.py`, `cli.py` | Strict settings, assembly and entry points |
| `data/`, `envs/`, `trainers/`, `utils/` | Sampling, OGBench, scheduling, isolated RNG/session/checkpoint services |
