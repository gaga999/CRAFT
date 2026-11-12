# Directory and file map

```text
CRAFT/
├── README.md                 setup, commands, architecture summary
├── AGENTS.md                 rules for coding agents (engineering + documentation standards)
├── pyproject.toml / uv.lock  package "craft" (uv_build), console script `craft = craft:main`
├── configs/experiment/       shipped experiment YAMLs (see configs-scripts-tests.md)
├── scripts/                  thin python entry points (train.py, evaluate.py)
├── docs/                     this documentation
├── tests/                    unittest suite (run from repo root)
└── src/craft/                the package
```

## `src/craft/` by layer

AGENTS.md defines the layering: *models compute, policies compose inference, agents
own learning updates, trainers schedule work, run services own persistence*.

| Layer | File | Lines | Key symbols | Responsibility |
| --- | --- | --- | --- | --- |
| entry | `__init__.py` | 8 | `main` | Console-script target → `cli.train_main`. |
| entry | `cli.py` | 45 | `run`, `train_main`, `evaluate_main` | argparse: `--config`, `--dry-run`, `--restore`, `--steps`, `--output-dir`, `--allow-code-change`, dotlist overrides. |
| config | `config.py` | 225 | `ExperimentConfig` dataclasses, `load_config`, `validate_config`, `validate_agent` | Strict schema; rejects unknown keys, wrong types, NaN/Inf, invalid ranges; resolves paths to absolute. |
| assembly | `experiments.py` | 274 | `run_experiment`, `check_restore`, `checkpoint_metadata` | Builds everything for train / resume / evaluate, wires callbacks, writes status. |
| agent | `agents/craft.py` | 209 | `CRAFTAgent.update`, `.ae_objective`, `.prior_objective`, `.sample_action_chunk`, `.validation_metrics` | Owns the four optimizers and the full update step. |
| models | `models/tokens.py` | 207 | `TokenPointwiseLinear`, `MixerBlock`, `TokenBackbone`, `GoalEncoder`, `TokenCritic` | Critic phi/psi networks and the token-mixer block. |
| models | `models/cvae.py` | 119 | `TokenMixer`, `GaussianHead`, `Posterior`, `Prior`, `Decoder`, `sample_gaussian` | CVAE components. |
| models | `models/factory.py` | 22 | `make_poc_models` | Instantiates critic, posterior, decoder, prior from the agent config. |
| models | `models/mlp.py` | 18 | `MLP` | GELU MLP. |
| models | `models/similarity.py` | 10 | `siglip_logits` | `exp(log_scale)·phi·psi + bias`, paired or all-pairs. |
| losses | `losses/contrastive.py` | 19 | `contrastive_loss` | SigLIP sigmoid B×B loss + critic metrics. |
| losses | `losses/cvae.py` | 14 | `gaussian_kl` | Diagonal Gaussian KL(q‖p), summed over time/latent, averaged over batch. |
| losses | `losses/reconstruction.py` | 11 | `reconstruction_mse` | **Unused** standalone example. |
| policy | `policies/chunk.py` | 20 | `sample_action_chunk` | Inference: prior → sample → decoder (condition always on). |
| data | `data/offline.py` | 145 | `OfflineDataset`, `synthetic_dataset` | Validates arrays, builds valid chunk starts, samples indices/goals, gathers batches, fingerprints. |
| data | `data/loading.py` | 35 | `load_split`, `load_ogbench_datasets` | Downloads (via ogbench) and loads train/val NPZ in compact layout, concurrently. |
| data | `data/prefetch.py` | 169 | `PrefetchBatches` | Background planner + gather workers + async `device_put`; accept/commit RNG protocol. |
| env | `envs/factory.py` | 68 | `make_env`, `make_env_and_datasets` | OGBench env construction with space checks; closes env on failure. |
| trainer | `trainers/offline.py` | 60 | `train_offline` | Iterates stages and steps, enforces the finite-update barrier, timing metrics, calls `on_step`. |
| eval | `evaluators/rollout.py` | 68 | `evaluate_rollouts` | Per-task rollouts with chunk execution, deterministic seeds, success metrics. |
| run services | `utils/runs.py` | 136 | `allocate_session`, `provenance`, `write_json` | Unique session directories, source archive, atomic JSON. |
| run services | `utils/checkpoint.py` | 100 | `save_checkpoint`, `read_checkpoint`, `restore_checkpoint`, `pure_state` | Atomic msgpack checkpoints (v5), validation before mutation. |
| run services | `utils/random.py` | 86 | `make_random_engines`, `RandomEngine` | All RNG construction; separate training/evaluation engines and streams. |
| run services | `utils/logging.py` | 49 | `Logger` | `metrics.jsonl` (exclusive create) + optional W&B + stdout. |
| misc | `types.py` | 7 | `Batch`, `Metrics` | Type aliases (currently not imported anywhere). |

## Call graph (training)

```text
craft:main → cli.train_main → cli.run
  → config.load_config → validate_config
  → experiments.run_experiment
      → utils.runs.provenance
      → utils.checkpoint.read_checkpoint / check_restore            (resume / evaluate)
      → utils.random.make_random_engines
      → utils.runs.allocate_session
      → envs.factory.make_env_and_datasets → data.loading → data.offline.OfflineDataset
        (or data.offline.synthetic_dataset)
      → agents.craft.CRAFTAgent.create → models.factory.make_poc_models
      → utils.checkpoint.restore_checkpoint                         (resume)
      → utils.logging.Logger
      → trainers.offline.train_offline
           → data.prefetch.PrefetchBatches
           → CRAFTAgent.update → losses.*, models.*
           → on_step: CRAFTAgent.validation_metrics, evaluators.rollout.evaluate_rollouts
                      (→ CRAFTAgent.sample_action_chunk → policies.chunk), save_checkpoint
```

Evaluation (`scripts/evaluate.py` → `cli.evaluate_main`) follows the same path
with `evaluate=True`. It loads no dataset, runs `evaluate_rollouts` once and
writes a fresh session.
