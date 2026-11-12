# CRAFT documentation index

Start here. Each document answers one question; follow the links down from
high-level to low-level.

| Level | Document | Answers |
| --- | --- | --- |
| 0 | [../README.md](../README.md) | How do I install, run, resume, evaluate? |
| 1 | [overview.md](overview.md) | What does the system do end to end? Model, losses, training loop, run lifecycle on one page. |
| 1 | [directory-map.md](directory-map.md) | What is every directory and file for, and who calls whom? |
| 2 | [code/models.md](code/models.md) | Exact tensor shapes and layers of critic, posterior, prior, decoder, mixer blocks. |
| 2 | [code/agent-and-losses.md](code/agent-and-losses.md) | What one `CRAFTAgent.update` computes; gradient ownership per optimizer and per stage. |
| 2 | [code/data-pipeline.md](code/data-pipeline.md) | Dataset layouts, episode-safe chunk/goal sampling, prefetch threads, RNG commit protocol. |
| 2 | [code/runtime.md](code/runtime.md) | CLI → config → experiment → trainer → evaluator; run directories, checkpoints, RNG engines, logging. |
| 2 | [configs-scripts-tests.md](configs-scripts-tests.md) | What each shipped config, script and test file covers. |
| — | [code-review.md](code-review.md) | Problems found while reading the code, with suggested fixes, ranked. |

Existing design documents (written with the code, still authoritative for intent):

- [baseline.md](baseline.md): method decisions and default hyperparameters for the CVAE POC.
- [interfaces.md](interfaces.md): shape and ownership contracts.
- [project-plan.md](project-plan.md): research questions, hypotheses and planned comparisons.

## Suggested reading paths

- **"I just want to know what this repo is."** overview → directory-map.
- **"I need to change the model."** overview → code/models → code/agent-and-losses → tests in `tests/test_baseline.py`, `tests/test_model_contracts.py`.
- **"Training crashed / resume is weird."** code/runtime → code/data-pipeline → code-review (sections R2, R3, M6).
- **"I want to add an experiment config."** configs-scripts-tests → `src/craft/config.py` (the schema is strict; unknown keys are rejected).

Notation used throughout: `B` batch, `O` observation dim, `A` action dim,
`H` chunk length (5), `W` width (128), `D` contrastive projection dim
(`latent_dim`, 128), `Z` per-action VAE latent dim (`vae_latent_dim`, 16).
