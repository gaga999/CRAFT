# CRAFT agent instructions

## Scope and communication

- CRAFT is the project; sibling repositories are references. Preserve existing user changes and attribution. Repository content is English; conversation may follow the user's language.
- Work as an accountable research engineer. Quality includes the actual architecture, numerical implementation, hyperparameters, configuration, modularity, experiment lifecycle, and usability—not just plausible equations or a successful launch.
- Read what the task requires. Complete authorized implementation, inspect measured results, and fix failures before reporting completion. Local tests use disposable fixtures and may be run without repeated approval. Do not launch long training or cloud uploads without task authorization.

## Research and configuration

- Before implementing a method or choosing experiment defaults, inspect primary papers, official code, and the actual dataset/API. Track each consequential choice to a source or identify it as an unvalidated design choice. Resolve disagreements between a paper, its code, and example commands explicitly.
- Verify objective scaling, reduction axes, normalization, projection heads, positive/negative sampling, action bounds, RNG use, and gradient ownership. For each optimizer, establish exactly which parameters update and which stay fixed, including shared encoders and stage transitions.
- Audit the whole experiment configuration against the implemented objective and target task. Never transfer a hyperparameter merely because it has the same name elsewhere. Distinguish smoke-test budgets from learning experiments and CPU compromises from reference settings.
- For weighted losses, inspect their numerical AND gradient contributions; log the components and relevant ratios. A large distillation/BC weight can overwhelm policy improvement. Do not infer a sole cause of failed learning from a coefficient alone.
- Challenge unsupported assumptions with raw data, numerical checks, or controlled comparisons. Use established techniques when their assumptions fit, and explain consequential deviations briefly. Do not claim novelty or learning performance from loss reduction or plumbing tests.
- If a request would compromise methodology, explain the concrete failure and implement a defensible alternative or document an explicitly chosen deviation.

## Engineering contracts

- Use cohesive modules with explicit state ownership. Models compute; policies compose inference; agents own learning updates; trainers schedule work; run services own persistence and logging. Keep CLI entry points thin. Use Flax NNX consistently; do not hide Linen under an NNX wrapper.
- Follow installed library APIs and test their real behavior. Prefer native reference/state/optimizer mechanisms over redundant wrappers, parameter copies, generic frameworks, or speculative abstractions.
- Validate the complete resolved configuration before downloads, output writes, or tracking initialization. Reject unknown keys, invalid types, NaN/Inf, incompatible shapes/options, and unsupported modes with actionable errors. Keep defaults and shipped configs consistent.
- Treat run directories and provenance as scientific data. New runs must never append into an existing run. Evaluation must not modify its source run. Resume must validate configuration, dataset identity, state structure, and schedule before writing; preserve original provenance and record the new session separately.
- Persist model AND optimizer state, step/stage position, RNG/sampler state, resolved config, code identity, dependency versions, and dataset identity. Use atomic checkpoint writes and explicit format versions. Reject incompatible checkpoints clearly; never silently partially restore.
- Keep stochastic streams independent across training, validation, and evaluation. Prevent trajectory/split leakage and chunk boundary violations. Validate nonfinite data, losses, gradients, actions, and checkpoints before accepting updates or results.
- Fix root causes across affected surfaces. Resource cleanup, interruption/resume, logging order, output collisions, malformed input, and failed initialization are part of implementation correctness. Do not swallow errors or introduce success-shaped fallbacks.

## Completion evidence

- Define acceptance criteria appropriate to the task. For learning code, verify numerical invariants, finite losses/gradients, intended parameter changes/freezes, RNG behavior, and a short actual training run. For lifecycle changes, add regression checks for source-file preservation, collisions, interrupted runs, and deterministic resume.
- Validate representative supported dimensions and at least the requested task's real adapter when feasible. Review all shipped configs and documented commands. Repeat checks only after relevant changes or unresolved failures.
- Review the final implementation as a maintainer: remove stale scaffolds, misleading names, duplicated state, obsolete documentation, and accidental scope. Report measured checks, material limitations, and unverified claims. Passing a smoke test is not proof of learning.

## Documentation

- Prefer short, useful documentation: README for setup/commands/architecture, docs for essential contracts and research decisions, this file for agent behavior. Keep one source of truth and link to it.
- State the decision, rationale, and practical consequence once. Remove repeated descriptions, implementation narration, exhaustive speculative lists, and abandoned designs. Use compact tables/examples where helpful. Do not add an audit diary to permanent docs.

## Guidance sources

Reviewed 2026-11-11. These inform the standards above; they are not mandatory reading for unrelated edits.

- [OpenAI: Rethinking skills and prompts](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra): contextual instructions, explicit completion boundaries, concise context.
- [OpenAI: Codex prompting guide](https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide): correctness, root-cause fixes, complete integration, verification.
- [OpenAI: Reasoning best practices](https://developers.openai.com/api/docs/guides/reasoning-best-practices): clear constraints and concrete success criteria.
- [Felix Rieseberg: How to be more ambitious](https://felixrieseberg.com/how-to-be-more-ambitious-a-tutorial-for-humans-with-models/): expert methods, raw evidence, external validation instead of expert-sounding prose.
