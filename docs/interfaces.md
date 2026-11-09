# Module contracts

These are initial conventions, not mandatory base classes. Keep signatures, shapes,
state ownership, and focused checks aligned when adding an implementation. Research
choices remain in [project-plan.md](project-plan.md).

## Data and models

- A data source exposes `sample(batch_size) -> Batch`. The initial offline source
  owns its sampling RNG. A future replay buffer can share this sampling interface;
  online collection and insertion require additional APIs.
- Proposed state-based batch fields: `observations [B, O]`, `goals [B, G]`,
  `action_chunks [B, H, A]`, and `chunk_mask [B, H]` when chunks are padded.
  Define required fields for each stage before implementing it. Split trajectories
  before forming overlapping chunks and never cross episode boundaries.
- An encoder maps `[..., O] -> [..., Z]`. Interchangeable implementations must
  agree on input structure, output dimension, dtype, and RNG/state behavior.
- `models/mlp.py` uses Flax Linen solely as an executable example. Linen versus
  NNX for the full method is undecided. With Linen, factories return uninitialized
  modules; the agent owns initialized variables and optimizer states.
- `losses/reconstruction.py` is an equal-shape MSE example for fully valid chunks.
  It is not the CRAFT objective and does not handle masked/padded chunks.

## Agent, policy, and stages

- `agent.update(batch, stage=...) -> (new_agent, metrics)` owns losses, optimizer
  steps, frozen/shared parameter decisions, target updates, and training RNG.
  Stage dispatch happens outside JIT. The example dataclass is not a registered
  JAX PyTree; choose the state representation before compiling updates.
- One agent may have multiple stages. Define stage-specific batches, parameter
  sharing, optimizer creation/reset, and freeze behavior in the concrete agent.
- The policy composes models and receives current parameters explicitly through
  `sample_action_chunk(params, observations, goals, rng) -> [B, H, A]`.
  It does not own a second copy of trainable parameters.
- Bind current agent parameters into an inference callable before evaluation.
  A future execution adapter must track each environment's cached chunk, reset it
  on episode boundaries, and replan after `execution_length` actions.
- The trainer chooses stages/data and schedules callbacks; it does not choose loss
  formulas or gradient paths. Its example callback receives a global step, stage,
  current agent, and metrics. Logging, evaluation, and persistence are not wired yet.
- Resume support must restore stage and local/global step, parameters, optimizer
  and target states, agent/sampler RNG, plus configuration and code metadata.

## Adding components

1. Read the relevant role contract and the closest example.
2. Implement the component in its role directory; use a base class only when shared
   behavior requires it. Optional `Protocol` types can be added as interfaces settle.
3. Register selectable model implementations explicitly in `models/factory.py`.
4. Add configuration and verify shapes, finite outputs, RNG/state behavior, and
   compatibility with the consuming policy or agent. Learning updates additionally
   require gradient/update checks and a training smoke test.

Future online support should add an interaction loop and replay insertion, while
reusing policy inference, compatible batches, and agent updates. It is not implemented.
