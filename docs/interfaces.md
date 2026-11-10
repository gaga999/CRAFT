# Contracts

Interfaces are documented roles, without mandatory base classes. Components use
Flax NNX with eager initialization and explicit dimensions.

| Component | Contract |
| --- | --- |
| Data source | `sample(B)` returns float32 `observations [B,O]`, `action_chunks [B,H,A]`, `value_goals [B,O]`, `actor_goals [B,O]`; `B>=2`. Owns its NumPy RNG. |
| Token backbone | `[B,O]`, `[B,H,A]`, optional condition → `[B,H+1,width]`. Shared channel FFNs; periodic token MLPs; ordered positions. |
| Critic | `(state, chunk, goal) → phi, psi`, each `[2,B,D]`. `encode(state) → [B,width]` precedes contrastive projection. |
| Chunk policy | State/goal, detached features, noise/action tokens, time → `[B,H,A]`; teacher predicts velocity, student predicts actions. |
| Agent | `update(batch, stage=...) → metrics`, mutating NNX state. Separate optimizers; rejected nonfinite updates leave state/RNG/step unchanged. |
| Inference | `sample_action_chunk(state, goal, key)` accepts `[O]` or `[B,O]`, returns `[H,A]` or `[B,H,A]`. Fresh key per call. |
| Trainer | Schedules stages from `agent.step`, samples batches, invokes callbacks; rolls back sampling RNG if an update is rejected. |
| Evaluation | Owns per-task RNG and per-episode action cache, obeys termination/truncation, validates finite bounded actions. |
| Run services | Exclusive session directories, immutable original provenance, versioned atomic checkpoints, no metrics append/reuse. |

`representation` updates critic only; `action` updates teacher/student only; `joint`
updates all from pre-update parameters. Optimizer states persist between stages.
NNX references remain live after updates, so bound inference does not copy parameters.
Resume validates semantic dataset hashes, config, code identity, and state structure.
Add components within their role, document shapes/state/RNG, then verify gradient
ownership, numerical behavior, and the consuming workflow.
