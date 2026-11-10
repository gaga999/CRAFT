# Component contracts

Interfaces are role conventions with executable checks, not mandatory base classes.
The current models use Flax Linen and functional PyTree training state.

- `OfflineDataset.sample(B)` returns float32 `observations [B,O]`,
  `action_chunks [B,H,A]`, `value_goals [B,O]`, and `actor_goals [B,O]`.
  `B >= 2`; chunks are complete and episode-safe. Sampling RNG is source-owned.
- `TokenBackbone(state, actions, condition)` returns `[B,H+1,width]`. The first
  token is state and subsequent tokens are ordered actions. Each layer shares one
  channel FFN across tokens; exchange layers use a token MLP shared across channels.
  `exchange_every >= 1` controls exchange frequency, with a final exchange always
  included. No attention, dropout, or mutable batch statistics are used.
- `TokenCritic` returns `phi, psi`, each `[2,B,latent_dim]`. `encode(states)` returns
  pre-projection features `[B,width]`; actor conditions use these with stop-gradient.
- `TokenActor(state, goal, embeddings, noisy_actions, time)` returns `[B,H,A]`.
  Teacher output is velocity; student output is directly clipped action proposals.
- `CRAFTAgent.update(batch, stage=...) -> (new_agent, metrics)` owns gradient paths
  and optimizer steps. Stages are static JIT arguments. `representation` updates
  critic only; `action` updates teacher/student only; `joint` updates all three.
- `agent.sample_action_chunk(observations, goals, rng)` supports `[O]` or `[B,O]`
  and returns `[H,A]` or `[B,H,A]`. Caller supplies a fresh RNG key. Bound policy
  facades must be recreated after agent updates to avoid stale parameters.
- `train_offline` controls stage order and calls `on_step(global_step, stage,
  current_agent, metrics)`. `start_step` skips already-completed steps. The CLI
  assembles components and schedules logging/evaluation/checkpoint callbacks.
- Evaluators consume an inference callable and own per-episode chunk caching.
  The offline trainer never needs model internals or loss formulas.
- Checkpoints use Flax msgpack plus JSON run metadata. Restoration requires a
  matching initialized agent template (including architecture and optimizer).

To add a model, document its shapes/state/RNG behavior, implement it in `models/`,
and expose construction through a factory or agent assembly. Verify shapes, token
interaction, finite gradients, and downstream parameter updates. To add an agent,
implement the update/inference contracts and add a runnable smoke configuration.
See [baseline decisions](baseline.md) for objectives and [project plan](project-plan.md)
for research scope. Online support needs collection/replay insertion in addition
to the existing sampling contract.
