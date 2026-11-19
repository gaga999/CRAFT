# Contracts

- Data: observations `[B,O]`, action_chunks `[B,H,A]`, value_goals `[B,O]`
  (geometric future, CRL), actor_goals `[B,O]` (uniform future, actor). Sampling
  is episode-safe float32; chunks never cross episodes. Worker threads only
  gather; one private RNG plans samples; the checkpointed sampler advances only
  after an accepted update.
- Representation: `latent(s, A) -> z [B,H,Z]` with mean 0 and RMS 1 per token;
  `phi`, `psi` unit `[B,D]`; `decode(s, z) -> [B,H,A]` in (-1, 1) via tanh;
  `score(s, z, g) = t phi(z, e_s(s)) . psi(g)` (no bias), `t = min(exp(log_scale), max_scale)`.
- Actor: `(s, g) -> z_hat`, normalized like `z`; guided by
  `score(s, latent(s, decode(s, z_hat)), g)` (`guidance: reencoded`). Inference is
  `decode(s, actor(s, g))`; no dataset action or randomness is used.
- Ownership: CRL updates encoder, state/goal encoders, head, `t`, `b`;
  reconstruction updates encoder and decoder; actor loss updates only the actor.
  All gradients of one step use pre-update parameters; `joint` equals the
  separate stages' updates (up to XLA rounding).
- Updates are pure functions of split NNX state, committed only if every
  gradient, metric and input is finite. The only training randomness is the
  latent noise (one sample per step, read by the head and the decoder), drawn
  from the checkpointed `updates` stream only in
  stages that train reconstruction; a rejected update rolls it back.
- Checkpoint v6 holds both models, both (AdamW) optimizers, step and both RNG
  streams; resume restores everything and rejects optimizer layouts from older
  code. `init.checkpoint` copies selected module parameters only, so earlier v6
  runs can still initialize new runs.
