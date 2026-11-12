# POC contracts

- Data: observations `[B,O]`, action_chunks `[B,5,A]`, value_goals `[B,O]`.
  CVAE and CRL use the same value goal. Sampling is episode-safe and float32.
  OGBench storage retains terminal observation slots, excludes their dummy actions,
  and never materializes a full next-observation array. Worker threads only gather;
  one private RNG plans samples in order. The live checkpointed RNG advances only
  after an accepted update, independent of prefetch depth or worker count.
  Worker failures propagate, and context cleanup joins workers/releases buffers.
  Dataset arrays must not be mutated while a prefetch iterator is active.
- Critic: phi features `[B,6,W]`, psi features `[B,W]`; normalized projections
  `[B,D]`. Independent phi/psi weights; only two scalar calibration parameters.
- Posterior/prior: `(mean, log_std)` each `[B,5,Z]`; reparameterized sampling.
- Decoder: `(Z, psi(s), psi(g), mask[B,1,1]) -> bounded actions[B,5,A]`.
  Gate is applied after condition projection, including biases. No raw condition
  tokens bypass dropout. Posterior features may receive AE gradients by config.
- Agent update mutates NNX state; prior-only guidance must not update decoder or
  critic weights. No hard clip or stop-gradient may break the guide input path.
- Inference composes prior, sample, decoder; no posterior or dataset action needed.
- RNG engines, dataset fingerprints, checkpoint atomicity and independent evaluation
  follow the README. New runs add a unique suffix; resume uses the saved actual path.
  Checkpoint v5 rejects older architecture/state layouts.
