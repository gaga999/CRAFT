# System overview

CRAFT (Contrastive Representation for Action Flow and Temporal Abstraction) is a
Flax NNX proof of concept for **offline goal-conditioned RL on OGBench** (state
vectors, continuous actions in `[-1, 1]`). It asks whether the representation
learned by a contrastive RL (CRL) critic can directly help *generate* long action
chunks. The current implementation is a CRL critic plus a conditional VAE (CVAE)
whose latent prior is additionally steered by the critic. It is a POC: tests check
plumbing and numerics, not learning performance.

## 1. The model in one picture

```text
                 dataset (s, A = a_t..a_{t+4}, g = future state)
                                   │
        ┌──────────────────────────┼─────────────────────────────┐
        │ CRITIC (TokenCritic)     │                              │
        │  phi(s,A): token mixer over [s, a1..a5] → mean → MLP → unit  [B,D]
        │  psi(g):   MLP → LayerNorm (=encode, [B,W]) → MLP → unit     [B,D]
        │  logit = exp(log_scale)·<phi,psi> + bias        (SigLIP style)
        └──────────────┬───────────────────────────────┬──────────────┘
          phi tokens [B,6,W], encode(g)          encode(s), encode(g) (detached)
                       │                               │
              POSTERIOR q(Z|s,A,g)              PRIOR p(Z|s,g)
              token mixer → [B,5,Z] Gaussian    MLP → [B,5,Z] Gaussian
                       │                               │
                  Z ~ q (training)               Z ~ p (inference / guidance)
                       └────────────┬──────────────────┘
                                DECODER D(Z, encode(s), encode(g), mask)
                                token mixer over 5 latent tokens → tanh → Â [B,5,A]
```

- **Critic** learns which future goals follow a (state, action chunk).
- **Posterior + decoder** form an autoencoder over action chunks, conditioned on
  critic features (a CVAE).
- **Prior** learns to produce the posterior's latent from (s, g) alone, so at test
  time no dataset action is needed: `prior → sample → decoder` is the policy
  (`src/craft/policies/chunk.py`).
- **Guidance**: the prior also maximizes the critic's logit of its *decoded*
  actions. Gradients flow through the frozen decoder and critic but update
  only the prior.

Details: [code/models.md](code/models.md).

## 2. Losses and who they update

| Optimizer (Adam + global-norm clip 10) | Loss it minimizes | LR |
| --- | --- | --- |
| critic | SigLIP sigmoid loss over the B×B phi/psi logits (+ CVAE loss if `cvae_updates_crl`) | 3e-4 |
| posterior | `MSE(D(z_q), A) + β·KL(q ‖ sg(p))` | 3e-4 |
| decoder | `MSE(D(z_q), A)` | 3e-4 |
| prior | `β·KL(sg(q) ‖ p) − λ·mean(logit(s, D(z_p), g))` | 1e-4 |

Defaults: β = `kl_weight` = 0.01 and λ = `guide_weight` = 0.1. Training stages
(`train.stages`) choose which optimizers are applied: `joint` (all of them),
`representation` (critic only), `action` (everything except the critic). Full
derivation: [code/agent-and-losses.md](code/agent-and-losses.md).

## 3. One training step

```text
PrefetchBatches ──next()──► batch {observations [B,O], action_chunks [B,5,A], value_goals [B,O]}
        │                         │
        │                  CRAFTAgent.update (one nnx.jit)
        │                    1. critic CRL loss + grads
        │                    2. posterior/decoder (and optional critic) CVAE grads
        │                    3. prior KL grads and guidance grads (separate VJPs, summed)
        │                    4. all finite? → apply stage-selected optimizers, step += 1
        │                       otherwise   → roll back the RNG counter, change nothing
        │                         │
        └──accept()◄── trainer raises FloatingPointError if the step was not finite
                                  │
                       on_step: log / validate / evaluate / checkpoint at intervals
```

Data sampling (`src/craft/data/`) guarantees that chunks never cross episode
boundaries. Goals are sampled at a geometric (γ = `data.discount`) offset into the
future of the same episode. The sampler's RNG state advances only when an update is
**accepted**, so resume is deterministic no matter how many batches were
prefetched. See [code/data-pipeline.md](code/data-pipeline.md).

## 4. Run lifecycle

```text
craft --config X.yaml [overrides]          (cli.py)
  └─ load_config: strict schema, type checks, absolute paths       (config.py)
  └─ run_experiment                                                 (experiments.py)
       ├─ new run   → runs/<prefix>-<UTC>-<uuid>/   (never reuses a directory)
       ├─ --restore → <run>/sessions/resume-<UTC>-<uuid>/  (config, data, code identity checked)
       └─ evaluate  → runs/evaluations/<run>-<token>/      (source run untouched)
  each session: config.yaml, metadata.json (provenance), source.zip, dataset.json,
                status.json, metrics.jsonl, checkpoint.msgpack (format v5)
```

See [code/runtime.md](code/runtime.md).

## 5. Where the project is going

The CVAE is the first candidate of a broader research plan: CRL-informed latent
generation, embedding-conditioned actors, subgoal embeddings and auxiliary
objectives. [project-plan.md](project-plan.md) lays out the hypotheses and the
comparisons needed before claiming anything. [baseline.md](baseline.md) records
why the current defaults were chosen.
