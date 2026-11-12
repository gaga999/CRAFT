# Agent update and losses (`agents/craft.py`, `losses/`)

## Losses

### CRL: `contrastive_loss` (`losses/contrastive.py`)
```text
logits[i,j] = exp(log_scale)·<phi_i, psi_j> + bias          [B,B]
labels      = +1 on the diagonal, −1 elsewhere
loss        = mean_i  Σ_j  −log σ(labels[i,j] · logits[i,j])
```
This is SigLIP: sum over the pairs of each anchor, mean over anchors. The
positives are `(s_i, A_i, g_i)`, where `g_i` is the geometric future goal of
sample i. Every other goal in the batch is a negative. Metrics:
`critic/loss`, `critic/scale`, `critic/bias`, `critic/accuracy` (argmax over the
row = diagonal) and `critic/positive` (mean diagonal logit).

### Gaussian KL: `gaussian_kl(q, p)` (`losses/cvae.py`)
Closed form of `KL(N(μq,σq) ‖ N(μp,σp))`, **summed** over `[H, Z]` and averaged
over the batch.

### Reconstruction
Computed inline as `mean((D(z_q) − A)²)` over batch, time and action.
`losses/reconstruction.py` is not used.

## `CRAFTAgent.update(batch, stage)` step by step (`agents/craft.py:98-178`)

`update` is one `nnx.jit` with `stage` static. **All quantities use the parameters
from before the update**, and every optimizer is applied at the end.

1. **Keys.** It saves `rngs.updates.count`, then splits the next `updates` key
   into `qkey` (posterior sample), `pkey` (prior sample) and `mkey` (condition
   dropout mask).
2. **Critic CRL gradient `cg`.** `value_and_grad(contrastive_loss ∘ critic)`.
3. **Prior target.** `pp = sg(prior(s, g, sg(hs), sg(hg)))`.
4. **Mask.** `Bernoulli(1 − condition_drop_prob)` with shape `[B,1,1]`.
5. **CVAE gradients `(acg, pg, dg)`** for (critic, posterior, decoder) from
   `ae_objective`:
   ```text
   tokens, hg, hs = critic.features(s,A), critic.encode(g), critic.encode(s)
   (detached unless cvae_updates_crl)
   q   = posterior(tokens, hg)
   Â   = decoder(sample(q, qkey), hs, hg, mask)
   L_ae = MSE(Â, A) + β · KL(q ‖ pp)
   ```
6. **Prior gradients.** `prior_objective` returns the two terms
   `[β·KL(sg(q) ‖ p), λ·guide]`, where
   ```text
   guide = −mean(critic.score(s, decoder(sample(p, pkey), hs, hg, 1), g))   (0 if guidance disabled)
   ```
   One `nnx.vjp` over the prior is pulled back twice, with cotangents `[1,0]`
   (`kg`) and `[0,1]` (`gg`), so the KL and guidance gradient norms can be
   logged separately. `rg = kg + gg`. Only the prior is differentiated: the
   decoder and critic pass input gradients but their weights get none.
7. **Joint coupling.** In stage `joint` the critic gradient becomes `cg + acg`.
   `acg` is zero unless `cvae_updates_crl`.
8. **Finite barrier.** `finite = all_finite(gradients, metrics, batch)`.
   `nnx.cond(finite, commit, reject)`:
   - `commit` applies optimizer `i` according to the stage table below and
     increments `step`.
   - `reject` restores `rngs.updates.count` and leaves everything else unchanged.
   It returns the metrics plus `numerics/finite`. The trainer turns `False` into a
   `FloatingPointError`.

### Gradient ownership

| Gradient source | critic | posterior | decoder | prior |
| --- | --- | --- | --- | --- |
| CRL loss | ✔ | | | |
| MSE reconstruction | only if `cvae_updates_crl` and stage `joint` | ✔ | ✔ | |
| β·KL(q‖sg p) (posterior side) | only if `cvae_updates_crl` and stage `joint` | ✔ | | |
| β·KL(sg q‖p) (prior side) | | | | ✔ |
| λ·guidance | (input path only) | | (input path only) | ✔ |

### Stage → optimizers applied

| Stage | critic | posterior | decoder | prior |
| --- | --- | --- | --- | --- |
| `joint` | ✔ | ✔ | ✔ | ✔ |
| `representation` | ✔ | | | |
| `action` | | ✔ | ✔ | ✔ |

Every gradient is computed in every stage. Gradients for frozen groups are
discarded, and optimizer state persists across stages.

## Logged training metrics

| Key | Meaning |
| --- | --- |
| `critic/*` | see CRL above |
| `vae/reconstruction`, `vae/kl` | posterior-side MSE and KL(q‖p) |
| `vae/posterior_std`, `prior/std` | mean σ of q and p |
| `prior/kl`, `prior/guidance` | unweighted prior KL and guidance loss (−logit) |
| `prior/weighted_kl`, `prior/weighted_guidance` | β·KL, λ·guide |
| `prior/kl_grad_norm`, `prior/guide_grad_norm`, `prior/guide_to_kl_grad_ratio`, `prior/grad_norm` | gradient balance on the prior (before clipping) |
| `decoder/condition_gate`, `decoder/condition_keep` | exp(log_gate); fraction of samples that kept the condition |
| `time/data_wait_seconds`, `time/update_seconds` | added by the trainer |

## Validation metrics (`validation_metrics`, `agents/craft.py:186-209`)

These are computed on a batch from the official validation split at each
`log_interval`, with a fixed key (`evaluation_engine.task_key(0)`), so the noise
is identical every time. The batch is fresh. Each key is logged with a
`validation/` prefix:

- all `critic/*` CRL metrics
- `vae/reconstruction`: decode the posterior sample (mask = 1)
- `vae/kl`: KL(q‖p)
- `vae/prior_sample_mse`: decode a prior sample and compare it with the dataset
  chunk (a generation-quality proxy)
- `vae/shuffled_latent_mse`: decode posterior latents rolled by one sample. A value
  close to `reconstruction` means the decoder ignores the latent.
