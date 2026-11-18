# Method decisions

Choices below are POC settings. Each states its source or the measurement that
selected it; anything else is an unvalidated design choice.

## Representation

| Choice | Decision | Source / rationale |
| --- | --- | --- |
| Critic score | `exp(t) * unit(phi) . unit(psi) + b`, learned `t`, `b` | [SigLIP](https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py). Bias absorbs the 1:(B-1) base rate; the sphere gives an angular metric and bounded guidance. |
| Initialization | `t = 10`, `b = -log(B-1)` | Optimal binary-NCE offset for one positive per B-1 negatives. |
| Loss reduction | sum pairs per anchor, mean over anchors | SigLIP. The reference CRL+AC averages all B^2 pairs, i.e. B times smaller gradients; reference coefficients are not transferable. |
| Goals | geometric future (`discount`) for CRL; uniform future in trajectory for the actor | Reference CRL+AC (`value_geom_sample=True`, `actor_geom_sample=False`). |
| Encoder | token Mixer over `[s, a_1..a_H]`; action tokens -> `Linear(W, Z)` -> parameter-free LayerNorm | Tokens share per-step weights for long chunks. Normalization fixes the latent support (a (Z-2)-sphere of radius sqrt(Z) per token), replacing a KL prior. |
| Head | `MLP([flatten(z); e_s])`, one hidden layer | Nonlinear but shallow so `z` keeps CRL geometry. Flatten is position-aware for fixed H. |
| State path | separate state encoder, also an additive skip into every decoder token | The actor score uses the true state, so the actor cannot fake state content; reconstruction must still use `z`. |
| Reconstruction | MSE of `decoder(s, normalize(z + sigma * eps))`, gradient into encoder; CRL uses clean `z` | Keeps action detail that CRL's many-to-one objective may discard. Input noise makes the decoder smooth near the encoder's image, where imperfect actor latents land, replacing the decoder-smoothing effect a VAE's sampling had ([RAE](https://arxiv.org/abs/1903.12436)). Re-normalizing keeps noisy inputs on the latent support. |

`reconstruction_weight = 1.0`: on `cube-single-noisy-v0` (joint, batch 256),
validation CRL/reconstruction gradient norms on the shared encoder were 1.6-5.2
at 1k-4k steps, and reconstruction MSE stayed near 0.01 versus 0.82-0.92 with
shuffled latents. Long-run balance is not yet measured.

`latent_noise = 0.2`: actor latents in the `bc_weight` test below had cycle MSE
0.04 per dimension, i.e. an off-image error of RMS about 0.2. Larger noise makes
the decoder average neighbouring chunks (it learns `E[A | noisy z]`) and pulls
actor latents back toward data, weakening guidance. A decoder trained WITHOUT
noise (same 4k-step checkpoint) is sharply sensitive: validation MSE 0.0097 clean,
0.048 / 0.149 / 0.371 at sigma 0.1 / 0.2 / 0.4, 0.85 with shuffled latents.
Training with noise is not yet measured;
watch `rec/noisy_latent_mse` against `rec/mse`, `rec/shuffled_latent_mse`,
`actor/score_gain` and success. `0` disables it.

## Actor

`z_hat = LayerNorm(MLP(s, g))`, deterministic, raw inputs. Loss
`-score / sg(mean|score|) + bc_weight * mean((z_hat - sg(z))^2)`; the score
normalization is the reference CRL+AC actor's `q_loss`. Score omits `b` (no
gradient) so normalization reflects similarity; `t` then cancels exactly.

`bc_weight = 10`: representation from 4k joint steps on `cube-single-noisy-v0`,
then 3k actor-only steps (dataset action variance 0.47):

| bc_weight | latent MSE | action MSE | score gain | exploit gap | cycle MSE |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.49 | 0.76 | 3.34 | 1.78 | 0.19 |
| 10 | 0.17 | 0.35 | 0.31 | 0.11 | 0.04 |
| 100 | 0.17 | 0.34 | 0.06 | 0.01 | 0.04 |

At 1 the actor leaves the encoder's image (its decoded actions are farther from
data than the data variance) while its direct score rises; 10 is the smallest
tested weight with a small exploit gap. The same default is used for all
tasks without per-task tuning.

## Other settings

- `latent_dim`: 8 for 5-D arm actions (6 free dimensions per token), 4 for 2-D
  pointmaze; unvalidated.
- `discount`: 0.95 for manipulation (reference default); 0.99 for pointmaze
  navigation (OGBench convention for long navigation horizons); unvalidated here.
- Size: batch 256, width/embedding 128, no critic ensemble. The reference uses
  batch 1024, 512-wide networks and a two-critic minimum.
- Deterministic actor: in multimodal data it regresses to the mean latent. That
  this mean decodes to useful actions is the hypothesis under test, not a given.

## Diagnostics to read before conclusions

Success alone is insufficient. Check `validation/rec/shuffled_latent_mse` (decoder
uses `z`), `latent/action_to_state_effect` (critic uses actions),
`actor/score_exploit_gap` and `actor/cycle_mse` (actor stays on the encoder's
image), `grad/*_ratio` (objective balance), multiple seeds and compute.
