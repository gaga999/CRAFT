# Method decisions

Choices below are POC settings. Each states its source or the measurement that
selected it; anything else is an unvalidated design choice.

## Representation

| Choice | Decision | Source / rationale |
| --- | --- | --- |
| Critic score | `exp(t) * unit(phi) . unit(psi) + b`, learned `t`, `b` | [SigLIP](https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py). Bias absorbs the 1:(B-1) base rate; the sphere gives an angular metric and bounded guidance. |
| Initialization | `t = 10`, `b = -log(B-1)` | Optimal binary-NCE offset for one positive per B-1 negatives. |
| Temperature cap | `t = min(exp(log_scale), max_scale)`, `max_scale = 30` | Measured below: a growing `t` made the critic overconfident. CLIP caps its scale for the same reason (at 100). |
| Weight decay | AdamW, `weight_decay = 0.01` on weight matrices only (`kernel`); none on biases, LayerNorm, positions, `t`, `b`; actor 0 | Train/validation CRL gap below. Unvalidated value; sweep 1e-3 / 1e-2. Decoupled decay acts as lr x wd per step. |
| Loss reduction | sum pairs per anchor, mean over anchors | SigLIP. The reference CRL+AC averages all B^2 pairs, i.e. B times smaller gradients; reference coefficients are not transferable. |
| Goals | geometric future (`discount`) for CRL; uniform future in trajectory for the actor | Reference CRL+AC (`value_geom_sample=True`, `actor_geom_sample=False`). |
| Encoder | token Mixer over `[s, a_1..a_H]`; action tokens -> `Linear(W, Z)` -> parameter-free LayerNorm | Tokens share per-step weights for long chunks. Normalization fixes the latent support (a (Z-2)-sphere of radius sqrt(Z) per token), replacing a KL prior. |
| Head | `MLP([flatten(z); e_s])`, one hidden layer | Nonlinear but shallow so `z` keeps CRL geometry. Flatten is position-aware for fixed H. |
| State path | separate state encoder, also an additive skip into every decoder token | The actor score uses the true state, so the actor cannot fake state content; reconstruction must still use `z`. |
| Reconstruction | MSE of `decoder(s, z~)`, `z~ = normalize(z + sigma * eps)`, gradient into encoder | Keeps action detail that CRL's many-to-one objective may discard. Input noise makes the decoder smooth near the encoder's image, where imperfect actor latents land, replacing the decoder-smoothing effect a VAE's sampling had ([RAE](https://arxiv.org/abs/1903.12436)). Re-normalizing keeps noisy inputs on the latent support. |
| Noisy CRL | the head reads the SAME `z~` as the decoder | Trains the head on a neighbourhood of the encoder's image instead of only on it, where the 1M-step runs showed it extrapolating; also regularizes the critic. Validation CRL metrics use clean `z`. |

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
Trained with noise (1M-step cube run), validation MSE was 0.0104 clean and 0.0165
at sigma 0.2. `0` disables it.

## Evidence from the first 1M-step runs

Shipped settings before this change (no decay, no cap, CRL on clean `z`, latent
guidance), `cube-single-noisy-v0` and `puzzle-3x3-noisy-v0`, one seed each.

| Cube | 25k | 100k | 300k | 1M |
| --- | --- | --- | --- | --- |
| CRL loss train / validation | 1.34 / 1.67 | 1.16 / 1.67 | 0.93 / 1.75 | 0.78 / 2.11 |
| validation top-1 | 0.79 | 0.80 | 0.81 | 0.78 |
| `t` | 35 | 37 | 41 | 47 |

Puzzle validation CRL loss bottomed near 1.98 at 300k and rose to 2.26. At the
final cube checkpoint, rescaling `t` with a re-fitted bias gave validation loss
1.80 at `t=30` (vs 2.11 learned) and 1.97 at `t=20`; train/validation still
differed (0.85 / 1.80 at `t=30`, top-1 0.92 / 0.78). So roughly half the rise is
calibration (the cap) and half is overfitting (decay, noisy CRL).

Latent guidance was exploited: at 1M the cube actor's direct score was 9.67,
its re-encoded (executed) score 5.81 and the dataset score 5.89 (puzzle 7.48 /
6.04 / 5.91). Moving from dataset latents toward the actor's latents raised the
direct score 5.44 -> 9.31 while the re-encoded score stayed 5.41 -> 5.36,
although decoded actions changed (MSE 0.27): the head extrapolates off the
encoder's image. Success (cube 0.20 -> 0.54, puzzle <= 0.22, 10 episodes/task)
is therefore attributable to latent BC, not guidance. Hence `guidance:
reencoded` and noisy CRL; `actor/score_gain` now measures re-encoded minus
dataset score.

## Actor

`z_hat = LayerNorm(MLP(s, g))`, deterministic, raw inputs. Loss
`-score / sg(mean|score|) + bc_weight * mean((z_hat - sg(z))^2)`; the score
normalization is the reference CRL+AC actor's `q_loss`. Score omits `b` (no
gradient) so normalization reflects similarity; `t` then cancels exactly. The
default score is re-encoded: `score(s, E(s, D(s, z_hat)), g)`, costing one more
backbone pass per actor step.

`bc_weight = 1`. With latent guidance (representation from 4k joint steps on
`cube-single-noisy-v0`, then 3k actor-only steps; dataset action variance 0.47):

| bc_weight | latent MSE | action MSE | score gain | exploit gap | cycle MSE |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.49 | 0.76 | 3.34 | 1.78 | 0.19 |
| 10 | 0.17 | 0.35 | 0.31 | 0.11 | 0.04 |
| 100 | 0.17 | 0.34 | 0.06 | 0.01 | 0.04 |

At 1 the actor leaves the encoder's image (its decoded actions are farther from
data than the data variance) while its direct score rises; 10 is the smallest
tested weight with a small exploit gap under that guidance.

With re-encoded guidance (representation of the 1M cube run, 3k actor-only steps;
`score_gain` = executed-chunk score minus dataset score):

| guidance, bc_weight | score gain | exploit gap | action MSE | cycle MSE |
| --- | --- | --- | --- | --- |
| latent, 10 | 0.02 | 1.97 | 0.34 | 0.20 |
| reencoded, 10 | 0.03 | 0.51 | 0.34 | 0.18 |
| reencoded, 1 | 0.62 | 0.42 | 0.38 | 0.18 |
| reencoded, 0.1 | 1.71 | 0.41 | 0.71 | 0.27 |

Re-encoding removes the latent exploit; BC at 10 then leaves no real gain, and
at 0.1 decoded actions leave the data (MSE above the data variance), where the
critic is itself out of distribution. `bc_weight = 1` is the default for all
tasks. Success under these settings is not yet measured.

## Other settings

- `latent_dim`: 8 for 5-D arm actions (6 free dimensions per token), 4 for 2-D
  pointmaze; unvalidated.
- `discount`: 0.95 for manipulation (reference default); 0.99 for pointmaze
  navigation (OGBench convention for long navigation horizons); unvalidated here.
- Size: batch 256, width/embedding 128, no critic ensemble. The reference uses
  batch 1024, 512-wide networks and a two-critic minimum.
- Deterministic actor: in multimodal data it regresses to the mean latent. That
  this mean decodes to useful actions is the hypothesis under test, not a given.

## Critic horizon (puzzle)

With the current settings `cube-single-noisy-v0` reached 0.94 success and
`puzzle-3x3-noisy-v0` about 0.2 (1M steps, one seed). Puzzle is lights-out: a
press toggles a button and its neighbours. The data presses about every 21
steps, and the evaluation tasks need 2 / 5 / 8 / 9 / 7 presses (tasks 1-5);
only task 1 (2 presses) succeeds reliably. Pairwise AUC of the final critic
against goals exactly k steps ahead:

| k (steps) | 5 | 20 | 50 | 100 | 200 |
| --- | --- | --- | --- | --- | --- |
| puzzle | 1.00 | 1.00 | 0.89 | 0.68 | 0.58 |
| cube | 1.00 | 0.99 | 0.97 | 0.88 | 0.75 |

With `discount = 0.95` (mean goal offset about 20 steps) the critic is near
chance at the 100-190 steps the failing tasks need, so guidance cannot help
there. `validation/crl/auc_<k>` tracks this during training.

## Diagnostics to read before conclusions

Success alone is insufficient. The [README metrics table](../README.md#metrics)
lists what each logged metric diagnoses; also compare multiple seeds and compute.
