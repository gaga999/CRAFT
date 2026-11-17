# CVAE POC decisions

## Models

Phi processes state + 5 ordered action tokens with shared FFNs and periodic token
MLPs, then mean-pools ALL 6 tokens before its projection. Token exchange uses two
kernel-size-1 convolutions with tokens as channels, preserving `[B,T,W]` without
explicit transposes; channel FFNs remain dense. GPU layout/performance is unverified.
Psi is a separate MLP shared across goals (also applied to state for compatible
observation/goal spaces).
Only projected contrastive features are L2-normalized. Posterior concatenates all
pre-projection phi tokens with the psi goal feature, applies a Mixer, and produces
5 diagonal Gaussian latents. Prior uses raw state/goal and detached psi features
and produces the same latent shape. Decoder receives sampled latents, learned
positions, and an additive condition `exp(log_gate) * P(Ps(psi(s))+Pg(psi(g)))`.
The gate is initialized to 0.1; only its logarithm is learned.

Decoder condition dropout uses one 0/1 mask per sample, shared over positions,
after all projections/biases. No inverted-dropout scaling or condition bypass is
present. Guidance/inference use mask 1. This encourages, but does not guarantee,
latent usage; compare shuffled/zero latents and condition-on/off reconstruction.

`cvae_representation` (default `critic`) selects the CVAE feature source. `own`
removes every critic feature from the CVAE: posterior gets its own phi
`TokenBackbone` and psi-style `MLP+LayerNorm` goal encoder, decoder gets one psi-style
encoder shared by state and goal, and prior drops the detached psi inputs. Copying
the replaced architectures keeps each CVAE function class identical, so the ablation
isolates the representation's training signal (CRL vs. reconstruction/KL). The own
encoders add trainable CVAE parameters (posterior +555k, decoder +20k at width 128);
prior loses `2*W*W` input weights. The critic is still trained by CRL and supplies
the guidance score. `own` is incompatible with `cvae_updates_crl=true`. Checkpoints
that predate the option are read as `critic`.

## Objectives and gradient ownership

Following [SigLIP](https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py),
`logits = exp(log_scale) * phi @ psi.T + bias`; diagonal positives, all off-diagonal
negatives. Loss sums pairs per anchor then averages anchors. Initial scale is 10,
bias is `-log(B-1)` once at initialization. Negative count remains batch dependent.
The mean-pooling/token architecture and ratio-based bias initialization are POC
choices, not a SigLIP reproduction or a calibrated reachability guarantee.

The [CVAE](https://papers.nips.cc/paper_files/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html)
reconstructs ground-truth chunks from posterior samples. Reconstruction MSE averages
batch/time/action; KL(q||p) sums time/latent and averages batch. Posterior/decoder
minimize reconstruction + beta KL with prior targets fixed. Prior minimizes the
same KL with posterior fixed plus `-guide_weight * mean(logit(s,D(Zp),g))`.
Guide and CRL share `logit = exp(log_scale) * dot(phi,psi) + bias`, without an
extra sigmoid. Scale multiplies the prior guidance gradient; the fixed critic bias
shifts the loss but has no prior gradient. Guide weight remains untuned.
These are the separate parameter derivatives of one KL, not double its weight.
Guide gradients update ONLY prior weights; fixed decoder/critic still transmit
input gradients. By default (`guide_through_std=false`) the guide sample uses a
detached prior log std, so guidance moves only the prior mean and KL alone sets the
std. With gradients through the std, a cube-single-noisy run drove prior std to
`exp(log_std_max)` within ~20k steps; the posterior followed, its saturated std
stopped learning, and validation reconstruction rose 0.10 -> 0.20 as the decoder
stopped using latents. Guidance still widens the prior indirectly: KL's optimal
prior variance is posterior variance plus squared mean gap. Monitor `prior/std`.
Checkpoints that predate the option are read as `guide_through_std=true`. Conditions for prior are always detached. `cvae_updates_crl`
controls only AE-to-backbone gradients. All groups use pre-update parameters.

Default beta=0.01, guide_weight=0.1, latent=16 PER action, width/projection=128,
encoder/decoder depth=4, posterior depth=2, exchange interval=2. Gaussian log std
is smoothly bounded to [-4,1], initialized near 0. Decoder uses differentiable
tanh: no hard clip; severe tanh saturation can still reduce gradients. Prior LR is
1e-4; other groups use 3e-4, Adam, global gradient norm cap 10. These are untuned
CPU-sized starting settings. No full learning-performance claim is made.

## Experiment contract

Use the same geometric future goal (discount 0.95) for CRL and CVAE; complete chunks
never cross episodes. Official OGBench train/validation splits remain separate.
Training/evaluation RNG engines and atomic finite-update rejection are retained.
`representation` trains critic only, `action` trains posterior/decoder/prior with
critic frozen, `joint` enables all. Optimizer states persist across stages.
Report prior-vs-posterior generation, KL, latent sensitivity, standard deviations,
condition gate, score calibration, guidance/KL gradient balance, task success,
compute and multiple-seed variability before drawing conclusions.
