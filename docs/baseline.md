# Token CRL baseline decisions

## Sources and objective

The local reference is `../action-chunked-contrastive-rl/action-chunked-contrastive-rl-offline/agents/accrl.py`.
This is a new implementation of its CRL/FQL training pattern with a different
architecture, rather than copied model code or a claim of identical performance.

- [Three Steps at a Time: paper](https://arxiv.org/abs/2608.30640) and
  [official implementation](https://github.com/M-Korniak/action-chunked-contrastive-rl-offline).
- [Flow Q-Learning](https://arxiv.org/abs/2502.02538): behavior flow matching plus
  a critic-guided, teacher-regularized one-step actor.
- [MLP-Mixer](https://arxiv.org/abs/2105.01601): token-mixing and channel-mixing
  MLPs with residual connections. Periodic exchange is a CRAFT design choice; the
  original vision results do not establish its effectiveness for control.
- [OGBench loader](https://github.com/seohongpark/ogbench/blob/master/ogbench/utils.py)
  and [evaluation](https://github.com/seohongpark/ogbench/blob/master/impls/utils/evaluation.py).

For each of two critics, `logits[i,j] = phi(s_i,A_i) @ psi(g_j) / sqrt(d)`.
Mean sigmoid binary cross entropy uses diagonal positives and off-diagonal negatives,
matching the reference. No feature L2 normalization or additional temperature is
introduced because this would change that score convention. A nonlinear goal
projection separates actor-facing features from contrastive features; its benefit
here is a hypothesis requiring ablation, not established by results in vision.

The teacher regresses `a-noise` at `(1-t)*noise + t*a`. Euler integration produces
a detached teacher target from the same noise used by the one-step student.
The actor loss is `-mean(min(Q1,Q2))/stopgrad(mean(abs(min(Q1,Q2)))+1e-6)` plus
`alpha * mean((student-teacher)^2)`. Predictions/targets are clipped to action
bounds as in the reference FQL branch. Saturation can reduce gradients; monitor
it during actual experiments. There is no TD target, reward bootstrapping, value
network, or target critic in this contrastive baseline.

## Architectural changes

Critics have separate token backbones and phi/psi heads but share a state/goal
encoder trunk. They are therefore not fully independent ensembles. Each backbone
has a state token, per-action tokens, and learned ordered positions. Each layer
applies a pre-normalized residual GELU channel FFN independently to each token,
sharing its weights across tokens. Every `exchange_every` layers and at the final
layer, a pre-normalized residual token MLP mixes `[B,C,T]` along its last axis,
sharing weights across channels. Its hidden size is `2*T`; channel FFN hidden size
is `4*width`. Different layers have independent weights. This is an all-MLP
backbone without attention; the exchange schedule is configurable. The state output
pools the
sequence into phi; action embeddings remain distinct throughout the backbone.

Teacher and student receive raw state, raw goal, detached encoder(state) and
encoder(goal), noise/action tokens, and time features. A projected condition is
added to every token at the input. Exchange layers mix all token positions because the chunk is generated jointly. Actor gradients propagate through the
critic to actions, while critic parameters and representation conditions are held
fixed for actor optimization. Teacher, actor, and critic have separate Adam states
and global gradient clipping at 10. Joint updates all use the pre-update parameters.

## Sampling, evaluation, and limitations

The official noncompact OGBench loader supplies valid transitions and their final
next states. Only complete chunks wholly inside an episode are sampled; no padding
or mask semantics are needed. Critic future goals use clipped geometric offsets
with `p=1-discount`; actor goals use uniform positive future offsets. Both include
the final next state. This explicitly differs from sampling only stored current
observations in the compact reference; use matched samplers for reproduction studies.
No random-goal mixture, frame stacking, augmentation, reward relabeling, or observation
normalization is added. Goal/state vectors must inhabit the same space. Standard
state-based goal-conditioned OGBench datasets fit this contract; image, discrete,
oracle-representation, and single-task modes are rejected by the adapter.

Evaluation resets each task/episode with a fixed seed, generates fresh noise for
each chunk, executes `execution_length` actions, and discards cached actions at
termination/truncation. It records any success within the episode. Tasks are equally
weighted. Evaluation uses a separate RNG and never advances training randomness.
No rendered video is collected in this initial baseline.

Checkpoint resume restores optimizer and agent RNG, sampler RNG, validation RNG,
and global schedule position. Environment state is unnecessary for offline resume.
Resume requires the original resolved config and unchanged dataset contents;
strict reproducibility also requires the same code, package versions, hardware,
and deterministic kernels. Revision/dirty status and environment versions are
recorded, but dataset hashing and source snapshots are not implemented.

Validation includes synthetic updates and a real pointmaze smoke run; neither
establishes learning performance. Compare original flattened CRL+AC, token-only,
and token-plus-embedding variants under matching budgets and multiple seeds before
making a performance claim. Chunk length, executed prefix, and dataset goal sampling
must be controlled separately.
