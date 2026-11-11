# Baseline decisions

## Method

[CRL+AC](https://github.com/M-Korniak/action-chunked-contrastive-rl-offline)
provides the binary-NCE/FQL reference. The new architecture is an experiment, not
a reproduction claim. Per critic, logits are `phi(s_i,A_i)·psi(g_j)/sqrt(D)`;
diagonal pairs are positive and all others negative, using mean sigmoid BCE.
A shared goal encoder supplies actor features before nonlinear contrastive heads.

[MLP-Mixer](https://arxiv.org/abs/2105.01601)-style residual channel FFNs share weights
across tokens. Every `exchange_every` layers, and at the end, token MLPs exchange
information across positions. Layer parameters are independent; FFN hidden size
is `4*width`, token-MLP hidden size is `2*(H+1)`. State and actions stay tokenized.
Critic interaction trunks are separate; their goal encoder is shared.

The [FQL](https://arxiv.org/abs/2502.02538) teacher regresses `a-noise` along a linear
noise/action interpolation. Euler integration supplies a detached teacher target.
The student minimizes `-mean(Q)/stopgrad(mean(abs(Q))+1e-6)` plus
`distillation_weight * MSE(raw_student, teacher)`, with `Q=min(Q1,Q2)`. Actor conditions
use raw state/goal and detached critic features. Critic parameters are fixed during
actor differentiation, but gradients pass through critic inputs to actions.
Distillation uses raw student outputs so out-of-bounds values retain a recovery
gradient; only the Q input is clipped. Teacher targets are clipped after integration
and detached, while velocities remain unbounded. Rollout sampling clips final
actions. This follows [official FQL](https://github.com/seohongpark/fql/blob/master/agents/fql.py)
and corrects the local CRL+AC branch's early clip before distillation.
There is no TD target critic or reward bootstrap in this objective.

## Configuration rationale

| Setting | Starting choice and source/limitation |
| --- | --- |
| Learning rate / flow steps | `3e-4` / `10`, reference `agents/accrl.py:get_config`. |
| Distillation weight | `0.1`, reference code default; replaces the unsupported universal `alpha=10`. Reference README grids also use `1,3,10`: these are experiment-specific alternatives, not evidence that any value is optimal here. |
| Discount | `0.95`, reference default; controls geometric future-goal offsets, not a TD backup. |
| Cube/Puzzle chunk / execution | `3` / `1`. Chunk 3 follows the reference manipulation grid; one-step execution isolates extraction and differs from its full-chunk grid protocol. |
| CPU capacity / batch | Width/latent 128, depth 4, batch 256 are CPU compromises; reference networks are substantially larger with batch 1024. No performance equivalence is claimed. |
| Optimizer | Separate Adam states, global norm clipping at 10 (a configurable stability choice). |

Log weighted losses, Q/distillation gradient norms and their ratio, saturation,
representation norms/variation, and held-out discrimination. Gradient diagnostics
add backward-pass cost; coefficients must be tuned using learning/evaluation data,
not loss values alone. `cube_single_noisy.yaml` and `cpu_puzzle.yaml` are starting experiments; `smoke.yaml` only
tests plumbing. Full learning performance and the sole cause of earlier failed
cube training remain unverified.

## Data and experiments

The [official OGBench loader](https://github.com/seohongpark/ogbench/blob/master/ogbench/utils.py)
provides separate train/validation splits. Regular transitions allow complete
within-episode chunks and their final next state. Critic future offsets are clipped
geometric; actor offsets are uniform positive futures. Including the final next
state differs from the compact reference sampler; match samplers for reproduction.
No observation normalization, frame stacking, augmentation, or random-goal mixture
is silently added. Only standard vector-state/continuous-action datasets are supported.

`make_random_engines(seed)` creates training/evaluation engines at entry and seeds
legacy Python/NumPy APIs once. Explicit initialization/update/data/validation streams
are independent; evaluation derives fixed per-task keys and per-episode environment
seeds, so task order or evaluation cadence cannot advance training randomness.
NumPy Generator states and typed JAX keys/counters are saved and restored.
Task success is any success during an episode, averaged equally across selected tasks. A fresh session
stores each evaluation/resume, with code/dependency snapshots and dataset hashes.
Checkpoint format v3 requires the centralized RNG layout; v2 is rejected. Exact
resume is checked at accepted-update boundaries; diagnostics interrupted mid-callback can be absent from that session.
Multiple seeds, matched sampling/capacity/budgets, and learning curves are required
before concluding that tokens, embeddings, or a particular coefficient help.
