"""CRAFT latent-actor agent.

Gradient ownership:
- representation optimizer (AdamW, kernels decayed): CRL (backbone, latent
  projection, state/goal encoders, head, temperature, bias) +
  reconstruction_weight * reconstruction (backbone, latent projection,
  decoder). The head and the decoder read the SAME noisy latent
  normalize(z + latent_noise * eps), so both stay smooth near the encoder's
  image where actor latents land.
- actor optimizer: normalized critic score + (optionally advantage-weighted)
  latent BC. With guidance "reencoded" the score is that of the executed chunk,
  E(s, D(s, z_hat)). The score is differentiated THROUGH fixed representation
  weights; representation features fed to the actor are detached.
All gradients use pre-update parameters. Stages: representation | actor | joint.

Metric names are `<component>/<name>_<split>` with split train or val, so each
W&B section holds one component and train/val curves sit side by side.
"""

from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from craft.losses.actor import actor_loss, advantage_weights
from craft.losses.contrastive import contrastive_loss, pairwise_auc
from craft.losses.reconstruction import reconstruction_mse
from craft.models.representation import (
    LatentActor,
    Representation,
    normalize_latent,
)
from craft.policies.chunk import act, generate

STAGES = ("representation", "actor", "joint")


def all_finite(tree):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree)]))


def kernel_mask(params):
    """Decay weight matrices only: not biases, norms, positions, scale or bias."""

    def is_kernel(path, _):
        names = [k.key for k in path if isinstance(k, jax.tree_util.DictKey)]
        return bool(names) and names[-1] == "kernel"

    return jax.tree_util.tree_map_with_path(is_kernel, params)


def optimizer(module, learning_rate, weight_decay, max_grad_norm):
    return nnx.Optimizer(
        module,
        optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adamw(learning_rate, weight_decay=weight_decay, mask=kernel_mask),
        ),
        wrt=nnx.Param,
    )


class CRAFTAgent(nnx.Module):
    def __init__(self, observation_dim, action_dim, config, *, rngs, batch_size):
        self.representation = Representation(
            observation_dim, action_dim, config, rngs=rngs, batch_size=batch_size
        )
        self.actor = LatentActor(observation_dim, config, rngs=rngs)
        self.representation_optimizer = optimizer(
            self.representation,
            config["learning_rate"],
            config["weight_decay"],
            config["max_grad_norm"],
        )
        self.actor_optimizer = optimizer(
            self.actor,
            config["actor_learning_rate"],
            config["actor_weight_decay"],
            config["max_grad_norm"],
        )
        self.observation_dim, self.action_dim = observation_dim, action_dim
        self.reconstruction_weight = config["reconstruction_weight"]
        self.bc_weight = config["bc_weight"]
        self.latent_noise = config["latent_noise"]
        self.guidance = config["guidance"]
        self.actor_inputs = tuple(config["actor_inputs"])
        self.bc_weighting = config["bc_weighting"]
        self.advantage = (
            config["advantage_beta"],
            config["advantage_max_weight"],
            config["advantage_baseline"],
        )
        # params: initialization; updates: decoder-input noise (checkpointed).
        self.rngs = rngs
        self.step = nnx.Variable(jnp.array(0, jnp.int32))

    @classmethod
    def create(cls, observation_dim, action_dim, config, *, rngs, batch_size=256):
        from craft.config import validate_agent

        validate_agent(config)
        if min(observation_dim, action_dim) > 0 and batch_size >= 2:
            return cls(
                observation_dim, action_dim, config, rngs=rngs, batch_size=batch_size
            )
        raise ValueError("Positive dimensions and batch_size >= 2 required")

    # Objectives -----------------------------------------------------------

    def perturb(self, z, key):
        """Noise on the latent support: re-apply the parameter-free LayerNorm."""
        if self.latent_noise == 0:
            return z
        return normalize_latent(
            z + self.latent_noise * jax.random.normal(key, z.shape, z.dtype)
        )

    def representation_objective(self, representation, batch, key):
        s, a, g = batch["observations"], batch["action_chunks"], batch["value_goals"]
        z = representation.latent(s, a)
        noisy = self.perturb(z, key)  # one sample shared by head and decoder
        phi = representation.embed(noisy, representation.state_features(s))
        crl = contrastive_loss(
            representation.logits(phi, representation.goal(g), pairwise=True)
        )
        rec = reconstruction_mse(representation.decode(s, noisy), a)
        metrics = {"critic/loss_train": crl, "decoder/rec_train": rec}
        if representation.similarity == "cosine":
            metrics["critic/scale_train"] = representation.temperature()
        return crl + self.reconstruction_weight * rec, (metrics, z)

    def guided_score(self, representation, states, z, goals):
        if self.guidance == "reencoded":
            actions = representation.decode(states, z)
            z = representation.latent(states, actions)
        return representation.score(states, z, goals)

    def actor_terms(self, actor, representation, states, goals, z_data):
        """Actor loss on (s, g) against dataset latents; shared by train and val."""
        z_hat = generate(representation, actor, states, goals, self.actor_inputs)
        score = self.guided_score(representation, states, z_hat, goals)
        data_score = jax.lax.stop_gradient(representation.score(states, z_data, goals))
        weights = None
        info = {}
        if self.bc_weighting == "advantage":
            beta, max_weight, baseline_kind = self.advantage
            baseline = (
                jax.lax.stop_gradient(score)
                if baseline_kind == "actor"
                else data_score.mean()
            )
            weights, adv = advantage_weights(data_score, baseline, beta, max_weight)
            info = {
                "advantage": adv.mean(),
                "weight_mean": weights.mean(),
                "weight_clipped": (weights >= max_weight).mean(),
            }
        loss, (q_loss, bc) = actor_loss(score, z_hat, z_data, self.bc_weight, weights)
        info.update(
            {
                "q_loss": q_loss,
                "bc": bc,
                # Guided minus dataset score (re-encoded guidance: real gain).
                "score_gain": (score - data_score).mean(),
            }
        )
        return loss, info

    def actor_objective(self, actor, batch, z_data):
        loss, info = self.actor_terms(
            actor,
            self.representation,
            batch["observations"],
            batch["actor_goals"],
            z_data,
        )
        return loss, {f"actor/{k}_train": v for k, v in info.items()}

    # Updates ----------------------------------------------------------------

    def train_step(self, batch, stage):
        """Trace-time update that mutates this (merged) agent; caller commits."""
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        s, a = batch["observations"], batch["action_chunks"]
        metrics, grads = {}, {}
        if stage == "actor":
            z = self.representation.latent(s, a)
        else:
            key = self.rngs.updates() if self.latent_noise > 0 else None
            (_, (metrics, z)), grads["representation"] = nnx.value_and_grad(
                lambda model: self.representation_objective(model, batch, key),
                has_aux=True,
            )(self.representation)
            metrics["balance/grad_norm_representation_train"] = optax.tree.norm(
                grads["representation"]
            )
        if stage != "representation":
            z_data = jax.lax.stop_gradient(z)
            (_, actor_metrics), grads["actor"] = nnx.value_and_grad(
                lambda model: self.actor_objective(model, batch, z_data),
                has_aux=True,
            )(self.actor)
            metrics.update(actor_metrics)
            metrics["balance/grad_norm_actor_train"] = optax.tree.norm(grads["actor"])
        finite = all_finite((grads, metrics, batch))
        if "representation" in grads:
            self.representation_optimizer.update(
                self.representation, grads["representation"]
            )
        if "actor" in grads:
            self.actor_optimizer.update(self.actor, grads["actor"])
        self.step[...] += 1
        return metrics, finite

    def update(self, batch, *, stage="joint"):
        """One update; state is committed only when every value is finite."""
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        graphdef, state = nnx.split(self)
        state, metrics = _step(graphdef, state, batch, stage)
        if bool(metrics["numerics/finite"]):
            nnx.update(self, state)
        return metrics

    def update_chunk(self, batches, *, stage="joint"):
        """`update` over batches stacked on a leading axis, in one dispatch.

        Returns last-step metrics; a nonfinite step rejects the whole chunk.
        """
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        graphdef, state = nnx.split(self)
        state, metrics = _scan(graphdef, state, batches, stage)
        if bool(metrics["numerics/finite"]):
            nnx.update(self, state)
        return metrics

    # Inference and diagnostics ---------------------------------------------

    def policy(self):
        """Snapshot of current weights as `(observations, goals, key) -> chunk`.

        One `jax.jit` over explicit state avoids per-call NNX graph traversal
        during rollouts. The policy is deterministic; `key` is accepted for
        the evaluator interface. Create a new policy after further updates.
        """
        rep_graph, rep_state = nnx.split(self.representation)
        actor_graph, actor_state = nnx.split(self.actor)

        inputs = self.actor_inputs

        def policy(observations, goals, key=None):
            return _act(
                rep_graph,
                actor_graph,
                inputs,
                rep_state,
                actor_state,
                observations,
                goals,
            )

        return policy

    @nnx.jit
    def validation_metrics(self, batch, key, horizon_batches):
        """Validation-split diagnostics (clean latents), grouped by component.

        critic: generalization (loss), calibration (top-1, mean positive and
          negative logits), horizon (auc_<k>), action use (action vs state
          effect), embedding norms.
        decoder: reconstruction from clean, noisy and batch-shuffled latents.
        latent: per-dimension batch spread (collapse).
        actor: real gain over data, latent exploitation, distance from data
          (actions, latents), cycle consistency, goal sensitivity.
        balance: gradient ratios of each optimizer's objective terms.
        """
        rep, actor = self.representation, self.actor
        s, a = batch["observations"], batch["action_chunks"]
        g, actor_goals = batch["value_goals"], batch["actor_goals"]
        n = s.shape[0]
        roll = partial(jnp.roll, shift=1, axis=0)
        z, es = rep.latent(s, a), rep.state_features(s)
        phi_raw = rep.head_output(z, es)
        psi_raw = rep.goal_encoder(g)
        phi, psi = rep.embed(z, es), rep.goal(g)
        logits = rep.logits(phi, psi, pairwise=True)
        eye = jnp.eye(n)
        # Interventional sensitivity: swap actions OR states between samples.
        action_effect = jnp.linalg.norm(
            phi - rep.embed(rep.latent(s, roll(a)), es), axis=-1
        ).mean()
        state_effect = jnp.linalg.norm(
            phi - rep.embed(rep.latent(roll(s), a), rep.state_features(roll(s))),
            axis=-1,
        ).mean()
        reconstruction = rep.decode(s, z)

        z_hat = generate(rep, actor, s, actor_goals, self.actor_inputs)
        actions_hat = rep.decode(s, z_hat)
        z_cycle = rep.latent(s, actions_hat)
        score_direct = rep.score(s, z_hat, actor_goals)
        score_executed = rep.score(s, z_cycle, actor_goals)
        score_data = rep.score(s, z, actor_goals)
        swapped = rep.decode(
            s, generate(rep, actor, s, roll(actor_goals), self.actor_inputs)
        )
        action_variance = a.reshape(-1, a.shape[-1]).var(0).mean() + 1e-8

        # Gradient balance of the TRAINING objectives (same noisy latent).
        def crl_term(model):
            noisy = self.perturb(model.latent(s, a), key)
            phi = model.embed(noisy, model.state_features(s))
            return contrastive_loss(model.logits(phi, model.goal(g), pairwise=True))

        def rec_term(model):
            prediction = model.decode(s, self.perturb(model.latent(s, a), key))
            return self.reconstruction_weight * reconstruction_mse(prediction, a)

        def shared_norm(grad):
            return optax.tree.norm((grad["backbone"], grad["latent_projection"]))

        def actor_term(model, name):
            _, info = self.actor_terms(model, rep, s, actor_goals, z)
            return info["q_loss"] if name == "q" else self.bc_weight * info["bc"]

        q_grad = nnx.grad(lambda m: actor_term(m, "q"))(actor)
        bc_grad = nnx.grad(lambda m: actor_term(m, "bc"))(actor)
        crl_grad = shared_norm(nnx.grad(crl_term)(rep))
        rec_grad = shared_norm(nnx.grad(rec_term)(rep))
        _, actor_info = self.actor_terms(actor, rep, s, actor_goals, z)

        metrics = {
            "critic/loss": contrastive_loss(logits),
            "critic/top1": (logits.argmax(-1) == jnp.arange(n)).mean(),
            "critic/logit_positive": jnp.diag(logits).mean(),
            "critic/logit_negative": (logits * (1 - eye)).sum() / (n * (n - 1)),
            "critic/action_effect": action_effect,
            "critic/state_effect": state_effect,
            "critic/action_to_state_effect": action_effect / (state_effect + 1e-8),
            "critic/phi_norm": jnp.linalg.norm(phi_raw, axis=-1).mean(),
            "critic/psi_norm": jnp.linalg.norm(psi_raw, axis=-1).mean(),
            "decoder/rec": reconstruction_mse(reconstruction, a),
            "decoder/rec_first": reconstruction_mse(reconstruction[:, 0], a[:, 0]),
            "decoder/rec_noisy": reconstruction_mse(
                rep.decode(s, self.perturb(z, key)), a
            ),
            "decoder/rec_shuffled": reconstruction_mse(rep.decode(s, roll(z)), a),
            "latent/batch_std": z.std(0).mean(),
            "actor/score_data": score_data.mean(),
            "actor/score_gain": (score_executed - score_data).mean(),
            "actor/exploit_gap": (score_direct - score_executed).mean(),
            "actor/action_mse_rel": reconstruction_mse(actions_hat, a)
            / action_variance,
            "actor/latent_mse": reconstruction_mse(z_hat, z),
            "actor/cycle_mse": reconstruction_mse(z_cycle, z_hat),
            "actor/goal_sensitivity": reconstruction_mse(actions_hat, swapped)
            / action_variance,
            "balance/crl_to_rec": crl_grad / (rec_grad + 1e-8),
            "balance/q_to_bc": optax.tree.norm(q_grad)
            / (optax.tree.norm(bc_grad) + 1e-8),
        }
        if self.bc_weighting == "advantage":
            metrics["actor/weight_mean"] = actor_info["weight_mean"]
            metrics["actor/weight_clipped"] = actor_info["weight_clipped"]
        for offset, horizon in horizon_batches.items():
            horizon_logits = rep.logits(
                rep.phi(horizon["observations"], horizon["action_chunks"]),
                rep.goal(horizon["value_goals"]),
                pairwise=True,
            )
            metrics[f"critic/auc_{offset}"] = pairwise_auc(horizon_logits)
        return {f"{k}_val": v for k, v in metrics.items()}


@partial(jax.jit, static_argnames=("graphdef", "stage"))
def _step(graphdef, state, batch, stage):
    agent = nnx.merge(graphdef, state)
    metrics, finite = agent.train_step(batch, stage)
    return nnx.state(agent), {**metrics, "numerics/finite": finite}


@partial(jax.jit, static_argnames=("graphdef", "stage"))
def _scan(graphdef, state, batches, stage):
    # No per-step select: a nonfinite step rejects the whole chunk on the host,
    # so later steps of a rejected chunk never reach committed state.
    def body(state, batch):
        agent = nnx.merge(graphdef, state)
        metrics, finite = agent.train_step(batch, stage)
        return nnx.state(agent), (metrics, finite)

    state, (metrics, finite) = jax.lax.scan(body, state, batches)
    last = jax.tree.map(lambda x: x[-1], metrics)
    return state, {**last, "numerics/finite": jnp.all(finite)}


@partial(jax.jit, static_argnames=("rep_graph", "actor_graph", "inputs"))
def _act(rep_graph, actor_graph, inputs, rep_state, actor_state, observations, goals):
    return act(
        nnx.merge(rep_graph, rep_state),
        nnx.merge(actor_graph, actor_state),
        observations,
        goals,
        inputs,
    )
