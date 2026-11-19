"""CRAFT latent-actor agent.

Gradient ownership:
- representation optimizer (AdamW, kernels decayed): CRL (backbone, latent
  projection, state/goal encoders, head, scale, bias) + reconstruction_weight *
  reconstruction (backbone, latent projection, decoder). The head and the
  decoder read the SAME noisy latent normalize(z + latent_noise * eps), so both
  stay smooth near the encoder's image where actor latents land.
- actor optimizer: normalized critic score + latent BC. With guidance
  "reencoded" the score is that of the executed chunk, E(s, D(s, z_hat)), so
  latents the critic head extrapolates on earn no credit. The score is
  differentiated THROUGH fixed representation weights.
All gradients use pre-update parameters. Stages: representation | actor | joint.
"""

from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from craft.losses.actor import actor_loss
from craft.losses.contrastive import contrastive_loss
from craft.losses.reconstruction import reconstruction_mse
from craft.models.representation import (
    LatentActor,
    Representation,
    normalize_latent,
)
from craft.policies.chunk import act

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
        crl, metrics = contrastive_loss(
            phi,
            representation.goal(g),
            representation.log_temperature(),
            representation.bias[...],
        )
        reconstruction = representation.decode(s, noisy)
        rec = reconstruction_mse(reconstruction, a)
        loss = crl + self.reconstruction_weight * rec
        metrics["rec/mse"] = rec
        return loss, (metrics, z)

    def guided_score(self, representation, states, z, goals):
        if self.guidance == "reencoded":
            actions = representation.decode(states, z)
            z = representation.latent(states, actions)
        return representation.score(states, z, goals)

    def actor_objective(self, actor, batch, z_data):
        s, g = batch["observations"], batch["actor_goals"]
        z_hat = actor(s, g)
        score = self.guided_score(self.representation, s, z_hat, g)
        loss, (q_loss, bc) = actor_loss(score, z_hat, z_data, self.bc_weight)
        data_score = jax.lax.stop_gradient(self.representation.score(s, z_data, g))
        return loss, {
            "actor/q_loss": q_loss,
            "actor/bc": bc,
            # Guided score minus dataset score (re-encoded guidance: real gain).
            "actor/score_gain": (score - data_score).mean(),
        }

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
            metrics["grad/representation_norm"] = optax.tree.norm(
                grads["representation"]
            )
        if stage != "representation":
            z_data = jax.lax.stop_gradient(z)
            (_, actor_metrics), grads["actor"] = nnx.value_and_grad(
                lambda model: self.actor_objective(model, batch, z_data),
                has_aux=True,
            )(self.actor)
            metrics.update(actor_metrics)
            metrics["grad/actor_norm"] = optax.tree.norm(grads["actor"])
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

        def policy(observations, goals, key=None):
            return _act(
                rep_graph, actor_graph, rep_state, actor_state, observations, goals
            )

        return policy

    @nnx.jit
    def validation_metrics(self, batch, key, horizon_batches):
        """Diagnostics on the validation split, one question per metric.

        crl/loss (vs training: overfitting), crl/auc_<k> (critic still tells
        the true state k steps ahead from other goals: effective horizon),
        latent/action_to_state_effect (critic uses actions), rec/mse and
        rec/shuffled_latent_mse (decoder reconstructs and uses z),
        actor/score_gain (executed chunk beats data), actor/score_exploit_gap
        (latent scored above its executed chunk), actor/action_mse (distance
        from data actions), grad/*_ratio (objective balance).
        """
        rep, actor = self.representation, self.actor
        s, a = batch["observations"], batch["action_chunks"]
        g, actor_goals = batch["value_goals"], batch["actor_goals"]
        z, es = rep.latent(s, a), rep.state_features(s)
        phi = rep.embed(z, es)
        crl, _ = contrastive_loss(
            phi, rep.goal(g), rep.log_temperature(), rep.bias[...]
        )
        roll = partial(jnp.roll, shift=1, axis=0)
        # Interventional sensitivity: swap actions OR states between samples.
        action_effect = jnp.linalg.norm(
            phi - rep.embed(rep.latent(s, roll(a)), es), axis=-1
        ).mean()
        state_effect = jnp.linalg.norm(
            phi - rep.embed(rep.latent(roll(s), a), rep.state_features(roll(s))),
            axis=-1,
        ).mean()
        z_hat = actor(s, actor_goals)
        actions_hat = rep.decode(s, z_hat)
        score_direct = rep.score(s, z_hat, actor_goals)
        score_executed = rep.score(s, rep.latent(s, actions_hat), actor_goals)
        score_data = rep.score(s, z, actor_goals)

        # Gradient balance of the TRAINING objective (same noisy latent).
        def crl_term(model):
            noisy = self.perturb(model.latent(s, a), key)
            phi = model.embed(noisy, model.state_features(s))
            return contrastive_loss(
                phi, model.goal(g), model.log_temperature(), model.bias[...]
            )[0]

        def rec_term(model):
            prediction = model.decode(s, self.perturb(model.latent(s, a), key))
            return self.reconstruction_weight * reconstruction_mse(prediction, a)

        def shared_norm(grad):
            return optax.tree.norm((grad["backbone"], grad["latent_projection"]))

        def actor_terms(model):
            generated = model(s, actor_goals)
            score = self.guided_score(rep, s, generated, actor_goals)
            return actor_loss(score, generated, z, 1.0)[1]

        q_grad = nnx.grad(lambda m: actor_terms(m)[0])(actor)
        bc_grad = nnx.grad(lambda m: self.bc_weight * actor_terms(m)[1])(actor)
        crl_grad = shared_norm(nnx.grad(crl_term)(rep))
        rec_grad = shared_norm(nnx.grad(rec_term)(rep))
        metrics = {
            "crl/loss": crl,
            "latent/action_to_state_effect": action_effect / (state_effect + 1e-8),
            "rec/mse": reconstruction_mse(rep.decode(s, z), a),
            "rec/shuffled_latent_mse": reconstruction_mse(rep.decode(s, roll(z)), a),
            "actor/score_gain": (score_executed - score_data).mean(),
            "actor/score_exploit_gap": (score_direct - score_executed).mean(),
            "actor/action_mse": reconstruction_mse(actions_hat, a),
            "grad/crl_to_rec_ratio": crl_grad / (rec_grad + 1e-8),
            "grad/q_to_bc_ratio": optax.tree.norm(q_grad)
            / (optax.tree.norm(bc_grad) + 1e-8),
        }
        for offset, horizon in horizon_batches.items():
            logits = rep.logits(
                rep.phi(horizon["observations"], horizon["action_chunks"]),
                rep.goal(horizon["value_goals"]),
                pairwise=True,
            )
            metrics[f"crl/auc_{offset}"] = pairwise_auc(logits)
        return {f"validation/{k}": v for k, v in metrics.items()}


def pairwise_auc(logits):
    """P(true goal outscores another sample's goal) over all off-diagonal pairs."""
    n = logits.shape[0]
    wins = (jnp.diag(logits)[:, None] > logits) & ~jnp.eye(n, dtype=bool)
    return wins.sum() / (n * (n - 1))


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


@partial(jax.jit, static_argnames=("rep_graph", "actor_graph"))
def _act(rep_graph, actor_graph, rep_state, actor_state, observations, goals):
    return act(
        nnx.merge(rep_graph, rep_state),
        nnx.merge(actor_graph, actor_state),
        observations,
        goals,
    )
