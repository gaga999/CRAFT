"""NNX CRL/FQL updates with explicit ownership and atomic finite-update guards."""

from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from craft.losses.contrastive import contrastive_loss, q_values
from craft.models.factory import make_crl_models
from craft.policies.chunk import encode_condition, integrate_flow


def all_finite(tree):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree)]))


class CRAFTAgent(nnx.Module):
    def __init__(self, observation_dim, action_dim, config, *, rngs):
        self.critic, self.teacher, self.actor = make_crl_models(
            config, observation_dim, action_dim, rngs=rngs
        )
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.flow_steps = config["flow_steps"]
        self.distillation_weight = config["distillation_weight"]
        self.chunk_length = config["chunk_length"]
        self.optimizers = nnx.List(
            [
                nnx.Optimizer(
                    model,
                    optax.chain(
                        optax.clip_by_global_norm(config["max_grad_norm"]),
                        optax.adam(config["learning_rate"]),
                    ),
                    wrt=nnx.Param,
                )
                for model in (self.critic, self.teacher, self.actor)
            ]
        )
        self.rngs = rngs
        self.step = nnx.Variable(jnp.array(0, jnp.int32))

    @classmethod
    def create(cls, observation_dim, action_dim, config, *, rngs):
        from craft.config import validate_agent

        validate_agent(config)
        if min(observation_dim, action_dim) <= 0:
            raise ValueError("Observation/action dimensions must be positive")
        return cls(observation_dim, action_dim, config, rngs=rngs)

    @partial(nnx.jit, static_argnames=("stage",))
    def update(self, batch, *, stage="joint"):
        if stage not in ("joint", "representation", "action"):
            raise ValueError(f"Unknown stage: {stage}")
        s, actions = batch["observations"], batch["action_chunks"]
        metrics = {}
        grads = []
        if stage != "action":

            def critic_loss(model):
                return contrastive_loss(*model(s, actions, batch["value_goals"]))

            (_, info), critic_grad = nnx.value_and_grad(critic_loss, has_aux=True)(
                self.critic
            )
            grads.append(critic_grad)
            metrics.update(info)
            metrics["critic/grad_norm"] = optax.tree.norm(critic_grad)
        previous_count = self.rngs.updates.count[...]
        nk, tk, zk = jax.random.split(self.rngs.updates(), 3)
        if stage != "representation":
            goals = batch["actor_goals"]
            embedding = encode_condition(self.critic, s, goals)
            noise = jax.random.normal(nk, actions.shape)
            t = jax.random.uniform(tk, (actions.shape[0],))
            xt = (1 - t[:, None, None]) * noise + t[:, None, None] * actions

            def flow_loss(model):
                return jnp.square(
                    model(s, goals, embedding, xt, t) - (actions - noise)
                ).mean()

            fm, teacher_grad = nnx.value_and_grad(flow_loss)(self.teacher)
            z = jax.random.normal(zk, actions.shape)
            target = jax.lax.stop_gradient(
                integrate_flow(self.teacher, s, goals, embedding, z, self.flow_steps)
            )

            def actor_losses(model):
                raw = model(s, goals, embedding, z, jnp.zeros(actions.shape[0]))
                proposed = jnp.clip(raw, -1, 1)
                q = q_values(*self.critic(s, proposed, goals)).min(0)
                qloss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
                # Keep the recovery gradient when raw actions exceed the bounds.
                distill = jnp.square(raw - target).mean()
                return jnp.stack([qloss, self.distillation_weight * distill]), (
                    q.mean(),
                    distill,
                    (jnp.abs(raw) >= 1).mean(),
                )

            # One reverse pass per component exposes the actual gradient balance.
            components, pullback, (q, distill, saturation) = nnx.vjp(
                actor_losses, self.actor, has_aux=True, graph_updates=False
            )
            qgrad = pullback(jnp.array([1.0, 0.0]))[0]
            bcgrad = pullback(jnp.array([0.0, 1.0]))[0]
            actor_grad = jax.tree.map(lambda a, b: a + b, qgrad, bcgrad)
            grads.extend([teacher_grad, actor_grad])
            metrics.update(
                {
                    "actor/loss": components.sum(),
                    "actor/q_loss": components[0],
                    "actor/weighted_distillation": components[1],
                    "actor/distillation": distill,
                    "actor/flow_loss": fm,
                    "actor/q": q,
                    "actor/saturation": saturation,
                    "actor/q_grad_norm": optax.tree.norm(qgrad),
                    "actor/distill_grad_norm": optax.tree.norm(bcgrad),
                    "actor/grad_ratio_distill_to_q": optax.tree.norm(bcgrad)
                    / (optax.tree.norm(qgrad) + 1e-8),
                }
            )
        finite = all_finite((grads, metrics, batch))

        def commit(agent):
            index = 0
            if stage != "action":
                agent.optimizers[0].update(agent.critic, grads[index])
                index += 1
            if stage != "representation":
                agent.optimizers[1].update(agent.teacher, grads[index])
                agent.optimizers[2].update(agent.actor, grads[index + 1])
            agent.step[...] += 1

        def reject(agent):
            agent.rngs.updates.count[...] = previous_count

        nnx.cond(finite, commit, reject, self)
        return {**metrics, "numerics/finite": finite}

    @nnx.jit
    def sample_action_chunk(self, observations, goals, rng):
        return self.actor.sample(self.critic, observations, goals, rng)

    @nnx.jit
    def validation_metrics(self, batch):
        _, metrics = contrastive_loss(
            *self.critic(
                batch["observations"], batch["action_chunks"], batch["value_goals"]
            )
        )
        return {f"validation/{key}": value for key, value in metrics.items()}
