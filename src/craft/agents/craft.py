"""Token CRL + representation-conditioned FQL; independent optimizer states."""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from craft.losses.contrastive import contrastive_loss, q_values
from craft.models.factory import make_crl_models
from craft.models.tokens import TokenCritic


@struct.dataclass
class CRAFTAgent:
    critic: TrainState
    teacher: TrainState
    actor: TrainState
    rng: Any
    chunk_length: int = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    flow_steps: int = struct.field(pytree_node=False, default=10)
    alpha: float = struct.field(pytree_node=False, default=10.0)

    @classmethod
    def create(cls, seed, observation_dim, action_dim, config):
        h = int(config["chunk_length"])
        width, depth, exchange_every, latent = [
            int(config[k]) for k in ("width", "depth", "exchange_every", "latent_dim")
        ]
        if (
            min(
                h,
                width,
                depth,
                exchange_every,
                latent,
                action_dim,
                observation_dim,
                int(config["flow_steps"]),
            )
            <= 0
        ):
            raise ValueError(
                "Dimensions, depth, exchange_every, and flow steps must be positive"
            )
        if config["alpha"] < 0 or config["learning_rate"] <= 0:
            raise ValueError("Invalid optimizer or distillation settings")
        keys = jax.random.split(jax.random.PRNGKey(seed), 4)
        s = jnp.zeros((1, observation_dim))
        a = jnp.zeros((1, h, action_dim))
        critic, actor = make_crl_models(config, action_dim)
        cp = critic.init(keys[0], s, a, s)["params"]
        ap = actor.init(keys[1], s, s, jnp.zeros((1, width * 2)), a, jnp.zeros(1))[
            "params"
        ]
        tp = actor.init(keys[2], s, s, jnp.zeros((1, width * 2)), a, jnp.zeros(1))[
            "params"
        ]

        def state(module, params):
            return TrainState.create(
                apply_fn=module.apply,
                params=params,
                tx=optax.chain(
                    optax.clip_by_global_norm(10.0), optax.adam(config["learning_rate"])
                ),
            )

        return cls(
            state(critic, cp),
            state(actor, tp),
            state(actor, ap),
            keys[3],
            h,
            action_dim,
            int(config["flow_steps"]),
            float(config["alpha"]),
        )

    def embeddings(self, observations, goals):
        # Reuse the critic's goal/state encoder, but detach actor conditioning.
        def encode(x):
            return self.critic.apply_fn(
                {"params": self.critic.params}, x, method=TokenCritic.encode
            )

        return jax.lax.stop_gradient(
            jnp.concatenate([encode(observations), encode(goals)], -1)
        )

    def integrate(self, observations, goals, embeddings, noise):
        def step(x, i):
            t = jnp.full((x.shape[0],), i / self.flow_steps)
            v = self.teacher.apply_fn(
                {"params": self.teacher.params}, observations, goals, embeddings, x, t
            )
            return x + v / self.flow_steps, None

        return jnp.clip(
            jax.lax.scan(step, noise, jnp.arange(self.flow_steps))[0], -1, 1
        )

    @partial(jax.jit, static_argnames=("stage",))
    def update(self, batch, *, stage="joint"):
        if stage not in ("joint", "representation", "action"):
            raise ValueError(f"Unknown stage: {stage}")
        s, a = batch["observations"], batch["action_chunks"]

        def critic_loss(params):
            phi, psi = self.critic.apply_fn(
                {"params": params}, s, a, batch["value_goals"]
            )
            return contrastive_loss(phi, psi)

        critic = self.critic
        metrics = {}
        if stage != "action":
            (_, info), grad = jax.value_and_grad(critic_loss, has_aux=True)(
                critic.params
            )
            critic = critic.apply_gradients(grads=grad)
            metrics.update(info)
        # Actor updates use the pre-update critic, matching simultaneous reference updates.
        rng, nk, tk, zk = jax.random.split(self.rng, 4)
        teacher, actor = self.teacher, self.actor
        if stage != "representation":
            goals = batch["actor_goals"]
            embedding = self.embeddings(s, goals)
            noise = jax.random.normal(nk, a.shape)
            t = jax.random.uniform(tk, (a.shape[0],))
            xt = (1 - t[:, None, None]) * noise + t[:, None, None] * a

            def flow_loss(p):
                pred = teacher.apply_fn({"params": p}, s, goals, embedding, xt, t)
                return jnp.square(pred - (a - noise)).mean()

            fm, grad = jax.value_and_grad(flow_loss)(teacher.params)
            teacher = teacher.apply_gradients(grads=grad)
            z = jax.random.normal(zk, a.shape)
            target = jax.lax.stop_gradient(self.integrate(s, goals, embedding, z))

            def actor_loss(p):
                actions = jnp.clip(
                    actor.apply_fn(
                        {"params": p}, s, goals, embedding, z, jnp.zeros(a.shape[0])
                    ),
                    -1,
                    1,
                )
                phi, psi = self.critic.apply_fn(
                    {"params": self.critic.params}, s, actions, goals
                )
                q = q_values(phi, psi).min(0)
                qloss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
                distill = jnp.square(actions - target).mean()
                loss = qloss + self.alpha * distill
                return loss, {
                    "actor/loss": loss,
                    "actor/q": q.mean(),
                    "actor/distillation": distill,
                    "actor/flow_loss": fm,
                }

            (_, info), grad = jax.value_and_grad(actor_loss, has_aux=True)(actor.params)
            actor = actor.apply_gradients(grads=grad)
            metrics.update(info)
        return self.replace(
            critic=critic, teacher=teacher, actor=actor, rng=rng
        ), metrics

    @jax.jit
    def sample_action_chunk(self, observations, goals, rng):
        single = observations.ndim == 1
        if single:
            observations, goals = observations[None], goals[None]
        embedding = self.embeddings(observations, goals)
        noise = jax.random.normal(
            rng, (observations.shape[0], self.chunk_length, self.action_dim)
        )
        actions = jnp.clip(
            self.actor.apply_fn(
                {"params": self.actor.params},
                observations,
                goals,
                embedding,
                noise,
                jnp.zeros(observations.shape[0]),
            ),
            -1,
            1,
        )
        return actions[0] if single else actions
