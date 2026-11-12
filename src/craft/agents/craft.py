"""CRL + conditional VAE POC. Guidance updates only the stochastic prior."""

from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from craft.losses.contrastive import contrastive_loss
from craft.losses.cvae import gaussian_kl
from craft.models.cvae import sample_gaussian
from craft.models.factory import make_poc_models
from craft.policies.chunk import sample_action_chunk


def all_finite(tree):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree)]))


class CRAFTAgent(nnx.Module):
    def __init__(self, observation_dim, action_dim, config, *, rngs, batch_size):
        self.critic, self.posterior, self.decoder, self.prior = make_poc_models(
            config, observation_dim, action_dim, rngs=rngs, batch_size=batch_size
        )
        self.observation_dim, self.action_dim = observation_dim, action_dim
        self.chunk_length = config["chunk_length"]
        self.kl_weight, self.guide_weight = config["kl_weight"], config["guide_weight"]
        self.guidance_enabled, self.cvae_updates_crl = (
            config["guidance_enabled"],
            config["cvae_updates_crl"],
        )
        self.condition_drop_prob = config["condition_drop_prob"]
        self.optimizers = nnx.List(
            [
                nnx.Optimizer(
                    model,
                    optax.chain(
                        optax.clip_by_global_norm(config["max_grad_norm"]),
                        optax.adam(
                            config["prior_learning_rate"]
                            if i == 3
                            else config["learning_rate"]
                        ),
                    ),
                    wrt=nnx.Param,
                )
                for i, model in enumerate(
                    (self.critic, self.posterior, self.decoder, self.prior)
                )
            ]
        )
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

    def ae_objective(
        self, critic, posterior, decoder, batch, key, mask, prior_parameters
    ):
        s, a, g = batch["observations"], batch["action_chunks"], batch["value_goals"]
        tokens, hg, hs = critic.features(s, a), critic.encode(g), critic.encode(s)
        if not self.cvae_updates_crl:
            tokens, hg, hs = jax.tree.map(jax.lax.stop_gradient, (tokens, hg, hs))
        q = posterior(tokens, hg)
        reconstruction = decoder(sample_gaussian(q, key), hs, hg, mask)
        rec = jnp.square(reconstruction - a).mean()
        kl = gaussian_kl(q, prior_parameters)
        return rec + self.kl_weight * kl, (rec, kl, q)

    def prior_objective(self, prior, batch, q, key):
        s, g = batch["observations"], batch["value_goals"]
        hs, hg = jax.tree.map(
            jax.lax.stop_gradient, (self.critic.encode(s), self.critic.encode(g))
        )
        p = prior(s, g, hs, hg)
        kl = gaussian_kl(jax.tree.map(jax.lax.stop_gradient, q), p)
        # No hard clip: decoder's differentiable tanh output is already bounded.
        actions = self.decoder(
            sample_gaussian(p, key), hs, hg, jnp.ones((s.shape[0], 1, 1))
        )
        guide = (
            -self.critic.score(s, actions, g).mean()
            if self.guidance_enabled
            else jnp.array(0.0)
        )
        return self.kl_weight * kl + self.guide_weight * guide, (kl, guide)

    @partial(nnx.jit, static_argnames=("stage",))
    def update(self, batch, *, stage="joint"):
        if stage not in ("joint", "representation", "action"):
            raise ValueError(f"Unknown stage: {stage}")
        s, a, g = batch["observations"], batch["action_chunks"], batch["value_goals"]
        previous = self.rngs.updates.count[...]
        qkey, pkey, mkey = jax.random.split(self.rngs.updates(), 3)

        def crl(model):
            return contrastive_loss(
                *model(s, a, g), model.log_scale[...], model.bias[...]
            )

        (_, metrics), cg = nnx.value_and_grad(crl, has_aux=True)(self.critic)
        hs, hg = jax.tree.map(
            jax.lax.stop_gradient, (self.critic.encode(s), self.critic.encode(g))
        )
        pp = jax.tree.map(jax.lax.stop_gradient, self.prior(s, g, hs, hg))
        mask = jax.random.bernoulli(
            mkey, 1 - self.condition_drop_prob, (s.shape[0], 1, 1)
        ).astype(s.dtype)

        def ae(c, p, d):
            return self.ae_objective(c, p, d, batch, qkey, mask, pp)

        (_, (rec, kl, q)), (acg, pg, dg) = nnx.value_and_grad(
            ae, argnums=(0, 1, 2), has_aux=True
        )(self.critic, self.posterior, self.decoder)

        def prior_terms(p):
            _, (pkl, guide) = self.prior_objective(p, batch, q, pkey)
            return jnp.stack([self.kl_weight * pkl, self.guide_weight * guide]), (
                pkl,
                guide,
            )

        terms, backward, (pkl, guide) = nnx.vjp(
            prior_terms, self.prior, has_aux=True, graph_updates=False
        )
        kg = backward(jnp.array([1.0, 0.0]))[0]
        gg = backward(jnp.array([0.0, 1.0]))[0]
        rg = jax.tree.map(lambda x, y: x + y, kg, gg)
        if stage == "joint":
            cg = jax.tree.map(lambda x, y: x + y, cg, acg)
        metrics.update(
            {
                "vae/reconstruction": rec,
                "vae/kl": kl,
                "prior/kl": pkl,
                "prior/guidance": guide,
                "prior/grad_norm": optax.tree.norm(rg),
                "prior/weighted_kl": terms[0],
                "prior/weighted_guidance": terms[1],
                "prior/kl_grad_norm": optax.tree.norm(kg),
                "prior/guide_grad_norm": optax.tree.norm(gg),
                "prior/guide_to_kl_grad_ratio": optax.tree.norm(gg)
                / (optax.tree.norm(kg) + 1e-8),
                "decoder/condition_gate": self.decoder.gate,
                "decoder/condition_keep": mask.mean(),
                "vae/posterior_std": jnp.exp(q[1]).mean(),
                "prior/std": jnp.exp(pp[1]).mean(),
            }
        )
        gradients = (cg, pg, dg, rg)
        finite = all_finite((gradients, metrics, batch))

        def commit(agent):
            for i, model in enumerate(
                (agent.critic, agent.posterior, agent.decoder, agent.prior)
            ):
                if (stage != "action" or i != 0) and (
                    stage != "representation" or i == 0
                ):
                    agent.optimizers[i].update(model, gradients[i])
            agent.step[...] += 1

        def reject(agent):
            agent.rngs.updates.count[...] = previous

        nnx.cond(finite, commit, reject, self)
        return {**metrics, "numerics/finite": finite}

    @nnx.jit
    def sample_action_chunk(self, observations, goals, rng):
        return sample_action_chunk(
            self.critic, self.prior, self.decoder, observations, goals, rng
        )

    @nnx.jit
    def validation_metrics(self, batch, key):
        s, a, g = batch["observations"], batch["action_chunks"], batch["value_goals"]
        _, metrics = contrastive_loss(
            *self.critic(s, a, g), self.critic.log_scale[...], self.critic.bias[...]
        )
        qkey, pkey = jax.random.split(key)
        hs, hg = self.critic.encode(s), self.critic.encode(g)
        q = self.posterior(self.critic.features(s, a), hg)
        p = self.prior(s, g, hs, hg)
        z = sample_gaussian(q, qkey)
        mask = jnp.ones((s.shape[0], 1, 1))
        reconstructed = self.decoder(z, hs, hg, mask)
        sampled = self.decoder(sample_gaussian(p, pkey), hs, hg, mask)
        shuffled = self.decoder(jnp.roll(z, 1, axis=0), hs, hg, mask)
        metrics.update(
            {
                "vae/reconstruction": jnp.square(reconstructed - a).mean(),
                "vae/kl": gaussian_kl(q, p),
                "vae/prior_sample_mse": jnp.square(sampled - a).mean(),
                "vae/shuffled_latent_mse": jnp.square(shuffled - a).mean(),
            }
        )
        return {f"validation/{k}": v for k, v in metrics.items()}
