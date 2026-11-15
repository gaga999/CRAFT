"""Numerical and gradient-ownership contracts for the CVAE POC."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from craft.agents.craft import CRAFTAgent
from craft.config import load_config
from craft.data.offline import OfflineDataset, synthetic_dataset
from craft.losses.cvae import gaussian_kl
from craft.models.cvae import sample_gaussian
from craft.utils.checkpoint import pure_state, restore_checkpoint, save_checkpoint
from craft.utils.random import make_random_engines


def equal(a, b):
    x, xt = jax.tree.flatten(a)
    y, yt = jax.tree.flatten(b)
    return xt == yt and all(np.array_equal(i, j) for i, j in zip(x, y))


def assert_equivalent_training(test, expected, actual, steps, config, batch):
    """Same updates through different XLA programs (e.g. scanned vs single).

    Rounding differs between programs. Token-mixer output biases have
    analytically zero gradients (a per-token constant cancelled by the next
    LayerNorm), and Adam normalizes their rounding noise into up to one
    learning-rate step per update, so parameters are bounded by lr * steps
    while outputs must agree tightly.
    """
    test.assertEqual(int(expected.step[...]), int(actual.step[...]))
    test.assertEqual(
        int(expected.rngs.updates.count[...]), int(actual.rngs.updates.count[...])
    )
    key = jax.random.key(0)
    want = expected.validation_metrics(batch, key)
    got = actual.validation_metrics(batch, key)
    for name in want:
        np.testing.assert_allclose(got[name], want[name], rtol=1e-4, atol=1e-5)
    gaps = sorted(
        (
            float(np.max(np.abs(np.asarray(x) - np.asarray(y)), initial=0)),
            jax.tree_util.keystr(path),
        )
        for (path, x), (_, y) in zip(
            jax.tree.flatten_with_path(pure_state(expected))[0],
            jax.tree.flatten_with_path(pure_state(actual))[0],
        )
    )
    bound = steps * max(config["learning_rate"], config["prior_learning_rate"])
    test.assertLess(gaps[-1][0], bound, msg=gaps[-5:])


def seeded_agent(seed, observation_dim, action_dim, config):
    engine, _ = make_random_engines(seed)
    return CRAFTAgent.create(observation_dim, action_dim, config, rngs=engine.jax)


def seeded_source(seed=0, chunk_length=5, observation_dim=6, action_dim=2):
    engine, _ = make_random_engines(seed)
    data = synthetic_dataset(
        rng=engine.numpy("synthetic"),
        observation_dim=observation_dim,
        action_dim=action_dim,
    )
    return OfflineDataset(data, chunk_length, rng=engine.numpy()), engine


class FakeEnv:
    action_space = SimpleNamespace(shape=(2,))
    observation_space = SimpleNamespace(shape=(6,))
    spec = SimpleNamespace(max_episode_steps=3)
    task_infos = ({"task_name": "test"},)
    closed = False

    @property
    def unwrapped(self):
        return self

    def reset(self, **kwargs):
        self.steps = 0
        return np.zeros(6, np.float32), {"goal": np.ones(6, np.float32)}

    def step(self, action):
        self.steps += 1
        return (
            np.zeros(6, np.float32),
            0,
            False,
            self.steps == 3,
            {"success": self.steps == 2},
        )

    def close(self):
        self.closed = True


class POCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = load_config("configs/experiment/smoke.yaml")["agent"]
        cls.initial = seeded_agent(0, 6, 2, cls.c)
        cls.b = seeded_source()[0].sample(4)

    def setUp(self):
        self.a = nnx.clone(self.initial)

    def test_shapes_and_sampling(self):
        s, a, g = self.b["observations"], self.b["action_chunks"], self.b["value_goals"]
        tokens = self.a.critic.features(s, a)
        self.assertEqual(tokens.shape, (4, 6, 16))
        hg, hs = self.a.critic.encode(g), self.a.critic.encode(s)
        q = self.a.posterior(s, a, g, (tokens, hg))
        p = self.a.prior(s, g, (hs, hg))
        for value in (*q, *p):
            self.assertEqual(value.shape, (4, 5, 3))
        key = make_random_engines(3)[1].task_key(1)
        actions = self.a.sample_action_chunk(s, g, key)
        self.assertEqual(actions.shape, (4, 5, 2))
        self.assertTrue(np.isfinite(actions).all() and (np.abs(actions) <= 1).all())
        self.assertEqual(self.a.sample_action_chunk(s[0], g[0], key).shape, (5, 2))

    def test_all_modules_and_scalars_update_reproducibly(self):
        other = nnx.clone(self.a)
        before = pure_state(self.a)
        m = self.a.update(self.b)
        other.update(self.b)
        self.assertTrue(all(np.isfinite(x) for x in m.values()))
        self.assertTrue(equal(pure_state(self.a), pure_state(other)))
        for role in ("critic", "posterior", "decoder", "prior"):
            self.assertFalse(equal(before[role], pure_state(self.a)[role]))
        self.assertNotEqual(
            float(self.a.critic.log_scale[...]),
            float(self.initial.critic.log_scale[...]),
        )
        self.assertNotEqual(
            float(self.a.critic.bias[...]), float(self.initial.critic.bias[...])
        )

    def test_ae_gradient_switch(self):
        key = make_random_engines(0)[1].task_key(1)
        s, g = self.b["observations"], self.b["value_goals"]
        p = self.a.prior(s, g, (self.a.critic.encode(s), self.a.critic.encode(g)))
        for enabled in (False, True):
            self.a.cvae_updates_crl = enabled
            grad = nnx.grad(
                lambda c: self.a.ae_objective(
                    c,
                    self.a.posterior,
                    self.a.decoder,
                    self.b,
                    key,
                    jnp.ones((4, 1, 1)),
                    p,
                )[0]
            )(self.a.critic)
            norm = sum(float(jnp.square(x).sum()) for x in jax.tree.leaves(grad))
            self.assertEqual(norm > 0, enabled)

    def test_dropout_blocks_bias_and_latent_still_matters(self):
        s, g = self.b["observations"], self.b["value_goals"]
        hs, hg = self.a.critic.encode(s), self.a.critic.encode(g)
        z = jnp.ones((4, 5, 3))
        mask = jnp.zeros((4, 1, 1))
        first = self.a.decoder(z, s, g, mask, (hs, hg))
        second = self.a.decoder(z, s, g, mask, (hs + 2, hg - 3))
        np.testing.assert_array_equal(first, second)
        self.assertFalse(
            np.allclose(first, self.a.decoder(z * 0, s, g, mask, (hs, hg)))
        )
        self.assertFalse(
            np.allclose(first, self.a.decoder(z, s, g, jnp.ones_like(mask), (hs, hg)))
        )

    def test_guidance_gradient_through_frozen_decoder_without_clip(self):
        self.a.kl_weight = 0.0
        self.a.decoder.output.bias[...] = jnp.ones(2) * 2
        s, a, g = self.b["observations"], self.b["action_chunks"], self.b["value_goals"]
        q = self.a.posterior(
            s, a, g, (self.a.critic.features(s, a), self.a.critic.encode(g))
        )
        before = pure_state(self.a)
        key = make_random_engines(0)[1].task_key(1)
        grad = nnx.grad(lambda p: self.a.prior_objective(p, self.b, q, key)[0])(
            self.a.prior
        )
        self.assertGreater(
            sum(float(jnp.square(x).sum()) for x in jax.tree.leaves(grad)), 0
        )
        self.assertTrue(equal(before, pure_state(self.a)))
        self.a.guidance_enabled = False
        grad = nnx.grad(lambda p: self.a.prior_objective(p, self.b, q, key)[0])(
            self.a.prior
        )
        self.assertEqual(
            sum(float(jnp.square(x).sum()) for x in jax.tree.leaves(grad)), 0
        )

    def test_guidance_changes_only_prior_updates(self):
        other = nnx.clone(self.a)
        other.guidance_enabled = False
        self.a.update(self.b)
        other.update(self.b)
        for role in ("critic", "posterior", "decoder"):
            self.assertTrue(equal(pure_state(self.a)[role], pure_state(other)[role]))
        self.assertFalse(equal(pure_state(self.a)["prior"], pure_state(other)["prior"]))
        s, g = self.b["observations"], self.b["value_goals"]
        hs, hg = self.a.critic.encode(s), self.a.critic.encode(g)
        grad = jax.grad(lambda h: self.a.prior(s, g, (h, hg))[0].sum())(hs)
        np.testing.assert_array_equal(grad, jnp.zeros_like(hs))

    def test_validation_is_finite_repeatable_and_read_only(self):
        key = make_random_engines(1)[1].task_key(0)
        before = pure_state(self.a)
        first = self.a.validation_metrics(self.b, key)
        second = self.a.validation_metrics(self.b, key)
        self.assertTrue(equal(first, second))
        self.assertTrue(equal(before, pure_state(self.a)))
        self.assertTrue(all(np.isfinite(value) for value in first.values()))
        self.assertIn("validation/vae/shuffled_latent_mse", first)

    def test_kl_formula_and_reduction(self):
        mean = jnp.zeros((2, 5, 3))
        logstd = jnp.zeros_like(mean)
        self.assertAlmostEqual(float(gaussian_kl((mean, logstd), (mean, logstd))), 0)
        self.assertAlmostEqual(
            float(gaussian_kl((mean + 1, logstd), (mean, logstd))), 7.5
        )

    def test_nonfinite_atomic_and_stage_freezing(self):
        before = pure_state(self.a)
        bad = {**self.b, "observations": np.full_like(self.b["observations"], np.nan)}
        self.assertFalse(bool(self.a.update(bad)["numerics/finite"]))
        self.assertTrue(equal(before, pure_state(self.a)))
        self.a.update(self.b, stage="representation")
        for role in ("posterior", "decoder", "prior"):
            self.assertTrue(equal(before[role], pure_state(self.a)[role]))
        critic = pure_state(self.a)["critic"]
        self.a.update(self.b, stage="action")
        self.assertTrue(equal(critic, pure_state(self.a)["critic"]))

    def test_update_chunk_matches_sequential_updates_and_rejects_atomically(self):
        source = seeded_source()[0]
        batches = [source.sample(4, include_actor_goals=False) for _ in range(3)]
        stacked = jax.tree.map(lambda *x: jnp.stack(x), *batches)
        sequential = nnx.clone(self.initial)
        for batch in batches:
            last = sequential.update(batch, stage="joint")
        metrics = self.a.update_chunk(stacked, stage="joint")
        self.assertTrue(bool(metrics["numerics/finite"]))
        self.assertEqual(int(self.a.step[...]), 3)
        assert_equivalent_training(self, sequential, self.a, 3, self.c, batches[0])
        self.assertAlmostEqual(
            float(metrics["critic/loss"]), float(last["critic/loss"]), places=5
        )
        before = pure_state(self.a)
        bad = jax.tree.map(np.array, stacked)
        bad["observations"][1] = np.nan
        self.assertFalse(bool(self.a.update_chunk(bad)["numerics/finite"]))
        self.assertTrue(equal(before, pure_state(self.a)))

    def test_checkpoint_and_gaussian_reparameterization(self):
        self.a.update(self.b)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cp"
            save_checkpoint(path, agent=self.a, run_state={})
            other = nnx.clone(self.initial)
            restore_checkpoint(path, agent=other)
            self.a.update(self.b)
            other.update(self.b)
            self.assertTrue(equal(pure_state(self.a), pure_state(other)))
        key = make_random_engines(0)[1].task_key(1)
        grad = jax.grad(
            lambda mu: sample_gaussian((mu, jnp.zeros_like(mu)), key).sum()
        )(jnp.zeros((1, 5, 3)))
        np.testing.assert_array_equal(grad, np.ones((1, 5, 3)))

    def test_mean_pool_and_action_order(self):
        c = self.a.critic
        s, a, g = self.b["observations"], self.b["action_chunks"], self.b["value_goals"]
        expected = c.phi_head(c.features(s, a).mean(1))
        expected = expected / jnp.sqrt(
            jnp.square(expected).sum(-1, keepdims=True) + 1e-8
        )
        np.testing.assert_allclose(c(s, a, g)[0], expected)
        self.assertFalse(np.allclose(c(s, a, g)[0], c(s, a[:, ::-1].copy(), g)[0]))


def norm2(tree):
    return sum(float(jnp.square(x).sum()) for x in jax.tree.leaves(tree))


class OwnRepresentationTests(unittest.TestCase):
    """`cvae_representation=own`: CVAE never reads critic features."""

    @classmethod
    def setUpClass(cls):
        cls.c = load_config(
            "configs/experiment/smoke.yaml", ["agent.cvae_representation=own"]
        )["agent"]
        cls.initial = seeded_agent(0, 6, 2, cls.c)
        cls.critic_mode = seeded_agent(
            0, 6, 2, load_config("configs/experiment/smoke.yaml")["agent"]
        )
        cls.b = seeded_source()[0].sample(4)

    def setUp(self):
        self.a = nnx.clone(self.initial)

    def perturbed_critic(self):
        other = nnx.clone(self.a)
        state = nnx.state(other.critic, nnx.Param)
        nnx.update(other.critic, jax.tree.map(lambda x: x + 0.5, state))
        return other

    def test_cvae_outputs_ignore_critic_parameters(self):
        other = self.perturbed_critic()
        s, a, g = self.b["observations"], self.b["action_chunks"], self.b["value_goals"]
        z = jnp.ones((4, 5, 3))
        mask = jnp.ones((4, 1, 1))
        for x, y in (
            (self.a.posterior(s, a, g), other.posterior(s, a, g)),
            (self.a.prior(s, g), other.prior(s, g)),
            (self.a.decoder(z, s, g, mask), other.decoder(z, s, g, mask)),
        ):
            self.assertTrue(equal(x, y))
        key = make_random_engines(3)[1].task_key(1)
        np.testing.assert_array_equal(
            self.a.sample_action_chunk(s, g, key), other.sample_action_chunk(s, g, key)
        )
        mine, theirs = (x.validation_metrics(self.b, key) for x in (self.a, other))
        for name in mine:
            if "/vae/" in name:
                self.assertEqual(float(mine[name]), float(theirs[name]), name)
        self.assertNotEqual(
            float(mine["validation/critic/loss"]),
            float(theirs["validation/critic/loss"]),
        )

    def test_ae_gradients_reach_own_encoders_not_critic(self):
        key = make_random_engines(0)[1].task_key(1)
        s, g = self.b["observations"], self.b["value_goals"]
        p = self.a.prior(s, g)
        critic, posterior, decoder = nnx.grad(
            lambda c, q, d: self.a.ae_objective(
                c, q, d, self.b, key, jnp.ones((4, 1, 1)), p
            )[0],
            argnums=(0, 1, 2),
        )(self.a.critic, self.a.posterior, self.a.decoder)
        self.assertEqual(norm2(critic), 0)
        for name, grad in (
            ("posterior.backbone", posterior.backbone),
            ("posterior.goal_encoder", posterior.goal_encoder),
            ("decoder.encoder", decoder.encoder),
        ):
            self.assertGreater(norm2(grad), 0, name)

    def test_guidance_uses_critic_score_and_updates_only_prior(self):
        key = make_random_engines(0)[1].task_key(1)
        s, a, g = self.b["observations"], self.b["action_chunks"], self.b["value_goals"]
        q = self.a.posterior(s, a, g)
        self.a.kl_weight = 0.0

        def prior_grad(agent):
            return nnx.grad(lambda p: agent.prior_objective(p, self.b, q, key)[0])(
                agent.prior
            )

        grad = prior_grad(self.a)
        self.assertGreater(norm2(grad), 0)
        other = self.perturbed_critic()
        other.kl_weight = 0.0
        self.assertFalse(equal(grad, prior_grad(other)))
        self.a.kl_weight = self.initial.kl_weight
        guided, unguided = nnx.clone(self.a), nnx.clone(self.a)
        unguided.guidance_enabled = False
        guided.update(self.b)
        unguided.update(self.b)
        for role in ("critic", "posterior", "decoder"):
            self.assertTrue(
                equal(pure_state(guided)[role], pure_state(unguided)[role]), role
            )
        self.assertFalse(
            equal(pure_state(guided)["prior"], pure_state(unguided)["prior"])
        )

    def test_update_changes_all_modules_and_checkpoint_resumes(self):
        before = pure_state(self.a)
        metrics = self.a.update(self.b)
        self.assertTrue(bool(metrics["numerics/finite"]))
        self.assertTrue(all(np.isfinite(x) for x in metrics.values()))
        after = pure_state(self.a)
        for role in ("critic", "posterior", "decoder", "prior"):
            self.assertFalse(equal(before[role], after[role]), role)
        for role, name in (
            ("posterior", "backbone"),
            ("posterior", "goal_encoder"),
            ("decoder", "encoder"),
        ):
            self.assertFalse(equal(before[role][name], after[role][name]), name)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cp"
            save_checkpoint(path, agent=self.a, run_state={})
            other = nnx.clone(self.initial)
            restore_checkpoint(path, agent=other)
            self.a.update(self.b)
            other.update(self.b)
            self.assertTrue(equal(pure_state(self.a), pure_state(other)))
            with self.assertRaises(ValueError):
                restore_checkpoint(path, agent=nnx.clone(self.critic_mode))

    def test_feature_contract_is_explicit(self):
        s, a, g = self.b["observations"], self.b["action_chunks"], self.b["value_goals"]
        critic = self.critic_mode
        hs, hg = critic.critic.encode(s), critic.critic.encode(g)
        with self.assertRaisesRegex(ValueError, "rejects"):
            self.a.prior(s, g, (hs, hg))
        with self.assertRaisesRegex(ValueError, "requires"):
            critic.decoder(jnp.ones((4, 5, 3)), s, g, jnp.ones((4, 1, 1)))
        with self.assertRaisesRegex(ValueError, "requires"):
            critic.posterior(s, a, g)
