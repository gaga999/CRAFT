"""Numerical, gradient-ownership and state contracts of the latent-actor agent."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from craft.agents.craft import CRAFTAgent, pairwise_auc
from craft.config import load_config
from craft.data.offline import OfflineDataset, synthetic_dataset
from craft.losses.actor import actor_loss
from craft.losses.contrastive import contrastive_loss
from craft.losses.reconstruction import reconstruction_mse
from craft.models.tokens import MixerBlock
from craft.utils.checkpoint import (
    load_modules,
    pure_state,
    read_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
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
    key = jax.random.key(0)
    want = expected.validation_metrics(batch, key, {})
    got = actual.validation_metrics(batch, key, {})
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
    bound = steps * max(config["learning_rate"], config["actor_learning_rate"])
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


def norm(tree):
    return sum(float(jnp.square(x).sum()) for x in jax.tree.leaves(tree))


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


class MixerTests(unittest.TestCase):
    def test_convolution_matches_dense_outputs_and_gradients(self):
        def dense_reference(block, x):
            x = block.local(x)
            y = jnp.swapaxes(block.token_norm(x), -1, -2)
            first, second = block.token_mlp.layers
            y = nnx.gelu(y @ first.kernel[0] + first.bias[...])
            y = y @ second.kernel[0] + second.bias[...]
            return x + jnp.swapaxes(y, -1, -2)

        for tokens, width in ((5, 16), (6, 16), (6, 128)):
            with self.subTest(tokens=tokens, width=width):
                engine, _ = make_random_engines(0)
                block = MixerBlock(width, tokens, True, rngs=engine.jax)
                x = jax.random.normal(engine.task_key(0), (2, tokens, width))
                np.testing.assert_allclose(
                    block(x), dense_reference(block, x), atol=1e-6
                )
                actual = nnx.grad(lambda m, v: jnp.square(m(v)).mean(), argnums=(0, 1))(
                    block, x
                )
                expected = nnx.grad(
                    lambda m, v: jnp.square(dense_reference(m, v)).mean(),
                    argnums=(0, 1),
                )(block, x)
                for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
                    np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)
                primitives = str(jax.make_jaxpr(block.token_mlp)(x))
                self.assertEqual(primitives.count("conv_general_dilated["), 2)
                self.assertNotIn("transpose[", primitives)


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = load_config("configs/experiment/smoke.yaml")["agent"]
        cls.initial = seeded_agent(0, 6, 2, cls.c)
        cls.b = seeded_source()[0].sample(4)

    def setUp(self):
        self.a = nnx.clone(self.initial)

    def inputs(self):
        return (
            self.b[k]
            for k in ("observations", "action_chunks", "value_goals", "actor_goals")
        )

    def test_shapes_normalization_and_policy(self):
        rep = self.a.representation
        s, a, g, _ = self.inputs()
        z = rep.latent(s, a)
        self.assertEqual(z.shape, (4, 5, self.c["latent_dim"]))
        np.testing.assert_allclose(z.mean(-1), 0, atol=1e-5)
        np.testing.assert_allclose(jnp.square(z).mean(-1), 1, atol=1e-3)
        z_hat = self.a.actor(s, g)
        np.testing.assert_allclose(jnp.square(z_hat).mean(-1), 1, atol=1e-3)
        for vector in (rep.phi(s, a), rep.goal(g)):
            np.testing.assert_allclose(jnp.linalg.norm(vector, axis=-1), 1, atol=1e-5)
        policy = self.a.policy()
        actions = policy(s, g, None)
        self.assertEqual(actions.shape, (4, 5, 2))
        self.assertTrue(np.isfinite(actions).all() and (np.abs(actions) <= 1).all())
        self.assertEqual(policy(s[0], g[0]).shape, (5, 2))
        np.testing.assert_allclose(
            actions, rep.decode(s, self.a.actor(s, g)), rtol=1e-5, atol=1e-6
        )
        self.assertFalse(np.allclose(rep.phi(s, a), rep.phi(s, a[:, ::-1].copy())))

    def test_score_matches_siglip_logit_without_bias(self):
        rep = self.a.representation
        rep.log_scale[...] = jnp.log(jnp.array(3.0))
        rep.bias[...] = jnp.array(-2.0)
        s, a, g, _ = self.inputs()
        phi, psi = rep.phi(s, a), rep.goal(g)
        logits = 3 * (phi @ psi.T) - 2
        np.testing.assert_allclose(
            rep.score(s, rep.latent(s, a), g), jnp.diag(logits) + 2, atol=1e-5
        )
        loss, metrics = contrastive_loss(phi, psi, rep.log_scale[...], rep.bias[...])
        labels = 2 * jnp.eye(4) - 1
        np.testing.assert_allclose(
            loss, -jax.nn.log_sigmoid(labels * logits).sum(-1).mean(), rtol=1e-6
        )
        np.testing.assert_allclose(metrics["crl/scale"], 3.0, rtol=1e-6)
        # AUC: fraction of off-diagonal pairs the true goal outscores.
        wins = sum(
            float(logits[i, i] > logits[i, j])
            for i in range(4)
            for j in range(4)
            if i != j
        )
        np.testing.assert_allclose(pairwise_auc(logits), wins / 12)
        np.testing.assert_allclose(pairwise_auc(jnp.eye(4)), 1.0)
        np.testing.assert_allclose(pairwise_auc(-jnp.eye(4)), 0.0)

    def test_component_gradients_reach_only_their_owners(self):
        rep = self.a.representation
        s, a, g, _ = self.inputs()

        def crl(model):
            phi = model.embed(model.latent(s, a), model.state_features(s))
            return contrastive_loss(
                phi, model.goal(g), model.log_scale[...], model.bias[...]
            )[0]

        def rec(model):
            return reconstruction_mse(model.decode(s, model.latent(s, a)), a)

        crl_grad, rec_grad = nnx.grad(crl)(rep), nnx.grad(rec)(rep)
        self.assertEqual(norm(crl_grad["decoder"]), 0)
        for name in ("state_encoder", "head", "goal_encoder", "log_scale", "bias"):
            self.assertEqual(norm(rec_grad[name]), 0, name)
        for grad in (crl_grad, rec_grad):
            self.assertGreater(norm(grad["backbone"]), 0)
            self.assertGreater(norm(grad["latent_projection"]), 0)

    def test_actor_gradient_passes_through_frozen_representation(self):
        s, a, _, actor_goals = self.inputs()
        rep = self.a.representation
        z_data = rep.latent(s, a)
        before = pure_state(rep)

        def q_only(actor):
            z = actor(s, actor_goals)
            return actor_loss(rep.score(s, z, actor_goals), z, z_data, 0.0)[0]

        rep.log_scale[...] = jnp.log(jnp.array(2.0))
        first = nnx.grad(q_only)(self.a.actor)
        rep.log_scale[...] = jnp.log(jnp.array(9.0))
        scaled = nnx.grad(q_only)(self.a.actor)
        self.assertGreater(norm(first), 0)
        # Score normalization removes the learned temperature from the actor step.
        for x, y in zip(jax.tree.leaves(first), jax.tree.leaves(scaled)):
            np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-6)
        rep.log_scale[...] = before["log_scale"]
        self.assertTrue(equal(before, pure_state(rep)))

    def test_stage_ownership_and_joint_equals_separate_updates(self):
        before = pure_state(self.initial)
        states = {}
        for stage in ("representation", "actor", "joint"):
            agent = nnx.clone(self.initial)
            metrics = agent.update(self.b, stage=stage)
            self.assertTrue(bool(metrics["numerics/finite"]))
            states[stage] = pure_state(agent)
            self.assertEqual(int(agent.step[...]), 1)
        rep_keys = ("representation", "representation_optimizer")
        actor_keys = ("actor", "actor_optimizer")
        for key in actor_keys:
            self.assertTrue(equal(before[key], states["representation"][key]))
            self.assertFalse(equal(before[key], states["actor"][key]))
        for key in rep_keys:
            self.assertTrue(equal(before[key], states["actor"][key]))
            self.assertFalse(equal(before[key], states["representation"][key]))
        # Joint = each optimizer's own step from the same pre-update parameters.
        for key in rep_keys:
            self.assertTrue(equal(states["joint"][key], states["representation"][key]))
        # The actor-stage program computes z without a representation backward
        # pass; XLA fusion differs, so allow rounding-level differences.
        for key in actor_keys:
            for x, y in zip(
                jax.tree.leaves(states["joint"][key]),
                jax.tree.leaves(states["actor"][key]),
            ):
                np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-7)
        self.assertEqual(int(self.initial.rngs.updates.count[...]), 0)

    def test_latent_noise_stays_on_support_and_uses_update_rng_only(self):
        s, a, _, _ = self.inputs()
        z = self.a.representation.latent(s, a)
        key = make_random_engines(2)[1].task_key(0)
        noisy = self.a.perturb(z, key)
        self.assertGreater(self.c["latent_noise"], 0)
        self.assertFalse(np.allclose(noisy, z))
        np.testing.assert_allclose(noisy.mean(-1), 0, atol=1e-5)
        np.testing.assert_allclose(jnp.square(noisy).mean(-1), 1, atol=1e-3)
        self.a.latent_noise = 0.0
        np.testing.assert_array_equal(self.a.perturb(z, key), z)
        # Decoder-input noise draws from the update stream only when
        # reconstruction is trained, and a rejected update rolls it back.
        counts = {}
        for stage in ("representation", "actor", "joint"):
            agent = nnx.clone(self.initial)
            agent.update(self.b, stage=stage)
            counts[stage] = int(agent.rngs.updates.count[...])
        self.assertEqual(counts, {"representation": 1, "actor": 0, "joint": 1})
        agent = nnx.clone(self.initial)
        bad = {**self.b, "observations": np.full_like(self.b["observations"], np.nan)}
        agent.update(bad)
        self.assertEqual(int(agent.rngs.updates.count[...]), 0)
        # Head and decoder read the same noisy latent: noise changes both terms,
        # and a fixed key reproduces both exactly.
        quiet = nnx.clone(self.initial)
        quiet.latent_noise = 0.0
        rep = self.initial.representation
        noisy_terms = self.initial.representation_objective(rep, self.b, key)[1][0]
        again = self.initial.representation_objective(rep, self.b, key)[1][0]
        quiet_terms = quiet.representation_objective(rep, self.b, key)[1][0]
        for name in ("crl/loss", "rec/mse"):
            self.assertNotEqual(float(noisy_terms[name]), float(quiet_terms[name]))
            self.assertEqual(float(noisy_terms[name]), float(again[name]))

    def test_weight_decay_shrinks_only_kernels_and_scale_is_capped(self):
        rep = self.a.representation
        before = pure_state(rep)
        zero = jax.tree.map(jnp.zeros_like, nnx.state(rep, nnx.Param))
        self.a.representation_optimizer.update(rep, zero)
        after = pure_state(rep)
        changed = {
            jax.tree_util.keystr(p): not np.array_equal(x, y)
            for (p, x), (_, y) in zip(
                jax.tree.flatten_with_path(before)[0],
                jax.tree.flatten_with_path(after)[0],
            )
        }
        self.assertTrue(any(changed.values()))
        for path, moved in changed.items():
            self.assertEqual(moved, path.endswith("['kernel']"), path)
        s, a, g, _ = self.inputs()
        z = rep.latent(s, a)
        rep.log_scale[...] = jnp.log(jnp.array(1000.0))
        capped = rep.score(s, z, g)
        rep.log_scale[...] = jnp.log(jnp.array(self.c["max_scale"]))
        np.testing.assert_allclose(capped, rep.score(s, z, g), rtol=1e-6)
        grad = nnx.grad(
            lambda m: m.logits(m.phi(s, a), m.goal(g), pairwise=True).sum()
        )(rep)
        self.assertNotEqual(float(grad["log_scale"][...]), 0)
        rep.log_scale[...] = jnp.log(jnp.array(1000.0))
        grad = nnx.grad(
            lambda m: m.logits(m.phi(s, a), m.goal(g), pairwise=True).sum()
        )(rep)
        self.assertEqual(float(grad["log_scale"][...]), 0)

    def test_reencoded_guidance_scores_the_executed_chunk(self):
        rep = self.a.representation
        s, _, _, goals = self.inputs()
        z_hat = self.a.actor(s, goals)
        executed = rep.latent(s, rep.decode(s, z_hat))
        np.testing.assert_allclose(
            self.a.guided_score(rep, s, z_hat, goals),
            rep.score(s, executed, goals),
            rtol=1e-6,
        )
        before = pure_state(rep)
        grads = {}
        for mode in ("reencoded", "latent"):
            self.a.guidance = mode
            grads[mode] = nnx.grad(
                lambda m: self.a.guided_score(rep, s, m(s, goals), goals).mean()
            )(self.a.actor)
            self.assertGreater(norm(grads[mode]), 0)
        self.assertFalse(equal(grads["reencoded"], grads["latent"]))
        self.assertTrue(equal(before, pure_state(rep)))

    def test_updates_are_reproducible_and_validation_read_only(self):
        other = nnx.clone(self.a)
        self.a.update(self.b)
        other.update(self.b)
        self.assertTrue(equal(pure_state(self.a), pure_state(other)))
        before = pure_state(self.a)
        key = make_random_engines(1)[1].task_key(0)
        source = seeded_source()[0]
        horizons = {
            5: source.sample_at_offset(4, 5),
            20: source.sample_at_offset(4, 20),
        }
        first = self.a.validation_metrics(self.b, key, horizons)
        second = self.a.validation_metrics(self.b, key, horizons)
        self.assertTrue(equal(first, second))
        self.assertTrue(equal(before, pure_state(self.a)))
        self.assertTrue(all(np.isfinite(v) for v in first.values()))
        self.assertEqual(
            set(first),
            {
                f"validation/{name}"
                for name in (
                    "crl/loss",
                    "crl/auc_5",
                    "crl/auc_20",
                    "latent/action_to_state_effect",
                    "rec/mse",
                    "rec/shuffled_latent_mse",
                    "actor/score_gain",
                    "actor/score_exploit_gap",
                    "actor/action_mse",
                    "grad/crl_to_rec_ratio",
                    "grad/q_to_bc_ratio",
                )
            },
        )

    def test_offset_sampler_returns_exact_future_states_or_none(self):
        source = seeded_source()[0]
        before = source.rng.bit_generator.state
        batch = source.sample_at_offset(8, 20, rng=np.random.default_rng(0))
        self.assertEqual(before, source.rng.bit_generator.state)
        obs, nxt = source.data["observations"], source.data["next_observations"]
        for s, goal in zip(batch["observations"], batch["value_goals"]):
            i = int(np.flatnonzero((obs == s).all(1))[0])
            np.testing.assert_array_equal(goal, nxt[i + 19])
            self.assertEqual(i // 32, (i + 19) // 32)  # same synthetic episode
        self.assertIsNone(source.sample_at_offset(8, 33))

    def test_nonfinite_updates_are_rejected_atomically(self):
        before = pure_state(self.a)
        bad = {**self.b, "observations": np.full_like(self.b["observations"], np.nan)}
        for stage in ("representation", "actor", "joint"):
            self.assertFalse(bool(self.a.update(bad, stage=stage)["numerics/finite"]))
            self.assertTrue(equal(before, pure_state(self.a)))
        with self.assertRaises(ValueError):
            self.a.update(self.b, stage="action")

    def test_update_chunk_matches_sequential_updates_and_rejects_atomically(self):
        source = seeded_source()[0]
        batches = [source.sample(4) for _ in range(3)]
        stacked = jax.tree.map(lambda *x: jnp.stack(x), *batches)
        sequential = nnx.clone(self.initial)
        for batch in batches:
            last = sequential.update(batch)
        metrics = self.a.update_chunk(stacked)
        self.assertTrue(bool(metrics["numerics/finite"]))
        self.assertEqual(int(self.a.step[...]), 3)
        assert_equivalent_training(self, sequential, self.a, 3, self.c, batches[0])
        self.assertAlmostEqual(
            float(metrics["crl/loss"]), float(last["crl/loss"]), places=4
        )
        before = pure_state(self.a)
        bad = jax.tree.map(np.array, stacked)
        bad["observations"][1] = np.nan
        self.assertFalse(bool(self.a.update_chunk(bad)["numerics/finite"]))
        self.assertTrue(equal(before, pure_state(self.a)))

    def test_checkpoint_restore_and_module_initialization(self):
        self.a.update(self.b)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cp"
            save_checkpoint(path, agent=self.a, run_state={})
            other = nnx.clone(self.initial)
            restore_checkpoint(path, agent=other)
            self.a.update(self.b)
            other.update(self.b)
            self.assertTrue(equal(pure_state(self.a), pure_state(other)))
            # Initialization copies parameters only; optimizers and step stay fresh.
            fresh = nnx.clone(self.initial)
            load_modules(read_checkpoint(path), agent=fresh, modules=["representation"])
            state = pure_state(fresh)
            saved = read_checkpoint(path)["agent"]
            self.assertTrue(
                equal(
                    state["representation"],
                    jax.tree.map(np.asarray, saved["representation"]),
                )
            )
            for key in ("actor", "representation_optimizer", "actor_optimizer", "step"):
                self.assertTrue(equal(state[key], pure_state(self.initial)[key]))
            # A shape mismatch in any requested module leaves the agent unchanged.
            wide = seeded_agent(0, 6, 2, {**self.c, "actor_width": 8})
            save_checkpoint(path, agent=wide, run_state={})
            before = pure_state(fresh)
            with self.assertRaisesRegex(ValueError, "module 'actor'"):
                load_modules(
                    read_checkpoint(path),
                    agent=fresh,
                    modules=["representation", "actor"],
                )
            self.assertTrue(equal(before, pure_state(fresh)))
