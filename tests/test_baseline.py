"""Numerical contracts for the NNX CRL implementation."""

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
from craft.evaluators.rollout import evaluate_rollouts
from craft.models.tokens import TokenBackbone
from craft.policies.chunk import encode_condition
from craft.utils.checkpoint import (
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


def seeded_agent(seed, observation_dim, action_dim, config):
    engine, _ = make_random_engines(seed)
    return CRAFTAgent.create(observation_dim, action_dim, config, rngs=engine.jax)


def seeded_source(seed=0, chunk_length=2, observation_dim=6, action_dim=2):
    engine, _ = make_random_engines(seed)
    raw = synthetic_dataset(
        rng=engine.numpy("synthetic"),
        observation_dim=observation_dim,
        action_dim=action_dim,
    )
    return OfflineDataset(raw, chunk_length, rng=engine.numpy()), engine


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


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("configs/experiment/smoke.yaml")
        cls.initial = seeded_agent(0, 6, 2, cls.config["agent"])
        cls.batch = seeded_source()[0].sample(4)

    def setUp(self):
        self.agent = nnx.clone(self.initial)

    def test_joint_update_reproducibility_and_parameter_changes(self):
        other = nnx.clone(self.agent)
        before = pure_state(self.agent)
        metrics = self.agent.update(self.batch)
        other.update(self.batch)
        self.assertTrue(equal(pure_state(self.agent), pure_state(other)))
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        for role in ("critic", "teacher", "actor"):
            self.assertFalse(equal(before[role], pure_state(self.agent)[role]))
        actions = self.agent.sample_action_chunk(
            self.batch["observations"], self.batch["actor_goals"], jax.random.PRNGKey(4)
        )
        self.assertEqual(actions.shape, (4, 2, 2))
        self.assertTrue(np.isfinite(actions).all() and (np.abs(actions) <= 1).all())

    def test_stage_ownership(self):
        before = pure_state(self.agent)
        self.agent.update(self.batch, stage="representation")
        for key in ("teacher", "actor"):
            self.assertTrue(equal(before[key], pure_state(self.agent)[key]))
        critic = nnx.to_pure_dict(nnx.state(self.agent.critic))
        self.agent.update(self.batch, stage="action")
        self.assertTrue(equal(critic, nnx.to_pure_dict(nnx.state(self.agent.critic))))
        self.assertEqual([int(o.step[...]) for o in self.agent.optimizers], [1, 1, 1])

    def test_nonfinite_update_is_atomic(self):
        before = pure_state(self.agent)
        batch = {
            **self.batch,
            "observations": np.full_like(self.batch["observations"], np.nan),
        }
        metrics = self.agent.update(batch)
        self.assertFalse(bool(metrics["numerics/finite"]))
        self.assertTrue(equal(before, pure_state(self.agent)))

    def test_loss_weight_and_gradient_scale(self):
        other = nnx.clone(self.agent)
        other.distillation_weight = self.agent.distillation_weight * 10
        low = self.agent.update(self.batch)
        high = other.update(self.batch)
        np.testing.assert_allclose(
            high["actor/distill_grad_norm"],
            10 * low["actor/distill_grad_norm"],
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            high["actor/q_grad_norm"], low["actor/q_grad_norm"], rtol=1e-5
        )
        np.testing.assert_allclose(
            low["actor/loss"],
            low["actor/q_loss"] + low["actor/weighted_distillation"],
            rtol=1e-5,
        )

    def test_mixer_sharing_exchange_and_order(self):
        net = TokenBackbone(
            6, 2, 2, width=7, depth=5, exchange_every=2, rngs=nnx.Rngs(3)
        )
        self.assertEqual(
            [b.exchange for b in net.blocks], [False, True, False, True, True]
        )
        self.assertEqual(net.blocks[0].ffn.layers[0].kernel.shape, (7, 28))
        self.assertEqual(net.blocks[1].token_mlp.layers[0].kernel.shape, (3, 6))
        s = jnp.ones((1, 6))
        a = jnp.array([[[0.2, -0.3], [0.7, 0.1]]])
        changed = a.at[:, 0].add(0.5)
        before = net.blocks[0].local(net.tokenize(s, a))
        after = net.blocks[0].local(net.tokenize(s, changed))
        np.testing.assert_array_equal(before[:, [0, 2]], after[:, [0, 2]])
        self.assertFalse(np.allclose(net(s, a)[:, 0], net(s, changed)[:, 0]))
        self.assertFalse(np.allclose(net(s, a)[:, 0], net(s, a[:, ::-1])[:, 0]))
        phi = lambda actions: self.agent.critic(
            self.batch["observations"], actions, self.batch["value_goals"]
        )[0].sum()
        grad = jax.grad(phi)(jnp.asarray(self.batch["action_chunks"]))
        self.assertTrue((np.linalg.norm(grad, axis=(0, 2)) > 0).all())

    def test_condition_is_detached_but_actor_uses_it(self):
        s, g = self.batch["observations"], self.batch["actor_goals"]
        grads = nnx.grad(lambda c: encode_condition(c, s, g).sum())(self.agent.critic)
        self.assertTrue(all((np.asarray(v) == 0).all() for v in jax.tree.leaves(grads)))
        embedding = encode_condition(self.agent.critic, s, g)
        args = (self.batch["action_chunks"], jnp.zeros(4))
        first = self.agent.actor(s, g, embedding, *args)
        second = self.agent.actor(s, g, jnp.zeros_like(embedding), *args)
        self.assertFalse(np.allclose(first, second))

    def test_other_shapes(self):
        config = {**self.config["agent"], "chunk_length": 1}
        agent = seeded_agent(7, 10, 3, config)
        batch = seeded_source(7, 1, observation_dim=10, action_dim=3)[0].sample(3)
        self.assertTrue(bool(agent.update(batch)["numerics/finite"]))
        self.assertEqual(
            agent.sample_action_chunk(
                batch["observations"][0], batch["actor_goals"][0], jax.random.PRNGKey(1)
            ).shape,
            (1, 3),
        )

    def test_checkpoint_exact_resume_and_incompatible_shape(self):
        ds, engine = seeded_source()
        self.agent.update(ds.sample(4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.msgpack"
            save_checkpoint(
                path,
                agent=self.agent,
                run_state={"engine": engine.numpy_state()},
            )
            restored = nnx.clone(self.initial)
            state = restore_checkpoint(path, agent=restored)
            other_ds, other_engine = seeded_source()
            other_engine.restore_numpy(state["engine"])
            b1, b2 = ds.sample(4), other_ds.sample(4)
            self.assertTrue(equal(b1, b2))
            self.agent.update(b1)
            restored.update(b2)
            self.assertTrue(equal(pure_state(self.agent), pure_state(restored)))
            mismatch = seeded_agent(0, 7, 2, self.config["agent"])
            before = pure_state(mismatch)
            with self.assertRaisesRegex(ValueError, "paths, shapes, or dtypes"):
                restore_checkpoint(path, agent=mismatch)
            self.assertTrue(equal(before, pure_state(mismatch)))
            broken = read_checkpoint(path)
            broken["run_state"] = "invalid JSON"
            snapshot = pure_state(restored)
            with self.assertRaises(ValueError):
                restore_checkpoint(path, agent=restored, payload=broken)
            self.assertTrue(equal(snapshot, pure_state(restored)))
            path.write_bytes(b"not a valid checkpoint")
            with self.assertRaises(ValueError):
                restore_checkpoint(path, agent=restored)

    def test_episode_boundaries_and_fingerprint(self):
        s = np.array([[0], [1], [2], [100], [101], [102]], np.float32)
        data = {
            "observations": s,
            "next_observations": s + 1,
            "actions": s / 200,
            "terminals": np.array([0, 0, 1, 0, 0, 1]),
        }
        ds = OfflineDataset(data, 2, rng=make_random_engines(0)[0].numpy())
        batch = ds.sample(200)
        for key in ("value_goals", "actor_goals"):
            np.testing.assert_array_equal(
                batch[key] // 100, batch["observations"] // 100
            )
        np.testing.assert_allclose(
            np.diff(batch["action_chunks"], axis=1), 1 / 200, atol=1e-7
        )
        first = ds.fingerprint()
        data["actions"][0] += 0.1
        self.assertNotEqual(first, ds.fingerprint())
        data["observations"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            OfflineDataset(data, 2, rng=make_random_engines(0)[0].numpy())

    def test_rollout_resets_and_rejects_invalid_actions(self):
        calls = []

        def act(s, g, rng):
            calls.append(np.asarray(jax.random.key_data(rng)))
            return np.zeros((2, 2))

        metrics = evaluate_rollouts(
            act,
            FakeEnv(),
            engine=make_random_engines(0)[1],
            episodes=2,
            execution_length=2,
        )
        self.assertEqual(len(calls), 4)
        self.assertEqual(metrics["evaluation/overall_success"], 1)
        with self.assertRaisesRegex(ValueError, "invalid/nonfinite"):
            evaluate_rollouts(
                lambda *a: np.full((2, 2), np.nan),
                FakeEnv(),
                engine=make_random_engines(0)[1],
            )
        with self.assertRaisesRegex(ValueError, "Invalid evaluation task"):
            evaluate_rollouts(
                act, FakeEnv(), engine=make_random_engines(0)[1], task_ids=[2]
            )
