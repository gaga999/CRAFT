"""Numerical, ownership, sampling, persistence, and rollout contract checks."""

import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from craft.agents.craft import CRAFTAgent
from craft.data.offline import OfflineDataset, synthetic_dataset
from craft.evaluators.rollout import evaluate_rollouts
from craft.models.tokens import TokenBackbone
from craft.utils.checkpoint import restore_checkpoint, save_checkpoint


def equal(a, b):
    return all(
        np.array_equal(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))
    )


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = OmegaConf.to_container(
            OmegaConf.load("configs/experiment/smoke.yaml"), resolve=True
        )
        cls.agent = CRAFTAgent.create(0, 6, 2, cls.config["agent"])
        cls.dataset = OfflineDataset(synthetic_dataset(), 2)
        cls.batch = cls.dataset.sample(4)

    def test_joint_update_and_reproducibility(self):
        a, metrics = self.agent.update(self.batch)
        b, _ = self.agent.update(self.batch)
        self.assertTrue(equal(a, b))
        self.assertTrue(all(np.isfinite(x) for x in metrics.values()))
        for before, after in [
            (self.agent.critic, a.critic),
            (self.agent.teacher, a.teacher),
            (self.agent.actor, a.actor),
        ]:
            self.assertFalse(equal(before.params, after.params))
        action = a.sample_action_chunk(
            self.batch["observations"], self.batch["actor_goals"], jax.random.PRNGKey(1)
        )
        self.assertEqual(action.shape, (4, 2, 2))
        self.assertTrue(np.all(np.isfinite(action)))
        self.assertTrue(np.all(np.abs(action) <= 1))

    def test_other_dimensions_and_single_action(self):
        config = {**self.config["agent"], "chunk_length": 1}
        agent = CRAFTAgent.create(7, 10, 3, config)
        data = OfflineDataset(
            synthetic_dataset(7, observation_dim=10, action_dim=3), 1, 7
        )
        batch = data.sample(3)
        agent, info = agent.update(batch)
        self.assertTrue(all(np.isfinite(value) for value in info.values()))
        actions = agent.sample_action_chunk(
            batch["observations"][0], batch["actor_goals"][0], jax.random.PRNGKey(2)
        )
        self.assertEqual(actions.shape, (1, 3))

    def test_stage_ownership(self):
        rep, _ = self.agent.update(self.batch, stage="representation")
        self.assertTrue(equal(rep.actor, self.agent.actor))
        self.assertTrue(equal(rep.teacher, self.agent.teacher))
        act, _ = self.agent.update(self.batch, stage="action")
        self.assertTrue(equal(act.critic, self.agent.critic))

    def test_tokens_exchange_and_order(self):
        c = self.agent.critic
        b = self.batch
        actions = jnp.asarray(b["action_chunks"])
        phi, _ = c.apply_fn(
            {"params": c.params}, b["observations"], actions, b["value_goals"]
        )
        reversed_phi, _ = c.apply_fn(
            {"params": c.params}, b["observations"], actions[:, ::-1], b["value_goals"]
        )
        self.assertFalse(np.allclose(phi, reversed_phi))
        jac = jax.jacrev(
            lambda x: c.apply_fn(
                {"params": c.params}, b["observations"], x, b["value_goals"]
            )[0].sum()
        )(actions)
        self.assertTrue(np.all(np.linalg.norm(np.asarray(jac), axis=(0, 2)) > 0))
        embedding = self.agent.embeddings(b["observations"], b["actor_goals"])

        def actor_output(a):
            return self.agent.actor.apply_fn(
                {"params": self.agent.actor.params},
                b["observations"],
                b["actor_goals"],
                embedding,
                a,
                jnp.zeros(4),
            )

        changed = actions.at[:, 0].add(0.5)
        self.assertFalse(
            np.allclose(actor_output(actions)[:, 1], actor_output(changed)[:, 1])
        )

    def test_shared_ffn_and_periodic_exchange(self):
        # Odd width also verifies removal of attention's head-divisibility rule.
        net = TokenBackbone(width=7, depth=5, exchange_every=2)
        state = jnp.ones((1, 6))
        actions = jnp.asarray([[[0.2, -0.3], [0.7, 0.1]]])
        variables = net.init(jax.random.PRNGKey(5), state, actions)
        params = variables["params"]
        exchanges = {key for key in params if key.startswith("token_contract_")}
        self.assertEqual(
            exchanges, {"token_contract_1", "token_contract_3", "token_contract_4"}
        )
        self.assertEqual(params["expand_0"]["kernel"].shape, (7, 28))
        self.assertEqual(params["token_expand_1"]["kernel"].shape, (3, 6))

        def apply(a):
            return net.apply(
                variables,
                state,
                a,
                capture_intermediates=lambda module, method: (
                    module.name == "contract_0"
                ),
                mutable=["intermediates"],
            )

        first, features = apply(actions)
        second, changed = apply(actions.at[:, 0].add(0.5))
        before = features["intermediates"]["contract_0"]["__call__"][0]
        after = changed["intermediates"]["contract_0"]["__call__"][0]
        # Before exchange, changes stay local to action token 0 (position 1).
        np.testing.assert_array_equal(before[:, [0, 2]], after[:, [0, 2]])
        self.assertFalse(np.allclose(before[:, 1], after[:, 1]))
        # After exchange, state and other action tokens receive the information.
        self.assertFalse(np.allclose(first[:, 0], second[:, 0]))
        self.assertFalse(np.allclose(first[:, 2], second[:, 2]))

    def test_embedding_is_detached(self):
        def loss(p):
            a = self.agent.replace(critic=self.agent.critic.replace(params=p))
            return a.embeddings(
                self.batch["observations"], self.batch["actor_goals"]
            ).sum()

        grads = jax.grad(loss)(self.agent.critic.params)
        self.assertTrue(all(np.all(np.asarray(g) == 0) for g in jax.tree.leaves(grads)))

    def test_resume_exact_next_update_and_sampler(self):
        a, _ = self.agent.update(self.batch)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "checkpoint.msgpack"
            save_checkpoint(
                path,
                agent=a,
                run_state={"step": 1, "rng": self.dataset.rng.bit_generator.state},
            )
            restored, state = restore_checkpoint(path, agent=self.agent)
            self.assertTrue(equal(a, restored))
            data = OfflineDataset(synthetic_dataset(), 2)
            data.rng.bit_generator.state = state["rng"]
            first, second = self.dataset.sample(4), data.sample(4)
            self.assertTrue(equal(first, second))
            next_a, _ = a.update(first)
            next_b, _ = restored.update(second)
            self.assertTrue(equal(next_a, next_b))

    def test_no_cross_episode_chunks_or_goals(self):
        s = np.array([[0], [1], [2], [100], [101], [102]], dtype=np.float32)
        data = {
            "observations": s,
            "next_observations": s + 1,
            "actions": s / 200,
            "terminals": np.array([0, 0, 1, 0, 0, 1]),
        }
        ds = OfflineDataset(data, 2)
        b = ds.sample(200)
        for k in ["value_goals", "actor_goals"]:
            self.assertTrue(np.all(b[k] // 100 == b["observations"] // 100))
            self.assertTrue(np.all(b[k] > b["observations"]))
        self.assertTrue(np.allclose(np.diff(b["action_chunks"], axis=1), 1 / 200))

    def test_rollout_cache_resets(self):
        class Env:
            def __init__(self):
                self.task_infos = [{"task_name": "test"}]

            @property
            def unwrapped(self):
                return self

            def reset(self, **kwargs):
                self.steps = 0
                return np.zeros(6), {"goal": np.ones(6)}

            def step(self, action):
                self.steps += 1
                return (
                    np.zeros(6),
                    0,
                    False,
                    self.steps == 3,
                    {"success": self.steps == 2},
                )

        calls = []

        def act(s, g, rng):
            calls.append(np.asarray(rng))
            return np.zeros((2, 2))

        metrics = evaluate_rollouts(act, Env(), episodes=2, execution_length=2)
        self.assertEqual(len(calls), 4)
        self.assertEqual(metrics["evaluation/overall_success"], 1)
        self.assertFalse(np.array_equal(calls[0], calls[1]))


if __name__ == "__main__":
    unittest.main()
