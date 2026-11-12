"""Regression tests for RNG isolation and recovery from saturated student outputs."""

import ast
import json
import random
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import jax
import numpy as np
from flax import nnx, serialization
from test_baseline import FakeEnv, equal, seeded_agent, seeded_source

from craft.agents.craft import CRAFTAgent
from craft.config import load_config
from craft.evaluators.rollout import evaluate_rollouts
from craft.utils.checkpoint import (
    pure_state,
    read_checkpoint,
)
from craft.utils.random import make_random_engines


class EngineTests(unittest.TestCase):
    def test_seeded_globals_and_independent_streams(self):
        train, evaluation = make_random_engines(17)
        expected_legacy = (np.random.random(), random.random())
        before = train.numpy_state()
        # Consume every evaluation stream and validation; training data must not move.
        evaluation.numpy().normal(size=20)
        evaluation.jax.updates()
        train.numpy("validation").normal(size=10)
        self.assertEqual(
            before["streams"]["data"], train.numpy_state()["streams"]["data"]
        )
        other, other_eval = make_random_engines(17)
        self.assertEqual(expected_legacy, (np.random.random(), random.random()))
        np.testing.assert_array_equal(
            train.numpy().normal(size=10), other.numpy().normal(size=10)
        )
        np.testing.assert_array_equal(
            jax.random.key_data(train.jax.updates()),
            jax.random.key_data(other.jax.updates()),
        )
        self.assertFalse(
            np.array_equal(
                jax.random.key_data(train.task_key(1)),
                jax.random.key_data(evaluation.task_key(1)),
            )
        )
        self.assertFalse(
            np.array_equal(
                jax.random.key_data(other.task_key(1)),
                jax.random.key_data(make_random_engines(18)[0].task_key(1)),
            )
        )
        self.assertNotEqual(train.episode_seed(1, 0), evaluation.episode_seed(1, 0))
        self.assertEqual(evaluation.episode_seed(1, 0), other_eval.episode_seed(1, 0))

    def test_seed_boundary_validation(self):
        for seed in (True, -1, 2**32, 0.5, "0"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                make_random_engines(seed)
        config = load_config(
            "configs/experiment/cube_single_noisy.yaml", [f"seed={2**32 - 1}"]
        )
        first, evaluation = make_random_engines(config["seed"])
        second, _ = make_random_engines(config["seed"])
        np.testing.assert_array_equal(
            first.numpy().normal(size=4), second.numpy().normal(size=4)
        )
        np.testing.assert_array_equal(
            jax.random.key_data(first.jax.updates()),
            jax.random.key_data(second.jax.updates()),
        )
        self.assertTrue(0 <= evaluation.episode_seed(1, 0) < 2**32)

    def test_numpy_restore_is_atomic_and_keeps_live_references(self):
        train, evaluation = make_random_engines(3)
        rng = train.numpy()
        state = train.numpy_state()
        first = rng.normal(size=4)
        # Real metadata round-trip rather than an in-memory state copy.
        train.restore_numpy(json.loads(json.dumps(state)))
        self.assertIs(rng, train.numpy())
        np.testing.assert_array_equal(first, rng.normal(size=4))
        before = train.numpy_state()
        invalid = deepcopy(before)
        invalid["streams"]["validation"]["bit_generator"] = "MT19937"
        with self.assertRaises(ValueError):
            train.restore_numpy(invalid)
        self.assertEqual(before, train.numpy_state())
        with self.assertRaises(ValueError):
            evaluation.restore_numpy(before)

    def test_factory_uses_live_engine_and_distinct_initializations(self):
        train, _ = make_random_engines(8)
        config = load_config("configs/experiment/smoke.yaml")["agent"]
        agent = CRAFTAgent.create(6, 2, config, rngs=train.jax)
        self.assertIs(agent.rngs, train.jax)
        self.assertGreater(int(train.jax.params.count[...]), 0)
        self.assertEqual(int(train.jax.updates.count[...]), 0)
        self.assertFalse(
            equal(pure_state(agent)["posterior"], pure_state(agent)["prior"])
        )
        fresh = seeded_agent(8, 6, 2, config)
        self.assertTrue(equal(pure_state(agent), pure_state(fresh)))

    def test_key_and_generator_construction_are_centralized(self):
        prohibited = {"PRNGKey", "key", "default_rng", "Rngs", "SeedSequence"}
        for path in Path("src/craft").rglob("*.py"):
            if path == Path("src/craft/utils/random.py"):
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(
                        node.func.attr,
                        prohibited,
                        f"{path}:{node.lineno} creates its own RNG",
                    )

    def test_evaluation_order_and_cadence_do_not_change_training(self):
        train, evaluation = make_random_engines(9)

        class Env(FakeEnv):
            task_infos = ({"task_name": "one"}, {"task_name": "two"})

            def reset(self, **kwargs):
                self.task = kwargs["options"]["task_id"]
                return super().reset(**kwargs)

        def rollout(tasks):
            env = Env()
            calls = {1: [], 2: []}

            def act(s, g, key):
                calls[env.task].append(np.asarray(jax.random.key_data(key)))
                return np.zeros((2, 2))

            evaluate_rollouts(
                act,
                env,
                engine=evaluation,
                episodes=2,
                execution_length=2,
                task_ids=tasks,
            )
            return calls

        before = train.numpy_state()
        count = int(train.jax.updates.count[...])
        first, reordered, repeated = rollout([1, 2]), rollout([2, 1]), rollout([1, 2])
        for task in (1, 2):
            np.testing.assert_array_equal(first[task], reordered[task])
            np.testing.assert_array_equal(first[task], repeated[task])
        self.assertEqual(before, train.numpy_state())
        self.assertEqual(count, int(train.jax.updates.count[...]))
        # Extra validation/evaluation also leave the next actual parameter update identical.
        config = load_config("configs/experiment/smoke.yaml")["agent"]
        agent = CRAFTAgent.create(6, 2, config, rngs=train.jax)
        other = nnx.clone(agent)
        source, _ = seeded_source(9)
        batch = source.sample(4)
        rollout([1])
        train.numpy("validation").normal(size=100)
        agent.update(batch)
        other.update(batch)
        self.assertTrue(equal(pure_state(agent), pure_state(other)))

    def test_old_checkpoint_format_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.msgpack"
            path.write_bytes(
                serialization.msgpack_serialize(
                    {"format_version": 2, "agent": {}, "run_state": "{}"}
                )
            )
            with self.assertRaisesRegex(ValueError, "expected v5"):
                read_checkpoint(path)
