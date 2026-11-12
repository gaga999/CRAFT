"""Data-only fixtures and fake updates: no experiment runs or training."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from threading import Barrier, Event, Lock, current_thread
from threading import enumerate as threads
from unittest.mock import patch

import jax
import numpy as np
from ogbench.utils import load_dataset

from craft.config import load_config
from craft.data.loading import load_ogbench_datasets, load_split
from craft.data.offline import OfflineDataset, synthetic_dataset
from craft.data.prefetch import PrefetchBatches
from craft.envs.factory import make_env_and_datasets
from craft.experiments import check_restore
from craft.trainers.offline import train_offline
from craft.utils.random import make_random_engines


def source(seed=0):
    engine, _ = make_random_engines(seed)
    return OfflineDataset(
        synthetic_dataset(rng=engine.numpy("synthetic")), 5, rng=engine.numpy()
    )


def assert_batches(test, actual, expected):
    test.assertEqual(set(actual), set(expected))
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])


class CompactTests(unittest.TestCase):
    def raw(self):
        # Include a too-short episode and terminal-only dummy action slots.
        observations, actions, terminals = [], [], []
        for episode, length in enumerate((2, 5, 8)):
            observations.extend([[episode, step] for step in range(length + 1)])
            actions.extend([[step / 10] for step in range(length)] + [[np.nan]])
            terminals.extend([False] * length + [True])
        return {
            "observations": np.array(observations, np.float32),
            "actions": np.array(actions, np.float32),
            "terminals": np.array(terminals),
        }

    def test_compact_matches_official_transitions_samples_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.npz"
            np.savez(path, **self.raw())
            a, _ = make_random_engines(4)
            b, _ = make_random_engines(4)
            regular_data = load_dataset(path)
            regular = OfflineDataset(regular_data, 3, rng=a.numpy())
            compact = load_split(path, 3, 0.95, b.numpy())
            self.assertNotIn("next_observations", compact.data)
            self.assertEqual(compact.fingerprint(), regular.fingerprint())
            self.assertLess(
                sum(x.nbytes for x in compact.data.values()),
                sum(x.nbytes for x in regular.data.values()),
            )
            path.unlink()
            for _ in range(20):
                actual, expected = compact.sample(16), regular.sample(16)
                assert_batches(self, actual, expected)
                # No action chunk crosses an episode; short episodes excluded.
                self.assertTrue((actual["observations"][:, 0] > 0).all())
                self.assertTrue(np.isfinite(actual["action_chunks"]).all())

    def test_regular_input_is_cast_once_and_uses_existing_float32_buffers(self):
        ds = source()
        same = OfflineDataset(ds.data, 5, rng=deepcopy(ds.rng))
        for key in ds.data:
            self.assertIs(same.data[key], ds.data[key])
        float64 = {k: v.astype(np.float64) for k, v in ds.data.items()}
        converted = OfflineDataset(float64, 5, rng=deepcopy(ds.rng))
        self.assertTrue(all(v.dtype == np.float32 for v in converted.data.values()))
        assert_batches(self, converted.sample(8), same.sample(8))

    def test_compact_rejects_invalid_real_actions_and_missing_boundaries(self):
        for kind in ("nonfinite", "bounds", "boundary", "binary"):
            with self.subTest(kind=kind):
                raw = self.raw()
                if kind == "nonfinite":
                    raw["actions"][0] = np.nan
                elif kind == "bounds":
                    raw["actions"][0] = 2
                elif kind == "boundary":
                    raw["terminals"][-1] = False
                else:
                    raw["terminals"] = raw["terminals"].astype(np.float32)
                    raw["terminals"][0] = 0.5
                with self.assertRaises(ValueError):
                    OfflineDataset(raw, 3, rng=source().rng, compact=True)

    def test_adapter_loads_both_splits_and_closes_env_on_load_failure(self):
        class Env:
            closed = False

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            train = Path(directory) / "fixture.npz"
            val = Path(directory) / "fixture-val.npz"
            np.savez(train, **self.raw())
            np.savez(val, **self.raw())
            engine, _ = make_random_engines(0)
            with patch(
                "ogbench.utils.download_datasets",
                side_effect=AssertionError("download"),
            ):
                a, b = load_ogbench_datasets(
                    "fixture-v0",
                    directory=directory,
                    path=train,
                    chunk_length=3,
                    discount=0.95,
                    rng=engine.numpy(),
                    validation_rng=engine.numpy("validation"),
                )
            self.assertIsNot(a.rng, b.rng)
            self.assertEqual(a.fingerprint(), b.fingerprint())
            env = Env()
            with (
                patch("craft.envs.factory.make_env", return_value=env),
                self.assertRaises(FileNotFoundError),
            ):
                make_env_and_datasets(
                    "fixture-v0",
                    chunk_length=3,
                    rng=a.rng,
                    validation_rng=b.rng,
                    dataset_path=Path(directory) / "missing.npz",
                )
            self.assertTrue(env.closed)


class PrefetchTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(
            any(
                t.name.startswith(("craft-prefetch", "craft-sample", "craft-load"))
                for t in threads()
            )
        )

    def test_order_and_live_rng_match_synchronous_for_every_worker_count(self):
        for workers in (0, 1, 3):
            with self.subTest(workers=workers):
                ds, reference = source(2), source(2)
                before = deepcopy(ds.rng.bit_generator.state)
                with PrefetchBatches(
                    ds, 8, 6, workers=workers, prefetch_batches=4
                ) as batches:
                    for _ in range(6):
                        actual = next(batches)
                        self.assertEqual(ds.rng.bit_generator.state, before)
                        self.assertTrue(
                            all(isinstance(x, jax.Array) for x in actual.values())
                        )
                        assert_batches(
                            self, actual, reference.sample(8, include_actor_goals=False)
                        )
                        batches.accept()
                        before = deepcopy(reference.rng.bit_generator.state)
                        self.assertEqual(ds.rng.bit_generator.state, before)
                    with self.assertRaises(StopIteration):
                        next(batches)

    def test_chunks_stack_sequential_batches_and_accept_as_one_unit(self):
        for workers in (0, 2):
            with self.subTest(workers=workers):
                ds, reference = source(3), source(3)
                before = deepcopy(ds.rng.bit_generator.state)
                with PrefetchBatches(ds, 8, 5, workers=workers) as batches:
                    chunk = batches.next_chunk(3)
                    self.assertEqual(ds.rng.bit_generator.state, before)
                    with self.assertRaises(RuntimeError):
                        batches.next_chunk(1)
                    for i in range(3):
                        assert_batches(
                            self,
                            {k: v[i] for k, v in chunk.items()},
                            reference.sample(8, include_actor_goals=False),
                        )
                    batches.accept()
                    self.assertEqual(
                        ds.rng.bit_generator.state, reference.rng.bit_generator.state
                    )
                    with self.assertRaises(StopIteration):
                        batches.next_chunk(3)
                    with self.assertRaises(ValueError):
                        batches.next_chunk(0)
                    batches.next_chunk(2)
                    batches.accept()

    def test_unaccepted_batch_and_lookahead_regenerate_after_json_restore(self):
        ds, reference = source(5), source(5)
        with PrefetchBatches(ds, 8, 50, workers=3) as batches:
            next(batches)
            batches.accept()
            reference.sample(8)
            saved = json.loads(json.dumps(ds.rng.bit_generator.state))
            unaccepted = next(batches)
        ds.rng.bit_generator.state = saved
        with PrefetchBatches(ds, 8, 1, workers=1) as resumed:
            batch = next(resumed)
            assert_batches(self, batch, unaccepted)
            assert_batches(self, batch, reference.sample(8, include_actor_goals=False))
            resumed.accept()
        self.assertEqual(ds.rng.bit_generator.state, reference.rng.bit_generator.state)

    def test_workers_actually_gather_concurrently(self):
        ds = source()
        gather = ds.gather
        barrier, lock, names = Barrier(2, timeout=5), Lock(), set()

        def concurrent(indices, **kwargs):
            with lock:
                names.add(current_thread().name)
            barrier.wait()
            return gather(indices, **kwargs)

        with (
            patch.object(ds, "gather", side_effect=concurrent),
            PrefetchBatches(ds, 8, 2, workers=2, prefetch_batches=2) as batches,
        ):
            next(batches)
            batches.accept()
            next(batches)
            batches.accept()
        self.assertEqual(len(names), 2)

    def test_worker_failure_propagates_without_consuming_rng(self):
        ds = source()
        before = deepcopy(ds.rng.bit_generator.state)
        with (
            patch.object(ds, "gather", side_effect=ValueError("bad gather")),
            PrefetchBatches(ds, 8, 10) as batches,
            self.assertRaisesRegex(ValueError, "bad gather"),
        ):
            next(batches)
        self.assertEqual(ds.rng.bit_generator.state, before)

    def test_full_queue_closes_without_deadlock_or_rng_advance(self):
        ds = source()
        ready = Event()
        gather = ds.gather

        def signal(indices, **kwargs):
            ready.set()
            return gather(indices, **kwargs)

        before = deepcopy(ds.rng.bit_generator.state)
        with (
            patch.object(ds, "gather", side_effect=signal),
            PrefetchBatches(ds, 8, 1000, workers=2, device_prefetch=1),
        ):
            self.assertTrue(ready.wait(5))
        self.assertEqual(ds.rng.bit_generator.state, before)


class FakeAgent:
    def __init__(self, *, fail=False, interrupt=False):
        self.step = np.array(0)
        self.fail, self.interrupt = fail, interrupt
        self.calls = []

    def update(self, batch, *, stage):
        self.calls.append((stage, jax.device_get(batch)))
        if self.fail:
            return {"numerics/finite": False}
        self.step[...] += 1
        if self.interrupt:
            raise KeyboardInterrupt
        return {"numerics/finite": True, "loss": 0.0}


class SchedulingTests(unittest.TestCase):
    def test_fake_updates_preserve_stage_order_and_resume_batches(self):
        ds, reference, agent = source(), source(), FakeAgent()
        stages = [
            {"name": "representation", "steps": 2},
            {"name": "action", "steps": 3},
        ]
        observed = []
        train_offline(
            agent,
            ds,
            stages,
            batch_size=8,
            max_steps=3,
            on_step=lambda step, stage, metrics: observed.append((step, metrics)),
        )
        train_offline(agent, ds, stages, batch_size=8, workers=1)
        self.assertEqual(
            [s for s, _ in agent.calls], ["representation"] * 2 + ["action"] * 3
        )
        for _, batch in agent.calls:
            assert_batches(self, batch, reference.sample(8, include_actor_goals=False))
        self.assertEqual(ds.rng.bit_generator.state, reference.rng.bit_generator.state)
        self.assertEqual([s for s, _ in observed], [1, 2, 3])
        self.assertTrue(all(m["time/data_wait_seconds"] >= 0 for _, m in observed))

    def test_rejection_keeps_rng_but_interrupt_after_commit_accepts_it(self):
        ds, reference = source(), source()
        before = deepcopy(ds.rng.bit_generator.state)
        stages = [{"name": "joint", "steps": 1}]
        with self.assertRaises(FloatingPointError):
            train_offline(FakeAgent(fail=True), ds, stages, batch_size=8)
        self.assertEqual(before, ds.rng.bit_generator.state)
        with self.assertRaises(KeyboardInterrupt):
            train_offline(FakeAgent(interrupt=True), ds, stages, batch_size=8)
        reference.sample(8)
        self.assertEqual(reference.rng.bit_generator.state, ds.rng.bit_generator.state)

    def test_prefetch_config_validation_and_restore_contract(self):
        for path in Path("configs/experiment").glob("*.yaml"):
            config = load_config(path)
            self.assertGreaterEqual(config["data"]["workers"], 0)
        for override in (
            "data.workers=-1",
            "data.workers=true",
            "data.prefetch_batches=0",
            "data.device_prefetch=0",
        ):
            with self.assertRaises(ValueError):
                load_config("configs/experiment/smoke.yaml", [override])
        config = load_config("configs/experiment/smoke.yaml")
        saved = {"step": 0, "config": deepcopy(config)}
        for key in ("workers", "prefetch_batches", "device_prefetch"):
            del saved["config"]["data"][key]
        check_restore(config, saved, evaluate=False)
        config["data"]["workers"] = 4
        check_restore(config, saved, evaluate=False)
        config["data"]["discount"] = 0.9
        with self.assertRaisesRegex(ValueError, "mismatch"):
            check_restore(config, saved, evaluate=False)
