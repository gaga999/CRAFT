"""Regression tests for config validation and scientific run isolation."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
from flax import nnx
from omegaconf.errors import OmegaConfBaseException
from test_baseline import FakeEnv, equal, seeded_agent, seeded_source

from craft.config import load_config
from craft.experiments import run_experiment
from craft.utils.checkpoint import pure_state, read_checkpoint, restore_checkpoint
from craft.utils.logging import Logger


def snapshot(path):
    return {
        str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()
    }


class ConfigTests(unittest.TestCase):
    def test_all_shipped_configs_and_coefficient(self):
        for path in Path("configs/experiment").glob("*.yaml"):
            config = load_config(path)
            self.assertEqual(config["agent"]["guide_weight"], 0.1)
            self.assertEqual(config["data"]["discount"], 0.95)

    def test_invalid_configs_fail_before_any_run(self):
        for override in [
            "agent.alpha=10",
            "agent.width=1.5",
            "agent.width=true",
            "agent.kl_weight=nan",
            "agent.learning_rate=inf",
            "agent.width=0",
            "data.discount=1",
            "logging.backend=typo",
            "train.stages=[]",
            "evaluation.execution_length=99",
            "seed=-1",
            "data.batch_size=1",
            "data.synthetic=1",
        ]:
            with (
                self.subTest(override=override),
                self.assertRaises((ValueError, OmegaConfBaseException)),
            ):
                load_config("configs/experiment/smoke.yaml", [override])


class RuntimeTests(unittest.TestCase):
    def config(self, directory):
        config = load_config("configs/experiment/smoke.yaml")
        config["output_dir"] = str(Path(directory) / "run")
        return config

    def test_fresh_run_collision_resume_and_metadata_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            c = self.config(directory)
            run = run_experiment(c, max_steps=1)
            before = snapshot(run)
            repeated = run_experiment(c, max_steps=1)
            self.assertNotEqual(run, repeated)
            self.assertEqual(before, snapshot(run))
            c = load_config(run / "config.yaml")
            resumed = run_experiment(c, checkpoint=run / "checkpoint.msgpack")
            for path, content in before.items():
                self.assertEqual((run / path).read_bytes(), content)
            self.assertEqual(
                json.loads((resumed / "status.json").read_text())["step"], 3
            )
            # Compare resumed result against an uninterrupted run, including optimizers.
            c2 = deepcopy(c)
            c2["output_dir"] = str(Path(directory) / "whole")
            full = run_experiment(c2)
            a = seeded_agent(0, 6, 2, c["agent"])
            b = nnx.clone(a)
            restore_checkpoint(resumed / "checkpoint.msgpack", agent=a)
            restore_checkpoint(full / "checkpoint.msgpack", agent=b)
            self.assertTrue(equal(pure_state(a), pure_state(b)))
            rows = [
                json.loads(line)
                for line in (resumed / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["step"] for row in rows], [2, 3])

    def test_eval_source_immutable_and_no_dataset_load(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            run = run_experiment(config, max_steps=1)
            # A fake real adapter avoids downloading while exercising production orchestration.
            config = load_config(run / "config.yaml")
            config["data"]["synthetic"] = False
            saved = read_checkpoint(run / "checkpoint.msgpack")
            meta = json.loads(saved["run_state"])
            meta["config"] = config
            from craft.utils.checkpoint import save_checkpoint

            agent = seeded_agent(0, 6, 2, config["agent"])
            restore_checkpoint(run / "checkpoint.msgpack", agent=agent)
            save_checkpoint(run / "checkpoint.msgpack", agent=agent, run_state=meta)
            before = snapshot(run)
            env = FakeEnv()
            with (
                patch("craft.experiments.make_env", return_value=env),
                patch(
                    "craft.experiments.make_env_and_datasets",
                    side_effect=AssertionError("evaluation loaded data"),
                ),
            ):
                destination = run_experiment(
                    config, checkpoint=run / "checkpoint.msgpack", evaluate=True
                )
            self.assertTrue(env.closed)
            self.assertEqual(before, snapshot(run))
            self.assertNotEqual(run, destination)
            self.assertTrue((destination / "metadata.json").is_file())
            with (
                patch("craft.experiments.make_env", return_value=FakeEnv()),
                self.assertRaisesRegex(ValueError, "outside"),
            ):
                run_experiment(
                    config,
                    checkpoint=run / "checkpoint.msgpack",
                    evaluate=True,
                    destination=run / "eval",
                )
            self.assertEqual(before, snapshot(run))

    def test_resume_rejects_dataset_and_config_changes_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            c = self.config(directory)
            run = run_experiment(c, max_steps=1)
            c = load_config(run / "config.yaml")
            before = snapshot(run)
            invalid = deepcopy(c)
            invalid["agent"]["guide_weight"] = 10
            with self.assertRaisesRegex(ValueError, "mismatch"):
                run_experiment(invalid, checkpoint=run / "checkpoint.msgpack")
            changed = seeded_source()[0].data
            changed["actions"][0] += 0.1
            with (
                patch("craft.experiments.synthetic_dataset", return_value=changed),
                self.assertRaisesRegex(ValueError, "Dataset contents"),
            ):
                run_experiment(c, checkpoint=run / "checkpoint.msgpack")
            self.assertEqual(before, snapshot(run))

    def test_interrupt_checkpoint_and_sampler_rollback(self):
        from craft.trainers.offline import train_offline

        ds, _ = seeded_source()
        agent = seeded_agent(
            0, 6, 2, load_config("configs/experiment/smoke.yaml")["agent"]
        )
        previous = deepcopy(ds.rng.bit_generator.state)
        bad = {**ds.data, "actions": np.full_like(ds.data["actions"], np.nan)}
        ds.data = bad
        with self.assertRaises(FloatingPointError):
            train_offline(agent, ds, [{"name": "joint", "steps": 1}], batch_size=4)
        self.assertEqual(previous, ds.rng.bit_generator.state)
        self.assertEqual(int(agent.step[...]), 0)
        with tempfile.TemporaryDirectory() as directory:
            c = self.config(directory)

            def interrupted(agent, data, stages, **kwargs):
                train_offline(
                    agent,
                    data,
                    stages,
                    batch_size=4,
                    max_steps=1,
                    on_step=kwargs["on_step"],
                )
                raise KeyboardInterrupt

            with (
                patch("craft.experiments.train_offline", side_effect=interrupted),
                self.assertRaises(KeyboardInterrupt),
            ):
                run_experiment(c)
            actual = next(Path(directory).glob("run-*"))
            status = json.loads((actual / "status.json").read_text())
            self.assertEqual(status, {"status": "interrupted", "step": 1})
            self.assertTrue((actual / "checkpoint.msgpack").is_file())

    def test_logger_rejects_duplicate_steps_and_append(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = load_config("configs/experiment/smoke.yaml")
            logger = Logger(root, c)
            try:
                logger.log({"metric": 1.0}, step=1)
                with self.assertRaises(ValueError):
                    logger.log({"metric": 2.0}, step=1)
                with self.assertRaises(ValueError):
                    logger.log({"metric": float("nan")}, step=2)
            finally:
                logger.close()
            with self.assertRaises(FileExistsError):
                Logger(root, c)
