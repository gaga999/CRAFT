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
from test_agent import (
    FakeEnv,
    assert_equivalent_training,
    equal,
    seeded_agent,
    seeded_source,
)

from craft.config import load_config
from craft.experiments import run_experiment
from craft.utils.checkpoint import pure_state, read_checkpoint, restore_checkpoint
from craft.utils.logging import Logger


def snapshot(path):
    return {
        str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()
    }


class ConfigTests(unittest.TestCase):
    def test_all_shipped_configs_load_with_defaults_where_unset(self):
        paths = sorted(Path("configs/experiment").glob("*.yaml"))
        self.assertEqual(
            [p.stem for p in paths],
            [
                "cube_double_play",
                "cube_single_noisy",
                "pointmaze_medium",
                "puzzle_3x3_noisy",
                "smoke",
            ],
        )
        for path in paths:
            config = load_config(path)
            stages = [s["name"] for s in config["train"]["stages"]]
            expected = (
                ["representation", "actor", "joint"]
                if path.stem == "smoke"
                else ["joint"]
            )
            self.assertEqual(stages, expected)
            self.assertIsNone(config["init"]["checkpoint"])
            agent = config["agent"]
            self.assertEqual(
                (
                    agent["bc_weight"],
                    agent["latent_noise"],
                    agent["guidance"],
                    agent["max_scale"],
                    agent["weight_decay"],
                    agent["actor_weight_decay"],
                ),
                (1.0, 0.2, "reencoded", 30.0, 0.01, 0.0),
            )

    def test_invalid_configs_fail_before_any_run(self):
        for override in [
            "agent.alpha=10",
            "agent.kl_weight=0.1",
            "agent.latent_dim=2",
            "agent.bc_weight=-1",
            "agent.latent_noise=-0.1",
            "agent.latent_noise=nan",
            "agent.guidance=direct",
            "agent.max_scale=1",
            "agent.weight_decay=-0.1",
            "agent.width=128",
            "agent.similarity=l2",
            "agent.actor_inputs=[goal]",
            "agent.actor_inputs=[goal_embedding,goal_embedding]",
            "agent.bc_weighting=awr",
            "agent.advantage_baseline=value",
            "agent.advantage_max_weight=0.5",
            "train.stages=[{name: action, steps: 1}]",
            "init.modules=[]",
            "init.modules=[decoder]",
            "init.modules=[actor,actor]",
            "agent.action_encoder_width=1.5",
            "agent.goal_encoder_depth=true",
            "agent.reconstruction_weight=nan",
            "agent.learning_rate=inf",
            "agent.head_depth=0",
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

    def test_init_from_checkpoint_starts_a_new_run_with_fresh_optimizers(self):
        with tempfile.TemporaryDirectory() as directory:
            source = run_experiment(self.config(directory), max_steps=2)
            before = snapshot(source)
            c = self.config(directory)
            c["output_dir"] = str(Path(directory) / "child")
            c["init"]["checkpoint"] = str(source / "checkpoint.msgpack")
            c["init"]["modules"] = ["representation"]
            c["train"]["stages"] = [{"name": "actor", "steps": 1}]
            child = run_experiment(c)
            self.assertEqual(before, snapshot(source))
            meta = json.loads((child / "metadata.json").read_text())
            self.assertEqual(meta["initialization"]["modules"], ["representation"])
            self.assertEqual(meta["initialization"]["source_step"], 2)
            parent, loaded = (
                seeded_agent(0, 6, 2, c["agent"]),
                seeded_agent(0, 6, 2, c["agent"]),
            )
            restore_checkpoint(source / "checkpoint.msgpack", agent=parent)
            restore_checkpoint(child / "checkpoint.msgpack", agent=loaded)
            # Actor-only training kept the loaded representation exactly.
            self.assertTrue(
                equal(
                    pure_state(parent)["representation"],
                    pure_state(loaded)["representation"],
                )
            )
            self.assertEqual(int(loaded.step[...]), 1)
            # A different dataset is rejected before any output is written.
            changed = seeded_source()[0].data
            changed["actions"][0] += 0.1
            c["output_dir"] = str(Path(directory) / "rejected")
            with (
                patch("craft.experiments.synthetic_dataset", return_value=changed),
                self.assertRaisesRegex(ValueError, "different dataset"),
            ):
                run_experiment(c)
            self.assertEqual(list(Path(directory).glob("rejected*")), [])

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
            invalid["agent"]["bc_weight"] = 3.0
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

    def test_chunked_training_matches_single_steps(self):
        from craft.trainers.offline import train_offline

        config = load_config("configs/experiment/smoke.yaml")["agent"]
        stages = [
            {"name": "representation", "steps": 3},
            {"name": "actor", "steps": 2},
            {"name": "joint", "steps": 5},
        ]
        results = {}
        for steps_per_call in (1, 4):
            ds, _ = seeded_source()
            agent = seeded_agent(0, 6, 2, config)
            observed = []
            # Resume mid-schedule to cover unaligned chunk starts.
            for max_steps in (2, None):
                train_offline(
                    agent,
                    ds,
                    stages,
                    batch_size=4,
                    max_steps=max_steps,
                    steps_per_call=steps_per_call,
                    on_step=lambda step, stage, m, seen=observed: seen.append(
                        (step, stage)
                    ),
                )
            results[steps_per_call] = (agent, ds.rng.bit_generator.state, observed)
        (single, single_rng, single_steps) = results[1]
        (chunked, chunked_rng, chunked_steps) = results[4]
        self.assertEqual(int(chunked.step[...]), 10)
        self.assertEqual(single_rng, chunked_rng)
        assert_equivalent_training(
            self, single, chunked, 10, config, seeded_source(1)[0].sample(8)
        )
        self.assertEqual(
            chunked_steps,
            [
                (2, "representation"),
                (3, "representation"),
                (4, "actor"),
                (5, "actor"),
                (8, "joint"),
                (10, "joint"),
            ],
        )
        self.assertTrue(set(chunked_steps) <= set(single_steps))
        with self.assertRaises(ValueError):
            train_offline(chunked, ds, stages, batch_size=4, steps_per_call=0)

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

    def test_wandb_display_name_defaults_to_session_and_can_be_set(self):
        import sys
        import types

        calls = []
        fake = types.SimpleNamespace(
            init=lambda **kw: (
                calls.append(kw)
                or types.SimpleNamespace(log=lambda *a, **k: None, finish=lambda: None)
            )
        )
        c = load_config("configs/experiment/smoke.yaml", ["logging.backend=wandb"])
        self.assertIsNone(c["logging"]["name"])
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"wandb": fake}),
        ):
            for name, expected in ((None, "session-a"), ("bc0.1-awbc", "bc0.1-awbc")):
                session = Path(directory) / f"session-{len(calls) and 'b' or 'a'}"
                session.mkdir()
                c["logging"]["name"] = name
                Logger(session, c).close()
                self.assertEqual(calls[-1]["name"], expected)
        with self.assertRaises(ValueError):
            load_config("configs/experiment/smoke.yaml", ["logging.name=''"])
        with (
            patch(
                "sys.argv",
                [
                    "craft",
                    "--config",
                    "configs/experiment/smoke.yaml",
                    "--dry-run",
                    "--name",
                    "0.1",
                ],
            ),
            patch("builtins.print") as printed,
        ):
            from craft.cli import run

            run()
        self.assertIn("name: '0.1'", printed.call_args[0][0])

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
