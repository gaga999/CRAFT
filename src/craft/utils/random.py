"""Entry-point-owned training/evaluation engines with fixed independent streams."""

import random
from copy import deepcopy
from dataclasses import dataclass

import jax
import numpy as np
from flax import nnx

_NUMPY_STREAMS = {"data": 100, "validation": 101, "synthetic": 102}


@dataclass
class RandomEngine:
    seed: int
    role: int
    jax: nnx.Rngs
    _numpy: dict[str, np.random.Generator]

    def numpy(self, stream="data"):
        """Return a persistent Generator; never reseed on sampling calls."""
        return self._numpy[stream]

    def task_key(self, task_id):
        """Repeatable evaluation per task, independent of task ordering/cadence."""
        return jax.random.fold_in(self.jax.evaluation.key[...], task_id)

    def episode_seed(self, task_id, episode):
        sequence = np.random.SeedSequence([self.seed, self.role, 200, task_id, episode])
        return int(sequence.generate_state(1, dtype=np.uint32)[0])

    def numpy_state(self):
        return {
            "version": 1,
            "seed": self.seed,
            "role": self.role,
            "streams": {
                name: deepcopy(rng.bit_generator.state)
                for name, rng in self._numpy.items()
            },
        }

    def restore_numpy(self, state):
        if (state.get("version"), state.get("seed"), state.get("role")) != (
            1,
            self.seed,
            self.role,
        ) or set(state.get("streams", {})) != set(self._numpy):
            raise ValueError(
                "Random engine state does not match seed, role, or streams"
            )
        # Validate all states before mutating any live Generator references.
        for value in state["streams"].values():
            probe = np.random.PCG64()
            probe.state = deepcopy(value)
        for name, value in state["streams"].items():
            self._numpy[name].bit_generator.state = deepcopy(value)


def make_random_engines(seed):
    """Seed legacy APIs once and return (training_engine, evaluation_engine).

    CRAFT uses the explicit streams, not the legacy global APIs. JAX engine state
    is stored through the agent's live NNX reference; NumPy state is run metadata.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    root = jax.random.key(seed, impl="threefry2x32")

    def engine(role):
        key = jax.random.fold_in(root, role)
        rngs = nnx.Rngs(
            params=jax.random.fold_in(key, 0),
            updates=jax.random.fold_in(key, 1),
            evaluation=jax.random.fold_in(key, 2),
        )
        generators = {
            name: np.random.default_rng(np.random.SeedSequence([seed, role, domain]))
            for name, domain in _NUMPY_STREAMS.items()
        }
        return RandomEngine(seed, role, rngs, generators)

    return engine(0), engine(1)
