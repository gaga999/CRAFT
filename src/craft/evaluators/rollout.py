"""Environment evaluation uses a bound policy callable, not training losses."""


def evaluate_rollouts(act, env, *, episodes: int, seed: int):
    """Plan reset, RNG, chunk execution/replanning, and benchmark metrics here."""
    raise NotImplementedError("Rollout evaluation and protocol are pending")
