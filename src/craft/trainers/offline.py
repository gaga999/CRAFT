"""Multi-stage scheduling with rollback of rejected samples and finite updates."""

from copy import deepcopy
from time import perf_counter


def train_offline(agent, dataset, stages, *, batch_size, max_steps=None, on_step=None):
    start = int(agent.step[...])
    total = sum(stage["steps"] for stage in stages)
    if not 0 <= start <= total:
        raise ValueError("Checkpoint step is outside the stage schedule")
    end = total if max_steps is None else min(total, start + max_steps)
    offset = 0
    for stage in stages:
        stage_end = offset + stage["steps"]
        begin = max(start, offset)
        for _ in range(begin, min(end, stage_end)):
            started = perf_counter()
            previous_rng = deepcopy(dataset.rng.bit_generator.state)
            previous_step = int(agent.step[...])
            try:
                metrics = agent.update(dataset.sample(batch_size), stage=stage["name"])
                if not bool(metrics.pop("numerics/finite")):
                    raise FloatingPointError(
                        "Rejected nonfinite batch, loss, or gradient; state was not updated"
                    )
            except BaseException:
                if int(agent.step[...]) == previous_step:
                    dataset.rng.bit_generator.state = previous_rng
                raise
            metrics["time/update_seconds"] = perf_counter() - started
            step = int(agent.step[...])
            if on_step is not None:
                on_step(step, stage["name"], metrics)
        offset = stage_end
    return agent
