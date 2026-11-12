"""Multi-stage scheduling with rollback of rejected samples and finite updates."""

from time import perf_counter

from craft.data.prefetch import PrefetchBatches


def train_offline(
    agent,
    dataset,
    stages,
    *,
    batch_size,
    max_steps=None,
    on_step=None,
    workers=2,
    prefetch_batches=4,
    device_prefetch=2,
):
    start = int(agent.step[...])
    total = sum(stage["steps"] for stage in stages)
    if not 0 <= start <= total:
        raise ValueError("Checkpoint step is outside the stage schedule")
    end = total if max_steps is None else min(total, start + max_steps)
    step, offset = start, 0
    with PrefetchBatches(
        dataset,
        batch_size,
        end - start,
        workers=workers,
        prefetch_batches=prefetch_batches,
        device_prefetch=device_prefetch,
    ) as batches:
        for stage in stages:
            stage_end = offset + stage["steps"]
            begin = max(start, offset)
            for _ in range(begin, min(end, stage_end)):
                started = perf_counter()
                batch = next(batches)
                sampled = perf_counter()
                try:
                    metrics = agent.update(batch, stage=stage["name"])
                    # Keep the finite-update barrier, but avoid repeated device
                    # reads of agent.step on the successful path.
                    if not bool(metrics.pop("numerics/finite")):
                        raise FloatingPointError(
                            "Rejected nonfinite batch, loss, or gradient; state was not updated"
                        )
                    batches.accept()
                except BaseException:
                    if batches.pending and int(agent.step[...]) != step:
                        batches.accept()
                    raise
                step += 1
                metrics["time/data_wait_seconds"] = sampled - started
                metrics["time/update_seconds"] = perf_counter() - sampled
                if on_step is not None:
                    on_step(step, stage["name"], metrics)
            offset = stage_end
    return agent
