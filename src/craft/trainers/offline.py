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
    steps_per_call=1,
):
    """Train through `stages`, dispatching up to `steps_per_call` updates at once.

    Chunks end at multiples of `steps_per_call`, stage ends and the step limit,
    and use `agent.update_chunk`; single steps use `agent.update`. `on_step`
    runs once per chunk with its final step and last-step metrics, so callbacks
    due at multiples of `steps_per_call` fire at the same steps as with
    `steps_per_call=1`. A nonfinite update rejects its whole chunk: the agent
    and the sampler stay at the chunk start.
    """
    if type(steps_per_call) is not int or steps_per_call < 1:
        raise ValueError("steps_per_call must be an integer >= 1")
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
            stop = min(end, offset + stage["steps"])
            step = max(step, offset)
            while step < stop:
                count = min(steps_per_call - step % steps_per_call, stop - step)
                started = perf_counter()
                if count == 1:
                    batch, update = next(batches), agent.update
                else:
                    batch, update = batches.next_chunk(count), agent.update_chunk
                sampled = perf_counter()
                try:
                    metrics = update(batch, stage=stage["name"])
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
                step += count
                metrics["time/data_wait_seconds"] = (sampled - started) / count
                metrics["time/update_seconds"] = (perf_counter() - sampled) / count
                if on_step is not None:
                    on_step(step, stage["name"], metrics)
            offset += stage["steps"]
    return agent
