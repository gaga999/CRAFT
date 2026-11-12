"""Bounded CPU workers and asynchronous JAX device prefetch, in sample order."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Thread

import jax


@dataclass
class PreparedBatch:
    batch: dict
    rng_state: dict


@dataclass
class WorkerFailure:
    error: BaseException


class PrefetchBatches:
    """Only commit accepted samples to the dataset's checkpointed live RNG.

    One private planner draws indices in order. NumPy workers share resident
    arrays, not RNGs or dataset copies. A producer transfers completed batches
    with nonblocking device_put into a bounded queue. Pending batches can be
    discarded and regenerated exactly after interruption/resume.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        count,
        *,
        workers=2,
        prefetch_batches=4,
        device_prefetch=2,
    ):
        for name, value, minimum in (
            ("batch_size", batch_size, 2),
            ("count", count, 0),
            ("workers", workers, 0),
            ("prefetch_batches", prefetch_batches, 1),
            ("device_prefetch", device_prefetch, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.dataset, self.batch_size, self.count = dataset, batch_size, count
        self.workers, self.prefetch_batches = workers, prefetch_batches
        self._rng = deepcopy(dataset.rng)
        # Capture the caller's device context before entering a background thread.
        self._device = jax.config.jax_default_device or jax.devices()[0]
        self._queue = Queue(maxsize=device_prefetch)
        self._stop = Event()
        self._thread = None
        self._current = None
        self._consumed = 0
        self._entered = False

    def __enter__(self):
        if self._entered or self._stop.is_set():
            raise RuntimeError("Prefetch iterator cannot be re-entered")
        self._entered = True
        if self.workers and self.count:
            self._thread = Thread(target=self._produce, name="craft-prefetch")
            self._thread.start()
        return self

    def _plan(self):
        indices = self.dataset.sample_indices(self.batch_size, rng=self._rng)
        return indices, deepcopy(self._rng.bit_generator.state)

    def _gather(self, indices):
        # actor_goals is unused by CRAFT; its RNG draw is still in the plan.
        return self.dataset.gather(indices, include_actor_goals=False)

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except Full:
                continue

    def _produce(self):
        pending = deque()
        try:
            with ThreadPoolExecutor(
                max_workers=self.workers, thread_name_prefix="craft-sample"
            ) as pool:
                submitted = 0

                def submit():
                    nonlocal submitted
                    indices, state = self._plan()
                    pending.append((pool.submit(self._gather, indices), state))
                    submitted += 1

                for _ in range(min(self.prefetch_batches, self.count)):
                    submit()
                while pending and not self._stop.is_set():
                    future, state = pending.popleft()
                    host_batch = future.result()
                    self._put(
                        PreparedBatch(jax.device_put(host_batch, self._device), state)
                    )
                    if submitted < self.count and not self._stop.is_set():
                        submit()
                for future, _ in pending:
                    future.cancel()
        except BaseException as exc:  # noqa: BLE001 -- propagate worker failures to the consumer
            self._put(WorkerFailure(exc))

    def __iter__(self):
        return self

    def __next__(self):
        if not self._entered or self._stop.is_set():
            raise RuntimeError("Prefetch iterator must be used inside its context")
        if self._current is not None:
            raise RuntimeError("Accept the current batch before requesting another")
        if self._consumed == self.count:
            raise StopIteration
        if self.workers:
            while True:
                try:
                    item = self._queue.get(timeout=0.1)
                    break
                except Empty:
                    if self._thread is None or not self._thread.is_alive():
                        raise RuntimeError(
                            "Prefetch producer stopped without a batch"
                        ) from None
        else:
            indices, state = self._plan()
            item = PreparedBatch(
                jax.device_put(self._gather(indices), self._device), state
            )
        if isinstance(item, WorkerFailure):
            raise item.error
        self._current = item
        return item.batch

    def accept(self):
        if self._current is None:
            raise RuntimeError("No batch is awaiting acceptance")
        self.dataset.rng.bit_generator.state = self._current.rng_state
        self._current = None
        self._consumed += 1

    @property
    def pending(self):
        return self._current is not None

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._current = None
        # Release queued device buffers, including failures with traceback refs.
        while not self._queue.empty():
            self._queue.get_nowait()

    def __exit__(self, *args):
        self.close()
