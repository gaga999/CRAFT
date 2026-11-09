"""Offline data-source example; sampling state belongs to the data source."""

from craft.types import Batch


class OfflineDataset:
    """Implement loading and episode-aware goal/chunk sampling here."""

    def sample(self, batch_size: int) -> Batch:
        """Return a batch satisfying the selected agent's documented contract."""
        raise NotImplementedError("Dataset loading and sampling are pending")
