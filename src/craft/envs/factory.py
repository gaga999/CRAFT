"""Environment adapters belong here, separate from dataset loading."""


def make_env(name: str, *, seed: int):
    """Construct an environment using an explicitly selected benchmark API."""
    raise NotImplementedError(f"Environment adapter not implemented: {name!r}")
