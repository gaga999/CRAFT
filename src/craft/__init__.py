"""CRAFT: Contrastive Representation for Action Flow and Temporal Abstraction."""


def main() -> None:
    """Run the package CLI; also exposed through scripts/train.py."""
    from craft.cli import train_main

    train_main()
