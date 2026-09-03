"""Compatibility wrapper for the SafeLLM4SE sampler command."""

import sys
from pathlib import Path


def main() -> None:
    """Run the packaged SafeLLM4SE sampler command."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from safellm4se.sample import main as package_main

    package_main()


if __name__ == "__main__":
    main()
