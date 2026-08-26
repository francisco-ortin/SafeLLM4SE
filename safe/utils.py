"""Miscellaneous utility functions for local scripts."""

import logging

logger: logging.Logger = logging.getLogger(__name__)


def read_api_key_from_file(file_path: str) -> str:
    """Read and return an API key from a plain text file."""
    with open(file_path, "r", encoding="utf-8") as api_key_file:
        api_key: str = api_key_file.read().strip()
    return api_key
