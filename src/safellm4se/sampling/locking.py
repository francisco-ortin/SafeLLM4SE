"""Cross-platform process-safe file locking."""

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def interprocess_file_lock(lock_path: Path) -> Generator[None, None, None]:
    """Serialize file updates across independent Python processes.
    Args:
        lock_path: Path to the lock file used for interprocess coordination.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()
