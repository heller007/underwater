"""Quiet console for Kaggle: full logs to file, short progress to stdout only."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


_REAL_STDOUT: TextIO = sys.stdout
_REAL_STDERR: TextIO = sys.stderr


def progress(msg: str) -> None:
    """Always visible in the notebook cell (bypasses log redirect)."""
    _REAL_STDOUT.write(str(msg).rstrip() + "\n")
    _REAL_STDOUT.flush()


@contextmanager
def quiet_run(log_path: str | Path, also_stderr: bool = True) -> Iterator[Path]:
    """
    Redirect stdout/stderr to log_path. Use progress() for cell-visible lines.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = log_f  # type: ignore[assignment]
        if also_stderr:
            sys.stderr = log_f  # type: ignore[assignment]
        # Mute noisy loggers that bypass stdout
        for name in ("ultralytics", " Ultralytics"):
            lg = logging.getLogger(name)
            lg.setLevel(logging.WARNING)
            lg.propagate = False
        progress(f"[log] writing details -> {log_path}")
        yield log_path
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        log_f.close()


def silence_ultralytics() -> None:
    """Reduce Ultralytics console spam (call before YOLO import/train)."""
    os.environ.setdefault("YOLO_VERBOSE", "False")
    try:
        from ultralytics.utils import LOGGER

        LOGGER.setLevel(logging.WARNING)
    except Exception:
        pass
