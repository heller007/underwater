"""Quiet console for Kaggle: full logs to file, minimal progress to cell."""

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
    Redirect stdout/stderr to log_path. Use progress() for rare cell-visible lines.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = log_f  # type: ignore[assignment]
        if also_stderr:
            sys.stderr = log_f  # type: ignore[assignment]
        for name in ("ultralytics", "ultralytics.yolo", "torch", "torch.distributed"):
            lg = logging.getLogger(name)
            lg.setLevel(logging.ERROR)
            lg.propagate = False
        yield log_path
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        log_f.close()


def silence_ultralytics() -> None:
    """Kill Ultralytics console spam (call before YOLO train/val)."""
    os.environ["YOLO_VERBOSE"] = "False"
    os.environ["TQDM_DISABLE"] = "1"
    try:
        from ultralytics.utils import LOGGER

        LOGGER.setLevel(logging.ERROR)
    except Exception:
        pass
