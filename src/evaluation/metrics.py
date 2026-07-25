"""Evaluation helpers (metrics aggregation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.run import save_json


def aggregate_split_metrics(results: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    agg = {"splits": {}}
    for r in results:
        split = r.get("split", "unknown")
        agg["splits"][split] = r.get("metrics", r)
    save_json(out_path, agg)
    return agg
