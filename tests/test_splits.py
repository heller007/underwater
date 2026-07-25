"""Tests for LOSO split contamination logic."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.seaclear import ImageRecord, SeaClearDataset
from src.data.splits import build_loso_fold, contamination_check


def _fake_ds():
    images = []
    iid = 0
    for site in ("A", "B", "C"):
        for seq in ("s1", "s2"):
            for i in range(4):
                images.append(
                    ImageRecord(
                        image_id=iid,
                        file_name=f"{site}/{seq}/{i}.jpg",
                        path=Path(f"{site}/{seq}/{i}.jpg"),
                        width=64,
                        height=64,
                        site=site,
                        camera="cam0",
                        sequence=seq,
                        group_id=f"{site}|cam0|{seq}",
                    )
                )
                iid += 1
    return SeaClearDataset(root=Path("."), images=images, annotations=[], categories=[])


def test_loso_no_contamination():
    ds = _fake_ds()
    df = build_loso_fold(ds, held_out_site="C", seed=0)
    check = contamination_check(df)
    assert check["pass"]
    assert (df[df["site"] == "C"]["split"] == "test").all()
    assert (df[df["split"] == "test"]["site"] == "C").all()


def test_groups_single_split():
    ds = _fake_ds()
    df = build_loso_fold(ds, held_out_site="A", seed=1)
    g = df.groupby("group_id")["split"].nunique()
    assert (g == 1).all()
