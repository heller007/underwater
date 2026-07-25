"""E7: lightweight utility-regression gates (descriptor / CNN / combined)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.common.quiet import progress
from src.common.run import save_json
from src.detection.train import predict_split
from src.enhancement.descriptors import descriptor_vector
from src.routing.dataset_utils import (
    list_split_images,
    load_oracle_table,
    materialize_selected_split,
    summarize_selection,
    yolo_alias_for_split,
)


class GateImageDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        oracle_df: pd.DataFrame,
        actions: list[str],
        imgsz: int = 160,
        desc_mean: np.ndarray | None = None,
        desc_std: np.ndarray | None = None,
    ):
        self.image_dir = Path(image_dir)
        self.actions = list(actions)
        self.imgsz = imgsz
        self.table = oracle_df.set_index("stem")
        files = sorted(
            p
            for p in self.image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            and p.stem in self.table.index
        )
        if not files:
            raise SystemExit(f"No overlapping images/oracle rows in {image_dir}")
        self.files = files
        sample = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
        keys, vec = descriptor_vector(sample)
        self.desc_keys = keys
        self.desc_dim = int(len(vec))
        if desc_mean is None:
            mats = []
            for p in files:
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                _, v = descriptor_vector(img)
                mats.append(v)
            arr = np.stack(mats, axis=0)
            self.desc_mean = arr.mean(axis=0).astype(np.float32)
            self.desc_std = arr.std(axis=0).astype(np.float32) + 1e-6
        else:
            self.desc_mean = desc_mean.astype(np.float32)
            self.desc_std = desc_std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        stem = path.stem
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            bgr = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        thumb = cv2.resize(bgr, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        # RGB float CHW in [0,1]
        rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x_img = torch.from_numpy(rgb.transpose(2, 0, 1))
        _, vec = descriptor_vector(bgr)
        x_desc = torch.from_numpy((vec - self.desc_mean) / self.desc_std)
        row = self.table.loc[stem]
        utils = torch.tensor(
            [float(row[f"utility_{a}"]) for a in self.actions], dtype=torch.float32
        )
        ora = str(row["oracle"])
        y_cls = self.actions.index(ora) if ora in self.actions else 0
        return {
            "image": x_img,
            "desc": x_desc,
            "utilities": utils,
            "oracle_idx": y_cls,
            "stem": stem,
        }


class DescriptorGate(nn.Module):
    def __init__(self, desc_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(desc_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, image, desc):
        return self.net(desc)


class CNNGate(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, n_actions)

    def forward(self, image, desc):
        h = self.features(image).flatten(1)
        return self.head(h)


class CombinedGate(nn.Module):
    def __init__(self, desc_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + desc_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, image, desc):
        h = self.features(image).flatten(1)
        return self.head(torch.cat([h, desc], dim=1))


def build_gate(kind: str, desc_dim: int, n_actions: int) -> nn.Module:
    if kind == "descriptor":
        return DescriptorGate(desc_dim, n_actions)
    if kind == "cnn":
        return CNNGate(n_actions)
    if kind == "combined":
        return CombinedGate(desc_dim, n_actions)
    raise ValueError(f"Unknown gate kind: {kind}")


def _collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "desc": torch.stack([b["desc"] for b in batch]),
        "utilities": torch.stack([b["utilities"] for b in batch]),
        "oracle_idx": torch.tensor([b["oracle_idx"] for b in batch], dtype=torch.long),
        "stem": [b["stem"] for b in batch],
    }


def train_one_gate(
    kind: str,
    train_ds: GateImageDataset,
    val_ds: GateImageDataset | None,
    *,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    alpha: float = 0.5,
    temperature: float = 1.0,
    device: str = "cuda",
) -> tuple[nn.Module, dict[str, Any]]:
    model = build_gate(kind, train_ds.desc_dim, len(train_ds.actions)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    huber = nn.SmoothL1Loss()
    ce = nn.CrossEntropyLoss()
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=_collate
    )
    history = []
    best_state = None
    best_val = float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for batch in loader:
            img = batch["image"].to(device)
            desc = batch["desc"].to(device)
            utils = batch["utilities"].to(device)
            y = batch["oracle_idx"].to(device)
            pred = model(img, desc)
            loss = huber(pred, utils) + alpha * ce(pred / temperature, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(y)
            n += len(y)
        train_loss = total / max(n, 1)
        val_loss = train_loss
        if val_ds is not None and len(val_ds) > 0:
            val_loss = _eval_loss(model, val_ds, device, alpha, temperature, huber, ce)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss})
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            progress(f"[e7/{kind}] epoch {ep}/{epochs} loss={train_loss:.4f} val={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": history, "best_val_loss": best_val if best_state else train_loss}


def _eval_loss(model, ds, device, alpha, temperature, huber, ce) -> float:
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=_collate)
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            desc = batch["desc"].to(device)
            utils = batch["utilities"].to(device)
            y = batch["oracle_idx"].to(device)
            pred = model(img, desc)
            loss = huber(pred, utils) + alpha * ce(pred / temperature, y)
            total += float(loss.item()) * len(y)
            n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def predict_actions(
    model: nn.Module,
    ds: GateImageDataset,
    device: str,
) -> dict[str, str]:
    model.eval()
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=_collate)
    out: dict[str, str] = {}
    for batch in loader:
        pred = model(batch["image"].to(device), batch["desc"].to(device))
        idx = pred.argmax(dim=1).cpu().tolist()
        for stem, i in zip(batch["stem"], idx):
            out[stem] = ds.actions[int(i)]
    return out


def run_e7_gate(
    weights: Path,
    src_yolo_root: Path,
    out_dir: Path,
    actions: list[str],
    oracle_gate: Path,
    oracle_test: Path | None,
    device: str,
    *,
    gate_kinds: list[str] | None = None,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    alpha: float = 0.5,
    temperature: float = 1.0,
    imgsz: int = 160,
    conf: float = 0.25,
    drop_enhanced: bool = True,
    torch_device: str | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kinds = gate_kinds or ["descriptor", "cnn", "combined"]
    tdev = torch_device or ("cuda" if torch.cuda.is_available() else "cpu")
    if tdev.startswith("cuda") and "," in str(device):
        tdev = "cuda:0"

    gate_df = load_oracle_table(oracle_gate)
    test_df = load_oracle_table(oracle_test) if oracle_test and Path(oracle_test).exists() else None

    gate_img = Path(src_yolo_root) / "images" / "gate"
    test_img = Path(src_yolo_root) / "images" / "test"
    train_ds = GateImageDataset(gate_img, gate_df, actions, imgsz=imgsz)

    # hold out 15% of gate for early stopping (by group-ish: last 15% stems)
    n = len(train_ds)
    n_val = max(1, int(0.15 * n))
    # simple split: last n_val as val
    indices = list(range(n))
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]

    class _Sub(Dataset):
        def __init__(self, base, idxs):
            self.base = base
            self.idxs = idxs
            self.actions = base.actions
            self.desc_dim = base.desc_dim
            self.desc_mean = base.desc_mean
            self.desc_std = base.desc_std

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, i):
            return self.base[self.idxs[i]]

    train_sub = _Sub(train_ds, train_idx)
    val_sub = _Sub(train_ds, val_idx)

    summary: dict[str, Any] = {
        "weights": str(weights),
        "actions": actions,
        "gate_kinds": kinds,
        "epochs": epochs,
        "alpha": alpha,
        "temperature": temperature,
        "imgsz": imgsz,
        "n_gate": n,
        "gates": {},
    }

    test_ds = None
    if test_df is not None and test_img.exists():
        test_ds = GateImageDataset(
            test_img,
            test_df,
            actions,
            imgsz=imgsz,
            desc_mean=train_ds.desc_mean,
            desc_std=train_ds.desc_std,
        )

    for kind in kinds:
        progress(f"[e7] training gate={kind}")
        model, train_info = train_one_gate(
            kind,
            train_sub,
            val_sub,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            alpha=alpha,
            temperature=temperature,
            device=tdev,
        )
        ckpt = out_dir / f"gate_{kind}.pt"
        torch.save(
            {
                "kind": kind,
                "actions": actions,
                "desc_mean": train_ds.desc_mean,
                "desc_std": train_ds.desc_std,
                "desc_keys": train_ds.desc_keys,
                "state_dict": model.state_dict(),
                "imgsz": imgsz,
            },
            ckpt,
        )

        gate_pred = predict_actions(model, train_ds, tdev)
        gate_stats = summarize_selection(gate_df, gate_pred, actions)

        gate_entry: dict[str, Any] = {
            "checkpoint": str(ckpt),
            "train": train_info,
            "gate_split": gate_stats,
        }

        if test_ds is not None:
            test_pred = predict_actions(model, test_ds, tdev)
            test_stats = summarize_selection(test_df, test_pred, actions)
            alias = yolo_alias_for_split("test")
            root = out_dir / f"selected_{kind}_test"
            data_yaml = materialize_selected_split(
                src_yolo_root, root, "test", test_pred, yolo_alias=alias
            )
            metrics = predict_split(
                weights=weights,
                data_yaml=data_yaml,
                split=alias,
                out_dir=out_dir / f"eval_{kind}_test",
                device=device,
                conf=conf,
                quiet=True,
            )
            gate_entry["test_split"] = {
                **test_stats,
                "metrics": metrics.get("metrics", metrics),
            }
            # save predictions
            pd.DataFrame(
                [{"stem": s, "pred": a} for s, a in test_pred.items()]
            ).to_csv(out_dir / f"preds_{kind}_test.csv", index=False)
            progress(
                f"[e7/{kind}] test mAP50={gate_entry['test_split'].get('metrics', {}).get('mAP50')} "
                f"acc={test_stats.get('action_accuracy_vs_oracle')}"
            )
            if drop_enhanced:
                import shutil

                shutil.rmtree(root / "images", ignore_errors=True)

        summary["gates"][kind] = gate_entry

    # baselines on test for context
    if test_df is not None:
        baselines = {}
        for name, action in [("always_t0", "T0"), ("always_t4", "T4" if "T4" in actions else actions[-1])]:
            stem_to = {s: action for s in test_df["stem"].tolist()}
            # only stems that exist
            test_stems = {p.stem for p in list_split_images(src_yolo_root, "test")}
            stem_to = {s: a for s, a in stem_to.items() if s in test_stems}
            alias = yolo_alias_for_split("test")
            root = out_dir / f"selected_{name}_test"
            data_yaml = materialize_selected_split(
                src_yolo_root, root, "test", stem_to, yolo_alias=alias
            )
            metrics = predict_split(
                weights=weights,
                data_yaml=data_yaml,
                split=alias,
                out_dir=out_dir / f"eval_{name}_test",
                device=device,
                conf=conf,
                quiet=True,
            )
            baselines[name] = {
                **summarize_selection(test_df, stem_to, actions),
                "metrics": metrics.get("metrics", metrics),
            }
            if drop_enhanced:
                import shutil

                shutil.rmtree(root / "images", ignore_errors=True)
        summary["baselines"] = baselines

    save_json(out_dir / "e7_results.json", summary)
    return summary
