# Selective Enhancement for Marine-Debris Detection

Cost-aware selective enhancement + deferral research code (SeaClear LOSO, YOLOv8n, dual-T4 Kaggle).

See [`main.md`](main.md) for the full research plan. This repo implements **Phase 1**: data readiness + raw baseline (E0–E1). Later stages (E2–E13) are scaffolded.

## GitHub → Kaggle workflow

Yes — edit here, push to GitHub, clone on Kaggle. That is the intended loop.

| Do | Don't |
| --- | --- |
| Attach SeaClear as a **Kaggle Dataset** under `/kaggle/input/...` | Download the 1.7GB RAR every session from 4TU |
| Set Accelerator to **GPU T4 x2** | Assume single GPU (`device=0` only) |
| Enable Internet for `git clone` / `pip install` | Reinstall PyTorch/CUDA (Kaggle already has it) |
| Write outputs to `/kaggle/working` | Cache all enhanced 1920×1080 variants for every action |
| Use `scripts/run_stage.py --stage smoke\|prep\|e1` | Rely on notebook cell state as the source of truth |

### One-time setup

1. Upload (or attach) [SeaClear](https://data.4tu.nl/datasets/4f1dff25-e157-4399-a5d4-478055461689) as a Kaggle Dataset.
2. Push this repo to **public** GitHub (or add a token for private).
3. Create a Kaggle Notebook, attach the dataset, Accelerator = **GPU T4 x2**.
4. Open [`notebooks/kaggle_e1_baseline.ipynb`](notebooks/kaggle_e1_baseline.ipynb) or paste the same cells.

If auto-discovery fails, set the path explicitly:

```python
SEACLEAR_ROOT = "/kaggle/input/your-dataset-slug"
```

Update `configs/env/kaggle.yaml` `seaclear_candidates` with your dataset folder name.

## Local quickstart

```bash
pip install -r requirements.txt
# Put SeaClear under data/raw/seaclear/ (COCO JSON + images)

python scripts/run_stage.py --stage prep --held-out-site Lokrum
python scripts/run_stage.py --stage smoke --held-out-site Lokrum   # 2 epochs
python scripts/run_stage.py --stage e1 --held-out-site Lokrum     # full baseline
```

Or step-by-step:

```bash
python scripts/audit_data.py
python scripts/build_splits.py --held-out-site Lokrum
python scripts/prepare_yolo.py --held-out-site Lokrum
python scripts/train_detector.py --experiment configs/experiments/e1_baseline.yaml
python scripts/evaluate.py --weights runs/<run_id>/train/weights/best.pt --held-out-site Lokrum
```

## Dual T4

Training uses Ultralytics multi-GPU via `device=0,1` (auto on Kaggle / when 2 GPUs are visible). Batch defaults to 32 (≈16/GPU). If OOM, pass `--batch 16` or `--batch 8`.

## Experiment tracking (paper)

After each stage, update [`reports/EXPERIMENT_LOG.md`](reports/EXPERIMENT_LOG.md) with protocol, metrics, run IDs, and findings. That file is the source for paper tables/figures.


## Repo layout

```text
configs/     env, data, detector, experiments
src/         data, enhancement, detection, evaluation, ...
scripts/     CLI entrypoints
notebooks/   Kaggle notebook
tests/       unit tests
reports/     audit artifacts
runs/        training runs (gitignored)
```

## Tests

```bash
pytest -q
```
