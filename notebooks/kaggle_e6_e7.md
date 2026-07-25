# E6 Quality selectors + E7 Utility gate (Kaggle)

**Attach:** SeaClear + E4 weights dataset + (optional) dataset with `oracle_table_*.csv`  
**Accelerator:** GPU T4 x2 · Internet ON  

E5 said pivot — still run E6/E7 for paper baselines (H6 + gate ablation). Expect small gains.

```python
import os, subprocess, sys, shutil
from pathlib import Path

REPO_URL = "https://github.com/heller007/underwater.git"
REPO_DIR = "/kaggle/working/underwater"

# E4 mixed weights (file or folder)
E4_WEIGHTS = "/kaggle/input/datasets/pandeyvineet/<your-e4-dataset>/e4_best.pt"

# Optional: if oracle CSVs are in a Kaggle dataset (else use copies in the cloned repo)
ORACLE_DS = None  # e.g. "/kaggle/input/datasets/pandeyvineet/e5-oracle-tables"

os.chdir("/kaggle/working")
if (Path(REPO_DIR) / ".git").exists():
    subprocess.check_call(["git", "-C", REPO_DIR, "fetch", "--all"])
    subprocess.check_call(["git", "-C", REPO_DIR, "reset", "--hard", "origin/main"])
else:
    subprocess.check_call(["git", "clone", REPO_URL, REPO_DIR])
os.chdir(REPO_DIR)

%pip install -q ultralytics imagehash scikit-image pycocotools opencv-python-headless

if ORACLE_DS:
    for name in ("oracle_table_gate.csv", "oracle_table_test.csv", "e5_results.json"):
        src = Path(ORACLE_DS) / name
        if src.exists():
            shutil.copy2(src, REPO_DIR / name)

w = Path(E4_WEIGHTS)
best = w if w.suffix == ".pt" else next(w.rglob("best.pt"))
assert best.exists(), best

# --- E6 ---
subprocess.check_call([
    sys.executable, "scripts/run_stage.py",
    "--stage", "e6", "--env", "kaggle",
    "--held-out-site", "Lokrum", "--device", "0,1",
    "--weights", str(best),
    "--actions", "T0,T4", "--drop-enhanced",
    "--oracle-gate", str(Path(REPO_DIR) / "oracle_table_gate.csv"),
    "--oracle-test", str(Path(REPO_DIR) / "oracle_table_test.csv"),
], cwd=REPO_DIR)

# --- E7 ---
subprocess.check_call([
    sys.executable, "scripts/run_stage.py",
    "--stage", "e7", "--env", "kaggle",
    "--held-out-site", "Lokrum", "--device", "0,1",
    "--weights", str(best),
    "--actions", "T0,T4", "--drop-enhanced",
    "--oracle-gate", str(Path(REPO_DIR) / "oracle_table_gate.csv"),
    "--oracle-test", str(Path(REPO_DIR) / "oracle_table_test.csv"),
], cwd=REPO_DIR)
```

**Save after run:** `e6/e6_results.json`, `e7/e7_results.json`, and `e7/gate_*.pt` (small).
