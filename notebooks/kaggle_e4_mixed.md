# E4 Mixed-path — quiet cell output (Kaggle)

**Attach:** SeaClear + `pandeyvineet/e3-t4-chkpt` (for later comparison; not required to train E4)  
**Accelerator:** GPU T4 x2 · Internet ON

Cell output shows only progress lines. Full Ultralytics logs → `runs/.../e4_console.log`.

```python
import os, subprocess, sys
from pathlib import Path

REPO_URL = "https://github.com/heller007/underwater.git"
REPO_DIR = "/kaggle/working/underwater"
E3_T4 = "/kaggle/input/datasets/pandeyvineet/e3-t4-chkpt"  # your saved T4 chkpt root

os.chdir("/kaggle/working")
if (Path(REPO_DIR) / ".git").exists():
    subprocess.check_call(["git", "-C", REPO_DIR, "fetch", "--all"])
    subprocess.check_call(["git", "-C", REPO_DIR, "reset", "--hard", "origin/main"])
else:
    subprocess.check_call(["git", "clone", REPO_URL, REPO_DIR])
os.chdir(REPO_DIR)

%pip install -q ultralytics imagehash scikit-image pycocotools opencv-python-headless

# optional: locate T4 best.pt for your records
cands = list(Path(E3_T4).rglob("best.pt"))
print("E3-T4 best.pt:", cands[0] if cands else "NOT FOUND (ok for E4 train)")

subprocess.check_call([
    sys.executable, "scripts/run_stage.py",
    "--stage", "e4",
    "--env", "kaggle",
    "--held-out-site", "Lokrum",
    "--device", "0,1",
    "--actions", "T0,T4",
    "--drop-enhanced",
], cwd=REPO_DIR)
```

After finish, zip the printed run folder (contains `train/weights/best.pt` + `e4_results.json` + `e4_console.log`).
