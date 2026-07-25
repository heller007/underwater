# Experiment Log — Selective Enhancement for Marine-Debris Detection

Living record of executed experiments for the paper.  
Update this file after every stage. Every number must trace to a run folder / artifact.

**Primary fold (development):** leave-one-site-out, held-out **Lokrum**  
**Detector:** YOLOv8n, 640, AMP, HSV color aug off, AdamW, dual T4 (`device=0,1`)  
**Labels:** SeaClear supercategories `debris`, `bio`, `robot`  
**Repo:** https://github.com/heller007/underwater  

---

## Artifact index

| Stage | Run / artifact | Notes |
| --- | --- | --- |
| E1 | `fold-lokrum_model-t0_exp-e1_baseline_seed-0_20260725T084249Z` | Also mirrored as Kaggle dataset `pandeyvineet/e1weights` |
| E2 | `fold-lokrum_model-e1frozen_exp-e2_naive_enhance_seed-0_20260725T103126Z` | See `e2/e2_results.json` |
| E3 | `fold-lokrum_model-fixed_exp-e3_fixed_path_seed-0_20260725T112111Z` | T4 weights saved; T2 metrics from log after freeze |
| E4 | *next* | Mixed-path detector |

Split sizes (LOSO Lokrum, full SeaClear): **train 6201 / val 918 / test 972** (8610 images).

---

## E0 — Data readiness

| Item | Result |
| --- | --- |
| Dataset | SeaClear (Kaggle input; 8610 images, 31555 anns after mapping) |
| Audit | `pass=True`, warnings=[] |
| Split | LOSO held-out Lokrum; source sites → 70/15/15 by group |
| Contamination | Required check in `manifests/loso_lokrum/contamination_check.json` |

---

## E1 — Raw baseline (train + eval on \(T_0\))

### Protocol
- **Action:** raw only (\(T_0\))
- **Train:** detector train split (source sites)
- **Model selection:** best val mAP50-95; early stopping patience 20
- **Never** use Lokrum test for training or early stopping
- **Hardware:** 2× Tesla T4

### Training outcome
- Best epoch: **51** (stopped at epoch **71**, ~1.26 h)
- Weights: `train/weights/best.pt`

### Metrics

| Split | Domain | P | R | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| **val** | seen sites | 0.629 | 0.408 | **0.436** | **0.303** |
| **test** | Lokrum held-out | 0.215 | 0.111 | **0.074** | **0.041** |

**Per-class mAP50-95**

| Class | val | test |
| --- | ---: | ---: |
| debris | 0.422 | 0.022 |
| bio | 0.132 | 0.010 |
| robot | 0.355 | 0.090 |

### Finding (paper-ready)
In-domain val performance is moderate (mAP50 ≈ 0.44), but **cross-site generalization to Lokrum collapses** (mAP50 ≈ 0.07). This gap is the primary motivation for domain-aware selective enhancement and deferral.

### Claim boundary
E1 is the **raw fixed pipeline** baseline, not the proposed method.

---

## E2 — Naive test-time enhancement (frozen E1 detector)

### Protocol
- **Detector:** frozen E1 `best.pt` (no retraining)
- **Actions:** \(T_0\) raw, \(T_1\) gray-world, \(T_2\) LAB-CLAHE, \(T_3\) adaptive gamma, \(T_4\) fusion
- **Eval:** apply \(T_k\) to val and test images only; same labels
- **Purpose:** measure when test-time enhancement helps vs harms

### Metrics (mAP50-95)

| Action | Name | val | test (Lokrum) | Δ test vs \(T_0\) |
| --- | --- | ---: | ---: | ---: |
| **T0** | raw | **0.303** | 0.041 | — |
| T1 | gray-world | 0.173 | 0.053 | +0.012 |
| T2 | LAB-CLAHE | 0.269 | 0.053 | +0.013 |
| T3 | adaptive gamma | 0.293 | 0.043 | +0.002 |
| **T4** | fusion | 0.267 | **0.057** | **+0.017** |

**mAP50 on Lokrum test:** T0 0.074 → T4 **0.096** (best).

### Ranking (test mAP50-95)
1. T4 fusion  
2. T2 LAB-CLAHE ≈ T1 gray-world  
3. T3 adaptive gamma  
4. T0 raw  

### Findings (paper-ready)
1. **Enhancement harm on in-domain val:** every enhanced path *hurts* val vs raw (worst: T1, −0.13 mAP50-95).
2. **Mild help on held-out Lokrum:** all \(T_{k>0}\) ≥ raw; best is fusion (**+0.017** mAP50-95).
3. **Gains do not close the domain gap:** best enhanced test (0.057) remains far below val raw (0.303).
4. Supports the thesis that **always-on enhancement is wrong**; a selective policy is justified.

### Shortlist carried to E3
| Keep | Reason |
| --- | --- |
| **T0** | Best in-domain; necessary raw/no-op action |
| **T4** | Best Lokrum test under naive enhance |
| **T2** | Second-best enhanced path; cheap classical alternative to fusion |

Drop for main fixed-path comparison: T1, T3 (optional ablations later).

### Claim boundary
E2 is **test-time preprocessing without adaptation**, not a fair fixed-enhancement pipeline. Fair comparison requires E3.

---

## E3 — Fixed-path training (train & test consistently on \(T_k\))

### Protocol
- Shortlist from E2: \(T_0\), \(T_2\), \(T_4\) (T1/T3 dropped after E2)
- For each \(T_k\): train on \(T_k\) train images; eval on \(T_k\) val + Lokrum test
- **T0:** reused E1 weights/metrics (no retrain)
- Same detector config as E1; dual T4
- Run id: `fold-lokrum_model-fixed_exp-e3_fixed_path_seed-0_20260725T112111Z`

### Metrics

| Action | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 | Δ test mAP50-95 vs T0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| T0 raw | 0.436 | 0.303 | 0.074 | 0.041 | — |
| T2 LAB-CLAHE | 0.397 | 0.283 | 0.092 | 0.069 | +0.028 |
| **T4 fusion** | **0.450** | 0.289 | **0.212** | **0.157** | **+0.116** |

**Per-class test mAP50-95 (T4):** debris 0.022, bio 0.006, robot **0.443**  
**Per-class test mAP50-95 (T2):** debris 0.016, bio 0.003, robot 0.187  

### Ranking (Lokrum test mAP50-95)
1. **T4 fusion** (0.157)  
2. T2 CLAHE (0.069)  
3. T0 raw (0.041)  

### Findings (paper-ready)
1. **Fixed-path fusion is much stronger than naive test-time fusion** (E2-T4 test mAP50-95 0.057 → E3-T4 **0.157**).
2. Best fixed pipeline on held-out Lokrum is **always-enhance-with-T4 + retrain**.
3. Domain gap remains: T4 val 0.289 vs test 0.157 — still motivates selective routing / deferral.
4. T2 helps vs raw but is clearly worse than T4 under consistent training.

### Saved artifacts (after Kaggle freeze)
| Item | Status |
| --- | --- |
| T4 `best.pt` + T4 results | Saved by user |
| T2 `best.pt` / full `e3_results.json` | Lost in freeze — **metrics recovered from `e3.log`** |
| T0 / E1 weights | Still on Kaggle dataset `pandeyvineet/e1weights` |

### Claim boundary
E3 provides the **fair fixed-enhancement baselines**. Main comparison for later methods: proposed system vs **E3-T4** (best fixed) and **E1-T0** (raw).

---

## Later stages (planned — not yet run)

| ID | Name | Goal |
| --- | --- | --- |
| E4 | Mixed-path detector | Shared detector for gate actions |
| E5 | Oracle study | Go/no-go for routing (≥ ~2 mAP or clear harm reduction) |
| E6 | Quality-metric selectors | UCIQE/UIQM baselines |
| E7 | Learned utility gate | Main method |
| E8 | Cost-aware λ sweep | Accuracy–latency Pareto |
| E9 | Reliability / defer | Risk–coverage |
| E10 | Full LOSO | All five held-out sites |
| E11 | TrashCan external | Frozen system |
| E12–E13 | Multi-seed + stats | CIs / bootstrap |

---

## Figures / tables planned from current evidence

1. **Domain gap bar chart:** E1 val vs Lokrum test (mAP50 / mAP50-95).  
2. **E2 enhancement matrix:** action × (val, test) mAP; annotate harm vs help.  
3. **E2 delta plot:** Δ mAP vs \(T_0\) on val (negative) vs test (positive).  
4. After E3: fixed-path table replacing naive E2 for main baseline comparisons.

---

## Reproducibility checklist

- [x] E1 seed 0, Lokrum holdout, dual T4  
- [x] E2 frozen E1 weights, T0–T4, val+test  
- [ ] E3 fixed-path T0/T2/T4  
- [ ] Save `e*_results.json` + `best.pt` per stage to a Kaggle Dataset / local archive  
- [ ] Cite SeaClear; note community Kaggle mirror used for compute  

---

## Changelog

| Date | Update |
| --- | --- |
| 2026-07-25 | E1 baseline logged from Kaggle run `...T084249Z` |
| 2026-07-25 | E2 naive enhance logged from run `...T103126Z`; shortlist T0/T2/T4 |
| 2026-07-25 | E3 fixed-path logged (T0/T2/T4); T4 best on Lokrum test; T2 weights lost in Kaggle freeze but metrics retained from log |
