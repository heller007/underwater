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
| E4 | `fold-lokrum_model-mixed_exp-e4_mixed_path_seed-0_20260725T144908Z` | Mixed T0+T4; see `e4_results.json` |
| E5 | `fold-lokrum_model-oracle_exp-e5_oracle_seed-0_20260725T170204Z` | **PIVOT_CALIBRATION** (gap 0.008) |

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

## E4 — Mixed-path detector (shared \(T_0\)+\(T_4\))

### Protocol
- Each image assigned one action \(\in\{T_0,T_4\}\) deterministically (~50/50)
- Train one YOLOv8n on mixed train; early-stop on mixed val
- Eval: mixed val/test + pure \(T_0\) Lokrum test + pure \(T_4\) Lokrum test
- Run id: `fold-lokrum_model-mixed_exp-e4_mixed_path_seed-0_20260725T144908Z`
- Mix counts (train): T0 3073 / T4 3128

### Metrics

| Eval setting | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| mixed val | 0.457 | 0.311 |
| mixed test (Lokrum) | 0.179 | 0.131 |
| **pure T0 test** (Lokrum) | **0.183** | **0.133** |
| pure T4 test (Lokrum) | 0.177 | 0.128 |

**Per-class test mAP50-95 (E4 on T0 test):** debris 0.023, bio 0.002, robot 0.376  

### Comparison on Lokrum test (mAP50-95)

| System | Input at test | mAP50-95 |
| --- | --- | ---: |
| E1 raw | \(T_0\) | 0.041 |
| E3 fixed T4 | \(T_4\) | **0.157** |
| E4 mixed | \(T_0\) | 0.133 |
| E4 mixed | \(T_4\) | 0.128 |

### Findings (paper-ready)
1. **Mixed training works for both paths:** E4 on raw Lokrum (**0.133**) ≫ E1 raw (**0.041**); on fusion test it is close to, but below, specialist E3-T4 (**0.128** vs **0.157**).
2. **Specialist still wins when the path is fixed to T4** — expected; the gate/oracle should try to recover part of that +0.029 gap by choosing T4 only when useful.
3. E4 is the correct **shared backbone** for E5 oracle / E7 gate (can ingest raw or fusion).

### Claim boundary
E4 is not the final method; it enables selective routing without training separate heads for each action.

---

## E5 — Oracle study (frozen E4 mixed)

### Protocol
- **Detector:** frozen E4 mixed `best.pt`
- **Actions:** \(T_0\), \(T_4\)
- **Splits:** gate + test (Lokrum); per-image utilities → oracle labels
- **Go rule:** gap ≥ 0.02 mAP50, both actions ≥10%, none >90%

### Test (Lokrum) outcome

| Quantity | Value |
| --- | ---: |
| Oracle mAP50 | 0.0871 |
| Best fixed mAP50 | 0.0789 |
| **Gap (oracle − best fixed)** | **0.0082** (~0.8 mAP pts) |
| Oracle action counts | T0 **830** (85%), T4 **142** (15%) |
| **Decision** | **PIVOT_CALIBRATION** |

Run: `...T170204Z` · artifacts: `e5/e5_results.json`

### Finding (paper-ready)
Selective enhancement oracle beats the best fixed path by **&lt; 1 mAP50 point** on Lokrum. Per `main.md` go/no-go, a learned action gate is unlikely to add a strong contribution. **Pivot** to degradation-aware confidence calibration + selective deferral (risk–coverage), keeping the same detector, LOSO protocol, and descriptors.

### Claim boundary
E5 rejects H1-style routing for this shortlist/detector; it does **not** invalidate E1–E4 domain-shift or enhancement findings.

---

## Later stages (revised after E5 pivot)

| ID | Name | Goal | Status |
| --- | --- | --- | --- |
| E6 | Quality selectors | UCIQE/UIQM/heuristic baselines (H6) | **Ready to run** (ablation / negative evidence) |
| E7 | Learned utility gate | Descriptor/CNN/combined | **Ready to run** (expect limited gain) |
| E8 | Cost-aware λ sweep | Accuracy–latency | Optional / skip if E7 weak |
| E9 | Reliability / defer | Risk–coverage, calibration | After E6/E7 (main pivot path) |
| E10 | Full LOSO | All five held-out sites | Later |
| E11 | TrashCan external | Frozen system | Later |
| E12–E13 | Multi-seed + stats | CIs / bootstrap | Later |

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
- [x] E3 fixed-path T0/T2/T4  
- [x] E4 mixed T0+T4  
- [x] E5 oracle → PIVOT_CALIBRATION  
- [ ] Save `e*_results.json` + `best.pt` per stage to a Kaggle Dataset / local archive  
- [ ] Cite SeaClear; note community Kaggle mirror used for compute  

---

## Changelog

| Date | Update |
| --- | --- |
| 2026-07-25 | E1 baseline logged from Kaggle run `...T084249Z` |
| 2026-07-25 | E2 naive enhance logged from run `...T103126Z`; shortlist T0/T2/T4 |
| 2026-07-25 | E3 fixed-path logged (T0/T2/T4); T4 best on Lokrum test; T2 weights lost in Kaggle freeze but metrics retained from log |
| 2026-07-25 | E4 mixed-path logged; strong on both T0/T4 test vs E1; below E3-T4 specialist on T4 test |
| 2026-07-25 | E5 oracle logged (`...T170204Z`); gap 0.008 → **PIVOT_CALIBRATION**; skip E6–E8 routing |
