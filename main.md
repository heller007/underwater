# Detect, Enhance, or Defer

Cost-Aware Selective Enhancement for Reliable Marine-Debris Detection

*Complete Research and Experiment Execution Plan*

**Author Name**

## Abstract

Underwater image enhancement is frequently applied before object detection to compensate for color attenuation, haze, low contrast, and non-uniform illumination. Existing evidence, however, shows that perceptually improved images do not consistently improve detection and that fixed enhancement does not remove cross-site or cross-camera domain shift. This project studies a selective alternative: for every frame, a lightweight gate chooses between the raw image and several inexpensive enhancement operations, while a reliability module defers predictions whose expected detection error is high. The gate is trained from downstream detection utility rather than human-oriented visual quality. Experiments use the multi-domain SeaClear dataset for site-held-out evaluation and TrashCan for external validation. This document specifies the novelty boundary, datasets, transformations, model design, data splits, baselines, metrics, statistical tests, computational budget, coding-agent work packages, ablations, stopping criteria, and publication deliverables. The complete design is intended for a single 12 GB GPU or a standard Kaggle accelerator.

---

## Executive Decision

The paper should **not** claim a new underwater enhancement model. That space already contains reinforcement-learning enhancement policies, joint enhancement--detection modules, task-driven feature enhancement, deep unfolding, and robust underwater detectors. A broad comparison of enhancement methods has also shown that perceptual enhancement does not guarantee improved object detection. SeaClear's original validation similarly reported that fusion-based enhancement did not solve cross-site or cross-camera generalization.

The defensible paper is instead:

> **A domain-aware, cost-sensitive selective detection system that predicts whether a marine-debris frame should be processed raw, enhanced by a particular operator, or deferred because no available path is reliable.**

The strongest contribution is the combined evaluation of enhancement utility, enhancement harm, real domain shift, computational cost, calibration, and selective risk. The method should remain deliberately small; the experimental design and reliability analysis must carry the paper.

---

## Research Gap and Novelty Boundary

### Closest Work

The closest conceptual predecessor is Wang et al., which uses detector-score increments as rewards to learn a sequence of visual enhancement actions with reinforcement learning. The current project must be distinguished through all of the following design choices:

1. a one-step, inexpensive utility gate rather than a multi-step enhancement policy;
2. an explicit raw/no-operation action with a raw-favoring tie rule;
3. explicit transformation latency in the action utility;
4. a calibrated defer option, evaluated through risk--coverage curves;
5. site- and camera-held-out evaluation using real domains rather than only aggregate or synthetic-shift evaluation;
6. external testing between SeaClear and TrashCan; and
7. an oracle analysis that quantifies whether selective enhancement is learnable before presenting the proposed gate.

If these components are removed, the work risks becoming an incremental reimplementation of earlier task-driven enhancement research.

### Positioning Against Related Work

| Work | Primary contribution | Role in this project |
| --- | --- | --- |
| Wang et al. | Reinforcement-learning configuration of enhancement action sequences using detector improvement as reward. | Closest prior work. Use as the main conceptual baseline and clearly contrast one-step cost-aware routing, real domain holdouts, and calibrated deferral. |
| UnitModule | A 31K-parameter task-driven enhancement module trained jointly with an object detector. | Demonstrates that task-specific enhancement can be lightweight. It is an optional learned-enhancement baseline, not the proposed contribution. |
| LUIEO | Integrated lightweight enhancement and object detection using physical self-supervision. | Represents multi-task enhancement--detection models that optimize all images through one learned path. |
| SeaClear | Multi-domain shallow-water debris dataset and cross-site/camera validation; fixed fusion enhancement does not eliminate domain shift. | Provides the main dataset, domain structure, and direct motivation for selective rather than unconditional enhancement. |
| Awad et al. | Systematic study of multiple enhancement models and detectors. | Motivates the enhancement-harm analysis and the need for downstream rather than perceptual action targets. |
| AquaFeat | Task-driven multi-scale feature enhancement trained from detector loss. | Strong recent example of feature enhancement. Our method changes the input path per frame and measures selective reliability. |
| DGU-Net | Detection-guided deep unfolding for joint underwater enhancement and detection. | Recent high-capacity joint approach; cite to avoid claiming novelty in detection-guided enhancement itself. |
| RHCNet | Residual-guided feature enhancement and hierarchical feature calibration for robust underwater detection. | Recent detector-level robustness method. “Calibration” here concerns features, whereas this project studies probabilistic reliability and deferral. |
| Wille et al. | Domain-aware benchmarking based on real appearance, scene, and acquisition factors. | Supports domain-specific reporting instead of relying only on aggregate mAP. |
| Kuzucu et al. | Analysis and practical post-hoc calibration methods for object detectors. | Supplies calibration methodology and warns against reporting calibration error without accuracy-aware evaluation. |

---

## Formal Problem Statement

Let $x_i$ be an underwater frame, $y_i$ its object-detection annotation, and

$$\mathcal{T}=\{T_0,T_1,\ldots,T_K\}, \qquad T_0(x)=x,$$

the available image paths. Let $D$ be a shared object detector. The system must choose one transformation without evaluating every transformation through the detector at inference time.

For each image--action pair, define an image-level matching error

$$e_{i,k}=\frac{\sum_{j\in\mathrm{TP}_{i,k}}(1-\operatorname{IoU}_j) + N^{\mathrm{FP}}_{i,k}+N^{\mathrm{FN}}_{i,k}}{N^{\mathrm{TP}}_{i,k}+N^{\mathrm{FP}}_{i,k}+N^{\mathrm{FN}}_{i,k}+\epsilon},$$

inspired by Localization Recall Precision (LRP). Matching is performed class-wise at IoU $\geq 0.5$ using a confidence threshold fixed on validation data. Empty frames require a separately documented convention: zero error if the detector returns no false positives, and unit error otherwise.

The cost-aware target utility is

$$u_{i,k}=1-e_{i,k}-\lambda\widetilde{c}_k,$$

where $\widetilde{c}_k \in [0,1]$ is normalized preprocessing latency. The oracle action is

$$k_i^*=\operatorname*{arg\,max}_{k\in\{0,\ldots,K\}} u_{i,k}.$$

If the best enhanced action improves utility over the raw action by less than a small margin $\delta$, the oracle label is reset to raw. This tie rule prevents unnecessary enhancement for negligible gains.

At inference, a gate $G$ predicts the action from a low-resolution raw frame:

$$\hat{k}_i = G(x_i).$$

A reliability model $R$ predicts the expected error of the selected result. The system defers when

$$\operatorname{defer}(x_i)=\mathbb{I}\left[R\left(x_i,D(T_{\hat{k}_i}(x_i))\right)>\tau\right].$$

---

## Hypotheses and Success Criteria

1. **H1:** The best transformation varies across images and domains; therefore, an oracle selector exceeds the best fixed path by at least 2 absolute mAP points on the pilot split.
2. **H2:** A learned gate recovers at least 50% of the oracle improvement over the raw baseline.
3. **H3:** The learned gate reduces enhancement harm rate by at least 25% relative to the best fixed enhanced path.
4. **H4:** A cost-aware gate achieves better mAP per millisecond than always applying the most expensive enhancer.
5. **H5:** At 80% coverage, calibrated deferral reduces retained-frame error by at least 20% relative to accepting every frame.
6. **H6:** Human-oriented quality scores such as UCIQE and UIQM are weaker action selectors than the learned downstream-utility gate. This is plausible because their limitations for enhanced underwater imagery are documented in prior work.

These values are predeclared engineering targets, not guaranteed results. The paper must report confidence intervals and negative findings even if a target is missed.

---

## Datasets and Data Governance

### SeaClear: Primary Multi-Domain Dataset

SeaClear contains 8,610 images at 1920x1080 resolution, COCO-format annotations, 40 detailed categories, and folders corresponding to site--camera pairs. Images come from five sites: Bistrina, Jakljan, Lokrum, Slano, and Marseille. The data are released under CC BY 4.0. The primary experiments will collapse the original taxonomy into the official supercategories debris, bio, and robot.

### TrashCan: External Validation Dataset

TrashCan contains 7,212 annotated underwater images with bounding-box and segmentation annotations. It provides material-based and instance-based label configurations for trash together with biological objects and ROV observations. TrashCan must remain external: it may not be used to tune the SeaClear model, gate, confidence threshold, or defer threshold.

### Cross-Dataset Label Spaces

Two external evaluation label spaces will be reported:

1. **Binary primary mapping:** debris versus non-debris. This is the safest comparison because fine-grained category definitions differ.
2. **Three-class secondary mapping:** debris, biological, and robot, after manually auditing every source category. Ambiguous and unknown categories will be ignored rather than forced into an incorrect class.

| SeaClear | TrashCan | Common label |
| --- | --- | --- |
| Debris categories | Trash material/instance categories | Debris |
| Animal and vegetation categories | Animal and plant categories | Biological |
| ROV and robot-part categories | ROV categories | Robot |
| --- | Unknown or irreconcilable categories | Ignore |

### Mandatory Data Audit

Before training, generate the following artifacts:

- reports/data_audit.json
- reports/domain_manifest.csv
- a separate PDF/HTML audit report

These artifacts must contain:

- image and annotation counts per dataset, site, camera, sequence, and category;
- bounding-box size, aspect-ratio, and object-count distributions;
- number of negative frames and images with invalid annotations;
- class imbalance and rare categories;
- duplicate and near-duplicate analysis using exact hashes and perceptual hashes;
- example grids from every site--camera domain;
- missing files, corrupt files, polygons outside image bounds, and zero-area boxes; and
- the final class mapping with an explicit reason for every ignored category.

No model training should begin until the audit passes.

---

## Leakage-Free Split Protocol

Random image splitting is prohibited because consecutive underwater frames may be visually near-identical. The main evaluation will use leave-one-site-out (LOSO) testing over the five SeaClear sites.

For outer fold $s$:

1. all images from site $s$ form the untouched test set;
2. the remaining sites are grouped by camera and capture sequence;
3. within the source sites, groups are divided into 70% detector training, 15% gate-target training, and 15% calibration/validation;
4. no image group may appear in more than one split;
5. the detector is trained only on the detector-training split;
6. the frozen detector generates action utilities on the gate split;
7. the gate is trained only on the gate split; and
8. confidence thresholds, isotonic/Platt calibration, and defer threshold are fitted only on the calibration split.

If reliable sequence identifiers are unavailable, use site--camera folder, filename order, temporal adjacency, and perceptual-hash clusters to create groups. The split generator must be deterministic and save a CSV manifest for every fold.

---

## Candidate Image Paths

The core paper should use deterministic, inexpensive actions so that the gate's benefit is not confused with training a large enhancement model.

| ID | Path | Implementation specification | Expected cost |
| --- | --- | --- | --- |
| $T_0$ | Raw/no operation | Decode, resize, normalize only. | Lowest |
| $T_1$ | Gray-world white balance | Scale RGB channel means to a common mean; clip safely to the image range. | Very low |
| $T_2$ | LAB-CLAHE | Convert to LAB, apply CLAHE only to luminance using fixed clip limit and tile size, then convert back. | Low |
| $T_3$ | Adaptive gamma | Estimate gamma from mean luminance, clip gamma to a predeclared interval, and record the value per image. | Very low |
| $T_4$ | Fusion enhancement | Use the implementation associated with SeaClear, based on white-balanced and contrast-enhanced inputs. | Medium |

A pretrained learned enhancer may be added only as an optional experiment after the core pipeline works. It must not delay the raw/classical/fusion study.

For every transformation:

- write deterministic unit tests;
- preserve original annotations exactly;
- store parameters and software versions;
- measure latency on CPU and GPU with warm-up and at least 200 timed images; and
- create visual contact sheets to detect clipping, saturation, channel reversal, or halo artifacts.

---

## Model Design

### Shared Detector

Use a small, stable one-stage detector such as YOLOv8n initialized from COCO weights. The contribution is not the detector, so the exact package version and model checkpoint must be pinned. Use the same architecture for all main experiments.

Recommended initial configuration:

- input resolution: 640x640 with aspect-ratio-preserving padding;
- precision: automatic mixed precision;
- batch size: 8 initially, increase to 16 only after a memory test;
- epochs: 100 maximum with patience 20;
- optimizer: AdamW, initial learning rate $10^{-3}$, weight decay $5 \times 10^{-4}$;
- schedule: cosine decay with three warm-up epochs;
- initialization: identical pretrained checkpoint across experiments;
- seeds: 0 for development, then 0, 1, and 2 for final comparisons; and
- checkpoint selection: highest validation mAP50:95, never test performance.

Use geometric augmentation consistently across all models. Disable generic HSV color augmentation in the main controlled comparison because color transformations are the independent variable. Add HSV/UCRT-style color augmentation as a separate robustness baseline rather than silently mixing it into every experiment.

### Detector Training Regimes

Three detector regimes are required:

1. **Raw detector:** train and test on $T_0$.
2. **Fixed-path detectors:** for each shortlisted $T_k$, train and test consistently on $T_k$. This is the fairest fixed-enhancement baseline.
3. **Mixed-path detector:** sample one action uniformly for each training image. The gate uses this shared detector so it can accept any selected path.

Also evaluate the raw detector directly on enhanced test images. Label this clearly as test-time preprocessing without adaptation; do not confuse it with a consistently trained fixed pipeline.

### Utility Gate

The gate receives a 160x160 raw thumbnail. Use MobileNetV3-Small or a four-block depthwise-separable CNN. Concatenate its pooled visual representation with the following descriptors:

- per-channel mean and standard deviation;
- red/green and blue/green mean ratios;
- luminance mean, contrast, entropy, and saturation;
- variance of Laplacian as a blur indicator;
- UCIQE; and
- UIQM.

The gate outputs one predicted utility $\hat{u}_{i,k}$ for each action. Train with

$$\mathcal{L}_{\mathrm{gate}} = \frac{1}{K+1}\sum_{k=0}^{K} \operatorname{Huber}(\hat{u}_{i,k},u_{i,k}) + \alpha \, \operatorname{CE}(\operatorname{softmax}(\hat{\mathbf{u}}_i/T),k_i^*),$$

where $T$ is a softmax temperature and $\alpha$ is tuned only on the validation split. Use class weights or balanced sampling if oracle actions are imbalanced.

The final gate selects

$$\hat{k}_i=\begin{cases}0, & \max_{k>0}\hat{u}_{i,k}-\hat{u}_{i,0}<\delta, \\ \operatorname*{arg\,max}_{k}\hat{u}_{i,k}, & \text{otherwise}.\end{cases}$$

### Reliability Model and Deferral

The primary reliability target is the observed image error in the earlier equation after the gate-selected path. A small MLP will use:

- the complete predicted utility vector;
- the margin between the two best utilities;
- selected action and image-quality descriptors;
- number of detections;
- maximum, mean, and standard deviation of box confidence;
- median predicted box area; and
- fraction of boxes close to image boundaries.

Train the risk regressor using Huber loss, then compare Platt scaling and isotonic regression on the calibration split. Calibration must be assessed together with detection quality. The primary selective result is retained-image risk at coverage levels 100%, 95%, 90%, 80%, 70%, 60%, and 50%.

Per-box calibration using localization-aware calibration is an optional secondary analysis. It should not replace the frame-level defer experiment.

---

## Complete Experiment Matrix

| ID | Name | Execution | Required output |
| --- | --- | --- | --- |
| E0 | Data audit | Validate files/annotations, generate domain manifest, class maps, duplicate clusters, and site/camera statistics. | Audit report; no invalid split membership; final class-map JSON. |
| E1 | Raw baseline | Train raw YOLOv8n on one development fold, then evaluate source validation, held-out site, and TrashCan. | mAP, precision, recall, LRP components, per-domain results, failure examples. |
| E2 | Naive test-time enhancement | Apply every $T_k$ to the E1 test images without retraining the detector. | Direct evidence of when test-time enhancement helps or harms. |
| E3 | Fixed-path training | Train one detector consistently for each action on the development fold. | Fair comparison of fixed pipelines and shortlist the best two enhanced paths. |
| E4 | Mixed-path detector | Train a shared detector by sampling raw and shortlisted enhanced views. | Detector capable of processing all gate actions; comparison with E1/E3. |
| E5 | Oracle action study | Run all candidate paths through the frozen mixed detector on gate and test splits. Compute image errors, utilities, action distribution, oracle mAP, and harm rate. | Go/no-go result. Continue only if the oracle has a meaningful advantage and action diversity. |
| E6 | Quality-metric selectors | Select actions using UCIQE, UIQM, and simple degradation heuristics. | Non-learned selection baselines and correlation with downstream utility. |
| E7 | Learned utility gate | Train descriptor-only, CNN-only, and combined gates. Evaluate action accuracy, utility regret, detector mAP, harm rate, and latency. | Main method comparison and gate ablation. |
| E8 | Cost-aware routing | Sweep $\lambda$ and report accuracy--latency Pareto curves. Use measured, not theoretical, preprocessing time. | Pareto plot; recommended operating point selected on validation only. |
| E9 | Reliability and defer | Train frame-risk model, calibrate on held-out source sequences, and sweep $\tau$. | Risk--coverage curves, AURC, calibration plots, retained-error table. |
| E10 | LOSO evaluation | Repeat E1, E3(best paths), E4, E7, and E9 for all five held-out SeaClear sites. | Mean and per-site performance; analysis of clear, turbid, cluttered, and camera-shift domains. |
| E11 | External validation | Freeze everything and evaluate on TrashCan using binary and audited three-class mappings. | Cross-dataset mAP, risk--coverage, calibration deterioration, and qualitative failures. |
| E12 | Final multi-seed runs | Run seeds 0, 1, and 2 for raw, best fixed, mixed, and complete proposed system. | Mean, standard deviation, confidence intervals, and saved checkpoints. |
| E13 | Statistical analysis | Paired bootstrap by image and domain; compare proposed method to raw and best fixed baselines. | 95% confidence intervals, effect sizes, and corrected significance tests. |

---

## Evaluation Metrics

### Detection

Report:

- mAP50 and mAP50:95;
- precision, recall, and F1 at a validation-selected threshold;
- AP for debris, bio, and robot;
- AP for small, medium, and large objects;
- LRP error and its localization, false-positive, and false-negative components; and
- metrics separately for every site and camera, not only pooled values.

### Routing

Report:

- oracle action distribution per site;
- gate top-1 action accuracy;
- confusion matrix over actions;
- mean utility regret;
- fraction of oracle gain recovered;
- raw-selection rate and unnecessary-enhancement rate; and
- enhancement harm rate.

### Reliability

Report:

- risk--coverage curve and area under the risk--coverage curve (AURC);
- observed risk at the fixed coverage levels listed in E9;
- Brier score and reliability diagram for a binarized acceptable-frame event;
- calibration error before and after calibration; and
- false-high-confidence rate on held-out sites and TrashCan.

### Efficiency

Measure on a fixed machine after 50 warm-up iterations and at least 200 timed frames:

- preprocessing time per action;
- gate time;
- detector time;
- complete end-to-end latency;
- peak GPU memory;
- model parameters and approximate FLOPs; and
- expected latency under the gate's empirical action distribution.

---

## Ablation Plan

The complete method must be decomposed through the following ablations:

1. descriptor-only gate;
2. CNN-only gate;
3. combined gate;
4. action classification versus utility regression;
5. removal of UCIQE/UIQM descriptors;
6. removal of the raw action;
7. removal of the raw-favoring margin $\delta$;
8. removal of the cost term $\lambda\widetilde{c}_k$;
9. uniform versus action-balanced gate sampling;
10. gate resolutions 96, 160, and 224 pixels;
11. no calibration versus Platt versus isotonic calibration;
12. native detector confidence versus learned frame-risk deferral;
13. random image split versus leakage-resistant domain split, reported only to demonstrate inflation caused by random splitting; and
14. binary versus three-class external mapping.

Do not run every ablation over every LOSO fold initially. Run all ablations on the development fold, then carry only the important variants into final LOSO evaluation.

---

## Statistical Analysis

For final methods, train three seeds. Report mean and standard deviation across seeds. In addition, use paired bootstrap resampling because predictions are available on the same test images:

1. resample images with replacement within each held-out domain;
2. recompute the metric difference between the proposed and baseline method for 10,000 bootstrap replicates;
3. report the median difference and 95% percentile interval;
4. repeat using site--camera groups as the resampling unit to avoid overstating certainty from correlated video frames; and
5. apply Holm correction when simultaneously testing against multiple fixed-path baselines.

The primary comparison is complete system versus best fixed pipeline on pooled LOSO predictions. Secondary comparisons are versus raw and mixed-without-gate. Select the primary comparison before looking at final test results.

---

## Pilot Study and Go/No-Go Rules

The pilot should use one representative outer fold, one seed, 30--50 detector epochs, and approximately 1,000--2,000 images if full training is too slow.

### Continue with Routing If

- at least three actions are oracle-optimal for non-trivial image subsets;
- no single action wins on more than 90% of images;
- the oracle improves at least 2 absolute mAP points over the best fixed path, or produces an equally compelling reduction in LRP/harm rate; and
- low-resolution features can predict action utility better than UCIQE, UIQM, and majority-action baselines.

### Pivot If

If the oracle gap is below 1 mAP point, the routing mechanism cannot provide a strong contribution. Pivot to:

> **Degradation-Aware Confidence Calibration and Selective Marine-Debris Detection under Real Underwater Domain Shift.**

Retain the same detector, LOSO split, domain manifest, quality descriptors, and TrashCan external evaluation. Remove the action gate and focus on calibration, failure prediction, false-high-confidence analysis, and risk--coverage. This pivot is supported by recent domain-aware underwater benchmarking and the broader object-detection calibration literature.

---

## Implementation Blueprint for Coding Agents

### Repository Structure

```text
project/
  configs/
    data/
    detector/
    gate/
    calibration/
    experiments/
  data/
    raw/seaclear/
    raw/trashcan/
    processed/
    manifests/
  src/
    data/
    enhancement/
    detection/
    routing/
    calibration/
    evaluation/
    visualization/
  scripts/
    download_data.py
    audit_data.py
    build_splits.py
    prepare_labels.py
    benchmark_transforms.py
    train_detector.py
    generate_utilities.py
    train_gate.py
    train_reliability.py
    evaluate.py
    aggregate_results.py
  tests/
  runs/
  reports/
  paper_assets/
  requirements.txt
  README.md
```

All scripts must accept a configuration file, seed, input path, and output path. No scientific result should depend on an unrecorded notebook state.

### Work Packages

| WP | Task | Required implementation | Acceptance test |
| --- | --- | --- | --- |
| WP0 | Project scaffold | Configuration loading, logging, deterministic seeding, run manifests, dependency lock, and test runner. | One smoke command creates a run folder containing config, seed, environment, and Git revision. |
| WP1 | Dataset ingestion | Parse SeaClear COCO annotations and both TrashCan configurations; validate paths and convert to the detector format without changing source files. | Converted box counts match source counts after documented ignores. |
| WP2 | Audit and splits | Produce class/domain statistics, perceptual hashes, sequence groups, LOSO manifests, and contamination checks. | No group overlap; every image belongs to exactly one split in each fold. |
| WP3 | Enhancement library | Implement raw, gray-world, LAB-CLAHE, adaptive gamma, and Fusion wrappers with a common API and deterministic parameters. | Pixel-range tests pass; output dimensions unchanged; contact sheet approved. |
| WP4 | Quality and latency | Implement UCIQE, UIQM, simple descriptors, and reproducible CPU/GPU timing. | Match trusted reference implementations on a small test set within tolerance. |
| WP5 | Detector pipeline | Train/evaluate raw, fixed, and mixed detectors; export COCO-format predictions and per-image matching details. | Re-running evaluation from saved predictions reproduces all metrics. |
| WP6 | Utility generation | Match predictions to ground truth, compute the image-error metric, latency-normalized utilities, oracle labels, regret, and harm rate. | Hand-checked toy examples and empty-frame cases return expected values. |
| WP7 | Gate training | Descriptor-only, CNN-only, and combined utility-regression gates; checkpoints, action confusion matrix, and utility-regret evaluation. | Majority and random baselines included; no detector-test images used. |
| WP8 | Reliability | Frame-risk regressor, Platt/isotonic calibration, threshold sweep, AURC, and risk--coverage plots. | Coverage is monotonic and all thresholds are fitted without test labels. |
| WP9 | Experiment runner | YAML experiment matrix, resumable jobs, seed/fold loops, failure logging, and result aggregation. | A dry run prints all jobs; interrupted jobs resume without overwriting results. |
| WP10 | Paper assets | Generate publication-ready tables, plots, qualitative grids, and CSV source data directly from run artifacts. | Every paper number has a source run ID and can be regenerated by one command. |

### Example Agent Instructions

For each work package, instruct the coding agent to:

1. inspect existing files before editing;
2. implement only the assigned package and its documented interfaces;
3. add unit tests for edge cases;
4. avoid silently discarding images or annotations;
5. write a short README describing inputs, outputs, and example commands;
6. run a small smoke test before declaring completion; and
7. never use test labels for model choice, threshold selection, or calibration.

---

## Experiment Tracking and Reproducibility

Every run folder must contain:

- resolved configuration;
- seed, fold, model, action set, and class mapping;
- package versions, hardware name, and Git revision;
- train/validation/test manifests or their hashes;
- checkpoints and epoch history;
- COCO-format predictions;
- raw metric JSON/CSV files;
- transformation latency measurements; and
- generated figures with the script and source data used to create them.

Use one experiment naming scheme, for example `fold-lokrum_model-mixed_gate-combined_seed-0`. Never reuse a result directory for a different configuration.

---

## Compute-Aware Execution Funnel

The full Cartesian product of actions, folds, and seeds is unnecessary. Use the following funnel:

1. **Smoke stage:** 100 images, 2 epochs, one fold; validate code.
2. **Pilot stage:** one fold, one seed, 30--50 epochs, all actions.
3. **Shortlist stage:** retain raw, the two best fixed actions, and mixed training.
4. **Domain stage:** run shortlisted methods over all five LOSO folds using seed 0.
5. **Final stage:** run seeds 1 and 2 only for raw, best fixed, mixed, and complete proposed system.

Approximate planning budget:

| Stage | GPU work | Purpose |
| --- | --- | --- |
| Smoke and audit | Less than 1 GPU hour | Verify data, memory, and output format. |
| Pilot detectors | Approximately 10--25 GPU hours | Compare all paths and estimate oracle gap. |
| Gate and reliability development | Approximately 5--15 GPU hours | Small models; most cost is generating frozen-detector predictions. |
| Five LOSO folds, seed 0 | Approximately 30--60 GPU hours | Main domain study. |
| Final additional seeds | Approximately 40--80 GPU hours | Statistical stability for shortlisted methods only. |

These are scheduling estimates, not results. A 12 GB GPU is sufficient for the proposed model sizes; total wall-clock time is the larger constraint.

---

## Six-Week Schedule

| Week | Engineering | Research output |
| --- | --- | --- |
| 1 | WP0--WP4: download, parse, audit, split, enhancements, quality metrics. | Dataset section, domain table, example images, leakage report. |
| 2 | WP5: raw/fixed/mixed pilot detectors. | Baseline table and enhancement impact analysis. |
| 3 | WP6--WP7: oracle study and learned gate. | Go/no-go decision, oracle figure, gate ablation. |
| 4 | WP8: calibration and defer system; first external test. | Risk--coverage and calibration results. |
| 5 | LOSO folds and final multi-seed shortlist. | Final quantitative tables, bootstrap intervals, qualitative errors. |
| 6 | Paper writing, related work, figures, reproducibility cleanup. | Complete manuscript and supplementary repository. |

---

## Required Paper Figures and Tables

Plan the manuscript around the following evidence:

1. system diagram: raw/enhance/defer pipeline;
2. domain grid: representative frames from all SeaClear site--camera pairs;
3. oracle action distribution by site;
4. oracle gain versus best fixed path;
5. scatter plots of UCIQE/UIQM change versus detection-utility change;
6. gate confusion matrix;
7. accuracy--latency Pareto plot over $\lambda$;
8. risk--coverage curves for native confidence and learned risk;
9. main LOSO results table;
10. cross-dataset SeaClear-to-TrashCan table;
11. gate and defer ablation table;
12. examples where enhancement helps, hurts, is correctly skipped, and should have been deferred; and
13. TIDE/LRP-style decomposition showing whether improvements come from localization, false positives, or false negatives.

---

## Reading Plan

Read papers in this order:

1. **SeaClear dataset paper**: learn the site/camera domains, label taxonomy, original baselines, and enhancement result.
2. **RL visual enhancement**: understand the closest action-selection method and write a precise novelty comparison.
3. **Enhancement impact study**: identify which enhancers and detectors were already compared and how enhancement failures were measured.
4. **UnitModule**: understand task-guided lightweight enhancement and its computational claims.
5. **AquaFeat and DGU-Net**: cover recent downstream-guided enhancement methods.
6. **Why Domain Matters**: adopt domain-specific reporting and avoid relying on pooled performance.
7. **UCIQE/UIQM and their limitations**: justify why perceptual quality is a baseline rather than the optimization objective.
8. **LRP**: implement interpretable per-image error targets and decompose detector failures.
9. **Object-detection calibration**: design calibration and reliability evaluation without misleading metrics.

---

## Expected Contributions

If experiments support the hypotheses, the paper can claim:

1. a domain-aware benchmark of per-image enhancement utility for marine-debris detection;
2. a lightweight gate that predicts downstream action utility and explicitly retains the raw path when enhancement is unnecessary;
3. a cost-aware routing formulation with measured accuracy--latency trade-offs;
4. a calibrated defer mechanism for reliable operation under unseen underwater domains; and
5. cross-site, cross-camera, and cross-dataset evidence with oracle, harm-rate, calibration, and failure-decomposition analyses.

Do not claim state of the art unless the complete method is compared under an identical split, label mapping, image resolution, and detector protocol.

---

## Limitations and Threats to Validity

- The action set may not contain a transformation capable of recovering severely lost visual evidence.
- Perceptual quality descriptors are imperfect and may encode site or camera shortcuts.
- The SeaClear sites are limited and do not represent every depth, habitat, camera, or water type.
- TrashCan and SeaClear differ in environment and taxonomy, making three-class mapping approximate.
- Annotation incompleteness can incorrectly label true detections as false positives, especially for small biological objects.
- Frame-level defer decisions require an operational fallback, such as human review, slower reprocessing, or another sensor.
- Training separate fixed-path models increases compute; the staged funnel is required to keep the study feasible.
- A successful oracle does not guarantee a low-resolution gate can infer the correct action under unseen domains.

---

## Final Checklist Before Writing Results

1. Dataset license and citations recorded.
2. No duplicate/sequence leakage across splits.
3. Test labels never used for training, thresholding, or calibration.
4. Raw, fixed, mixed, heuristic, oracle, gate, and defer baselines all present.
5. Per-site results reported alongside pooled results.
6. Enhancement latency measured on the same machine.
7. At least three seeds for shortlisted final methods.
8. Confidence intervals and effect sizes reported.
9. Negative and failure cases included.
10. Every table and figure regenerates from saved artifacts.
11. Novelty is contrasted explicitly with Wang et al., UnitModule, AquaFeat, DGU-Net, and SeaClear's enhancement experiment.
12. Claims are restricted to the evidence produced by the stated split and label mapping.

---

## Conclusion

The experiment should answer a narrow but practically important question: when should an underwater debris detector enhance an image, preserve the raw frame, or decline to make an automatic decision? The project is viable on a 12 GB GPU because the detector and gate are deliberately small, the enhancement operations are mostly deterministic, and expensive experiments are shortlisted through an oracle pilot. Publication strength will come from a leakage-resistant domain protocol, comparison to the closest enhancement selection work, calibrated selective evaluation, and transparent reporting of when enhancement helps and when it causes harm.

---

## References

- Duraš et al. (2024). "A Dataset for Detection and Segmentation of Underwater Marine Debris in Shallow Waters." Scientific Data.
- Duraš et al. (2024). "Seaclear Marine Debris Detection & Segmentation Dataset." 4TU.ResearchData.
- Hong, Fulton, and Sattar (2020). "TrashCan: A Semantically-Segmented Dataset towards Visual Detection of Marine Debris." arXiv preprint arXiv:2007.08097.
- Wang et al. (2023). "A Reinforcement Learning Paradigm of Configuring Visual Enhancement for Object Detection in Underwater Scenes." IEEE Journal of Oceanic Engineering.
- Liu et al. (2024). "UnitModule: A Lightweight Joint Image Enhancement Module for Underwater Object Detection." Pattern Recognition.
- Li et al. (2024). "LUIEO: A Lightweight Model for Integrating Underwater Image Enhancement and Object Detection." arXiv preprint arXiv:2412.07009.
- Awad et al. (2024). "Beneath the Surface: The Role of Underwater Image Enhancement in Object Detection." arXiv preprint arXiv:2411.14626.
- Silva et al. (2025). "AquaFeat: A Features-Based Image Enhancement Model for Underwater Object Detection." arXiv preprint arXiv:2508.12343.
- Yu, Vo, and Lee (2026). "Detection-Guided Deep Unfolding for Joint Underwater Image Enhancement and Object Detection." IEEE Access.
- Wang et al. (2026). "RHCNet: Residual-Guided Hierarchical Calibration Network for Robust Underwater Object Detection." CVPR.
- Wille et al. (2026). "Why Domain Matters: Domain-Aware Benchmarking of Underwater Object Detection and Annotation Quality." arXiv preprint arXiv:2607.10575.
- Ancuti et al. (2012). "Enhancing Underwater Images and Videos by Fusion." CVPR.
- Yang and Sowmya (2015). "An Underwater Color Image Quality Evaluation Metric." IEEE Transactions on Image Processing.
- Panetta, Gao, and Agaian (2016). "Human-Visual-System-Inspired Underwater Image Quality Measures." IEEE Journal of Oceanic Engineering.
- Li and Cavallaro (2022). "On the Limits of Perceptual Quality Measures for Enhanced Underwater Images." arXiv preprint arXiv:2207.05470.
- Oksuz et al. (2018). "Localization Recall Precision (LRP): A New Performance Metric for Object Detection." ECCV.
- Guo et al. (2017). "On Calibration of Modern Neural Networks." ICML.
- Pathiraja et al. (2023). "Multiclass Confidence and Localization Calibration for Object Detection." CVPR.
- Kuzucu et al. (2024). "On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines." arXiv preprint arXiv:2405.20459.
