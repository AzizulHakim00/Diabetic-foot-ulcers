# DFU-ImageGuard

Source code accompanying the manuscript **“DFU-ImageGuard: An Explainable and Reliability-Validated Framework for Diabetic Foot Ulcer Image Classification.”**

The repository implements the complete paper workflow: similarity-aware data auditing, group-locked candidate screening, fold-only probability calibration, sensitivity-oriented threshold selection, repeatability analysis, selective referral, frozen external evaluation, overlap-filtered sensitivity analysis, and Grad-CAM inspection.

## Experimental design

- Primary cohort: 1,055 images from the explicit `Patches` hierarchy of the public Diabetic Foot Ulcer dataset.
- Leakage-control unit: exact- and near-similarity groups; no patient identifiers are available.
- Candidates: ConvNeXtV2-Tiny, DenseNet121, and MobileNetV3-Large.
- Evaluation: five `StratifiedGroupKFold` outer folds and seeds 2026, 2027, and 2028.
- Training: ImageNet initialization, 224 × 224 inputs, weighted binary cross-entropy, AdamW, cosine annealing, mixed precision, gradient clipping, and selection-partition early stopping.
- Calibration: temperature scaling fitted only on the calibration role within each outer-training fold.
- Decision rule: maximize specificity subject to calibration sensitivity of at least 0.95, with balanced accuracy as the fallback.
- External cohort: the public Wound Classification dataset, evaluated with frozen model, calibration, and threshold state.

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── scripts/
│   ├── run_training.py
│   ├── analyze_trustworthiness.py
│   ├── paper_statistics.py
│   ├── external_validation.py
│   └── external_sensitivity_analysis.py
└── src/
    ├── config_data.py
    ├── evaluation.py
    ├── models_training.py
    ├── reliable_analysis.py
    ├── reliable_models.py
    ├── reliable_runner_v2.py
    └── reliable_storage_v2.py
```

## Data

The code downloads the two public datasets through `kagglehub`:

- Primary: `laithjj/diabetic-foot-ulcer-dfu`
- External: `ibrahimfateen/wound-classification`

Only the explicit `Patches` class hierarchy is accepted for the primary analysis. Ambiguous root folders and transfer-learning subsets are not used. Dataset images are not redistributed in this repository.

## Environment

Python 3.10 or newer is recommended. A CUDA-capable runtime is required by the primary training runner.

```bash
python -m pip install -r requirements.txt
```

## Primary experiment

The following command runs the 45 matched candidate trials and writes resumable checkpoints, predictions, calibration parameters, metrics, and provenance records:

```bash
python scripts/run_training.py \
  --run-id DFU_IMAGEGUARD_PAPER \
  --drive-root /content/drive/MyDrive/DFU-ImageGuard
```

The default configuration uses all five outer folds, all three fixed seeds, and all three candidate architectures.

## Reliability analysis

After the primary experiment has produced its final prediction export:

```bash
python scripts/analyze_trustworthiness.py
```

This stage produces aggregate model performance, probability-reliability tables, operating-threshold audits, hard-case consistency, selective-referral results, group-aware resampling summaries, and representative Grad-CAM outputs.

The manuscript’s matched Wilcoxon tests and ten-test Holm adjustment can be regenerated from the fold/seed metric table:

```bash
python scripts/paper_statistics.py \
  /content/drive/MyDrive/DFU-ImageGuard/runs/DFU_IMAGEGUARD_PAPER/tables/fold_seed_metrics.csv \
  --output paired_wilcoxon_holm.csv
```

## External evaluation

Frozen external evaluation uses the checkpoint inventory and primary out-of-fold prediction tables:

```bash
python scripts/external_validation.py
python scripts/external_sensitivity_analysis.py
```

The first script performs cross-dataset fingerprint auditing and frozen inference on the full external cohort. The second regenerates the analysis after excluding conservative near-pHash candidates. No external fitting, recalibration, threshold selection, or model selection is performed.

For a Kaggle runtime, set `DFU_FROZEN_BUNDLE_HANDLE` to the handle containing the frozen checkpoint bundle. In Google Colab, the scripts read the corresponding run artifacts from Google Drive.

## Manuscript-to-code map

| Manuscript component | Code |
|---|---|
| Dataset audit, hashing, similarity groups, group-locked folds | `src/config_data.py` |
| Model construction, augmentation, and loading | `src/reliable_models.py`, `src/models_training.py` |
| Matched training, early stopping, calibration, thresholds, outer-fold inference | `src/reliable_runner_v2.py`, `src/evaluation.py` |
| Internal performance, calibration, repeatability, referral, and Grad-CAM | `scripts/analyze_trustworthiness.py` |
| Paired Wilcoxon tests and Holm correction | `scripts/paper_statistics.py` |
| Frozen cross-dataset evaluation | `scripts/external_validation.py` |
| Overlap-filtered external sensitivity analysis | `scripts/external_sensitivity_analysis.py` |

## Scope

This is retrospective research software for reproducibility and methodological review. It is not a medical device and must not be used to replace clinical assessment. Similarity-aware grouping controls detectable related-image leakage but does not establish patient-disjoint evaluation.
