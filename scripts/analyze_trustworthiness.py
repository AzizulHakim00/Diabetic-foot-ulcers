from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

VERSION = "DFU_PHASE2_TRUSTWORTHINESS_FULL_V1_20260811"
RUN_ID = "RELIABLE_DFU_CV_V3_MISSING38"
EXPECTED_TRIALS = 45
EXPECTED_OOF_ROWS = 9495
EXPECTED_UNIQUE_IMAGES = 1055
EXPECTED_MODELS = ("convnextv2_tiny", "densenet121", "mobilenetv3_large")
EXPECTED_SEEDS = (2026, 2027, 2028)
EXPECTED_FOLDS = (1, 2, 3, 4, 5)
EXPECTED_FOLD_SIZES = {1: 209, 2: 206, 3: 217, 4: 214, 5: 209}
EXPECTED_LOCKED_SPLIT_SHA256 = "d392c8dd1458b5054a74ceafdc2d63ad88e5f39bceec2558338a0980170c2eb1"
EXPECTED_INPUT_NAME = "DFU_REPAIR45_FINAL_EXPORT.zip"
PHASE2_DIRNAME = "PHASE2_FULL"
BOOTSTRAP_REPS = 1000
LEAKAGE_SIM_REPS = 1000
RNG_SEED = 2026
TARGET_SENSITIVITY = 0.95
REFERRAL_RATES = (0.00, 0.05, 0.10, 0.15, 0.20)
COVERAGES = tuple(1.0 - r for r in REFERRAL_RATES)
XAI_CASE_TYPES = (
    "confident_true_positive",
    "confident_true_negative",
    "persistent_false_negative",
    "persistent_false_positive",
    "uncertain_dfu",
    "uncertain_normal",
)
FORBIDDEN_CALL_NAMES = ("train_trial", "train_model", "make_outer_folds", "assign_duplicate_groups")

MODEL_NAMES = {
    "convnextv2_tiny": "ConvNeXtV2-Tiny",
    "densenet121": "DenseNet121",
    "mobilenetv3_large": "MobileNetV3-Large",
}
MODEL_CANDIDATES = {
    "convnextv2_tiny": (
        "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "convnextv2_tiny.fcmae_ft_in1k",
        "convnextv2_tiny.fcmae",
    ),
    "densenet121": ("densenet121.ra_in1k", "densenet121"),
    "mobilenetv3_large": ("mobilenetv3_large_100.ra_in1k", "mobilenetv3_large_100"),
}


def now_ns() -> int:
    return time.time_ns()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def model_name(key: str) -> str:
    return MODEL_NAMES.get(str(key), str(key))


def ensure_packages(packages: dict[str, str]) -> None:
    import importlib.util
    missing = [pip_name for import_name, pip_name in packages.items() if importlib.util.find_spec(import_name) is None]
    if missing:
        print("Installing Phase-2 analysis dependencies only:", ", ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


def configure_matplotlib() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 400,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig, png_dir: Path, pdf_dir: Path, stem: str) -> None:
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_dir / f"{stem}.png", bbox_inches="tight", dpi=400)
    fig.savefig(pdf_dir / f"{stem}.pdf", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def snapshot_trial_tree(run_root: Path | None) -> dict[str, tuple[int, int]]:
    if run_root is None:
        return {}
    trials = Path(run_root) / "trials"
    if not trials.exists():
        return {}
    snap: dict[str, tuple[int, int]] = {}
    for p in sorted(trials.rglob("*")):
        if p.is_file():
            st = p.stat()
            snap[str(p.relative_to(trials))] = (int(st.st_size), int(st.st_mtime_ns))
    return snap


def static_self_audit() -> dict[str, Any]:
    import ast
    source_path = Path(__file__) if "__file__" in globals() else None
    text = source_path.read_text(encoding="utf-8") if source_path and source_path.exists() else ""
    tree = ast.parse(text) if text else ast.parse("")
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALL_NAMES:
            found.append(func.id)
        elif isinstance(func, ast.Attribute):
            if func.attr in FORBIDDEN_CALL_NAMES:
                found.append(func.attr)
            if func.attr == "step" and isinstance(func.value, ast.Name) and func.value.id == "optimizer":
                found.append("optimizer.step")
    found = sorted(set(found))
    if found:
        raise RuntimeError(f"Phase-2 source contains forbidden training call(s): {found}")
    return {
        "status": "PASS",
        "forbidden_training_calls_present": found,
        "training_performed": False,
        "outer_folds_regenerated": False,
        "duplicate_groups_regenerated": False,
    }


def load_export_zip(zip_path: Path) -> dict[str, Any]:
    required = {
        "tables/REPAIR45_RECONCILIATION.csv",
        "REPAIR45_FINAL_VERIFICATION.json",
        "tables/fold_seed_metrics.csv",
        "tables/all_oof_predictions.csv",
        "tables/selective_prediction.csv",
        "tables/error_audit.csv",
        "reliable_dfu_reproducibility_repair45.pkl",
    }
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(f"Final export missing required members: {missing}")
        frames = {
            "reconciliation": pd.read_csv(io.BytesIO(zf.read("tables/REPAIR45_RECONCILIATION.csv"))),
            "metrics": pd.read_csv(io.BytesIO(zf.read("tables/fold_seed_metrics.csv"))),
            "oof": pd.read_csv(io.BytesIO(zf.read("tables/all_oof_predictions.csv"))),
            "selective_original": pd.read_csv(io.BytesIO(zf.read("tables/selective_prediction.csv"))),
            "error_original": pd.read_csv(io.BytesIO(zf.read("tables/error_audit.csv"))),
            "final_verification": json.loads(zf.read("REPAIR45_FINAL_VERIFICATION.json").decode("utf-8")),
            "original_pkl_bytes": zf.read("reliable_dfu_reproducibility_repair45.pkl"),
        }
        if "tables/paired_bootstrap.json" in names:
            frames["old_paired_bootstrap"] = json.loads(zf.read("tables/paired_bootstrap.json").decode("utf-8"))
        else:
            frames["old_paired_bootstrap"] = None
    return frames


def integrity_gate(data: dict[str, Any], input_zip: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rec = data["reconciliation"].copy()
    met = data["metrics"].copy()
    oof = data["oof"].copy()
    final = data["final_verification"]

    if final.get("status") != "PASS":
        raise RuntimeError(f"Primary final verification is not PASS: {final.get('status')}")
    if int(final.get("completed_unique_trials", -1)) != EXPECTED_TRIALS:
        raise RuntimeError("Primary final verification does not contain 45 completed trials")
    if int(final.get("prediction_rows", -1)) != EXPECTED_OOF_ROWS:
        raise RuntimeError("Primary final verification prediction row count changed")
    if str(final.get("locked_split_sha256")) != EXPECTED_LOCKED_SPLIT_SHA256:
        raise RuntimeError("Locked split SHA256 changed")
    if len(met) != EXPECTED_TRIALS:
        raise RuntimeError(f"fold_seed_metrics has {len(met)} rows, expected {EXPECTED_TRIALS}")
    if len(oof) != EXPECTED_OOF_ROWS:
        raise RuntimeError(f"OOF has {len(oof)} rows, expected {EXPECTED_OOF_ROWS}")

    for col in ("model_key", "seed", "outer_fold", "image_id", "group_id", "label", "pred", "prob_raw", "prob_calibrated"):
        if col not in oof.columns:
            raise RuntimeError(f"OOF missing required column {col}")
    for col in ("model_key", "seed", "outer_fold"):
        if col not in met.columns:
            raise RuntimeError(f"Metrics missing required column {col}")

    oof["model_key"] = oof["model_key"].astype(str)
    oof["seed"] = pd.to_numeric(oof["seed"], errors="raise").astype(int)
    oof["outer_fold"] = pd.to_numeric(oof["outer_fold"], errors="raise").astype(int)
    oof["label"] = pd.to_numeric(oof["label"], errors="raise").astype(int)
    oof["pred"] = pd.to_numeric(oof["pred"], errors="raise").astype(int)
    oof["image_id"] = oof["image_id"].astype(str)
    oof["group_id"] = oof["group_id"].astype(str)

    if set(oof["model_key"].unique()) != set(EXPECTED_MODELS):
        raise RuntimeError(f"Unexpected model set: {sorted(oof['model_key'].unique())}")
    if set(oof["seed"].unique()) != set(EXPECTED_SEEDS):
        raise RuntimeError(f"Unexpected seed set: {sorted(oof['seed'].unique())}")
    if set(oof["outer_fold"].unique()) != set(EXPECTED_FOLDS):
        raise RuntimeError(f"Unexpected fold set: {sorted(oof['outer_fold'].unique())}")
    if not set(oof["label"].unique()).issubset({0, 1}) or not set(oof["pred"].unique()).issubset({0, 1}):
        raise RuntimeError("Binary label/prediction domain violation")
    for col in ("prob_raw", "prob_calibrated"):
        vals = pd.to_numeric(oof[col], errors="coerce").to_numpy(float)
        if not np.isfinite(vals).all() or (vals < 0).any() or (vals > 1).any():
            raise RuntimeError(f"Invalid probability values in {col}")

    dup = oof.duplicated(["model_key", "seed", "image_id"], keep=False)
    if dup.any():
        raise RuntimeError("Duplicate model-seed-image OOF prediction rows detected")

    counts = oof.groupby(["model_key", "seed"]).agg(rows=("image_id", "size"), unique_images=("image_id", "nunique"))
    if not ((counts["rows"] == EXPECTED_UNIQUE_IMAGES) & (counts["unique_images"] == EXPECTED_UNIQUE_IMAGES)).all():
        raise RuntimeError("Not every model-seed contains exactly 1055 unique images")

    image_meta = oof[["image_id", "group_id", "label", "outer_fold"] + (["relative_path"] if "relative_path" in oof.columns else [])].copy()
    consistency_cols = [c for c in image_meta.columns if c != "image_id"]
    for col in consistency_cols:
        n = image_meta.groupby("image_id")[col].nunique(dropna=False)
        if (n > 1).any():
            raise RuntimeError(f"Historical OOF metadata inconsistent by image_id for {col}")
    unique_images = image_meta.drop_duplicates("image_id").reset_index(drop=True)
    if len(unique_images) != EXPECTED_UNIQUE_IMAGES:
        raise RuntimeError(f"Unique image count {len(unique_images)} != {EXPECTED_UNIQUE_IMAGES}")
    fold_sizes = unique_images.groupby("outer_fold").size().sort_index().to_dict()
    if fold_sizes != EXPECTED_FOLD_SIZES:
        raise RuntimeError(f"Fold sizes changed: {fold_sizes}")
    if unique_images.groupby("group_id")["outer_fold"].nunique().max() != 1:
        raise RuntimeError("Similarity group crosses outer folds in frozen primary experiment")

    rec_id = rec[["model_key", "seed", "outer_fold"]].drop_duplicates()
    if len(rec_id) != EXPECTED_TRIALS:
        raise RuntimeError("Reconciliation does not contain 45 unique identities")

    proof = {
        "status": "PASS",
        "version": VERSION,
        "input_zip": str(input_zip),
        "input_zip_sha256": sha256_file(input_zip),
        "completed_unique_trials": EXPECTED_TRIALS,
        "oof_rows": EXPECTED_OOF_ROWS,
        "unique_images": EXPECTED_UNIQUE_IMAGES,
        "models": sorted(EXPECTED_MODELS),
        "seeds": list(EXPECTED_SEEDS),
        "fold_sizes": {str(k): int(v) for k, v in fold_sizes.items()},
        "locked_split_sha256": EXPECTED_LOCKED_SPLIT_SHA256,
        "wrong_image_to_fold_assignments": 0,
        "cross_fold_similarity_group_leakage": 0,
        "verified_at_ns": now_ns(),
    }
    print("PHASE2 integrity gate: PASS | 45 trials | 9495 OOF | 1055 images | frozen fold/group membership")
    return unique_images, proof


def metric_dict(y: np.ndarray, p: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision_score,
        average_precision_score,
        roc_auc_score,
        confusion_matrix,
    )
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = np.asarray(pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    eps = 1e-12
    ece = expected_calibration_error(y, p, bins=15)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, eps, 1 - eps), labels=[0, 1])),
        "ece": float(ece),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "n": int(len(y)),
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    total = len(y)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += (mask.sum() / total) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def calibration_curve_equal_frequency(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"y": np.asarray(y, int), "p": np.asarray(p, float)}).sort_values("p", kind="stable").reset_index(drop=True)
    q = min(bins, len(frame))
    frame["bin"] = pd.qcut(np.arange(len(frame)), q=q, labels=False, duplicates="drop")
    out = frame.groupby("bin", as_index=False).agg(
        n=("y", "size"),
        mean_probability=("p", "mean"),
        observed_rate=("y", "mean"),
    )
    return out


def primary_performance(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    metric_cols = [
        "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "mcc",
        "roc_auc", "pr_auc", "brier", "log_loss", "ece",
    ]
    rows = []
    for key in EXPECTED_MODELS:
        g = metrics.loc[metrics["model_key"] == key]
        row: dict[str, Any] = {"model_key": key, "model_name": model_name(key), "n_fold_seed_trials": int(len(g))}
        for m in metric_cols:
            vals = pd.to_numeric(g[m], errors="raise").to_numpy(float)
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_sd"] = float(vals.std(ddof=1))
            row[f"{m}_min"] = float(vals.min())
            row[f"{m}_max"] = float(vals.max())
        rows.append(row)
    return pd.DataFrame(rows)


def weighted_metric_dict(y: np.ndarray, p: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = np.asarray(pred, dtype=int)
    w = np.asarray(weights, dtype=float)
    active = w > 0
    y = y[active]; p = p[active]; pred = pred[active]; w = w[active]
    if len(y) == 0 or np.sum(w) <= 0:
        raise ValueError("weighted metric received no active rows")
    tp = float(w[(y == 1) & (pred == 1)].sum())
    tn = float(w[(y == 0) & (pred == 0)].sum())
    fp = float(w[(y == 0) & (pred == 1)].sum())
    fn = float(w[(y == 1) & (pred == 0)].sum())
    n = float(w.sum())
    accuracy = (tp + tn) / n
    sensitivity = tp / max(tp + fn, 1e-12)
    specificity = tn / max(tn + fp, 1e-12)
    precision = tp / max(tp + fp, 1e-12)
    f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)
    balanced = 0.5 * (sensitivity + specificity)
    roc = float(roc_auc_score(y, p, sample_weight=w)) if len(np.unique(y)) == 2 else float("nan")
    pr = float(average_precision_score(y, p, sample_weight=w)) if len(np.unique(y)) == 2 else float("nan")
    brier = float(np.average((p - y) ** 2, weights=w))
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    ll = float(np.average(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)), weights=w))
    ece = weighted_ece(y, p, w, bins=15)
    return {
        "accuracy": float(accuracy), "balanced_accuracy": float(balanced),
        "sensitivity": float(sensitivity), "specificity": float(specificity),
        "f1": float(f1), "roc_auc": roc, "pr_auc": pr,
        "brier": brier, "log_loss": ll, "ece": float(ece),
    }


def weighted_ece(y: np.ndarray, p: np.ndarray, w: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(w.sum())
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if not mask.any():
            continue
        wb = w[mask]
        mass = float(wb.sum())
        if mass <= 0:
            continue
        obs = float(np.average(y[mask], weights=wb))
        conf = float(np.average(p[mask], weights=wb))
        out += (mass / total) * abs(obs - conf)
    return float(out)


def group_cluster_bootstrap(oof: pd.DataFrame, reps: int = BOOTSTRAP_REPS, seed: int = RNG_SEED) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metrics_to_report = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc", "brier", "log_loss", "ece"]
    groups = sorted(oof["group_id"].astype(str).unique().tolist())
    group_to_int = {gid: i for i, gid in enumerate(groups)}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    observed: dict[str, dict[str, float]] = {}
    for m in EXPECTED_MODELS:
        g = oof.loc[oof["model_key"] == m].reset_index(drop=True)
        arrays[m] = {
            "y": g["label"].to_numpy(int),
            "p": g["prob_calibrated"].to_numpy(float),
            "pred": g["pred"].to_numpy(int),
            "group_index": g["group_id"].astype(str).map(group_to_int).to_numpy(int),
        }
        observed[m] = metric_dict(arrays[m]["y"], arrays[m]["p"], arrays[m]["pred"])

    rng = np.random.default_rng(seed)
    boot = {m: {metric: np.empty(int(reps), dtype=float) for metric in metrics_to_report} for m in EXPECTED_MODELS}
    pairs = [(EXPECTED_MODELS[i], EXPECTED_MODELS[j]) for i in range(len(EXPECTED_MODELS)) for j in range(i + 1, len(EXPECTED_MODELS))]
    delta_store = {pair: {metric: np.empty(int(reps), dtype=float) for metric in metrics_to_report} for pair in pairs}
    valid = 0
    for rep in range(int(reps)):
        sampled_int = rng.integers(0, len(groups), size=len(groups))
        cluster_counts = np.bincount(sampled_int, minlength=len(groups)).astype(float)
        rep_metrics: dict[str, dict[str, float]] = {}
        invalid = False
        for m in EXPECTED_MODELS:
            a = arrays[m]
            row_weights = cluster_counts[a["group_index"]]
            active_labels = np.unique(a["y"][row_weights > 0])
            if len(active_labels) != 2:
                invalid = True
                break
            rep_metrics[m] = weighted_metric_dict(a["y"], a["p"], a["pred"], row_weights)
        if invalid:
            continue
        for m in EXPECTED_MODELS:
            for metric in metrics_to_report:
                boot[m][metric][valid] = rep_metrics[m][metric]
        for a, b in pairs:
            for metric in metrics_to_report:
                delta_store[(a, b)][metric][valid] = rep_metrics[a][metric] - rep_metrics[b][metric]
        valid += 1

    if valid < max(100, int(reps * 0.9)):
        raise RuntimeError(f"Too few valid group bootstrap replicates: {valid}/{reps}")

    ci_rows = []
    for m in EXPECTED_MODELS:
        for metric in metrics_to_report:
            arr = boot[m][metric][:valid]
            lo, hi = np.quantile(arr, [0.025, 0.975])
            ci_rows.append({
                "model_key": m, "model_name": model_name(m), "metric": metric,
                "observed": float(observed[m][metric]), "ci95_lower": float(lo), "ci95_upper": float(hi),
                "bootstrap_reps": int(valid), "cluster_unit": "group_id",
                "repeated_seed_rows_preserved_within_cluster": True,
            })
    ci_df = pd.DataFrame(ci_rows)

    pair_rows = []
    for (a, b), metrics in delta_store.items():
        for metric, values in metrics.items():
            arr = values[:valid]
            lo, hi = np.quantile(arr, [0.025, 0.975])
            obs_delta = float(observed[a][metric] - observed[b][metric])
            p_two = float(min(1.0, 2.0 * min(np.mean(arr <= 0), np.mean(arr >= 0))))
            pair_rows.append({
                "model_a": model_name(a), "model_a_key": a, "model_b": model_name(b), "model_b_key": b,
                "metric": metric, "observed_delta_a_minus_b": obs_delta,
                "ci95_lower": float(lo), "ci95_upper": float(hi), "bootstrap_p_two_sided": p_two,
                "significant_ci_excludes_zero": bool(not (lo <= 0 <= hi)),
                "bootstrap_reps": int(valid), "cluster_unit": "group_id",
            })
    pair_df = pd.DataFrame(pair_rows)
    pair_df["holm_p"] = np.nan
    for metric, idx in pair_df.groupby("metric").groups.items():
        ids = list(idx)
        pvals = pair_df.loc[ids, "bootstrap_p_two_sided"].to_numpy(float)
        order = np.argsort(pvals)
        adjusted = np.empty_like(pvals)
        running = 0.0
        mtests = len(pvals)
        for rank, pos in enumerate(order):
            val = min(1.0, (mtests - rank) * pvals[pos])
            running = max(running, val)
            adjusted[pos] = running
        pair_df.loc[ids, "holm_p"] = adjusted
    pair_df["significant_holm_0_05"] = pair_df["holm_p"] < 0.05
    higher_better = {"accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc"}
    def favored(row):
        if not row["significant_holm_0_05"]:
            return "No clear difference"
        delta = float(row["observed_delta_a_minus_b"])
        if row["metric"] in higher_better:
            return row["model_a"] if delta > 0 else row["model_b"]
        return row["model_a"] if delta < 0 else row["model_b"]
    pair_df["favored_model"] = pair_df.apply(favored, axis=1)
    meta = {
        "valid_bootstrap_reps": int(valid), "requested_reps": int(reps), "n_groups": len(groups), "seed": seed,
        "implementation": "cluster_count_weighted_bootstrap",
    }
    return ci_df, pair_df, meta


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
        model.fit(logit, y)
        return float(model.intercept_[0]), float(model.coef_[0, 0])
    except Exception:
        model = LogisticRegression(C=1e8, solver="lbfgs", max_iter=2000)
        model.fit(logit, y)
        return float(model.intercept_[0]), float(model.coef_[0, 0])


def calibration_audit(metrics: pd.DataFrame, oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    reliability_rows = []
    for m in EXPECTED_MODELS:
        gm = metrics.loc[metrics["model_key"] == m]
        go = oof.loc[oof["model_key"] == m]
        raw_intercept, raw_slope = calibration_intercept_slope(go["label"].to_numpy(), go["prob_raw"].to_numpy(float))
        cal_intercept, cal_slope = calibration_intercept_slope(go["label"].to_numpy(), go["prob_calibrated"].to_numpy(float))
        summary_rows.append({
            "model_key": m,
            "model_name": model_name(m),
            "raw_calibration_intercept": raw_intercept,
            "raw_calibration_slope": raw_slope,
            "calibrated_calibration_intercept": cal_intercept,
            "calibrated_calibration_slope": cal_slope,
            "raw_brier_mean": float(gm["raw_brier"].mean()),
            "calibrated_brier_mean": float(gm["brier"].mean()),
            "delta_brier_cal_minus_raw": float(gm["brier"].mean() - gm["raw_brier"].mean()),
            "raw_ece_mean": float(gm["raw_ece"].mean()),
            "calibrated_ece_mean": float(gm["ece"].mean()),
            "delta_ece_cal_minus_raw": float(gm["ece"].mean() - gm["raw_ece"].mean()),
            "raw_log_loss_mean": float(gm["raw_log_loss"].mean()),
            "calibrated_log_loss_mean": float(gm["log_loss"].mean()),
            "delta_log_loss_cal_minus_raw": float(gm["log_loss"].mean() - gm["raw_log_loss"].mean()),
            "temperature_mean": float(gm["temperature"].mean()),
            "temperature_sd": float(gm["temperature"].std(ddof=1)),
            "temperature_min": float(gm["temperature"].min()),
            "temperature_max": float(gm["temperature"].max()),
            "threshold_mean": float(gm["threshold"].mean()),
            "threshold_sd": float(gm["threshold"].std(ddof=1)),
            "threshold_min": float(gm["threshold"].min()),
            "threshold_max": float(gm["threshold"].max()),
        })
        for probability_type, col in (("raw", "prob_raw"), ("calibrated", "prob_calibrated")):
            curve = calibration_curve_equal_frequency(go["label"].to_numpy(), go[col].to_numpy(float), bins=10)
            curve.insert(0, "probability_type", probability_type)
            curve.insert(0, "model_name", model_name(m))
            curve.insert(0, "model_key", m)
            reliability_rows.append(curve)
    return pd.DataFrame(summary_rows), pd.concat(reliability_rows, ignore_index=True)


def threshold_operating_audit(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = metrics[[
        "model_key", "seed", "outer_fold", "temperature", "threshold", "threshold_rule",
        "sensitivity", "specificity", "balanced_accuracy", "accuracy", "f1",
        "raw_brier", "brier", "raw_ece", "ece", "raw_log_loss", "log_loss",
    ]].copy()
    x.insert(1, "model_name", x["model_key"].map(model_name))
    x["target_sensitivity"] = TARGET_SENSITIVITY
    x["target_met_on_outer_test"] = x["sensitivity"] >= TARGET_SENSITIVITY
    x["extreme_threshold_ge_0_99"] = x["threshold"] >= 0.99
    x["extreme_threshold_le_0_01"] = x["threshold"] <= 0.01
    x["extreme_temperature_le_0_10"] = x["temperature"] <= 0.10
    x["brier_improved"] = x["brier"] < x["raw_brier"]
    x["ece_improved"] = x["ece"] < x["raw_ece"]
    x["log_loss_improved"] = x["log_loss"] < x["raw_log_loss"]
    summary = x.groupby(["model_key", "model_name"], as_index=False).agg(
        trials=("outer_fold", "size"),
        target_met_trials=("target_met_on_outer_test", "sum"),
        target_met_fraction=("target_met_on_outer_test", "mean"),
        sensitivity_mean=("sensitivity", "mean"),
        specificity_mean=("specificity", "mean"),
        threshold_mean=("threshold", "mean"),
        threshold_sd=("threshold", "std"),
        temperature_mean=("temperature", "mean"),
        extreme_threshold_trials=("extreme_threshold_ge_0_99", "sum"),
        extreme_temperature_trials=("extreme_temperature_le_0_10", "sum"),
        brier_improved_trials=("brier_improved", "sum"),
        ece_improved_trials=("ece_improved", "sum"),
        log_loss_improved_trials=("log_loss_improved", "sum"),
    )
    return x, summary


def selective_referral_analysis(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_rows = []
    capture_rows = []
    for (m, seed, fold), g0 in oof.groupby(["model_key", "seed", "outer_fold"], sort=False):
        g = g0.copy().reset_index(drop=True)
        p = g["prob_calibrated"].to_numpy(float)
        confidence = np.maximum(p, 1 - p)
        uncertainty = 1.0 - confidence
        order = np.argsort(-uncertainty, kind="stable")
        total_errors = int((g["pred"].to_numpy(int) != g["label"].to_numpy(int)).sum())
        total_fn = int(((g["label"] == 1) & (g["pred"] == 0)).sum())
        for referral_rate in REFERRAL_RATES:
            n_ref = int(round(referral_rate * len(g)))
            referred_idx = set(order[:n_ref].tolist())
            keep = np.array([i not in referred_idx for i in range(len(g))], dtype=bool)
            kept = g.loc[keep]
            if kept.empty or kept["label"].nunique() < 2:
                continue
            md = metric_dict(kept["label"].to_numpy(), kept["prob_calibrated"].to_numpy(float), kept["pred"].to_numpy(int))
            errors_referred = int(sum((g.loc[i, "pred"] != g.loc[i, "label"]) for i in referred_idx))
            fn_referred = int(sum((g.loc[i, "label"] == 1 and g.loc[i, "pred"] == 0) for i in referred_idx))
            trial_rows.append({
                "model_key": m, "model_name": model_name(m), "seed": int(seed), "outer_fold": int(fold),
                "coverage": float(1 - referral_rate), "referral_rate": float(referral_rate),
                "referred_n": int(n_ref), **md,
            })
            capture_rows.append({
                "model_key": m, "model_name": model_name(m), "seed": int(seed), "outer_fold": int(fold),
                "referral_rate": float(referral_rate), "coverage": float(1 - referral_rate),
                "total_errors": total_errors, "errors_referred": errors_referred,
                "error_capture_fraction": float(errors_referred / total_errors) if total_errors else 0.0,
                "total_false_negatives": total_fn, "false_negatives_referred": fn_referred,
                "fn_capture_fraction": float(fn_referred / total_fn) if total_fn else 0.0,
                "remaining_automated_error_rate": float(1 - md["accuracy"]),
            })
    trial_df = pd.DataFrame(trial_rows)
    cap_df = pd.DataFrame(capture_rows)
    return trial_df, cap_df


def similarity_leakage_audit(unique_images: pd.DataFrame, reps: int = LEAKAGE_SIM_REPS, seed: int = RNG_SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import StratifiedKFold
    sizes = unique_images.groupby("group_id", as_index=False).agg(
        n_images=("image_id", "size"),
        n_labels=("label", "nunique"),
        outer_folds=("outer_fold", "nunique"),
    )
    summary = pd.DataFrame([{
        "unique_images": int(len(unique_images)),
        "similarity_groups": int(len(sizes)),
        "singleton_groups": int((sizes["n_images"] == 1).sum()),
        "multi_image_groups": int((sizes["n_images"] > 1).sum()),
        "images_in_multi_image_groups": int(sizes.loc[sizes["n_images"] > 1, "n_images"].sum()),
        "fraction_images_in_multi_image_groups": float(sizes.loc[sizes["n_images"] > 1, "n_images"].sum() / len(unique_images)),
        "largest_group_size": int(sizes["n_images"].max()),
        "cross_label_group_conflicts": int((sizes["n_labels"] > 1).sum()),
        "actual_cross_fold_group_leakage": int((sizes["outer_folds"] > 1).sum()),
    }])

    frame = unique_images.reset_index(drop=True).copy()
    sim_rows = []
    for i in range(int(reps)):
        rs = int(seed + i)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
        simulated = np.empty(len(frame), dtype=int)
        for fold, (_, test_idx) in enumerate(skf.split(frame, frame["label"].to_numpy())):
            simulated[test_idx] = fold
        tmp = frame.assign(simulated_fold=simulated)
        span = tmp.groupby("group_id")["simulated_fold"].nunique()
        bad_groups = set(span[span > 1].index.astype(str))
        affected = int(tmp["group_id"].astype(str).isin(bad_groups).sum())
        sim_rows.append({
            "replicate": i + 1,
            "random_state": rs,
            "split_similarity_groups": int(len(bad_groups)),
            "affected_images": affected,
            "affected_image_fraction": float(affected / len(tmp)),
            "group_leakage_present": bool(len(bad_groups) > 0),
        })
    return summary, sizes, pd.DataFrame(sim_rows)


def hard_case_analysis(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = oof.copy()
    x["is_error"] = x["pred"] != x["label"]
    x["confidence"] = np.maximum(x["prob_calibrated"], 1 - x["prob_calibrated"])
    image = x.groupby(["image_id", "group_id", "label", "outer_fold"], as_index=False).agg(
        wrong_count_9=("is_error", "sum"),
        mean_probability=("prob_calibrated", "mean"),
        probability_sd=("prob_calibrated", "std"),
        mean_confidence=("confidence", "mean"),
    )
    image["wrong_fraction_9"] = image["wrong_count_9"] / 9.0
    image["difficulty_category"] = pd.cut(
        image["wrong_count_9"],
        bins=[-1, 0, 2, 5, 8, 9],
        labels=["easy_stable", "occasional_instability", "difficult", "persistent_difficult", "universal_failure"],
    ).astype(str)
    for m in EXPECTED_MODELS:
        gm = x.loc[x["model_key"] == m].groupby("image_id", as_index=False).agg(
            **{f"{m}_wrong_count_3": ("is_error", "sum")},
            **{f"{m}_probability_sd_3": ("prob_calibrated", "std")},
            **{f"{m}_mean_probability_3": ("prob_calibrated", "mean")},
        )
        image = image.merge(gm, on="image_id", how="left")

    majority_rows = []
    for m in EXPECTED_MODELS:
        gm = x.loc[x["model_key"] == m].groupby("image_id").agg(label=("label", "first"), votes=("pred", "sum"), mean_p=("prob_calibrated", "mean"))
        gm[f"{m}_majority_pred"] = (gm["votes"] >= 2).astype(int)
        majority_rows.append(gm[[f"{m}_majority_pred"]])
    majority = pd.concat(majority_rows, axis=1).reset_index()
    agree = majority.copy()
    pred_cols = [f"{m}_majority_pred" for m in EXPECTED_MODELS]
    agree["unanimous_architecture_agreement"] = agree[pred_cols].nunique(axis=1) == 1
    pair_rows = []
    for i, a in enumerate(EXPECTED_MODELS):
        for b in EXPECTED_MODELS[i + 1:]:
            pair_rows.append({
                "model_a": model_name(a), "model_b": model_name(b),
                "majority_prediction_agreement": float((agree[f"{a}_majority_pred"] == agree[f"{b}_majority_pred"]).mean()),
            })
    return image, agree, pd.DataFrame(pair_rows)


def stability_analysis(oof: pd.DataFrame, metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (m, image_id), g in oof.groupby(["model_key", "image_id"], sort=False):
        probs = g["prob_calibrated"].to_numpy(float)
        preds = g["pred"].to_numpy(int)
        rows.append({
            "model_key": m, "model_name": model_name(m), "image_id": image_id,
            "probability_mean": float(probs.mean()), "probability_sd_across_seeds": float(probs.std(ddof=1)),
            "prediction_agreement_3of3": bool(len(set(preds.tolist())) == 1),
            "positive_votes": int(preds.sum()),
        })
    per_image = pd.DataFrame(rows)
    summary = per_image.groupby(["model_key", "model_name"], as_index=False).agg(
        images=("image_id", "nunique"),
        mean_probability_sd_across_seeds=("probability_sd_across_seeds", "mean"),
        median_probability_sd_across_seeds=("probability_sd_across_seeds", "median"),
        prediction_agreement_fraction=("prediction_agreement_3of3", "mean"),
    )
    fold_var = metrics.groupby("model_key")["accuracy"].std(ddof=1).rename("accuracy_sd_15_trials").reset_index()
    fold_var["model_name"] = fold_var["model_key"].map(model_name)
    summary = summary.merge(fold_var[["model_key", "accuracy_sd_15_trials"]], on="model_key", how="left")
    return per_image, summary


def decision_curve(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    thresholds = np.linspace(0.05, 0.95, 37)
    # seed-average per image to avoid counting the same image three times in architecture-level DCA
    for m in EXPECTED_MODELS:
        gm = oof.loc[oof["model_key"] == m].groupby("image_id", as_index=False).agg(label=("label", "first"), probability=("prob_calibrated", "mean"))
        y = gm["label"].to_numpy(int)
        p = gm["probability"].to_numpy(float)
        n = len(gm)
        prevalence = float(y.mean())
        for pt in thresholds:
            pred = p >= pt
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            nb = tp / n - fp / n * (pt / (1 - pt))
            treat_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
            rows.extend([
                {"model_key": m, "model_name": model_name(m), "threshold_probability": float(pt), "strategy": "model", "net_benefit": float(nb)},
                {"model_key": m, "model_name": model_name(m), "threshold_probability": float(pt), "strategy": "treat_all", "net_benefit": float(treat_all)},
                {"model_key": m, "model_name": model_name(m), "threshold_probability": float(pt), "strategy": "treat_none", "net_benefit": 0.0},
            ])
    return pd.DataFrame(rows)


def checkpoint_inventory(run_root: Path | None) -> pd.DataFrame:
    rows = []
    if run_root is None:
        return pd.DataFrame(columns=["model_key", "model_name", "seed", "outer_fold", "path", "exists", "bytes", "sha256", "metadata_valid", "error"])
    import torch
    for m in EXPECTED_MODELS:
        for seed in EXPECTED_SEEDS:
            for fold in EXPECTED_FOLDS:
                path = Path(run_root) / "trials" / m / f"seed_{seed}" / f"fold_{fold}" / "best_model_portable_fp16.pt"
                record = {
                    "model_key": m, "model_name": model_name(m), "seed": seed, "outer_fold": fold,
                    "path": str(path), "exists": path.is_file(), "bytes": None, "sha256": None,
                    "metadata_valid": False, "error": "",
                }
                if path.is_file():
                    try:
                        record["bytes"] = int(path.stat().st_size)
                        record["sha256"] = sha256_file(path)
                        payload = torch.load(path, map_location="cpu", weights_only=False)
                        ok = (
                            str(payload.get("model_key")) == m
                            and int(payload.get("seed")) == seed
                            and int(payload.get("fold")) == fold - 1
                            and isinstance(payload.get("state_dict"), dict)
                        )
                        record["metadata_valid"] = bool(ok)
                        if not ok:
                            record["error"] = "checkpoint metadata mismatch"
                        del payload
                    except Exception as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(record)
    return pd.DataFrame(rows)


def choose_xai_cases(hard_cases: pd.DataFrame, oof: pd.DataFrame, fold: int = 1) -> pd.DataFrame:
    x = hard_cases.loc[hard_cases["outer_fold"] == fold].copy()
    if x.empty:
        return pd.DataFrame()
    x["uncertainty"] = np.abs(x["mean_probability"] - 0.5)
    candidates: list[dict[str, Any]] = []
    used: set[str] = set()
    def take(frame: pd.DataFrame, name: str, sort_cols: list[str], ascending: list[bool]):
        frame = frame.loc[~frame["image_id"].astype(str).isin(used)].copy()
        if frame.empty:
            return
        r = frame.sort_values(sort_cols, ascending=ascending, kind="stable").iloc[0]
        used.add(str(r["image_id"]))
        candidates.append({"case_type": name, **r.to_dict()})
    take(x[(x["label"] == 1) & (x["wrong_count_9"] == 0)], "confident_true_positive", ["mean_confidence"], [False])
    take(x[(x["label"] == 0) & (x["wrong_count_9"] == 0)], "confident_true_negative", ["mean_confidence"], [False])
    take(x[(x["label"] == 1) & (x["wrong_count_9"] > 0)], "persistent_false_negative", ["wrong_count_9", "mean_confidence"], [False, False])
    take(x[(x["label"] == 0) & (x["wrong_count_9"] > 0)], "persistent_false_positive", ["wrong_count_9", "mean_confidence"], [False, False])
    take(x[x["label"] == 1], "uncertain_dfu", ["uncertainty", "wrong_count_9"], [True, False])
    take(x[x["label"] == 0], "uncertain_normal", ["uncertainty", "wrong_count_9"], [True, False])
    out = pd.DataFrame(candidates)
    if not out.empty and "relative_path" in oof.columns:
        meta = oof[["image_id", "relative_path"]].drop_duplicates("image_id")
        out = out.merge(meta, on="image_id", how="left")
    return out


def resolve_dataset_root_if_needed(xai_cases: pd.DataFrame) -> Path | None:
    if xai_cases.empty:
        return None
    try:
        import kagglehub
        root = Path(kagglehub.dataset_download("laithjj/diabetic-foot-ulcer-dfu"))
        return root
    except Exception as exc:
        print("XAI dataset resolution unavailable:", type(exc).__name__, exc)
        return None


def build_model_for_xai(model_key: str):
    import timm
    errors = []
    for candidate in MODEL_CANDIDATES[model_key]:
        try:
            return timm.create_model(candidate, pretrained=False, num_classes=1, drop_rate=0.20)
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not instantiate XAI model: " + " | ".join(errors))


def xai_target_layer(model_key: str, model):
    if model_key == "convnextv2_tiny":
        return model.stages[-1]
    if model_key == "densenet121":
        return model.features
    if model_key == "mobilenetv3_large":
        return model.blocks[-1]
    raise ValueError(model_key)


def prepare_image_tensor(path: Path):
    from PIL import Image
    import torch
    from torchvision import transforms
    image = Image.open(path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return image, tf(image).unsqueeze(0)


def gradcam_map(model, model_key: str, tensor, device):
    import torch
    activations = []
    layer = xai_target_layer(model_key, model)
    handle = layer.register_forward_hook(lambda _m, _i, out: activations.append(out))
    try:
        output = model(tensor.to(device)).reshape(-1)[0]
        if not activations:
            raise RuntimeError("target layer produced no activation")
        act = activations[-1]
        grad = torch.autograd.grad(output, act, retain_graph=False, create_graph=False)[0]
        if act.ndim != 4 or grad.ndim != 4:
            raise RuntimeError(f"Grad-CAM requires 4D activation; got {tuple(act.shape)}")
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().float().cpu().numpy()
        cam -= cam.min()
        cam /= max(cam.max(), 1e-8)
        prob = float(torch.sigmoid(output.detach()).cpu())
        return cam, prob
    finally:
        handle.remove()


def lime_map(model, tensor_device, image, device):
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    import torch
    from torchvision import transforms
    arr = np.asarray(image.resize((224, 224)), dtype=np.uint8)
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    def classifier(images):
        batch = torch.stack([normalize(np.asarray(im, dtype=np.uint8)) for im in images]).to(device)
        with torch.inference_mode():
            p = torch.sigmoid(model(batch).reshape(-1)).float().cpu().numpy()
        return np.column_stack([1 - p, p])
    explainer = lime_image.LimeImageExplainer(random_state=RNG_SEED)
    explanation = explainer.explain_instance(arr, classifier, labels=(1,), num_samples=250, hide_color=0)
    _, mask = explanation.get_image_and_mask(1, positive_only=False, num_features=8, hide_rest=False)
    boundary = mark_boundaries(arr / 255.0, mask)
    return boundary


def shap_map(model, tensor, device):
    import shap
    import torch
    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, x):
            return self.inner(x).reshape(-1, 1)
    wrapped = Wrapper(model)
    background = torch.zeros((2, *tensor.shape[1:]), dtype=tensor.dtype, device=device)
    explainer = shap.GradientExplainer(wrapped, background)
    values = explainer.shap_values(tensor.to(device), nsamples=32)
    arr = values[0] if isinstance(values, list) else values
    arr = np.asarray(arr)
    if arr.ndim == 5:
        arr = arr[..., 0]
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[0] in (1, 3):
        heat = np.mean(np.abs(arr), axis=0)
    elif arr.shape[-1] in (1, 3):
        heat = np.mean(np.abs(arr), axis=-1)
    else:
        raise RuntimeError(f"Unexpected SHAP shape {arr.shape}")
    heat -= heat.min()
    heat /= max(heat.max(), 1e-8)
    return heat


def run_conditional_xai(checkpoints: pd.DataFrame, cases: pd.DataFrame, oof: pd.DataFrame, out_root: Path, png_dir: Path, pdf_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    usable = checkpoints[(checkpoints["exists"] == True) & (checkpoints["metadata_valid"] == True) & (checkpoints["outer_fold"] == 1)].copy()
    if usable.empty or cases.empty:
        return pd.DataFrame(), {"status": "SKIPPED", "reason": "no usable Fold-1 checkpoints or no selected cases", "training_performed": False}
    ensure_packages({"timm": "timm", "lime": "lime", "shap": "shap", "skimage": "scikit-image"})
    dataset_root = resolve_dataset_root_if_needed(cases)
    if dataset_root is None:
        return pd.DataFrame(), {"status": "SKIPPED", "reason": "dataset unavailable for XAI", "training_performed": False}

    import torch
    import matplotlib.pyplot as plt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xai_rows = []
    selected_models = []
    for m in EXPECTED_MODELS:
        candidates = usable[usable["model_key"] == m].sort_values(["seed", "outer_fold"])
        if candidates.empty:
            continue
        selected_models.append(m)
        ck = candidates.iloc[0]
        model = build_model_for_xai(m)
        payload = torch.load(Path(ck["path"]), map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"], strict=True)
        model = model.to(device).eval()
        for _, case in cases.iterrows():
            rel = str(case.get("relative_path", ""))
            image_path = dataset_root / rel
            if not image_path.is_file():
                matches = list(dataset_root.rglob(Path(rel).name))
                image_path = matches[0] if matches else image_path
            record = {
                "model_key": m, "model_name": model_name(m), "checkpoint_seed": int(ck["seed"]), "checkpoint_fold": int(ck["outer_fold"]),
                "case_type": str(case["case_type"]), "image_id": str(case["image_id"]), "relative_path": rel,
                "image_found": image_path.is_file(), "gradcam_status": "NOT_RUN", "lime_status": "NOT_RUN", "shap_status": "NOT_RUN",
                "probability": None, "error": "",
            }
            if not image_path.is_file():
                record["error"] = "image not found"
                xai_rows.append(record)
                continue
            try:
                image, tensor = prepare_image_tensor(image_path)
                cam, prob = gradcam_map(model, m, tensor, device)
                record["probability"] = prob
                record["gradcam_status"] = "PASS"
            except Exception as exc:
                image = None
                cam = None
                record["gradcam_status"] = "FAIL"
                record["error"] += f"GradCAM {type(exc).__name__}: {exc}; "
            lime_img = None
            shap_heat = None
            if image is not None:
                try:
                    lime_img = lime_map(model, tensor, image, device)
                    record["lime_status"] = "PASS"
                except Exception as exc:
                    record["lime_status"] = "FAIL"
                    record["error"] += f"LIME {type(exc).__name__}: {exc}; "
                try:
                    shap_heat = shap_map(model, tensor, device)
                    record["shap_status"] = "PASS"
                except Exception as exc:
                    record["shap_status"] = "FAIL"
                    record["error"] += f"SHAP {type(exc).__name__}: {exc}; "
            xai_rows.append(record)

            if image is not None:
                fig, axes = plt.subplots(1, 4, figsize=(15, 4))
                axes[0].imshow(image.resize((224, 224)))
                axes[0].set_title(f"Original\n{case['case_type']}")
                axes[0].axis("off")
                if cam is not None:
                    axes[1].imshow(image.resize((224, 224)))
                    axes[1].imshow(cam, alpha=0.45, cmap="jet")
                    axes[1].set_title(f"Grad-CAM\np(DFU)={prob:.3f}")
                else:
                    axes[1].text(0.5, 0.5, "Grad-CAM unavailable", ha="center", va="center")
                axes[1].axis("off")
                if lime_img is not None:
                    axes[2].imshow(lime_img)
                    axes[2].set_title("LIME")
                else:
                    axes[2].text(0.5, 0.5, "LIME unavailable", ha="center", va="center")
                axes[2].axis("off")
                if shap_heat is not None:
                    axes[3].imshow(image.resize((224, 224)))
                    axes[3].imshow(shap_heat, alpha=0.45, cmap="magma")
                    axes[3].set_title("SHAP")
                else:
                    axes[3].text(0.5, 0.5, "SHAP unavailable", ha="center", va="center")
                axes[3].axis("off")
                fig.suptitle(f"{model_name(m)} — {case['case_type']}", fontsize=13, fontweight="bold")
                save_figure(fig, png_dir, pdf_dir, f"XAI_{m}_{case['case_type']}")
        del model, payload
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    xai_df = pd.DataFrame(xai_rows)
    return xai_df, {
        "status": "PASS_WITH_CONDITIONAL_METHOD_STATUS" if not xai_df.empty else "SKIPPED",
        "representative_models": [model_name(m) for m in selected_models],
        "cases_requested": list(XAI_CASE_TYPES),
        "training_performed": False,
        "parameter_updates": False,
        "gradients_used_only_for_attribution": True,
    }


def make_figures(
    primary: pd.DataFrame,
    ci: pd.DataFrame,
    pairwise: pd.DataFrame,
    oof: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    reliability: pd.DataFrame,
    threshold_trials: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    selective: pd.DataFrame,
    capture: pd.DataFrame,
    leakage_summary: pd.DataFrame,
    leakage_groups: pd.DataFrame,
    leakage_sim: pd.DataFrame,
    hard_cases: pd.DataFrame,
    agreement_pairs: pd.DataFrame,
    stability_summary: pd.DataFrame,
    dca: pd.DataFrame,
    png_dir: Path,
    pdf_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    # Fig01 architecture
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
    boxes = [
        (0.4, 5.1, 2.1, 1.0, "Frozen Primary Run\n45 trials / 9,495 OOF"),
        (3.1, 5.1, 2.1, 1.0, "Integrity Gate\nfolds • groups • hashes"),
        (5.8, 5.1, 2.1, 1.0, "Primary Results\nmetrics • CI • statistics"),
        (8.5, 5.1, 2.1, 1.0, "Calibration +\nClinical Threshold"),
        (11.2, 5.1, 1.4, 1.0, "Referral\nAnalysis"),
        (1.2, 2.6, 2.2, 1.0, "Similarity / Leakage\n1,000 naive split simulations"),
        (4.2, 2.6, 2.2, 1.0, "Group-Clustered\nBootstrap + Holm"),
        (7.2, 2.6, 2.2, 1.0, "Stability +\nHard Cases"),
        (10.2, 2.6, 2.2, 1.0, "Checkpoint Audit\nConditional XAI"),
        (5.0, 0.45, 3.0, 1.0, "PHASE2_FULL_FINAL_EXPORT\nTables • PNG/PDF • PKL • Manifest"),
    ]
    for x, y, w, h, txt in boxes:
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", linewidth=1.4, facecolor="white", edgecolor="black")
        ax.add_patch(p); ax.text(x+w/2, y+h/2, txt, ha="center", va="center", fontsize=10)
    arrows = [((2.5,5.6),(3.1,5.6)),((5.2,5.6),(5.8,5.6)),((7.9,5.6),(8.5,5.6)),((10.6,5.6),(11.2,5.6)),
              ((4.2,5.1),(2.3,3.6)),((6.8,5.1),(5.3,3.6)),((9.4,5.1),(8.3,3.6)),((11.9,5.1),(11.3,3.6)),
              ((2.3,2.6),(6.0,1.45)),((5.3,2.6),(6.2,1.45)),((8.3,2.6),(6.5,1.45)),((11.3,2.6),(7.0,1.45))]
    for a,b in arrows:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="->",mutation_scale=12,linewidth=1.2))
    ax.set_title("DFU-ImageGuard Phase-2 Trustworthiness Analysis Architecture", fontsize=15, fontweight="bold")
    save_figure(fig, png_dir, pdf_dir, "Fig01_Evaluation_Architecture")

    # Fig02 performance overview
    metrics = ["accuracy_mean", "balanced_accuracy_mean", "sensitivity_mean", "specificity_mean", "f1_mean"]
    labels = ["Accuracy", "Balanced Acc.", "Sensitivity", "Specificity", "F1"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels)); width = 0.24
    for i, (_, r) in enumerate(primary.iterrows()):
        vals = [r[m] for m in metrics]
        ax.bar(x + (i-1)*width, vals, width=width, label=r["model_name"])
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0.90, 1.01); ax.set_ylabel("Mean over 15 fold-seed trials")
    ax.set_title("Primary Classification Performance", fontweight="bold"); ax.grid(axis="y", alpha=0.25); ax.legend(ncol=3, loc="lower center")
    save_figure(fig, png_dir, pdf_dir, "Fig02_Performance_Overview")

    # Fig03 ROC + inset
    fig, ax = plt.subplots(figsize=(8, 6.5)); axins = inset_axes(ax, width="45%", height="45%", loc="lower right", borderpad=2)
    for m in EXPECTED_MODELS:
        g = oof[oof["model_key"] == m]
        fpr, tpr, _ = roc_curve(g["label"], g["prob_calibrated"]); roc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{model_name(m)} (AUC={roc:.4f})"); axins.plot(fpr, tpr, lw=1.5)
    ax.plot([0,1],[0,1],ls="--",lw=1); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("OOF ROC Curves", fontweight="bold"); ax.grid(alpha=.25); ax.legend(loc="lower right")
    axins.set_xlim(0, .10); axins.set_ylim(.90, 1.002); axins.grid(alpha=.2); axins.set_title("Zoom", fontsize=9)
    save_figure(fig, png_dir, pdf_dir, "Fig03_ROC_with_Zoom")

    # Fig04 PR + inset
    fig, ax = plt.subplots(figsize=(8, 6.5)); axins = inset_axes(ax, width="45%", height="45%", loc="lower left", borderpad=2)
    for m in EXPECTED_MODELS:
        g = oof[oof["model_key"] == m]
        pr, rc, _ = precision_recall_curve(g["label"], g["prob_calibrated"]); val = auc(rc, pr)
        ax.plot(rc, pr, lw=2, label=f"{model_name(m)} (PR-AUC={val:.4f})"); axins.plot(rc, pr, lw=1.5)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("OOF Precision-Recall Curves", fontweight="bold"); ax.grid(alpha=.25); ax.legend(loc="lower left")
    axins.set_xlim(.85, 1.0); axins.set_ylim(.90, 1.002); axins.grid(alpha=.2); axins.set_title("Zoom", fontsize=9)
    save_figure(fig, png_dir, pdf_dir, "Fig04_PR_with_Zoom")

    # Fig05 forest selected metrics
    selected = pairwise[pairwise["metric"].isin(["balanced_accuracy","sensitivity","roc_auc","brier","ece"])].copy()
    fig, ax = plt.subplots(figsize=(10, max(6, 0.35*len(selected))))
    y = np.arange(len(selected)); delta = selected["observed_delta_a_minus_b"].to_numpy(float); lo = selected["ci95_lower"].to_numpy(float); hi = selected["ci95_upper"].to_numpy(float)
    ax.errorbar(delta, y, xerr=[delta-lo, hi-delta], fmt="o", capsize=3)
    labels2 = [f"{r.model_a} − {r.model_b} | {r.metric}" for r in selected.itertuples()]
    ax.set_yticks(y); ax.set_yticklabels(labels2); ax.axvline(0, ls="--", lw=1); ax.invert_yaxis(); ax.set_xlabel("Metric difference (A − B), group-clustered 95% CI"); ax.set_title("Pairwise Model Comparison Forest Plot", fontweight="bold"); ax.grid(axis="x", alpha=.25)
    save_figure(fig, png_dir, pdf_dir, "Fig05_Model_Comparison_Forest")

    # Fig06 raw vs calibrated reliability, 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, m in zip(axes, EXPECTED_MODELS):
        for ptype in ("raw","calibrated"):
            c = reliability[(reliability["model_key"] == m) & (reliability["probability_type"] == ptype)]
            ax.plot(c["mean_probability"], c["observed_rate"], marker="o", lw=1.7, label=ptype.title())
        ax.plot([0,1],[0,1],ls="--",lw=1,label="Perfect"); ax.set_title(model_name(m)); ax.set_xlabel("Mean predicted probability"); ax.grid(alpha=.25)
    axes[0].set_ylabel("Observed DFU rate"); axes[0].legend(); fig.suptitle("Raw vs Calibrated Reliability (Equal-Frequency Bins)", fontsize=14, fontweight="bold")
    save_figure(fig, png_dir, pdf_dir, "Fig06_Raw_vs_Calibrated_Reliability")

    # Fig07 calibration deltas
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, raw_col, cal_col, title in [
        (axes[0],"raw_brier_mean","calibrated_brier_mean","Brier Score"),
        (axes[1],"raw_ece_mean","calibrated_ece_mean","ECE"),
        (axes[2],"raw_log_loss_mean","calibrated_log_loss_mean","Log-loss")]:
        x = np.arange(len(calibration_summary)); w=.34
        ax.bar(x-w/2, calibration_summary[raw_col], width=w, label="Raw"); ax.bar(x+w/2, calibration_summary[cal_col], width=w, label="Calibrated")
        ax.set_xticks(x); ax.set_xticklabels(calibration_summary["model_name"], rotation=20, ha="right"); ax.set_title(title); ax.grid(axis="y",alpha=.25)
    axes[0].legend(); fig.suptitle("Effect of Temperature Scaling", fontsize=14, fontweight="bold")
    save_figure(fig, png_dir, pdf_dir, "Fig07_Calibration_Improvement")

    # Fig08 operating point
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(threshold_summary)); w=.28
    ax.bar(x-w/2, threshold_summary["sensitivity_mean"], width=w, label="Sensitivity"); ax.bar(x+w/2, threshold_summary["specificity_mean"], width=w, label="Specificity")
    ax.axhline(TARGET_SENSITIVITY, ls="--", lw=1.4, label="Target sensitivity 0.95")
    ax.set_xticks(x); ax.set_xticklabels(threshold_summary["model_name"]); ax.set_ylim(.85,1.01); ax.set_ylabel("Outer-test performance"); ax.set_title("Calibration-Selected Clinical Operating Point", fontweight="bold"); ax.grid(axis="y",alpha=.25); ax.legend()
    save_figure(fig, png_dir, pdf_dir, "Fig08_Clinical_Operating_Point")

    # Fig09 risk coverage
    agg = selective.groupby(["model_key","model_name","coverage"], as_index=False).agg(error_rate=("accuracy", lambda s: 1-float(np.mean(s))))
    fig, ax = plt.subplots(figsize=(8.5,6))
    for m in EXPECTED_MODELS:
        g=agg[agg.model_key==m].sort_values("coverage"); ax.plot(g.coverage,g.error_rate,marker="o",lw=2,label=model_name(m))
    ax.set_xlabel("Coverage (fraction automatically classified)"); ax.set_ylabel("Error rate"); ax.set_title("Risk–Coverage Curves",fontweight="bold"); ax.grid(alpha=.25); ax.legend()
    save_figure(fig,png_dir,pdf_dir,"Fig09_Risk_Coverage")

    # Fig10 referral capture
    agg2 = capture.groupby(["model_key","model_name","referral_rate"],as_index=False).agg(error_capture=("error_capture_fraction","mean"), fn_capture=("fn_capture_fraction","mean"))
    fig, axes=plt.subplots(1,2,figsize=(13,5))
    for m in EXPECTED_MODELS:
        g=agg2[agg2.model_key==m].sort_values("referral_rate"); axes[0].plot(g.referral_rate,g.error_capture,marker="o",lw=2,label=model_name(m)); axes[1].plot(g.referral_rate,g.fn_capture,marker="o",lw=2,label=model_name(m))
    axes[0].set_title("All-error capture"); axes[1].set_title("False-negative capture")
    for ax in axes: ax.set_xlabel("Referral rate"); ax.set_ylabel("Fraction captured"); ax.grid(alpha=.25)
    axes[0].legend(); fig.suptitle("How Much Error Is Removed by Referring Uncertain Cases?",fontsize=14,fontweight="bold")
    save_figure(fig,png_dir,pdf_dir,"Fig10_Referral_Error_Capture")

    # Fig11 stability
    fig, axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].bar(stability_summary.model_name,stability_summary.prediction_agreement_fraction); axes[0].set_ylim(.8,1.01); axes[0].set_title("3-seed hard-prediction agreement"); axes[0].set_ylabel("Agreement fraction"); axes[0].grid(axis="y",alpha=.25)
    axes[1].bar(stability_summary.model_name,stability_summary.mean_probability_sd_across_seeds); axes[1].set_title("Probability variability across seeds"); axes[1].set_ylabel("Mean probability SD"); axes[1].grid(axis="y",alpha=.25)
    fig.suptitle("Stochastic Stability",fontsize=14,fontweight="bold"); save_figure(fig,png_dir,pdf_dir,"Fig11_Model_Stability")

    # Fig12 duplicate group dist
    counts=leakage_groups.groupby("n_images").size()
    fig,ax=plt.subplots(figsize=(8,5.5)); ax.bar(counts.index.astype(str),counts.values); ax.set_xlabel("Images per similarity group"); ax.set_ylabel("Number of groups"); ax.set_title("Similarity-Group Size Distribution",fontweight="bold"); ax.grid(axis="y",alpha=.25)
    save_figure(fig,png_dir,pdf_dir,"Fig12_Duplicate_Group_Distribution")

    # Fig13 leakage comparison
    actual=int(leakage_summary.iloc[0]["actual_cross_fold_group_leakage"]); mean_naive=float(leakage_sim["split_similarity_groups"].mean())
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].hist(leakage_sim["split_similarity_groups"],bins=25); axes[0].axvline(actual,ls="--",lw=2,label=f"Group-aware actual = {actual}"); axes[0].set_xlabel("Similarity groups split across folds"); axes[0].set_ylabel("Simulation count"); axes[0].set_title(f"Naive image-wise CV (mean={mean_naive:.1f})"); axes[0].legend()
    axes[1].hist(leakage_sim["affected_image_fraction"]*100,bins=25); axes[1].axvline(0,ls="--",lw=2,label="Group-aware actual = 0%") ; axes[1].set_xlabel("Images affected by split groups (%)"); axes[1].set_ylabel("Simulation count"); axes[1].set_title("Potential similarity contamination"); axes[1].legend()
    fig.suptitle(f"Naive Image-wise 5-fold CV vs Similarity-Group-Safe CV ({LEAKAGE_SIM_REPS:,} simulations)",fontsize=14,fontweight="bold")
    save_figure(fig,png_dir,pdf_dir,"Fig13_Naive_vs_GroupSafe_Leakage")

    # Fig14 hard cases
    order=["easy_stable","occasional_instability","difficult","persistent_difficult","universal_failure"]
    counts=hard_cases["difficulty_category"].value_counts().reindex(order,fill_value=0)
    fig,ax=plt.subplots(figsize=(9,5.5)); xpos=np.arange(len(counts)); ax.bar(xpos,counts.values); ax.set_xticks(xpos); ax.set_xticklabels(counts.index,rotation=20,ha="right"); ax.set_ylabel("Unique images"); ax.set_title("Cross-Model/Seed Hard-Case Consistency",fontweight="bold"); ax.grid(axis="y",alpha=.25)
    save_figure(fig,png_dir,pdf_dir,"Fig14_Hard_Case_Consistency")

    # Fig15 architecture agreement
    fig,ax=plt.subplots(figsize=(8,5)); x=np.arange(len(agreement_pairs)); ax.bar(x,agreement_pairs["majority_prediction_agreement"]); ax.set_xticks(x); ax.set_xticklabels([f"{a}\nvs\n{b}" for a,b in zip(agreement_pairs.model_a,agreement_pairs.model_b)]); ax.set_ylim(.9,1.005); ax.set_ylabel("Agreement fraction"); ax.set_title("Cross-Architecture Majority-Prediction Agreement",fontweight="bold"); ax.grid(axis="y",alpha=.25)
    save_figure(fig,png_dir,pdf_dir,"Fig15_Cross_Model_Agreement")

    # Fig16 DCA
    fig,ax=plt.subplots(figsize=(9,6))
    for m in EXPECTED_MODELS:
        g=dca[(dca.model_key==m)&(dca.strategy=="model")]; ax.plot(g.threshold_probability,g.net_benefit,lw=2,label=model_name(m))
    ref=dca[(dca.model_key==EXPECTED_MODELS[0])]; allg=ref[ref.strategy=="treat_all"]; none=ref[ref.strategy=="treat_none"]
    ax.plot(allg.threshold_probability,allg.net_benefit,ls="--",lw=1.5,label="Treat all"); ax.plot(none.threshold_probability,none.net_benefit,ls=":",lw=1.5,label="Treat none")
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit"); ax.set_title("Decision Curve Analysis (Seed-Averaged OOF Probabilities)",fontweight="bold"); ax.set_ylim(-.1,.55); ax.grid(alpha=.25); ax.legend()
    save_figure(fig,png_dir,pdf_dir,"Fig16_Decision_Curve")



def build_contact_sheet(png_dir: Path, output_path: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageOps
        candidates = sorted([p for p in png_dir.glob("Fig*.png") if p.is_file()])[:16]
        if not candidates:
            return
        thumb_w, thumb_h = 560, 390
        margin, title_h, cols = 18, 34, 2
        rows = int(math.ceil(len(candidates) / cols))
        sheet = Image.new("RGB", (cols * (thumb_w + margin) + margin, rows * (thumb_h + title_h + margin) + margin), "white")
        draw = ImageDraw.Draw(sheet)
        for i, path in enumerate(candidates):
            image = Image.open(path).convert("RGB")
            fitted = ImageOps.contain(image, (thumb_w, thumb_h))
            row, col = divmod(i, cols)
            x = margin + col * (thumb_w + margin)
            y = margin + row * (thumb_h + title_h + margin)
            draw.text((x + 4, y + 5), path.stem, fill="black")
            sheet.paste(fitted, (x + (thumb_w - fitted.width) // 2, y + title_h))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output_path, quality=95)
    except Exception as exc:
        print("Preview contact sheet skipped:", type(exc).__name__, exc)

def build_manifest(root: Path) -> pd.DataFrame:
    rows=[]
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.endswith((".tmp", ".partial")):
            rows.append({"relative_path":str(p.relative_to(root)),"bytes":int(p.stat().st_size),"sha256":sha256_file(p)})
    return pd.DataFrame(rows)


def run_phase2(input_zip: Path, output_root: Path, run_root: Path | None = None, enable_xai: bool = True, bootstrap_reps: int = BOOTSTRAP_REPS, leakage_reps: int = LEAKAGE_SIM_REPS) -> dict[str, Any]:
    ensure_packages({"scipy":"scipy","sklearn":"scikit-learn","matplotlib":"matplotlib"})
    configure_matplotlib()
    source_audit = static_self_audit()
    input_zip = Path(input_zip)
    if not input_zip.is_file():
        raise FileNotFoundError(input_zip)

    output_root = Path(output_root)
    if output_root.exists():
        # never delete anything outside the dedicated Phase-2 directory
        if output_root.name != PHASE2_DIRNAME:
            raise RuntimeError(f"Refusing to clean unexpected output directory: {output_root}")
        shutil.rmtree(output_root)
    dirs = {
        "root": output_root,
        "tables": output_root / "tables",
        "figures_png": output_root / "figures_png",
        "figures_pdf": output_root / "figures_pdf",
        "statistics": output_root / "statistics",
        "manifests": output_root / "manifests",
        "xai": output_root / "xai",
    }
    for d in dirs.values(): Path(d).mkdir(parents=True, exist_ok=True)
    source_path = Path(__file__) if "__file__" in globals() else None
    if source_path is not None and source_path.is_file():
        shutil.copy2(source_path, dirs["manifests"] / "PHASE2_FULL_SOURCE.py")

    before_trials = snapshot_trial_tree(run_root)
    data = load_export_zip(input_zip)
    unique_images, integrity = integrity_gate(data, input_zip)
    atomic_json(dirs["manifests"] / "00_primary_integrity_verification.json", integrity)
    atomic_json(dirs["manifests"] / "00_phase2_source_safety.json", source_audit)

    metrics = data["metrics"].copy(); oof = data["oof"].copy()
    metrics["model_name"] = metrics["model_key"].map(model_name); oof["model_name"] = oof["model_key"].map(model_name)

    primary = primary_performance(metrics)
    atomic_csv(dirs["tables"] / "01_primary_performance.csv", primary)

    ci, pairwise, bootstrap_meta = group_cluster_bootstrap(oof, reps=bootstrap_reps)
    atomic_csv(dirs["statistics"] / "02_metric_confidence_intervals_group_clustered.csv", ci)
    atomic_csv(dirs["statistics"] / "03_pairwise_statistics_group_clustered_holm.csv", pairwise)
    atomic_json(dirs["statistics"] / "03_group_bootstrap_metadata.json", bootstrap_meta)

    # pooled confusion summary
    conf_rows=[]
    for m in EXPECTED_MODELS:
        g=oof[oof.model_key==m]; md=metric_dict(g.label.to_numpy(),g.prob_calibrated.to_numpy(float),g.pred.to_numpy(int)); conf_rows.append({"model_key":m,"model_name":model_name(m),**md,"unique_images":int(g.image_id.nunique()),"repeated_seed_predictions":int(len(g))})
    confusion=pd.DataFrame(conf_rows); atomic_csv(dirs["tables"] / "04_confusion_summary.csv", confusion)

    calibration_summary, reliability = calibration_audit(metrics,oof)
    atomic_csv(dirs["tables"] / "05_calibration_summary.csv",calibration_summary); atomic_csv(dirs["tables"] / "05_reliability_bins_equal_frequency.csv",reliability)

    threshold_trials, threshold_summary=threshold_operating_audit(metrics)
    atomic_csv(dirs["tables"] / "06_temperature_threshold_audit.csv",threshold_trials); atomic_csv(dirs["tables"] / "07_clinical_operating_points.csv",threshold_summary)

    selective,capture=selective_referral_analysis(oof)
    atomic_csv(dirs["tables"] / "08_selective_referral_trial_level.csv",selective); atomic_csv(dirs["tables"] / "08_referral_error_capture.csv",capture)

    leakage_summary, leakage_groups, leakage_sim=similarity_leakage_audit(unique_images,reps=leakage_reps)
    atomic_csv(dirs["tables"] / "09_similarity_group_audit.csv",leakage_summary); atomic_csv(dirs["tables"] / "09_similarity_group_sizes.csv",leakage_groups); atomic_csv(dirs["tables"] / "10_naive_split_leakage_simulation.csv",leakage_sim)

    hard_cases, architecture_agreement, agreement_pairs=hard_case_analysis(oof)
    atomic_csv(dirs["tables"] / "11_hard_case_consistency.csv",hard_cases); atomic_csv(dirs["tables"] / "11_architecture_majority_agreement_by_image.csv",architecture_agreement); atomic_csv(dirs["tables"] / "11_pairwise_architecture_agreement.csv",agreement_pairs)

    stability_per_image,stability_summary=stability_analysis(oof,metrics)
    atomic_csv(dirs["tables"] / "12_seed_stability_per_image.csv",stability_per_image); atomic_csv(dirs["tables"] / "12_model_stability_summary.csv",stability_summary)

    dca=decision_curve(oof); atomic_csv(dirs["tables"] / "13_decision_curve_analysis.csv",dca)

    checkpoints=checkpoint_inventory(run_root); atomic_csv(dirs["tables"] / "14_checkpoint_inventory.csv",checkpoints)
    xai_cases=choose_xai_cases(hard_cases,oof,fold=1); atomic_csv(dirs["tables"] / "15_xai_case_selection.csv",xai_cases)

    make_figures(primary,ci,pairwise,oof,calibration_summary,reliability,threshold_trials,threshold_summary,selective,capture,leakage_summary,leakage_groups,leakage_sim,hard_cases,agreement_pairs,stability_summary,dca,dirs["figures_png"],dirs["figures_pdf"])

    if enable_xai:
        xai_df,xai_meta=run_conditional_xai(checkpoints,xai_cases,oof,dirs["xai"],dirs["figures_png"],dirs["figures_pdf"])
    else:
        xai_df=pd.DataFrame(); xai_meta={"status":"DISABLED_FOR_TEST","training_performed":False}
    atomic_csv(dirs["tables"] / "16_xai_status.csv",xai_df); atomic_json(dirs["manifests"] / "16_xai_verification.json",xai_meta)
    build_contact_sheet(dirs["figures_png"], dirs["root"] / "PHASE2_FIGURE_PREVIEW_SHEET.png")

    headline = {
        "version": VERSION,
        "primary_winner_accuracy": primary.sort_values("accuracy_mean", ascending=False).iloc[0]["model_name"],
        "primary_winner_roc_auc": primary.sort_values("roc_auc_mean", ascending=False).iloc[0]["model_name"],
        "best_calibrated_by_ece": primary.sort_values("ece_mean", ascending=True).iloc[0]["model_name"],
        "similarity_groups": int(leakage_summary.iloc[0]["similarity_groups"]),
        "multi_image_groups": int(leakage_summary.iloc[0]["multi_image_groups"]),
        "images_in_multi_image_groups": int(leakage_summary.iloc[0]["images_in_multi_image_groups"]),
        "actual_cross_fold_group_leakage": 0,
        "naive_split_mean_split_groups": float(leakage_sim["split_similarity_groups"].mean()),
        "naive_split_mean_affected_image_fraction": float(leakage_sim["affected_image_fraction"].mean()),
        "xai_status": xai_meta.get("status"),
        "training_performed_during_phase2": False,
    }
    atomic_json(dirs["root"] / "PHASE2_HEADLINE_SUMMARY.json", headline)

    # Preserve original reproducibility object inside Phase-2 package, plus a new composite object.
    (dirs["manifests"] / "PRIMARY_REPRODUCIBILITY_OBJECT.pkl").write_bytes(data["original_pkl_bytes"])
    analysis_payload={
        "version":VERSION,"integrity":integrity,"source_safety":source_audit,"primary_performance":primary,
        "confidence_intervals":ci,"pairwise_statistics":pairwise,"calibration_summary":calibration_summary,
        "threshold_trials":threshold_trials,"operating_summary":threshold_summary,"selective_referral":selective,
        "referral_capture":capture,"similarity_summary":leakage_summary,"hard_cases":hard_cases,
        "stability_summary":stability_summary,"checkpoint_inventory":checkpoints,"xai_meta":xai_meta,
    }
    with (dirs["root"] / "PHASE2_FULL_ANALYSIS.pkl").open("wb") as handle: pickle.dump(analysis_payload,handle,protocol=pickle.HIGHEST_PROTOCOL)

    after_trials=snapshot_trial_tree(run_root)
    primary_unchanged=(before_trials==after_trials)
    if run_root is not None and before_trials and not primary_unchanged:
        changed=sorted(set(before_trials).symmetric_difference(after_trials) | {k for k in set(before_trials)&set(after_trials) if before_trials[k]!=after_trials[k]})
        raise RuntimeError(f"Primary trial tree changed during Phase-2; refusing PASS. First changes: {changed[:10]}")

    verification={
        "version":VERSION,"status":"PASS","primary_model_training_performed_during_phase2":False,
        "primary_45_trials_modified":False if run_root is not None and before_trials else None,
        "primary_trial_tree_unchanged":primary_unchanged if run_root is not None and before_trials else None,
        "outer_folds_modified":False,"duplicate_groups_regenerated":False,
        "integrity_status":integrity["status"],"group_cluster_bootstrap":bootstrap_meta,
        "similarity_groups":int(leakage_summary.iloc[0]["similarity_groups"]),
        "multi_image_groups":int(leakage_summary.iloc[0]["multi_image_groups"]),
        "images_in_multi_image_groups":int(leakage_summary.iloc[0]["images_in_multi_image_groups"]),
        "actual_cross_fold_group_leakage":int(leakage_summary.iloc[0]["actual_cross_fold_group_leakage"]),
        "naive_split_mean_split_groups":float(leakage_sim["split_similarity_groups"].mean()),
        "naive_split_mean_affected_image_fraction":float(leakage_sim["affected_image_fraction"].mean()),
        "checkpoint_usable_count":int(((checkpoints.get("exists",pd.Series(dtype=bool))==True)&(checkpoints.get("metadata_valid",pd.Series(dtype=bool))==True)).sum()) if not checkpoints.empty else 0,
        "xai_status":xai_meta.get("status"),"completed_at_ns":now_ns(),
    }
    atomic_json(dirs["root"] / "PHASE2_FULL_VERIFICATION.json",verification)

    manifest=build_manifest(dirs["root"]); atomic_csv(dirs["manifests"] / "PHASE2_FULL_MANIFEST.csv",manifest)
    final_zip=dirs["root"] / "PHASE2_FULL_FINAL_EXPORT.zip"
    with zipfile.ZipFile(final_zip,"w",zipfile.ZIP_DEFLATED,allowZip64=True) as zf:
        for p in sorted(dirs["root"].rglob("*")):
            if p.is_file() and p!=final_zip and not p.name.endswith((".tmp",".partial")):
                zf.write(p,p.relative_to(dirs["root"]))
    verification["final_export_zip"] = str(final_zip); verification["final_export_sha256"] = sha256_file(final_zip); verification["final_export_bytes"] = int(final_zip.stat().st_size)
    atomic_json(dirs["root"] / "PHASE2_FULL_VERIFICATION.json",verification)

    print("="*92)
    print("PHASE2 FULL VERIFICATION: PASS")
    print("="*92)
    print("Primary training performed: FALSE")
    print("Primary 45 trials modified:", "FALSE" if verification["primary_45_trials_modified"] is False else "NOT CHECKED (local test/no run root)")
    print("Group-cluster bootstrap reps:", bootstrap_meta["valid_bootstrap_reps"])
    print("Similarity groups:", verification["similarity_groups"], "| multi-image:", verification["multi_image_groups"], "| images in multi-image groups:", verification["images_in_multi_image_groups"])
    print("Actual cross-fold group leakage: 0")
    print("Naive CV mean split groups:", f"{verification['naive_split_mean_split_groups']:.1f}")
    print("Naive CV mean affected images (%):", f"{100*verification['naive_split_mean_affected_image_fraction']:.1f}")
    print("Usable checkpoints:", verification["checkpoint_usable_count"], "| XAI:", verification["xai_status"])
    print("Final export:", final_zip)
    return {"verification":verification,"output_root":str(output_root),"final_zip":str(final_zip)}


def resolve_colab_paths() -> tuple[Path, Path, Path]:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    run_root=Path("/content/drive/MyDrive/DFU-ImageGuard/runs")/RUN_ID
    input_zip=run_root/EXPECTED_INPUT_NAME
    if not input_zip.is_file():
        print(f"Could not find {input_zip}. Upload the exact {EXPECTED_INPUT_NAME}.")
        from google.colab import files
        uploaded=files.upload()
        if EXPECTED_INPUT_NAME in uploaded:
            local=Path("/content")/EXPECTED_INPUT_NAME; local.write_bytes(uploaded[EXPECTED_INPUT_NAME]); input_zip=local
        elif len(uploaded)==1:
            name,data=next(iter(uploaded.items())); local=Path("/content")/name; local.write_bytes(data); input_zip=local
        else:
            raise RuntimeError("Exact final export ZIP was not resolved")
    output_root=run_root/PHASE2_DIRNAME
    return input_zip,output_root,run_root


def main() -> None:
    print("="*92); print(VERSION); print("READ-ONLY PHASE-2: no model training, no fold/group regeneration, no primary trial writes."); print("="*92)
    input_zip,output_root,run_root=resolve_colab_paths()
    result=run_phase2(input_zip,output_root,run_root=run_root,enable_xai=True)
    print("PHASE2_FULL_FINAL_EXPORT.zip SHA256:",result["verification"]["final_export_sha256"])


if __name__ == "__main__":
    main()

