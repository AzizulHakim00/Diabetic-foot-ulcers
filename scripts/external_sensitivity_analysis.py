from __future__ import annotations

import hashlib, json, os, pickle, random, zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "DFU_PHASE4_POSTHOC_NO_INFERENCE_V1_20260812"
RUN_ID = "RELIABLE_DFU_CV_V3_MISSING38"
RNG_SEED = 2026
BOOTSTRAP_REPS = 1000
DRIVE_ROOT = Path("/content/drive/MyDrive/DFU-ImageGuard")
RUN_ROOT = DRIVE_ROOT / "runs" / RUN_ID
PHASE4_ROOT = RUN_ROOT / "PHASE4_EXTERNAL_V6"
OUT_ROOT = RUN_ROOT / "PHASE4_POSTHOC_V1"
MODEL_NAMES = {
    "convnextv2_tiny": "ConvNeXtV2-Tiny",
    "densenet121": "DenseNet121",
    "mobilenetv3_large": "MobileNetV3-Large",
}


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def mount_drive() -> None:
    if Path("/content/drive/MyDrive").is_dir():
        return
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & ((p <= edges[i + 1]) if i == bins - 1 else (p < edges[i + 1]))
        if mask.any():
            ece += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def metric_row(y: np.ndarray, p: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
        roc_auc_score, average_precision_score, brier_score_loss, log_loss, confusion_matrix,
    )
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "ece": ece_score(y, p),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def load_inputs() -> dict[str, Any]:
    required = {
        "phase4_verification": PHASE4_ROOT / "PHASE4_EXTERNAL_V6_VERIFICATION.json",
        "overlap_details": PHASE4_ROOT / "tables" / "02_cross_dataset_overlap_details.csv",
        "external_checkpoint_predictions": PHASE4_ROOT / "tables" / "04_external_checkpoint_predictions.csv",
        "external_ensemble_predictions": PHASE4_ROOT / "tables" / "06_external_ensemble_predictions.csv",
        "external_ensemble_metrics": PHASE4_ROOT / "tables" / "07_external_ensemble_metrics.csv",
        "oof": RUN_ROOT / "tables" / "all_oof_predictions.csv",
    }
    for name, p in required.items():
        if not p.exists():
            raise FileNotFoundError(f"Required saved result missing ({name}): {p}")
    ver = json.loads(required["phase4_verification"].read_text(encoding="utf-8"))
    if ver.get("status") != "PASS":
        raise RuntimeError("Phase-4 V6 verification is not PASS")
    for forbidden in ["primary_training_performed", "fine_tuning_performed", "external_label_fitting_for_threshold_or_temperature"]:
        if bool(ver.get(forbidden, False)):
            raise RuntimeError(f"Phase-4 safety invariant violated: {forbidden}=True")
    return {
        "paths": required,
        "verification": ver,
        "overlap": pd.read_csv(required["overlap_details"]),
        "ext_ckpt": pd.read_csv(required["external_checkpoint_predictions"]),
        "ext_ens_saved": pd.read_csv(required["external_ensemble_predictions"]),
        "ext_metrics_saved": pd.read_csv(required["external_ensemble_metrics"]),
        "oof": pd.read_csv(required["oof"]),
    }


def validate_columns(inputs: dict[str, Any]) -> None:
    overlap = inputs["overlap"]
    ext = inputs["ext_ckpt"]
    oof = inputs["oof"]
    req_overlap = {"relative_path", "near_primary_candidate", "included_for_external_eval"}
    req_ext = {"model_key", "seed", "outer_fold", "relative_path", "external_group_id", "label", "prob_raw", "prob_calibrated", "pred"}
    req_oof = {"model_key", "seed", "outer_fold", "image_id", "label", "prob_raw", "prob_calibrated", "pred"}
    for name, frame, req in [("overlap", overlap, req_overlap), ("external predictions", ext, req_ext), ("OOF", oof, req_oof)]:
        missing = req - set(frame.columns)
        if missing:
            raise RuntimeError(f"{name} missing columns: {sorted(missing)}")


def ensemble_external(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_key, rel), g in preds.groupby(["model_key", "relative_path"], sort=True):
        rows.append({
            "model_key": model_key,
            "model_name": MODEL_NAMES.get(model_key, model_key),
            "relative_path": str(rel),
            "external_group_id": str(g.external_group_id.iloc[0]),
            "label": int(g.label.iloc[0]),
            "prob_raw_mean": float(g.prob_raw.mean()),
            "prob_calibrated_mean": float(g.prob_calibrated.mean()),
            "pred_majority": int(g.pred.mean() >= 0.5),
            "n_checkpoints": int(len(g)),
        })
    return pd.DataFrame(rows)


def external_metrics(ens: pd.DataFrame, analysis_set: str) -> pd.DataFrame:
    rows = []
    for model_key, g in ens.groupby("model_key", sort=True):
        rows.append({
            "analysis_set": analysis_set,
            "model_key": model_key,
            "model_name": MODEL_NAMES.get(model_key, model_key),
            "n": int(len(g)),
            "groups": int(g.external_group_id.astype(str).nunique()),
            **metric_row(g.label.values, g.prob_calibrated_mean.values, g.pred_majority.values),
        })
    return pd.DataFrame(rows)


def cluster_bootstrap(ens: pd.DataFrame, analysis_set: str, reps: int = BOOTSTRAP_REPS) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    metrics = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc", "roc_auc", "pr_auc", "brier", "log_loss", "ece"]
    rows = []
    for model_key, g in ens.groupby("model_key", sort=True):
        groups = g.external_group_id.astype(str).unique()
        idxmap = {gr: np.where(g.external_group_id.astype(str).values == gr)[0] for gr in groups}
        vals = {m: [] for m in metrics}
        for _ in range(reps):
            sampled = rng.choice(groups, size=len(groups), replace=True)
            idx = np.concatenate([idxmap[gr] for gr in sampled])
            b = g.iloc[idx]
            if b.label.nunique() < 2:
                continue
            mm = metric_row(b.label.values, b.prob_calibrated_mean.values, b.pred_majority.values)
            for m in metrics:
                vals[m].append(mm[m])
        point = metric_row(g.label.values, g.prob_calibrated_mean.values, g.pred_majority.values)
        for m in metrics:
            arr = np.asarray(vals[m], dtype=float)
            rows.append({
                "analysis_set": analysis_set,
                "model_key": model_key,
                "model_name": MODEL_NAMES.get(model_key, model_key),
                "metric": m,
                "estimate": float(point[m]),
                "ci_low": float(np.nanquantile(arr, 0.025)) if len(arr) else np.nan,
                "ci_high": float(np.nanquantile(arr, 0.975)) if len(arr) else np.nan,
                "valid_bootstrap_reps": int(len(arr)),
            })
    return pd.DataFrame(rows)


def near_candidate_analysis(inputs: dict[str, Any]) -> dict[str, pd.DataFrame]:
    overlap = inputs["overlap"].copy()
    ext = inputs["ext_ckpt"].copy()
    included = bool_series(overlap["included_for_external_eval"])
    near = bool_series(overlap["near_primary_candidate"])
    near_paths = set(overlap.loc[included & near, "relative_path"].astype(str))
    all_paths = set(overlap.loc[included, "relative_path"].astype(str))
    if not near_paths:
        raise RuntimeError("No near-primary candidates found; expected Phase-4 audit candidates")
    ext_all = ext[ext.relative_path.astype(str).isin(all_paths)].copy()
    ext_filtered = ext_all[~ext_all.relative_path.astype(str).isin(near_paths)].copy()
    ens_all = ensemble_external(ext_all)
    ens_filtered = ensemble_external(ext_filtered)
    met_all = external_metrics(ens_all, "all_external")
    met_filtered = external_metrics(ens_filtered, "exclude_near_primary_phash_le_4")
    merged = met_all.merge(met_filtered, on=["model_key", "model_name"], suffixes=("_all", "_filtered"))
    metric_cols = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc", "roc_auc", "pr_auc", "brier", "log_loss", "ece"]
    delta_rows = []
    for _, r in merged.iterrows():
        for m in metric_cols:
            delta_rows.append({
                "model_key": r.model_key, "model_name": r.model_name, "metric": m,
                "all_external": r[f"{m}_all"], "filtered_external": r[f"{m}_filtered"],
                "filtered_minus_all": r[f"{m}_filtered"] - r[f"{m}_all"],
            })
    audit = pd.DataFrame([{
        "external_images_before": int(len(all_paths)),
        "near_primary_candidates_removed": int(len(near_paths)),
        "external_images_after": int(len(all_paths - near_paths)),
        "removed_fraction": float(len(near_paths) / max(len(all_paths), 1)),
        "positive_removed": int(overlap.loc[included & near, "label"].sum()) if "label" in overlap.columns else np.nan,
        "negative_removed": int((1 - overlap.loc[included & near, "label"]).sum()) if "label" in overlap.columns else np.nan,
    }])
    boot_filtered = cluster_bootstrap(ens_filtered, "exclude_near_primary_phash_le_4")
    return {
        "audit": audit,
        "metrics_all": met_all,
        "metrics_filtered": met_filtered,
        "delta": pd.DataFrame(delta_rows),
        "bootstrap_filtered": boot_filtered,
        "ensemble_filtered": ens_filtered,
        "near_paths": pd.DataFrame({"relative_path": sorted(near_paths)}),
    }


def checkpoint_identities(ext_ckpt: pd.DataFrame) -> pd.DataFrame:
    ids = ext_ckpt[["model_key", "seed", "outer_fold"]].drop_duplicates().sort_values(["model_key", "seed", "outer_fold"]).reset_index(drop=True)
    counts = ids.groupby("model_key").size().to_dict()
    if int(len(ids)) != 7:
        raise RuntimeError(f"Expected 7 external checkpoint identities, found {len(ids)}")
    return ids


def per_checkpoint_metrics(frame: pd.DataFrame, domain: str, prob_col: str = "prob_calibrated", pred_col: str = "pred") -> pd.DataFrame:
    rows = []
    for keys, g in frame.groupby(["model_key", "seed", "outer_fold"], sort=True):
        rows.append({
            "domain": domain,
            "model_key": keys[0], "model_name": MODEL_NAMES.get(keys[0], keys[0]),
            "seed": int(keys[1]), "outer_fold": int(keys[2]), "n": int(len(g)),
            **metric_row(g.label.values, g[prob_col].values, g[pred_col].values),
        })
    return pd.DataFrame(rows)


def ensemble_internal_matched(oof_selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Same selected checkpoint identities; within each model these are the same outer fold across seeds in the current run.
    for (model_key, image_id), g in oof_selected.groupby(["model_key", "image_id"], sort=True):
        if g.label.nunique() != 1:
            raise RuntimeError(f"Inconsistent label for internal image {image_id}")
        rows.append({
            "model_key": model_key,
            "model_name": MODEL_NAMES.get(model_key, model_key),
            "image_id": str(image_id),
            "label": int(g.label.iloc[0]),
            "prob_raw_mean": float(g.prob_raw.mean()),
            "prob_calibrated_mean": float(g.prob_calibrated.mean()),
            "pred_majority": int(g.pred.mean() >= 0.5),
            "n_checkpoints": int(len(g)),
        })
    return pd.DataFrame(rows)


def matched_transfer_analysis(inputs: dict[str, Any], filtered_ens: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ext = inputs["ext_ckpt"].copy()
    oof = inputs["oof"].copy()
    ids = checkpoint_identities(ext)
    key = pd.MultiIndex.from_frame(ids[["model_key", "seed", "outer_fold"]])
    oof_key = pd.MultiIndex.from_frame(oof[["model_key", "seed", "outer_fold"]])
    oof_sel = oof[oof_key.isin(key)].copy()
    ext_key = pd.MultiIndex.from_frame(ext[["model_key", "seed", "outer_fold"]])
    ext_sel = ext[ext_key.isin(key)].copy()
    if oof_sel.empty or ext_sel.empty:
        raise RuntimeError("Matched checkpoint selection produced empty data")

    internal_pc = per_checkpoint_metrics(oof_sel, "internal_oof_matched")
    external_pc = per_checkpoint_metrics(ext_sel, "external_full")
    pc = internal_pc.merge(external_pc, on=["model_key", "model_name", "seed", "outer_fold"], suffixes=("_internal", "_external"))
    metric_cols = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc", "roc_auc", "pr_auc", "brier", "log_loss", "ece"]
    rows = []
    for _, r in pc.iterrows():
        for m in metric_cols:
            rows.append({
                "model_key": r.model_key, "model_name": r.model_name, "seed": int(r.seed), "outer_fold": int(r.outer_fold),
                "metric": m, "internal": r[f"{m}_internal"], "external": r[f"{m}_external"],
                "external_minus_internal": r[f"{m}_external"] - r[f"{m}_internal"],
            })
    pc_long = pd.DataFrame(rows)

    int_ens = ensemble_internal_matched(oof_sel)
    ext_full_ens = ensemble_external(ext_sel)
    int_met = []
    for model_key, g in int_ens.groupby("model_key", sort=True):
        int_met.append({"model_key": model_key, "model_name": MODEL_NAMES.get(model_key, model_key), "domain": "internal_matched_ensemble", "n": len(g), **metric_row(g.label.values, g.prob_calibrated_mean.values, g.pred_majority.values)})
    int_met = pd.DataFrame(int_met)
    ext_full_met = external_metrics(ext_full_ens, "external_full").drop(columns=["groups"]).rename(columns={"analysis_set": "domain"})
    ext_filtered_met = external_metrics(filtered_ens, "external_filtered_no_near").drop(columns=["groups"]).rename(columns={"analysis_set": "domain"})
    combined = pd.concat([int_met, ext_full_met, ext_filtered_met], ignore_index=True, sort=False)
    long_rows = []
    for _, r in combined.iterrows():
        for m in metric_cols:
            long_rows.append({"model_key": r.model_key, "model_name": r.model_name, "domain": r.domain, "n": int(r.n), "metric": m, "value": r[m]})
    long_df = pd.DataFrame(long_rows)
    pivot = long_df.pivot_table(index=["model_key", "model_name", "metric"], columns="domain", values="value", aggfunc="first").reset_index()
    if "internal_matched_ensemble" in pivot.columns:
        for col in ["external_full", "external_filtered_no_near"]:
            if col in pivot.columns:
                pivot[f"{col}_minus_internal_matched"] = pivot[col] - pivot["internal_matched_ensemble"]

    identity_audit = ids.copy()
    identity_audit["internal_oof_rows"] = identity_audit.apply(lambda r: len(oof_sel[(oof_sel.model_key == r.model_key) & (oof_sel.seed == r.seed) & (oof_sel.outer_fold == r.outer_fold)]), axis=1)
    identity_audit["external_rows"] = identity_audit.apply(lambda r: len(ext_sel[(ext_sel.model_key == r.model_key) & (ext_sel.seed == r.seed) & (ext_sel.outer_fold == r.outer_fold)]), axis=1)
    return {
        "identities": identity_audit,
        "per_checkpoint_long": pc_long,
        "matched_ensemble_long": long_df,
        "matched_ensemble_compare": pivot,
        "internal_ensemble_predictions": int_ens,
    }


def configure_plotting() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({"figure.dpi": 120, "savefig.dpi": 350, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})


def save_fig(fig, stem: str) -> None:
    import matplotlib.pyplot as plt
    for folder, ext, dpi in [(OUT_ROOT / "figures_png", ".png", 350), (OUT_ROOT / "figures_pdf", ".pdf", None)]:
        folder.mkdir(parents=True, exist_ok=True)
        kwargs = {"bbox_inches": "tight"}
        if dpi: kwargs["dpi"] = dpi
        fig.savefig(folder / f"{stem}{ext}", **kwargs)
    plt.close(fig)


def make_figures(near: dict[str, pd.DataFrame], matched: dict[str, pd.DataFrame]) -> None:
    import matplotlib.pyplot as plt
    cls_metrics = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc"]
    delta = near["delta"]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(cls_metrics)); width = 0.24
    for j, model_key in enumerate(MODEL_NAMES):
        g = delta[(delta.model_key == model_key) & (delta.metric.isin(cls_metrics))].set_index("metric").reindex(cls_metrics)
        ax.bar(x + (j - 1) * width, g.filtered_minus_all.values, width, label=MODEL_NAMES[model_key])
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(cls_metrics, rotation=25, ha="right")
    ax.set_ylabel("Filtered minus all-external")
    ax.set_title("Effect of Excluding Near-Primary pHash Candidates")
    ax.legend()
    save_fig(fig, "FigP4Post_01_NearCandidate_Exclusion_Delta")

    comp = matched["matched_ensemble_long"]
    fig, ax = plt.subplots(figsize=(12, 5))
    metrics = cls_metrics
    domains = ["internal_matched_ensemble", "external_full", "external_filtered_no_near"]
    labels = ["Internal matched", "External full", "External filtered"]
    x = np.arange(len(metrics)); width = 0.085
    offset = 0
    for m_idx, model_key in enumerate(MODEL_NAMES):
        for d_idx, domain in enumerate(domains):
            g = comp[(comp.model_key == model_key) & (comp.domain == domain) & (comp.metric.isin(metrics))].set_index("metric").reindex(metrics)
            pos = x + (offset - 4) * width
            ax.bar(pos, g.value.values, width, label=f"{MODEL_NAMES[model_key]} | {labels[d_idx]}")
            offset += 1
    ax.set_xticks(x); ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_ylim(0, 1.05); ax.set_title("Matched-Checkpoint Internal vs External Transfer")
    ax.legend(ncol=3, fontsize=7)
    save_fig(fig, "FigP4Post_02_Matched_Internal_External")

    cal_metrics = ["roc_auc", "pr_auc", "brier", "log_loss", "ece"]
    pivot = matched["matched_ensemble_compare"]
    fig, ax = plt.subplots(figsize=(10, 5))
    rows = pivot[pivot.metric.isin(cal_metrics)].copy()
    names = []
    vals = []
    for model_key in MODEL_NAMES:
        for metric in cal_metrics:
            r = rows[(rows.model_key == model_key) & (rows.metric == metric)]
            if r.empty or "external_full_minus_internal_matched" not in r.columns:
                continue
            names.append(f"{MODEL_NAMES[model_key]}\n{metric}")
            vals.append(float(r.iloc[0]["external_full_minus_internal_matched"]))
    ax.bar(np.arange(len(vals)), vals)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(np.arange(len(vals))); ax.set_xticklabels(names, rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("External full minus internal matched")
    ax.set_title("Matched Transfer Deltas: Discrimination and Calibration")
    save_fig(fig, "FigP4Post_03_Matched_Transfer_Deltas")


def display_results(tables: list[tuple[str, Path]]) -> None:
    from IPython.display import display, Markdown, Image as IPImage
    for title, p in tables:
        display(Markdown(f"## {title}"))
        display(pd.read_csv(p))
    for p in sorted((OUT_ROOT / "figures_png").glob("*.png")):
        display(Markdown(f"### {p.stem}"))
        display(IPImage(filename=str(p), width=1100))


def export_final(verification: dict[str, Any]) -> Path:
    atomic_json(OUT_ROOT / "PHASE4_POSTHOC_VERIFICATION.json", verification)
    pkl = OUT_ROOT / "PHASE4_POSTHOC_ANALYSIS.pkl"
    with pkl.open("wb") as f:
        pickle.dump(verification, f, protocol=pickle.HIGHEST_PROTOCOL)
    zpath = OUT_ROOT / "PHASE4_POSTHOC_FINAL_EXPORT.zip"
    tmp = zpath.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for sub in ["tables", "figures_png", "figures_pdf"]:
            root = OUT_ROOT / sub
            if root.exists():
                for p in root.rglob("*"):
                    if p.is_file(): zf.write(p, arcname=str(p.relative_to(OUT_ROOT)))
        for p in [OUT_ROOT / "PHASE4_POSTHOC_VERIFICATION.json", pkl]:
            if p.exists(): zf.write(p, arcname=p.name)
    os.replace(tmp, zpath)
    return zpath


def run() -> None:
    random.seed(RNG_SEED); np.random.seed(RNG_SEED)
    mount_drive(); configure_plotting(); OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print(VERSION)
    print("POST-HOC ONLY | NO TRAINING | NO CNN INFERENCE | EXISTING SAVED PREDICTIONS ONLY")
    print("=" * 100)
    inputs = load_inputs(); validate_columns(inputs)
    print("Phase-4 V6 verification/input audit: PASS")
    print("Computing near-primary exclusion sensitivity analysis...")
    near = near_candidate_analysis(inputs)
    print("Near-primary sensitivity analysis: PASS")
    print("Computing matched 7-checkpoint internal→external transfer...")
    matched = matched_transfer_analysis(inputs, near["ensemble_filtered"])
    print("Matched checkpoint transfer analysis: PASS")

    table_map = [
        ("01_near_candidate_removal_audit.csv", near["audit"]),
        ("02_near_candidate_paths.csv", near["near_paths"]),
        ("03_external_metrics_all_vs_filtered_delta.csv", near["delta"]),
        ("04_external_filtered_bootstrap_ci.csv", near["bootstrap_filtered"]),
        ("05_matched_checkpoint_identities.csv", matched["identities"]),
        ("06_matched_per_checkpoint_internal_external.csv", matched["per_checkpoint_long"]),
        ("07_matched_ensemble_internal_external_long.csv", matched["matched_ensemble_long"]),
        ("08_matched_ensemble_comparison.csv", matched["matched_ensemble_compare"]),
        ("09_external_filtered_ensemble_predictions.csv", near["ensemble_filtered"]),
        ("10_internal_matched_ensemble_predictions.csv", matched["internal_ensemble_predictions"]),
    ]
    for name, df in table_map:
        atomic_csv(OUT_ROOT / "tables" / name, df)
    make_figures(near, matched)

    input_hashes = {k: sha256_file(p) for k, p in inputs["paths"].items()}
    verification = {
        "status": "PASS", "version": VERSION, "run_id": RUN_ID,
        "training_performed": False, "cnn_inference_performed": False, "fine_tuning_performed": False,
        "external_threshold_or_calibration_fitting": False,
        "source_phase4_status": inputs["verification"].get("status"),
        "external_images_before": int(near["audit"].iloc[0].external_images_before),
        "near_primary_candidates_removed": int(near["audit"].iloc[0].near_primary_candidates_removed),
        "external_images_after": int(near["audit"].iloc[0].external_images_after),
        "matched_checkpoint_identities": int(len(matched["identities"])),
        "bootstrap_reps": BOOTSTRAP_REPS,
        "input_sha256": input_hashes,
        "output_root": str(OUT_ROOT),
    }
    zip_path = export_final(verification)
    display_tables = [
        ("NEAR-PRIMARY REMOVAL AUDIT", OUT_ROOT / "tables" / "01_near_candidate_removal_audit.csv"),
        ("EXTERNAL ALL vs FILTERED DELTAS", OUT_ROOT / "tables" / "03_external_metrics_all_vs_filtered_delta.csv"),
        ("FILTERED EXTERNAL BOOTSTRAP CI", OUT_ROOT / "tables" / "04_external_filtered_bootstrap_ci.csv"),
        ("MATCHED CHECKPOINT IDENTITIES", OUT_ROOT / "tables" / "05_matched_checkpoint_identities.csv"),
        ("MATCHED PER-CHECKPOINT INTERNAL→EXTERNAL", OUT_ROOT / "tables" / "06_matched_per_checkpoint_internal_external.csv"),
        ("MATCHED ENSEMBLE INTERNAL→EXTERNAL", OUT_ROOT / "tables" / "08_matched_ensemble_comparison.csv"),
    ]
    display_results(display_tables)
    print("\n" + "=" * 100)
    print("PHASE-4 POST-HOC VERIFICATION: PASS")
    print("Training performed: FALSE")
    print("CNN inference performed: FALSE")
    print("Near-primary candidates removed:", verification["near_primary_candidates_removed"])
    print("Matched checkpoint identities:", verification["matched_checkpoint_identities"])
    print("Final export:", zip_path)
    print("=" * 100)


if __name__ == "__main__":
    run()

