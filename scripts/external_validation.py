from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "DFU_PHASE4_UNIVERSAL_SAFE_V6_20260812"
RUN_ID = "RELIABLE_DFU_CV_V3_MISSING38"
EXTERNAL_DATASET_HANDLE = "ibrahimfateen/wound-classification"
PRIMARY_DATASET_HANDLE = "laithjj/diabetic-foot-ulcer-dfu"
PRIVATE_FROZEN_HANDLE = os.environ.get("DFU_FROZEN_BUNDLE_HANDLE", "")
EXPECTED_CHECKPOINTS = 7
IMAGE_SIZE = 224
CPU_BATCH_SIZE = 16
GPU_BATCH_SIZE = 64
PHASH_NEAR_THRESHOLD = 4
BOOTSTRAP_REPS = 1000
RNG_SEED = 2026
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

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


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def ensure_packages() -> None:
    packages = {
        "timm": "timm>=1.0.15",
        "kagglehub": "kagglehub>=0.3.0",
        "sklearn": "scikit-learn>=1.4.0",
        "PIL": "pillow>=10.0.0",
        "matplotlib": "matplotlib>=3.8.0",
    }
    missing = [pip for mod, pip in packages.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print("Installing missing Phase-4 packages:", ", ".join(missing), flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


def detect_and_prepare_runtime() -> dict[str, Any]:
    """Drive-first. If user accidentally remains on Kaggle runtime, fall back automatically."""
    in_colab = importlib.util.find_spec("google.colab") is not None
    if in_colab:
        from google.colab import drive
        if not Path("/content/drive/MyDrive").is_dir():
            drive.mount("/content/drive", force_remount=False)
        if not Path("/content/drive/MyDrive").is_dir():
            raise RuntimeError("Google Drive mount did not become available in Colab")
        base = Path("/content/drive/MyDrive/DFU-ImageGuard")
        run_root = base / "runs" / RUN_ID
        return {
            "mode": "colab_drive",
            "base": base,
            "run_root": run_root,
            "out_root": run_root / "PHASE4_EXTERNAL_V6",
        }

    # If Drive is already mounted by a compatible Jupyter runtime, prefer it.
    if Path("/content/drive/MyDrive").is_dir():
        base = Path("/content/drive/MyDrive/DFU-ImageGuard")
        run_root = base / "runs" / RUN_ID
        return {
            "mode": "drive_mounted_jupyter",
            "base": base,
            "run_root": run_root,
            "out_root": run_root / "PHASE4_EXTERNAL_V6",
        }

    if Path("/kaggle").exists() or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        base = Path("/kaggle/working/DFU-ImageGuard")
        run_root = base / "runs" / RUN_ID
        return {
            "mode": "kaggle_auto_private",
            "base": base,
            "run_root": run_root,
            "out_root": run_root / "PHASE4_EXTERNAL_V6",
        }

    raise RuntimeError(
        "Unsupported runtime: neither Google Colab/Drive nor Kaggle was detected. "
        "Open the notebook in Google Colab or Kaggle."
    )


def find_one(root: Path, name: str) -> Path:
    matches = [p for p in root.rglob(name) if p.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name} under {root}; found {len(matches)}")
    return matches[0]


def normalize_kaggle_frozen_bundle(downloaded: Path) -> Path:
    """Accept unpacked bundle, Kaggle CLI folder-zips, or the master ZIP."""
    downloaded = Path(downloaded)

    manifests = [p for p in downloaded.rglob("FROZEN_INPUT_MANIFEST.json") if p.is_file()]
    for manifest in manifests:
        root = manifest.parent
        if (
            (root / "tables" / "14_checkpoint_inventory.csv").exists()
            and (root / "tables" / "all_oof_predictions.csv").exists()
            and (root / "tables" / "fold_seed_metrics.csv").exists()
            and (root / "checkpoints").is_dir()
        ):
            return root

    extract_root = Path("/kaggle/working/dfu_phase4_frozen_v6") if Path("/kaggle").exists() else Path.cwd() / "dfu_phase4_frozen_v6"

    # Kaggle CLI -r zip uploads each directory as checkpoints.zip/tables.zip.
    for manifest in manifests:
        root = manifest.parent
        checkpoints_zip = root / "checkpoints.zip"
        tables_zip = root / "tables.zip"
        if checkpoints_zip.exists() and tables_zip.exists():
            if extract_root.exists():
                shutil.rmtree(extract_root)
            (extract_root / "tables").mkdir(parents=True, exist_ok=True)
            (extract_root / "checkpoints").mkdir(parents=True, exist_ok=True)
            shutil.copy2(manifest, extract_root / "FROZEN_INPUT_MANIFEST.json")
            for optional in ["PHASE2_FULL_VERIFICATION.json", "REPAIR45_FINAL_VERIFICATION.json"]:
                p = root / optional
                if p.exists():
                    shutil.copy2(p, extract_root / optional)
            raw_t = extract_root / "_raw_tables"
            raw_c = extract_root / "_raw_checkpoints"
            raw_t.mkdir(); raw_c.mkdir()
            with zipfile.ZipFile(tables_zip, "r") as zf:
                zf.extractall(raw_t)
            with zipfile.ZipFile(checkpoints_zip, "r") as zf:
                zf.extractall(raw_c)
            for name in ["14_checkpoint_inventory.csv", "all_oof_predictions.csv", "fold_seed_metrics.csv"]:
                shutil.copy2(find_one(raw_t, name), extract_root / "tables" / name)
            inv = pd.read_csv(extract_root / "tables" / "14_checkpoint_inventory.csv")
            if "bundle_relative_path" not in inv.columns:
                raise RuntimeError("Frozen inventory is missing bundle_relative_path")
            for rel in inv["bundle_relative_path"].astype(str):
                src = find_one(raw_c, Path(rel).name)
                dst = extract_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            shutil.rmtree(raw_t, ignore_errors=True)
            shutil.rmtree(raw_c, ignore_errors=True)
            return extract_root

    # Master ZIP fallback.
    master_zips = [p for p in downloaded.rglob("DFU_PHASE4_FROZEN_INPUT.zip") if p.is_file()]
    if master_zips:
        if extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(master_zips[0], "r") as zf:
            zf.extractall(extract_root)
        return normalize_kaggle_frozen_bundle(extract_root)

    raise FileNotFoundError(
        f"Could not normalize frozen input bundle under {downloaded}. "
        "Expected unpacked bundle, checkpoints.zip+tables.zip, or DFU_PHASE4_FROZEN_INPUT.zip."
    )


def resolve_frozen_inputs(runtime: dict[str, Any]) -> dict[str, Any]:
    mode = runtime["mode"]
    if mode in {"colab_drive", "drive_mounted_jupyter"}:
        run_root = runtime["run_root"]
        phase2 = run_root / "PHASE2_FULL"
        inv_path = phase2 / "tables" / "14_checkpoint_inventory.csv"
        oof_path = run_root / "tables" / "all_oof_predictions.csv"
        metrics_path = run_root / "tables" / "fold_seed_metrics.csv"
        phase2_ver = phase2 / "PHASE2_FULL_VERIFICATION.json"
        for p in [inv_path, oof_path, metrics_path, phase2_ver]:
            if not p.exists():
                raise FileNotFoundError(f"Frozen Drive input missing: {p}")
        inv = pd.read_csv(inv_path)
        usable = inv.loc[(inv["exists"] == True) & (inv["metadata_valid"] == True)].copy()
        return {
            "root": run_root,
            "inventory": usable,
            "oof": pd.read_csv(oof_path),
            "metrics": pd.read_csv(metrics_path),
            "verification": json.loads(phase2_ver.read_text(encoding="utf-8")),
        }

    if mode == "kaggle_auto_private":
        import kagglehub
        if not PRIVATE_FROZEN_HANDLE:
            raise FileNotFoundError(
                "Set DFU_FROZEN_BUNDLE_HANDLE to the Kaggle handle containing "
                "the frozen checkpoint bundle, or provide the artifacts through Google Drive."
            )
        print("Kaggle runtime detected; fetching the configured frozen bundle.", flush=True)
        downloaded = Path(kagglehub.dataset_download(PRIVATE_FROZEN_HANDLE))
        bundle = normalize_kaggle_frozen_bundle(downloaded)
        inv = pd.read_csv(bundle / "tables" / "14_checkpoint_inventory.csv")
        if "bundle_relative_path" not in inv.columns:
            raise RuntimeError("Frozen inventory is missing bundle_relative_path")
        inv["path"] = inv["bundle_relative_path"].astype(str).map(lambda x: str(bundle / x))
        return {
            "root": bundle,
            "inventory": inv,
            "oof": pd.read_csv(bundle / "tables" / "all_oof_predictions.csv"),
            "metrics": pd.read_csv(bundle / "tables" / "fold_seed_metrics.csv"),
            "verification": json.loads((bundle / "FROZEN_INPUT_MANIFEST.json").read_text(encoding="utf-8")),
        }

    raise RuntimeError(f"Unknown runtime mode: {mode}")


def verify_frozen_inputs(frozen: dict[str, Any]) -> pd.DataFrame:
    ver = frozen["verification"]
    if isinstance(ver, dict) and ver.get("status") not in (None, "PASS"):
        raise RuntimeError(f"Frozen verification status is not PASS: {ver.get('status')}")
    inv = frozen["inventory"].copy()
    if len(inv) != EXPECTED_CHECKPOINTS:
        raise RuntimeError(f"Expected exactly {EXPECTED_CHECKPOINTS} frozen checkpoints; found {len(inv)}")
    unknown = set(inv["model_key"].astype(str)) - set(MODEL_NAMES)
    if unknown:
        raise RuntimeError(f"Unexpected model keys in frozen inventory: {sorted(unknown)}")
    for _, r in inv.iterrows():
        p = Path(str(r["path"]))
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint missing: {p}")
        actual = sha256_file(p)
        if actual != str(r["sha256"]):
            raise RuntimeError(f"Checkpoint SHA mismatch: {p}\nexpected={r['sha256']}\nactual={actual}")
    print(f"Frozen input verification: PASS ({len(inv)} checkpoints)", flush=True)
    return inv.sort_values(["model_key", "seed", "outer_fold"]).reset_index(drop=True)


def download_public_dataset(handle: str) -> Path:
    import kagglehub
    path = Path(kagglehub.dataset_download(handle))
    if not path.exists():
        raise FileNotFoundError(f"Dataset download did not produce a path for {handle}")
    return path


def norm_text(x: str) -> str:
    return " ".join(str(x).lower().replace("_", " ").replace("-", " ").split())


def classify_external_path(path: Path, root: Path) -> int | None:
    folder_text = " / ".join(norm_text(x) for x in path.relative_to(root).parts[:-1])
    positives = ("diabetic wounds", "diabetic wound", "diabetic foot", "diabeticfoot", "dfu")
    negatives = ("normal", "healthy foot", "healthy feet", "healthy")
    if any(tok in folder_text for tok in positives):
        return 1
    if any(tok in folder_text for tok in negatives):
        return 0
    return None


def discover_external_dataset(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_dirs: dict[str, int] = {}
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = p.relative_to(root)
        parent = norm_text(rel.parent.name)
        all_dirs[parent] = all_dirs.get(parent, 0) + 1
        label = classify_external_path(p, root)
        if label is not None:
            rows.append({"path": str(p), "relative_path": str(rel), "label": int(label), "class_dir": parent})
    out = pd.DataFrame(rows)
    if out.empty or set(out["label"].unique()) != {0, 1}:
        top = sorted(all_dirs.items(), key=lambda kv: kv[1], reverse=True)[:30]
        raise RuntimeError(f"External class discovery failed. Top detected image directories: {top}")
    out = out.drop_duplicates("path").sort_values("relative_path").reset_index(drop=True)
    return out


def resolve_primary_image_path(root: Path, relative_path: str) -> Path:
    rel = Path(str(relative_path))
    direct = root / rel
    if direct.exists():
        return direct
    parts = rel.parts
    suffix_matches = []
    for p in root.rglob(rel.name):
        if not p.is_file():
            continue
        if len(p.parts) >= len(parts) and tuple(p.parts[-len(parts):]) == tuple(parts):
            suffix_matches.append(p)
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    basename_matches = [p for p in root.rglob(rel.name) if p.is_file()]
    if len(basename_matches) == 1:
        return basename_matches[0]
    raise RuntimeError(f"Could not uniquely resolve primary image {relative_path}; candidates={len(basename_matches)}")


def dhash64(path: Path) -> int:
    from PIL import Image
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((9, 8)), dtype=np.uint8)
    bits = a[:, 1:] > a[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return int(value)


def fingerprint_one(path: str) -> tuple[str, str, str, int]:
    p = Path(path)
    return path, sha256_file(p), str(dhash64(p)), int(p.stat().st_size)


def build_fingerprints(df: pd.DataFrame, cache_path: Path, key_col: str, workers: int = 8) -> pd.DataFrame:
    expected_keys = set(df[key_col].astype(str))
    if cache_path.exists():
        cached = pd.read_csv(cache_path, dtype={"key": str, "phash64": str})
        required = {"key", "path", "sha256", "phash64", "size"}
        if required.issubset(cached.columns) and set(cached["key"].astype(str)) == expected_keys:
            existing = cached["path"].astype(str).map(lambda x: Path(x).exists())
            if bool(existing.all()):
                print("RESTORED fingerprint cache:", cache_path, flush=True)
                return cached
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(workers), os.cpu_count() or 2, 12))
    print(f"Building fingerprints with {workers} CPU workers: {len(df)} images", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fingerprint_one, str(r["path"])): str(r[key_col]) for _, r in df.iterrows()}
        total = len(futures)
        for done, fut in enumerate(as_completed(futures), start=1):
            key = futures[fut]
            path, sha, ph, size = fut.result()
            rows.append({"key": key, "path": path, "sha256": sha, "phash64": ph, "size": size})
            if done % 100 == 0 or done == total:
                print(f"  fingerprint {done}/{total}", flush=True)
    out = pd.DataFrame(rows).sort_values("key").reset_index(drop=True)
    atomic_csv(cache_path, out)
    return out


def hamming64(a: int, b: int) -> int:
    return int((int(a) ^ int(b)).bit_count())


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def audit_cross_dataset_overlap(external: pd.DataFrame, ext_fp: pd.DataFrame, primary_fp: pd.DataFrame, out_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ext = external.merge(ext_fp[["key", "sha256", "phash64"]], left_on="relative_path", right_on="key", how="left", validate="one_to_one")
    if ext[["sha256", "phash64"]].isna().any().any():
        raise RuntimeError("External fingerprint merge produced missing values")
    primary_sha = set(primary_fp["sha256"].astype(str))
    p_hashes = [int(x) for x in primary_fp["phash64"].astype(str)]
    ext["exact_primary_overlap"] = ext["sha256"].astype(str).isin(primary_sha)
    mins = []
    for i, ph in enumerate(ext["phash64"].astype(str), start=1):
        v = int(ph)
        best = min((hamming64(v, pv) for pv in p_hashes), default=64)
        mins.append(best)
        if i % 100 == 0 or i == len(ext):
            print(f"  cross-dataset pHash {i}/{len(ext)}", flush=True)
    ext["min_primary_phash_distance"] = mins
    ext["near_primary_candidate"] = ext["min_primary_phash_distance"] <= PHASH_NEAR_THRESHOLD
    # Only exact byte-identical overlaps are automatically excluded. Near candidates are reported, not silently removed.
    ext["included_for_external_eval"] = ~ext["exact_primary_overlap"]
    details = ext[[
        "relative_path", "label", "class_dir", "sha256", "phash64",
        "exact_primary_overlap", "min_primary_phash_distance", "near_primary_candidate",
        "included_for_external_eval",
    ]].copy()
    summary = pd.DataFrame([{
        "external_selected_images": int(len(details)),
        "external_positive": int((details["label"] == 1).sum()),
        "external_negative": int((details["label"] == 0).sum()),
        "exact_primary_overlap": int(details["exact_primary_overlap"].sum()),
        "near_primary_candidates_phash_le_4": int(details["near_primary_candidate"].sum()),
        "external_eval_images_after_exact_exclusion": int(details["included_for_external_eval"].sum()),
    }])
    atomic_csv(out_root / "tables" / "02_cross_dataset_overlap_details.csv", details)
    atomic_csv(out_root / "tables" / "02_cross_dataset_overlap_summary.csv", summary)
    return summary, details


def assign_external_similarity_groups(details: pd.DataFrame) -> pd.DataFrame:
    d = details.loc[details["included_for_external_eval"] == True].copy().reset_index(drop=True)
    n = len(d)
    uf = UnionFind(n)
    ph = [int(x) for x in d["phash64"].astype(str)]
    sha = d["sha256"].astype(str).tolist()
    first_sha: dict[str, int] = {}
    for i, s in enumerate(sha):
        if s in first_sha:
            uf.union(i, first_sha[s])
        else:
            first_sha[s] = i
    for i in range(n):
        for j in range(i + 1, n):
            if hamming64(ph[i], ph[j]) <= PHASH_NEAR_THRESHOLD:
                uf.union(i, j)
    mapping: dict[int, str] = {}
    ids = []
    for i in range(n):
        root = uf.find(i)
        if root not in mapping:
            mapping[root] = f"EG{len(mapping)+1:05d}"
        ids.append(mapping[root])
    d["external_group_id"] = ids
    return d


def build_base_model(model_key: str):
    import timm
    errors = []
    for name in MODEL_CANDIDATES[model_key]:
        try:
            model = timm.create_model(name, pretrained=False, num_classes=1, drop_rate=0.20)
            return model, name
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
    raise RuntimeError("Could not build frozen architecture: " + " | ".join(errors))


def load_checkpoint_model(model_key: str, checkpoint: Path, device):
    import torch
    model, arch = build_base_model(model_key)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") or payload.get("model")
    if state is None:
        raise RuntimeError(f"Checkpoint has no state_dict/model: {checkpoint}")
    state = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint/model mismatch {model_key}: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval().to(device)
    return model, arch


def image_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class ExternalDataset:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)
        self.tf = image_transform()
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, idx: int):
        from PIL import Image
        r = self.frame.iloc[idx]
        with Image.open(r["path"]) as im:
            x = self.tf(im.convert("RGB"))
        return x, int(r["label"]), str(r["relative_path"]), str(r["external_group_id"])


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    z = np.exp(x[~pos])
    out[~pos] = z / (1.0 + z)
    return out


def frozen_operating_point(oof: pd.DataFrame, model_key: str, seed: int, outer_fold: int) -> tuple[float, float]:
    g = oof[(oof["model_key"] == model_key) & (oof["seed"] == seed) & (oof["outer_fold"] == outer_fold)]
    if g.empty:
        raise RuntimeError(f"No frozen OOF rows for {model_key} seed={seed} fold={outer_fold}")
    for col in ["temperature", "threshold"]:
        if col not in g.columns:
            raise RuntimeError(f"OOF table lacks required frozen column: {col}")
    temps = pd.Series(g["temperature"]).dropna().astype(float).unique()
    thrs = pd.Series(g["threshold"]).dropna().astype(float).unique()
    if len(temps) != 1 or len(thrs) != 1:
        raise RuntimeError(
            f"Expected one frozen temperature/threshold for {model_key} seed={seed} fold={outer_fold}; "
            f"found temperature={len(temps)}, threshold={len(thrs)}"
        )
    return float(temps[0]), float(thrs[0])


def inference_cache_is_valid(cache: Path, ext_eval: pd.DataFrame, expected_sha: str) -> pd.DataFrame | None:
    if not cache.exists():
        return None
    try:
        df = pd.read_csv(cache)
    except Exception:
        return None
    needed = {
        "relative_path", "label", "logit", "prob_raw", "prob_calibrated", "pred",
        "external_group_id", "checkpoint_sha256",
    }
    if not needed.issubset(df.columns):
        return None
    if len(df) != len(ext_eval):
        return None
    if set(df["relative_path"].astype(str)) != set(ext_eval["relative_path"].astype(str)):
        return None
    if set(df["checkpoint_sha256"].astype(str)) != {str(expected_sha)}:
        return None
    return df


def run_checkpoint_inference(inv_row: pd.Series, oof: pd.DataFrame, ext_eval: pd.DataFrame, device, out_root: Path) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader

    model_key = str(inv_row["model_key"])
    seed = int(inv_row["seed"])
    fold = int(inv_row["outer_fold"])
    checkpoint = Path(str(inv_row["path"]))
    checkpoint_sha = str(inv_row["sha256"])
    temperature, threshold = frozen_operating_point(oof, model_key, seed, fold)

    cache = out_root / "checkpoint_predictions" / f"{model_key}_seed{seed}_fold{fold}.csv"
    restored = inference_cache_is_valid(cache, ext_eval, checkpoint_sha)
    if restored is not None:
        print(f"RESTORED checkpoint prediction: {model_key} seed={seed} fold={fold}", flush=True)
        return restored

    print(f"INFERENCE START [{device.type}]: {model_key} seed={seed} fold={fold}", flush=True)
    model, arch = load_checkpoint_model(model_key, checkpoint, device)
    dataset = ExternalDataset(ext_eval)
    batch_size = GPU_BATCH_SIZE if device.type == "cuda" else CPU_BATCH_SIZE
    workers = min(2 if device.type == "cpu" else 4, os.cpu_count() or 1)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    rows = []
    seen = 0
    with torch.inference_mode():
        for xb, yb, rels, groups in loader:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            logits = model(xb).reshape(-1).float().cpu().numpy()
            p_raw = sigmoid_np(logits)
            p_cal = sigmoid_np(logits / max(temperature, 1e-8))
            pred = (p_cal >= threshold).astype(int)
            for i in range(len(logits)):
                rows.append({
                    "model_key": model_key,
                    "model_name": MODEL_NAMES[model_key],
                    "seed": seed,
                    "outer_fold": fold,
                    "checkpoint_sha256": checkpoint_sha,
                    "architecture": arch,
                    "relative_path": str(rels[i]),
                    "external_group_id": str(groups[i]),
                    "label": int(yb[i]),
                    "logit": float(logits[i]),
                    "temperature": float(temperature),
                    "threshold": float(threshold),
                    "prob_raw": float(p_raw[i]),
                    "prob_calibrated": float(p_cal[i]),
                    "pred": int(pred[i]),
                })
            seen += len(logits)
            if seen % max(batch_size * 5, 1) == 0 or seen == len(dataset):
                print(f"  {model_key} s{seed} f{fold}: {seen}/{len(dataset)}", flush=True)

    out = pd.DataFrame(rows)
    atomic_csv(cache, out)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"INFERENCE PASS [{device.type}]: {model_key} seed={seed} fold={fold}", flush=True)
    return out


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & ((p <= edges[i + 1]) if i == bins - 1 else (p < edges[i + 1]))
        if mask.any():
            ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def metric_row(y: np.ndarray, p: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
        confusion_matrix, f1_score, log_loss, matthews_corrcoef, roc_auc_score,
    )
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = np.asarray(pred, dtype=int)
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
        "ece": float(ece_score(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def checkpoint_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["model_key", "model_name", "seed", "outer_fold"]
    for values, g in predictions.groupby(keys, sort=True):
        rows.append(dict(zip(keys, values)) | metric_row(g["label"], g["prob_calibrated"], g["pred"]))
    return pd.DataFrame(rows)


def ensemble_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_key, rel), g in predictions.groupby(["model_key", "relative_path"], sort=True):
        p = float(g["prob_calibrated"].mean())
        mean_threshold = float(g["threshold"].mean())
        rows.append({
            "model_key": model_key,
            "model_name": MODEL_NAMES[model_key],
            "relative_path": rel,
            "external_group_id": str(g["external_group_id"].iloc[0]),
            "label": int(g["label"].iloc[0]),
            "prob_raw_mean": float(g["prob_raw"].mean()),
            "prob_calibrated_mean": p,
            "pred_majority": int(g["pred"].mean() >= 0.5),
            "n_checkpoints": int(len(g)),
            "mean_threshold": mean_threshold,
            "decision_margin": float(abs(p - mean_threshold)),
        })
    return pd.DataFrame(rows)


def ensemble_metrics(ens: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_key, g in ens.groupby("model_key", sort=True):
        rows.append({
            "model_key": model_key,
            "model_name": MODEL_NAMES[model_key],
            "n": int(len(g)),
            **metric_row(g["label"], g["prob_calibrated_mean"], g["pred_majority"]),
        })
    return pd.DataFrame(rows)


def cluster_bootstrap_ci(ens: pd.DataFrame, reps: int = BOOTSTRAP_REPS) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    metric_names = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc", "roc_auc", "pr_auc", "brier", "log_loss", "ece"]
    rows = []
    for model_key, g in ens.groupby("model_key", sort=True):
        group_ids = g["external_group_id"].astype(str).unique()
        idx_map = {gid: np.flatnonzero(g["external_group_id"].astype(str).values == gid) for gid in group_ids}
        values = {m: [] for m in metric_names}
        for _ in range(reps):
            sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
            idx = np.concatenate([idx_map[x] for x in sampled])
            b = g.iloc[idx]
            if b["label"].nunique() < 2:
                continue
            mm = metric_row(b["label"], b["prob_calibrated_mean"], b["pred_majority"])
            for m in metric_names:
                values[m].append(mm[m])
        full = metric_row(g["label"], g["prob_calibrated_mean"], g["pred_majority"])
        for m in metric_names:
            a = np.asarray(values[m], dtype=float)
            rows.append({
                "model_key": model_key,
                "model_name": MODEL_NAMES[model_key],
                "metric": m,
                "estimate": float(full[m]),
                "ci_low": float(np.nanquantile(a, 0.025)) if len(a) else np.nan,
                "ci_high": float(np.nanquantile(a, 0.975)) if len(a) else np.nan,
                "valid_bootstrap_reps": int(len(a)),
            })
    return pd.DataFrame(rows)


def selective_referral(ens: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_key, g0 in ens.groupby("model_key", sort=True):
        g0 = g0.copy()
        g0["correct"] = (g0["label"].astype(int) == g0["pred_majority"].astype(int)).astype(int)
        g0["uncertainty"] = 1.0 - np.abs(g0["prob_calibrated_mean"].astype(float) - 0.5) * 2.0
        total_errors = int((g0["correct"] == 0).sum())
        for coverage in [1.00, 0.95, 0.90, 0.80]:
            keep_n = max(1, int(round(len(g0) * coverage)))
            kept = g0.sort_values(["uncertainty", "relative_path"], ascending=[True, True]).iloc[:keep_n]
            referred = g0.loc[~g0.index.isin(kept.index)]
            rows.append({
                "model_key": model_key,
                "model_name": MODEL_NAMES[model_key],
                "coverage": float(keep_n / len(g0)),
                "referral_fraction": float(1 - keep_n / len(g0)),
                "accuracy": float(kept["correct"].mean()),
                "errors_retained": int((kept["correct"] == 0).sum()),
                "errors_captured_by_referral": int((referred["correct"] == 0).sum()),
                "error_capture_fraction": float((referred["correct"] == 0).sum() / max(total_errors, 1)),
            })
    return pd.DataFrame(rows)


def internal_vs_external(fold_metrics: pd.DataFrame, ext_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_aliases = {
        "accuracy": ["accuracy", "acc"],
        "balanced_accuracy": ["balanced_accuracy", "bal_acc", "ba"],
        "sensitivity": ["sensitivity", "sens", "recall"],
        "specificity": ["specificity", "spec"],
        "f1": ["f1", "f1_score"],
        "mcc": ["mcc"],
        "roc_auc": ["roc_auc", "auc"],
        "pr_auc": ["pr_auc", "average_precision", "ap"],
        "brier": ["brier", "brier_score"],
        "log_loss": ["log_loss", "logloss"],
        "ece": ["ece"],
    }
    model_col = "model_key" if "model_key" in fold_metrics.columns else None
    if model_col is None:
        raise RuntimeError("fold_seed_metrics.csv lacks model_key")
    rows = []
    for _, er in ext_metrics.iterrows():
        mk = str(er["model_key"])
        internal = fold_metrics[fold_metrics[model_col].astype(str) == mk]
        if internal.empty:
            raise RuntimeError(f"No internal fold metrics for {mk}")
        for metric, candidates in metric_aliases.items():
            col = next((c for c in candidates if c in internal.columns), None)
            if col is None:
                continue
            iv = float(pd.to_numeric(internal[col], errors="coerce").mean())
            ev = float(er[metric])
            rows.append({
                "model_key": mk,
                "model_name": MODEL_NAMES[mk],
                "metric": metric,
                "internal_macro_mean": iv,
                "external_ensemble": ev,
                "external_minus_internal": ev - iv,
            })
    return pd.DataFrame(rows)


def configure_plotting() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 350,
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save_fig(fig, out_root: Path, stem: str) -> None:
    png = out_root / "figures_png" / f"{stem}.png"
    pdf = out_root / "figures_pdf" / f"{stem}.pdf"
    png.parent.mkdir(parents=True, exist_ok=True)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, bbox_inches="tight", dpi=350)
    fig.savefig(pdf, bbox_inches="tight")


def make_figures(ext_eval: pd.DataFrame, overlap_summary: pd.DataFrame, ens: pd.DataFrame, extm: pd.DataFrame, ive: pd.DataFrame, referral: pd.DataFrame, out_root: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

    counts = ext_eval["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(["Normal", "Diabetic Wound"], [counts.get(0, 0), counts.get(1, 0)]); ax.set_ylabel("Images"); ax.set_title("External Evaluation Class Distribution"); save_fig(fig, out_root, "FigP4V6_01_Class_Distribution"); plt.close(fig)

    s = overlap_summary.iloc[0]
    fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(["Exact overlap", "Near candidates", "Retained"], [int(s["exact_primary_overlap"]), int(s["near_primary_candidates_phash_le_4"]), int(s["external_eval_images_after_exact_exclusion"])]); ax.set_ylabel("Images"); ax.set_title("Cross-Dataset Similarity Audit"); save_fig(fig, out_root, "FigP4V6_02_Overlap_Audit"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for mk, g in ens.groupby("model_key"):
        fpr, tpr, _ = roc_curve(g["label"], g["prob_calibrated_mean"]); auc = extm.loc[extm["model_key"] == mk, "roc_auc"].iloc[0]; ax.plot(fpr, tpr, label=f"{MODEL_NAMES[mk]} AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("External ROC Curves"); ax.legend(); save_fig(fig, out_root, "FigP4V6_03_ROC"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for mk, g in ens.groupby("model_key"):
        precision, recall, _ = precision_recall_curve(g["label"], g["prob_calibrated_mean"]); ap = extm.loc[extm["model_key"] == mk, "pr_auc"].iloc[0]; ax.plot(recall, precision, label=f"{MODEL_NAMES[mk]} AP={ap:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("External Precision-Recall Curves"); ax.legend(); save_fig(fig, out_root, "FigP4V6_04_PR"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for mk, g in ens.groupby("model_key"):
        frac, mean = calibration_curve(g["label"], g["prob_calibrated_mean"], n_bins=min(10, max(2, len(g)//20)), strategy="quantile"); ax.plot(mean, frac, marker="o", label=MODEL_NAMES[mk])
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1); ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed positive fraction"); ax.set_title("External Reliability Diagram"); ax.legend(); save_fig(fig, out_root, "FigP4V6_05_Reliability"); plt.close(fig)

    selected = ive[ive["metric"].isin(["accuracy", "sensitivity", "specificity", "roc_auc", "brier", "ece"])].copy()
    if not selected.empty:
        metrics = selected["metric"].drop_duplicates().tolist(); names = extm["model_name"].tolist(); x = np.arange(len(metrics)); width = 0.12
        fig, ax = plt.subplots(figsize=(10, 5))
        for j, name in enumerate(names):
            sub = selected[selected["model_name"] == name].set_index("metric").reindex(metrics)
            ax.bar(x + (j*2)*width, sub["internal_macro_mean"].values, width, label=f"{name} internal")
            ax.bar(x + (j*2+1)*width, sub["external_ensemble"].values, width, label=f"{name} external")
        ax.set_xticks(x + width * max(len(names)-0.5, 0)); ax.set_xticklabels(metrics, rotation=30, ha="right"); ax.set_title("Internal vs External Performance"); ax.legend(ncol=2, fontsize=8); save_fig(fig, out_root, "FigP4V6_06_Internal_vs_External"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for mk, g in referral.groupby("model_key"):
        ax.plot(g["coverage"], g["accuracy"], marker="o", label=MODEL_NAMES[mk])
    ax.set_xlabel("Coverage"); ax.set_ylabel("Accuracy after referral"); ax.set_title("External Risk-Coverage"); ax.legend(); save_fig(fig, out_root, "FigP4V6_07_Risk_Coverage"); plt.close(fig)

    models = list(MODEL_NAMES)
    fig, axes = plt.subplots(1, len(models), figsize=(12, 3.6))
    for ax, mk in zip(axes, models):
        g = ens[ens["model_key"] == mk]; cm = confusion_matrix(g["label"], g["pred_majority"], labels=[0, 1]); ax.imshow(cm)
        for i in range(2):
            for j in range(2): ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(["Normal","DFU"]); ax.set_yticklabels(["Normal","DFU"]); ax.set_title(MODEL_NAMES[mk]); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.suptitle("External Frozen-Decision Confusion Matrices"); save_fig(fig, out_root, "FigP4V6_08_Confusion_Matrices"); plt.close(fig)


def display_results(out_root: Path) -> None:
    from IPython.display import Image as IPImage, Markdown, display
    tables = [
        ("DATASET AUDIT", "01_dataset_audit.csv"),
        ("CROSS-DATASET OVERLAP", "02_cross_dataset_overlap_summary.csv"),
        ("FROZEN CHECKPOINT PROTOCOL", "03_checkpoint_protocol.csv"),
        ("EXTERNAL CHECKPOINT METRICS", "05_external_checkpoint_metrics.csv"),
        ("EXTERNAL ENSEMBLE METRICS", "07_external_ensemble_metrics.csv"),
        ("INTERNAL VS EXTERNAL", "08_internal_vs_external.csv"),
        ("GROUP-CLUSTER BOOTSTRAP CI", "09_external_bootstrap_ci.csv"),
        ("SELECTIVE REFERRAL", "10_external_selective_referral.csv"),
    ]
    for title, name in tables:
        p = out_root / "tables" / name
        if p.exists():
            display(Markdown(f"## {title}")); display(pd.read_csv(p))
    for p in sorted((out_root / "figures_png").glob("*.png")):
        display(Markdown(f"### {p.stem}")); display(IPImage(filename=str(p), width=1000))


def export_final(out_root: Path, verification: dict[str, Any]) -> Path:
    atomic_json(out_root / "PHASE4_EXTERNAL_V6_VERIFICATION.json", verification)
    pkl = out_root / "PHASE4_EXTERNAL_V6_ANALYSIS.pkl"
    with pkl.open("wb") as f:
        pickle.dump(verification, f, protocol=pickle.HIGHEST_PROTOCOL)
    zip_path = out_root / "PHASE4_EXTERNAL_V6_FINAL_EXPORT.zip"
    wanted = []
    for sub in ["tables", "figures_png", "figures_pdf", "checkpoint_predictions", "cache"]:
        root = out_root / sub
        if root.exists(): wanted.extend(p for p in root.rglob("*") if p.is_file())
    wanted += [out_root / "PHASE4_EXTERNAL_V6_VERIFICATION.json", pkl]
    tmp = zip_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in wanted:
            if p.exists() and p != zip_path:
                zf.write(p, arcname=str(p.relative_to(out_root)))
    os.replace(tmp, zip_path)
    return zip_path


def run() -> None:
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    print("=" * 100)
    print(VERSION)
    print("DRIVE-FIRST COLAB | AUTOMATIC KAGGLE FALLBACK | CPU/GPU AUTO | RESUME-SAFE")
    print("NO TRAINING | NO FINE-TUNING | NO EXTERNAL THRESHOLD/CALIBRATION FITTING")
    print("=" * 100, flush=True)

    ensure_packages()
    runtime = detect_and_prepare_runtime()
    out_root = Path(runtime["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    print("Runtime mode:", runtime["mode"], flush=True)
    print("Output root:", out_root, flush=True)

    frozen = resolve_frozen_inputs(runtime)
    inv = verify_frozen_inputs(frozen)
    oof = frozen["oof"].copy()
    fold_metrics = frozen["metrics"].copy()

    protocol_rows = []
    for _, r in inv.iterrows():
        temperature, threshold = frozen_operating_point(oof, str(r["model_key"]), int(r["seed"]), int(r["outer_fold"]))
        protocol_rows.append({
            "model_key": str(r["model_key"]),
            "model_name": MODEL_NAMES[str(r["model_key"])],
            "seed": int(r["seed"]),
            "outer_fold": int(r["outer_fold"]),
            "checkpoint_sha256": str(r["sha256"]),
            "temperature_frozen": temperature,
            "threshold_frozen": threshold,
            "training": False,
            "fine_tuning": False,
            "external_label_fitting": False,
        })
    atomic_csv(out_root / "tables" / "03_checkpoint_protocol.csv", pd.DataFrame(protocol_rows))

    print("Fetching external dataset by URL/API:", EXTERNAL_DATASET_HANDLE, flush=True)
    external_root = download_public_dataset(EXTERNAL_DATASET_HANDLE)
    external = discover_external_dataset(external_root)
    atomic_csv(out_root / "tables" / "01_dataset_audit.csv", pd.DataFrame([{
        "dataset_handle": EXTERNAL_DATASET_HANDLE,
        "selected_images": int(len(external)),
        "positive_images": int((external["label"] == 1).sum()),
        "negative_images": int((external["label"] == 0).sum()),
        "task": "independent diabetic-wound-vs-normal domain-shift stress test",
        "manual_download_required": False,
    }]))
    print(f"External class discovery: PASS n={len(external)} positive={(external['label']==1).sum()} negative={(external['label']==0).sum()}", flush=True)

    print("Fetching primary dataset by URL/API for overlap audit:", PRIMARY_DATASET_HANDLE, flush=True)
    primary_root = download_public_dataset(PRIMARY_DATASET_HANDLE)
    primary_unique = oof[["image_id", "relative_path"]].drop_duplicates("image_id").copy()
    primary_unique["path"] = primary_unique["relative_path"].map(lambda x: str(resolve_primary_image_path(primary_root, x)))

    ext_fp = build_fingerprints(external, out_root / "cache" / "external_fingerprints.csv", "relative_path", workers=12)
    prim_for_fp = primary_unique.rename(columns={"image_id": "primary_key"})
    primary_fp = build_fingerprints(prim_for_fp, out_root / "cache" / "primary_fingerprints.csv", "primary_key", workers=12)
    overlap_summary, overlap_details = audit_cross_dataset_overlap(external, ext_fp, primary_fp, out_root)

    clustered = assign_external_similarity_groups(overlap_details)
    ext_eval = external.merge(clustered[["relative_path", "external_group_id"]], on="relative_path", how="inner", validate="one_to_one")
    if ext_eval.empty or ext_eval["label"].nunique() != 2:
        raise RuntimeError("External evaluation set is empty or lost one class after exact-overlap exclusion")
    print(f"External evaluation set: PASS n={len(ext_eval)} similarity_groups={ext_eval['external_group_id'].nunique()}", flush=True)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Inference device:", device, flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    else:
        print("CPU inference enabled; checkpoint-level resume is active.", flush=True)

    prediction_frames = []
    for i, (_, r) in enumerate(inv.iterrows(), start=1):
        print(f"Checkpoint {i}/{len(inv)}", flush=True)
        prediction_frames.append(run_checkpoint_inference(r, oof, ext_eval, device, out_root))
        atomic_json(out_root / "PHASE4_V6_PROGRESS.json", {
            "status": "RUNNING",
            "version": VERSION,
            "runtime_mode": runtime["mode"],
            "inference_device": str(device),
            "completed_checkpoints": i,
            "total_checkpoints": int(len(inv)),
            "last_checkpoint": f"{r['model_key']}_seed{int(r['seed'])}_fold{int(r['outer_fold'])}",
            "updated_at_unix": time.time(),
        })

    predictions = pd.concat(prediction_frames, ignore_index=True)
    atomic_csv(out_root / "tables" / "04_external_checkpoint_predictions.csv", predictions)
    ckptm = checkpoint_metrics(predictions); atomic_csv(out_root / "tables" / "05_external_checkpoint_metrics.csv", ckptm)
    ens = ensemble_predictions(predictions); atomic_csv(out_root / "tables" / "06_external_ensemble_predictions.csv", ens)
    extm = ensemble_metrics(ens); atomic_csv(out_root / "tables" / "07_external_ensemble_metrics.csv", extm)
    ive = internal_vs_external(fold_metrics, extm); atomic_csv(out_root / "tables" / "08_internal_vs_external.csv", ive)
    boot = cluster_bootstrap_ci(ens, reps=BOOTSTRAP_REPS); atomic_csv(out_root / "tables" / "09_external_bootstrap_ci.csv", boot)
    referral = selective_referral(ens); atomic_csv(out_root / "tables" / "10_external_selective_referral.csv", referral)
    make_figures(ext_eval, overlap_summary, ens, extm, ive, referral, out_root)

    verification = {
        "status": "PASS",
        "version": VERSION,
        "runtime_mode": runtime["mode"],
        "primary_training_performed": False,
        "fine_tuning_performed": False,
        "external_label_fitting_for_threshold_or_temperature": False,
        "verified_checkpoints_used": int(len(inv)),
        "external_dataset_handle": EXTERNAL_DATASET_HANDLE,
        "primary_dataset_handle_for_overlap_audit": PRIMARY_DATASET_HANDLE,
        "external_selected_images": int(len(external)),
        "external_eval_images": int(len(ext_eval)),
        "external_similarity_groups": int(ext_eval["external_group_id"].nunique()),
        "exact_primary_overlap_excluded": int(overlap_summary.iloc[0]["exact_primary_overlap"]),
        "near_primary_candidates_phash_le_4": int(overlap_summary.iloc[0]["near_primary_candidates_phash_le_4"]),
        "bootstrap_reps": BOOTSTRAP_REPS,
        "inference_device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "output_root": str(out_root),
    }
    zip_path = export_final(out_root, verification)
    atomic_json(out_root / "PHASE4_V6_PROGRESS.json", {**verification, "status": "PASS", "completed_checkpoints": int(len(inv))})
    display_results(out_root)

    print("\n" + "=" * 100)
    print("PHASE-4 V6 EXTERNAL VERIFICATION: PASS")
    print("Runtime mode:", runtime["mode"])
    print("Inference device:", device)
    print("Primary training performed: FALSE")
    print("Fine-tuning performed: FALSE")
    print("External threshold/calibration fitting: FALSE")
    print("Final export:", zip_path)
    print("=" * 100)


if __name__ == "__main__":
    run()
