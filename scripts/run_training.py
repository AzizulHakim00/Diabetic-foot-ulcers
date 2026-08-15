"""Run the matched DFU-ImageGuard candidate-screening experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reliable_runner_v2 import ReliableSettingsV2, run_reliable_framework_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train ConvNeXtV2-Tiny, DenseNet121, and MobileNetV3-Large on "
            "matched group-locked outer folds with fold-only calibration."
        )
    )
    parser.add_argument("--run-id", default="DFU_IMAGEGUARD_PAPER")
    parser.add_argument(
        "--drive-root",
        default="/content/drive/MyDrive/DFU-ImageGuard",
        help="Root directory for checkpoints, predictions, tables, and figures.",
    )
    parser.add_argument(
        "--backup-root",
        default="/content/drive/MyDrive/DFU-ImageGuard-Backup",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        choices=[1, 2, 3, 4, 5],
        help="One-based outer-fold numbers.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["convnextv2_tiny", "densenet121", "mobilenetv3_large"],
        choices=["convnextv2_tiny", "densenet121", "mobilenetv3_large"],
    )
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--target-sensitivity", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = ReliableSettingsV2(
        run_id=args.run_id,
        drive_root=args.drive_root,
        backup_root=args.backup_root,
        seeds=tuple(args.seeds),
        folds=tuple(fold - 1 for fold in args.folds),
        models=tuple(args.models),
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_sensitivity=args.target_sensitivity,
        source_commit="paper-review-release",
    )
    result = run_reliable_framework_v2(settings)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
