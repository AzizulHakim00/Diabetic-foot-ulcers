"""Paired Wilcoxon tests and Holm correction used for model comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SELECTED_MODEL = "densenet121"
COMPARATORS = ("convnextv2_tiny", "mobilenetv3_large")
METRICS = ("accuracy", "balanced_accuracy", "f1", "sensitivity", "mcc")
PAIR_KEYS = ("outer_fold", "seed")


def rank_biserial(differences: np.ndarray) -> float:
    from scipy.stats import rankdata

    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences) & (differences != 0)]
    if differences.size == 0:
        return 0.0
    ranks = rankdata(np.abs(differences), method="average")
    positive = float(ranks[differences > 0].sum())
    negative = float(ranks[differences < 0].sum())
    return (positive - negative) / (positive + negative)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    total = len(p_values)
    for position, index in enumerate(order):
        candidate = min(1.0, (total - position) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    required = {"model_key", *PAIR_KEYS, *METRICS}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows: list[dict[str, float | int | str | bool]] = []
    for comparator in COMPARATORS:
        selected = metrics.loc[metrics.model_key == SELECTED_MODEL]
        baseline = metrics.loc[metrics.model_key == comparator]
        paired = selected.merge(
            baseline,
            on=list(PAIR_KEYS),
            how="inner",
            suffixes=("_selected", "_comparator"),
            validate="one_to_one",
        ).sort_values(list(PAIR_KEYS))
        if paired.empty:
            raise ValueError(f"No matched evaluations for {SELECTED_MODEL} and {comparator}")

        for metric in METRICS:
            selected_values = paired[f"{metric}_selected"].to_numpy(float)
            comparator_values = paired[f"{metric}_comparator"].to_numpy(float)
            differences = selected_values - comparator_values
            if np.allclose(differences, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                test = wilcoxon(
                    selected_values,
                    comparator_values,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                statistic, p_value = float(test.statistic), float(test.pvalue)
            rows.append(
                {
                    "selected_model": SELECTED_MODEL,
                    "comparator": comparator,
                    "metric": metric,
                    "matched_evaluations": int(len(paired)),
                    "selected_mean": float(selected_values.mean()),
                    "comparator_mean": float(comparator_values.mean()),
                    "mean_difference": float(differences.mean()),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "rank_biserial": rank_biserial(differences),
                }
            )

    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result.p_value.to_numpy(float))
    result["significant_holm_0_05"] = result.p_holm < 0.05
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paired_wilcoxon_holm.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = paired_tests(pd.read_csv(args.metrics_csv))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
