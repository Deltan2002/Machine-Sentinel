"""Print descriptive statistics for a processed feature table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from machine_sentinel.features import FEATURE_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "features",
        nargs="?",
        type=Path,
        default=Path("data/processed/features.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    summary = features.groupby("condition")[list(FEATURE_COLUMNS)].agg(["mean", "std"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
