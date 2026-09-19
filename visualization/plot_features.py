"""Generate feature-space plots from processed data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "features",
        nargs="?",
        type=Path,
        default=Path("data/processed/features.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/figures"))
    return parser.parse_args()


def _scatter_by_condition(
    features: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    destination: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    for condition, subset in features.groupby("condition", sort=True):
        axis.scatter(subset[x], subset[y], label=condition, alpha=0.7)
    axis.set(xlabel=x.replace("_", " ").title(), ylabel=y.replace("_", " ").title())
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plots = (
        ("dominant_freq", "rms", "Vibration Conditions", "frequency_vs_rms.png"),
        ("energy_mid", "energy_high", "Frequency Band Energy", "band_energy.png"),
    )
    for x, y, title, filename in plots:
        destination = args.output_dir / filename
        _scatter_by_condition(
            features,
            x=x,
            y=y,
            title=title,
            destination=destination,
        )
        print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
