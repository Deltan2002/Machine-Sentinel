"""Build processed feature tables from raw accelerometer recordings."""

from __future__ import annotations

import argparse
from pathlib import Path

from machine_sentinel.datasets import build_challenge_dataset, build_training_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/final"),
        help="Folder containing condition subfolders (default: data/raw/final)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Destination for generated CSV files (default: data/processed)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features = build_training_dataset(args.raw_dir)
    challenge = build_challenge_dataset(args.raw_dir)
    features.to_csv(args.output_dir / "features.csv", index=False)
    challenge.to_csv(args.output_dir / "challenge.csv", index=False)

    print(f"Wrote {len(features)} training rows to {args.output_dir / 'features.csv'}")
    print(f"Wrote {len(challenge)} challenge rows to {args.output_dir / 'challenge.csv'}")


if __name__ == "__main__":
    main()
