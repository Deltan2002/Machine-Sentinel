"""Train, evaluate, and export the two-feature edge classifier."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from machine_sentinel.modeling import (
    EDGE_FEATURES,
    EXPERIMENTS,
    deployment_constants,
    evaluate_classifier,
    split_by_recording,
    train_classifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--test-suffix", default="_3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    train, test = split_by_recording(features, args.test_suffix)

    results: dict[str, object] = {}
    for name, columns in EXPERIMENTS.items():
        model = train_classifier(train, columns)
        results[name] = evaluate_classifier(model, test, columns)

    edge_model = train_classifier(train, EDGE_FEATURES)
    normal_rms = train.loc[train["condition"] == "normal_40", "rms"]
    constants = deployment_constants(edge_model, EDGE_FEATURES)
    constants["idle_threshold"] = float(0.5 * normal_rms.min())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "edge_model.json").write_text(
        json.dumps(constants, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "edge_model.pkl").open("wb") as model_file:
        pickle.dump(edge_model, model_file)

    print(json.dumps(results, indent=2))
    print(f"Exported model artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
