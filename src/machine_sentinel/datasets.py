"""Dataset loading and preparation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .features import extract_recording_features

TRAINING_CONDITIONS = {
    "normal_40": 0,
    "loud_40": 1,
    "freq_60": 1,
}

CHALLENGE_CONDITIONS = ("stationary", "tapping")


def extract_folder(
    folder: Path,
    *,
    condition: str,
    label: int,
    trim_seconds: float,
) -> pd.DataFrame:
    """Extract features from every CSV recording in a folder."""
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV recordings found in {folder}")

    tables = []
    for path in files:
        recording = pd.read_csv(path)
        tables.append(
            extract_recording_features(
                recording,
                recording_id=path.stem,
                condition=condition,
                label=label,
                trim_seconds=trim_seconds,
            )
        )
    return pd.concat(tables, ignore_index=True)


def build_training_dataset(raw_final_dir: Path) -> pd.DataFrame:
    """Build the labeled training table from the configured conditions."""
    tables = [
        extract_folder(
            raw_final_dir / condition,
            condition=condition,
            label=label,
            trim_seconds=1,
        )
        for condition, label in TRAINING_CONDITIONS.items()
    ]
    return pd.concat(tables, ignore_index=True)


def build_challenge_dataset(raw_final_dir: Path) -> pd.DataFrame:
    """Build the unlabeled stationary/tapping validation table."""
    tables = [
        extract_folder(
            raw_final_dir / condition,
            condition=condition,
            label=-1,
            trim_seconds=0,
        )
        for condition in CHALLENGE_CONDITIONS
    ]
    return pd.concat(tables, ignore_index=True)
