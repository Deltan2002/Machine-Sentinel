"""Pure vibration feature extraction with no plotting or file-system side effects."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

SAMPLE_RATE_HZ = 200
WINDOW_SIZE = 200
COUNTS_PER_G = 4096.0
ACCEL_COLUMNS = ("ax", "ay", "az")
FEATURE_COLUMNS = (
    "rms",
    "ptp",
    "dominant_freq",
    "energy_low",
    "energy_mid",
    "energy_high",
)


def _validate_acceleration_frame(frame: pd.DataFrame) -> None:
    missing = set(ACCEL_COLUMNS).difference(frame.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Missing accelerometer columns: {names}")


def band_power(
    frequencies: np.ndarray,
    power: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    """Return total spectral power in the half-open interval [low_hz, high_hz)."""
    mask = (frequencies >= low_hz) & (frequencies < high_hz)
    return float(np.sum(power[mask]))


def extract_window_features(
    window: pd.DataFrame,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    counts_per_g: float = COUNTS_PER_G,
) -> dict[str, float]:
    """Extract time- and frequency-domain features from one sample window."""
    _validate_acceleration_frame(window)
    if len(window) < 2:
        raise ValueError("A feature window must contain at least two samples")
    if sample_rate_hz <= 0 or counts_per_g <= 0:
        raise ValueError("sample_rate_hz and counts_per_g must be positive")

    acceleration = window.loc[:, ACCEL_COLUMNS].to_numpy(dtype=float) / counts_per_g
    centered = acceleration - acceleration.mean(axis=0)
    dynamic_magnitude = np.linalg.norm(centered, axis=1)

    rms = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    ptp = float(np.ptp(dynamic_magnitude))

    fft = np.fft.rfft(centered, axis=0)
    frequencies = np.fft.rfftfreq(len(centered), d=1 / sample_rate_hz)
    power = np.sum(np.abs(fft) ** 2, axis=1)
    dominant_index = int(np.argmax(power[1:]) + 1)

    return {
        "rms": rms,
        "ptp": ptp,
        "dominant_freq": float(frequencies[dominant_index]),
        "energy_low": band_power(frequencies, power, 1, 30),
        "energy_mid": band_power(frequencies, power, 30, 50),
        "energy_high": band_power(frequencies, power, 50, 101),
    }


def extract_recording_features(
    recording: pd.DataFrame,
    *,
    recording_id: str,
    condition: str,
    label: int,
    trim_seconds: float = 1,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    window_size: int = WINDOW_SIZE,
) -> pd.DataFrame:
    """Convert a recording into one feature row per complete window."""
    _validate_acceleration_frame(recording)
    if trim_seconds < 0:
        raise ValueError("trim_seconds cannot be negative")
    if window_size < 2:
        raise ValueError("window_size must be at least two")

    trim_samples = int(trim_seconds * sample_rate_hz)
    prepared = recording
    if trim_samples and len(recording) > 2 * trim_samples:
        prepared = recording.iloc[trim_samples:-trim_samples].reset_index(drop=True)

    rows: list[Mapping[str, object]] = []
    for start in range(0, len(prepared) - window_size + 1, window_size):
        features = extract_window_features(
            prepared.iloc[start : start + window_size],
            sample_rate_hz=sample_rate_hz,
        )
        rows.append(
            {
                "recording_id": recording_id,
                "condition": condition,
                "label": label,
                "window": start // window_size,
                "start_time_s": start / sample_rate_hz,
                **features,
            }
        )

    columns = [
        "recording_id",
        "condition",
        "label",
        "window",
        "start_time_s",
        *FEATURE_COLUMNS,
    ]
    return pd.DataFrame(rows, columns=columns)
