from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from machine_sentinel.features import extract_recording_features, extract_window_features


class FeatureExtractionTests(unittest.TestCase):
    def test_constant_signal_has_zero_dynamic_features(self) -> None:
        frame = pd.DataFrame(
            {
                "ax": np.full(200, 100),
                "ay": np.full(200, -200),
                "az": np.full(200, 4096),
            }
        )

        features = extract_window_features(frame)

        self.assertAlmostEqual(features["rms"], 0.0)
        self.assertAlmostEqual(features["ptp"], 0.0)

    def test_incomplete_window_is_ignored(self) -> None:
        frame = pd.DataFrame(
            {
                "ax": np.arange(450),
                "ay": np.arange(450),
                "az": np.arange(450),
            }
        )

        result = extract_recording_features(
            frame,
            recording_id="sample_1",
            condition="normal_40",
            label=0,
            trim_seconds=0,
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result["window"].tolist(), [0, 1])

    def test_missing_axis_is_rejected(self) -> None:
        frame = pd.DataFrame({"ax": [0, 1], "az": [0, 1]})
        with self.assertRaisesRegex(ValueError, "ay"):
            extract_window_features(frame)


if __name__ == "__main__":
    unittest.main()
