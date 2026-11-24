"""Regression tests for v6 diagnostic parity, not a simulation test."""
import math
import unittest

from paper_b.experiments.run_v6_pilot import parity_payload
from paper_b.experiments.run_matched_pilot import canonical_hash


class V6ParityTests(unittest.TestCase):
    def test_matching_nested_nan(self):
        a = [{"metric": float("nan"), "mse": 1.25}]
        b = [{"mse": 1.25, "metric": float("nan")}]
        self.assertEqual(parity_payload(a), parity_payload(b))

    def test_nan_is_not_null_or_finite(self):
        for replacement in (None, 0.0, "NaN"):
            self.assertNotEqual(parity_payload({"x": math.nan}),
                                parity_payload({"x": replacement}))

    def test_numeric_change_still_detected(self):
        self.assertNotEqual(parity_payload([math.nan, 1.25]),
                            parity_payload([math.nan, 1.26]))

    def test_infinity_rejected(self):
        for value in (math.inf, -math.inf):
            with self.assertRaises(ValueError):
                parity_payload({"x": value})

    def test_design_hash_remains_strict(self):
        with self.assertRaises(ValueError):
            canonical_hash({"x": math.nan})


if __name__ == "__main__":
    unittest.main()
