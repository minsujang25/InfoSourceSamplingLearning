"""Smoke-test the Paper B wrapper against the validated Mesa 3.5 model."""

from __future__ import annotations

import json
import numpy as np

from model.InfoSourceSamplingLearning import InfoSampleModel
from paper_b.metrics import belief_summary, structural_source_counts
from paper_b.validation import assert_finite_state
from scripts.smoke_test import build_smoke_config


def main():
    np.random.seed(12345)
    model = InfoSampleModel(model_attribute=build_smoke_config(), rng=12345)

    for _ in range(3):
        model.step()
        assert_finite_state(model, context="paper-b smoke")

    payload = {
        "steps": int(model.steps),
        "period": int(model.period),
        "belief_summary": belief_summary(model),
        "structural_source_counts": structural_source_counts(model),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
