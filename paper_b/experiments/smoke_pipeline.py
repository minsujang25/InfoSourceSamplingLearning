"""Smoke-test the theory-aligned Paper B wrapper."""

from __future__ import annotations

import json

from model.InfoSourceSamplingLearning import InfoSampleModel
from paper_b.metrics import (
    realized_flow_composition,
    structural_source_counts,
    theory_metrics,
)
from paper_b.validation import assert_finite_state
from scripts.smoke_test import build_smoke_config


def main():
    config = build_smoke_config()
    config["reliance_mode"] = "adaptive"

    model = InfoSampleModel(model_attribute=config, rng=12345)

    for _ in range(3):
        model.step()
        assert_finite_state(model, context="paper-b smoke")

    payload = {
        "steps": int(model.steps),
        "period": int(model.period),
        "theory_metrics": theory_metrics(model),
        "realized_flow_composition": realized_flow_composition(model),
        "structural_source_counts": structural_source_counts(model),
        "reliance_log_periods": len(model.reliance_history),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
