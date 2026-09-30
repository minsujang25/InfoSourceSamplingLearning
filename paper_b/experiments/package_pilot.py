"""Create a compact shareable bundle from a completed Paper B pilot.

The resumable per-block shards remain on UCloud.  This bundle contains the
consolidated scientific outputs needed for result audit and manuscript analysis.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path


BUNDLE_FILES = (
    "pilot_manifest.json",
    "block_manifest.csv",
    "runs.csv",
    "jammer_contrasts.csv",
    "adaptive_frozen_contrasts.csv",
    "terminal_beliefs.csv.gz",
    "lambda_checkpoints.csv",
    "belief_checkpoints.csv",
    "jammer_strategy_trajectory.csv.gz",
    "reliance_checkpoints.csv.gz",
    "pilot_summary.json",
    "pilot_gate.json",
)


def package_pilot(run_root: str | Path) -> Path:
    run_root = Path(run_root)
    manifest = json.loads(
        (run_root / "pilot_manifest.json").read_text(encoding="utf-8")
    )
    design_id = manifest["design_id"]
    zip_path = run_root.parent / f"paper_b_pilot_{design_id}_shareable.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in BUNDLE_FILES:
            path = run_root / name
            if path.exists():
                archive.write(path, arcname=name)

    return zip_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pilot_dir")
    args = parser.parse_args()
    print(package_pilot(args.pilot_dir))
