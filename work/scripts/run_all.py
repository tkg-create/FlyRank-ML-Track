"""Run the full w07 pipeline: load/score, then build the ranked queue.

Mirrors scripts/run_all.py's structure — prompts for the HF token once
(getpass, never hardcoded), sets it in the environment so every subprocess
step inherits it, then runs each step in order.

Usage (from a notebook, matching notebooks/01_first_look_and_discovery.ipynb's
pattern of shelling out to the reference pipeline):

    import sys
    !{sys.executable} work/scripts/run_all.py

Or directly from a terminal:

    python work/scripts/run_all.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w07_pipeline_utils import OUTPUT_DIR, get_hf_token, read_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "work" / "scripts"

STEPS = [
    ("01_load_and_score.py", "Load — warehouse features, baseline rule, out-of-fold RF scoring"),
    ("02_build_queue.py", "Build queue — archetypes, confidence, combined score, exports"),
]


def run_step(index: int, script: str, label: str) -> None:
    print(f"\n{'=' * 70}\n▶ Step {index}/{len(STEPS)} — {label}\n{'=' * 70}", flush=True)
    subprocess.run([sys.executable, str(SCRIPTS_DIR / script)], cwd=ROOT, check=True)


def main() -> None:
    # Prompt once here so both steps inherit the same token via the
    # environment — avoids prompting twice, and avoids the unreliable
    # stdin behavior of prompting for a secret *inside* a subprocess
    # launched via Colab's `!` shell magic.
    os.environ["HF_TOKEN"] = get_hf_token()

    for index, (script, label) in enumerate(STEPS, start=1):
        run_step(index, script, label)

    metrics_path = OUTPUT_DIR / "w07_metrics.json"
    if metrics_path.exists():
        metrics = read_json(metrics_path)
        print("\nPipeline complete")
        print(f"Population size: {metrics['population_size']:,}")
        print(f"Archetype counts: {metrics['archetype_counts']}")
        print(f"Metrics: {metrics_path}")
        print(f"Report: {OUTPUT_DIR / 'w07_report.md'}")
        print(f"Charts: {OUTPUT_DIR / 'charts'}")


if __name__ == "__main__":
    main()
