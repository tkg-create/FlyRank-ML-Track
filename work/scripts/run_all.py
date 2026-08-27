"""Run the full w07 pipeline: load/score, then build the ranked queue.

Only 01 and 02 run here — those are the two steps that produce the deployable queue.
03_validate_combined_score.py and 04_check_fold_representation.py are validation/audit scripts: they consume this pipeline's output and check it, they don't feed anything back into what gets built. 
Run them separately, after this, when you want to (re)validate rather than every time you rebuild the queue.

Prompts for the HF token once (getpass, not hardcoded), sets it in the environment so every subprocess step inherits it, then runs each step in order.

Usage (from a notebook):

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
    ("01_load_and_score.py", "Load — warehouse features, baseline rule, out-of-fold RF scoring, calibration"),
    ("02_build_queue.py", "Build queue — archetypes, confidence, combined score (calibrated), exports"),
]


def run_step(index: int, script: str, label: str) -> None:
    print(f"\n{'=' * 70}\n▶ Step {index}/{len(STEPS)} — {label}\n{'=' * 70}", flush=True)
    # "-u" forces the child process's stdout/stderr to be unbuffered. 
    # Without it, Python fully buffers output when stdout isn't a real terminal (true for any subprocess),
    # so print() statements inside 01/02 would queue up invisibly and only appear in a burst at the end instead of streaming.
    subprocess.run([sys.executable, "-u", str(SCRIPTS_DIR / script)], cwd=ROOT, check=True)


def main() -> None:
    # Prompt once here so both steps inherit the same token via the environment — avoids prompting twice,
    # and avoids the unreliable stdin behavior of prompting for a secret inside a subprocess launched via Colab's `!` shell magic.
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
        print(
            "\nQueue built. Run 03_validate_combined_score.py and 04_check_fold_representation.py separately to (re)validate it — not run here, see this script's docstring."
        )


if __name__ == "__main__":
    main()
