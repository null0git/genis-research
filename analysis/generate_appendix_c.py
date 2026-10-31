"""Generate Appendix C: full disaggregated per-fold/per-seed results for
every model in every experiment, unrounded, unsummarized.

Reads: results/processed/e0{1..7}_metrics.json (any that exist)
Writes: results/processed/appendix_c_full_results.csv

Usage:
    python analysis/generate_appendix_c.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import get_repo_root  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

EXPERIMENT_PREFIXES = ["e01", "e02", "e03", "e04", "e06", "e07"]  # e05 has a nested structure, handled separately


def main() -> None:
    root = get_repo_root()
    rows = []

    for prefix in EXPERIMENT_PREFIXES:
        path = root / f"results/processed/{prefix}_metrics.json"
        if not path.exists():
            logger.warning("%s not found; skipping (run the experiment first if this is unexpected).", path)
            continue
        with open(path) as f:
            data = json.load(f)

        if "models" not in data:
            continue

        for model_name, model_result in data["models"].items():
            for run in model_result.get("per_run_metrics", []):
                row = {
                    "experiment": prefix,
                    "model": model_name,
                    "seed_or_fold": run.get("seed", run.get("held_out_scenario", "N/A")),
                    "accuracy": run.get("accuracy"),
                    "macro_f1": run.get("macro_f1"),
                    "weighted_f1": run.get("weighted_f1"),
                    "macro_precision": run.get("macro_precision"),
                    "macro_recall": run.get("macro_recall"),
                    "false_positive_rate": run.get("false_positive_rate"),
                    "false_negative_rate": run.get("false_negative_rate"),
                    "n_samples": run.get("n_samples"),
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    out_path = root / "results/processed/appendix_c_full_results.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows across %d experiment file(s) found).", out_path, len(df), len(EXPERIMENT_PREFIXES))


if __name__ == "__main__":
    main()
