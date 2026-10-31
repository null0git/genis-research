"""Generate Table 05: per-class recall/precision broken down by held-out
scenario, for the error analysis (Section 16 of the manuscript).

Reads: results/processed/{source}_metrics.json (any LOSO experiment, e.g. e02 or e04)
Writes: results/processed/table_05_per_scenario_breakdown_{source}.csv

Usage:
    python analysis/generate_table_05.py --source e02
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import get_repo_root  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="e02", help="Experiment prefix to read (must be a LOSO experiment).")
    args = parser.parse_args()

    root = get_repo_root()
    path = root / f"results/processed/{args.source}_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run experiments/{args.source}/run.py first.")
    with open(path) as f:
        data = json.load(f)

    if data.get("split_strategy") != "loso":
        raise ValueError(
            f"{args.source}_metrics.json has split_strategy='{data.get('split_strategy')}', "
            f"not 'loso'. Table 05 requires a LOSO experiment's per-fold, "
            f"per-scenario breakdown."
        )

    rows = []
    for model_name, model_result in data["models"].items():
        for run in model_result["per_run_metrics"]:
            held_out = run.get("held_out_scenario", "Not reported")
            for class_label, class_metrics in run["per_class"].items():
                rows.append(
                    {
                        "model": model_name,
                        "held_out_scenario": held_out,
                        "class": class_label,
                        "precision": class_metrics["precision"],
                        "recall": class_metrics["recall"],
                        "f1": class_metrics["f1"],
                        "support": class_metrics["support"],
                    }
                )

    df = pd.DataFrame(rows)
    out_path = root / f"results/processed/table_05_per_scenario_breakdown_{args.source}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows).", out_path, len(df))


if __name__ == "__main__":
    main()
