"""Generate Table 04: split strategy x model grid of macro-F1 mean +/- SD.

Reads: results/processed/e01_metrics.json, e02, e03, e04
Writes: results/processed/table_04_split_x_model_grid.csv

Usage:
    python analysis/generate_table_04.py
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

SOURCES = {
    "random_split": ["e01", "e03"],
    "loso": ["e02", "e04"],
}


def _load(root: Path, prefix: str) -> dict:
    path = root / f"results/processed/{prefix}_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run experiments/{prefix}/run.py first.")
    with open(path) as f:
        return json.load(f)


def main() -> None:
    root = get_repo_root()
    rows = []

    for strategy, prefixes in SOURCES.items():
        for prefix in prefixes:
            data = _load(root, prefix)
            for model_name, model_result in data["models"].items():
                agg = model_result["aggregate"]
                rows.append(
                    {
                        "split_strategy": strategy,
                        "model": model_name,
                        "macro_f1_mean": agg.get("macro_f1_mean"),
                        "macro_f1_std": agg.get("macro_f1_std"),
                        "n_runs": agg.get("n_runs"),
                        "source_experiment": prefix,
                    }
                )

    df = pd.DataFrame(rows).sort_values(["split_strategy", "model"])
    out_path = root / "results/processed/table_04_split_x_model_grid.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows).", out_path, len(df))


if __name__ == "__main__":
    main()
