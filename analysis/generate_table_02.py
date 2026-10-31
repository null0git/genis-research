"""Generate Table 02: model hyperparameters, read directly from
configs/models/*.yaml so this can never drift from what was actually run.

Writes: results/processed/table_02_hyperparameters.csv

Usage:
    python analysis/generate_table_02.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_yaml, get_repo_root  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    root = get_repo_root()
    config_paths = sorted(glob.glob(str(root / "configs/models/*.yaml")))

    rows = []
    for path in config_paths:
        cfg = load_yaml(path)
        for hp_name, hp_value in cfg["hyperparameters"].items():
            rows.append(
                {
                    "model_name": cfg["model_name"],
                    "model_family": cfg["model_family"],
                    "framework": cfg["framework"],
                    "hyperparameter": hp_name,
                    "value": hp_value,
                }
            )

    df = pd.DataFrame(rows)
    out_path = root / "results/processed/table_02_hyperparameters.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows across %d models).", out_path, len(df), df["model_name"].nunique())


if __name__ == "__main__":
    main()
