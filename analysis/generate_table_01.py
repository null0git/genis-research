"""Generate Table 01: full feature list, retained vs. dropped, from the
actual data audit -- never typed by hand.

Reads: results/processed/data_audit_summary.json, configs/data.yaml
Writes: results/processed/table_01_features.csv

Usage:
    python analysis/generate_table_01.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_yaml, get_repo_root  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    root = get_repo_root()
    audit_path = root / "results/processed/data_audit_summary.json"
    if not audit_path.exists():
        raise FileNotFoundError(
            f"{audit_path} not found. Run `python -m src.data.audit` first."
        )
    with open(audit_path) as f:
        audit = json.load(f)

    data_cfg = load_yaml("configs/data.yaml")
    drop_cols = set(data_cfg.get("drop_columns", []))
    categorical_cols = set(data_cfg.get("categorical_columns", []))
    label_cols = {
        data_cfg["column_map"].get("binary_label"),
        data_cfg["column_map"].get("category_label"),
        data_cfg["column_map"].get("subcategory_label"),
        data_cfg["column_map"].get("scenario_id"),
        data_cfg["column_map"].get("start_time"),
    }

    all_columns = audit["files"][0]["columns"] if audit["files"] else []
    rows = []
    for col in all_columns:
        if col in label_cols:
            status = "label/metadata (excluded from feature matrix)"
        elif col in drop_cols:
            status = "dropped (topology-identifying or bookkeeping)"
        elif col in categorical_cols:
            status = "retained (one-hot encoded)"
        else:
            status = "retained (numeric)"
        rows.append({"column": col, "status": status})

    df = pd.DataFrame(rows)
    out_path = root / "results/processed/table_01_features.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d columns).", out_path, len(df))


if __name__ == "__main__":
    main()
