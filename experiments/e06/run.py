"""Run experiment E06: within-scenario temporal-split robustness check.

Runs all 5 models under the chronological (temporal) split. Flow-independent
models (rf, xgboost, mlp) run with use_sequences=False; sequence-aware
models (cnn1d, gru) run with use_sequences=True at the fixed sequence
length used elsewhere in the project (10), read from the config. Each
model is run as its own one-model experiment so a failure in one model
does not block the others, and results are merged into a single
e06_metrics.json.

Usage:
    python experiments/e06/run.py --config configs/experiments/e06_temporal_split_robustness.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data import preprocessing  # noqa: E402
from src.data.audit import find_interval_files  # noqa: E402
from src.training.train import run_experiment_from_dataframe, SEQUENCE_AWARE_FAMILIES  # noqa: E402
from src.utils.config import load_experiment_config, get_repo_root  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/e06_temporal_split_robustness.yaml")
    args = parser.parse_args()

    base_cfg = load_experiment_config(args.config)
    data_cfg = base_cfg["data_config"]

    preprocessed_dir = get_repo_root() / data_cfg["preprocessed_data_dir"]
    interval_files = find_interval_files(preprocessed_dir, data_cfg["flow_interval_seconds"])
    df = preprocessing.load_flows(preprocessed_dir, interval_files)

    output_dir = get_repo_root() / "results/processed"
    merged = {"experiment_id": "E06", "split_strategy": "temporal", "models": {}}

    for model_cfg in base_cfg["models"]:
        model_name = model_cfg["model_name"]
        sub_cfg = copy.deepcopy(base_cfg)
        sub_cfg["models"] = [model_cfg]
        sub_cfg["use_sequences"] = model_name in SEQUENCE_AWARE_FAMILIES
        sub_cfg["output_prefix"] = f"e06_{model_name}"

        logger.info("Running E06 for model=%s (use_sequences=%s)", model_name, sub_cfg["use_sequences"])
        result = run_experiment_from_dataframe(df, sub_cfg, output_dir, base_cfg.get("label_column", "binary_label"))
        merged["models"].update(result["models"])

    merged_path = output_dir / "e06_metrics.json"
    with open(merged_path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2, default=str)
    logger.info("Wrote combined E06 results to %s", merged_path)


if __name__ == "__main__":
    main()
