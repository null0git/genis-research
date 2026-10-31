"""Run experiment E05: sequence-length ablation.

For each sequence length in configs/experiments/e05_*.yaml's
sequence_lengths list, and each split strategy in splits_to_run, this
builds a one-off experiment config (reusing e03's random-split settings and
e04's LOSO settings as templates) and runs it via
src.training.train.run_experiment_from_dataframe, so all results land in a
single results/processed/e05_metrics.json rather than being scattered
across per-length files.

Usage:
    python experiments/e05/run.py --config configs/experiments/e05_sequence_length_ablation.yaml
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
from src.training.train import run_experiment_from_dataframe  # noqa: E402
from src.utils.config import load_experiment_config, load_yaml, get_repo_root  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/e05_sequence_length_ablation.yaml")
    args = parser.parse_args()

    base_cfg = load_experiment_config(args.config)
    data_cfg = base_cfg["data_config"]

    preprocessed_dir = get_repo_root() / data_cfg["preprocessed_data_dir"]
    interval_files = find_interval_files(preprocessed_dir, data_cfg["flow_interval_seconds"])
    df = preprocessing.load_flows(preprocessed_dir, interval_files)

    all_results = {}
    output_dir = get_repo_root() / "results/processed"

    for strategy in base_cfg["splits_to_run"]:
        for seq_len in base_cfg["sequence_lengths"]:
            sub_cfg = copy.deepcopy(base_cfg)
            sub_cfg["sequence_length"] = seq_len
            sub_cfg["use_sequences"] = True
            sub_cfg["output_prefix"] = f"e05_{strategy}_seqlen{seq_len}"

            if strategy == "random":
                sub_cfg["split"] = {
                    "strategy": "random",
                    "train_frac": 0.70,
                    "val_frac": 0.15,
                    "test_frac": 0.15,
                    "seeds": [42, 43, 44, 45, 46],
                }
            elif strategy == "loso":
                sub_cfg["split"] = {"strategy": "loso"}
            else:
                raise ValueError(f"Unrecognized split strategy in splits_to_run: {strategy}")

            logger.info("Running E05 sub-experiment: strategy=%s, sequence_length=%d", strategy, seq_len)
            result = run_experiment_from_dataframe(df, sub_cfg, output_dir, base_cfg.get("label_column", "binary_label"))
            all_results[f"{strategy}_seqlen{seq_len}"] = result

    combined_path = output_dir / "e05_metrics.json"
    with open(combined_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    logger.info("Wrote combined E05 results to %s", combined_path)


if __name__ == "__main__":
    main()
