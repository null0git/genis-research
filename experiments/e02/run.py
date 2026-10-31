"""Run experiment E02. See configs/experiments/e02_*.yaml for the exact
models, split strategy, and hyperparameters this uses.

Usage:
    python experiments/e02/run.py --config configs/experiments/e02_*.yaml

Reproducibility instructions (see README.md for the full protocol):
    1. Run this script, redirecting stdout to results/raw/e02_terminal_output.txt
    2. The script itself also writes results/processed/e02_metrics.json
    3. Do not hand-edit either output file -- regenerate by re-running this script.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.training.train import run_experiment  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def _resolve_config(pattern: str) -> str:
    if Path(pattern).exists():
        return pattern
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No config matched: {pattern}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/e02_*.yaml")
    args = parser.parse_args()

    config_path = _resolve_config(args.config)
    logger.info("Running E02 with config: %s", config_path)
    results = run_experiment(config_path)

    for model_name, model_results in results["models"].items():
        agg = model_results["aggregate"]
        logger.info(
            "%s | macro_f1 = %.4f +/- %.4f (n=%d)",
            model_name,
            agg.get("macro_f1_mean", float("nan")),
            agg.get("macro_f1_std", 0.0),
            agg.get("n_runs", 0),
        )


if __name__ == "__main__":
    main()
