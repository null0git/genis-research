"""Generate Table 06: deployment cost (RQ3) -- training time, inference
latency, model size, peak memory, per model.

Reads: results/processed/e07_metrics.json
Writes: results/processed/table_06_deployment_cost.csv

Usage:
    python analysis/generate_table_06.py
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


def main() -> None:
    root = get_repo_root()
    path = root / "results/processed/e07_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run experiments/e07/run.py first.")
    with open(path) as f:
        e07 = json.load(f)

    rows = []
    for model_name, metrics in e07["models"].items():
        rows.append(
            {
                "model": model_name,
                "training_time_seconds": metrics["training_time_seconds"],
                "inference_latency_ms_mean": metrics["inference_latency_ms_mean"],
                "inference_latency_ms_std": metrics["inference_latency_ms_std"],
                "model_size_kb": metrics["model_size_bytes"] / 1024,
                "peak_memory_mb_process_cumulative": metrics["peak_memory_mb_process_cumulative"],
            }
        )

    df = pd.DataFrame(rows).sort_values("inference_latency_ms_mean")
    out_path = root / "results/processed/table_06_deployment_cost.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)
    logger.warning(
        "peak_memory_mb_process_cumulative is process-wide cumulative, not "
        "isolated per model -- see experiments/e07/run.py's closing warning."
    )


if __name__ == "__main__":
    main()
