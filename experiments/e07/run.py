"""Run experiment E07: deployment/computational cost (RQ3).

Trains each of the 5 models once on a single random split (seed fixed via
the config), then measures:
  - training_time_seconds (wall clock)
  - inference_latency_ms (mean/std per single-flow prediction, batch size 1)
  - model_size_bytes (serialized size on disk)
  - peak_memory_mb (via resource.getrusage; Linux/macOS only -- this is
    noted explicitly since ru_maxrss is not portable to Windows, where this
    script will raise rather than silently report a wrong number)

Usage:
    python experiments/e07/run.py --config configs/experiments/e07_deployment_cost.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data import preprocessing, splits as split_mod  # noqa: E402
from src.data.audit import find_interval_files  # noqa: E402
from src.training.train import _build_and_fit_model, _prepare_split_arrays, SEQUENCE_AWARE_FAMILIES  # noqa: E402
from src.utils.config import load_experiment_config, get_repo_root  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seeding import set_all_seeds  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

logger = get_logger(__name__)


def _peak_memory_mb() -> float:
    if platform.system() not in ("Linux", "Darwin"):
        raise RuntimeError(
            "Peak-memory measurement via resource.getrusage is only supported "
            "on Linux/macOS in this script. On Windows, report this metric as "
            "'Not measured on this platform' rather than a fabricated number."
        )
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KB on Linux, bytes on macOS -- normalize to MB.
    if platform.system() == "Darwin":
        return peak_kb / (1024 * 1024)
    return peak_kb / 1024


def _model_size_bytes(model: Any, model_name: str, tmp_path: Path) -> int:
    if model_name in SEQUENCE_AWARE_FAMILIES or model_name == "mlp":
        import torch

        torch.save(model.model.state_dict(), tmp_path)
    else:
        with open(tmp_path, "wb") as f:
            pickle.dump(model, f)
    size = tmp_path.stat().st_size
    tmp_path.unlink()
    return size


def _measure_sklearn_latency(model: Any, X: np.ndarray, n_repeats: int = 100) -> Dict[str, float]:
    sample = X[:1]
    _ = model.predict(sample)  # warm-up
    timings = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        _ = model.predict(sample)
        timings.append((time.perf_counter() - start) * 1000.0)
    return {"mean_latency_ms": float(np.mean(timings)), "std_latency_ms": float(np.std(timings))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/e07_deployment_cost.yaml")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    data_cfg = cfg["data_config"]

    preprocessed_dir = get_repo_root() / data_cfg["preprocessed_data_dir"]
    interval_files = find_interval_files(preprocessed_dir, data_cfg["flow_interval_seconds"])
    df = preprocessing.load_flows(preprocessed_dir, interval_files)
    df = preprocessing.derive_scenario_column(df, data_cfg)
    X, y, scenario_id, start_time = preprocessing.clean_and_encode(df, data_cfg, cfg.get("label_column", "binary_label"))

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.values)
    n_classes = len(label_encoder.classes_)

    seed = cfg["split"]["seeds"][0]
    set_all_seeds(seed)
    train_idx, val_idx, test_idx = split_mod.random_stratified_split(
        pd.Series(y_encoded),
        cfg["split"]["train_frac"],
        cfg["split"]["val_frac"],
        cfg["split"]["test_frac"],
        seed,
    )
    class_weights = preprocessing.compute_class_weights(pd.Series(y_encoded[train_idx]))

    results: Dict[str, Any] = {"experiment_id": "E07", "models": {}}
    tmp_dir = get_repo_root() / "results" / "raw"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for model_cfg in cfg["models"]:
        model_name = model_cfg["model_name"]
        use_sequences = model_name in SEQUENCE_AWARE_FAMILIES
        seq_len = cfg.get("sequence_length") if use_sequences else None

        arrays = _prepare_split_arrays(
            X, y_encoded, scenario_id, start_time, train_idx, val_idx, test_idx, use_sequences, seq_len
        )
        model, train_time = _build_and_fit_model(
            model_cfg, *arrays["train"], *arrays["val"], n_classes, class_weights, use_sequences
        )

        X_test, _y_test = arrays["test"]
        if hasattr(model, "measure_inference_latency"):
            latency = model.measure_inference_latency(X_test, batch_size=cfg["inference_batch_size"])
        else:
            latency = _measure_sklearn_latency(model, X_test)

        size_bytes = _model_size_bytes(model, model_name, tmp_dir / f"_tmp_{model_name}.bin")
        peak_mem = _peak_memory_mb()

        results["models"][model_name] = {
            "training_time_seconds": train_time,
            "inference_latency_ms_mean": latency["mean_latency_ms"],
            "inference_latency_ms_std": latency["std_latency_ms"],
            "model_size_bytes": size_bytes,
            "peak_memory_mb_process_cumulative": peak_mem,
        }
        logger.info(
            "%s | train=%.2fs | inference=%.3fms | size=%d bytes | peak_mem=%.1f MB",
            model_name, train_time, latency["mean_latency_ms"], size_bytes, peak_mem,
        )

    out_path = get_repo_root() / "results/processed/e07_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Wrote E07 results to %s", out_path)
    logger.warning(
        "peak_memory_mb_process_cumulative is the process-wide cumulative peak "
        "(resource.getrusage), not an isolated per-model measurement -- since "
        "all 5 models run in one process, later models' numbers include earlier "
        "models' peak. Report this caveat in the manuscript; do not present it "
        "as a clean per-model figure without the caveat."
    )


if __name__ == "__main__":
    main()
