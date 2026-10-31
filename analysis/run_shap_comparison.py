"""Run the Section 9 SHAP comparison: does a LOSO-trained model rely on
different features than its random-split-trained counterpart?

This is the actual consumer of the model files produced when an
experiment config has `save_models: true` (see
src/models/persistence.py and README.md's "Trained models: saved or
not?" section). It does not retrain anything -- it loads two already-
saved models (one from a random-split experiment, one from a specific
LOSO fold) and compares what each relies on.

Prerequisites:
  1. Run E01 (or e01_random_split_baselines_no_torch) with
     `save_models: true` added to the config.
  2. Run E02 (or e02_loso_baselines_no_torch) with `save_models: true`
     added to the config.
  3. Then run this script.

Usage:
    python analysis/run_shap_comparison.py --model-name random_forest \
        --random-seed 42 --held-out-scenario scenario-7-authtest \
        --random-models-dir results/models/e01 \
        --loso-models-dir results/models/e02
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import preprocessing, splits as split_mod  # noqa: E402
from src.data.audit import find_interval_files  # noqa: E402
from src.evaluation.explain import (  # noqa: E402
    compute_tree_shap_importance,
    compare_importance_rankings,
)
from src.models.persistence import load_model  # noqa: E402
from src.training.train import _prepare_split_arrays, _carve_validation  # noqa: E402
from src.utils.config import load_yaml, get_repo_root  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

TREE_MODEL_FAMILIES = {"random_forest", "xgboost"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default="random_forest",
        help="Which model family to compare (must have been saved for both arms). "
        "Currently only tree models (random_forest, xgboost) are supported by "
        "this script -- TreeExplainer is fast enough to run without extra "
        "sampling machinery; the torch model families would need "
        "compute_torch_shap_importance instead (see src/evaluation/explain.py), "
        "not yet wired up here.",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Which E01 seed's saved model to use.")
    parser.add_argument("--held-out-scenario", required=True, help="Which E02 LOSO fold's saved model to use.")
    parser.add_argument("--random-models-dir", default="results/models/e01")
    parser.add_argument("--loso-models-dir", default="results/models/e02")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--sample-size", type=int, default=200, help="Rows to explain per arm.")
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    if args.model_name not in TREE_MODEL_FAMILIES:
        raise ValueError(
            f"--model-name '{args.model_name}' not supported by this script yet "
            f"(only {TREE_MODEL_FAMILIES}). See the --model-name help text above."
        )

    root = get_repo_root()
    data_cfg = load_yaml(args.data_config)

    preprocessed_dir = root / data_cfg["preprocessed_data_dir"]
    interval_files = find_interval_files(preprocessed_dir, data_cfg["flow_interval_seconds"])
    df = preprocessing.load_flows(preprocessed_dir, interval_files)
    df = preprocessing.derive_scenario_column(df, data_cfg)
    X, y, scenario_id, start_time = preprocessing.clean_and_encode(df, data_cfg, "binary_label")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.values)

    # --- Random-split arm: reproduce the exact same split the saved model was trained on ---
    train_idx, val_idx, test_idx = split_mod.random_stratified_split(
        pd.Series(y_encoded), 0.70, 0.15, 0.15, args.random_seed
    )
    random_arrays = _prepare_split_arrays(
        X, y_encoded, scenario_id, start_time, train_idx, val_idx, test_idx, False, None
    )
    X_random_test, _ = random_arrays["test"]

    random_model_path = (
        root / args.random_models_dir / f"{args.model_name}_seed{args.random_seed}.pkl"
    )
    if not random_model_path.exists():
        raise FileNotFoundError(
            f"{random_model_path} not found. Did you run E01 with save_models: true "
            f"for seed={args.random_seed}?"
        )
    random_model = load_model(args.model_name, random_model_path)
    logger.info("Loaded random-split model from %s", random_model_path)

    # --- LOSO arm: reproduce the specific held-out fold ---
    loso_train_idx = loso_test_idx = None
    for held_out, tr_idx, te_idx in split_mod.leave_one_scenario_out(scenario_id):
        if held_out == args.held_out_scenario:
            loso_train_idx, loso_test_idx = tr_idx, te_idx
            break
    if loso_test_idx is None:
        raise ValueError(
            f"--held-out-scenario '{args.held_out_scenario}' not found among this "
            f"data's scenarios: {sorted(scenario_id.unique())}"
        )
    inner_train_idx, loso_val_idx = _carve_validation(
        loso_train_idx, pd.Series(y_encoded), val_frac=0.15, seed=42
    )  # matches train.py's LOSO branch exactly (same seed=42), so the imputer/scaler
    # fit here are fit on the SAME 85% inner-train subset the saved model actually
    # saw -- not the full LOSO train set, which would scale test rows slightly
    # differently than what the model was trained against.
    loso_arrays = _prepare_split_arrays(
        X, y_encoded, scenario_id, start_time, inner_train_idx, loso_val_idx, loso_test_idx, False, None
    )
    X_loso_test, _ = loso_arrays["test"]

    loso_model_path = (
        root / args.loso_models_dir / f"{args.model_name}_heldout_{args.held_out_scenario}.pkl"
    )
    if not loso_model_path.exists():
        raise FileNotFoundError(
            f"{loso_model_path} not found. Did you run E02 with save_models: true "
            f"for held-out scenario={args.held_out_scenario}?"
        )
    loso_model = load_model(args.model_name, loso_model_path)
    logger.info("Loaded LOSO model (held out=%s) from %s", args.held_out_scenario, loso_model_path)

    # --- Compute and compare SHAP importance ---
    feature_names = list(X.columns)
    n_random = min(args.sample_size, X_random_test.shape[0])
    n_loso = min(args.sample_size, X_loso_test.shape[0])

    logger.info("Computing SHAP importance for the random-split model (%d rows)...", n_random)
    random_importance = compute_tree_shap_importance(
        random_model, X_random_test[:n_random], feature_names
    )
    logger.info("Computing SHAP importance for the LOSO model (%d rows)...", n_loso)
    loso_importance = compute_tree_shap_importance(
        loso_model, X_loso_test[:n_loso], feature_names
    )

    comparison = compare_importance_rankings(random_importance, loso_importance, top_k=args.top_k)

    result = {
        "model_name": args.model_name,
        "random_seed": args.random_seed,
        "held_out_scenario": args.held_out_scenario,
        "random_split_importance": random_importance,
        "loso_importance": loso_importance,
        "comparison": comparison,
    }

    out_path = root / f"results/processed/shap_comparison_{args.model_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)

    logger.info("Wrote %s", out_path)
    logger.info(
        "Top-%d feature overlap (Jaccard) between random-split and LOSO models: %.3f",
        args.top_k, comparison["jaccard_overlap"],
    )
    logger.info("Random-split top features: %s", comparison["top_k_a"][:5])
    logger.info("LOSO top features: %s", comparison["top_k_b"][:5])


if __name__ == "__main__":
    main()
