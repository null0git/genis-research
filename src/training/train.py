"""Core training orchestration.

run_experiment() is the single entry point every experiments/eXX/run.py
script calls. It is split into two layers deliberately:

  - run_experiment(config_path, ...) -- CLI-facing: loads real GeNIS data
    from disk and calls the layer below.
  - run_experiment_from_dataframe(df, config, ...) -- takes an
    already-loaded DataFrame directly, so it can be exercised in
    tests/test_leakage.py and in local smoke-testing against a small
    synthetic dataset without needing the real (large) GeNIS download.

Both layers write the same JSON-serializable metrics structure to
results/processed/{output_prefix}_metrics.json, and print structured log
lines that results/raw/{output_prefix}_terminal_output.txt is meant to
capture (see README.md's reproducibility instructions for the exact shell
redirection to use).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from src.data import preprocessing, splits as split_mod, sequences as seq_mod
from src.data.audit import find_interval_files
from src.evaluation.metrics import compute_classification_metrics, aggregate_metrics_across_runs
from src.utils.config import load_experiment_config, get_repo_root
from src.utils.logging import get_logger
from src.utils.seeding import set_all_seeds

logger = get_logger(__name__)

SEQUENCE_AWARE_FAMILIES = {"cnn1d", "gru"}


def _carve_validation(
    train_idx: np.ndarray, y: pd.Series, val_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a training index set into (train, val) for early stopping.

    Used when a split strategy (LOSO, temporal) does not itself define a
    validation set. Stratified by label where possible; falls back to an
    unstratified split if a class has too few members to stratify.

    Args:
        train_idx: Positional indices making up the training set.
        y: Full label Series (indexed positionally, aligned with train_idx).
        val_frac: Fraction of train_idx to carve out for validation.
        seed: Random seed.

    Returns:
        (inner_train_idx, val_idx), both subsets of train_idx.
    """
    labels = y.values[train_idx]
    try:
        inner_train, val = train_test_split(
            train_idx, test_size=val_frac, stratify=labels, random_state=seed
        )
    except ValueError:
        logger.warning(
            "Stratified validation carve-out failed (a class likely has too "
            "few members in this fold); falling back to an unstratified split."
        )
        inner_train, val = train_test_split(train_idx, test_size=val_frac, random_state=seed)
    return inner_train, val


def _build_and_fit_model(
    model_cfg: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    class_weights: Dict[int, float],
    use_sequences: bool,
) -> Tuple[Any, float]:
    """Build and fit one model, returning (fitted_model, train_time_seconds).

    Args:
        model_cfg: A loaded configs/models/*.yaml dict.
        X_train, y_train, X_val, y_val: Arrays (2D for flow-independent
            models, 3D (n, seq_len, n_features) for sequence-aware models).
        n_classes: Number of target classes.
        class_weights: Per-class weights computed on the training split.
        use_sequences: Whether this is a sequence-aware run (affects only
            logging/validation here; the array shapes already reflect it).

    Returns:
        Tuple of the fitted model object (interface varies by family, but
        every returned object supports .predict(X) and .predict_proba(X))
        and the wall-clock training time in seconds.

    Raises:
        ValueError: If model_cfg["model_name"] is not recognized.
    """
    name = model_cfg["model_name"]
    hp = model_cfg["hyperparameters"]
    start = time.perf_counter()

    if name == "random_forest":
        from src.models.tree_models import build_random_forest

        model = build_random_forest(hp)
        model.fit(X_train, y_train)
    elif name == "xgboost":
        from src.models.tree_models import build_xgboost, fit_xgboost

        model = build_xgboost(hp, n_classes)
        model = fit_xgboost(model, X_train, y_train, class_weights)
    elif name == "mlp":
        from src.models.mlp import build_mlp_wrapper

        model = build_mlp_wrapper(hp, n_features=X_train.shape[1], n_classes=n_classes, class_weights=class_weights)
        model.fit(X_train, y_train, X_val, y_val)
    elif name == "cnn1d":
        from src.models.cnn1d import build_cnn1d_wrapper

        model = build_cnn1d_wrapper(
            hp,
            n_features=X_train.shape[2],
            n_classes=n_classes,
            class_weights=class_weights,
            sequence_length=X_train.shape[1],
        )
        model.fit(X_train, y_train, X_val, y_val)
    elif name == "gru":
        from src.models.gru import build_gru_wrapper

        model = build_gru_wrapper(
            hp,
            n_features=X_train.shape[2],
            n_classes=n_classes,
            class_weights=class_weights,
            sequence_length=X_train.shape[1],
        )
        model.fit(X_train, y_train, X_val, y_val)
    else:
        raise ValueError(f"Unrecognized model_name: {name}")

    train_time = time.perf_counter() - start
    logger.info("Fitted %s in %.2fs.", name, train_time)
    return model, train_time


def _save_model_if_enabled(
    save_models: bool,
    models_dir: Optional[Path],
    model: Any,
    model_name: str,
    output_prefix: str,
    run_label: str,
    n_features: int,
) -> None:
    """Save a fitted model if save_models is enabled, else no-op.

    Centralizes the extension choice (pkl for sklearn/XGBoost, pt for
    torch) and the save_path convention so all three split-strategy
    branches below call one thing instead of repeating this logic.

    Args:
        save_models: Whether saving is enabled for this experiment.
        models_dir: Base directory for saved models (None if disabled).
        model: The fitted model object to save.
        model_name: Model family name.
        output_prefix: The experiment's output_prefix.
        run_label: Seed number or held-out scenario name, as a string.
        n_features: Recorded alongside the save (not currently persisted
            separately, but kept as a parameter for callers -- reloading a
            torch model requires this value, which the caller must supply
            explicitly to persistence.load_model, matching what was used
            at save time).
    """
    if not save_models:
        return
    from src.models.persistence import save_model, model_save_path, TORCH_MODEL_FAMILIES

    extension = "pt" if model_name in TORCH_MODEL_FAMILIES else "pkl"
    path = model_save_path(models_dir, output_prefix, model_name, run_label, extension)
    save_model(model, model_name, path)


def _prepare_split_arrays(
    X: pd.DataFrame,
    y_encoded: np.ndarray,
    scenario_id: pd.Series,
    start_time: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    use_sequences: bool,
    sequence_length: Optional[int],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Scale features (fit on train only) and optionally build sequences.

    Args:
        X: Full, unscaled feature DataFrame.
        y_encoded: Full integer-encoded label array, aligned with X.
        scenario_id: Full scenario_id Series, aligned with X.
        start_time: Full start_time Series, aligned with X.
        train_idx, val_idx, test_idx: Positional index arrays defining the split.
        use_sequences: If True, build leakage-safe sequences per split subset.
        sequence_length: Required if use_sequences is True.

    Returns:
        Dict with keys "train", "val", "test", each mapping to an
        (X_array, y_array) tuple, ready to pass to _build_and_fit_model.
    """
    X_train_raw = X.iloc[train_idx]
    imputer = preprocessing.fit_imputer(X_train_raw)
    X_train_imputed = preprocessing.apply_imputer(imputer, X_train_raw)
    scaler = preprocessing.fit_scaler(X_train_imputed)

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        X_imputed = preprocessing.apply_imputer(imputer, X.iloc[idx])
        X_split = preprocessing.apply_scaler(scaler, X_imputed)
        y_split = y_encoded[idx]

        if use_sequences:
            assert sequence_length is not None
            seq_arr, seq_labels, _mask = seq_mod.build_sequences(
                X_split,
                pd.Series(y_split, index=X_split.index),
                scenario_id.iloc[idx],
                start_time.iloc[idx],
                sequence_length,
            )
            out[split_name] = (seq_arr, seq_labels)
        else:
            out[split_name] = (X_split.values, y_split)

    return out


def run_experiment_from_dataframe(
    df: pd.DataFrame,
    experiment_cfg: Dict[str, Any],
    output_dir: Path,
    label_column_key: str = "binary_label",
) -> Dict[str, Any]:
    """Run a full experiment given an already-loaded flow DataFrame.

    This is the layer exercised directly by tests and by local smoke tests
    against synthetic data; run_experiment() (below) is a thin wrapper that
    loads real data from disk and calls this.

    Args:
        df: Raw flow DataFrame (as returned by preprocessing.load_flows),
            not yet scenario-derived, cleaned, or encoded.
        experiment_cfg: A loaded (via load_experiment_config) experiment
            config dict, with data_config and models already inlined.
        output_dir: Directory to write {output_prefix}_metrics.json into.
        label_column_key: Which column_map entry to use as the target.

    Returns:
        The full results dict, also written to disk as JSON.
    """
    data_cfg = experiment_cfg["data_config"]
    output_prefix = experiment_cfg["output_prefix"]

    df = preprocessing.derive_scenario_column(df, data_cfg)
    X, y, scenario_id, start_time = preprocessing.clean_and_encode(df, data_cfg, label_column_key)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.values)
    n_classes = len(label_encoder.classes_)
    logger.info("Encoded %d classes: %s", n_classes, list(label_encoder.classes_))

    use_sequences = experiment_cfg.get("use_sequences", False)
    sequence_length = experiment_cfg.get("sequence_length")

    # Opt-in model persistence -- OFF by default (see
    # src/models/persistence.py's module docstring for why). Enable by
    # adding `save_models: true` to an experiment config. Models are
    # written under results/models/{output_prefix}/, a sibling of
    # output_dir (results/processed/).
    save_models = experiment_cfg.get("save_models", False)
    models_dir = output_dir.parent / "models" if save_models else None
    if save_models:
        logger.info(
            "save_models=true: trained models will be written under %s -- "
            "this can be a meaningful amount of disk, especially across "
            "multiple seeds/folds. See README.md for storage guidance.",
            models_dir,
        )

    strategy = experiment_cfg["split"]["strategy"]
    results: Dict[str, Any] = {
        "experiment_id": experiment_cfg["experiment_id"],
        "split_strategy": strategy,
        "label_encoder_classes": [str(c) for c in label_encoder.classes_],
        "models": {},
    }

    for model_cfg in experiment_cfg["models"]:
        model_name = model_cfg["model_name"]
        per_run_metrics: List[Dict[str, Any]] = []
        per_run_timing: List[Dict[str, Any]] = []

        if strategy == "random":
            seeds = experiment_cfg["split"]["seeds"]
            for seed in seeds:
                set_all_seeds(seed)
                train_idx, val_idx, test_idx = split_mod.random_stratified_split(
                    pd.Series(y_encoded),
                    experiment_cfg["split"]["train_frac"],
                    experiment_cfg["split"]["val_frac"],
                    experiment_cfg["split"]["test_frac"],
                    seed,
                )
                class_weights = preprocessing.compute_class_weights(
                    pd.Series(y_encoded[train_idx])
                )
                arrays = _prepare_split_arrays(
                    X, y_encoded, scenario_id, start_time,
                    train_idx, val_idx, test_idx, use_sequences, sequence_length,
                )
                model, train_time = _build_and_fit_model(
                    model_cfg, *arrays["train"], *arrays["val"], n_classes, class_weights, use_sequences
                )
                X_test, y_test = arrays["test"]
                y_pred = model.predict(X_test)
                metrics = compute_classification_metrics(y_test, y_pred)
                metrics["seed"] = seed
                per_run_metrics.append(metrics)
                per_run_timing.append({"seed": seed, "train_time_seconds": train_time})
                _save_model_if_enabled(
                    save_models, models_dir, model, model_name, output_prefix,
                    f"seed{seed}", X.shape[1],
                )

        elif strategy == "loso":
            for held_out, train_idx, test_idx in split_mod.leave_one_scenario_out(scenario_id):
                split_mod.validate_no_scenario_leakage(scenario_id, train_idx, test_idx)
                set_all_seeds(42)
                inner_train_idx, val_idx = _carve_validation(
                    train_idx, pd.Series(y_encoded), val_frac=0.15, seed=42
                )
                class_weights = preprocessing.compute_class_weights(
                    pd.Series(y_encoded[inner_train_idx])
                )
                arrays = _prepare_split_arrays(
                    X, y_encoded, scenario_id, start_time,
                    inner_train_idx, val_idx, test_idx, use_sequences, sequence_length,
                )
                model, train_time = _build_and_fit_model(
                    model_cfg, *arrays["train"], *arrays["val"], n_classes, class_weights, use_sequences
                )
                X_test, y_test = arrays["test"]
                y_pred = model.predict(X_test)
                metrics = compute_classification_metrics(y_test, y_pred)
                metrics["held_out_scenario"] = held_out
                per_run_metrics.append(metrics)
                per_run_timing.append({"held_out_scenario": held_out, "train_time_seconds": train_time})
                _save_model_if_enabled(
                    save_models, models_dir, model, model_name, output_prefix,
                    f"heldout_{held_out}", X.shape[1],
                )

        elif strategy == "temporal":
            train_idx, test_idx = split_mod.temporal_split_within_scenario(
                scenario_id, start_time, experiment_cfg["split"]["train_frac"]
            )
            set_all_seeds(42)
            inner_train_idx, val_idx = _carve_validation(
                train_idx, pd.Series(y_encoded), val_frac=0.15, seed=42
            )
            class_weights = preprocessing.compute_class_weights(pd.Series(y_encoded[inner_train_idx]))
            arrays = _prepare_split_arrays(
                X, y_encoded, scenario_id, start_time,
                inner_train_idx, val_idx, test_idx, use_sequences, sequence_length,
            )
            model, train_time = _build_and_fit_model(
                model_cfg, *arrays["train"], *arrays["val"], n_classes, class_weights, use_sequences
            )
            X_test, y_test = arrays["test"]
            y_pred = model.predict(X_test)
            metrics = compute_classification_metrics(y_test, y_pred)
            per_run_metrics.append(metrics)
            per_run_timing.append({"train_time_seconds": train_time})
            _save_model_if_enabled(
                save_models, models_dir, model, model_name, output_prefix,
                "temporal", X.shape[1],
            )

        else:
            raise ValueError(f"Unrecognized split strategy: {strategy}")

        results["models"][model_name] = {
            "per_run_metrics": per_run_metrics,
            "aggregate": aggregate_metrics_across_runs(per_run_metrics),
            "timing": per_run_timing,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{output_prefix}_metrics.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Wrote metrics to %s", out_path)

    return results


def run_experiment(experiment_config_path: str, results_dir: str = "results/processed") -> Dict[str, Any]:
    """CLI-facing entry point: load real GeNIS data, then run the experiment.

    Args:
        experiment_config_path: Path to a configs/experiments/*.yaml file.
        results_dir: Output directory for the metrics JSON, relative to
            the repository root unless absolute.

    Returns:
        The results dict (same structure as run_experiment_from_dataframe).
    """
    experiment_cfg = load_experiment_config(experiment_config_path)
    data_cfg = experiment_cfg["data_config"]

    preprocessed_dir = get_repo_root() / data_cfg["preprocessed_data_dir"]
    interval_files = find_interval_files(preprocessed_dir, data_cfg["flow_interval_seconds"])
    df = preprocessing.load_flows(preprocessed_dir, interval_files)

    output_dir = get_repo_root() / results_dir
    return run_experiment_from_dataframe(df, experiment_cfg, output_dir, experiment_cfg.get("label_column", "binary_label"))
