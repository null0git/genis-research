"""Save and reload trained models.

By default, experiments do NOT save trained models -- only their computed
metrics (results/processed/*.json), per the reproducibility design in
README.md's "Result provenance" section. This module exists for the one
place that genuinely needs an actual trained model object rather than its
metrics: the SHAP explainability comparison (Section 9 of
STAGE2_PROPOSAL_AND_METHODOLOGY.md, src/evaluation/explain.py), which
compares what features a random-split-trained model relies on against a
LOSO-trained version of the same model -- that comparison needs both
actual fitted models available, not just their scores.

Saving is opt-in per experiment config (save_models: true) precisely
because trained-model files can be large, especially across 5 seeds or 8
LOSO folds -- most experiments don't need this and shouldn't pay the
storage cost by default. See README.md for the storage-size guidance.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

from src.utils.logging import get_logger

logger = get_logger(__name__)

TORCH_MODEL_FAMILIES = {"mlp", "cnn1d", "gru"}
SKLEARN_MODEL_FAMILIES = {"random_forest", "xgboost"}


def save_model(model: Any, model_name: str, filepath: Path) -> None:
    """Save a fitted model to disk, dispatching by model family.

    Args:
        model: A fitted model object -- either a plain sklearn/XGBoost
            estimator (random_forest, xgboost) or a TorchClassifierWrapper
            (mlp, cnn1d, gru), as returned by
            src/training/train.py::_build_and_fit_model.
        model_name: One of "random_forest", "xgboost", "mlp", "cnn1d",
            "gru" -- determines the save format.
        filepath: Where to write the file. Parent directories are created
            if needed. Convention: use a ".pkl" extension for sklearn/
            XGBoost models and ".pt" for torch models, but this function
            does not enforce that -- the caller controls the extension.

    Raises:
        ValueError: If model_name is not a recognized model family.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if model_name in SKLEARN_MODEL_FAMILIES:
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
    elif model_name in TORCH_MODEL_FAMILIES:
        import torch

        torch.save(
            {
                "state_dict": model.model.state_dict(),
                "hyperparameters": model.hp,
                "n_classes": model.n_classes,
                "model_name": model_name,
            },
            filepath,
        )
    else:
        raise ValueError(
            f"Unrecognized model_name '{model_name}' -- must be one of "
            f"{SKLEARN_MODEL_FAMILIES | TORCH_MODEL_FAMILIES}."
        )

    size_mb = filepath.stat().st_size / (1024 * 1024)
    logger.info("Saved %s model to %s (%.2f MB).", model_name, filepath, size_mb)


def load_model(
    model_name: str,
    filepath: Path,
    n_features: int | None = None,
    sequence_length: int | None = None,
) -> Any:
    """Load a previously-saved model.

    Args:
        model_name: One of "random_forest", "xgboost", "mlp", "cnn1d",
            "gru" -- must match what save_model was called with.
        filepath: Path to the saved file.
        n_features: Required for torch model families -- the number of
            input features (or features per timestep, for sequence-aware
            models), needed to rebuild the network architecture before
            loading its weights. Not needed for sklearn/XGBoost.
        sequence_length: Required for cnn1d specifically (its architecture
            is built with a fixed sequence length); optional for gru
            (kept only for interface parity, see build_gru_wrapper).

    Returns:
        The reloaded model object, in exactly the same fit-ready state as
        what save_model was originally given (a plain sklearn/XGBoost
        estimator, or a TorchClassifierWrapper with weights restored).

    Raises:
        ValueError: If model_name is not recognized, or if a torch model
            family is requested without n_features.
    """
    if model_name in SKLEARN_MODEL_FAMILIES:
        with open(filepath, "rb") as f:
            return pickle.load(f)

    if model_name in TORCH_MODEL_FAMILIES:
        import torch

        if n_features is None:
            raise ValueError(
                f"n_features is required to reload a '{model_name}' model "
                f"(needed to rebuild its architecture before loading weights)."
            )
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        hp = checkpoint["hyperparameters"]
        n_classes = checkpoint["n_classes"]

        if model_name == "mlp":
            from src.models.mlp import build_mlp_wrapper

            wrapper = build_mlp_wrapper(hp, n_features=n_features, n_classes=n_classes)
        elif model_name == "cnn1d":
            from src.models.cnn1d import build_cnn1d_wrapper

            wrapper = build_cnn1d_wrapper(
                hp, n_features=n_features, n_classes=n_classes, sequence_length=sequence_length
            )
        else:  # gru
            from src.models.gru import build_gru_wrapper

            wrapper = build_gru_wrapper(
                hp, n_features=n_features, n_classes=n_classes, sequence_length=sequence_length
            )

        wrapper.model.load_state_dict(checkpoint["state_dict"])
        wrapper.model.to(wrapper.device)
        return wrapper

    raise ValueError(
        f"Unrecognized model_name '{model_name}' -- must be one of "
        f"{SKLEARN_MODEL_FAMILIES | TORCH_MODEL_FAMILIES}."
    )


def model_save_path(
    output_dir: Path, output_prefix: str, model_name: str, run_label: str, extension: str
) -> Path:
    """Construct a consistent save path for one model/seed-or-fold combination.

    Args:
        output_dir: Base directory for saved models (e.g. results/models/).
        output_prefix: The experiment's output_prefix (e.g. "e01").
        model_name: Model family name.
        run_label: Seed number (as a string) or held-out scenario name --
            whatever distinguishes this run from others of the same model
            within the same experiment.
        extension: File extension without the dot, e.g. "pkl" or "pt".

    Returns:
        A Path like results/models/e01/random_forest_seed42.pkl.
    """
    return output_dir / output_prefix / f"{model_name}_{run_label}.{extension}"
