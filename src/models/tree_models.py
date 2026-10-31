"""Random Forest and XGBoost builders.

Both return a fitted-on-demand estimator with the standard scikit-learn
fit/predict/predict_proba interface, so src/training/train.py can treat
them uniformly alongside the torch-wrapped models.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_random_forest(hyperparameters: Dict[str, Any]) -> RandomForestClassifier:
    """Build an (unfit) RandomForestClassifier from a hyperparameter dict.

    Args:
        hyperparameters: The "hyperparameters" block from
            configs/models/rf.yaml.

    Returns:
        An unfit RandomForestClassifier instance.
    """
    return RandomForestClassifier(
        n_estimators=hyperparameters["n_estimators"],
        max_depth=hyperparameters.get("max_depth"),
        min_samples_leaf=hyperparameters.get("min_samples_leaf", 1),
        class_weight=hyperparameters.get("class_weight", "balanced"),
        n_jobs=hyperparameters.get("n_jobs", -1),
        random_state=hyperparameters.get("random_state", 42),
    )


def build_xgboost(hyperparameters: Dict[str, Any], n_classes: int) -> "XGBClassifier":
    """Build an (unfit) XGBClassifier from a hyperparameter dict.

    XGBoost does not accept class_weight="balanced" directly; class
    imbalance is instead handled via a per-sample weight vector passed at
    fit time (see fit_xgboost below), computed from
    preprocessing.compute_class_weights on the training split only.

    Args:
        hyperparameters: The "hyperparameters" block from
            configs/models/xgboost.yaml.
        n_classes: Number of target classes (2 for binary, >2 for
            multiclass), used to select the correct XGBoost objective.

    Returns:
        An unfit XGBClassifier instance.
    """
    from xgboost import XGBClassifier

    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    kwargs: Dict[str, Any] = dict(
        n_estimators=hyperparameters["n_estimators"],
        max_depth=hyperparameters["max_depth"],
        learning_rate=hyperparameters["learning_rate"],
        subsample=hyperparameters.get("subsample", 1.0),
        colsample_bytree=hyperparameters.get("colsample_bytree", 1.0),
        eval_metric=hyperparameters.get("eval_metric", "logloss"),
        n_jobs=hyperparameters.get("n_jobs", -1),
        random_state=hyperparameters.get("random_state", 42),
        objective=objective,
        use_label_encoder=False,
    )
    if n_classes > 2:
        kwargs["num_class"] = n_classes
    return XGBClassifier(**kwargs)


def fit_xgboost(
    model: "XGBClassifier",
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weights: Optional[Dict[int, float]] = None,
) -> XGBClassifier:
    """Fit an XGBClassifier with optional per-sample class weighting.

    Args:
        model: An unfit XGBClassifier from build_xgboost.
        X_train: Training features.
        y_train: Training labels (integer-encoded).
        class_weights: Optional dict mapping class label -> weight,
            computed on the training split only
            (preprocessing.compute_class_weights). If None, all samples
            are weighted equally.

    Returns:
        The fitted model (same object, mutated in place, also returned for
        convenience).
    """
    if class_weights is not None:
        sample_weight = np.array([class_weights[label] for label in y_train])
    else:
        sample_weight = None
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model
