"""SHAP-based explainability, per Section 9 of
STAGE2_PROPOSAL_AND_METHODOLOGY.md.

This module answers one specific question -- do the features driving a
model's predictions differ between its random-split-trained version and
its LOSO-trained version -- rather than producing generic SHAP plots for
their own sake. TreeExplainer is used for RF/XGBoost; for the torch models
a KernelExplainer or GradientExplainer is used on a background sample,
kept small deliberately (see sample_size) because SHAP on deep models is
expensive and this project's compute budget is laptop-scale.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import shap

from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_tree_shap_importance(
    model: Any, X_background: pd.DataFrame, feature_names: List[str]
) -> Dict[str, float]:
    """Mean absolute SHAP value per feature for a tree-based model.

    Args:
        model: A fitted RandomForestClassifier or XGBClassifier.
        X_background: A sample of feature rows to explain (kept modest in
            size by the caller -- SHAP cost grows with sample size).
        feature_names: Column names matching X_background's columns.

    Returns:
        Dict mapping feature name -> mean absolute SHAP value, sorted
        descending by importance.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_background)

    # For multiclass tree models, shap_values is a list (one array per
    # class); average the absolute value across classes for an overall
    # importance ranking. For binary/regression-style output it is a
    # single array.
    if isinstance(shap_values, list):
        stacked = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        stacked = np.abs(shap_values)

    mean_abs = np.mean(stacked, axis=0)
    importance = dict(zip(feature_names, mean_abs.tolist()))
    return dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))


def compute_torch_shap_importance(
    predict_fn: Any,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    sample_size: int = 100,
) -> Dict[str, float]:
    """Mean absolute SHAP value per feature for a torch-wrapped model.

    Uses shap.KernelExplainer, which is model-agnostic but computationally
    expensive -- sample_size caps how many rows are actually explained, and
    this cap (not the full test set) is what is reported. This is stated
    explicitly rather than left implicit, since it changes how much weight
    the resulting importances should be given relative to a full-dataset
    tree-model SHAP computation.

    Args:
        predict_fn: A callable mapping a 2D numpy array of flow-independent
            features to class probabilities (e.g. model_wrapper.predict_proba
            for the MLP; for sequence-aware models, feature_names/X should
            already be flattened to (n, seq_len * n_features) by the caller,
            since KernelExplainer expects 2D input).
        X_background: Background sample for the explainer's baseline
            (typically 20-50 rows).
        X_explain: The rows to actually compute SHAP values for.
        feature_names: Names for the (possibly flattened) feature columns.
        sample_size: Max number of rows from X_explain actually explained.

    Returns:
        Dict mapping feature name -> mean absolute SHAP value, sorted
        descending by importance.
    """
    n = min(sample_size, X_explain.shape[0])
    logger.info(
        "Computing KernelExplainer SHAP values for %d of %d available rows "
        "(sample_size cap=%d) -- this is a subsample, not the full test set.",
        n,
        X_explain.shape[0],
        sample_size,
    )
    explainer = shap.KernelExplainer(predict_fn, X_background)
    shap_values = explainer.shap_values(X_explain[:n], nsamples="auto")

    if isinstance(shap_values, list):
        stacked = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        stacked = np.abs(shap_values)

    mean_abs = np.mean(stacked, axis=0)
    importance = dict(zip(feature_names, mean_abs.tolist()))
    return dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))


def compare_importance_rankings(
    importance_a: Dict[str, float], importance_b: Dict[str, float], top_k: int = 15
) -> Dict[str, Any]:
    """Compare two SHAP importance rankings (e.g. random-split vs. LOSO).

    Args:
        importance_a: Feature -> importance dict, condition A.
        importance_b: Feature -> importance dict, condition B.
        top_k: How many top features (by condition A's ranking) to compare.

    Returns:
        Dict with top_k_a, top_k_b (the ranked feature lists), and
        jaccard_overlap (the fraction of A's top-k features that also
        appear in B's top-k), a simple, interpretable measure of whether
        the two models rely on similar features.
    """
    top_a = list(importance_a.keys())[:top_k]
    top_b = list(importance_b.keys())[:top_k]
    overlap = set(top_a) & set(top_b)
    jaccard = len(overlap) / len(set(top_a) | set(top_b)) if (top_a or top_b) else 0.0
    return {
        "top_k": top_k,
        "top_k_a": top_a,
        "top_k_b": top_b,
        "shared_features": sorted(overlap),
        "jaccard_overlap": jaccard,
    }
