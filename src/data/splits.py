"""Train/val/test splitting strategies for GeNIS.

Three strategies, matching Section 3 of STAGE2_PROPOSAL_AND_METHODOLOGY.md:

1. random_stratified_split -- shuffled, stratified by label. The
   "optimistic" baseline; flows from the same scenario can land in both
   train and test.
2. leave_one_scenario_out -- a generator yielding one (train_idx, test_idx)
   pair per scenario, with the held-out scenario entirely absent from
   training.
3. temporal_split_within_scenario -- earliest N% of each scenario's flows
   for training, latest (100-N)% for testing, scenario boundaries intact.

Every function here operates on positional/label indices only and does not
touch feature values, so it cannot itself leak preprocessing statistics --
that responsibility belongs to preprocessing.fit_scaler / apply_scaler,
called separately by src/training/train.py using the indices these
functions return.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logging import get_logger

logger = get_logger(__name__)


def random_stratified_split(
    y: pd.Series,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random split, stratified by label, ignoring scenario identity.

    Args:
        y: Full label Series (index used as the row identifier).
        train_frac: Fraction for training.
        val_frac: Fraction for validation.
        test_frac: Fraction for test. train_frac + val_frac + test_frac
            must sum to 1.0 (within floating-point tolerance).
        seed: Random seed for reproducibility.

    Returns:
        (train_idx, val_idx, test_idx) as arrays of positional indices
        into y (i.e. indices for use with .iloc, not .loc).

    Raises:
        ValueError: If the fractions do not sum to 1.0.
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError(
            f"train_frac + val_frac + test_frac must sum to 1.0, got "
            f"{train_frac + val_frac + test_frac}"
        )

    idx = np.arange(len(y))
    labels = y.values

    train_idx, rest_idx, y_train, y_rest = train_test_split(
        idx, labels, train_size=train_frac, stratify=labels, random_state=seed
    )
    rel_val_frac = val_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        rest_idx, train_size=rel_val_frac, stratify=y_rest, random_state=seed
    )
    logger.info(
        "Random split (seed=%d): train=%d val=%d test=%d",
        seed,
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )
    return train_idx, val_idx, test_idx


def leave_one_scenario_out(
    scenario_id: pd.Series,
) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    """Yield one (held_out_scenario, train_idx, test_idx) tuple per scenario.

    Args:
        scenario_id: Series aligned with the full dataset, giving each
            row's scenario identity.

    Yields:
        Tuples of (held_out_scenario_name, train_idx, test_idx), where
        train_idx covers every row NOT in the held-out scenario and
        test_idx covers every row IN the held-out scenario. Both are
        positional index arrays.
    """
    scenarios = sorted(scenario_id.unique())
    idx = np.arange(len(scenario_id))
    values = scenario_id.values

    for held_out in scenarios:
        test_mask = values == held_out
        train_idx = idx[~test_mask]
        test_idx = idx[test_mask]
        logger.info(
            "LOSO fold (held out=%s): train=%d test=%d",
            held_out,
            len(train_idx),
            len(test_idx),
        )
        yield held_out, train_idx, test_idx


def temporal_split_within_scenario(
    scenario_id: pd.Series, start_time: pd.Series, train_frac: float = 0.70
) -> Tuple[np.ndarray, np.ndarray]:
    """Chronological split: earliest train_frac of each scenario for training.

    Scenario boundaries are respected -- flows are ranked by start_time
    separately within each scenario, so training and test both contain
    every scenario, split by time, not one whole scenario going to either
    side.

    Args:
        scenario_id: Series aligned with the full dataset.
        start_time: Series aligned with the full dataset, used for
            within-scenario chronological ordering.
        train_frac: Fraction of each scenario's flows (by time order) to
            place in training; the remainder goes to test.

    Returns:
        (train_idx, test_idx) as positional index arrays.
    """
    df = pd.DataFrame(
        {"scenario_id": scenario_id.values, "start_time": start_time.values},
        index=np.arange(len(scenario_id)),
    )
    train_positions: List[int] = []
    test_positions: List[int] = []

    for scenario, group in df.groupby("scenario_id"):
        ordered = group.sort_values("start_time")
        n_train = int(np.floor(len(ordered) * train_frac))
        train_positions.extend(ordered.index[:n_train].tolist())
        test_positions.extend(ordered.index[n_train:].tolist())

    logger.info(
        "Temporal split: train=%d test=%d (train_frac=%.2f)",
        len(train_positions),
        len(test_positions),
        train_frac,
    )
    return np.array(sorted(train_positions)), np.array(sorted(test_positions))


def validate_no_scenario_leakage(
    scenario_id: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray
) -> None:
    """Assert that no scenario appears on both sides of a LOSO-style split.

    Used only for the LOSO strategy (random and temporal splits are
    expected to share scenarios across train/test by design, so this check
    does not apply to them). Raises rather than warns, since this is a
    correctness invariant, not a heuristic.

    Args:
        scenario_id: Full scenario_id Series.
        train_idx: Positional training indices.
        test_idx: Positional test indices.

    Raises:
        AssertionError: If any scenario appears in both train and test.
    """
    train_scenarios = set(scenario_id.values[train_idx])
    test_scenarios = set(scenario_id.values[test_idx])
    overlap = train_scenarios & test_scenarios
    assert not overlap, f"Scenario leakage detected: {overlap} present in both train and test."
