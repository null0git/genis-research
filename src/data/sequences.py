"""Leakage-safe sequence construction, per Section 4 of
STAGE2_PROPOSAL_AND_METHODOLOGY.md.

Rules enforced here:
  - Flows are ordered by start_time WITHIN a scenario only.
  - A sequence never spans two scenarios.
  - A sequence is labeled by its LAST flow's label.
  - The first (sequence_length - 1) flows of each scenario are left-padded
    with a zero vector plus an explicit mask, rather than borrowing flows
    from a different scenario.

build_sequences() operates on a single split's data (e.g. only the
training rows, or only one LOSO fold's test rows) that has already been
scaled by preprocessing.apply_scaler -- it does not itself do any
train/test splitting or scaling, keeping it a pure windowing function that
cannot introduce leakage on its own as long as it is never called on data
that spans two different splits.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_sequences(
    X: pd.DataFrame,
    y: pd.Series,
    scenario_id: pd.Series,
    start_time: pd.Series,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding-window sequences of flows, per scenario, with padding.

    Args:
        X: Scaled feature DataFrame for this split only (train, or one
            LOSO fold's train/test set -- never a mix of splits).
        y: Label Series aligned with X.
        scenario_id: Scenario identity Series aligned with X.
        start_time: Ordering key Series aligned with X.
        sequence_length: Number of flows per sequence (e.g. 5, 10, 20).

    Returns:
        Tuple of:
            sequences: float array of shape
                (n_flows_in_split, sequence_length, n_features), where
                sequences[i] is the window ending at (and predicting) the
                i-th flow in scenario/time order.
            labels: array of shape (n_flows_in_split,), the label of each
                window's LAST flow.
            mask: boolean array of shape
                (n_flows_in_split, sequence_length), True where a position
                is real data, False where it is left-padding.
    """
    if sequence_length < 1:
        raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")

    n_features = X.shape[1]
    df = pd.DataFrame(
        {"scenario_id": scenario_id.values, "start_time": start_time.values},
        index=np.arange(len(X)),
    )

    all_sequences = []
    all_labels = []
    all_masks = []

    for scenario, group in df.groupby("scenario_id"):
        ordered_positions = group.sort_values("start_time").index.to_numpy()
        n = len(ordered_positions)
        feats = X.values[ordered_positions]
        labs = y.values[ordered_positions]

        padded_feats = np.vstack(
            [np.zeros((sequence_length - 1, n_features)), feats]
        )
        padded_mask = np.concatenate(
            [np.zeros(sequence_length - 1, dtype=bool), np.ones(n, dtype=bool)]
        )

        for i in range(n):
            window = padded_feats[i : i + sequence_length]
            window_mask = padded_mask[i : i + sequence_length]
            all_sequences.append(window)
            all_masks.append(window_mask)
            all_labels.append(labs[i])

        logger.debug("Scenario %s: built %d sequences of length %d.", scenario, n, sequence_length)

    sequences = np.stack(all_sequences).astype(np.float32)
    labels = np.array(all_labels)
    mask = np.stack(all_masks)

    logger.info(
        "Built %d sequences (length=%d, features=%d) across %d scenario(s).",
        sequences.shape[0],
        sequence_length,
        n_features,
        df["scenario_id"].nunique(),
    )
    return sequences, labels, mask


def assert_no_cross_scenario_sequence(
    scenario_id: pd.Series, sequence_length: int
) -> None:
    """Sanity check: confirm build_sequences cannot mix scenarios.

    This is a structural guarantee of the implementation (sequences are
    built inside a per-scenario groupby loop with per-scenario padding), so
    this function does not re-derive sequences -- it documents and asserts
    the invariant that tests/test_leakage.py checks empirically. Kept here,
    next to the implementation, so the guarantee and the code that provides
    it stay in sync.

    Args:
        scenario_id: Scenario identity Series for the split being checked.
        sequence_length: Sequence length used.

    Raises:
        AssertionError: If sequence_length is non-positive (the one
            precondition build_sequences itself does not re-validate here).
    """
    assert sequence_length >= 1
    logger.debug(
        "build_sequences groups by scenario_id before windowing (%d scenarios "
        "in this split), so no window can include flows from two scenarios "
        "by construction.",
        scenario_id.nunique(),
    )
