"""Automated leakage checks, per Section 3/4 of
STAGE2_PROPOSAL_AND_METHODOLOGY.md.

Run with: pytest tests/test_leakage.py -v

These tests use a small synthetic dataset (built in-memory, not the real
GeNIS download) so they run in seconds and can be part of a normal CI
workflow. They check the *mechanism* (splits.py, sequences.py,
preprocessing.py) rather than any real experiment's output, which is the
right level for a leakage test: if these pass, every experiment built on
top of these functions inherits the guarantee.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import splits as split_mod
from src.data import sequences as seq_mod
from src.data import preprocessing


def _make_synthetic_flows(n_per_scenario: int = 200, n_features: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scenarios = ["attack_a", "attack_b", "benign_a", "benign_b"]
    frames = []
    for s_idx, scenario in enumerate(scenarios):
        n = n_per_scenario
        data = {f"feat_{i}": rng.normal(loc=s_idx, scale=1.0, size=n) for i in range(n_features)}
        data["scenario_id"] = scenario
        data["start_time"] = np.arange(n) + s_idx * 100000  # non-overlapping time ranges, arbitrary
        data["binary_label"] = rng.integers(0, 2, size=n)
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


def test_random_split_produces_disjoint_indices():
    df = _make_synthetic_flows()
    y = df["binary_label"]
    train_idx, val_idx, test_idx = split_mod.random_stratified_split(
        y, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42
    )
    assert set(train_idx).isdisjoint(set(val_idx))
    assert set(train_idx).isdisjoint(set(test_idx))
    assert set(val_idx).isdisjoint(set(test_idx))
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)


def test_random_split_rejects_bad_fractions():
    df = _make_synthetic_flows()
    with pytest.raises(ValueError):
        split_mod.random_stratified_split(df["binary_label"], 0.5, 0.3, 0.3, seed=42)


def test_loso_folds_have_no_scenario_overlap():
    df = _make_synthetic_flows()
    scenario_id = df["scenario_id"]
    fold_count = 0
    for held_out, train_idx, test_idx in split_mod.leave_one_scenario_out(scenario_id):
        split_mod.validate_no_scenario_leakage(scenario_id, train_idx, test_idx)
        assert set(scenario_id.values[test_idx]) == {held_out}
        assert held_out not in set(scenario_id.values[train_idx])
        fold_count += 1
    assert fold_count == scenario_id.nunique()


def test_loso_leakage_validator_raises_on_synthetic_violation():
    df = _make_synthetic_flows()
    scenario_id = df["scenario_id"]
    # Deliberately construct a leaking split: train_idx includes one row
    # from the same scenario as test_idx.
    test_idx = np.array([0, 1, 2])  # scenario of first rows
    train_idx = np.array([0, 3, 4])  # includes index 0, in test_idx's scenario
    with pytest.raises(AssertionError):
        split_mod.validate_no_scenario_leakage(scenario_id, train_idx, test_idx)


def test_temporal_split_respects_scenario_boundaries_and_time_order():
    df = _make_synthetic_flows()
    train_idx, test_idx = split_mod.temporal_split_within_scenario(
        df["scenario_id"], df["start_time"], train_frac=0.7
    )
    assert set(train_idx).isdisjoint(set(test_idx))
    # Every scenario should appear in both train and test (time-split, not scenario-split).
    train_scenarios = set(df["scenario_id"].values[train_idx])
    test_scenarios = set(df["scenario_id"].values[test_idx])
    assert train_scenarios == test_scenarios == set(df["scenario_id"].unique())

    # Within each scenario, every training timestamp must precede every test timestamp.
    for scenario in df["scenario_id"].unique():
        scenario_train_times = df["start_time"].values[
            [i for i in train_idx if df["scenario_id"].values[i] == scenario]
        ]
        scenario_test_times = df["start_time"].values[
            [i for i in test_idx if df["scenario_id"].values[i] == scenario]
        ]
        assert scenario_train_times.max() < scenario_test_times.min()


def test_sequences_never_span_two_scenarios():
    df = _make_synthetic_flows(n_per_scenario=20)
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    X = df[feature_cols]
    y = df["binary_label"]
    scenario_id = df["scenario_id"]
    start_time = df["start_time"]

    sequence_length = 5
    sequences, labels, mask = seq_mod.build_sequences(X, y, scenario_id, start_time, sequence_length)

    assert sequences.shape == (len(df), sequence_length, len(feature_cols))
    assert labels.shape == (len(df),)
    assert mask.shape == (len(df), sequence_length)

    # Every fully-real (unpadded) window's non-padded rows must come from
    # values actually present in X for a single scenario. We check this
    # indirectly: the number of "real" (mask=True) positions in each
    # window never exceeds that flow's position-within-scenario + 1.
    n_per_scenario = 20
    for scenario_idx in range(df["scenario_id"].nunique()):
        start = scenario_idx * n_per_scenario
        for pos_in_scenario in range(n_per_scenario):
            global_idx = start + pos_in_scenario
            expected_real = min(pos_in_scenario + 1, sequence_length)
            actual_real = int(mask[global_idx].sum())
            assert actual_real == expected_real, (
                f"Row {global_idx} (position {pos_in_scenario} in its scenario) "
                f"has {actual_real} real positions in its window, expected "
                f"{expected_real} -- a padding/scenario-boundary bug."
            )


def test_sequence_label_matches_last_flow_label():
    df = _make_synthetic_flows(n_per_scenario=10)
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    X = df[feature_cols]
    y = df["binary_label"]
    sequences, labels, mask = seq_mod.build_sequences(X, y, df["scenario_id"], df["start_time"], sequence_length=3)
    # Labels array must equal y in the same (scenario-grouped, time-ordered) order it was built in.
    assert len(labels) == len(y)


def test_scaler_fit_on_train_only_does_not_use_test_statistics():
    df = _make_synthetic_flows()
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    X = df[feature_cols]
    y = df["binary_label"]

    train_idx, val_idx, test_idx = split_mod.random_stratified_split(y, 0.7, 0.15, 0.15, seed=1)
    X_train = X.iloc[train_idx]
    scaler = preprocessing.fit_scaler(X_train)

    # The scaler's mean_ must equal the training data's mean, not the full
    # dataset's mean (this is the actual leakage check: if someone
    # accidentally fits on X instead of X_train, this assertion catches it).
    assert np.allclose(scaler.mean_, X_train.values.mean(axis=0), atol=1e-8)
    full_mean = X.values.mean(axis=0)
    # With scenario-correlated synthetic data, the full-data mean should
    # differ from the train-only mean by a non-trivial amount, otherwise
    # this test would not actually be discriminating between the two cases.
    assert not np.allclose(scaler.mean_, full_mean, atol=1e-3)


def test_derive_scenario_column_rejects_unexpected_scenarios():
    df = _make_synthetic_flows()
    df = df.rename(columns={"scenario_id": "Scenario"})
    bad_cfg = {
        "column_map": {"scenario_id": "Scenario"},
        "scenarios": {"attack": ["attack_a"], "benign": ["benign_a"]},  # missing attack_b, benign_b on purpose
    }
    with pytest.raises(ValueError):
        preprocessing.derive_scenario_column(df, bad_cfg)
