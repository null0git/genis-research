"""Load and preprocess GeNIS 60-second flow data.

This module never fits any statistic (scaler, feature selector, class
weights) on data outside the training split it is given -- every function
that could leak (fit_scaler, fit_class_weights) takes a training-only
DataFrame explicitly and returns a fitted object to be *applied* to
val/test separately. See tests/test_leakage.py for the automated check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_flows(preprocessed_dir: Path, interval_files: List[Path]) -> pd.DataFrame:
    """Concatenate one or more flow CSVs into a single DataFrame.

    Args:
        preprocessed_dir: Root of the preprocessed data (used only for
            logging context).
        interval_files: The specific CSV files to load, as returned by
            src.data.audit.find_interval_files.

    Returns:
        A single concatenated DataFrame, with a `_source_file` column
        recording which file each row came from (used by
        derive_scenario_column when no explicit scenario column exists).
    """
    frames = []
    for path in interval_files:
        df = pd.read_csv(path, low_memory=False)
        df["_source_file"] = path.stem
        frames.append(df)
    if not frames:
        raise ValueError(f"No files to load from {preprocessed_dir}.")
    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d flows from %d file(s).", len(combined), len(frames))
    return combined


def derive_scenario_column(df: pd.DataFrame, data_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Ensure a scenario-identity column exists, deriving it if necessary.

    If configs/data.yaml's column_map.scenario_id is already a column in
    df, this is a no-op (beyond validating its values against the expected
    scenario list). Otherwise, this falls back to using `_source_file`
    (set by load_flows) as a best-effort scenario identifier, on the
    assumption that GeNIS ships one file per scenario -- an assumption that
    MUST be confirmed against the real audit.py output, not trusted blindly.

    Args:
        df: Flow DataFrame, as returned by load_flows.
        data_cfg: The loaded configs/data.yaml dict.

    Returns:
        df with a guaranteed `scenario_id` column (internal name, distinct
        from whatever column_map maps it to in the source files).

    Raises:
        ValueError: If a scenario identifier is discovered that is not in
            the expected scenario list in configs/data.yaml -- this is
            intentionally strict, so a silent mismatch cannot leak into
            split construction.
    """
    df = df.copy()
    scenario_col = data_cfg["column_map"].get("scenario_id")

    if scenario_col and scenario_col in df.columns:
        df["scenario_id"] = df[scenario_col].astype(str)
    else:
        logger.warning(
            "Column '%s' not found in the loaded data; deriving scenario_id "
            "from source filename instead. Confirm this is correct with "
            "src.data.audit before trusting downstream splits.",
            scenario_col,
        )
        df["scenario_id"] = df["_source_file"].astype(str)

    expected = set(data_cfg["scenarios"]["attack"]) | set(data_cfg["scenarios"]["benign"])
    discovered = set(df["scenario_id"].unique())
    unexpected = discovered - expected
    if unexpected:
        raise ValueError(
            f"Discovered scenario identifiers not listed in configs/data.yaml's "
            f"scenarios block: {sorted(unexpected)}. Update configs/data.yaml's "
            f"scenarios.attack / scenarios.benign lists to match the real data "
            f"(see the file-by-file breakdown in "
            f"results/processed/data_audit_summary.json) before proceeding -- "
            f"this check exists specifically to prevent silently mislabeled or "
            f"unrecognized scenarios from entering the LOSO split logic."
        )
    return df


def _parse_start_time(raw: pd.Series) -> pd.Series:
    """Parse GeNIS's StartTime column into a sortable numeric value.

    CONFIRMED against a real scenarios-60-sec/scenario-7-authtest.csv file:
    StartTime is a zero-padded "HH:MM:SS" time-of-day string (e.g.
    "08:55:38"), NOT a Unix epoch numeric value, and NOT a full datetime
    (no date component -- the real file's values ranged 08:55:38 to
    16:59:57, a single day, no midnight rollover). This function converts
    that format to seconds-since-midnight (a plain float), which sorts
    identically to the original string for a single-day scenario capture.

    If a scenario capture ever DOES cross midnight, seconds-since-midnight
    would incorrectly wrap around (23:59:59 -> 00:00:01 looks like a huge
    decrease, not a 2-second increase) -- this function does NOT attempt to
    detect or correct for that case; if a future scenario file exhibits a
    non-monotonic parsed result, verify with audit.py before trusting
    sequence order built from it, and consider adding a monotonic
    day-rollover correction here rather than assuming it away.

    Falls back to plain numeric parsing (pd.to_numeric) for any value that
    doesn't match "HH:MM:SS", so this remains compatible with a possible
    future GeNIS release that ships a true Unix-epoch StartTime instead.

    Args:
        raw: The raw StartTime column, as loaded from CSV (string dtype).

    Returns:
        A float Series of seconds-since-midnight (or the original numeric
        value, for any row that didn't match HH:MM:SS), NaN for anything
        unparseable by either path.
    """
    parsed_time = pd.to_datetime(raw, format="%H:%M:%S", errors="coerce")
    seconds_since_midnight = (
        parsed_time.dt.hour * 3600 + parsed_time.dt.minute * 60 + parsed_time.dt.second
    ).astype(float)

    still_missing = seconds_since_midnight.isna()
    if still_missing.any():
        numeric_fallback = pd.to_numeric(raw[still_missing], errors="coerce")
        seconds_since_midnight.loc[still_missing] = numeric_fallback

    return seconds_since_midnight


def _coerce_numeric_with_hex(series: pd.Series) -> pd.Series:
    """Coerce a string column to numeric, handling GeNIS's hex-formatted values.

    CONFIRMED against real, combined 8-scenario data: Sport/Dport contain
    hex-formatted strings ("0x0008", "0x7ce9", etc.) for protocols without
    real TCP/UDP ports (ARP, ICMP, and similar) -- these are not garbage,
    they are Argus's encoding for a non-applicable-port sentinel or a
    protocol-specific code (e.g. an ICMP type/code packed into the field).
    One value in particular, "0x7ce9", appears as a fixed constant exactly
    65,536 (=2^16) times across the combined dataset -- a strong signal
    it's a placeholder value for "no real port," not per-flow information,
    but it is parsed and kept as a legitimate (if unusual) numeric value
    rather than discarded, since dropping it would silently throw away the
    "this flow's protocol has no port" signal that its consistency
    encodes.

    Args:
        series: A raw string column (e.g. X['Sport']).

    Returns:
        A float Series: plain decimal strings parsed as decimal, "0x..."
        strings parsed as their base-16 integer value, anything else NaN.
    """
    decimal_parsed = pd.to_numeric(series, errors="coerce")

    still_missing = decimal_parsed.isna()
    if still_missing.any():
        candidates = series[still_missing].astype(str).str.strip()
        is_hex = candidates.str.match(r"^0[xX][0-9a-fA-F]+$")
        hex_values = candidates[is_hex]
        if len(hex_values) > 0:
            hex_parsed = hex_values.apply(lambda s: int(s, 16)).astype(float)
            decimal_parsed.loc[hex_parsed.index] = hex_parsed

    return decimal_parsed


def clean_and_encode(
    df: pd.DataFrame, data_cfg: Dict[str, Any], label_column_key: str = "binary_label"
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Drop non-modeling columns, one-hot encode categoricals, separate labels.

    Args:
        df: Flow DataFrame with a scenario_id column already present
            (see derive_scenario_column).
        data_cfg: The loaded configs/data.yaml dict.
        label_column_key: Which entry in column_map to use as the
            classification target ("binary_label" or "category_label").

    Returns:
        Tuple of (X, y, scenario_id, start_time):
            X: feature DataFrame, numeric only, ready for scaling.
            y: label Series.
            scenario_id: Series aligned with X's index.
            start_time: Series aligned with X's index (used for the
                temporal split and for sequence ordering); NaN-filled with
                a warning if the configured start_time column is absent.
    """
    df = df.copy()
    col_map = data_cfg["column_map"]
    label_col = col_map[label_column_key]

    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' (from column_map.{label_column_key}) "
            f"not found. Available columns: {list(df.columns)}"
        )
    y = df[label_col].copy()

    start_time_col = col_map.get("start_time")
    if start_time_col and start_time_col in df.columns:
        start_time = _parse_start_time(df[start_time_col])
        if start_time.isna().any():
            logger.warning(
                "%d rows have an unparseable or missing start_time value; these "
                "will sort last within their scenario for sequence construction.",
                int(start_time.isna().sum()),
            )
    else:
        logger.warning(
            "start_time column '%s' not found; using row order within scenario "
            "as a fallback ordering. This is a materially weaker assumption "
            "than a real timestamp and MUST be flagged in the manuscript's "
            "Limitations section if it is actually used.",
            start_time_col,
        )
        start_time = pd.Series(np.arange(len(df)), index=df.index)

    scenario_id = df["scenario_id"].copy()

    drop_cols = set(data_cfg.get("drop_columns", []))
    drop_cols |= {
        col_map.get("binary_label"),
        col_map.get("category_label"),
        col_map.get("subcategory_label"),
        col_map.get("scenario_id"),
        col_map.get("start_time"),
        "scenario_id",
        "_source_file",
    }
    drop_cols = {c for c in drop_cols if c and c in df.columns}
    X = df.drop(columns=list(drop_cols))

    categorical_cols = [c for c in data_cfg.get("categorical_columns", []) if c in X.columns]

    # Safety guard against a real failure mode: one-hot encoding a
    # high-cardinality column (e.g. a port number field with tens of
    # thousands of distinct values, if it were ever misconfigured into
    # categorical_columns) creates one dummy column PER DISTINCT VALUE.
    # With hundreds of thousands of rows, that produces an array with more
    # columns than there is RAM to hold -- observed in practice as
    # `numpy._core._exceptions._ArrayMemoryError: unable to allocate ~51 GiB
    # for an array with shape (544585, 101389) and data type bool` when
    # Sport/Dport ended up in categorical_columns instead of being treated
    # as plain numeric features. Raise a clear, actionable error here
    # instead of letting pandas attempt the allocation and crash with an
    # opaque MemoryError deep inside get_dummies.
    MAX_CATEGORICAL_CARDINALITY = 50
    high_cardinality = {
        c: X[c].nunique(dropna=True) for c in categorical_cols
        if X[c].nunique(dropna=True) > MAX_CATEGORICAL_CARDINALITY
    }
    if high_cardinality:
        raise ValueError(
            f"configs/data.yaml's categorical_columns includes column(s) with "
            f"far too many distinct values to one-hot encode safely: "
            f"{high_cardinality} (cardinality shown per column; limit is "
            f"{MAX_CATEGORICAL_CARDINALITY}). One-hot encoding these would "
            f"attempt to allocate one column PER DISTINCT VALUE and exhaust "
            f"memory (this is almost certainly what caused an "
            f"ArrayMemoryError if you're seeing this after one). Remove "
            f"these columns from categorical_columns in configs/data.yaml -- "
            f"if they are genuinely numeric (e.g. Sport, Dport, or any port/"
            f"ID-like field), they do not belong in categorical_columns at "
            f"all; they are handled by the numeric-coercion step below "
            f"instead. The confirmed-correct categorical_columns for this "
            f"project's real GeNIS schema is exactly: State, Flgs, Proto."
        )

    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, dummy_na=True)

    # Real GeNIS data (confirmed against real combined data) contains
    # numeric-valued columns (e.g. Sport, Dport) stored as inconsistent
    # strings -- plain decimals ("138"), decimals with a trailing ".0"
    # ("35672.0"), hex-formatted codes for non-TCP/UDP protocols
    # ("0x7ce9"), and empty strings for protocols with no port at all --
    # rather than a clean numeric dtype. _coerce_numeric_with_hex handles
    # all of these. Only columns that still can't be coerced after that
    # (i.e. contain real text, not just inconsistently-formatted numbers)
    # trigger the error below. Diagnostic info (which values actually
    # failed to parse) is captured here, not just at raise time, so a
    # failure is immediately actionable instead of requiring a second
    # investigation round-trip -- this matters in practice, since a column
    # can coerce cleanly on one scenario file but fail once multiple files
    # are combined.
    still_non_numeric = X.select_dtypes(exclude=[np.number, bool]).columns.tolist()
    coercion_diagnostics: Dict[str, Dict[str, Any]] = {}
    for col in still_non_numeric:
        coerced = _coerce_numeric_with_hex(X[col])
        original_non_null_mask = X[col].notna() & (X[col].astype(str).str.strip() != "")
        original_non_null = int(original_non_null_mask.sum())
        coerced_non_null = int(coerced.notna().sum())
        success_fraction = coerced_non_null / max(original_non_null, 1) if original_non_null > 0 else 1.0

        if original_non_null == 0 or success_fraction > 0.99:
            X[col] = coerced
        else:
            # Capture WHICH values failed, so the error is actionable
            # without a second round-trip: a value failed if it was
            # genuinely present in the original column but pd.to_numeric
            # could not parse it.
            failed_mask = original_non_null_mask & coerced.isna()
            failing_values = X.loc[failed_mask, col].value_counts().head(10)
            coercion_diagnostics[col] = {
                "success_fraction": round(success_fraction, 4),
                "original_non_null_count": original_non_null,
                "coerced_non_null_count": coerced_non_null,
                "sample_failing_values": failing_values.to_dict(),
            }
            # else: leave as-is, will be caught by the check below.

    non_numeric = X.select_dtypes(exclude=[np.number, bool]).columns.tolist()
    if non_numeric:
        detail_lines = []
        for col in non_numeric:
            diag = coercion_diagnostics.get(col)
            if diag:
                detail_lines.append(
                    f"  '{col}': {diag['success_fraction']*100:.2f}% of "
                    f"{diag['original_non_null_count']} non-empty values parsed as "
                    f"numeric; most common values that did NOT parse: "
                    f"{diag['sample_failing_values']}"
                )
            else:
                detail_lines.append(f"  '{col}': (no diagnostic captured)")
        raise ValueError(
            "Columns remain non-numeric after encoding and numeric coercion: "
            f"{non_numeric}.\n" + "\n".join(detail_lines) + "\n"
            "This is almost always caused by one scenario file's export using "
            "different formatting than the others (e.g. resolved service names "
            "like 'http'/'ssh' instead of numeric ports in just one file) -- "
            "the 'sample_failing_values' above should show exactly what's "
            "different. If the failing values are genuinely non-numeric text "
            "that appears in a small, known set (like service names), add "
            "this column to configs/data.yaml's categorical_columns ONLY IF "
            "its cardinality is low (see the ArrayMemoryError guard above) -- "
            "otherwise, map the known text values to their numeric port "
            "equivalents before this step, or drop the column."
        )
    X = X.astype(float)

    logger.info(
        "Prepared feature matrix: %d rows x %d columns (label='%s').",
        X.shape[0],
        X.shape[1],
        label_col,
    )
    return X, y, scenario_id, start_time


def fit_imputer(X_train: pd.DataFrame) -> SimpleImputer:
    """Fit a median imputer on training features only.

    Real GeNIS data (confirmed against an actual scenario file) contains
    genuine missing values in otherwise-legitimate numeric columns -- e.g.
    sTos/dTos are NaN for protocols where that field doesn't apply, Sport/
    Dport are NaN for ARP flows (no ports). StandardScaler raises on NaN
    input, so imputation MUST happen before scaling, and -- per this
    project's leakage rules -- must be fit on the training split only.

    Median imputation is used rather than mean, since several of these
    columns (byte/packet counts, rates) are heavily right-skewed, and the
    median is more robust to that skew than the mean.

    Args:
        X_train: Training-split feature DataFrame (after clean_and_encode,
            before scaling).

    Returns:
        A fitted SimpleImputer.
    """
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train.values)
    return imputer


def apply_imputer(imputer: SimpleImputer, X: pd.DataFrame) -> pd.DataFrame:
    """Apply an already-fitted imputer to a feature DataFrame.

    Args:
        imputer: A SimpleImputer previously fit on a training split.
        X: Feature DataFrame to transform (train, val, or test).

    Returns:
        A new DataFrame with the same columns/index, NaN-imputed.
    """
    imputed = imputer.transform(X.values)
    return pd.DataFrame(imputed, columns=X.columns, index=X.index)


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on training features only.

    Args:
        X_train: Training-split feature DataFrame.

    Returns:
        A fitted StandardScaler.
    """
    scaler = StandardScaler()
    scaler.fit(X_train.values)
    return scaler


def apply_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    """Apply an already-fitted scaler to a feature DataFrame.

    Args:
        scaler: A StandardScaler previously fit on a training split.
        X: Feature DataFrame to transform (val, test, or train itself).

    Returns:
        A new DataFrame with the same columns/index, scaled.
    """
    scaled = scaler.transform(X.values)
    return pd.DataFrame(scaled, columns=X.columns, index=X.index)


def compute_class_weights(y_train: pd.Series) -> Dict[Any, float]:
    """Compute inverse-frequency class weights from training labels only.

    Args:
        y_train: Training-split label Series.

    Returns:
        Dict mapping each class label to its weight (higher weight for
        rarer classes), normalized so weights average to 1.0.
    """
    counts = y_train.value_counts()
    n_classes = len(counts)
    n_samples = len(y_train)
    weights = {cls: n_samples / (n_classes * count) for cls, count in counts.items()}
    return weights
