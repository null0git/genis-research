"""Audit the downloaded GeNIS 60-second flow data before any modeling runs.

This script exists because the Stage 2 methodology document is explicit
that several facts assumed in configs/data.yaml (exact column names,
scenario identifiers, per-scenario flow counts, class balance) are
placeholders inherited from the published GeNIS descriptor and the FPS 2025
benchmark paper's split, not yet measured on this project's own load of the
data. Every downstream script (preprocessing.py, splits.py, sequences.py)
assumes this audit has been run and its column_map confirmed/corrected.

Usage:
    python -m src.data.audit --interval 60 --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_yaml, get_repo_root  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def find_interval_files(preprocessed_dir: Path, interval_seconds: int) -> list[Path]:
    """Locate CSV files for a given flow-export interval.

    GeNIS's preprocessed folder structure (and exact file naming, e.g.
    whether train/holdout portions are separate files) is not assumed
    here. Matching uses a regex that requires the interval number to be
    bounded by non-digit characters (so interval=5 does not spuriously
    match a file containing "60s"), and every candidate CSV under
    preprocessed_dir is logged with its match/no-match status, so a
    silently-wrong match is visible in the log rather than hidden.

    IMPORTANT: this function intentionally does NOT try to distinguish
    "train" vs. "holdout" vs. "test" files -- if GeNIS ships the 60s
    interval as two separate files (e.g. one shuffled training portion,
    one holdout portion), BOTH are expected to match here and BOTH get
    loaded and concatenated by preprocessing.load_flows into one pool.
    This project builds its own random/LOSO/temporal splits from that
    combined pool rather than using GeNIS's shipped split -- see
    DATASET.md, "Sequence construction and the shipped train/holdout
    split", for why this is the correct approach here, not an oversight.

    Args:
        preprocessed_dir: Root directory of the preprocessed GeNIS data.
        interval_seconds: Flow interval to look for (5, 10, 30, or 60).

    Returns:
        List of matched CSV file paths.
    """
    if not preprocessed_dir.exists():
        raise FileNotFoundError(
            f"{preprocessed_dir} does not exist. Run "
            f"`python -m src.data.download --config configs/data.yaml "
            f"--only-preprocessed` first, then extract any archives into this "
            f"directory."
        )

    candidates = sorted(preprocessed_dir.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV files found anywhere under {preprocessed_dir}. Confirm "
            f"the download was extracted here, not left as a .zip."
        )

    # Interval number bounded by non-digit characters on both sides. Matched
    # against the FULL RELATIVE PATH (folder names included), not just the
    # filename -- GeNIS's real layout (confirmed against an actual download)
    # encodes the interval in the PARENT FOLDER (e.g.
    # "scenarios-60-sec/scenario-7-authtest.csv"), not the filename itself,
    # so filename-only matching would silently find nothing.
    pattern = re.compile(rf"(?<!\d){interval_seconds}(?!\d)")

    logger.info("All CSV files found under %s (%d total):", preprocessed_dir, len(candidates))
    matched = []
    for c in candidates:
        rel_path = str(c.relative_to(preprocessed_dir))
        is_match = bool(pattern.search(rel_path))
        logger.info("  [%s] %s", "MATCH" if is_match else "     ", rel_path)
        if is_match:
            matched.append(c)

    if not matched:
        logger.warning(
            "No filename matched interval=%d among %d CSVs found under %s. "
            "Falling back to ALL discovered CSVs -- this almost certainly mixes "
            "multiple flow intervals together, which is WRONG for this project "
            "(60s-interval-only scope). Inspect the [MATCH]/[ ] list printed "
            "above and fix this function's matching pattern or GeNIS's actual "
            "file layout before trusting anything downstream.",
            interval_seconds,
            len(candidates),
            preprocessed_dir,
        )
        return candidates

    logger.info(
        "%d file(s) matched interval=%d. If GeNIS ships this interval as "
        "separate train/holdout files, BOTH are expected here and will be "
        "combined into one pool by preprocessing.load_flows (see this "
        "function's docstring and DATASET.md for why that's correct for "
        "this project's own re-splitting design, not a bug).",
        len(matched),
        interval_seconds,
    )
    return matched


def audit_files(files: list[Path], column_map: Dict[str, str]) -> Dict[str, Any]:
    """Inspect one or more flow CSVs and summarize schema and label balance.

    Args:
        files: CSV files to inspect.
        column_map: The project's expected column_map from configs/data.yaml,
            checked against what is actually present in each file.

    Returns:
        A JSON-serializable summary dict.
    """
    summary: Dict[str, Any] = {"files": []}
    total_rows = 0

    for path in files:
        logger.info("Auditing %s", path)
        df = pd.read_csv(path, nrows=None, low_memory=False)
        n_rows = len(df)
        total_rows += n_rows

        missing_expected = [
            v for v in column_map.values() if v and v not in df.columns
        ]
        file_summary: Dict[str, Any] = {
            "path": str(path),
            "n_rows": n_rows,
            "n_columns": df.shape[1],
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "missing_expected_columns": missing_expected,
        }

        binary_col = column_map.get("binary_label")
        if binary_col in df.columns:
            file_summary["binary_label_counts"] = (
                df[binary_col].value_counts(dropna=False).to_dict()
            )
        category_col = column_map.get("category_label")
        if category_col in df.columns:
            file_summary["category_label_counts"] = (
                df[category_col].value_counts(dropna=False).to_dict()
            )
        scenario_col = column_map.get("scenario_id")
        if scenario_col and scenario_col in df.columns:
            counts = df[scenario_col].value_counts(dropna=False).to_dict()
            file_summary["scenario_counts"] = counts
            file_summary["scenario_source"] = "explicit_column"
            n_distinct = len(counts)
            if n_distinct >= 8:
                verdict = "LIKELY FULL SCENARIO GRANULARITY (>=8 distinct values, matches the confirmed 8-scenario structure) -- proceed as designed."
            elif 2 <= n_distinct < 8:
                verdict = (
                    f"ONLY {n_distinct} DISTINCT VALUES -- likely too coarse for the "
                    f"project's 8-fold LOSO design (probably CategoryLabel-level, not "
                    f"true per-scenario). See DATASET.md, Section 8, for what to do next."
                )
            else:
                verdict = "SUSPICIOUS: fewer than 2 distinct values -- this column is probably not real scenario identity."
            file_summary["scenario_granularity_verdict"] = verdict
        else:
            # CONFIRMED design (see DATASET.md Section 8): GeNIS's real files
            # have no explicit scenario column, and configs/data.yaml
            # intentionally sets scenario_id to null for exactly this reason
            # -- src/data/preprocessing.py::derive_scenario_column falls
            # back to using each file's own filename as its scenario
            # identity, which is the mechanism this project actually uses
            # and which has been confirmed working end-to-end on real data.
            # This is NOT a failure case; report it as the expected,
            # working fallback rather than an error, and use the filename
            # itself as this file's single scenario value so the COMBINED
            # cross-file check in main() can still verify overall
            # granularity (N distinct filenames = N candidate scenarios).
            derived_scenario_name = path.stem
            file_summary["scenario_counts"] = {derived_scenario_name: n_rows}
            file_summary["scenario_source"] = "derived_from_filename"
            file_summary["scenario_granularity_verdict"] = (
                f"No explicit scenario column configured (column_map.scenario_id "
                f"is null) -- this is EXPECTED for GeNIS's real file layout, not "
                f"an error. Scenario identity for this file is derived from its "
                f"filename: '{derived_scenario_name}'. See the combined verdict "
                f"below for whether this holds up across all loaded files."
            )

        n_missing = int(df.isna().sum().sum())
        file_summary["total_missing_values"] = n_missing
        n_duplicate_rows = int(df.duplicated().sum())
        file_summary["duplicate_rows"] = n_duplicate_rows

        summary["files"].append(file_summary)

    summary["total_rows_across_files"] = total_rows

    # Combined cross-file scenario granularity: a single train file might
    # only show a subset of scenarios (e.g. if one scenario landed entirely
    # in the holdout portion), so the real check is the union across every
    # file this audit loaded, not any one file in isolation.
    all_scenario_values: set = set()
    any_scenario_column = False
    any_filename_derived = False
    for f in summary["files"]:
        if "scenario_counts" in f:
            any_scenario_column = True
            if f.get("scenario_source") == "derived_from_filename":
                any_filename_derived = True
            all_scenario_values.update(f["scenario_counts"].keys())

    source_note = (
        " (derived from filename, per the CONFIRMED design for GeNIS's real "
        "file layout -- this is expected, not a fallback failure)"
        if any_filename_derived
        else ""
    )

    if not any_scenario_column:
        combined_verdict = (
            "NO SCENARIO COLUMN FOUND IN ANY FILE, and no filename fallback "
            "was applied either -- this should not happen given the current "
            "code; check that configs/data.yaml loaded correctly. See "
            "DATASET.md, Section 8."
        )
    elif len(all_scenario_values) >= 8:
        combined_verdict = (
            f"{len(all_scenario_values)} distinct scenario values found across all "
            f"loaded files{source_note} -- matches the confirmed 8-scenario "
            f"structure. Proceed as designed."
        )
    else:
        combined_verdict = (
            f"Only {len(all_scenario_values)} distinct scenario value(s) found "
            f"across all loaded files{source_note} -- expected 8 (one per real "
            f"GeNIS scenario file: scenario-1-disrupt ... scenario-8-authtest). "
            f"If you're missing some, confirm all 8 files from "
            f"3-scenarios.zip/scenarios-60-sec/ were actually extracted into "
            f"preprocessed_data_dir -- see DATASET.md Section 8."
        )
    summary["combined_scenario_granularity_verdict"] = combined_verdict
    summary["combined_distinct_scenario_values"] = sorted(all_scenario_values)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--output",
        default="results/processed/data_audit_summary.json",
        help="Where to write the JSON summary (relative to repo root unless absolute).",
    )
    args = parser.parse_args()

    data_cfg = load_yaml(args.config)
    preprocessed_dir = get_repo_root() / data_cfg["preprocessed_data_dir"]

    files = find_interval_files(preprocessed_dir, args.interval)
    logger.info("Found %d candidate file(s) for the %ds interval.", len(files), args.interval)
    for f in files:
        logger.info("  candidate: %s", f)

    summary = audit_files(files, data_cfg["column_map"])

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = get_repo_root() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info("Audit summary written to %s", out_path)
    logger.info("SCENARIO GRANULARITY VERDICT: %s", summary["combined_scenario_granularity_verdict"])

    # Only sound an alarm if something is actually wrong -- a previous
    # version of this message printed unconditionally, which confused at
    # least one real user into thinking a successful "proceed as designed"
    # verdict meant something needed fixing. Check the two things that
    # actually matter before deciding which message to show.
    files_with_missing_columns = [
        f["path"] for f in summary["files"] if f.get("missing_expected_columns")
    ]
    scenario_granularity_ok = len(summary["combined_distinct_scenario_values"]) >= 8

    if not files_with_missing_columns and scenario_granularity_ok:
        logger.info(
            "ALL CHECKS PASSED. No missing expected columns in any file, and "
            "scenario granularity matches the confirmed 8-scenario structure. "
            "No changes to configs/data.yaml are needed -- you can proceed to "
            "preprocessing / running an experiment."
        )
    else:
        problems = []
        if files_with_missing_columns:
            problems.append(
                f"{len(files_with_missing_columns)} file(s) are missing an expected "
                f"column (see missing_expected_columns per file in the JSON summary): "
                f"{files_with_missing_columns}"
            )
        if not scenario_granularity_ok:
            problems.append(
                f"scenario granularity looks too coarse "
                f"({len(summary['combined_distinct_scenario_values'])} distinct values "
                f"found, expected 8) -- see the combined_scenario_granularity_verdict "
                f"above for detail."
            )
        logger.warning(
            "ACTION REQUIRED -- %s. Correct configs/data.yaml's column_map / "
            "drop_columns / scenarios list to match what was actually found "
            "(see the JSON summary at %s), and re-run this audit until this "
            "message no longer appears, before running any preprocessing or "
            "experiment script.",
            "; ".join(problems),
            out_path,
        )


if __name__ == "__main__":
    main()
