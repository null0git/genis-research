"""Generate Table 03: this project's E01 (random-split baseline) results
next to the published Silva et al. (2026) numbers, for direct comparison.

The published values below are hard-coded WITH A CITATION, exactly once,
in this script -- they are never re-typed anywhere else in the manuscript.
If Silva et al.'s numbers are revised (e.g. a published erratum), this is
the only place to update them.

Reads: results/processed/e01_metrics.json
Writes: results/processed/table_03_baseline_comparison.csv

Usage:
    python analysis/generate_table_03.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import get_repo_root  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

# Silva, M., Pinto, D., Vitorino, J., Maia, E., Praca, I., Amorim, I.,
# Viamonte, M.J. (2026). "Binary and Multiclass Cyberattack Classification
# on GeNIS Dataset." FPS 2025, LNCS 16402, Springer. (also arXiv:2511.08660)
# Values below are the binary-classification, full-feature-set results from
# their Table of results, as reported in the paper.
PUBLISHED_BASELINE_F1 = {
    "random_forest": 0.999949,
    "xgboost": 0.999949,
    # LightGBM and their non-sequential LSTM are not reproduced by this
    # project (see Section 5 of STAGE2_PROPOSAL_AND_METHODOLOGY.md for why),
    # so they are omitted here rather than left blank with no explanation.
}


def main() -> None:
    root = get_repo_root()
    e01_path = root / "results/processed/e01_metrics.json"
    if not e01_path.exists():
        raise FileNotFoundError(f"{e01_path} not found. Run experiments/e01/run.py first.")
    with open(e01_path) as f:
        e01 = json.load(f)

    rows = []
    for model_name, model_result in e01["models"].items():
        agg = model_result["aggregate"]
        rows.append(
            {
                "model": model_name,
                "this_project_macro_f1_mean": agg.get("macro_f1_mean"),
                "this_project_macro_f1_std": agg.get("macro_f1_std"),
                "silva_et_al_2026_f1": PUBLISHED_BASELINE_F1.get(model_name, "Not reported"),
                "note": (
                    "Silva et al. use a single 70/30 random split, not repeated "
                    "seeds; this project's std is not directly comparable to a "
                    "single-run number."
                ),
            }
        )

    df = pd.DataFrame(rows)
    out_path = root / "results/processed/table_03_baseline_comparison.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
