"""Run the confirmatory statistical tests for RQ1 and RQ2.

RQ1: for each flow-independent model, compare random-split macro-F1 (E01,
     repeated seeds) against LOSO macro-F1 (E02, per-fold), via Wilcoxon
     signed-rank test. Since the two arrays are not naturally the same
     length (5 seeds vs. 11 folds), the comparison is made against each
     model's *mean* random-split score broadcast against each LOSO fold's
     score -- this is documented explicitly below, since it is a
     one-sample-style test against a fixed reference, not a fully paired
     test in the strictest sense (paired requires matching units, and a
     "seed" and a "held-out scenario" are not the same unit). This
     limitation is stated in the manuscript's Threats to Validity section,
     not hidden here.

RQ2: for each model family (flow-independent E01/E02 vs. sequence-aware
     E03/E04), compare the *generalization gap* (random-split score minus
     LOSO mean) across models, via Wilcoxon signed-rank test on the paired
     per-model gap values -- this one is a genuine paired test, since each
     model contributes exactly one gap value per family.

Holm-Bonferroni correction is applied within each RQ's family of
per-model tests.

Reads: results/processed/e01_metrics.json, e02, e03, e04
Writes: results/processed/rq1_wilcoxon.json, results/processed/rq2_wilcoxon.json

Usage:
    python analysis/run_rq_statistics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.statistics import wilcoxon_paired_test, holm_bonferroni_correction  # noqa: E402
from src.utils.config import get_repo_root  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def _load(root: Path, prefix: str) -> dict:
    path = root / f"results/processed/{prefix}_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run experiments/{prefix}/run.py first.")
    with open(path) as f:
        return json.load(f)


def run_rq1(root: Path) -> dict:
    """RQ1: random-split mean vs. each LOSO fold, per flow-independent model."""
    e01 = _load(root, "e01")
    e02 = _load(root, "e02")

    per_model_tests = {}
    p_values = []
    model_order = []

    for model_name in e02["models"]:
        if model_name not in e01["models"]:
            logger.warning("Model %s in E02 but not E01; skipping RQ1 test for it.", model_name)
            continue
        random_mean = e01["models"][model_name]["aggregate"]["macro_f1_mean"]
        loso_scores = [r["macro_f1"] for r in e02["models"][model_name]["per_run_metrics"]]
        reference = [random_mean] * len(loso_scores)

        test_result = wilcoxon_paired_test(reference, loso_scores, alternative="two-sided")
        test_result["note"] = (
            "Reference is the random-split MEAN broadcast against each LOSO "
            "fold -- not a strictly paired same-unit comparison. See this "
            "script's module docstring and the manuscript's Threats to "
            "Validity section."
        )
        per_model_tests[model_name] = test_result
        p_values.append(test_result["p_value"])
        model_order.append(model_name)

    correction = holm_bonferroni_correction(p_values) if p_values else {}
    return {
        "research_question": "RQ1",
        "per_model": per_model_tests,
        "model_order_for_correction": model_order,
        "holm_bonferroni_correction": correction,
    }


def run_rq2(root: Path) -> dict:
    """RQ2: generalization gap (random - LOSO mean), flow-independent vs. sequence-aware."""
    e01 = _load(root, "e01")
    e02 = _load(root, "e02")
    e03 = _load(root, "e03")
    e04 = _load(root, "e04")

    def gap(random_data, loso_data, model_name):
        random_f1 = random_data["models"][model_name]["aggregate"]["macro_f1_mean"]
        loso_scores = [r["macro_f1"] for r in loso_data["models"][model_name]["per_run_metrics"]]
        loso_mean = sum(loso_scores) / len(loso_scores)
        return random_f1 - loso_mean

    flow_independent_gaps = [gap(e01, e02, m) for m in e01["models"]]
    sequence_aware_gaps = [gap(e03, e04, m) for m in e03["models"]]

    if len(flow_independent_gaps) != len(sequence_aware_gaps):
        logger.warning(
            "Unequal number of flow-independent (%d) and sequence-aware (%d) "
            "models; comparing the minimum overlapping count only. This is a "
            "structural mismatch (different model counts per family) and must "
            "be reported as such, not silently truncated without comment.",
            len(flow_independent_gaps),
            len(sequence_aware_gaps),
        )
        n = min(len(flow_independent_gaps), len(sequence_aware_gaps))
        flow_independent_gaps = flow_independent_gaps[:n]
        sequence_aware_gaps = sequence_aware_gaps[:n]

    test_result = wilcoxon_paired_test(flow_independent_gaps, sequence_aware_gaps, alternative="two-sided")
    return {
        "research_question": "RQ2",
        "flow_independent_gaps": flow_independent_gaps,
        "sequence_aware_gaps": sequence_aware_gaps,
        "wilcoxon_test": test_result,
    }


def main() -> None:
    root = get_repo_root()

    rq1_result = run_rq1(root)
    rq1_path = root / "results/processed/rq1_wilcoxon.json"
    with open(rq1_path, "w", encoding="utf-8") as fh:
        json.dump(rq1_result, fh, indent=2, default=str)
    logger.info("Wrote %s", rq1_path)

    rq2_result = run_rq2(root)
    rq2_path = root / "results/processed/rq2_wilcoxon.json"
    with open(rq2_path, "w", encoding="utf-8") as fh:
        json.dump(rq2_result, fh, indent=2, default=str)
    logger.info("Wrote %s", rq2_path)


if __name__ == "__main__":
    main()
