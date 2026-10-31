"""Statistical comparison tools, per Section 6 of
STAGE2_PROPOSAL_AND_METHODOLOGY.md.

Wilcoxon signed-rank (not a paired t-test) is used throughout because the
number of paired observations is small (11 LOSO folds, 5 random-split
seeds) and macro-F1 is bounded in [0, 1] and often clustered near a
ceiling, which violates the normality assumption a t-test relies on.
Rank-biserial correlation is reported as the effect size alongside every
p-value, since a significant-but-tiny effect and a large-but-marginal-p
effect carry different implications for this project's conclusions.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def wilcoxon_paired_test(
    scores_a: List[float], scores_b: List[float], alternative: str = "two-sided"
) -> Dict[str, Any]:
    """Wilcoxon signed-rank test on paired scores (e.g. per-fold macro-F1).

    Args:
        scores_a: Scores under condition A (e.g. random-split macro-F1 per
            seed, or per-model, per-fold LOSO macro-F1).
        scores_b: Scores under condition B, paired with scores_a
            element-for-element (same seed / same fold / same model).
        alternative: "two-sided", "greater", or "less" -- passed to
            scipy.stats.wilcoxon.

    Returns:
        Dict with statistic, p_value, n_pairs, rank_biserial_effect_size,
        and a warning field if any pair was tied (zero difference), which
        scipy drops by default (the "wilcox" zero-method) -- this is
        surfaced explicitly rather than silently handled.

    Raises:
        ValueError: If scores_a and scores_b have different lengths.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"scores_a and scores_b must be paired (same length): "
            f"got {len(scores_a)} vs {len(scores_b)}"
        )

    a = np.array(scores_a)
    b = np.array(scores_b)
    diffs = a - b
    n_zero = int(np.sum(diffs == 0))

    result: Dict[str, Any] = {"n_pairs": len(a), "n_tied_pairs_dropped": n_zero}

    if n_zero == len(a):
        result.update(
            {
                "statistic": None,
                "p_value": 1.0,
                "rank_biserial_effect_size": 0.0,
                "note": "All pairs tied (zero difference); no test performed.",
            }
        )
        return result

    statistic, p_value = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")

    n_effective = len(a) - n_zero
    max_w = n_effective * (n_effective + 1) / 2
    rank_biserial = 1 - (2 * statistic) / max_w if max_w > 0 else 0.0

    result.update(
        {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "rank_biserial_effect_size": float(rank_biserial),
        }
    )
    return result


def holm_bonferroni_correction(p_values: List[float]) -> Dict[str, Any]:
    """Apply Holm-Bonferroni correction to a family of p-values.

    Args:
        p_values: Uncorrected p-values from a family of related tests
            (e.g. one per model, within one research question).

    Returns:
        Dict with corrected_p_values (same order as input), reject (bool
        list at alpha=0.05), and the method name, so the manuscript can
        report both raw and corrected values explicitly rather than only
        one or the other.
    """
    reject, corrected, _, _ = multipletests(p_values, alpha=0.05, method="holm")
    return {
        "method": "holm-bonferroni",
        "alpha": 0.05,
        "raw_p_values": list(p_values),
        "corrected_p_values": [float(p) for p in corrected],
        "reject_null": [bool(r) for r in reject],
    }
