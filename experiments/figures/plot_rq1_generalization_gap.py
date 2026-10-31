"""Generate Figure 01: macro-F1 across LOSO folds, per model, with the
random-split mean shown as a reference line.

Reads: results/processed/e01_metrics.json, results/processed/e02_metrics.json
Writes: paper/figures/figure_01_generalization_gap.png

Usage:
    python experiments/figures/plot_rq1_generalization_gap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import get_repo_root  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    root = get_repo_root()
    e01_path = root / "results/processed/e01_metrics.json"
    e02_path = root / "results/processed/e02_metrics.json"

    for p in (e01_path, e02_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run experiments/e01/run.py and "
                f"experiments/e02/run.py first."
            )

    with open(e01_path) as f:
        e01 = json.load(f)
    with open(e02_path) as f:
        e02 = json.load(f)

    models = list(e02["models"].keys())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        loso_scores = [r["macro_f1"] for r in e02["models"][model_name]["per_run_metrics"]]
        random_mean = e01["models"][model_name]["aggregate"]["macro_f1_mean"]

        sns.stripplot(y=loso_scores, ax=ax, size=8, color="steelblue", jitter=0.05)
        sns.boxplot(y=loso_scores, ax=ax, width=0.3, boxprops=dict(alpha=0.3), showfliers=False)
        ax.axhline(random_mean, color="firebrick", linestyle="--", linewidth=2, label="Random-split mean")
        ax.set_title(model_name)
        ax.set_xlabel("Leave-one-scenario-out fold")
        ax.set_ylabel("Macro-F1")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("Macro-F1: leave-one-scenario-out folds vs. random-split mean")
    fig.tight_layout()

    out_path = root / "paper/figures/figure_01_generalization_gap.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
