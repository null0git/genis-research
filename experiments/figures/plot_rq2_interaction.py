"""Generate Figure 02: interaction plot -- split strategy (x-axis) vs.
macro-F1 (y-axis), one line per model, sequence-aware models shown dashed.

Reads: results/processed/e01_metrics.json, e02, e03, e04
Writes: paper/figures/figure_02_interaction.png

Usage:
    python experiments/figures/plot_rq2_interaction.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.training.train import SEQUENCE_AWARE_FAMILIES  # noqa: E402
from src.utils.config import get_repo_root  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the corresponding experiments/eXX/run.py first.")
    with open(path) as f:
        return json.load(f)


def main() -> None:
    root = get_repo_root()
    e01 = _load(root / "results/processed/e01_metrics.json")  # flow-independent, random
    e02 = _load(root / "results/processed/e02_metrics.json")  # flow-independent, LOSO
    e03 = _load(root / "results/processed/e03_metrics.json")  # sequence-aware, random
    e04 = _load(root / "results/processed/e04_metrics.json")  # sequence-aware, LOSO

    fig, ax = plt.subplots(figsize=(7, 6))
    x_labels = ["Random split", "Leave-one-scenario-out"]
    x_positions = [0, 1]

    for model_name in e01["models"]:
        random_f1 = e01["models"][model_name]["aggregate"]["macro_f1_mean"]
        loso_scores = [r["macro_f1"] for r in e02["models"][model_name]["per_run_metrics"]]
        loso_mean = sum(loso_scores) / len(loso_scores)
        ax.plot(x_positions, [random_f1, loso_mean], marker="o", linestyle="-", label=f"{model_name} (flow-independent)")

    for model_name in e03["models"]:
        random_f1 = e03["models"][model_name]["aggregate"]["macro_f1_mean"]
        loso_scores = [r["macro_f1"] for r in e04["models"][model_name]["per_run_metrics"]]
        loso_mean = sum(loso_scores) / len(loso_scores)
        ax.plot(x_positions, [random_f1, loso_mean], marker="s", linestyle="--", label=f"{model_name} (sequence-aware)")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.02)
    ax.set_title("Split strategy x model family interaction (RQ1 x RQ2)")
    ax.legend(loc="lower left", fontsize=8, ncol=1)
    fig.tight_layout()

    out_path = root / "paper/figures/figure_02_interaction.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
