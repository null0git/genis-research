"""Generate Figure 04: confusion matrix heatmaps.

Per Section 16 of the manuscript, this plots confusion matrices for
whichever models the error analysis identifies as the best- and
worst-generalizing under LOSO -- pass them explicitly via --models rather
than the script guessing, so the figure always matches what the text says.

Reads: results/processed/e02_metrics.json (flow-independent, LOSO) and/or
       results/processed/e04_metrics.json (sequence-aware, LOSO)
Writes: paper/figures/figure_04_confusion_matrices.png

Usage:
    python experiments/figures/plot_confusion_matrices.py --models random_forest gru \
        --sources e02 e04
"""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="Model names to plot, e.g. random_forest gru")
    parser.add_argument("--sources", nargs="+", required=True, help="Metrics file prefixes, one per model, e.g. e02 e04")
    args = parser.parse_args()

    if len(args.models) != len(args.sources):
        raise ValueError("--models and --sources must have the same length (one source per model).")

    root = get_repo_root()
    fig, axes = plt.subplots(1, len(args.models), figsize=(6 * len(args.models), 5))
    if len(args.models) == 1:
        axes = [axes]

    for ax, model_name, source in zip(axes, args.models, args.sources):
        path = root / f"results/processed/{source}_metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run the corresponding experiment first.")
        with open(path) as f:
            data = json.load(f)

        runs = data["models"][model_name]["per_run_metrics"]
        # Sum confusion matrices across LOSO folds / seeds for an aggregate view.
        labels = runs[0]["labels"]
        cm_sum = np.zeros((len(labels), len(labels)), dtype=int)
        for r in runs:
            cm_sum += np.array(r["confusion_matrix"])

        sns.heatmap(
            cm_sum, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax, cbar=False
        )
        ax.set_title(f"{model_name} ({source})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.suptitle("Aggregated confusion matrices across folds/seeds")
    fig.tight_layout()

    out_path = root / "paper/figures/figure_04_confusion_matrices.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
