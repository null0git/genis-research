"""Generate Figure 03: macro-F1 vs. sequence length, one line per model per
split strategy.

Reads: results/processed/e05_metrics.json
Writes: paper/figures/figure_03_sequence_length.png

Usage:
    python experiments/figures/plot_sequence_length_ablation.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import get_repo_root  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

KEY_PATTERN = re.compile(r"^(random|loso)_seqlen(\d+)$")


def main() -> None:
    root = get_repo_root()
    path = root / "results/processed/e05_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run experiments/e05/run.py first.")
    with open(path) as f:
        e05 = json.load(f)

    # curves[(strategy, model_name)] = {seq_len: macro_f1}
    curves = defaultdict(dict)
    for key, sub_result in e05.items():
        m = KEY_PATTERN.match(key)
        if not m:
            logger.warning("Skipping unrecognized key in e05_metrics.json: %s", key)
            continue
        strategy, seq_len = m.group(1), int(m.group(2))
        for model_name, model_result in sub_result["models"].items():
            f1 = model_result["aggregate"]["macro_f1_mean"]
            curves[(strategy, model_name)][seq_len] = f1

    fig, ax = plt.subplots(figsize=(7, 6))
    for (strategy, model_name), points in sorted(curves.items()):
        seq_lens = sorted(points.keys())
        f1s = [points[s] for s in seq_lens]
        linestyle = "--" if strategy == "loso" else "-"
        ax.plot(seq_lens, f1s, marker="o", linestyle=linestyle, label=f"{model_name} ({strategy})")

    ax.set_xlabel("Sequence length (flows per window)")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.02)
    ax.set_title("Sequence-length ablation")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()

    out_path = root / "paper/figures/figure_03_sequence_length.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
