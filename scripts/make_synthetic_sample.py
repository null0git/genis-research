"""Generate a small synthetic flow dataset with GeNIS-like structure, for
local pipeline testing (Stage A of the computational staging plan, Section
7 of STAGE2_PROPOSAL_AND_METHODOLOGY.md) without needing the real
(large) GeNIS download.

THIS DATA IS NOT REAL GENIS DATA. It exists only to verify the pipeline
runs end-to-end (loading, preprocessing, splitting, sequencing, training,
evaluation) without errors, before pointing the same code at the real
dataset. No number produced from this synthetic data may appear in the
manuscript.

Usage:
    python scripts/make_synthetic_sample.py --output data/preprocessed/synthetic_60s.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

SCENARIOS = {
    "recon_bruteforce_ftp": "attack",
    "recon_bruteforce_web": "attack",
    "recon_bruteforce_ad": "attack",
    "dos_ftp": "attack",
    "dos_web": "attack",
    "dos_ad": "attack",
    "multistage_1": "attack",
    "multistage_2": "attack",
    "user_activity": "benign",
    "admin_activity": "benign",
    "background_idle": "benign",
}

STATES = ["CON", "FIN", "RST", "REQ"]
FLAGS = ["FSPA", "S", "FA", "PA"]
PROTOCOLS = ["tcp", "udp", "icmp"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/preprocessed/synthetic_60s.csv")
    parser.add_argument("--n-per-scenario", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    frames = []

    for scenario_idx, (scenario, kind) in enumerate(SCENARIOS.items()):
        n = args.n_per_scenario
        is_attack = kind == "attack"

        data = {
            "Dur": rng.exponential(scale=2.0 + scenario_idx, size=n),
            "TotPkts": rng.poisson(lam=10 + 5 * is_attack, size=n),
            "TotBytes": rng.poisson(lam=1000 + 500 * is_attack, size=n),
            "SrcBytes": rng.poisson(lam=500, size=n),
            "DstBytes": rng.poisson(lam=500, size=n),
            "Rate": rng.exponential(scale=1.0, size=n),
            "State": rng.choice(STATES, size=n),
            "Flags": rng.choice(FLAGS, size=n),
            "Protocol": rng.choice(PROTOCOLS, size=n),
            "SrcIP": [f"10.0.{scenario_idx}.{i % 250}" for i in range(n)],
            "DstIP": [f"10.0.{(scenario_idx + 1) % 10}.{i % 250}" for i in range(n)],
            "FlowID": [f"{scenario}_{i}" for i in range(n)],
        }
        # Binary label: attacks mostly malicious (1), benign scenarios mostly benign (0),
        # with some label noise so the synthetic classification task is not trivial.
        base_label = 1 if is_attack else 0
        noise = rng.random(n) < 0.1
        data["BinaryLabel"] = np.where(noise, 1 - base_label, base_label)
        data["CategoryLabel"] = np.where(
            data["BinaryLabel"] == 1,
            rng.choice(["bruteforce", "dos", "recon"], size=n),
            "benign",
        )
        data["Scenario"] = scenario
        data["StartTime"] = np.arange(n) * 60 + scenario_idx * 1_000_000

        frames.append(pd.DataFrame(data))

    df = pd.concat(frames, ignore_index=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Wrote synthetic sample: %d rows -> %s", len(df), out_path)
    logger.warning("This is SYNTHETIC data for pipeline testing only -- not real GeNIS data.")


if __name__ == "__main__":
    main()
