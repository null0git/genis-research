# EXPERIMENTS.md — Experiment Registry

Status values used below: `NOT STARTED`, `RUNNING`, `COMPLETED`, `FAILED`,
`REQUIRES REVIEW`. **Update the Status column yourself as you actually run
each experiment — every row below currently says `NOT STARTED` because no
experiment has been run against the real GeNIS dataset yet.** Do not mark
anything `COMPLETED` until `results/processed/{output_prefix}_metrics.json`
exists and you've inspected it.

---

## E01 — Baseline reproduction (random split)

- **Research question addressed:** reproduction baseline for RQ1/RQ2.
- **Hypothesis:** none (descriptive reproduction).
- **Dataset subset:** GeNIS, 60s interval, full.
- **Preprocessing:** `configs/data.yaml` defaults (drop topology columns,
  one-hot encode `State`/`Flags`/`Protocol`, `StandardScaler` fit on train).
- **Models:** Random Forest, XGBoost, MLP.
- **Hyperparameters:** `configs/models/rf.yaml`, `xgboost.yaml`, `mlp.yaml`.
- **Command:**
  ```bash
  python experiments/e01/run.py --config configs/experiments/e01_random_split_baselines.yaml \
      > results/raw/e01_terminal_output.txt 2>&1
  ```
- **Expected runtime:** minutes (Stage C scale) per README.md's compute table.
- **Output files:** `results/processed/e01_metrics.json`, `results/raw/e01_terminal_output.txt`.
- **Metrics:** macro-F1, accuracy, precision, recall, FPR (per seed, aggregated).
- **Figure generated:** none directly (feeds Figures 01/02).
- **Table generated:** Table 03 (`analysis/generate_table_03.py`), Table 04.
- **Manuscript section:** 13.1.
- **Status:** NOT STARTED

## E02 — Cross-scenario generalization (LOSO)

- **Research question addressed:** RQ1.
- **Hypothesis:** H1a / H0a (Section 5 of the manuscript).
- **Dataset subset:** GeNIS, 60s interval, full, 8 LOSO folds.
- **Models:** Random Forest, XGBoost, MLP.
- **Command:**
  ```bash
  python experiments/e02/run.py --config configs/experiments/e02_loso_baselines.yaml \
      > results/raw/e02_terminal_output.txt 2>&1
  ```
- **Expected runtime:** ~8x a single E01 run (one fit per fold), Stage D scale.
- **Output files:** `results/processed/e02_metrics.json`, `results/raw/e02_terminal_output.txt`.
- **Metrics:** per-fold macro-F1, FPR, per-class recall.
- **Figure generated:** Figure 01 (`experiments/figures/plot_rq1_generalization_gap.py`).
- **Table generated:** Table 04, Table 05.
- **Manuscript section:** 13.2, 15, 16.
- **Status:** NOT STARTED

## E01-no-torch / E02-no-torch — RF + XGBoost only (torch-blocked fallback)

- **When to use:** torch cannot be installed or fixed in your environment
  (see README.md's Troubleshooting section for what to try first). These
  are official, supported configs, not a hand-edit workaround.
- **Research question addressed:** same as E01/E02, minus the MLP data
  point — RQ1 is still fully answerable from RF + XGBoost alone; RQ2
  (which needs the sequence-aware models) is not answerable until torch
  works somewhere.
- **Commands:**
  ```bash
  python experiments/e01/run.py --config configs/experiments/e01_random_split_baselines_no_torch.yaml \
      > results/raw/e01_no_torch_terminal_output.txt 2>&1
  python experiments/e02/run.py --config configs/experiments/e02_loso_baselines_no_torch.yaml \
      > results/raw/e02_no_torch_terminal_output.txt 2>&1
  ```
- **Output files:** `results/processed/e01_no_torch_metrics.json`, `results/processed/e02_no_torch_metrics.json`.
- **Manuscript impact:** if these are the only results available when you
  write up, say so explicitly in Limitations — RQ2 and the RQ1×RQ2
  interaction remain open questions pending a working torch install, not
  silently dropped.
- **Status:** NOT STARTED

## E03 — Sequence-aware models, random split

- **Research question addressed:** RQ2 (main effect, random-split arm).
- **Models:** 1D-CNN, GRU. Sequence length fixed at 10.
- **Command:**
  ```bash
  python experiments/e03/run.py --config configs/experiments/e03_random_split_sequence_aware.yaml \
      > results/raw/e03_terminal_output.txt 2>&1
  ```
- **Expected runtime:** Stage D scale; dominated by neural network training.
- **Output files:** `results/processed/e03_metrics.json`, `results/raw/e03_terminal_output.txt`.
- **Figure generated:** Figure 02.
- **Table generated:** Table 04.
- **Manuscript section:** 13.3.
- **Status:** NOT STARTED

## E04 — Sequence-aware models, LOSO

- **Research question addressed:** RQ2 (main effect, LOSO arm) + RQ1×RQ2 interaction.
- **Hypothesis:** H1b / H0b.
- **Models:** 1D-CNN, GRU. Sequence length fixed at 10.
- **Command:**
  ```bash
  python experiments/e04/run.py --config configs/experiments/e04_loso_sequence_aware.yaml \
      > results/raw/e04_terminal_output.txt 2>&1
  ```
- **Expected runtime:** Stage D scale, largest single experiment (8 folds x 2 models x neural training).
- **Output files:** `results/processed/e04_metrics.json`, `results/raw/e04_terminal_output.txt`.
- **Figure generated:** Figure 02, Figure 04 (if GRU/CNN selected for error analysis).
- **Table generated:** Table 04, Table 05.
- **Manuscript section:** 13.3, 15, 16.
- **Status:** NOT STARTED

## E05 — Sequence-length ablation

- **Research question addressed:** RQ2 (ablation).
- **Models:** 1D-CNN, GRU at sequence lengths {5, 10, 20}, both split strategies.
- **Command:**
  ```bash
  python experiments/e05/run.py --config configs/experiments/e05_sequence_length_ablation.yaml \
      > results/raw/e05_terminal_output.txt 2>&1
  ```
- **Expected runtime:** ~6x a single E03/E04-scale run (3 lengths x 2 strategies); this is the
  first thing to cut back to a single length if Stage D compute checkpoints run over budget
  (see README.md's fallback plan). **Reduced fallback config available:**
  `configs/experiments/e05_sequence_length_ablation_reduced.yaml` (sequence length 10 only,
  cuts 78 model fits down to 26) — use this on constrained hardware (e.g. free Colab) instead
  of silently skipping E05 or quietly running a smaller grid without disclosing it. If used,
  state this explicitly in the manuscript's Limitations section.
- **Output files:** `results/processed/e05_metrics.json` (full) or `e05_reduced_metrics.json`
  (reduced), `results/raw/e05_terminal_output.txt`.
- **Figure generated:** Figure 03.
- **Table generated:** feeds Section 14 (Ablation Study) directly.
- **Manuscript section:** 13.4, 14.
- **Status:** NOT STARTED

## E06 — Temporal-split robustness check

- **Research question addressed:** secondary robustness check (not a confirmatory RQ).
- **Models:** all 5 (RF, XGBoost, MLP flow-independent; 1D-CNN, GRU sequence-aware).
- **Command:**
  ```bash
  python experiments/e06/run.py --config configs/experiments/e06_temporal_split_robustness.yaml \
      > results/raw/e06_terminal_output.txt 2>&1
  ```
- **Expected runtime:** roughly one E01 run + one E03 run combined (single split, no CV folds).
- **Output files:** `results/processed/e06_metrics.json`, `results/raw/e06_terminal_output.txt`.
- **Figure generated:** none dedicated (referenced in Section 13.5 discussion).
- **Table generated:** none dedicated; values quoted directly from JSON in Section 13.5.
- **Manuscript section:** 13.5.
- **Status:** NOT STARTED

## E07 — Deployment cost (RQ3)

- **Research question addressed:** RQ3.
- **Models:** all 5, single random split, single seed.
- **Command:**
  ```bash
  python experiments/e07/run.py --config configs/experiments/e07_deployment_cost.yaml \
      > results/raw/e07_terminal_output.txt 2>&1
  ```
- **Expected runtime:** roughly one E01-scale run plus latency-measurement overhead (100
  repeated single-flow inference calls per model).
- **Output files:** `results/processed/e07_metrics.json`, `results/raw/e07_terminal_output.txt`.
- **Metrics:** training time, inference latency (mean/std), model size, peak memory.
- **Figure generated:** none dedicated.
- **Table generated:** Table 06 (`analysis/generate_table_06.py`).
- **Manuscript section:** 17.
- **Status:** NOT STARTED

---

## Prerequisite steps (before E01)

| Step | Command | Output | Status |
|---|---|---|---|
| Download | `python -m src.data.download --config configs/data.yaml --only-preprocessed` | `data/raw/*` | NOT STARTED |
| Extract | (manual, see DATASET.md Section 4) | `data/preprocessed/*.csv` | NOT STARTED |
| Audit | `python -m src.data.audit --interval 60 --config configs/data.yaml` | `results/processed/data_audit_summary.json` | NOT STARTED |
| Correct `configs/data.yaml` against audit output | (manual) | updated `configs/data.yaml` | NOT STARTED |
| Leakage tests | `pytest tests/test_leakage.py -v` | pass/fail in terminal | **PASSED on synthetic data during development — re-run against real config before trusting real experiments** |
| Synthetic pipeline smoke test | `python scripts/make_synthetic_sample.py && python experiments/e01/run.py ...` (point config at the synthetic file) | confirms the pipeline runs end-to-end | COMPLETED (see repository development notes) |

## Post-experiment steps (after E01–E07)

| Step | Command |
|---|---|
| Generate all tables | `python analysis/generate_table_0{1,2,3,4,5,6}.py`, `generate_abstract_stats.py`, `generate_appendix_c.py` |
| Generate all figures | `python experiments/figures/plot_rq1_generalization_gap.py`, `plot_rq2_interaction.py`, `plot_sequence_length_ablation.py`, `plot_confusion_matrices.py --models ... --sources ...` |
| Run RQ1/RQ2 statistical tests | `python analysis/run_rq_statistics.py` (writes `results/processed/rq1_wilcoxon.json` and `rq2_wilcoxon.json`) |
| Run the SHAP feature-reliance comparison (Section 9/16) | Requires E01 and E02 to have been run with `save_models: true` first (see README.md's "Trained models: saved or not?"). Then: `python analysis/run_shap_comparison.py --model-name random_forest --random-seed 42 --held-out-scenario <one of your 8 scenario names> --sample-size 200` (writes `results/processed/shap_comparison_random_forest.json`) |
| Fill manuscript placeholders | follow each `[RESULT PLACEHOLDER]` / `[TABLE PLACEHOLDER]` / `[FIGURE PLACEHOLDER]` in `paper/manuscript.md` exactly as instructed inline |
