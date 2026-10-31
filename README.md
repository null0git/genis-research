# Temporal Context and Cross-Scenario Generalization in Machine-Learning-Based Network Intrusion Detection Using the GeNIS 2025 Dataset

**Status: research infrastructure complete; no experiments have been run
against the real GeNIS dataset yet.** Every script here has been
syntax-checked and, where possible without the real dataset, smoke-tested
against a synthetic sample (see "What has actually been tested" below).
The manuscript (`paper/manuscript.md`) contains explicit placeholders for
every number, figure, and table — nothing in it is a fabricated result.

## Research question

Does cross-scenario evaluation (holding out entire attack scenarios,
rather than shuffling flows) reveal a generalization gap in
machine-learning network intrusion detection that a random flow split
hides — and does giving models explicit multi-flow temporal context
change that gap? See `STAGE2_PROPOSAL_AND_METHODOLOGY.md` for the full
research design, and `docs/STAGE1_LITERATURE_AND_TOPIC.md` for the
literature review that motivated it.

## Motivation and research gap

The only published machine-learning benchmark on GeNIS (Silva et al.,
2026) uses a single random 70/30 split and explicitly does not give its
LSTM a real sequence of flows. No published work evaluates GeNIS under a
scenario-based split or with genuine multi-flow sequence models. This
project closes both gaps together, since they interact — see Section 4 of
`paper/manuscript.md` for the full statement of the gap and its
verification scope.

## Contributions

- **Scientific:** an empirical answer to whether cross-scenario evaluation
  changes measured detection performance on GeNIS, and whether that
  interacts with temporal (sequence-aware) modeling.
- **Methodological:** a leakage-safe protocol for scenario-aware splitting
  and flow-sequence construction on GeNIS.
- **Engineering:** this full, open pipeline.
- **Reproducibility:** every reported number traces to a specific script,
  config, and raw output file (see "Result provenance" below).

## Dataset

**GeNIS (GECAD Network Intrusion Scenarios)**, 60-second flow interval
only. Zenodo DOI `10.5281/zenodo.14919237`. See `DATASET.md` for the full
acquisition, verification, and citation protocol.

```
@article{silva2025genis,
  author  = {Silva, M. and Pinto, D. and Vitorino, J. and Gon{\c{c}}alves, J. and Maia, E. and Pra{\c{c}}a, I.},
  title   = {GeNIS: A modular dataset for network intrusion detection and classification},
  journal = {Data in Brief},
  volume  = {60}, pages = {111487}, year = {2025},
  doi     = {10.1016/j.dib.2025.111487}
}
```

## Repository structure

```
genis-temporal-generalization/
├── README.md                         <- you are here
├── STAGE2_PROPOSAL_AND_METHODOLOGY.md <- full research design/methodology
├── docs/STAGE1_LITERATURE_AND_TOPIC.md <- literature review, topic selection
├── EXPERIMENTS.md                    <- experiment registry, commands, status
├── ENVIRONMENT.md                    <- version pinning, reproducibility notes
├── DATASET.md                        <- dataset acquisition/verification
├── LICENSE                           <- MIT (code only, see note inside)
├── references.bib
├── requirements.txt
├── configs/                          <- data.yaml, models/*.yaml, experiments/*.yaml
├── src/
│   ├── data/                         <- download, audit, preprocessing, splits, sequences
│   ├── models/                       <- tree_models, mlp, cnn1d, gru, torch_common
│   ├── training/train.py             <- core orchestration, used by every experiments/eXX/run.py
│   ├── evaluation/                   <- metrics, statistics, explain (SHAP)
│   └── utils/                        <- seeding, logging, config
├── experiments/
│   ├── e01 .. e07/run.py             <- one entry point per experiment (see EXPERIMENTS.md)
│   └── figures/                      <- one script per manuscript figure
├── analysis/                         <- one script per manuscript table + stats + abstract numbers
├── tests/test_leakage.py             <- automated leakage/correctness tests (passing, see below)
├── scripts/make_synthetic_sample.py  <- synthetic data for pipeline smoke-testing (NOT real GeNIS)
├── data/reference/                   <- genis-features.csv, genis-ground-truth.csv (official GeNIS metadata, small, version-controlled)
├── results/{raw,processed,figures}/  <- untouched terminal output / clean metrics / plots
└── paper/manuscript.md               <- full manuscript with explicit result/figure/table placeholders
```

## Installation

```bash
git clone https://github.com/null0git/genis-research.git
cd genis-research
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

See `ENVIRONMENT.md` for exact versions and what to record on your own
machine before publishing final results.

## Full reproduction protocol

```bash
# 1. Download 3-scenarios.zip from the Zenodo record (CONFIRMED correct
#    source -- NOT 4-preprocessed, which strips timing columns and ships
#    pre-shuffled; see DATASET.md Section 8 for the full verification).
#    src.data.download's --only-preprocessed flag currently targets
#    4-preprocessed-style filenames -- for now, download 3-scenarios.zip
#    manually from https://zenodo.org/records/14919237 (see DATASET.md
#    Section 3).

# 2. Extract ONLY the scenarios-60-sec/ folder from 3-scenarios.zip into
#    data/preprocessed/ (skip scenarios-5-sec/, -10-sec/, -30-sec/ --
#    not needed for this project's 60s-only scope). You should end up
#    with 8 files: data/preprocessed/scenario-1-disrupt.csv ...
#    scenario-8-authtest.csv (or nested one level under
#    data/preprocessed/scenarios-60-sec/ -- both layouts work, audit.py
#    searches recursively).

# 3. Audit the real data BEFORE anything else -- confirms column names,
#    scenario identifiers, and flow counts, and updates configs/data.yaml
#    if anything doesn't match the defaults.
python -m src.data.audit --interval 60 --config configs/data.yaml
# --> inspect results/processed/data_audit_summary.json
# --> correct configs/data.yaml's column_map / drop_columns / scenarios
#     block until missing_expected_columns is empty for every file

# 4. Run the leakage test suite -- must pass before trusting any experiment.
pytest tests/test_leakage.py -v

# 5. Run each experiment (see EXPERIMENTS.md for exact commands / expected
#    runtimes / what each one answers). Example:
python experiments/e01/run.py --config configs/experiments/e01_random_split_baselines.yaml \
    > results/raw/e01_terminal_output.txt 2>&1
# ... repeat for e02 through e07 ...

# 6. Run the RQ1/RQ2 statistical tests
python analysis/run_rq_statistics.py

# 7. Generate every table
python analysis/generate_table_01.py
python analysis/generate_table_02.py
python analysis/generate_table_03.py
python analysis/generate_table_04.py
python analysis/generate_table_05.py --source e02
python analysis/generate_table_06.py
python analysis/generate_abstract_stats.py
python analysis/generate_appendix_c.py

# 8. Generate every figure
python experiments/figures/plot_rq1_generalization_gap.py
python experiments/figures/plot_rq2_interaction.py
python experiments/figures/plot_sequence_length_ablation.py
python experiments/figures/plot_confusion_matrices.py --models random_forest gru --sources e02 e04

# 9. Fill every [RESULT PLACEHOLDER] / [FIGURE PLACEHOLDER] / [TABLE PLACEHOLDER]
#    in paper/manuscript.md using the files generated above -- follow each
#    placeholder's own inline instructions exactly; never hand-type a number
#    that a script already generated.
```

## What has actually been tested (read this before trusting anything above)

This codebase was written and checked in a sandboxed environment **without
network access and without `xgboost`/`torch`/`shap`/`statsmodels`
installed**, so testing splits into two phases: synthetic-data testing done
before any real GeNIS file was available, and real-data testing done once
you shared actual uploaded files from the dataset.

**Synthetic-data testing (proves the pipeline's *logic* is correct):**
- Every `.py` file compiles (`python -m py_compile`, zero syntax errors).
- `tests/test_leakage.py`'s 9 tests all pass, exercising: random-split
  disjointness, bad-fraction rejection, LOSO scenario isolation, the LOSO
  leakage validator correctly raising on a deliberately-broken split,
  temporal-split boundary/ordering correctness, sequence construction
  never crossing a scenario boundary (checked position-by-position),
  sequence labels matching the last flow in each window, the scaler being
  fit on training data only, and scenario-validation correctly rejecting
  an unexpected scenario identifier.
- Full synthetic end-to-end runs of E01 (random split) and E02 (LOSO)
  completed successfully with Random Forest.

**Real-data testing (proves the pipeline works on GeNIS itself, not just
on data shaped like it) — done against actual uploaded files:**
`genis-60-sec-test.csv` (from `4-preprocessed`) and `scenario-7-authtest.csv`
(from `3-scenarios/scenarios-60-sec/`, the confirmed correct source — see
`DATASET.md` Section 8 for the full story of how this was determined,
including two real bugs this process caught and fixed: `find_interval_files`
matching only filenames when GeNIS actually encodes the flow interval in
the *folder* name, and `IdleTime` silently containing a raw epoch
timestamp on every row rather than the small duration its name implies).
Specifically confirmed on real data:
- `configs/data.yaml`'s `drop_columns`/`categorical_columns` are not
  guessed — they were derived by programmatically diffing the real
  125-raw-column schema against Silva et al. (2026)'s published
  87-column schema.
- `StartTime` parses correctly from its real `HH:MM:SS` format and is
  confirmed monotonically ordered in-file (`_parse_start_time`, zero
  parse failures on 29,679 real rows).
- The full `clean_and_encode` → impute → scale → split → train pipeline
  ran successfully end-to-end on a real scenario file with Random Forest
  (this also caught and fixed a real crash: several legitimate numeric
  columns, e.g. `sTos`/`dTos`, contain genuine missing values that
  `StandardScaler` cannot handle without imputation first).
- `sequences.py::build_sequences` ran successfully on the real,
  genuinely time-ordered data, producing exactly the expected padding
  pattern (verified position-by-position, not just checked for the
  absence of a crash).

**Still NOT tested anywhere** (no network / missing packages in this
sandbox): the actual download against the live Zenodo API; XGBoost, MLP,
1D-CNN, or GRU training (these require `xgboost`/`torch`, which could not
be installed offline — the code compiles and follows the same interface
as the tested Random Forest path, but has not been executed); the SHAP
explainability module; the figure-plotting scripts (need real metrics
files as input, which don't exist yet); a full multi-scenario LOSO run
against real data (only one real scenario file has been tested so far —
LOSO mechanics themselves are proven correct via the synthetic tests
above, but running all 8 real files together has not been done).

**Before you rely on this for a real submission:** install the full
`requirements.txt` on a normal internet-connected machine, run
`pytest tests/test_leakage.py -v` again there, then run
`scripts/make_synthetic_sample.py` once more and do a full synthetic
dry run of E01–E07 (a few minutes, all models) before pointing anything at
the real 8-scenario download — this catches any environment-specific
issue (e.g. a torch/CUDA mismatch) cheaply, before a multi-hour real run.

## Computational requirements

| Stage | Purpose | Data | RAM | Storage | Runtime |
|---|---|---|---|---|---|
| A | Pipeline dev | 1 scenario pair, ~50k flows (or `scripts/make_synthetic_sample.py`) | 4–8 GB | 1–2 GB | minutes |
| B | Split/sequence validation | 3–4 scenarios, ~300k flows | 8 GB | 3–5 GB | tens of minutes |
| C | Full 60s data, flow-independent only | all 8 scenarios | 8–16 GB | 5–10 GB | <1 hr/model, CPU |
| D | Full sequence-aware + LOSO grid | all 8 scenarios | 16 GB recommended | 10–15 GB | several hours, CPU; no GPU required |

Full detail and fallback plan if hardware is insufficient (what to cut
first): `STAGE2_PROPOSAL_AND_METHODOLOGY.md`, Section 7.

## Trained models: saved or not?

**By default, no.** Every experiment computes and saves metrics
(`results/processed/*.json`) and discards the fitted model object once
that's done — analysis and figures are built entirely from the JSON, not
from any saved model file. This is the right default: most of this
project's questions (macro-F1, generalization gap, etc.) only need the
*scores*, and saving all 153 model fits across every experiment would use
real disk for no benefit (confirmed size: a single Random Forest at this
project's settings is ~4.3 MB — multiply by however many fits you'd
otherwise save).

**One exception needs an actual saved model, not just its score:** the
SHAP explainability comparison (manuscript Section 16 / methodology
Section 9) compares what features a random-split-trained model relies on
against a LOSO-trained version of the *same* model — that needs both
actual fitted models available, not their metrics. Enable saving only
where you need it, by adding `save_models: true` to a specific experiment
config (already commented as available in every experiment's YAML
schema; add the line yourself where needed — it isn't on by default in
any shipped config, including E01/E02, precisely so nobody pays this cost
without deciding to):

```yaml
# add this one line to e.g. configs/experiments/e01_random_split_baselines.yaml
save_models: true
```

Models are written to `results/models/{output_prefix}/{model_name}_{seed_or_fold}.{pkl,pt}`
— `.pkl` for Random Forest/XGBoost, `.pt` (a torch checkpoint dict, not a
raw `torch.save(model)`) for MLP/1D-CNN/GRU. Reload with
`src/models/persistence.py::load_model` (needs `n_features`, and
`sequence_length` for `cnn1d`, to rebuild the architecture before loading
weights — plain `pickle.load` won't work for the torch ones, use
`load_model` either way for a consistent interface).

**Practical recommendation:** only turn this on for E01 and E02 (the flow-
independent baselines, which is what the SHAP comparison actually needs) —
not E03–E05, where 5 seeds × 8 folds × multiple sequence lengths of
saved deep-learning checkpoints would add up fast for a comparison that
doesn't need them.

## Result provenance

Every number in the manuscript traces back through a fixed chain:

```
paper/manuscript.md (a specific placeholder, e.g. Table 04)
      ↓
analysis/generate_table_04.py
      ↓
results/processed/e01_metrics.json, e02_metrics.json, e03_metrics.json, e04_metrics.json
      ↓
experiments/e0{1,2,3,4}/run.py
      ↓
results/raw/e0{1,2,3,4}_terminal_output.txt (untouched stdout)
      ↓
configs/experiments/e0{1,2,3,4}_*.yaml (exact hyperparameters/split settings)
      ↓
ENVIRONMENT.md (library versions, git commit, random seeds)
```

No number is ever hand-typed into `paper/manuscript.md`; every
`[TABLE PLACEHOLDER]` names the exact script to run and the exact CSV to
read from.

## Troubleshooting

- **`numpy._core._exceptions._ArrayMemoryError: unable to allocate ... GiB
  for an array with shape (N, huge_number) and data type bool`:** this
  means a high-cardinality column (thousands of distinct values — most
  likely `Sport` or `Dport`, port numbers can take up to 65,535 distinct
  values) ended up in `configs/data.yaml`'s `categorical_columns` list.
  One-hot encoding creates one column per distinct value, so a
  high-cardinality column can try to allocate more memory than exists.
  **Fix:** `categorical_columns` should be exactly `["State", "Flgs",
  "Proto"]` — remove `Sport`/`Dport` or any other numeric ID-like field if
  present. As of this version, `clean_and_encode` also raises a clear
  error *before* attempting the allocation if a categorical column has
  more than 50 distinct values, so this should show as an informative
  `ValueError` naming the offending column(s) rather than an opaque
  memory crash going forward.
- **Audit reports `NO SCENARIO COLUMN FOUND` / scenario-granularity
  verdict looks like a failure even though your files are the correct 8
  real GeNIS scenario files:** this was a real bug, fixed. `audit.py` now
  correctly recognizes that `column_map.scenario_id: null` means
  "derive scenario identity from filename" (the confirmed, working design
  for GeNIS's real file layout — see `DATASET.md` Section 8), and reports
  that as the expected case rather than an error.
- **`No filename matched interval=60 among N CSVs found...` warning, even
  though your files are the correct real 60-second scenario files:** this
  fires whenever the interval number "60" doesn't literally appear
  anywhere in the file path — which happens if you extracted
  `scenarios-60-sec/*.csv` directly into `data/preprocessed/` without
  keeping the `scenarios-60-sec` folder name. The fallback behavior (use
  every CSV found) is still correct here as long as `data/preprocessed/`
  only contains the 8 real 60-second scenario files and nothing from the
  5/10/30-second folders — but to get a clean run with no warning, put the
  files under a folder whose name contains "60" (e.g.
  `data/preprocessed/scenarios-60-sec/scenario-1-disrupt.csv`, etc.).
- **`OSError: [WinError 1114] A dynamic link library (DLL) initialization
  routine failed` / mentions `c10.dll` (Windows only), OR any other torch
  install problem that survives a venv, a CPU-only reinstall, and a VC++
  Redistributable repair:** this is a broken `torch` install, not a bug in
  this codebase — it only affects the MLP/1D-CNN/GRU models (RF and
  XGBoost never import torch, see the lazy-import note below). Try, in
  order, before falling back to the no-torch path:
  1. Reinstall torch as the CPU-only build (this project doesn't need a
     GPU): `pip uninstall torch torchvision torchaudio -y`, then
     `pip cache purge`, then
     `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
     Verify in isolation before touching the pipeline:
     `python -c "import torch; print(torch.__version__); print(torch.zeros(3))"`.
  2. Install/repair the **Microsoft Visual C++ Redistributable (x64)** —
     torch's native DLLs link against it, and a missing or stale copy is
     the single most common cause of this exact error. Run the installer
     even if a version already shows as present; it repairs in place.
  3. Confirm your Python version is 3.8–3.12 (`torch==2.3.1` doesn't
     support 3.13+): `python --version`.
  4. **Try `conda` instead of `pip` for torch specifically**, even if
     everything else stays in `pip`/`venv`: `conda install pytorch
     cpuonly -c pytorch`. Conda's PyTorch build bundles its own copy of
     the MSVC runtime DLLs rather than relying on the system-wide
     Redistributable being correctly installed, which resolves this exact
     error on some Windows machines where the pip wheel doesn't.
  5. **Try a cloud notebook instead of your local machine** — Google
     Colab (free) ships with a working CPU/GPU torch install already, on
     Linux, which sidesteps Windows DLL issues entirely. Upload this
     repo's contents (or clone it if you push it to GitHub), install the
     non-torch requirements, and torch just works out of the box.
  6. **If torch still won't cooperate: use the official no-torch configs**
     — `configs/experiments/e01_random_split_baselines_no_torch.yaml` and
     `e02_loso_baselines_no_torch.yaml` run RF + XGBoost only (no MLP).
     These are real, supported experiment configs (not a hand-edit
     workaround), producing `e01_no_torch_metrics.json` /
     `e02_no_torch_metrics.json`:
     ```powershell
     python experiments/e01/run.py --config configs/experiments/e01_random_split_baselines_no_torch.yaml
     python experiments/e02/run.py --config configs/experiments/e02_loso_baselines_no_torch.yaml
     ```
     This gets you real RQ1 results today (the flow-independent half of
     the comparison), while sequence-aware RQ2 results (which need torch
     for 1D-CNN/GRU) wait on a working torch install, tried again later or
     on a different machine/environment. State this scope limitation
     plainly in the manuscript's Limitations section if it's still true
     when you write it up — don't silently drop RQ2.
- **`missing_expected_columns` non-empty after audit:** GeNIS's real
  column names don't match `configs/data.yaml`'s defaults (which were
  written from the published data descriptor, not from inspecting the
  live files). Open `results/processed/data_audit_summary.json`, find the
  real column names, and update `configs/data.yaml`'s `column_map`,
  `drop_columns`, and `scenarios` block to match.
- **`ValueError: Discovered scenario identifiers not listed...`:**
  same root cause — `configs/data.yaml`'s `scenarios.attack`/`benign`
  lists don't match what's actually in the data. This error is
  intentional (see `src/data/preprocessing.py::derive_scenario_column`) —
  it exists specifically to stop a mislabeled scenario from silently
  entering the LOSO split logic.
- **`ModuleNotFoundError: No module named 'xgboost'` (or `torch`, `shap`,
  `statsmodels`):** these are only imported lazily, inside the function
  that actually needs them, so Random Forest–only runs work without them —
  but XGBoost/MLP/CNN/GRU runs and SHAP explainability need the full
  `requirements.txt` installed.
- **Torch determinism warnings:** expected on some GPU operations; see
  `ENVIRONMENT.md`'s "Known non-determinism sources" section.

## Limitations (short form — full version in `paper/manuscript.md`, Section 20)

Single-simulator dataset; only 8 scenarios (modest LOSO statistical
power); severe class imbalance; single flow-export interval (60s) by
design; sequence-aware models don't mask zero-padding internally (stated
explicitly in `src/models/cnn1d.py` and `gru.py`); results may plausibly
be a negative/null finding given how close to ceiling flow-independent
models already are on GeNIS's random split.

## Ethics

Purely defensive research on a publicly released, simulator-generated
dataset. No attack tooling is developed or distributed; no real systems
are targeted; no personal data is involved (GeNIS traffic is synthetic,
generated on an isolated CyberRange simulation). See
`paper/manuscript.md`, Section 21.

## Citation

If you use this code, please cite this repository and the GeNIS dataset
paper (see `references.bib`). If you use GeNIS itself, cite Silva et al.
(2025) directly, per the dataset's own citation request.
