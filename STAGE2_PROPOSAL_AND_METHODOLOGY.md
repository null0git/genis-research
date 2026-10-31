# Stage 2 — Research Proposal, Methodology, and Experimental Design

Topic approved: Candidate D. Per your instruction on scope, the only prior GeNIS ML study (Silva et al., FPS 2025) used the **60-second flow interval only** — so this project also uses the 60 s interval only, as its primary and sole analysis. Flow-interval comparison is dropped entirely, not just demoted, since there's no existing 60-vs-other-intervals result on GeNIS to extend and no budget reason to add a dimension nobody else has touched yet in this line of work.

Working title (unchanged): **"Temporal Context and Cross-Scenario Generalization in Machine-Learning-Based Network Intrusion Detection Using the GeNIS 2025 Dataset."**

---

## 1. Final research questions and hypotheses

**RQ1.** Does cross-scenario evaluation reveal a generalization gap that random-split evaluation hides on GeNIS?

- IV: split strategy — random (stratified) vs. leave-one-scenario-out (LOSO).
- DV: macro-F1, per-class recall, false-positive rate.
- H1: mean macro-F1 differs between random-split and LOSO evaluation, for a fixed model.
- H0: no difference.
- Rationale: the dataset's own reference split is random and shuffled, and the only published GeNIS benchmark uses a random split. Neither has been checked against a harder, leakage-resistant alternative. Comparable work on other datasets (temporal splits on CICIDS-2017) shows detection rate can collapse under non-random splits, so there's a documented mechanism, not just a hunch.

**RQ2.** Does explicit multi-flow temporal context change model performance, and does that effect itself depend on the split strategy?

- IV: model family — flow-independent (MLP, RF, XGBoost) vs. sequence-aware (1D-CNN, GRU over ordered flow windows), crossed with split strategy from RQ1.
- DV: macro-F1, per-class recall, FPR, plus the *generalization gap* (random-split score − LOSO score) as a derived DV.
- H1: the difference in macro-F1 between sequence-aware and flow-independent models is not the same under random-split evaluation as under LOSO evaluation (interaction effect).
- H0: no interaction — any sequence-aware advantage (or disadvantage) is constant across split strategies.
- Rationale: this is the actual novel question. Nobody has run a sequence-aware model on GeNIS at all — the existing LSTM baseline was explicitly non-sequential — so RQ2's main effect is new by itself, and the interaction with RQ1 is new on top of that.

**RQ3 (secondary, resource-permitting).** What computational cost does sequence-aware modeling add relative to flow-independent modeling, at deployment-relevant batch sizes?

- DV: training time, inference latency per flow, model size on disk, peak memory.
- No hypothesis test — this is a descriptive engineering question that feeds the deployment-analysis section, not a statistical claim.

All three questions are designed so that "no difference," "sequence context makes things worse," and "the effect only shows up under one split strategy" are all valid, reportable endings. Nothing here is set up to require a positive result to be worth publishing.

---

## 2. Dataset

GeNIS, 60-second flow interval, `3-scenarios.zip`'s `scenarios-60-sec/` folder (Zenodo DOI `10.5281/zenodo.14919237`) — CONFIRMED against real downloaded data as the correct source; NOT `4-preprocessed`, which strips all timing columns and ships pre-shuffled (see `DATASET.md` Section 8 for the full verification trail). Full download/verification instructions live in `DATASET.md` (Stage 4). Summary of what matters for the design:

- 8 attack scenarios (multi-stage: recon → bruteforce → DoS, against FTP/web/AD services) + 3 benign scenarios (user, admin, background/idle).
- Labels used: `BinaryLabel` for RQ1/RQ2's primary analysis, `CategoryLabel` (benign / bruteforce / DoS / recon) for the multiclass arm.
- Known imbalance from the only prior benchmark: malicious ≈ 92.6% / benign ≈ 7.4% at binary level (Silva et al., 2026) — this number is theirs, from their split, not ours; our own class balance will be measured directly from the data we load and reported, not assumed to match theirs.
- A scenario identifier column is required to build LOSO folds and sequences correctly. If GeNIS's preprocessed CSVs don't carry an explicit scenario label already, one must be reconstructed from file provenance (which raw capture / attack-step folder each flow's source PCAPNG came from) during preprocessing — this is flagged as an open engineering task for Stage 3, not assumed to be trivial.

---

## 3. Splitting strategy — full justification

Three split strategies are implemented and compared, not just the two named in the RQs, because the middle one is needed to interpret the other two honestly:

1. **Random stratified split** (the "optimistic" baseline, matching the dataset's shipped split and the only prior paper). Flows are shuffled and split 70/15/15 (train/val/test), stratified by `BinaryLabel`. **What it measures:** how well a model fits the statistical distribution of flows when scenario identity is not held out — i.e., the same attack scenario can contribute flows to both train and test. **Why it's optimistic:** flows from the same scenario are temporally and behaviorally correlated (same attacker IP, same tool, same timing pattern), so a model can partly succeed by memorizing scenario-specific signatures rather than learning generalizable attack behavior. This is the standard "i.i.d. assumption" critique documented in the generalization literature (e.g., the unsupervised-generalization PMC paper found in Stage 1).

2. **Leave-one-scenario-out (LOSO)**, cross-validated across all 8 scenarios (CONFIRMED count — not 11 as originally estimated in Stage 1; GeNIS's real structure has no separate benign-only scenario files, benign traffic lives inside each of the 8 scenario recordings alongside its attack payload). In each fold, one full scenario is held out entirely for testing; the model trains on the remaining 7. **What it measures:** whether a model generalizes to an attack campaign it has never seen any flow from, which is the realistic deployment question ("will this catch tomorrow's attack, not just a replay of today's"). **Why 8 folds and not fewer:** using every scenario as a held-out fold once, rather than picking one fixed holdout, avoids cherry-picking a convenient or inconvenient scenario and gives a distribution of LOSO scores rather than one number — this directly enables the statistical analysis in Section 6.

3. **Temporal (chronological) split**, using each flow's start timestamp within a scenario: earliest 70% of a scenario's flows for training, latest 30% for testing, scenarios kept intact within a split. **What it measures:** short-horizon generalization within the same attack campaign — can the model pick up an attack partway through, having only seen its earlier stages? This split is **not** used for RQ1/RQ2's headline comparison (LOSO is the harder, more decision-relevant test), but it's included as a robustness check reported alongside the main results, because a model that does well on LOSO but poorly on the temporal split within-scenario would tell a different, also-useful story.

**Leakage rules that apply to all three splits:**
- No flow ever appears in more than one of train/val/test.
- For LOSO and temporal splits, all preprocessing statistics (scaler mean/variance, feature-selection scores, class weights) are computed on the training portion only and applied to val/test — never fit on the full dataset first.
- IP addresses, MAC addresses, VLAN IDs, and any other topology-identifying fields are dropped before training (following the precedent set by Silva et al., 2026, and justified independently here: a model that keys off a specific attacker IP is not learning an attack behavior, it's memorizing a fixture of this one simulated network).

---

## 4. Sequence construction (for the sequence-aware arm)

This needs its own section because the instructions are right that it's easy to get wrong.

- **Unit of sequencing:** flows, ordered by `StartTime` (or the closest available start-timestamp field — exact column name confirmed during Stage 3 data audit), **within a single scenario only**. A sequence never crosses a scenario boundary. This is what keeps LOSO honest: if sequences could span scenarios, a "held-out" scenario's flows could still leak temporal context from a training scenario into the test set.
- **Sequence length:** three values tested as an ablation — 5, 10, and 20 flows per sequence — chosen to span roughly one order of magnitude, from a short local context to a longer one, given that GeNIS attack scenarios are multi-stage and a longer window is more likely to span stage transitions (e.g., recon into bruteforce).
- **Windowing:** sliding window with stride 1 (each flow after the first `sequence_length − 1` flows in a scenario becomes the "last" flow of one sequence), so every flow still gets exactly one prediction, keeping the sequence-aware and flow-independent arms comparable flow-for-flow at evaluation time.
- **Labeling:** a sequence is labeled by the class of its **last** flow (the one being classified), not by majority vote across the window — this matches the actual deployment scenario of "classify the flow that just arrived, using recent history for context," rather than "classify a whole window."
- **Padding:** for the first `sequence_length − 1` flows of each scenario, where a full window isn't yet available, left-pad with zero-vectors and a learned padding mask (for GRU) or zero-padding with an explicit mask channel (for 1D-CNN). This is applied per scenario per split — never padded with flows borrowed from a different scenario.
- **Missing/irregular gaps:** GeNIS scenarios include idle/background periods; flows are not evenly spaced in time. Sequences are built by flow order, not by fixed wall-clock ticks, so large idle gaps become part of the feature space (via existing time-based features like inter-arrival time) rather than requiring separate imputation — this is a design choice, stated explicitly rather than assumed, and its effect is checked qualitatively in the error analysis.
- **Different flow intervals:** not applicable — single 60 s interval only, per the approved scope.

---

## 5. Models

**Flow-independent baselines** (justify each explicitly, not just include for count):
- **Random Forest** and **XGBoost** — included because they are the top performers in the only existing GeNIS benchmark and the standard tabular NIDS baselines field-wide; reproducing them under the new split strategies is what makes RQ1 answerable at all.
- **MLP** — included as the flow-independent deep-learning counterpart to the sequence-aware models below, so that any sequence-aware advantage can be attributed to sequence information specifically, not just to "being a neural network." LightGBM is dropped from the reproduction set (it tracked RF/XGBoost closely in the existing benchmark and doesn't add a distinct comparison point for this project's specific questions — a resource-allocation choice, stated here rather than silently made).

**Sequence-aware models:**
- **1D-CNN** over the flow-window (convolving across the sequence-of-flows axis) — included because it's cheap to train relative to a recurrent model and gives a "local temporal pattern" baseline before reaching for recurrence.
- **GRU** — included instead of LSTM specifically because (a) it has fewer parameters and trains faster on a laptop, and (b) using a *different* recurrent architecture than the existing GeNIS paper's LSTM avoids implying this project is "redoing their LSTM correctly" — it's a new model on a new task (real sequences), not a fix to their model.
- Transformer: **not included**, and this is a deliberate scope cut, stated honestly. A self-attention model over 5–20-length sequences is unlikely to show an advantage over a GRU at this scale and would roughly double the hyperparameter-tuning burden for a laptop-scale project. If Stage 3 compute checkpoints (Section 7) leave meaningful headroom, a single fixed-hyperparameter Transformer run can be added as an exploratory addendum — but it is not part of the core design or the confirmatory hypothesis tests.

---

## 6. Class imbalance and statistical analysis

- **Class weighting** (inverse-frequency) is applied during training for all models, fit from the training split only, reported per split/fold since class balance differs across scenarios (a DoS-heavy scenario looks very different from the idle "background" scenario).
- **No oversampling/SMOTE** in the primary analysis: SMOTE on flow-level tabular NIDS features has a documented risk of synthesizing between-flow interpolations that don't correspond to real traffic behavior, and — more importantly for this design — synthesizing sequence data with SMOTE is not well-defined without breaking the leakage rules in Section 4. Class weighting is used instead as a leakage-safe alternative; if time permits, an oversampling ablation is run on the flow-independent arm only (where it's well-defined) and reported as a secondary check, not folded into the headline comparison.
- **Repeated evaluation:** the random split (arm 1) is repeated with 5 different random seeds; LOSO (arm 2) naturally provides 8 folds. Both give a distribution of scores per model, not a single number.
- **Statistical test:** paired comparisons (same model, same data, different split strategy or different sequence-length) use a **Wilcoxon signed-rank test** rather than a paired t-test, because the number of folds/seeds is small (8 LOSO folds, 5 seeds) and macro-F1 scores are bounded and non-normally distributed near the ceiling — the Wilcoxon test doesn't assume normality, which a t-test does. Effect size is reported via **rank-biserial correlation** alongside the p-value, since a significant-but-tiny effect and a large-but-marginal-p effect mean different things for this project's conclusions and both need to be visible, not just the p-value.
- **Multiple comparisons:** RQ1 and RQ2 involve several paired comparisons (per model, per sequence length). Holm-Bonferroni correction is applied within each RQ's family of tests, reported alongside uncorrected p-values so a reader can see both.

---

## 7. Computational staging

| Stage | Purpose | Data | Expected RAM | Expected storage | Runtime (rough) |
|---|---|---|---|---|---|
| A | Pipeline development/debugging | 1 scenario pair (1 attack + 1 benign), subsampled to ~50k flows | 4–8 GB | 1–2 GB | Minutes per run |
| B | Validation of splits, sequence construction, leakage checks | 3–4 scenarios, ~300k flows | 8 GB | 3–5 GB | Tens of minutes per run |
| C | Full 60 s dataset, flow-independent models only | All 8 scenarios, full 60 s flow set (a single scenario file — scenario-7-authtest.csv — was confirmed to be 29,679 flows; total across all 8 will be measured directly via audit.py, not assumed) | 8–16 GB | 5–10 GB | Under an hour per model on CPU |
| D | Full sequence-aware experiments + LOSO cross-validation across all models, sequence lengths, and split strategies | Full 60 s dataset | 16 GB recommended | 10–15 GB (including saved models, logs, intermediate sequence arrays) | Several hours total, dominated by the 8-fold LOSO × 2 sequence-aware models × 3 sequence lengths grid; tree-based models are cheap, GRU/CNN training is the bottleneck |

**If hardware is insufficient:** the sequence-length ablation (3 lengths) is the first thing to cut back to a single length (10, the middle value); the temporal-split robustness check (Section 3, item 3) is the second thing to cut, since it's a secondary check, not one of the two confirmatory RQs; full LOSO can be downgraded to a stratified 5-of-11-scenario subsample if 11 full folds are not tractable, with this substitution stated plainly in Limitations rather than hidden. No GPU is assumed necessary — GRU/1D-CNN at this data scale (sequence length ≤ 20, tabular features, not raw packets) should train on CPU within the runtimes above; a GPU would speed Stage D up but is not required.

---

## 8. Evaluation metrics and why each is included

- **Macro-F1** — primary metric, since it weights each class equally regardless of its frequency, which matters directly here because LOSO folds have very different class balances from fold to fold.
- **Per-class recall** — because "detects attacks in general" and "detects this specific attack category" are different claims, and the error analysis needs class-level detail to say anything concrete.
- **False-positive rate** — a NIDS that never misses an attack but floods analysts with false alarms is not deployable; FPR is reported separately from recall for exactly this reason.
- **Accuracy** — reported for comparability with the existing GeNIS benchmark, but explicitly flagged in the manuscript as uninformative on its own given the class imbalance, so it isn't allowed to stand in for macro-F1 anywhere in the interpretation.
- **Confusion matrices** — per split strategy per model, feeding the error analysis.
- **Training time, inference latency, model size, peak memory** — RQ3's deployment-cost metrics.

---

## 9. Explainability

SHAP (via the `shap` library's `TreeExplainer` for RF/XGBoost and `DeepExplainer`/`GradientExplainer` for MLP/GRU/CNN) is applied, but **only to answer a specific question**: whether the features driving predictions differ between the random-split and LOSO-trained versions of the same model. If SHAP shows the LOSO model relying on more genuinely behavioral (time/quantity-based) features and less on anything that could be scenario-specific, that's evidence supporting a mechanism for RQ1's result, one way or the other. This is not a decorative addition — if RQ1 shows no generalization gap at all, the SHAP comparison is still run (it's still informative to know *why* generalization held up), but it stays a secondary, error-analysis-adjacent piece, not a headline result.

---

## 10. Repository structure

```
genis-temporal-generalization/
├── README.md
├── EXPERIMENTS.md
├── ENVIRONMENT.md
├── DATASET.md
├── LICENSE
├── references.bib
├── requirements.txt
├── configs/
│   ├── data.yaml
│   ├── models/
│   │   ├── rf.yaml
│   │   ├── xgboost.yaml
│   │   ├── mlp.yaml
│   │   ├── cnn1d.yaml
│   │   └── gru.yaml
│   └── experiments/
│       ├── e01_random_split_baselines.yaml
│       ├── e02_loso_baselines.yaml
│       ├── e03_random_split_sequence_aware.yaml
│       ├── e04_loso_sequence_aware.yaml
│       ├── e05_sequence_length_ablation.yaml
│       ├── e06_temporal_split_robustness.yaml
│       └── e07_deployment_cost.yaml
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── audit.py              # confirms column names, scenario labels, flow counts before anything else runs
│   │   ├── preprocessing.py
│   │   ├── splits.py             # random / LOSO / temporal split implementations
│   │   └── sequences.py          # leakage-safe windowing per Section 4
│   ├── models/
│   │   ├── tree_models.py
│   │   ├── mlp.py
│   │   ├── cnn1d.py
│   │   └── gru.py
│   ├── training/
│   │   └── train.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── statistics.py         # Wilcoxon, Holm-Bonferroni, effect sizes
│   │   └── explain.py            # SHAP
│   └── utils/
│       ├── logging.py
│       ├── seeding.py
│       └── config.py
├── experiments/
│   ├── e01/ ... e07/              # one runnable script or entry point per experiment, per EXPERIMENTS.md
│   └── figures/                   # plotting scripts, one per figure
├── analysis/
│   └── generate_table_XX.py       # one script per manuscript table
├── results/
│   ├── raw/                       # untouched terminal/log output
│   ├── processed/                 # cleaned metrics.json / .csv per experiment
│   └── figures/
├── paper/
│   ├── manuscript.md
│   └── figures/
└── tests/
    └── test_leakage.py            # automated check: no shared flow IDs across train/val/test; no cross-scenario sequences
```

`tests/test_leakage.py` is called out specifically because Section 3 and 4's leakage rules are exactly the kind of thing that silently breaks during refactoring — an automated test that fails loudly is worth more here than a comment.

---

## 11. What Stage 3 needs to deliver before any placeholder in the manuscript can be filled

1. `src/data/audit.py` run against the actual downloaded GeNIS 60 s CSVs, confirming: exact column names for timestamp/scenario identity, actual flow counts per scenario, actual class balance per scenario. Several numbers in this document (flow counts, exact imbalance ratio) are placeholders inherited from the *other* paper's split, not yet measured on ours, and are marked as such.
2. The full codebase per the structure above.
3. Stage A/B compute checkpoints, to confirm or revise the Stage 7 runtime estimates before committing to the full Stage C/D run.

This document is the methodology; Stage 3 is the implementation of it.
