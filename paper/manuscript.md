# Temporal Context and Cross-Scenario Generalization in Machine-Learning-Based Network Intrusion Detection Using the GeNIS 2025 Dataset

**Status: every section that does not depend on experimental results is finished — Introduction, Background, Related Work, Research Gap, Methodology, Feature Engineering, Ethics, Reproducibility, and the Threats to Validity framework are all complete prose, not outlines. No experiments have been run against real data yet, so the Abstract, Results, Ablation Study, Generalization Analysis, Error Analysis, Deployment Analysis, Discussion, Limitations, and Conclusion remain explicit placeholders — every numeric claim in this document is either cited from a published source or marked as a placeholder with the exact command needed to fill it. Three citations are marked UNVERIFIED in `references.bib` and must be confirmed before submission (see Section 25).**

---

## Abstract

[ABSTRACT PLACEHOLDER — fill in after all experiments complete]

Draft shell, to be completed last, once Results and Discussion are final:

> Network intrusion detection systems built on machine learning are usually evaluated on a random split of flows from a single dataset, which can overstate how well they generalize to attacks the model has not seen before. We study this question on GeNIS, a 2025 dataset of eight recorded scenarios on a simulated enterprise network, each combining a multi-stage attack campaign with concurrent benign traffic (typical user, administrator, and background activity). We compare a random stratified split against a leave-one-scenario-out split for five models — three flow-independent (Random Forest, XGBoost, MLP) and two sequence-aware (1D-CNN, GRU) models trained on ordered windows of flows — at the dataset's 60-second flow interval. [PLACEHOLDER: one to two sentences stating the headline generalization-gap finding, in whichever direction the data supports]. [PLACEHOLDER: one to two sentences stating whether sequence-aware models changed that gap, in whichever direction the data supports]. [PLACEHOLDER: one sentence on the deployment-cost trade-off]. These results indicate [PLACEHOLDER — state the actual implication once known; do not pre-write a conclusion].

**Do not fill this in by guessing at plausible numbers.** Run `analysis/generate_abstract_stats.py` after Results is finalized; it pulls the exact headline numbers from `results/processed/` so the abstract can't drift from the tables.

## Keywords

Network intrusion detection, machine learning, temporal modeling, cross-scenario generalization, GeNIS, sequence models, data leakage, reproducibility.

---

## 1. Introduction

Attackers who compromise a network rarely act in a single instant. A typical intrusion unfolds as a sequence: reconnaissance to map the network, a foothold gained through credential guessing or a known vulnerability, then movement toward whatever the attacker actually wants — data, disruption, or persistence. A network intrusion detection system (NIDS) that classifies each network flow on its own, with no memory of what came before it, sees only one frame of that sequence at a time.

Most machine-learning NIDS research still evaluates flow-independent classifiers. The dominant reason is practical: public NIDS datasets are usually distributed as flow tables, split into training and test sets by shuffling rows at random. A random split treats every flow as exchangeable, which is convenient for training but sits at odds with how attacks actually happen — flows from the same attack session share a source IP, a timing pattern, and often a specific tool's fingerprint. A model evaluated on a random split can partly succeed by recognizing that fingerprint rather than the underlying attack behavior, and a fingerprint is exactly the kind of thing that changes the next time a real attacker shows up.

This gap between how NIDS models are usually tested and how they need to perform in practice has been documented before, on other datasets. Split strategy alone has been shown to change detection rates substantially: models evaluated on a temporal split of CICIDS-2017 (train on earlier days, test on later ones) show markedly lower detection rates for some attack categories than the same models evaluated on a random split of the same data [PLACEHOLDER CITATION — verify author list before final submission, see references.bib entry marked UNVERIFIED]. Related work on cross-dataset generalization shows a similar pattern when a model trained on one network environment is tested on another [PLACEHOLDER CITATION — same verification note].

GeNIS, released in 2025, gives us a way to study this question under controlled conditions. Unlike datasets assembled from a single continuous capture, GeNIS is explicitly modular: it contains eight distinct multi-stage attack scenarios and three distinct benign-activity scenarios, each recorded as a separate, labeled unit on a simulated enterprise network [Silva et al., 2025]. That structure makes it possible to hold out entire attack scenarios for testing — not just a random slice of flows — which is a stronger and more realistic test of generalization than a shuffled split allows.

The only published machine-learning benchmark on GeNIS so far, by the dataset's own creators, evaluates Random Forest, XGBoost, LightGBM, an LSTM, and an MLP on a single random 70/30 split, and reports near-ceiling F1-scores for all five models [Silva et al., 2026]. That paper is explicit that its LSTM was not given a genuine sequence of flows — each input was treated as "one one-step sequence," meaning the recurrent model saw exactly the same single-flow input as the MLP. GeNIS's defining feature — that its scenarios are sequential, multi-stage attacks — is not yet exploited by any model that has been run on it, and its performance has not yet been checked against a split that respects scenario boundaries.

This project addresses both gaps together, because they interact: whether temporal context helps a model may depend on whether the model is being asked to recognize something it saw during training or to generalize to something new. We ask:

**RQ1.** Does cross-scenario evaluation reveal a generalization gap that random-split evaluation hides on GeNIS?

**RQ2.** Does explicit multi-flow temporal context change model performance, and does that effect depend on the split strategy?

**RQ3.** What computational cost does sequence-aware modeling add relative to flow-independent modeling?

### Contributions

- **Scientific contribution:** an empirical answer, on GeNIS, to whether cross-scenario evaluation changes measured detection performance relative to random-split evaluation, and whether any change interacts with the use of explicit temporal context — reported honestly regardless of which direction the result points.
- **Methodological contribution:** a leakage-safe protocol for constructing ordered flow sequences within scenario boundaries and for building leave-one-scenario-out folds on GeNIS, which the existing GeNIS literature has not defined.
- **Engineering contribution:** an open, documented pipeline — preprocessing, all three split strategies, sequence construction, five trained model families, evaluation, and statistical testing — released for others to reuse or contest.
- **Reproducibility contribution:** every reported number in this manuscript is traceable to a specific script, config file, and raw output file (Section 22).

---

## 2. Background

### 2.1 Flow-based network intrusion detection

A network flow is a summary of one conversation between two hosts — typically identified by source and destination IP, port, and protocol — collapsed into a handful of statistics: how long it lasted, how many packets and bytes moved in each direction, how the connection was opened and closed, and the timing between packets. Machine-learning NIDS research works almost entirely at this flow level rather than on raw packets, for a practical reason: a single flow can span thousands of packets, and classifying every packet individually is both far more expensive and far less informative than summarizing the conversation once and classifying that. Flow-level features also generalize better across different capture conditions than raw payload bytes do, since they describe behavior (how much data moved, how fast, in what pattern) rather than content that a defender may not even be permitted to inspect.

Turning packets into flows is the job of a flow exporter. Argus and, more recently, CICFlowMeter are the two most widely used in the NIDS literature, and both have been shown to introduce their own artifacts into the datasets built on top of them — inconsistent feature values, and in at least one documented case a flow-level implementation error that changed how an entire attack category looked to a downstream classifier. GeNIS instead uses HERA, a flow exporter built by the same research group that produced GeNIS, specifically to standardize this extraction step and avoid the tool-specific inconsistencies that later analysis has found in older benchmark datasets [PLACEHOLDER CITATION — Pinto et al., HERA tool paper — verify before final submission, see references.bib entry `pinto2024hera`].

### 2.2 Evaluation splits and generalization in NIDS research

Almost all published NIDS benchmarks report performance on a random split of the same dataset — flows are shuffled, some fraction is held out, and the model is scored on that holdout. This is convenient, but it rests on an assumption that doesn't hold for network traffic: that individual flows are independent and identically distributed. They aren't. Flows from the same session, the same attacker, or the same short time window share timing patterns, tool fingerprints, and often literal source addresses, so a random split routinely puts near-duplicate flows on both sides of the train/test boundary.

Two lines of prior work make this concrete. Comparisons of random splits against temporal splits (train on earlier days of a capture, test on later ones) on CICIDS-family datasets have found detection rates for some attack categories fall sharply once the split respects time order [PLACEHOLDER CITATION — verify author list before final submission, see references.bib entry marked UNVERIFIED]. Separately, work on cross-dataset generalization — training on one network's traffic and testing on another's — finds a comparable drop when a model is asked to transfer to genuinely new conditions rather than interpolate within a single capture [PLACEHOLDER CITATION — same verification note]. Kostas, Just, and Lones's IoTGeM study adds a related but distinct data point: under strict cross-dataset isolation, models trained on rolling-window features generalized better than models trained on single-flow features, suggesting that the extra context a window provides isn't just noise even when the underlying task is hard. None of this prior work touches GeNIS, and none of it separates "does a harder split reveal a gap" from "does temporal modeling change that gap" — which is exactly the combination this project examines.

### 2.3 Temporal and sequence-aware modeling in NIDS

"Temporal" has meant several different things in this literature, and the distinctions matter for what follows. At one end, a recurrent model (LSTM, GRU) can be applied to a single flow's own internal packet sequence — still a per-flow classifier, just with a recurrent architecture instead of a feedforward one. At the other end, a model can be given a sequence of multiple distinct flows, ordered by time, and asked to classify the most recent one using the others as context — a genuinely inter-flow, sequence-aware approach. Parlanti and Catania's taxonomy of the time-aware NIDS literature places these on a spectrum from static single-flow analysis through inter-flow windowing to full multi-stage, attack-campaign-aware modeling, and their review of roughly forty studies from 2020–2025 finds that the inter-flow and windowed categories are comparatively under-studied relative to static per-flow analysis, despite covering a broader range of attack stages when checked against the MITRE ATT&CK framework.

The only recurrent model run on GeNIS to date sits at the first end of that spectrum, not the second: Silva et al.'s LSTM baseline is explicit that it receives one flow at a time, with no multi-flow context at all. This project's sequence-aware arm (Section 7.2) sits at the second end — ordered windows of several distinct flows, built specifically so a model can use what came immediately before a given flow, not just what happened inside it.

---

## 3. Related Work

### 3.1 The GeNIS dataset and its only existing benchmark

GeNIS was introduced by its creators as a modular dataset designed specifically to address known flaws in earlier NIDS benchmarks [Silva et al., 2025]. Datasets such as CICIDS2017 have been shown, in independent follow-up work, to contain flow-extraction errors traceable to their flow-exporter tool, including a misimplemented attack and inconsistent feature values [PLACEHOLDER CITATION — Engelen et al., verify before final submission]. GeNIS instead uses HERA, a flow exporter built by the same research group specifically to standardize this process [PLACEHOLDER CITATION — Pinto et al., HERA tool paper].

The only existing classification study on GeNIS, by Silva et al. [2026], trains Random Forest, XGBoost, LightGBM, an LSTM, and an MLP on a single random 70/30 split at the 60-second flow interval, combined with a five-method feature-selection pipeline. All five models reach F1-scores above 99.9% on the full feature set. Two aspects of that study matter directly for the present work. First, the split is random and shuffled — the same scenario can and does contribute flows to both the training set and the test set, since the split operates on individual flows, not on scenario identity. Second, and more directly relevant to RQ2, the paper states plainly that its LSTM does not receive a sequence: each flow is passed to the model as a length-one input, "since this dataset is not inherently sequential" in the form the authors preprocessed it into. That sentence is the specific gap this project closes for RQ2 — nothing published on GeNIS to date evaluates a model that actually receives more than one flow of temporal context per prediction.

We reproduce Silva et al.'s core model families (Random Forest, XGBoost, MLP) as our flow-independent baselines, both to confirm their reported performance under our own pipeline and to give a fair, matched comparison point for the sequence-aware arm. This is a deliberate reproduction, not an independent claim of novelty for that part of the work.

### 3.2 Generalization beyond a single random split

The concern motivating RQ1 — that a random split can flatter a NIDS model by letting near-duplicate, same-session flows sit on both sides of the train/test boundary — is not new to this project; it is documented on datasets other than GeNIS. On the CICIDS family, comparing a random split against a temporal split (earlier days for training, later days for testing) has been shown to produce a substantially lower detection rate for at least some attack categories under the temporal split [PLACEHOLDER CITATION — verify author list before final submission, see references.bib entry marked UNVERIFIED]. Cross-dataset generalization work shows a related but mechanistically distinct effect: performance also degrades when a model trained on one network's traffic is tested on a different network's traffic entirely, which points to the same underlying problem — evaluation that is too close to the training distribution — surfacing through a different axis of "closeness" (time, in the first case; network environment, in the second) [PLACEHOLDER CITATION — same verification note]. Kostas, Just, and Lones's IoTGeM work is the closest prior evidence that additional context helps in exactly this harder-evaluation regime: under strict cross-dataset isolation, rolling-window features outperformed single-flow features, though their windowing is a feature-engineering step rather than a learned sequence model, and their setting is IoT traffic with no attack-scenario structure comparable to GeNIS's.

None of this prior work touches GeNIS. The mechanism it documents — that harder, more realistic splits reveal gaps a random split hides — is the reason to expect RQ1 might find something, not evidence that RQ1 is already answered on this particular dataset. GeNIS's modular, scenario-labeled structure is what makes a genuinely scenario-based (rather than merely temporal or cross-dataset) version of this question testable here for the first time.

### 3.3 Temporal and sequential approaches to intrusion detection

Parlanti and Catania's taxonomy of roughly forty time-aware NIDS studies published between 2020 and 2025 organizes the field along a spectrum: static single-flow analysis at one end, inter-flow and sliding-window methods in the middle, and multi-stage, campaign-aware modeling that tracks an attack across its full lifecycle at the other. Their central finding relevant here is that the middle category — models that see more than one flow at a time — is comparatively under-studied relative to static analysis, despite giving broader coverage across MITRE ATT&CK attack stages when the two are compared directly. Their survey does not run any experiments and does not touch GeNIS; it is included here to establish that the gap this project closes on GeNIS specifically is also a gap the wider field has flagged, not an idiosyncrasy of one dataset's literature being small.

This project's sequence-aware arm sits squarely in that middle, under-studied category: 1D-CNN and GRU models trained on leakage-safe windows of several consecutive flows from the same scenario (Section 7.2), built specifically so a prediction for one flow can draw on the several flows immediately preceding it. This is a different design point from the only recurrent model previously run on GeNIS, which — as Section 3.1 describes — received a single flow at a time and sits at the static end of Parlanti and Catania's spectrum despite being architecturally recurrent.

### Comparison table

| Study | Dataset | Split strategy | Temporal context used | Sequence explicitly modeled | Closest to this project because... | Differs from this project because... |
|---|---|---|---|---|---|---|
| Silva et al. (2026) | GeNIS | Random 70/30 | LSTM run, but single-flow input | No | Same dataset, same core baseline models | No scenario-based split; no genuine sequence input |
| [entropy paper, verify authors] | CICIDS-2017/2018, NSL-KDD | Random vs. temporal (compared) | Timestamps for split only | No | Demonstrates the random-vs-non-random split effect this project tests | Different dataset; no sequence modeling; no GeNIS scenario structure |
| Kostas, Just & Lones (IoTGeM) | IoT traffic | Cross-dataset, strict isolation | Rolling window features | Partial (windowed features, not an RNN/CNN sequence model) | Shows windowed context can improve cross-domain generalization | Different dataset family (IoT); windowing is feature-engineering, not a sequence model; no attack-scenario structure like GeNIS |
| Parlanti & Catania (2025) | N/A (survey) | N/A | N/A | Discussed, not implemented | Frames why inter-flow sequential methods matter | Pure survey, no experiments, does not touch GeNIS |

---

## 4. Research Gap

To the best of our systematic search of Google Scholar, arXiv, IEEE Xplore–indexed venues, ScienceDirect, PMC, and Zenodo as of August 2026 (search strategy and full literature matrix in the project's Stage 1 document, included as Appendix A), no published study evaluates GeNIS under a scenario-based or temporally ordered train/test split, and no published study on GeNIS trains a model on genuine multi-flow sequences. This is a narrow, verifiable gap, not a claim that temporal or generalization-aware NIDS research is new in general — Section 3 situates this work within a broader literature that has raised exactly these concerns on other datasets.

---

## 5. Research Questions and Hypotheses

(Restated from Section 1 with full statistical detail; see Stage 2 methodology document, Section 1, for the complete rationale behind each.)

**RQ1 / H1a:** Mean macro-F1 differs between random-split and leave-one-scenario-out (LOSO) evaluation, for a fixed model. **H0a:** no difference.

**RQ2 / H1b:** The difference in macro-F1 between sequence-aware and flow-independent models is not constant across split strategies (interaction effect). **H0b:** no interaction.

**RQ3:** Descriptive only — no hypothesis test.

---

## 6. Dataset

[Content largely stable, draftable now; only the flow-count and class-balance numbers below are placeholders pending Stage 3's data audit.]

GeNIS (GECAD Network Intrusion Scenarios) [Silva et al., 2025] is a labeled network-traffic dataset recorded on the Airbus CyberRange platform. It comprises eight recorded scenarios — five built around a denial-of-service payload ("disrupt") and three around credential bruteforce ("authtest"), each preceded by reconnaissance — with benign traffic (typical user activity, administrator activity, and idle background activity) captured concurrently within each scenario rather than as separate recordings. This differs from our own Stage 1 literature-review estimate of "8 attack + 3 benign scenarios" as 11 separate units; the real structure, confirmed directly against the released files, is 8 scenario recordings total. The dataset is distributed as raw PCAPNG captures (37,681,001 packets total, across all intervals), as an intermediate per-scenario flow export (`3-scenarios`, organized as one file per scenario per flow interval — the source this project uses), and as a final, combined and randomly shuffled flow table (`4-preprocessed`) at four flow-export intervals — 5, 10, 30, and 60 seconds — totaling 2,806,168 flows across all intervals combined [Silva et al., 2025]. This project uses the 60-second interval of the `3-scenarios` stage, not `4-preprocessed`: the final stage strips all timing columns and its rows are confirmed shuffled, which is incompatible with both the scenario-based splitting (Section 7.1) and the sequence construction (Section 7.2) this project's design requires. The only prior GeNIS classification study (Silva et al., 2026) uses `4-preprocessed`; our flow-independent baselines reproduce their feature-selection philosophy as closely as possible (Section 9) to keep that comparison meaningful despite the different source stage.

[DATA PLACEHOLDER — Stage 3 audit] Exact flow count at the 60 s interval, per-scenario flow counts, and per-scenario class balance:

```
Run: python src/data/audit.py --interval 60 --config configs/data.yaml
Raw output: results/raw/data_audit_output.txt
Processed summary: results/processed/data_audit_summary.json
```

Required values for this section: total 60 s flows; flows per scenario; `BinaryLabel` and `CategoryLabel` counts per scenario; confirmed column names for timestamp and scenario identity.

---

## 7. Methodology

[This section summarizes Section 3 ("Splitting strategy") and Section 4 ("Sequence construction") of the Stage 2 methodology document. Draftable in full now, since it describes the design, not results — reproduced here in condensed form; full detail lives in `STAGE2_PROPOSAL_AND_METHODOLOGY.md`.]

### 7.1 Split strategies

Three split strategies are used: a random stratified split (70/15/15, 5 repeated seeds), leave-one-scenario-out cross-validation (8 folds, one per scenario), and a within-scenario chronological split (earliest 70% vs. latest 30% of each scenario's flows), included as a secondary robustness check. All preprocessing statistics (scaling, feature selection, class weights) are fit on training data only, per split, per fold.

### 7.2 Sequence construction

For the sequence-aware model arm, flows are ordered by start time within a scenario and grouped into sliding windows of length 5, 10, or 20 flows, with stride 1. A sequence is labeled by its last flow's class. Sequences never cross a scenario boundary, which keeps LOSO folds free of cross-scenario temporal leakage. The first `sequence_length − 1` flows of each scenario are left-padded with a masked zero vector.

### 7.3 Leakage prevention checklist

- No flow ID appears in more than one of train/val/test within any split.
- No sequence spans two scenarios.
- Feature scaling and feature selection fit only on training data.
- Topology-identifying fields (IP, MAC, VLAN ID) removed prior to modeling.
- Automated test (`tests/test_leakage.py`) enforces the above and is run before any experiment result is accepted.

---

## 8. Experimental Design

| Experiment ID | Question addressed | Models | Split(s) | Notes |
|---|---|---|---|---|
| E01 | Baseline reproduction | RF, XGBoost, MLP | Random | Reproduces Silva et al. (2026) under our pipeline |
| E02 | RQ1 | RF, XGBoost, MLP | LOSO | |
| E03 | RQ2 (main effect, random split) | 1D-CNN, GRU | Random | Sequence length = 10 (fixed, see E05 for ablation) |
| E04 | RQ2 (main effect, LOSO) + RQ1×RQ2 interaction | 1D-CNN, GRU | LOSO | Sequence length = 10 |
| E05 | Sequence-length ablation | 1D-CNN, GRU | Random + LOSO | Lengths 5, 10, 20 |
| E06 | Temporal-split robustness check | All 5 models | Temporal | Secondary, not part of confirmatory hypothesis tests |
| E07 | RQ3, deployment cost | All 5 models | N/A (fixed test batch) | Training time, inference latency, model size, memory |

Full experiment registry with exact commands, hyperparameters, and status tracking lives in `EXPERIMENTS.md` (Stage 4 deliverable).

---

## 9. Data Preprocessing

Preprocessing follows a fixed, six-step pipeline (`src/data/preprocessing.py`), applied identically regardless of split strategy:

1. Load 60 s scenario CSVs (`3-scenarios/scenarios-60-sec/`); run `audit.py` checks.
2. Drop 57 topology-identifying, near-empty, or non-generalizable columns — derived not by independent judgment but by programmatically diffing this stage's 125 raw columns against Silva et al. (2026)'s published 87-column preprocessed schema, so this project's feature-selection philosophy matches their baseline exactly wherever the two designs don't have to differ. One column in their drop list is deliberately *not* dropped here: `StartTime`, which their pipeline discards and this project's RQ2 specifically needs. One column was independently confirmed unsafe regardless of Silva et al.'s choice: `IdleTime`, whose name and official documentation describe it as "time since last packet activity," but which was found on inspection to contain a raw Unix epoch timestamp on every row — keeping it would have let every model silently learn absolute calendar time as a feature, which would have invalidated any cross-scenario generalization result in either direction.
3. Coerce inconsistently-formatted numeric columns (`Sport`, `Dport` — real values mix `"138"`-style and `"35672.0"`-style strings, plus blanks for protocols with no ports) to numeric, and one-hot encode the three genuinely categorical fields (`State`, `Flgs`, `Proto`), again matching Silva et al.'s exact choice of which raw columns to encode.
4. Construct scenario-identity (from filename — no explicit scenario column exists in the source files) and ordering fields (`StartTime`, parsed from its real `HH:MM:SS` format to seconds-since-midnight) required for splitting and sequencing (Section 7).
5. Impute remaining missing values (median strategy, fit on the training portion of each split/fold only) — several legitimate numeric fields (e.g. `sTos`, `dTos`) are genuinely absent for protocols the field doesn't apply to, not merely dropped-and-forgotten, and a standard scaler cannot operate on missing values.
6. Fit `StandardScaler` on training data only, per split/fold; apply to val/test.

One structural property of the raw export is worth stating plainly rather than treating as a data-quality defect: a single long-lived connection (e.g. a persistent TCP session) can appear as multiple rows sharing one `FlowID` — one row per successive ~60-second reporting interval while the connection stays open, marked `Cause=Start` for the first and `Cause=Status` for each subsequent report. Each such row is a genuinely distinct observation of that connection's behavior during that interval, not a duplicate; this is exactly how flow-interval export is meant to work, and it means consecutive same-connection rows carry real temporal dependency, which is precisely the kind of structure the sequence-aware arm (Section 7.2) is designed to exploit.

[TABLE PLACEHOLDER — TABLE 01]
```
Run: python analysis/generate_table_01.py
```
Purpose: full feature list with type category (general/time/quantity/hybrid/context), and which are dropped vs. retained, mirroring Table with feature categories in the related-work benchmark for direct comparability.

---

## 10. Feature Engineering / Feature Selection

This project does not run a dedicated feature-selection step and uses the full retained feature set (everything surviving the topology-column removal and categorical encoding in Section 9) for every model. This is a deliberate choice, not an oversight, for two reasons. First, Silva et al. (2026) already ran a five-method feature-selection pipeline on GeNIS at the 60-second interval and found that reducing to a 16-feature subset came at a real cost — false-positive rate rose sharply for their neural models on the reduced set — so there is no clean, already-established "best" reduced feature set to inherit, and re-deriving one would be a separate methodological project of its own rather than a small addition to this one. Second, and more directly relevant to RQ2, feature selection interacts with sequence construction in a way that would blur the comparison this project is actually trying to make: if the flow-independent and sequence-aware arms were fit on different feature subsets, any difference in their performance could no longer be attributed cleanly to the presence or absence of temporal context, since it could also come from which features happened to be selected on which run. Holding the feature set fixed across every model and every split strategy keeps that one variable — temporal context — isolated, which is what RQ2 needs.

This means the tree-based and MLP baselines in this project should not be expected to exactly reproduce Silva et al.'s reported numbers, since those used the reduced 16-feature set; Section 13.1's baseline-comparison table reports both sets of numbers side by side rather than treating them as directly interchangeable.

---

## 11. Model Architectures

[TABLE PLACEHOLDER — TABLE 02: full hyperparameters for RF, XGBoost, MLP, 1D-CNN, GRU]
```
Run: python analysis/generate_table_02.py
```
Source: `configs/models/*.yaml`, read directly — this table must be generated from the actual config files used for training, not typed by hand, so it can never drift from what was actually run.

---

## 12. Evaluation Metrics

Macro-F1 (primary), per-class recall and precision, false-positive rate, accuracy (secondary, reported but not used for interpretation given class imbalance), confusion matrices, plus training time, inference latency, model size, and peak memory for RQ3. Rationale for each metric's inclusion is given in Section 8 of the Stage 2 methodology document.

---

## 13. Experimental Results

**No numbers in this section are to be filled in by estimation. Every value must come from `results/processed/`.**

### 13.1 Baseline reproduction (E01)

[RESULT PLACEHOLDER — EXPERIMENT E01]
```
Run: python experiments/e01/run.py --config configs/experiments/e01_random_split_baselines.yaml
Raw output: results/raw/e01_terminal_output.txt
Metrics: results/processed/e01_metrics.json
```
Required values: macro-F1, accuracy, precision, recall, FPR for RF, XGBoost, MLP, each averaged over 5 seeds with standard deviation.

[TABLE PLACEHOLDER — TABLE 03] Comparison of E01 numbers against Silva et al. (2026)'s published numbers, generated by `analysis/generate_table_03.py`, which reads both `results/processed/e01_metrics.json` and the published values hard-coded (with citation) in the script — never edited by hand in the manuscript.

### 13.2 RQ1 — random split vs. LOSO (E01 vs. E02)

[RESULT PLACEHOLDER — EXPERIMENT E02]
```
Run: python experiments/e02/run.py --config configs/experiments/e02_loso_baselines.yaml
Raw output: results/raw/e02_terminal_output.txt
Metrics: results/processed/e02_metrics.json
```
Required values: per-fold macro-F1 (8 folds) for RF, XGBoost, MLP; Wilcoxon signed-rank test comparing E01 vs. E02 macro-F1, with Holm-Bonferroni correction across the 3 models; rank-biserial effect size.

[FIGURE PLACEHOLDER — FIGURE 01]
```
Purpose: box/strip plot of macro-F1 across the 8 LOSO folds, one panel per model, with the random-split mean shown as a reference line.
Generate with: python experiments/figures/plot_rq1_generalization_gap.py
Save as: paper/figures/figure_01_generalization_gap.png
Required: labeled axes (macro-F1, scenario held out), legend, per-fold points visible (not just a box), reference line for random-split score, publication resolution (≥300 DPI).
```

### 13.3 RQ2 — sequence-aware vs. flow-independent, and the interaction with split strategy (E03, E04)

[RESULT PLACEHOLDER — EXPERIMENTS E03, E04]
```
Run: python experiments/e03/run.py --config configs/experiments/e03_random_split_sequence_aware.yaml
Run: python experiments/e04/run.py --config configs/experiments/e04_loso_sequence_aware.yaml
Raw output: results/raw/e03_terminal_output.txt, results/raw/e04_terminal_output.txt
Metrics: results/processed/e03_metrics.json, results/processed/e04_metrics.json
```
Required values: macro-F1 for 1D-CNN and GRU under random split (5 seeds) and LOSO (8 folds); the generalization gap (random-split score − LOSO score) for each sequence-aware model, compared against the same gap for the flow-independent models from 13.2, via Wilcoxon signed-rank test on the paired gap values; effect size.

[TABLE PLACEHOLDER — TABLE 04]
```
Run: python analysis/generate_table_04.py
```
Purpose: full 2×5 grid (split strategy × model) of macro-F1 mean ± SD, generated directly from results/processed/e01–e04 metrics files.

[FIGURE PLACEHOLDER — FIGURE 02]
```
Purpose: interaction plot — x-axis: split strategy (random, LOSO); y-axis: macro-F1; one line per model, sequence-aware models visually distinguished (e.g., dashed) from flow-independent models (solid), to make any interaction effect visible at a glance.
Generate with: python experiments/figures/plot_rq2_interaction.py
Save as: paper/figures/figure_02_interaction.png
Required: labeled axes, legend distinguishing model families, error bars (SD or CI), publication resolution.
```

### 13.4 Sequence-length ablation (E05)

[RESULT PLACEHOLDER — EXPERIMENT E05]
```
Run: python experiments/e05/run.py --config configs/experiments/e05_sequence_length_ablation.yaml
Raw output: results/raw/e05_terminal_output.txt
Metrics: results/processed/e05_metrics.json
```
Required values: macro-F1 for 1D-CNN and GRU at sequence lengths 5, 10, 20, under both random and LOSO splits.

[FIGURE PLACEHOLDER — FIGURE 03]
```
Purpose: macro-F1 vs. sequence length, one line per model per split strategy.
Generate with: python experiments/figures/plot_sequence_length_ablation.py
Save as: paper/figures/figure_03_sequence_length.png
```

### 13.5 Temporal-split robustness check (E06)

[RESULT PLACEHOLDER — EXPERIMENT E06] — same structure as above, `experiments/e06/run.py`, reported as a secondary check, not folded into the RQ1/RQ2 confirmatory tests.

---

## 14. Ablation Study

Consolidates E05 (sequence length) and, if time permits per Stage 2 Section 7's fallback plan, an oversampling-vs-class-weighting ablation on the flow-independent arm. [PLACEHOLDER — write once E05 and any secondary ablations are complete; this section interprets the ablation table, it does not introduce new numbers.]

---

## 15. Generalization Analysis

[PLACEHOLDER — synthesizes 13.2, 13.3, and 13.5 into a direct answer to RQ1 and RQ2. Write only after all three experiments are complete; this section is where H1a/H0a and H1b/H0b are formally accepted or rejected based on the Wilcoxon test results, not before.]

---

## 16. Error Analysis

[PLACEHOLDER — per-class confusion matrices for the best- and worst-generalizing model (identified from Section 15, not assumed in advance), broken down by which scenario was held out (for LOSO) and by attack category. Explanations must be grounded in what the confusion matrices and SHAP values actually show — e.g., "if reconnaissance flows are consistently misclassified as benign under LOSO but not under random split, check whether recon flows share more surface-level similarity with benign background traffic than with the DoS/bruteforce flows that dominate the training distribution," phrased as a hypothesis to check against the data, not asserted before the data exists.]

[TABLE PLACEHOLDER — TABLE 05: per-class recall/precision broken down by held-out scenario, generated by `analysis/generate_table_05.py`]

[FIGURE PLACEHOLDER — FIGURE 04: confusion matrix heatmaps, generated by `experiments/figures/plot_confusion_matrices.py`]

---

## 17. Computational / Deployment Analysis (RQ3)

[RESULT PLACEHOLDER — EXPERIMENT E07]
```
Run: python experiments/e07/run.py --config configs/experiments/e07_deployment_cost.yaml
Raw output: results/raw/e07_terminal_output.txt
Metrics: results/processed/e07_metrics.json
```
Required values: training time, per-flow inference latency, model size on disk, peak memory, for all 5 models.

[TABLE PLACEHOLDER — TABLE 06] generated by `analysis/generate_table_06.py`.

[PLACEHOLDER — narrative interpretation once E07 is complete: whether any macro-F1 gain from sequence-aware models (per Section 15) is worth its added latency/memory cost for a real-time deployment context, stated as a trade-off, not a verdict, since the answer depends on the deployment's own latency budget, which this project does not assume.]

---

## 18. Discussion

[Framework only — do not write final content until Section 15's actual results are known. Each branch below states what to write depending on which result obtains; only the branch matching the real outcome gets written into the final manuscript, and the others are deleted, not left in as hedging.]

- **If RQ1 shows a significant generalization gap (LOSO < random):** discuss this as evidence that GeNIS's own shipped random split, and the only prior published benchmark, likely overstate real-world generalization; connect to the same pattern documented on CICIDS-2017 (Section 3.2); discuss what specifically differs between train and held-out scenarios (via SHAP, Section 16) as a candidate mechanism.
- **If RQ1 shows no significant gap:** discuss this as a genuinely informative negative result — it would suggest GeNIS's behavioral features (time/quantity-based, with topology fields already removed) generalize across scenarios better than flow-level features on datasets like CICIDS-2017 do, and discuss what about GeNIS's design (behavioral feature focus, HERA's more careful extraction) might explain that, without overclaiming causality from one dataset's result.
- **If RQ2 shows sequence-aware models help more under LOSO than under random split:** this would be the strongest possible finding — evidence that temporal context specifically aids generalization to unseen attacks, not just memorization of seen ones — discuss mechanism via SHAP and error analysis.
- **If RQ2 shows sequence-aware models help equally or less under LOSO:** discuss honestly; a plausible explanation is that GeNIS's flow-independent statistical features (time/quantity-based) already encode enough of the relevant temporal signal (e.g., inter-arrival time, duration) that explicit sequencing is redundant for this dataset — state this as a candidate explanation to be checked against the SHAP comparison in Section 9 of the Stage 2 document, not asserted as fact.
- **If results are mixed** (e.g., helps for one model family, not the other, or only at one sequence length): report the mixed pattern directly; do not force a single clean narrative onto it.

---

## 19. Threats to Validity

- **Single-simulator dataset:** GeNIS traffic is generated on the Airbus CyberRange platform; findings may not transfer to production enterprise networks or to other simulation platforms.
- **Small number of scenarios:** 8 scenarios means 8 LOSO folds — modest statistical power; a non-significant result could reflect a true null or simply low power, and this must be stated explicitly wherever H0 is not rejected.
- **Class imbalance:** severe at the binary level (per the only prior benchmark's reported ratio; our own measured ratio is in the Stage 3 data audit); imbalance varies by held-out scenario in LOSO, which could make some folds harder for reasons unrelated to generalization per se.
- **Single flow-export interval:** results are specific to the 60 s interval; whether they hold at 5/10/30 s is untested by design (Section 1 of this document, scope decision).
- **Hyperparameter sensitivity:** models use a single fixed hyperparameter configuration per architecture (`configs/models/*.yaml`), not a per-split or per-fold search, for compute reasons — an exhaustive search across 3 split strategies × up to 8 folds × 5 models was not feasible at laptop scale. Hyperparameters were chosen from standard defaults for each architecture rather than tuned against this project's own validation data, so results likely understate what any given model could achieve with a dedicated search, and this is a limitation of the comparison's absolute numbers, not (on its own) of the relative comparison between split strategies or model families that RQ1/RQ2 actually test.
- **Simulated attacker behavior:** GeNIS's attacks were executed by two Kali machines running specific tools; real attackers may behave differently, and any generalization gap or lack thereof is a statement about generalizing across this dataset's scenario diversity, not about real-world zero-day robustness.
- **Random-split leakage discussion is dataset-specific:** the mechanism (same-scenario flows in train and test) is well-defined for GeNIS's modular design; whether it applies identically to datasets without explicit scenario boundaries is not addressed here.

---

## 20. Limitations

[PLACEHOLDER — condensed, reader-facing version of Section 19, plus any compute-driven scope cuts actually taken per Stage 2 Section 7's fallback plan (e.g., "sequence-length ablation was reduced from 3 to 1 value" or "LOSO was run on a subsample of scenarios" — state only if this actually happened; if the full design was run as planned, say so instead.]

---

## 21. Ethical Considerations

This project is purely defensive: it evaluates detection models on a publicly released, already-anonymized, simulator-generated dataset (GeNIS), and does not develop or test any attack tooling, does not target real systems, and does not involve human subjects or personally identifiable data (GeNIS's traffic is synthetic, generated on an isolated CyberRange simulation, not captured from real users). All experiments are run against publicly available data under its stated license (Section 6/Appendix), and all code and results are released openly to support independent verification, consistent with responsible-disclosure norms in defensive security research.

---

## 22. Reproducibility

Full protocol (clone → environment → dataset → verify → preprocess → train → evaluate → figures → tables) is documented in `README.md` and `EXPERIMENTS.md`. Environment and dependency versions are pinned in `ENVIRONMENT.md` and `requirements.txt`. All random seeds are fixed and logged per run (`src/utils/seeding.py`). Exact reproduction of every decimal digit is not guaranteed across different hardware or library minor-version updates (floating-point non-determinism in GPU-accelerated tree/DL libraries is a known, documented limitation of the underlying libraries, not of this pipeline specifically) — where this applies, `ENVIRONMENT.md` states the tested configuration explicitly so a close reproduction, rather than a bit-exact one, is the realistic target.

---

## 23. Conclusion

[PLACEHOLDER — 1 paragraph, written only after Sections 15 and 18 are final. Must state the actual answer to RQ1, RQ2, and RQ3 as found, including if that answer is a null result, and must not restate the Introduction's motivation as if it were a finding.]

---

## 24. Future Work

[PLACEHOLDER — to be written last, but likely candidates based on the design's own stated scope cuts: extending the split comparison to the 5/10/30 s intervals; adding a Transformer arm if RQ2's GRU/CNN result is close enough to warrant it; testing generalization across GeNIS and a second modular dataset if one becomes available; adversarial robustness under cross-scenario evaluation, echoing Silva et al. (2026)'s own stated future work.]

---

## 25. References

See `references.bib`. Entries marked UNVERIFIED in Stage 1 must be confirmed or removed before this section is finalized — no unverified citation may appear in a submitted version of this manuscript.

---

## Appendix A — Stage 1 literature matrix

Full literature search strategy, matrix, and topic-selection rationale: see [`docs/STAGE1_LITERATURE_AND_TOPIC.md`](../docs/STAGE1_LITERATURE_AND_TOPIC.md), Section 3, reproduced in full here once the manuscript is finalized.

## Appendix B — Full hyperparameters used

This project uses a single fixed hyperparameter configuration per model (no grid search — see the Threats to Validity discussion of hyperparameter sensitivity, Section 19), sourced directly from `configs/models/*.yaml`. Same table as Table 02; reproduced here for appendix completeness once generated:

```
Run: python analysis/generate_table_02.py
```

## Appendix C — Per-fold LOSO results (all models, all metrics)

[TABLE PLACEHOLDER — full disaggregated table, generated by `analysis/generate_appendix_c.py`, not summarized/rounded anywhere else in the main text.]
