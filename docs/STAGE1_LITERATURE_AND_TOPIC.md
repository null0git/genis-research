# Stage 1 — Literature Investigation and Topic Selection

**Status: DRAFT — awaiting your approval before Stage 2 begins.**

> **Errata (added after real GeNIS data was obtained and inspected in
> Stage 3):** this document's dataset description below is preserved
> as-written, as the honest record of what was knowable from published
> sources alone at the time. Two things stated here turned out to be
> wrong once real files were checked directly, and are corrected in
> `DATASET.md` (Section 8) and `configs/data.yaml`, not retroactively
> edited here:
> 1. **Scenario count.** This document assumes "8 attack scenarios + 3
>    benign scenarios" as 11 separate units. The real structure is 8
>    scenario recordings total; benign traffic (user/admin/background)
>    is captured concurrently within each of the 8 files, not as
>    separate benign-only recordings.
> 2. **Primary data source.** This document (and Stage 2, originally)
>    assumed `4-preprocessed` would be usable. It isn't for this
>    project's design — confirmed directly that it strips all timing
>    columns and ships pre-shuffled. `3-scenarios/scenarios-60-sec/` is
>    the correct source and is what the actual pipeline uses.
>
> Everything else below — the literature matrix, the gap analysis, the
> topic selection and its rationale — is unaffected by this and stands
> as originally written.

This document is the required first deliverable: a literature search strategy, a literature matrix, an honest account of what the closest existing studies already did, 3–5 candidate research questions, and a recommendation. No experiments have been run. No numeric results appear anywhere in this document except numbers taken directly from published sources (all cited).

---

## 1. Search strategy

**Databases used:** Google Scholar (via targeted web search), Semantic Scholar/PMC/ScienceDirect indexing, arXiv, IEEE/ACM/Springer indexes reachable through search, and the Zenodo record itself. I prioritized primary sources (the dataset paper, peer-reviewed conference/journal papers, arXiv preprints) over blogs or aggregator pages.

**Query families used** (paraphrased, not exhaustive — each was run as 2–4 concrete queries):
1. `GeNIS` + `GECAD` + `dataset` (find the original dataset publication and any paper that cites/uses it)
2. `GeNIS` + `intrusion detection` + `classification` / `feature selection` / `LSTM` / `unlearning` (find every downstream study)
3. `network intrusion detection` + `scenario-based split` / `unseen attack` / `generalization` / `data leakage` (establish how the field currently evaluates generalization, independent of GeNIS)
4. `temporal` / `sequential` / `concept drift` + `network intrusion detection` (establish what "temporal modeling" already means in this literature, and whether it has already been done on GeNIS or on comparable modular-scenario datasets)
5. `cross-dataset generalization` + `intrusion detection` (check whether the generalization gap I want to study has been closed via a different mechanism — cross-dataset transfer rather than cross-scenario)
6. GitHub/Zenodo checks for existing code from the GeNIS authors (`cybergecad`) to identify what infrastructure already exists and must be credited, not reproduced as "my own."

**Limitation of this search, stated explicitly:** this was a broad but not exhaustive systematic review (not a PRISMA-style protocol with a fixed query log and deduplication count). For a project of this scope that is proportionate, but the manuscript will say *"to the best of our systematic search of Google Scholar, arXiv, IEEE Xplore–indexed venues, ScienceDirect, and Zenodo as of August 2026..."* rather than an unqualified "no one has done this."

---

## 2. What already exists on GeNIS specifically

GeNIS is very new (Data Brief paper published 21 March 2025; Zenodo DOI `10.5281/zenodo.14919237`), so the downstream literature is small. I found **two** peer-reviewed/preprint studies that actually train models on GeNIS, plus the original dataset paper itself.

### 2.1 The dataset paper
**Silva, Pinto, Vitorino, Gonçalves, Maia, Praça (2025)**, *"GeNIS: A modular dataset for network intrusion detection and classification,"* **Data in Brief**, 60, 111487. DOI: `10.1016/j.dib.2025.111487`. Open access, CC BY license (confirmed on the article itself; the exact license tag on the Zenodo record page should still be checked directly at upload time, since Zenodo licenses are set per-deposit and I could not fully re-verify it independent of the article page).

Key facts I can verify from the paper/abstract and citing sources:
- Recorded on the Airbus CyberRange platform; **8 attack scenarios** (multi-stage: reconnaissance → credential attacks → DoS, targeting FTP, web servers, Active Directory) and **3 benign scenarios** (user activity, admin activity, weekend "background" idle traffic).
- Raw data: PCAPNG, **37,681,001 packets**, organized per intermediate attack step.
- Preprocessed data: **2,806,168 flows**, generated with the HERA flow exporter at **four flow intervals: 5 s, 10 s, 30 s, 60 s**.
- 125 raw extracted features, 87 after one-hot encoding of categorical fields (`State`, `Flags`, `Protocol`); three label columns (`BinaryLabel`, `CategoryLabel`, `SubCategoryLabel`).
- **Critical methodological fact for this project:** the dataset's own "4-preprocessed" folder ships flows "combined into a training set and a holdout set for each flow interval, created with shuffl[ing]" — i.e., **the official reference split is a random, shuffled split, not a scenario-based or temporally ordered split.**

### 2.2 The only existing ML benchmark on GeNIS
**Silva, Pinto, Vitorino, Maia, Praça, Amorim, Viamonte (2026)**, *"Binary and Multiclass Cyberattack Classification on GeNIS Dataset,"* in *Foundations and Practice of Security (FPS 2025)*, LNCS vol. 16402, Springer. Also on arXiv as `2511.08660`.

This is the paper closest to the original proposed idea, so I read it in full rather than from an abstract. What it actually did:
- **Split:** a single random train/validation/test split at the **60-second flow interval only** (70% train / 30% test, with 70/30 of training further split for validation). No scenario-based or temporal split is used or discussed.
- **Feature selection:** five methods (Information Gain, Chi-Squared, Recursive Feature Elimination, Mean Absolute Deviation, Dispersion Ratio) combined by summing normalized scores; top 16 features kept (≈70% of cumulative importance), for both binary and multiclass tasks separately.
- **Models:** RF, XGBoost, LightGBM (tree ensembles) and LSTM, MLP (deep learning), each tuned via grid search / manual architecture search, each run on the full feature set and the reduced 16-feature set.
- **Explicit statement about temporality:** *"Since this dataset is not inherently sequential, each data point is treated as one one-step sequence, enabling LSTMs to reveal complex nonlinear feature interactions."* This is the single most important sentence in the existing literature for this project: **the only LSTM ever run on GeNIS was NOT given a temporal sequence of multiple flows — it saw one flow at a time, exactly like the MLP.** No inter-flow, multi-flow-window, or sequence-aware model has been evaluated on GeNIS.
- **Results (for context only, not to be treated as our result):** near-ceiling performance for all five models on both tasks (F1-scores mostly above 99.9% with the full feature set; e.g., binary RF/XGB F1 ≈ 99.9949%, multiclass RF F1 ≈ 99.9817%), with tree ensembles slightly ahead of the neural models and dramatically cheaper to train/run. FPR increased notably (up to ~10×) for neural models when moving to the reduced 16-feature subset.
- **Explicitly stated future work (their words, paraphrased):** combining multiple datasets and studying transferability to unseen traffic; extending to adversarial robustness. **They do not propose scenario-based evaluation or sequence modeling as future work themselves — this is a gap they did not even flag, not just one they left open.**

### 2.3 Second existing use of GeNIS
**Magalhães et al. (2026)**, *"Machine Unlearning for the XGBoost Model with Network Intrusion Datasets,"* arXiv `2606.19220`. Uses GeNIS (alongside IoT-23) as one of two benchmark datasets for a machine-unlearning method (XGBoost-Forget) — i.e., studies how to make a trained XGBoost model "forget" specific training points efficiently, evaluated via predictive-performance retention and unlearning efficiency/attack-success-rate metrics. **Not related to temporal modeling, sequence modeling, or generalization evaluation** — different research question entirely, but confirms GeNIS is starting to be adopted as a general-purpose tabular NIDS benchmark beyond its originating group.

### 2.4 Adjacent taxonomy/survey work (context, not a GeNIS study)
**Parlanti & Catania (2025)**, *"Temporal Analysis Framework for Intrusion Detection Systems: A Novel Taxonomy for Time-Aware Cybersecurity,"* arXiv `2511.03799` (submitted to *Computer Networks*). A systematic review (40+ studies, 2020–2025) proposing a 5-category taxonomy (S.1–T.3) of how NIDS literature treats time, from static per-flow analysis to multi-window sequential modeling, mapped onto MITRE ATT&CK stages. Key findings relevant here: (a) they identify systematic bias in benchmark datasets (CIC-IDS2017, UNSW-NB15, NSL-KDD) toward late-stage/Impact attacks, which limits early-detection research; (b) they find inter-flow sequential and window-based methods give the broadest ATT&CK-stage coverage but are comparatively under-studied. This paper **does not run any experiment and does not touch GeNIS**, but it is directly relevant related work for framing why sequence-aware evaluation matters, and it independently corroborates that this is a real, recognized gap in the broader field, not just in GeNIS specifically.

**Conclusion of this sub-search:** no publication I could find — on GeNIS or, as far as I can verify, proposing to evaluate GeNIS this way — has (a) used a scenario-based or temporally-ordered train/test split on GeNIS, (b) constructed genuine multi-flow temporal sequences from GeNIS flows for a sequence model, or (c) compared GeNIS's four native flow intervals (5/10/30/60 s) against each other for detection or generalization performance. All three are open.

---

## 3. Literature matrix

"Not reported" is used verbatim wherever the source did not state a value; nothing below is inferred or estimated.

| # | Authors (Year) | Title (short) | Venue / DOI | Dataset | Samples/Flows | Attack types | Models | Feature selection | Train/test strategy | Temporal info used? | Sequences explicitly modeled? | Key metrics | Main finding | Limitations (stated or evident) | Code available | Gap relative to this project |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Silva, Pinto, Vitorino, Gonçalves, Maia, Praça (2025) | GeNIS: A modular dataset for NID and classification | *Data in Brief* 60, 111487 / 10.1016/j.dib.2025.111487 | GeNIS (introduces it) | 37,681,001 packets; 2,806,168 flows; intervals 5/10/30/60 s | 8 attack scenarios (bruteforce, DoS, recon sub-types) + 3 benign | N/A (dataset paper, no classifier trained) | N/A | Ships a pre-shuffled random train/holdout split per interval | Timestamps present in raw features | No | N/A | N/A | Dataset descriptor only; provides no baseline classification results | Not reported (dataset + descriptor code on Zenodo) | This is the data source, not a competing study |
| 2 | Silva, Pinto, Vitorino, Maia, Praça, Amorim, Viamonte (2026) | Binary and Multiclass Cyberattack Classification on GeNIS Dataset | FPS 2025, LNCS 16402, Springer / arXiv 2511.08660 | GeNIS, 60 s interval only | Train: 273,124 malicious / 21,720 benign (binary); multiclass counts per class reported partially (Not reported for full test split table due to formatting) | Binary + multiclass (DoS, Recon, Bruteforce, Benign) | RF, XGB, LGBM, LSTM, MLP | 5 methods combined (IG, Chi², RFE, MAD, Dispersion Ratio) → top 16 features | **Single random 70/30 split** (no scenario or temporal separation) | LSTM used, but explicitly as one-flow-at-a-time ("not inherently sequential... one one-step sequence") | **No** | F1, Acc, Recall, Precision, FPR, train/epoch/inference time | Near-ceiling F1 (>99.9%) for all models; tree ensembles best time/accuracy trade-off; FPR rises sharply for DL models with reduced feature set | Random split only; explicit non-use of temporal ordering; no cross-scenario test; authors themselves flag only cross-dataset transfer and adversarial robustness as future work | Not reported | **Directly overlapping baseline** — same dataset, same core model families. Does not test cross-scenario generalization or genuine sequence modeling. |
| 3 | Magalhães et al. (2026) | Machine Unlearning for the XGBoost Model with NI Datasets | arXiv 2606.19220 | GeNIS + IoT-23 | Not reported (not extracted from available text) | Not reported in detail | XGBoost (+ SISA baseline) | Not reported | Not reported | Not reported | No | Predictive-performance retention, unlearning efficiency, attack success rate (ASR) | XGBoost-Forget matches retraining performance with much faster unlearning | Different research question (machine unlearning, not detection generalization) | Not reported | Confirms GeNIS adoption beyond its authors; unrelated research question |
| 4 | Parlanti & Catania (2025) | Temporal Analysis Framework for IDS: taxonomy for time-aware cybersecurity | arXiv 2511.03799, submitted to *Computer Networks* | N/A (survey of 40+ studies, not an empirical dataset study) | N/A | N/A | N/A (taxonomy, not experiments) | N/A | N/A | Core subject of the paper | Discusses but does not implement | N/A | Identifies dataset bias toward late-stage attacks; ranks inter-flow/window methods as broadest ATT&CK coverage but under-studied | Pure literature review, no new experiments, does not touch GeNIS | Not reported | Motivates *why* temporal/sequence modeling matters; does not compete with an empirical GeNIS study |
| 5 | (survey, general) "On the Cross-Dataset Generalization of ML for NID" | arXiv 2402.10974 | Multiple public NIDS datasets (cross-dataset, not GeNIS) | Not reported | Not reported | XGBoost and others | Not reported | Cross-**dataset** train/test (train on dataset A, test on dataset B) | Not applicable (cross-dataset, not cross-scenario within one dataset) | No | Not reported | Studies generalization degradation across *different* datasets/network environments | Does not use GeNIS; addresses a different axis of generalization (dataset-to-dataset, not scenario-to-scenario within one controlled dataset) | Not reported | Related to, but methodologically distinct from, cross-scenario generalization within a single dataset |
| 6 | (example) "Multi-Level Distributional Entropy for Explainable NID" | arXiv 2606.29797 | CICIDS-2017, CICIDS-2018, NSL-KDD (not GeNIS) | Reports per-split n_test values in results tables | DDoS, PortScan, Bot, Infiltration, etc. | LightGBM, RF | Not reported (paper's own entropy features) | Compares **random stratified split vs. temporal split** (Mon–Thu train → Fri test) on the same dataset | Uses native timestamps for temporal split | Partial (evaluation-time only) | Detection rate (DR), false-alarm rate (FAR) | Detection rate **collapses** under temporal split for several attack categories versus random split (e.g., DDoS DR 0.182 vs. much higher under random CV) | Confirms random splits can be badly over-optimistic; does not use GeNIS | Not reported | **Directly supports the hypothesis motivating this project**: non-random splits reveal performance drops hidden by random splits, on a different dataset family |
| 7 | Kostas, Just, Lones — IoTGeM | arXiv 2401.01343 | IoT network traffic (not GeNIS) | Not reported | ACK, HTTP, SYN, MHD, PS, UDP attack types | Multiple ML models with a rolling-window feature-extraction approach | GA-guided, cross-dataset feedback | Strict train/test isolation; cross-**dataset** tests | Rolling window over packets | Yes — windowed features | Partially (window-based, not explicit sequence model) | F1 (up to 99% on several attack types under strict isolation) | Window-based features generalize better than flow-based ones under strict cross-dataset evaluation | Does not use GeNIS; windowing is feature-engineering-level, not a sequence model (e.g., not RNN/Transformer over ordered flows) | Not reported | Supports that temporal/windowed context can help generalization, but via a different mechanism and a different dataset |

---

## 4. Candidate research questions

### Candidate A — Cross-scenario generalization only
- **Title:** *"How Optimistic Are Random Splits? Cross-Scenario Generalization of Machine-Learning Intrusion Detection on the GeNIS Dataset"*
- **RQ:** Does classification performance degrade when models are evaluated on entirely held-out attack scenarios instead of a random flow split?
- **Hypothesis (testable, non-directional as required):** H1 — mean macro-F1 under scenario-based holdout differs from mean macro-F1 under random holdout. H0 — no significant difference.
- **IV:** split strategy (random vs. leave-one-scenario-out).
- **DV:** macro-F1, FPR, per-class recall.
- **Baselines:** RF, XGBoost, LightGBM, MLP (replicate the FPS 2025 paper's non-temporal models under a fairer split).
- **Overlap:** heavily reuses the FPS 2025 paper's model family and metrics; the *only* new axis is the split strategy.
- **Remaining gap it closes:** whether the published near-100% GeNIS results are split-dependent.
- **Difficulty:** Low–Medium. **Compute:** Low (tabular, tree models mainly).
- **Novelty:** Real but narrow — a single-axis methodological correction.
- **Risk of duplication:** Low, but the contribution alone is arguably "too thin" for a flagship project (it is closer to a methodological note than a full study).

### Candidate B — Sequence-aware vs. flow-independent classification (temporal-context only)
- **Title:** *"Does Sequence Matter? Flow-Independent vs. Sequence-Aware Deep Learning for Network Intrusion Detection on GeNIS"*
- **RQ:** Does explicit multi-flow temporal context (via LSTM/GRU/1D-CNN over ordered flow windows) improve detection compared to a flow-independent classifier on the same features?
- **Hypothesis:** H1 — sequence-aware models achieve different macro-F1 than flow-independent models on the same test set. H0 — no significant difference.
- **IV:** model family (flow-independent MLP/tree vs. sequence-aware LSTM/GRU/1D-CNN) and sequence length.
- **DV:** macro-F1, FPR, latency, model size.
- **Baselines:** MLP, RF/XGBoost as flow-independent baselines (matches FPS 2025 paper's setup, correctly cited as such).
- **Overlap:** the flow-independent half duplicates existing GeNIS baselines almost exactly; the sequence-aware half is genuinely new for GeNIS.
- **Remaining gap it closes:** whether GeNIS's explicitly "sequential attack scenario" design (its headline feature) is actually exploited by any model so far — currently it is not.
- **Difficulty:** Medium (requires correct sequence construction, ordering, leakage-safe windowing — this is exactly the part the instructions ask to justify carefully).
- **Compute:** Medium (RNN training on windowed data; still CPU/laptop-GPU feasible at moderate subset sizes).
- **Novelty:** Good — closes a gap the existing GeNIS paper explicitly and correctly identifies as unaddressed (their own LSTM was non-sequential).
- **Risk of duplication:** Low.

### Candidate C — Flow-interval sensitivity (5 s vs. 10 s vs. 30 s vs. 60 s)
- **Title:** *"Does Flow-Interval Duration Matter? A Multi-Interval Evaluation of Machine-Learning Intrusion Detection on GeNIS"*
- **RQ:** How does the choice of flow-export interval (5/10/30/60 s) affect detection performance and cross-scenario generalization?
- **Hypothesis:** H1 — performance/generalization varies significantly across flow intervals. H0 — no significant variation.
- **IV:** flow interval.
- **DV:** macro-F1, FPR, generalization gap (random-split score − scenario-split score).
- **Baselines:** same tree/MLP baselines, run once per interval.
- **Overlap:** none — the existing GeNIS paper used only the 60 s interval; no one has compared intervals.
- **Remaining gap it closes:** whether GeNIS's four-interval modular design (a distinguishing feature of the dataset versus older single-interval datasets) actually matters for detection quality.
- **Difficulty:** Low–Medium (mostly re-running the same pipeline four times; the interesting part is the analysis of *why* differences occur).
- **Compute:** Medium (4× data volume to process, though each interval's flow count differs).
- **Novelty:** Good, and cleanly novel, but by itself is a narrower "hyperparameter sensitivity" study rather than a methodological or modeling contribution.
- **Risk of duplication:** Very low.

### Candidate D — Combined: temporal context × cross-scenario generalization (refines the original proposed title)
- **Title:** *"Temporal Context and Cross-Scenario Generalization in Machine-Learning Network Intrusion Detection on the GeNIS Dataset"* (working title retained from your original proposal, scope tightened based on the literature)
- **RQ1:** Does cross-scenario evaluation reveal a generalization gap that random-split evaluation hides on GeNIS? **RQ2:** Does giving models explicit temporal (multi-flow sequence) context change that generalization gap, relative to flow-independent models? **RQ3 (secondary/ablation, time-permitting):** Is any effect from RQ1/RQ2 sensitive to the flow-export interval?
- **Hypotheses:** H1a (random vs. scenario-split performance differs); H1b (sequence-aware vs. flow-independent performance differs, and this difference itself changes between random-split and scenario-split evaluation — this is the actual novel interaction effect); H0a/H0b as the corresponding null hypotheses.
- **IV:** split strategy (random / leave-one-scenario-out) × model family (flow-independent vs. sequence-aware) × (optionally) flow interval.
- **DV:** macro-F1, per-class recall/precision, FPR, generalization gap, latency/throughput, model size.
- **Baselines:** RF, XGBoost, LightGBM, MLP (flow-independent, replicating and correctly citing the FPS 2025 paper) vs. 1D-CNN and/or LSTM/GRU over explicit flow windows (sequence-aware, novel for GeNIS).
- **Existing research that overlaps:** the flow-independent arm overlaps substantially with the FPS 2025 paper (must be cited as the baseline being reproduced/extended, not reinvented); the general "random split is optimistic" finding overlaps with non-GeNIS papers like the entropy paper (item 6 in the matrix) and the broader generalization literature (items 5, 7); Parlanti & Catania (item 4) motivates but does not implement this.
- **Remaining gap:** the specific **interaction** — whether temporal context specifically helps (or does not help, or only helps under harder splits) — is unaddressed anywhere in the literature I found, on GeNIS or, as far as verifiable, on any comparably modular dataset.
- **Difficulty:** Medium–High (this is a full 2×2 or 2×2×4 factorial design; it requires the most careful engineering — leakage-safe sequence construction, scenario-aware splitting, class imbalance handling per split).
- **Compute:** Medium, staged (see Section 7 of the eventual proposal — Stage 2 will define concrete subset sizes).
- **Novelty:** Strong — it is the union of gaps A and B, evaluated as an interaction rather than as two disconnected side studies, which is more scientifically interesting than either alone and is explicitly not something any existing GeNIS paper does or proposes as future work.
- **Risk of duplication:** Low, provided the flow-independent baselines are clearly framed as *reproducing and extending* the FPS 2025 paper's role (a legitimate and expected part of rigorous comparison), not as a novel contribution in themselves.
- **Expected scientific value:** High relative to A or C alone; comparable ambition to B but adds the generalization axis that turns "does temporal context help" into a more decision-relevant question ("does it help in the regime that matters for real deployment, i.e., unseen attack campaigns").

---

## 5. Scoring against your 10 selection criteria

| Criterion | A (scenario only) | B (sequence only) | C (interval only) | D (combined) |
|---|---|---|---|---|
| 1. Genuine research gap | Yes, narrow | Yes | Yes, narrow | Yes, and the *interaction* is the least explored |
| 2. Feasible implementation | Very easy | Moderate | Easy | Moderate–hard but scoped in stages |
| 3. Publicly reproducible data | Yes | Yes | Yes | Yes |
| 4. Reasonable compute | Yes (laptop) | Yes (laptop/small GPU) | Yes, larger storage | Yes if staged (Stage A–D subsets), see below |
| 5. Scientifically meaningful experiments | Moderate | Good | Moderate | Best |
| 6. Clear baseline | Yes (FPS25 paper) | Yes (FPS25 paper) | Yes (FPS25 paper, 60 s only) | Yes (FPS25 paper) |
| 7. Measurable contribution | Small | Medium | Small | Largest, but must be scoped honestly |
| 8. Cybersecurity/AI connection | Strong | Strong | Strong | Strong |
| 9. Can produce a meaningful negative result | Yes ("no significant drop") | Yes ("sequence context does not help") | Yes ("interval doesn't matter") | Yes, and *more informative* negative results are possible (e.g., "helps under random split but not under scenario split," which is itself a publishable finding) |
| 10. Reproducible by another researcher | Yes | Yes | Yes | Yes, with disciplined engineering (Stage 2 will spell out exact leakage-prevention rules) |

---

## 6. Recommendation

**Recommended topic: Candidate D**, with **RQ3 (flow-interval sensitivity) demoted to an optional ablation** rather than a primary research question, and with the **primary experimental focus on RQ1 × RQ2** (the interaction between split strategy and temporal modeling) at a **single flow interval** (60 s, matching the only prior GeNIS ML benchmark, to keep the comparison to existing work clean) — with interval sensitivity added back in only if Stage A/B compute checkpoints (defined in Stage 2) show it's affordable.

**Why not A, B, or C alone:** each is individually defensible and lower-risk, but each is closer to a single-variable methods note than a project with a real "story." D subsumes A and B as its two main effects and is the only candidate whose central finding cannot be predicted from the existing FPS 2025 paper or from generic ML intuition — it specifically asks whether a modeling choice (sequence-awareness) matters *differently* depending on how rigorously you evaluate generalization, which is a genuinely open, decision-relevant question and lets a negative, positive, or mixed result all be scientifically interesting.

**Why not switch datasets:** GeNIS remains the strongest choice. It is newer and more carefully engineered than CICIDS2017/HIKARI2021 (both of which have documented, citable flaws in flow-exporter feature extraction — see the FPS 2025 paper's related work, matrix row 2), it has almost no prior work on it (2 studies found, versus a saturated literature on CICIDS2017/NSL-KDD/UNSW-NB15), it was purpose-built with multi-stage sequential scenarios and multiple flow intervals — exactly the structure this project needs — and it is a manageable size (2.8M flows; the 60 s interval alone will be far smaller and laptop-tractable). No alternative dataset found in this search offered a stronger combination of scenario structure, recency, and low duplication risk.

**Novelty assessment:** Medium-high confidence this is a genuine gap, based on (a) direct reading of the only existing GeNIS ML paper, which explicitly states its LSTM was not given sequential input and does not propose scenario-based evaluation as future work, and (b) a taxonomy/survey paper (Parlanti & Catania 2025) that independently identifies inter-flow sequential and window-based methods as an under-studied category across the wider NIDS literature. Residual risk: a paper could exist in a venue not indexed well by the tools available here (e.g., a very recent non-English-language venue, a workshop paper without a strong web footprint, or something published between this search and when you start Stage 2). Stage 2 should open with one more targeted check immediately before implementation begins, and the manuscript's Related Work section must use "to the best of our systematic search" language rather than an unqualified novelty claim.

**Key risks to flag honestly before you approve:**
1. **Class imbalance is severe** (in the FPS 2025 paper's binary table, malicious:benign ≈ 92.6%:7.4%); scenario-based splitting will make this worse in some folds (e.g., a held-out scenario might contain very few benign flows), which must be handled explicitly (Stage 2/3) rather than papered over.
2. **Leave-one-scenario-out with only 8 attack + 3 benign scenarios** gives a small number of folds — statistical power for the cross-split comparison will be modest; this must be stated as a limitation, not hidden.
3. **Sequence construction is non-trivial and leakage-prone** — flows must be ordered and windowed *within* a scenario, and scenario boundaries must never be crossed by a sequence, or the "cross-scenario" test becomes contaminated. This is exactly why the instructions require an explicit, justified protocol in Stage 2 rather than an assumed one.
4. **The likely outcome may be a negative or mixed result** given how close to ceiling performance already is on GeNIS with flow-independent models (F1 > 99.9% in the FPS 2025 paper on their split). This is scientifically fine and explicitly welcomed by your instructions, but you should go in expecting that "no significant improvement, and here's why" is a very plausible and legitimate ending — not something to be avoided.

---

## 7. What Stage 2 will need from you before it starts

Nothing blocking — Stage 2 can start as soon as you approve a topic. Two decisions worth making explicitly now, though, since they shape the whole design:
- Confirm you're comfortable with the **60 s-interval-first, interval-sensitivity-as-optional-ablation** scoping (keeps the core project laptop-feasible; Stage 2 will give concrete RAM/CPU/runtime numbers per stage).
- Confirm the working title — keep your original, or use the slightly narrowed version above (*"Temporal Context and Cross-Scenario Generalization in Machine-Learning Network Intrusion Detection on the GeNIS Dataset"*)? The only change is dropping "sequential" ambiguity and being explicit that this is single-dataset, cross-scenario (not cross-dataset).

---

## References gathered so far (for `references.bib` in Stage 4)

1. Silva, M., Pinto, D., Vitorino, J., Gonçalves, J., Maia, E., Praça, I. (2025). GeNIS: A modular dataset for network intrusion detection and classification. *Data in Brief*, 60, 111487. https://doi.org/10.1016/j.dib.2025.111487
2. Silva, M., Pinto, D., Vitorino, J., Maia, E., Praça, I., Amorim, I., Viamonte, M.J. (2026). Binary and Multiclass Cyberattack Classification on GeNIS Dataset. In *Foundations and Practice of Security (FPS 2025)*, LNCS 16402, Springer. https://doi.org/10.1007/978-3-032-20018-1_18 (preprint: arXiv:2511.08660)
3. Magalhães, D. et al. (2026). Machine Unlearning for the XGBoost Model with Network Intrusion Datasets. arXiv:2606.19220.
4. Parlanti, T.S., Catania, C.A. (2025). Temporal Analysis Framework for Intrusion Detection Systems: A Novel Taxonomy for Time-Aware Cybersecurity. arXiv:2511.03799 (submitted to *Computer Networks*).
5. Pinto, D., Amorim, I., Maia, E., Praça, I. (2024). A novel approach to network traffic analysis: the HERA tool. *2024 IEEE 23rd International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom)*. https://doi.org/10.1109/TrustCom63139.2024.00255
6. Pinto, D. et al. (2025). A review on intrusion detection datasets: tools, processes, and features. *Computer Networks*, 262, 111177. https://doi.org/10.1016/j.comnet.2025.111177
7. arXiv:2402.10974 — On the Cross-Dataset Generalization of Machine Learning for Network Intrusion Detection. (Full author list not yet verified — mark unverified until confirmed before citing in the manuscript.)
8. arXiv:2606.29797 — Multi-Level Distributional Entropy for Explainable Network Intrusion Detection. (Author list not yet verified — mark unverified until confirmed.)
9. Kostas, K., Just, M., Lones, M.A. IoTGeM: Generalizable Models for Behaviour-Based IoT Attack Detection. arXiv:2401.01343.
10. Engelen, G. et al. (2021). Troubleshooting an intrusion detection dataset: the CICIDS2017 case study. *2021 IEEE Security and Privacy Workshops (SPW)*. https://doi.org/10.1109/SPW53761.2021.00009 (cited via FPS 2025 paper's related work — should be independently verified in Stage 4).

**Note:** items 7 and 8 are flagged "unverified" per your No-Fabrication Policy — I have their content and arXiv IDs from search snippets but have not independently confirmed full author lists. These will be re-verified (or dropped) before they appear in the final manuscript's reference list.

---

**STOP — this is the end of Stage 1.** Please review the recommended topic (Candidate D, scoped as above), the literature matrix, and the flagged risks, and let me know:
1. Approve Candidate D as-is, or choose A/B/C, or ask for a different combination.
2. Any change to the working title.
3. Any concern about the scoping (60 s-first, interval as optional ablation) before Stage 2 (final RQs, hypotheses, methodology, experimental design, repo structure) begins.
