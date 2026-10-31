# QUALITY_CONTROL_REVIEW.md — Self-Review Against the Project's QC Checklist

This runs the "Research Quality Control" checklist from the project brief
against what's actually in the repository, item by item, honestly —
including where something isn't done yet.

---

## Literature

- **Have all closely related papers been identified?** Best-effort yes,
  within the stated scope (`docs/STAGE1_LITERATURE_AND_TOPIC.md`, Section
  1: Google Scholar, arXiv, IEEE Xplore–indexed venues, ScienceDirect,
  PMC, Zenodo, as of August 2026). Not a PRISMA-protocol systematic
  review — the manuscript's Related Work section says so explicitly
  ("to the best of our systematic search...") rather than claiming
  completeness.
- **Has the closest GeNIS research been discussed?** Yes — both existing
  GeNIS ML papers (Silva et al. 2026, Magalhães et al. 2026) are read in
  full and cited specifically, not just listed.
- **Has novelty been checked?** Yes, and the specific mechanism is stated:
  the only GeNIS LSTM baseline explicitly says it wasn't given a real
  sequence; no GeNIS paper uses a scenario-based split. Residual risk
  (a paper this search missed) is stated plainly, not hidden.
- **Are citations real?** 7 of 10 `references.bib` entries are fully
  verified against a primary source (DOI or direct paper read). **3 are
  explicitly marked `UNVERIFIED`** in the file itself (`crossdataset2024`,
  `entropy2026nid`, plus a partial-verification note on
  `pinto2025reviewdatasets` and `magalhaes2026unlearning`) — these appear
  in the manuscript only as `[PLACEHOLDER CITATION — verify...]`, never as
  a resolved in-text citation. **Action still needed from you:** confirm
  or drop these before final submission — this project's No-Fabrication
  Policy means I won't silently resolve them for you.

## Methodology

- **Is there a clear research question?** Yes — RQ1/RQ2/RQ3, each with an
  explicit IV/DV and, for RQ1/RQ2, a stated null hypothesis
  (`STAGE2_PROPOSAL_AND_METHODOLOGY.md`, Section 1).
- **Are hypotheses testable?** Yes, and pre-registered against a specific
  test: Wilcoxon signed-rank, not chosen after seeing results (chosen in
  Stage 2, before any code was run).
- **Are baselines appropriate?** Yes, and each model's inclusion is
  justified individually (Section 5) rather than padded for a headcount —
  LightGBM and Transformer were deliberately excluded, with the reasoning
  stated rather than silently dropped.
- **Is data leakage prevented?** Yes, by construction, and this is the
  part of the project with the most direct evidence behind it: 9 automated
  tests in `tests/test_leakage.py` check exactly this (random-split
  disjointness, LOSO scenario isolation, temporal-split ordering,
  sequences never crossing a scenario boundary, scaler fit-on-train-only)
  and all 9 currently pass.
- **Are train/test splits justified?** Yes — Section 3 of the methodology
  doc states what each of the three splits measures and why, not just how
  to implement them.

## Experiments

- **Are experiments reproducible?** Structurally yes — every experiment
  has a config file, a runnable script, and a documented command
  (`EXPERIMENTS.md`). **Not yet verified against the real dataset** — see
  "What's still open" below.
- **Are results generated from code?** Yes by design — every table/figure
  script reads from `results/processed/*.json`, never from a hand-typed
  number, and the manuscript's placeholders name the exact script for
  each one.
- **Are multiple metrics used?** Yes — macro-F1, per-class precision/
  recall, FPR, FNR, accuracy (flagged as secondary), confusion matrices,
  plus training time/latency/model size/memory for RQ3.
- **Is class imbalance addressed?** Yes, via training-split-only class
  weighting, with the SMOTE/oversampling alternative explicitly rejected
  and reasoned through (Section 6) rather than silently omitted.
- **Is error analysis performed?** Structurally scaffolded (Section 16 of
  the manuscript, Table 05/Figure 04 generation scripts exist) but **the
  actual analysis cannot be written until real per-scenario results
  exist** — the manuscript placeholder is explicit that explanations must
  be grounded in real confusion matrices, not asserted in advance.
- **Are negative results allowed?** Yes, structurally: Section 18
  (Discussion) is written as a branching framework keyed to which result
  actually obtains, including an explicit "no significant gap" branch and
  a "sequence-aware models don't help, or help less under LOSO" branch,
  each with its own honest interpretation — not just a positive-result
  branch with a token caveat.

## Writing

- **Is every major claim supported?** Every claim in the Introduction/
  Related Work is either cited or explicitly flagged
  `[PLACEHOLDER CITATION]` pending verification. No invented numbers
  appear anywhere in `paper/manuscript.md` — I checked this directly:
  38 explicit `[...PLACEHOLDER...]` markers remain in the file, and the
  Abstract's numeric shell is explicitly marked "do not fill in by
  guessing."
- **Is the writing clear?** Reviewed once against the Humanizer skill's
  guidance (plain sentences, no inflated "opportunity"/"landscape"-style
  filler, no formulaic hedge-everything conclusions) during drafting; one
  flagged phrase ("purpose-built opportunity") was caught and rewritten.
  A second full pass specifically for AI-writing tells (over-qualification,
  em-dash overuse, "it's not just X, it's Y" constructions, false-range
  claims) has **not yet been run** on the final Results/Discussion
  sections, because those sections don't exist yet — they're
  placeholders. This pass is scheduled for Stage 6, once there's real
  prose to check.
- **Is it free from unnecessary filler?** Believed yes, no exhaustive
  re-audit performed beyond the check above.
- **Does it distinguish fact from interpretation?** Yes structurally —
  Section 18's branching framework is explicit that only the branch
  matching the real outcome should survive into the final manuscript, and
  each branch is phrased as a candidate explanation to check, not an
  assertion.
- **Does it avoid exaggerated novelty?** Yes — Section 4 uses
  "to the best of our systematic search," not "no one has done this,"
  and Section 3.1 explicitly frames the flow-independent baselines as a
  deliberate reproduction of Silva et al., not a novel contribution.

## Reproducibility

- **Can another researcher reproduce the experiments?** Structurally
  everything needed is present (config, script, documented command,
  environment pinning). **Not yet demonstrated on the real dataset** —
  see below.
- **Are commands documented?** Yes, in `EXPERIMENTS.md` and `README.md`,
  matched 1:1 with what the scripts actually accept (checked by running
  the RF path end-to-end, not just by reading the argparse definitions).
- **Are versions recorded?** `ENVIRONMENT.md` exists with the correct
  structure, but its version fields are still templated placeholders
  ("*fill in from `python --version`...*") — **this needs to be filled in
  from your actual training machine**, not from this sandbox, since this
  sandbox's Python/library versions aren't what you'll actually train on.
- **Are raw results preserved?** The mechanism exists (`results/raw/`,
  every experiment script's documented `> results/raw/eXX_terminal_output.txt`
  redirection) but no real raw output exists yet — only synthetic
  smoke-test output, which was deliberately deleted before packaging so
  it can't be mistaken for real results.
- **Are figures generated from scripts?** Yes, by construction — all 4
  figure scripts read only from `results/processed/*.json`.
- **Are tables generated from result files?** Yes, same as above, for all
  6 numbered tables plus the abstract-stats and appendix-C scripts.

---

## What's still open (the honest gap between "built" and "done")

1. **No *experiment* has been run against the full real GeNIS dataset yet
   — but the pipeline itself has now been validated against real GeNIS
   files, not just synthetic data.** You shared three real files
   (`genis-60-sec-test.csv`, `scenario-7-authtest.csv`, plus the official
   `genis-features.csv`/`genis-ground-truth.csv` metadata), and that
   process surfaced and fixed several real bugs a synthetic-only test
   never could have: wrong assumptions about `Ssaddr`/`Sdaddr`, a silent
   epoch-timestamp leak in `IdleTime`, a folder-vs-filename matching bug
   in `find_interval_files`, inconsistently-formatted `Sport`/`Dport`
   values, and missing-value handling that would have crashed
   `StandardScaler`. All fixed and re-tested against the real file. What
   remains is running the full 8-scenario dataset through E01–E07, not
   validating that the code works on real data at all.
2. **This sandbox still has no network access and cannot install
   `xgboost`/`torch`/`shap`/`statsmodels`.** The Random Forest path is
   proven correct on real data (Section above); XGBoost, MLP, 1D-CNN, and
   GRU are compiled and interface-consistent but unexecuted, and a full
   multi-scenario LOSO run (as opposed to LOSO's *mechanics*, which the
   synthetic tests do cover) has not been run against real data yet
   either.
3. **`ENVIRONMENT.md`'s version fields are placeholders** — fill these in
   from the machine you actually train on.
4. **3 citations are marked UNVERIFIED** and must be confirmed (or
   dropped) before submission.
5. **Stage 6 — writing the real Results, Discussion, Conclusion, and
   updating the Abstract — cannot start until real experiment output
   exists.** This is the actual next step, and it depends on you: run
   the pipeline per `README.md`'s reproduction protocol (or as much of it
   as your hardware allows, using the Stage A/B/C/D fallback plan in
   `STAGE2_PROPOSAL_AND_METHODOLOGY.md` Section 7 if needed), then share
   the resulting `results/processed/*.json` files back with me.

Once you have real output — even partial (e.g. just E01+E02, the core
RQ1 comparison) — I can move to Stage 6: verify it against the tests and
provenance chain, generate the real tables/figures, write the actual
Results/Discussion/Conclusion, update the Abstract, and do a final
consistency check that every number in the manuscript traces back to a
real file.
