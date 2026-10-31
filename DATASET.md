# DATASET.md — GeNIS Acquisition and Preparation

This project uses **GeNIS (GECAD Network Intrusion Scenarios)**, a 2025 dataset
created by GECAD/ISEP researchers, at the **60-second flow interval only**
(see `STAGE2_PROPOSAL_AND_METHODOLOGY.md`, Section 1, for why the scope is
limited to this one interval).

## 1. Official source and citation

- **Zenodo record:** https://zenodo.org/records/14919237
- **DOI:** `10.5281/zenodo.14919237`
- **Dataset paper:** Silva, M., Pinto, D., Vitorino, J., Gonçalves, J., Maia, E.,
  Praça, I. (2025). "GeNIS: A modular dataset for network intrusion detection
  and classification." *Data in Brief*, 60, 111487.
  https://doi.org/10.1016/j.dib.2025.111487

**Citation (BibTeX)** — see `references.bib`, entry `silva2025genis`.

**License:** the dataset paper is published open access under a CC BY license.
**Action required before publishing this repository:** open the Zenodo record
page directly and confirm the license tag shown there for the *dataset files
themselves* (Zenodo licenses are set per-deposit and can differ from the
license of the accompanying journal article) — do not assume CC BY for the
data files without checking the record page yourself.

## 2. What's actually in the dataset

- **Raw data:** PCAPNG packet captures. 37,681,001 packets total, across
  all scenarios and both benign/attack traffic.
- **Preprocessed data:** flow tables (CSV), generated with the HERA flow
  exporter, at **four flow-export intervals: 5s, 10s, 30s, 60s**.
  2,806,168 flows total across all four intervals combined (per the dataset
  paper). **This project uses the 60s-interval files only** — the 60s-only
  flow count is smaller than this total and must be measured directly by
  running `src/data/audit.py` (see Section 5 below), not assumed.
- **Scenarios:** 8 attack scenarios (multi-stage: reconnaissance →
  credential bruteforce → denial-of-service, targeting FTP, web, and Active
  Directory services) + 3 benign scenarios (typical user activity,
  administrator activity, idle/background traffic).
- **Labels:** `BinaryLabel` (benign/malicious), `CategoryLabel` (attack
  category), `SubCategoryLabel` (finer-grained attack sub-type).

## 3. Download procedure

```bash
# Preferred: download only the preprocessed flow CSVs (skips the much
# larger raw PCAPNG archives, which this project's 60s-interval analysis
# does not need).
python -m src.data.download --config configs/data.yaml --only-preprocessed
```

This script:
1. Queries the Zenodo REST API for the record's file index.
2. Streams each matching file to `data/raw/`.
3. Verifies each download's MD5 checksum against the value Zenodo reports.

If the automated download fails (e.g. Zenodo API shape changes, or a file
requires browser-based access), download manually from
https://zenodo.org/records/14919237 and place the files under `data/raw/`.

**Expected storage:** the preprocessed-only download should be on the order
of a few hundred MB to low single-digit GB (2.8M flows across 4 intervals,
~125 columns); the full raw PCAPNG archive set is much larger and is NOT
needed for this project. If you download the raw archives anyway, budget
several tens of GB and expect multi-hour download times depending on
connection speed.

## 4. Extraction

If the downloaded files are ZIP archives, extract them into
`data/preprocessed/`:

```bash
mkdir -p data/preprocessed
unzip data/raw/<preprocessed_archive>.zip -d data/preprocessed/
```

Expected result: one or more CSV files under `data/preprocessed/`, with
filenames that (per the dataset's documented structure) should indicate
which flow interval each file corresponds to. **The exact naming
convention has not been independently re-verified by this project beyond
the published dataset descriptor** — this is exactly what
`src/data/audit.py` checks in the next step.

## 5. Verification / audit (REQUIRED before any modeling)

```bash
python -m src.data.audit --interval 60 --config configs/data.yaml
```

This script:
- Locates candidate 60s-interval CSV files.
- Reports every column name, dtype, row count, label distribution, and
  scenario distribution it finds.
- Flags any column `configs/data.yaml`'s `column_map` expects but doesn't
  find.
- Writes a full JSON summary to `results/processed/data_audit_summary.json`.

**Do not proceed to preprocessing or any experiment until this audit's
`missing_expected_columns` field is empty for every file.** If it isn't,
correct `configs/data.yaml`'s `column_map`, `drop_columns`, and
`scenarios` block to match what the audit actually found, and re-run the
audit.

## 6. Minimum subset for development (Stage A)

For initial pipeline development (Section 7 of
`STAGE2_PROPOSAL_AND_METHODOLOGY.md`, "Stage A"), do not load the full 60s
dataset. Either:
- Use `scripts/make_synthetic_sample.py` to generate a small,
  structurally-similar (but **not real**) sample for pure pipeline-logic
  testing, or
- Subsample two real scenarios (one attack, one benign) from the real
  downloaded data for an initial real-data smoke test before scaling up.

## 7. Known limitations to carry into the manuscript's Limitations section

- This project's own measured flow counts, class balance, and scenario
  identifiers (from the audit step) are the source of truth for anything
  reported in the manuscript — not the published dataset paper's
  aggregate, all-interval numbers, and not the FPS 2025 benchmark paper's
  split-specific numbers.
- GeNIS traffic is generated on the Airbus CyberRange simulation platform
  by two Kali attacker machines running specific tools; it is not a
  capture of a production network or of real-world attacker behavior.

## 8. Sequence construction and the shipped train/holdout split — RESOLVED

**This was an open question; it is now resolved against real, uploaded
GeNIS files, not guessed.** Summary of what was actually found (full
forensic trail in git history / conversation record, condensed here):

- `4-preprocessed` (the dataset's shipped, ready-to-use split) was
  confirmed **unusable for this project**: it strips `StartTime`,
  `LastTime`, and all other timing columns entirely, and its rows are
  confirmed shuffled (checked directly: ~50% of consecutive rows show a
  reversed `Offset`/`Seq` value, far beyond what file-concatenation alone
  would cause).
- `2-flows` (organized by attack **technique**, e.g.
  `attack-dos-icmp.csv`) was confirmed the wrong level of granularity:
  cross-referencing against `0-info/genis-ground-truth.csv` showed that
  `dos-icmp`, `dos-pushack`, and `dos-udp` all came from the **same single
  recording session** but live in three separate `2-flows` files — using
  them as LOSO fold units would leave same-session flows in the training
  set while "holding out" only the label, a real leakage path.
- **`3-scenarios/scenarios-60-sec/` is the correct source and is what this
  project uses.** 8 files, one per actual recording session
  (`scenario-1-disrupt.csv` ... `scenario-8-authtest.csv`). Confirmed
  directly by loading `scenario-7-authtest.csv`:
  - `StartTime` is real, format `HH:MM:SS` (not Unix epoch, not a full
    datetime), confirmed monotonically increasing in row order — **this
    stage is not shuffled**.
  - Benign flows (`BinaryLabel=0`) live inside each scenario file
    alongside its attack payload (25,445 benign / 4,234 attack in the
    file checked) — confirming there are no separate benign-only scenario
    files, and the correct scenario count is **8, not 11** as originally
    guessed in Stage 1.
  - `configs/data.yaml`'s `drop_columns` and `categorical_columns` were
    derived by **diffing this stage's 125 raw columns against Silva et
    al. (2026)'s published 87-column preprocessed schema programmatically**
    (not retyped by hand), so this project's feature-selection philosophy
    matches their baseline exactly, with one disclosed, deliberate
    exception: `StartTime` is in their drop list (they don't use temporal
    ordering) and this project keeps it, since that's the entire point of
    RQ2.
  - `IdleTime` was independently confirmed to leak a **raw Unix epoch
    timestamp on every single row** (not "time since last activity" as
    its name suggests) — it's already in Silva et al.'s drop list for
    unrelated reasons, but this project's own check found the leak
    independently and would have flagged it regardless.
  - `Sport`/`Dport` needed a numeric-coercion fix
    (`src/data/preprocessing.py::_coerce_numeric_with_hex`) — real values
    mix plain decimals (`"138"`), decimals with a trailing `.0`
    (`"35672.0"`), empty strings (no port for that protocol), **and
    hex-formatted codes** (`"0x7ce9"`, `"0x0008"`, etc.) for protocols
    without real TCP/UDP ports (ARP, ICMP, and similar). Confirmed against
    the real combined 8-scenario dataset: `"0x7ce9"` appears as a fixed
    constant exactly 65,536 (=2^16) times, a strong signal it's Argus's
    placeholder for "no applicable port," not per-flow information — but
    it's parsed and kept as a legitimate numeric value (its base-16
    integer equivalent) rather than discarded, so the "this flow's
    protocol has no real port" signal its consistency encodes isn't
    silently thrown away.
  - Several legitimate numeric columns (`sTos`, `dTos`, coerced
    `Sport`/`Dport`) contain genuine `NaN` values — `StandardScaler`
    raises on `NaN`, so median imputation (fit on the training split
    only, same rule as scaling) was added
    (`preprocessing.fit_imputer`/`apply_imputer`, wired into
    `src/training/train.py::_prepare_split_arrays` before scaling).
  - `Ssaddr`/`Sdaddr` were originally (wrongly) assumed to be encoded IP
    identifiers and slated for dropping; the official feature dictionary
    (`data/reference/genis-features.csv`) and real values (1–3798,
    right-skewed) confirm they're legitimate "number of connections with
    same service+address" count features — they stay in the model's
    feature set.
  - A full pipeline smoke test — load real `scenario-7-authtest.csv` →
    clean/encode → impute → scale → split → train Random Forest — ran
    successfully end to end, and a full sequence-construction pass (real
    `StartTime` ordering, `sequence_length=10`) also ran successfully,
    producing exactly the expected padding pattern.
- `find_interval_files` had a real bug, also caught by this process: it
  matched the interval number against the filename only, but GeNIS's real
  layout encodes the interval in the **parent folder** name
  (`scenarios-60-sec/`), not any individual filename. Fixed to match
  against the full relative path.

**Practical instructions:** extract only `3-scenarios.zip`'s
`scenarios-60-sec/` folder (skip `scenarios-5-sec/`, `-10-sec/`,
`-30-sec/` — not needed for this project's scope) to
`preprocessed_data_dir` (`configs/data.yaml`), then run the standard audit
step to confirm your own download matches what's described above before
training anything:

```bash
python -m src.data.audit --interval 60 --config configs/data.yaml
```
