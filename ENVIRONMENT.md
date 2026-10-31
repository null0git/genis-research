# ENVIRONMENT.md — Recorded Environment for Reproducibility

This file records the environment this project's code was **written and
smoke-tested** against. Per `STAGE2_PROPOSAL_AND_METHODOLOGY.md` and the
manuscript's Reproducibility section, exact bit-for-bit reproduction across
different hardware/OS/library-minor-version combinations is not guaranteed
(this is a documented property of the underlying libraries — e.g.
floating-point non-determinism in multi-threaded tree ensembles and in GPU
kernels — not a gap specific to this codebase). A close reproduction on
the pinned versions below is the realistic target.

**Fill in the italicized fields below on your own machine, once you run
the full pipeline there — do not leave this file with the placeholder
values below in the version you publish.**

## Python

- Python version: *fill in from `python --version` on your training machine*
- Operating system: *fill in, e.g. "Ubuntu 22.04" / "macOS 14.5" / "Windows 11"*

## Core scientific-computing stack (pinned in `requirements.txt`)

```
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
PyYAML==6.0.1
scikit-learn==1.5.0
xgboost==2.0.3
statsmodels==0.14.2
torch==2.3.1
shap==0.45.1
matplotlib==3.9.0
seaborn==0.13.2
tqdm==4.66.4
requests==2.32.3
pytest==8.2.2
```

Install with:

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## GPU / CUDA (optional — this project's design does not require a GPU)

- GPU present: *yes/no*
- If yes: GPU model, CUDA version (`nvidia-smi`), and whether you installed
  a CUDA-enabled `torch` build instead of the CPU-only default above.

## Git

- Commit hash used to produce the manuscript's reported results:
  *fill in with `git rev-parse HEAD` at the time you run the final
  experiments — this is what makes a specific manuscript result
  traceable to a specific version of the code, per the Result
  Provenance system in EXPERIMENTS.md.*

## Dataset version

- Zenodo DOI: `10.5281/zenodo.14919237`
- Download date: *fill in*
- If Zenodo publishes dataset versions/revisions in the future, record
  the specific version used here.

## Random seeds

All experiment configs (`configs/experiments/*.yaml`) fix their own seeds
explicitly (see each file's `split.seeds` field, or `random_state: 42` in
model configs). `src/utils/seeding.py::set_all_seeds` is called at the
start of every seed/fold iteration in `src/training/train.py`. No seed is
left to its library default anywhere in the codebase.

## Known non-determinism sources (documented, not hidden)

- `RandomForestClassifier` and `XGBClassifier` with `n_jobs=-1` can show
  tiny run-to-run floating-point differences under multi-threading, even
  with a fixed `random_state`, depending on your CPU/thread count. If
  exact reproduction matters more than speed, re-run with `n_jobs=1` in
  the relevant `configs/models/*.yaml` file.
- `torch.use_deterministic_algorithms(True, warn_only=True)` is set in
  `set_all_seeds`, but some GPU kernels do not have a deterministic
  implementation and will fall back with a warning rather than an error —
  check your run's logs for any such warning if running on GPU.
