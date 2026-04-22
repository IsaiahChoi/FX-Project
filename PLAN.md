# Execution Plan — FX Gap Research Repository

**This document outlines exactly what will be produced. Review before approving Phase 2.**

---

## Source code clarification

You have two relevant uploaded code files:

- **`project_presentation_model-1.py`** (621 lines) — **PRIMARY SOURCE.** This is the paper's actual code. It generates all six paper figures (CCF, Granger heatmap, IRF, rolling R², HAC effect sizes, feature importance) plus Diebold-Mariano statistics. The main refactor in `src/` is based on this file.
- `economics_multi_model.py`, `Final_multi_model.py`, `Final_Project_Results.py` — **NOT INCLUDED.** Intermediate drafts and the trading simulation. Excluded from the public repo (see CODE_REVIEW_NOTES.md #10 for the principled rationale — the paper's signal decays within the opening auction, so a daily-frequency backtest would contradict the paper's own framing).

---

## Phase 1 — Scaffolding (COMPLETE)

- [x] Directory structure
- [x] `LICENSE` (MIT)
- [x] `.gitignore` (with strong data protections)
- [x] `requirements.txt`
- [x] `pyproject.toml`
- [x] `PLAN.md` (this file)
- [x] `CODE_REVIEW_NOTES.md`
- [x] `src/config.py` + `src/__init__.py` (starter refactored config)

## Phase 2 — Code Refactor (PENDING APPROVAL)

The 621-line `project_presentation_model-1.py` will be split into the following modules. **No methodological changes.** All hyperparameters, thresholds, and logic preserved exactly.

### `src/config.py` ✅
Already created. `ResearchConfig` dataclass with env-var-driven paths.

### `src/data_engine.py`
- `DataEngine` class with methods: `load_fx()`, `get_data(ticker)`, `_load_excel_sheets()`, `_find_sheet_for_ticker()`, `_parse_equity_sheet()`, `_inspect_and_load()`
- Timestamp-level lookahead guard preserved
- 10^-5 gap truncation preserved
- Overnight window (15:00–08:50 Asia/Tokyo) preserved

### `src/features.py`
- `combine_datasets(engine, tickers)` function
- Feature construction: FX_Vol, FX_Ret, FX_Ret_2D, FX_Ret_5D, Gap_Lag1, Gap_Lag2, day-of-week dummies, equity dummies
- All feature definitions preserved exactly

### `src/models.py`
- HAC OLS fitting helpers (core and full specifications) with Newey-West standard errors
- Random Forest regressor and classifier wrappers (500 trees, balanced class weights, random_state=42)
- VIF computation
- Newey-West lag selection heuristic: `m = floor(4 * (n/100)^(2/9))`

### `src/econometrics.py`
- **Cross-correlation function** (`sm.tsa.stattools.ccf`) at lags 0–5
- **Granger causality tests** (`grangercausalitytests`) at lags 1–5
- **VAR(2) fitting and orthogonalized impulse response functions** with bootstrap confidence intervals
- **Expanding-window rolling OOS R²** with 21-day moving average (from `compute_rolling_r2`)
- **Diebold-Mariano test** against naive zero-forecast benchmark
- **Mutual information** (`mutual_info_regression`)
- Directional accuracy + one-sided binomial test

### `src/evaluation.py`
- `ReportingEngine` class (refactored from the paper code)
- Consolidated results table generation
- Holm-Bonferroni multiple testing adjustment

### `src/plotting.py`
All six paper figures as separate functions:
1. `plot_ols_effect_sizes()` → Figure 1
2. `plot_cross_correlation()` → Figure 2
3. `plot_granger_heatmap()` → Figure 3
4. `plot_impulse_response()` → Figure 4
5. `plot_rolling_r2()` → Figure 5
6. `plot_feature_importance()` → Figure 6

### `src/main.py`
- CLI entry point via argparse
- Subcommands: `reproduce-paper`, `run-econometrics`
- Logging configuration

## Phase 3 — Documentation (PENDING)

### `README.md` (final version)
1. Title + one-line tagline
2. TL;DR (3–4 sentences, brutally honest)
3. Headline findings (no cherry-picking)
4. "What this paper is and isn't" explicit paragraph
5. Repository structure tree
6. Reproduction instructions
7. Data section (real data not included, schema documented)
8. Authors
9. AI disclosure (YOU CHOOSE from A/B/C in CODE_REVIEW_NOTES.md)
10. License
11. BibTeX citation

### `docs/methodology.md` (~2,500 words)
Plain-English technical writeup. Prose, not bullets.

### `docs/limitations.md`
Explicit enumeration of every caveat.

### `docs/website_summary.md` (~1,000 words)
Portfolio-page writeup.

### `docs/interpretation.md`
Short companion piece on how to read the results honestly.

## Phase 4 — Supporting Files (PENDING)

### `scripts/generate_synthetic_data.py`
Demo data matching expected schema so anyone can reproduce the pipeline without Bloomberg access.

### `scripts/reproduce_paper.py`
Single-command entry: `python scripts/reproduce_paper.py --data-dir data/synthetic` regenerates all figures and Table 1.

### `data/README.md`
Data schema documentation. Instructions for sourcing real Bloomberg data.

### `tests/`
- `test_lookahead_guard.py` — verify the timestamp cutoff rejects violating FX bars
- `test_feature_construction.py` — FX_Ret, FX_Vol, cumulative returns, overnight window mapping
- `test_data_schema.py` — validates synthetic data conforms to expected schema
- `test_econometrics_smoke.py` — smoke tests that econometrics functions run on synthetic data

### `CITATION_CHECKLIST.md`
15-item checklist for manual Google Scholar verification.

### `HANDOFF.md`
Final checklist of what you need to do before publishing.

---

## Stopping points

- **End of Phase 1** (now): review scaffolding and CODE_REVIEW_NOTES. ← YOU ARE HERE
- **End of Phase 2**: review refactored code for faithfulness to original
- **End of Phase 3**: review documentation tone and claims
- **End of Phase 4**: final review before packaging

## Hard constraints

1. No methodological changes to the analysis
2. No new experiments or reinterpretations
3. No fabricated or embellished results
4. No citation verification (flagged for manual review)
5. No raw data in the repo
6. All AI-assistance disclosure decisions deferred to you

---

## What I need from you before Phase 2

**All decisions resolved.** Ready to proceed with code refactor.

- [x] AI disclosure: Option C
- [x] Trading simulation: excluded (principled — see CODE_REVIEW_NOTES.md #10)
- [x] 60.1% framing: lead with HAC OLS betas, mention directional accuracy with context
