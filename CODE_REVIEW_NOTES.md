# Code Review Notes

Observations that require your judgment. None of these have been silently changed in the refactor — they are flagged for you to decide.

---

## 1. File landscape — RESOLVED

**Status: resolved.** `project_presentation_model-1.py` is the paper's actual code. It generates all six paper figures (CCF, Granger, IRF, rolling R², HAC effect sizes, feature importance) and the Diebold-Mariano statistics reported in Table 1.

The repo structure now reflects this:

- **Main pipeline** (`src/`): refactored from `project_presentation_model-1.py` — the paper reproduction
- **Extensions** (`extensions/trading_simulation/`): backtest code from `economics_multi_model.py`, labeled as illustrative

No fabrication. Earlier integrity concern retracted.

---

## 2. AI disclosure — RESOLVED (Option C)

**Status: resolved.** Option C selected. This will appear in `README.md` as:

> This project used AI tools throughout, including for code authorship assistance on portions of the machine learning pipeline, literature discovery, and prose drafting. The authors designed the research question, constructed the data pipeline's identification strategy (including the lookahead guard), selected the econometric framework, interpreted results, and bear responsibility for all claims in the paper. Citations were verified manually against primary sources.

**Note:** The final clause ("citations were verified manually") only becomes true *after* you complete `CITATION_CHECKLIST.md`. Do not publish the repo with that wording until citation verification is actually done.

---

## 3. Hardcoded path

**Severity: medium (blocks reproduction)**

```python
base_path: str = "/Users/isaiahchoi/Desktop/Junior Winter Term/Mathematical Finance/Final Project/final model"
```

Replaced with config-driven path (CLI flag / env var / default to repo-relative `data/`). No methodological change.

---

## 4. 10^-5 gap truncation — justify or flag

**Severity: medium (methodological)**

```python
df_target['Gap'] = np.where(df_target['Gap'].abs() < 1e-5, 0.0, df_target['Gap'])
```

With N ≈ 153 per ticker, threshold choice can matter. Before publishing, consider:
- How many observations this truncates (roughly)
- Why 10^-5 specifically vs. 10^-4 or 10^-6
- A robustness check showing results don't change materially under alternative thresholds

Interviewers will ask. Have an answer ready.

---

## 5. The 60.1% directional accuracy result

**Severity: high (recruiting-specific)**

60.1% is 1 of 7 configurations. The other 6 are ~49–53%. The paper's Section 5.4 correctly flags this:

> "The accuracy of 60.1% is the only result demonstrating strong unconditional predictive power. For the remaining tickers and baskets, the proximity of the results to 50% suggests that a simple directional strategy without selective participation filters is insufficient."

For the resume: lead with HAC OLS betas (robust across all 7 configs) rather than directional accuracy. We discussed this earlier.

For the public repo README: lead with the robust result, mention 60.1% with context.

---

## 6. Random seed reproducibility

**Severity: low (housekeeping)**

`random_state=42` is set on Random Forest. Train/test split is deterministic (chronological). VAR and Granger are deterministic. Main source of non-determinism: thread-count-dependent sklearn internals.

For strict reproducibility, `scripts/reproduce_paper.py` will set `OMP_NUM_THREADS=1` and `n_jobs=1`.

---

## 7. Rolling R² is display-capped

**Severity: low (disclosure)**

```python
loc_r2 = max(-0.5, 1 - (mse_model / (mse_naive + 1e-8)))
```

The rolling R² series is bounded at -0.5 for plotting clarity. The paper's Table 1 reports unbounded values. Worth noting in the methodology docs so readers understand the rolling R² plot's floor is display-only, not a methodological choice.

---

## 8. Hedging practices not controlled for

**Severity: medium (limitations)**

The three firms (Sumitomo, Mitsubishi, Mitsui) have different capital structures and hedging practices. The paper acknowledges this briefly but doesn't control for it. Worth expanding in `docs/limitations.md`.

---

## 9. Lookahead guard — add unit test

**Severity: medium (defensibility)**

The code enforces lookahead at runtime via assertion. Good. But no unit test explicitly constructs a violating case to confirm the guard fires. The refactor will add `tests/test_lookahead_guard.py` with explicit positive and negative test cases.

---

## 10. Trading simulation — EXCLUDED

**Status: resolved.** The `economics_multi_model.py` trading simulation is excluded from the public repo.

Rationale: the paper documents that the FX-to-gap signal decays within the opening auction (Granger causality null at lags 1+, impulse response reverses at step 1). A daily close-to-open backtest cannot cleanly capture this — exploitation requires intraday exit timing the current data doesn't support. Including a Sharpe-ratio backtest would contradict the paper's own framing of the effect's boundedness.

The exclusion is a feature, not a gap. The repo and paper now say the same thing. Intraday data + quanto pricing structures are flagged as the path forward in `docs/limitations.md` and the paper's Section 7/Future Work.

---

## What you must resolve before publishing

1. **Choose** one of the three AI disclosure options (A, B, or C)
2. **Verify** all 15 citations manually against Google Scholar
3. **Confirm** the resume framing of the 60.1% result
4. **Justify or robustness-check** the 10^-5 truncation rule

Items 1 and 2 are most important.
