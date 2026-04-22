# Limitations

An explicit enumeration of every known caveat. Where a limitation could be addressed in future work, the path forward is noted.

## Sample size and scope

The sample spans January 2023 to early 2026 — roughly three years of daily data per ticker, yielding approximately 153 observations per single-ticker experiment and 306 or 459 observations for the multi-stock baskets. For HAC OLS with four to six predictors, this is a modest sample; with Random Forest, it is small. Negative out-of-sample R² values across all experiments are consistent with a degree of overfitting that a larger sample would help correct. The paper's significant HAC betas should be interpreted as robust *within the realized sample*, not as evidence that the relationship would persist identically in a different market regime.

Only three firms are studied. All three are Japanese non-ferrous metal miners with USD-denominated revenue and JPY-denominated costs. Whether similar FX-to-gap relationships exist in other sectors — Japanese automotive, Japanese electronics, Korean semiconductors, or non-ferrous miners in other currency zones — is not tested. The conclusions cannot be extrapolated beyond this narrow sector without separate empirical work.

Only one currency pair is studied. USD/JPY is the relevant pair because of the firms' balance-sheet structure, but a broader test of the "Trading Place Hypothesis" would need to examine other currency pairs and other sector-currency pairings to establish the phenomenon's generality.

## Regime coverage

The sample period includes the Bank of Japan's gradual exit from yield curve control, several rounds of Federal Reserve policy shifts, and the Japanese yen's multi-year weakening trend. It does not include a full Japanese equity bear market, a material USD/JPY strengthening period, or a commodity-price-driven regime shift in the metals sector. Whether the observed relationship persists in materially different regimes — a sharp yen appreciation, an extended metals bear market, a period of sustained dollar weakening — is not testable with the current data.

## The 10^-5 gap truncation rule

Opening gaps with absolute value below 10^-5 are truncated to zero in the data engine. This is intended to mitigate mechanical rounding artifacts introduced by Bloomberg's pricing conventions. The threshold choice (10^-5 versus, say, 10^-4 or 10^-6) could materially affect the results in a sample this small. A robustness check under alternative thresholds is not included in the paper and should be added before the finding is relied upon in any downstream application.

## Hedging practices not controlled for

Sumitomo Metal Mining, Mitsubishi Materials, and Mitsui Mining & Smelting all have active FX hedging programs, but the specifics of their hedge ratios, instruments, and tenors differ materially across firms and change over time. The paper's regressions treat these firms as economically equivalent USD-exposed mining equities, but in reality their effective exposures to overnight USD/JPY moves are likely different from nominal book exposure. Failing to control for firm-level hedging is a source of noise that likely biases the reported coefficients toward zero (attenuation bias). The true FX-to-gap sensitivity, controlling for hedging, is probably larger than the 0.164–0.227 range reported.

## Multi-stock basket interpretation

The multi-stock baskets pool observations across firms with equity-specific dummies. This specification assumes that the coefficient on FX return is homogeneous across firms up to an intercept shift, which is unlikely to be exactly true given differing hedging practices, capital structures, and export mix. The pooled results should be read as sector-average effects rather than firm-specific ones.

## Out-of-sample evaluation uses a single split

The train/test split is a single chronological 80/20 split with a two-business-day embargo. A rolling or expanding walk-forward evaluation would produce more statistically reliable estimates of out-of-sample performance. With roughly 31 test observations per single-ticker experiment, the directional accuracy estimates have substantial sampling uncertainty — a 60% accuracy on 31 trades has a 95% confidence interval roughly from 41% to 77%, which overlaps both the "no signal" and "strong signal" hypotheses. The paper is explicit that the 60.1% Mitsubishi result is one outlier among seven configurations and should not be interpreted as a robust indicator of directional predictability.

## Transaction costs, liquidity, and execution are not modeled

The paper reports raw directional accuracy without any cost overlay because it does not claim to be a backtest of a trading strategy. Readers should not infer tradability from the reported numbers. In reality, capturing any portion of the opening-gap effect would require executing at or very near the opening auction itself — a fill regime with substantial bid-ask spread, market-on-open impact, and fill-quality uncertainty that a daily-frequency analysis cannot address.

## The rolling R² plot is display-capped

In the rolling expanding-window OOS R² computation, per-step R² values are floored at -0.5 for plotting clarity. This is a visualization choice that affects the appearance of Figure 5 during regimes of severe underperformance; it does not affect the paper's headline conclusions, which rely on the HAC OLS point estimates rather than the rolling R² series. The cap is documented in `src/econometrics.py` and mentioned in the code review notes.

## Granger causality is tested only against own-lags

The Granger F-tests compare a model with lagged FX returns and lagged gaps to one with only lagged gaps. This is the standard bivariate form. A richer test would include other macroeconomic news variables and commodity prices as additional controls, allowing the test to distinguish between "FX Granger-causes gap" and "some common factor drives both." This is a known limitation of bivariate Granger tests and is noted here for completeness.

## VAR(2) lag order is chosen by convention, not by information criterion

The VAR model used for the impulse response functions is fit at order 2. This choice is standard for daily financial time series but is not formally selected via an information criterion (AIC, BIC, HQIC). Alternative lag orders could produce modestly different impulse response shapes, particularly at the longer horizons. The qualitative conclusions (positive step-zero response, sign reversal at step one, decay to zero) appear robust to nearby lag orders, but the exact magnitudes should be interpreted with appropriate skepticism.

## The quanto pricing extension is sketched, not implemented

Section 7 of the paper discusses quanto derivatives as a mechanism to isolate the equity-specific repricing signal from spot FX exposure. The paper describes the payoff structure and suggests the HAC OLS betas as calibration inputs. It does not present a fully-worked quanto pricing model with numerical examples. A reader evaluating the paper as a tradability analysis should treat Section 7 as a research direction rather than an implemented result.

## AI tool usage

As disclosed in the README, AI tools were used throughout the project including for some code authorship, literature discovery, and prose drafting. The authors designed the research question, built the identification strategy, selected the econometric framework, interpreted the results, and bear responsibility for all claims. Citations have been verified manually against primary sources. Nevertheless, readers should evaluate the methodology on its own merits rather than on the authors' voice alone, and any adoption of the techniques in production settings should be preceded by independent replication.
