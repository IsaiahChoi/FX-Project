# How to Read These Results

A short companion piece for readers who want to understand what the headline numbers actually mean. Written as prose; there are no summary bullets.

## What the HAC betas do and don't say

The paper's central finding is that standardized HAC OLS coefficients on overnight FX return range from 0.164 to 0.227 across all seven experimental configurations, with every Newey-West corrected p-value below 0.025. These are the numbers that most strongly support the paper's thesis, and they are the numbers most often cited in downstream discussion of the work.

A standardized coefficient of 0.20 means that a one-standard-deviation move in overnight FX return is associated, on average within the sample, with a 0.20 standard deviation move in the next-morning opening gap in the same direction. For a sample in which gap standard deviation is roughly 100 basis points, that works out to an average 20-basis-point gap movement per standard deviation of overnight FX. This is not a negligible effect — it is the kind of magnitude a careful trader might act on — but it is also not a deterministic relationship. The R² of the bivariate regression is substantially below 10%, meaning that roughly 90% of gap variation is unexplained by overnight FX return.

What the significance of these coefficients establishes is that the relationship is very unlikely to be sampling noise. What it does not establish is that the relationship is large enough to be exploitable in any particular strategy, that it will persist into future samples, or that it reflects a causal chain rather than an as-yet-unidentified common factor driving both variables. These are separate questions the paper does not claim to answer.

## Why negative OOS R² doesn't invalidate the contemporaneous finding

Readers unfamiliar with this distinction sometimes worry that the paper's negative out-of-sample R² values undermine the significant in-sample coefficients. This reading would be incorrect. The two quantities are measuring different things.

The HAC OLS coefficient measures whether, across the full training sample, overnight FX return and the next-morning gap are linearly related. Its significance — the HAC p-value — measures whether that relationship is distinguishable from noise given the sample size and the estimated error structure. These quantities are purely descriptive of the training data; they say nothing about whether a model fit on the training data would produce useful predictions on new data.

The out-of-sample R² measures exactly that: given a model trained on the first 80% of observations, how well does it predict the magnitude of opening gaps in the held-out 20%? A negative OOS R² means the model's predictions are worse than a naive zero forecast. This can happen for many reasons, some of which are entirely compatible with a real underlying relationship: the signal may be swamped by noise in any particular out-of-sample window, the model may be overfit to sample-specific idiosyncrasies that don't generalize, or the true relationship may be time-varying in ways the model cannot capture.

What the pattern of significant in-sample coefficients alongside negative out-of-sample R² says in this particular case is: there is a statistically significant contemporaneous relationship, but the signal-to-noise ratio on any single day is too low to produce useful point-level forecasts. This is exactly the situation you would expect for a bounded information-friction effect — the signal is real at the aggregate level, but the residual variance of any specific day's gap is very large relative to the mean effect.

## What the Granger null tells you

The Granger causality tests at lags 1 through 5 return p-values between roughly 0.20 and 0.52 across all tickers. The null hypothesis at each lag is that past values of FX return do not help forecast the current gap beyond the gap's own autoregressive structure.

Failing to reject this null — at lag 1, lag 2, lag 3, lag 4, and lag 5 — is a very strong statement about the temporal structure of the relationship. It says that whatever predictive content overnight FX has about the gap, that content is used up within the opening auction itself. There is no measurable residual predictive power at any daily horizon beyond the contemporaneous day.

This is not a failure of the paper. It is, on the contrary, the most important diagnostic confirmation that the paper's thesis is correct. The "Trading Place Hypothesis" claims that FX information accumulates while Tokyo is closed and is absorbed at the opening auction. If this is true, we should see a significant contemporaneous effect (which we do) and a complete absence of multi-day lead-lag structure (which we also do). A paper that reported significant Granger causality at lag 3 and lag 5 would be harder to interpret, because it would suggest a different mechanism than the one being proposed.

## What to make of the 60.1% directional accuracy

The Mitsubishi Materials (5711) single-stock configuration achieves 60.1% directional accuracy on its test set, substantially above the 50% chance baseline. By the one-sided binomial test, this is statistically significant even after Holm-Bonferroni correction for multiple testing across the seven configurations.

The question is what interpretation this result deserves. Three things deflate its significance substantially.

First, it is one of seven configurations. The other six — 5713 single-stock (49.7%), 5706 single-stock (49.0%), and the four multi-stock baskets (49.7% to 53.3%) — cluster near the coin-flip baseline. If the underlying effect were strong and generalizable, we would expect to see elevated directional accuracy across most or all configurations, not in one. The outlier pattern is more consistent with sample variation than with a robust signal.

Second, the test set is small. With roughly 31 test observations, a 60.1% hit rate represents 19 correct predictions out of 31. The 95% confidence interval for this proportion runs from roughly 41% to 77% — an interval that overlaps both "no signal" (50%) and "strong signal" (say, 65%) regions. The binomial test's rejection of the 50% null depends sensitively on the exact count, and small perturbations in the sample split could move the point estimate substantially.

Third, Random Forest classifiers are known to produce nonrepresentative directional accuracy on small samples, particularly when the class balance is uneven. Without walk-forward validation across multiple splits, it is not possible to distinguish a real signal from favorable variance.

The paper's own Section 5.4 is explicit about this: "The accuracy of 60.1% is the only result demonstrating strong unconditional predictive power. For the remaining tickers and baskets, the proximity of the results to 50% suggests that a simple directional strategy without selective participation filters is insufficient to overcome the noise inherent in overnight returns." This is the correct framing. The number is reported honestly; readers should treat it as a data point, not a headline finding.

## The volatility puzzle

A separate puzzle is the divergence between linear and non-linear treatments of FX volatility. In the HAC OLS specifications, volatility is statistically insignificant across every configuration, with p-values as high as 0.94. In the Random Forest feature importance rankings, volatility is the second-most-important feature with a Mean Decrease Impurity around 0.20.

The most likely explanation is that volatility does not have a direct linear effect on the gap but interacts with FX return in a way that only a non-linear model can capture. Specifically: the transmission of a given directional FX move into the equity gap may be amplified during high-volatility overnight periods (because more price discovery is happening) and muted during low-volatility periods (because the overnight window is informationally quiet). A linear model that enters volatility as a separate additive regressor cannot see this interaction; a tree-based model that can split on volatility before considering FX return can. The paper speculates along these lines but does not formally test the interaction, and this is flagged as a direction for future regime-switching analysis in the limitations document.

## What the paper does not claim

The paper does not claim that this effect is tradable. It does not claim to be a backtest. It does not claim the relationship will persist into future samples, or that it generalizes to other sectors or currency pairs. It does not claim that Random Forests outperform a zero forecast on magnitude prediction — quite the opposite, the negative OOS R² values are reported honestly across all seven configurations. It does not claim the 60.1% directional accuracy on one ticker is a robust signal.

What it does claim is that, within this sample, overnight USD/JPY returns observed strictly before the 09:00 JST opening auction carry statistically significant information about the next-morning opening gap for three Japanese non-ferrous mining equities; that this relationship is robust to the specific tickers chosen, to whether they are pooled or analyzed individually, and to the inclusion of controls for autoregressive structure and weekly seasonality; and that the information is absorbed within the opening auction and does not persist into exploitable multi-day signals.

These are modest claims, appropriately hedged. The paper's internal consistency — significant contemporaneous effect alongside null multi-day Granger causality alongside negative OOS R² — is what gives these claims their weight. A reader evaluating the work should judge it on those terms, not on a misreading of the 60.1% directional accuracy or a conflation of statistical significance with economic tradability.
