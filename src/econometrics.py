"""Econometric time-series analyses reported in the paper.

This module implements the diagnostic and inferential tests that
support the paper's main argument about the temporal boundedness of
the FX-to-gap signal:

- **Cross-Correlation Function (CCF)** at lags 0–5 with 95% bands
- **Granger causality F-tests** at lags 1–5 (SSR F-test variant)
- **VAR(2) orthogonalized impulse response** with standard errors
- **Expanding-window rolling OOS R²** (21-day smoothed) per ticker
- **Diebold-Mariano test** against the naive zero-forecast benchmark
- **Mutual information** between FX features and gaps
- **Directional accuracy** with a one-sided binomial test

Methodology preserved exactly from the original paper code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Out-of-sample evaluation metrics for a single experiment."""

    oos_r2: float
    mutual_info: float
    dm_stat: float
    dm_p_val: float
    accuracy: float
    binom_p_val: float
    n_test: int


def evaluate_experiment(
    y_true_reg: np.ndarray,
    preds_reg: np.ndarray,
    y_true_clf: np.ndarray,
    preds_clf: np.ndarray,
    fx_features_test: np.ndarray,
    random_state: int = 42,
) -> EvaluationMetrics:
    """Compute the six evaluation metrics for one experiment's test set.

    The metrics match the columns in the paper's Table 1 exactly.
    """
    # 1. Out-of-sample R² on magnitude.
    oos_r2 = float(r2_score(y_true_reg, preds_reg))

    # 2. Mutual information between the FX feature vector and the gap.
    fx_reshaped = (
        fx_features_test.reshape(-1, 1) if fx_features_test.ndim == 1 else fx_features_test
    )
    mi_score = float(
        mutual_info_regression(fx_reshaped, y_true_reg, random_state=random_state)[0]
    )

    # 3. Diebold-Mariano test vs naive zero forecast.
    naive_pred = np.zeros_like(y_true_reg)
    e_model = y_true_reg - preds_reg
    e_naive = y_true_reg - naive_pred
    d = (e_naive**2) - (e_model**2)
    d_mean = float(np.mean(d))
    gamma_0 = float(np.var(d, ddof=1))

    if gamma_0 == 0:
        dm_stat, dm_p_val = 0.0, 1.0
    else:
        dm_stat = float(d_mean / np.sqrt(gamma_0 / len(d)))
        dm_p_val = float(2 * (1 - stats.norm.cdf(abs(dm_stat))))

    # 4. Directional accuracy + one-sided binomial test.
    actual_dir = np.sign(y_true_reg)
    pred_dir = np.where(preds_clf == 1, 1, -1)
    traded_mask = actual_dir != 0
    n_eval = int(traded_mask.sum())

    if n_eval > 0:
        correct = int((pred_dir[traded_mask] == actual_dir[traded_mask]).sum())
        acc = correct / n_eval
        binom_p = float(
            stats.binomtest(correct, n_eval, p=0.5, alternative="greater").pvalue
        )
    else:
        acc, binom_p = 0.0, 1.0

    return EvaluationMetrics(
        oos_r2=oos_r2,
        mutual_info=mi_score,
        dm_stat=dm_stat,
        dm_p_val=dm_p_val,
        accuracy=acc,
        binom_p_val=binom_p,
        n_test=len(y_true_reg),
    )


def cross_correlation(
    series_a: pd.Series, series_b: pd.Series, max_lag: int = 5
) -> tuple[np.ndarray, float]:
    """Cross-correlation of ``series_a`` leading ``series_b``.

    Returns the correlations at lags 0..max_lag and the 95% confidence
    threshold ``1.96 / sqrt(n)``. Values outside ±threshold reject the
    null of zero correlation.
    """
    ccf_vals = sm.tsa.stattools.ccf(series_a, series_b, adjusted=False)[: max_lag + 1]
    n = len(series_a)
    conf = 1.96 / np.sqrt(n)
    return ccf_vals, conf


def granger_p_values(
    df: pd.DataFrame,
    response: str = "Gap",
    predictor: str = "FX_Ret",
    max_lag: int = 5,
) -> list[float]:
    """Granger causality p-values from SSR F-test at lags 1..max_lag.

    Tests whether past values of ``predictor`` help forecast
    ``response`` beyond what ``response``'s own past predicts. The
    paper's null at every lag is that FX does not Granger-cause Gap.
    """
    gc_data = df[[response, predictor]].dropna()
    res = grangercausalitytests(gc_data, maxlag=max_lag, verbose=False)
    return [float(res[lag][0]["ssr_ftest"][1]) for lag in range(1, max_lag + 1)]


@dataclass
class IRFResult:
    """Orthogonalized IRF estimates with standard errors."""

    point_estimates: np.ndarray
    std_errors: np.ndarray
    steps: np.ndarray


def var_impulse_response(
    df: pd.DataFrame,
    response: str = "Gap",
    impulse: str = "FX_Ret",
    var_order: int = 2,
    irf_horizon: int = 5,
) -> IRFResult:
    """Fit VAR(order) and extract orthogonalized IRF of response→impulse.

    Parameters
    ----------
    df
        Standardized time-series DataFrame. Must contain ``impulse``
        and ``response`` columns.
    response, impulse
        Column names. IRF is response-variable's reaction to impulse
        variable's shock.
    var_order
        VAR lag order. Paper uses 2.
    irf_horizon
        Number of steps forward to compute IRF.

    Returns
    -------
    IRFResult with point estimates, standard errors, and step indices.
    """
    var_data = df[[impulse, response]].dropna()
    model = VAR(var_data)
    fitted = model.fit(var_order)
    irf = fitted.irf(irf_horizon)

    # Column order in var_data is [impulse, response].
    # Response to impulse → row=response_idx, col=impulse_idx.
    impulse_idx = 0
    response_idx = 1
    point = irf.orth_irfs[:, response_idx, impulse_idx]
    se = irf.stderr(orth=True)[:, response_idx, impulse_idx]
    steps = np.arange(len(point))
    return IRFResult(point_estimates=point, std_errors=se, steps=steps)


def compute_rolling_r2(
    df_std: pd.DataFrame,
    fx_col: str = "FX_Ret",
    gap_col: str = "Gap",
    window: int = 252,
    smoothing: int = 21,
    floor: float = -0.5,
) -> pd.Series:
    """Expanding-window out-of-sample R² with rolling smoothing.

    For each observation ``i >= window``, fits a HAC OLS on the first
    ``i`` observations and predicts observation ``i``. The per-step R²
    is computed against the naive zero forecast (``y_true^2`` in the
    denominator, matching a standardized series with mean ≈ 0).

    Values are floored at ``floor`` for display stability (the paper
    caps negative R² at -0.5 in Figure 5 for plotting clarity; the
    floor is visualization-only, not a methodological choice).

    Parameters
    ----------
    df_std
        Standardized single-ticker DataFrame.
    fx_col, gap_col
        Feature and target column names.
    window
        Minimum number of observations before OOS evaluation begins
        (≈ 1 trading year).
    smoothing
        Window size for the rolling mean applied to the raw OOS R²
        series before plotting. Paper uses 21.
    floor
        Lower bound applied per-step for plotting stability.

    Returns
    -------
    Smoothed rolling R² series, indexed by date.
    """
    df_clean = df_std[[fx_col, gap_col]].dropna()
    X = sm.add_constant(df_clean[fx_col])
    y = df_clean[gap_col]

    dates = df_clean.index[window:]
    rolling_r2: list[float] = []

    for i in range(window, len(df_clean)):
        X_train, y_train = X.iloc[:i], y.iloc[:i]
        X_test, y_test = X.iloc[i : i + 1], y.iloc[i : i + 1]

        try:
            model = sm.OLS(y_train, X_train).fit(
                cov_type="HAC", cov_kwds={"maxlags": 1}
            )
            pred = model.predict(X_test)
            mse_model = float((y_test.values[0] - pred.values[0]) ** 2)
            mse_naive = float(y_test.values[0] ** 2)
            # Guard tiny denominators to match paper's numerical handling.
            loc_r2 = 1 - (mse_model / (mse_naive + 1e-8))
            rolling_r2.append(max(floor, loc_r2))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Rolling R² fit failed at step %d: %s", i, exc)
            rolling_r2.append(float("nan"))

    return pd.Series(rolling_r2, index=dates).rolling(smoothing).mean()
