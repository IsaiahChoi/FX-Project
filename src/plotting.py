"""Figure generation for the paper's six visualizations.

Each of the six paper figures is produced by a dedicated function,
parameterized by the results DataFrame / time-series cache from
``evaluation.run_all_experiments``. Figures are saved as PNG at 300
DPI to ``cfg.figures_dir``.

Figure index (matching the paper):
    1. HAC OLS standardized betas + significance (scatter of p-values)
    2. Cross-correlation functions (per single ticker, lags 0-5)
    3. Granger causality heatmap (per ticker, lags 1-5)
    4. VAR(2) orthogonalized impulse responses
    5. Rolling expanding-window OOS R²
    6. Random Forest feature importance (Gini)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import ResearchConfig
from .econometrics import (
    compute_rolling_r2,
    cross_correlation,
    granger_p_values,
    var_impulse_response,
)

logger = logging.getLogger(__name__)

PAPER_STYLE = {"style": "whitegrid", "context": "paper", "font_scale": 1.1}


def _apply_style() -> None:
    """Apply consistent paper-quality style across figures."""
    sns.set_theme(**PAPER_STYLE)


def _save(fig: plt.Figure, cfg: ResearchConfig, filename: str) -> str:
    """Save figure to configured output directory and return the path."""
    os.makedirs(cfg.figures_dir, exist_ok=True)
    path = os.path.join(cfg.figures_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    logger.info("Saved figure: %s", path)
    return path


def plot_ols_effect_sizes(df_results: pd.DataFrame, cfg: ResearchConfig) -> plt.Figure:
    """Figure 1: standardized betas and HAC p-values across seven configs.

    Left panel: betas for FX_Vol and FX_Ret (core specification).
    Right panel: corresponding HAC p-values with α=0.05 reference line.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    df_ols = df_results.melt(
        id_vars=["Experiment"],
        value_vars=["Beta_Vol_Core", "Beta_Ret_Core"],
        var_name="Feature",
        value_name="Std_Beta",
    )
    df_pvals = df_results.melt(
        id_vars=["Experiment"],
        value_vars=["P_Vol_Core", "P_Ret_Core"],
        var_name="Feature",
        value_name="P_Value",
    )
    df_ols["P_Value"] = df_pvals["P_Value"].values
    df_ols["Feature"] = df_ols["Feature"].replace(
        {"Beta_Vol_Core": "FX Volatility", "Beta_Ret_Core": "FX Return"}
    )
    df_ols["Significant"] = df_ols["P_Value"] < 0.05

    sns.barplot(
        data=df_ols, x="Experiment", y="Std_Beta", hue="Feature",
        ax=axes[0], palette="viridis",
    )
    axes[0].set_title("Standardized Effect Sizes (HAC OLS)", fontweight="bold")
    axes[0].set_ylabel(r"Standardized Beta ($\beta$)")
    axes[0].tick_params(axis="x", rotation=45)

    sns.scatterplot(
        data=df_ols, x="Experiment", y="P_Value", hue="Feature",
        style="Significant", s=150, ax=axes[1], palette="viridis",
        markers={True: "*", False: "o"},
    )
    axes[1].axhline(0.05, color="darkred", linestyle="--", label=r"$\alpha = 0.05$")
    axes[1].set_title(r"Statistical Significance ($H_0: \beta = 0$)", fontweight="bold")
    axes[1].set_ylabel("HAC P-Value")
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    _save(fig, cfg, "1_OLS_Effect_Sizes.png")
    return fig


def plot_cross_correlation(
    ts_data_cache: dict[str, pd.DataFrame],
    cfg: ResearchConfig,
    max_lag: int = 5,
) -> Optional[plt.Figure]:
    """Figure 2: CCF stem plots per single ticker, lags 0..max_lag."""
    _apply_style()
    single_tickers = [k for k in ts_data_cache.keys() if " + " not in k]
    n = len(single_tickers)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    axes = axes.flatten()

    for i, ticker in enumerate(single_tickers):
        df_ts = ts_data_cache[ticker]
        ax = axes[i]

        ccf_vals, conf = cross_correlation(
            df_ts["FX_Ret"], df_ts["Gap"], max_lag=max_lag
        )
        lags = np.arange(max_lag + 1)
        ax.stem(lags, ccf_vals, basefmt="k-")
        ax.axhline(conf, color="red", linestyle="--", alpha=0.5, label="95% Significance")
        ax.axhline(-conf, color="red", linestyle="--", alpha=0.5)

        company = cfg.equities.get(ticker, "Unknown")
        ax.set_title(
            fr"CCF: FX Returns leading {ticker} ({company})"
            + "\n"
            + r"$H_0: \rho(FX_{t-k}, \mathrm{Gap}_t) = 0$",
            fontweight="bold",
            fontsize=10,
        )
        ax.set_xlabel(r"Lag $k$ (Days)")
        if i == 0:
            ax.set_ylabel("Pearson Correlation")
            ax.legend()

    fig.tight_layout()
    _save(fig, cfg, "2_Cross_Correlation_All.png")
    return fig


def plot_granger_heatmap(
    ts_data_cache: dict[str, pd.DataFrame],
    cfg: ResearchConfig,
    max_lag: int = 5,
) -> Optional[plt.Figure]:
    """Figure 3: Granger causality p-value heatmap (tickers × lags)."""
    _apply_style()
    single_tickers = [k for k in ts_data_cache.keys() if " + " not in k]
    n = len(single_tickers)
    if n == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 1.5 * n + 2))
    pvals: dict[str, list[float]] = {}
    for ticker in single_tickers:
        df_ts = ts_data_cache[ticker]
        p_list = granger_p_values(df_ts, max_lag=max_lag)
        company = cfg.equities.get(ticker, "Unknown")
        pvals[f"{ticker} ({company})"] = p_list

    heatmap_data = pd.DataFrame(pvals, index=range(1, max_lag + 1)).T
    sns.heatmap(
        heatmap_data, annot=True, cmap="coolwarm_r",
        cbar_kws={"label": "P-Value"}, vmin=0, vmax=0.1, ax=ax,
    )
    ax.set_title(
        r"Granger Causality: FX Returns $\rightarrow$ Equity Gap"
        + "\n"
        + r"$H_0$: FX does not Granger-cause Gap",
        fontweight="bold",
    )
    ax.set_xlabel("Lag Order")
    ax.set_ylabel("Ticker")

    fig.tight_layout()
    _save(fig, cfg, "3_Granger_Causality_All.png")
    return fig


def plot_impulse_response(
    ts_data_cache: dict[str, pd.DataFrame],
    cfg: ResearchConfig,
    var_order: int = 2,
    irf_horizon: int = 5,
) -> Optional[plt.Figure]:
    """Figure 4: orthogonalized IRF of Gap to FX shock, per ticker."""
    _apply_style()
    single_tickers = [k for k in ts_data_cache.keys() if " + " not in k]
    n = len(single_tickers)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    axes = axes.flatten()

    for i, ticker in enumerate(single_tickers):
        df_ts = ts_data_cache[ticker]
        ax = axes[i]

        result = var_impulse_response(
            df_ts, response="Gap", impulse="FX_Ret",
            var_order=var_order, irf_horizon=irf_horizon,
        )
        steps = result.steps
        point = result.point_estimates
        se = result.std_errors

        ax.plot(steps, point, color="#1f77b4", linewidth=2)
        ax.fill_between(
            steps, point - 1.96 * se, point + 1.96 * se,
            color="#1f77b4", alpha=0.2,
        )
        ax.axhline(0, color="black", linewidth=1)

        company = cfg.equities.get(ticker, "Unknown")
        ax.set_title(
            f"Orthogonalized Impulse Response: {ticker} ({company})\n"
            + "Effect of 1-SD FX Shock on Equity Gap",
            fontweight="bold",
            fontsize=10,
        )
        ax.set_xlabel("Days after shock")
        if i == 0:
            ax.set_ylabel("Response in Equity Gap (Z-Score)")
        else:
            ax.set_ylabel("")

    fig.tight_layout()
    _save(fig, cfg, "4_Impulse_Response_All.png")
    return fig


def plot_rolling_r2(
    ts_data_cache: dict[str, pd.DataFrame],
    cfg: ResearchConfig,
    window: int = 252,
    smoothing: int = 21,
) -> Optional[plt.Figure]:
    """Figure 5: rolling expanding-window OOS R² per ticker."""
    _apply_style()
    single_tickers = [k for k in ts_data_cache.keys() if " + " not in k]
    n = len(single_tickers)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    axes = axes.flatten()

    for i, ticker in enumerate(single_tickers):
        df_ts = ts_data_cache[ticker]
        ax = axes[i]

        r2_series = compute_rolling_r2(
            df_ts, fx_col="FX_Ret", gap_col="Gap",
            window=window, smoothing=smoothing,
        )
        ax.plot(r2_series.index, r2_series.values, color="indigo", linewidth=1.5)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)

        company = cfg.equities.get(ticker, "Unknown")
        ax.set_title(
            fr"Rolling Predictive $R^2$: {ticker} ({company})"
            + "\n"
            + r"$H_0$: Out-of-sample predictive power $\leq$ 0",
            fontweight="bold",
            fontsize=10,
        )
        ax.set_xlabel("Date")
        ax.tick_params(axis="x", rotation=45)
        if i == 0:
            ax.set_ylabel(r"Out-of-Sample $R^2$")
        else:
            ax.set_ylabel("")

    fig.tight_layout()
    _save(fig, cfg, "5_Rolling_R2_All.png")
    return fig


def plot_feature_importance(
    all_importances: dict[str, dict[str, float]],
    cfg: ResearchConfig,
    min_configs: int = 5,
) -> plt.Figure:
    """Figure 6: RF Gini importance, features present in >= min_configs runs."""
    _apply_style()
    df_imp = pd.DataFrame(all_importances).T
    common_features = df_imp.columns[df_imp.notna().sum() >= min_configs].tolist()
    df_imp.fillna(0, inplace=True)
    df_imp_common = df_imp[common_features]
    df_imp_mean = df_imp_common.mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        df_imp_mean.index, df_imp_mean.values,
        color="cadetblue", edgecolor="black", alpha=0.8,
    )
    ax.set_title(
        "Random Forest Non-Linear Feature Importance\n"
        + fr"(Features present in $\geq$ {min_configs}/7 configurations)",
        fontweight="bold",
    )
    ax.set_ylabel("Mean Decrease Impurity")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    _save(fig, cfg, "6_Feature_Importance.png")
    return fig


def plot_all(
    df_results: pd.DataFrame,
    ts_data_cache: dict[str, pd.DataFrame],
    all_importances: dict[str, dict[str, float]],
    cfg: ResearchConfig,
) -> dict[str, Optional[plt.Figure]]:
    """Produce every paper figure in one call."""
    return {
        "1_ols_effect_sizes": plot_ols_effect_sizes(df_results, cfg),
        "2_cross_correlation": plot_cross_correlation(ts_data_cache, cfg),
        "3_granger_heatmap": plot_granger_heatmap(ts_data_cache, cfg),
        "4_impulse_response": plot_impulse_response(ts_data_cache, cfg),
        "5_rolling_r2": plot_rolling_r2(ts_data_cache, cfg),
        "6_feature_importance": plot_feature_importance(all_importances, cfg),
    }
