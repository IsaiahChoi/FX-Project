"""Experiment orchestration: run all seven configurations.

Runs the three single-ticker and four multi-ticker configurations
reported in the paper, fits both parametric (HAC OLS) and non-
parametric (Random Forest) models, evaluates out-of-sample, and
applies Holm-Bonferroni correction to the Diebold-Mariano p-values
for multiple testing.

Output columns match the paper's Table 1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .config import ResearchConfig
from .data_engine import DataEngine
from .econometrics import EvaluationMetrics, evaluate_experiment
from .features import (
    BASE_FEATURES,
    combine_datasets,
    split_chronologically,
    standardize_columns,
)
from .models import (
    CORE_OLS_FEATURES,
    FULL_OLS_FEATURES,
    OLSFitResult,
    fit_hac_ols,
    fit_random_forest_classifier,
    fit_random_forest_regressor,
)

logger = logging.getLogger(__name__)


# Seven experimental configurations reported in the paper.
DEFAULT_EXPERIMENTS: list[list[str]] = [
    ["5713"],
    ["5711"],
    ["5706"],
    ["5711", "5706"],
    ["5711", "5713"],
    ["5706", "5713"],
    ["5711", "5706", "5713"],
]


@dataclass
class ExperimentResult:
    """Container for a single experiment's artefacts."""

    name: str
    tickers: list[str]
    df_std: pd.DataFrame  # standardized full panel (train + test)
    features: list[str]
    ols_core: OLSFitResult
    ols_full: OLSFitResult
    rf_feature_importance: dict[str, float]
    metrics: EvaluationMetrics


def run_single_experiment(
    engine: DataEngine,
    tickers: list[str],
    cfg: ResearchConfig,
) -> Optional[ExperimentResult]:
    """Run one experiment end-to-end.

    Returns ``None`` if insufficient data (< 20 rows after merge).
    """
    name = " + ".join(tickers)
    logger.info("Running experiment: %s", name)

    df, features = combine_datasets(engine, tickers)
    if df is None or len(df) < 20:
        logger.warning("Insufficient data for experiment %s", name)
        return None

    df_std = standardize_columns(df)
    train, test = split_chronologically(df_std, cfg.test_size)
    train_nonzero = train[train["Gap"] != 0.0]

    # Parametric: HAC OLS, two specifications.
    ols_core = fit_hac_ols(train_nonzero, CORE_OLS_FEATURES)
    ols_full = fit_hac_ols(train_nonzero, FULL_OLS_FEATURES)

    # Non-parametric: Random Forest regressor (magnitude) + classifier
    # (direction). Both use the full feature set.
    rf_reg = fit_random_forest_regressor(
        train_nonzero[features],
        train_nonzero["Gap"],
        n_estimators=cfg.n_estimators,
        random_state=cfg.random_state,
    )
    rf_clf = fit_random_forest_classifier(
        train_nonzero[features],
        train_nonzero["Target_Class"],
        n_estimators=cfg.n_estimators,
        random_state=cfg.random_state,
    )

    importance = dict(zip(features, rf_reg.feature_importances_))

    preds_reg = rf_reg.predict(test[features])
    preds_clf = rf_clf.predict(test[features])

    metrics = evaluate_experiment(
        y_true_reg=test["Gap"].values,
        preds_reg=preds_reg,
        y_true_clf=test["Target_Class"].values,
        preds_clf=preds_clf,
        fx_features_test=test["FX_Ret"].values,
        random_state=cfg.random_state,
    )

    return ExperimentResult(
        name=name,
        tickers=tickers,
        df_std=df_std,
        features=features,
        ols_core=ols_core,
        ols_full=ols_full,
        rf_feature_importance=importance,
        metrics=metrics,
    )


def run_all_experiments(
    cfg: Optional[ResearchConfig] = None,
    experiments: Optional[list[list[str]]] = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict[str, float]], ResearchConfig]:
    """Run all configured experiments and assemble a results table.

    Returns
    -------
    df_results
        Row-per-experiment DataFrame matching the paper's Table 1
        columns, with an added Holm-adjusted DM p-value.
    ts_data_cache
        Standardized single-ticker panels, keyed by ticker code.
        Used by the plotting module for CCF, Granger, IRF, and
        rolling R² figures.
    all_importances
        Feature importance dict per experiment.
    cfg
        The config actually used (helpful downstream).
    """
    if cfg is None:
        cfg = ResearchConfig()
    if experiments is None:
        experiments = DEFAULT_EXPERIMENTS

    engine = DataEngine(cfg)

    rows: list[dict] = []
    ts_data_cache: dict[str, pd.DataFrame] = {}
    all_importances: dict[str, dict[str, float]] = {}

    for tickers in experiments:
        result = run_single_experiment(engine, tickers, cfg)
        if result is None:
            continue

        # Cache single-ticker panels for time-series plots.
        if len(tickers) == 1:
            ts_data_cache[result.name] = result.df_std

        all_importances[result.name] = result.rf_feature_importance

        m = result.metrics
        rows.append(
            {
                "Experiment": result.name,
                "N_Test": m.n_test,
                "Beta_Vol_Core": result.ols_core.beta_vol,
                "P_Vol_Core": result.ols_core.p_vol,
                "Beta_Ret_Core": result.ols_core.beta_ret,
                "P_Ret_Core": result.ols_core.p_ret,
                "Beta_Vol_Full": result.ols_full.beta_vol,
                "P_Vol_Full": result.ols_full.p_vol,
                "Beta_Ret_Full": result.ols_full.beta_ret,
                "P_Ret_Full": result.ols_full.p_ret,
                "OOS_R2": m.oos_r2,
                "Mutual_Info": m.mutual_info,
                "DM_Stat": m.dm_stat,
                "DM_P_Val": m.dm_p_val,
                "Accuracy": m.accuracy,
                "Binom_P_Val": m.binom_p_val,
            }
        )

    df_results = pd.DataFrame(rows)

    # Holm-Bonferroni correction across the seven DM p-values.
    if not df_results.empty and "DM_P_Val" in df_results.columns:
        _, pvals_corrected, _, _ = multipletests(
            df_results["DM_P_Val"], alpha=0.05, method="holm"
        )
        df_results["DM_Adj_P_Val"] = pvals_corrected

    return df_results, ts_data_cache, all_importances, cfg


def format_results_table(df_results: pd.DataFrame) -> str:
    """Return a human-readable string of the headline columns."""
    cols = [
        "Experiment",
        "N_Test",
        "Beta_Ret_Core",
        "P_Ret_Core",
        "OOS_R2",
        "Mutual_Info",
        "DM_Stat",
        "DM_Adj_P_Val",
        "Accuracy",
    ]
    cols = [c for c in cols if c in df_results.columns]
    return df_results[cols].to_string(
        index=False,
        formatters={
            "Beta_Ret_Core": "{:.3f}".format,
            "P_Ret_Core": "{:.4f}".format,
            "OOS_R2": "{:.4f}".format,
            "Mutual_Info": "{:.4f}".format,
            "DM_Stat": "{:.3f}".format,
            "DM_Adj_P_Val": "{:.4f}".format,
            "Accuracy": "{:.3f}".format,
        },
    )
