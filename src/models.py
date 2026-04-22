"""Model fitting: HAC OLS and Random Forest.

Implements the two model classes used in the paper:

1. **Parametric HAC OLS** with Newey-West standard errors. Two
   specifications: ``Core`` (FX_Vol, FX_Ret) and ``Full`` (adds
   FX_Ret_2D, FX_Ret_5D).
2. **Non-parametric Random Forest** regressor (for gap magnitude) and
   classifier (for gap direction). 500 trees, ``random_state=42``.

All hyperparameters and specifications preserved from the original.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)

CORE_OLS_FEATURES = ["FX_Vol", "FX_Ret"]
FULL_OLS_FEATURES = ["FX_Vol", "FX_Ret", "FX_Ret_2D", "FX_Ret_5D"]


@dataclass
class OLSFitResult:
    """Container for a single HAC OLS fit's headline statistics."""

    beta_vol: float
    p_vol: float
    beta_ret: float
    p_ret: float
    n_obs: int
    model: Optional[sm.regression.linear_model.RegressionResultsWrapper] = None


def newey_west_maxlags(n_obs: int) -> int:
    """Newey-West automatic lag selection.

    Uses the Stock-Watson rule of thumb: ``floor(4 * (n/100)^(2/9))``.
    This is the same heuristic used in the paper.
    """
    return int(4 * (n_obs / 100) ** (2 / 9))


def fit_hac_ols(
    df: pd.DataFrame, features: list[str], target: str = "Gap"
) -> OLSFitResult:
    """Fit OLS with HAC (Newey-West) standard errors.

    Parameters
    ----------
    df
        Training DataFrame. Expected to be standardized already so that
        coefficients are interpretable as effect sizes.
    features
        Column names to use as regressors. A constant is added.
    target
        Dependent variable column.

    Returns
    -------
    OLSFitResult with betas and p-values for FX_Vol and FX_Ret (when
    present in ``features``). NaNs returned on any fit failure.
    """
    try:
        X = sm.add_constant(df[features])
        y = df[target]
        lags = newey_west_maxlags(len(df))
        model = sm.OLS(y, X).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": lags, "use_correction": True},
        )
        return OLSFitResult(
            beta_vol=float(model.params.get("FX_Vol", np.nan)),
            p_vol=float(model.pvalues.get("FX_Vol", np.nan)),
            beta_ret=float(model.params.get("FX_Ret", np.nan)),
            p_ret=float(model.pvalues.get("FX_Ret", np.nan)),
            n_obs=len(df),
            model=model,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("HAC OLS fit failed: %s", exc)
        return OLSFitResult(
            beta_vol=float("nan"),
            p_vol=float("nan"),
            beta_ret=float("nan"),
            p_ret=float("nan"),
            n_obs=len(df),
            model=None,
        )


def compute_vifs(df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    """Variance Inflation Factors for each feature (with a constant).

    VIF > 10 suggests problematic multicollinearity. Used in the paper
    as a diagnostic before the Full specification.
    """
    X = sm.add_constant(df[features]).values
    cols = ["const"] + list(features)
    return {c: float(variance_inflation_factor(X, i)) for i, c in enumerate(cols)}


def fit_random_forest_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 500,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Train a Random Forest regressor for gap magnitude.

    500 trees, default sklearn depth, ``random_state`` fixed for
    reproducibility. Matches the paper's configuration exactly.
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state
    )
    rf.fit(X_train, y_train)
    return rf


def fit_random_forest_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 500,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest classifier for gap direction.

    500 trees, ``class_weight='balanced'`` to handle directional
    imbalance, ``random_state`` fixed. Matches the paper exactly.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
    )
    rf.fit(X_train, y_train)
    return rf
