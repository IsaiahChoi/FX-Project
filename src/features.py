"""Feature engineering and dataset assembly.

Combines per-ticker DataFrames into a panel with day-of-week and
ticker dummies, standardizes features for interpretable effect sizes,
and splits chronologically with a business-day embargo.

All feature definitions, dummy encodings, and split parameters are
preserved exactly from the original paper code.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from .data_engine import DataEngine

logger = logging.getLogger(__name__)

# Columns standardized prior to HAC OLS so coefficients are comparable.
# (Original choice preserved from paper code.)
COLS_TO_STANDARDIZE = [
    "Gap",
    "FX_Vol",
    "FX_Ret",
    "FX_Ret_2D",
    "FX_Ret_5D",
    "Gap_Lag1",
    "Gap_Lag2",
]

# Base feature set used by every model. Day-of-week and equity
# dummies are appended dynamically.
BASE_FEATURES = ["FX_Vol", "FX_Ret", "FX_Ret_2D", "FX_Ret_5D", "Gap_Lag1", "Gap_Lag2"]


def combine_datasets(
    engine: DataEngine, tickers: list[str]
) -> tuple[Optional[pd.DataFrame], list[str]]:
    """Assemble a panel DataFrame across the requested tickers.

    Parameters
    ----------
    engine
        Initialized DataEngine. Will be used to load each ticker via
        ``get_data``. Internal caching means repeat calls are cheap.
    tickers
        TSE ticker codes to include, e.g. ``["5711", "5706"]``.

    Returns
    -------
    combined : DataFrame or None
        Row-stacked panel of per-ticker frames, with additional equity
        dummies (``Eq_...``) appended for multi-stock experiments.
        Returns ``None`` if no ticker returned data.
    features : list[str]
        Feature column names in a deterministic order suitable for
        passing directly to model ``fit`` calls.
    """
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        df = engine.get_data(ticker)
        if df is not None:
            frames.append(df)

    if not frames:
        return None, []

    combined = pd.concat(frames)

    features = list(BASE_FEATURES)
    dow_cols = sorted([c for c in combined.columns if c.startswith("DoW_")])
    features.extend(dow_cols)

    if len(tickers) > 1:
        dummies = pd.get_dummies(combined["Equity"], prefix="Eq", drop_first=False)
        dummies = dummies.reindex(sorted(dummies.columns), axis=1)
        combined = pd.concat([combined, dummies], axis=1)
        features.extend(dummies.columns.tolist())

    combined.sort_index(inplace=True)
    return combined, features


def standardize_columns(
    df: pd.DataFrame, cols: list[str] = None
) -> pd.DataFrame:
    """Z-score the listed columns (x - mean) / std.

    Copies the DataFrame; does not modify in place. Columns absent
    from ``df`` are silently skipped so this is safe for both single-
    and multi-ticker panels.
    """
    if cols is None:
        cols = COLS_TO_STANDARDIZE

    out = df.copy()
    for c in cols:
        if c in out.columns:
            std = out[c].std()
            if std > 0:
                out[c] = (out[c] - out[c].mean()) / std
    return out


def split_chronologically(
    df: pd.DataFrame, test_size: float = 0.2, embargo_bdays: int = 2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-indexed panel chronologically with a business-day embargo.

    The embargo prevents label leakage from short-term autocorrelation
    between the train tail and test head — an issue called out explicitly
    in the paper's identification strategy.

    Parameters
    ----------
    df
        Panel DataFrame indexed by date (possibly with duplicate dates
        across tickers).
    test_size
        Fraction of unique dates allocated to the test set.
    embargo_bdays
        Number of business days to drop between train end and test start.

    Returns
    -------
    train, test : (DataFrame, DataFrame)

    Raises
    ------
    AssertionError
        If either split is empty.
    """
    unique_dates = np.sort(df.index.unique())
    split_idx = int(len(unique_dates) * (1 - test_size))
    split_date = unique_dates[split_idx]

    train = df[df.index < split_date]
    train_end = train.index.max()
    embargo_cutoff = train_end + BDay(embargo_bdays)
    test = df[df.index > embargo_cutoff]

    assert len(train) > 0 and len(test) > 0, "Split failed: train or test is empty."
    return train, test
