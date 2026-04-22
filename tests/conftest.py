"""Shared test fixtures.

Provides minimal synthetic FX and equity dataframes that match the
structure the pipeline expects, so tests can run without access to
proprietary Bloomberg data.

These fixtures are for *testing the pipeline's behavior*, not for
producing plausible numerical results. They are deliberately tiny
and structured to make assertions about parsing, lookahead handling,
and feature construction easy to verify.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fx_bars_clean() -> pd.DataFrame:
    """FX bars with every timestamp strictly before 09:00 Asia/Tokyo.

    Covers three business days with bars every 5 minutes between 15:00
    and 23:55 (evening) and between 00:00 and 08:50 (morning). All
    Tokyo-local. No bars at or after 09:00, so this should pass the
    lookahead guard cleanly.
    """
    rng = np.random.default_rng(seed=42)
    rows = []
    for day in pd.bdate_range("2024-01-02", periods=3, tz="Asia/Tokyo"):
        # Evening window: 15:00 - 23:55, mapped to next business day.
        for h in range(15, 24):
            for m in range(0, 60, 5):
                rows.append(
                    {
                        "Dates": day.replace(hour=h, minute=m),
                        "Open": 150 + rng.standard_normal() * 0.1,
                        "Close": 150 + rng.standard_normal() * 0.1,
                        "High": 150.2,
                        "Low": 149.8,
                    }
                )
        # Morning window (next day): 00:00 - 08:50.
        next_day = day + pd.Timedelta(days=1)
        for h in range(0, 9):
            for m in range(0, 60, 5):
                if h == 8 and m >= 50:
                    break  # stop before 08:50
                rows.append(
                    {
                        "Dates": next_day.replace(hour=h, minute=m),
                        "Open": 150 + rng.standard_normal() * 0.1,
                        "Close": 150 + rng.standard_normal() * 0.1,
                        "High": 150.2,
                        "Low": 149.8,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def fx_bars_with_lookahead_violation() -> pd.DataFrame:
    """FX bars including one bar at exactly 09:00 Tokyo (should fail the guard)."""
    day = pd.Timestamp("2024-01-02", tz="Asia/Tokyo")
    rows = [
        {"Dates": day.replace(hour=15, minute=0), "Open": 150.0, "Close": 150.1,
         "High": 150.2, "Low": 149.8},
        {"Dates": day.replace(hour=20, minute=0), "Open": 150.1, "Close": 150.0,
         "High": 150.2, "Low": 149.8},
        # This bar is at 09:00 on the NEXT day — a lookahead violation.
        {
            "Dates": (day + pd.Timedelta(days=1)).replace(hour=9, minute=0),
            "Open": 150.0, "Close": 150.2, "High": 150.3, "Low": 149.9,
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def equity_sheet_clean() -> pd.DataFrame:
    """Minimal daily equity DataFrame matching the post-parsing schema.

    Represents what ``_parse_equity_sheet`` returns after header
    detection and type coercion: a DataFrame with `Dates`, `PX_OPEN`,
    `PX_LAST` columns and clean numeric values.
    """
    dates = pd.bdate_range("2024-01-02", periods=10)
    return pd.DataFrame(
        {
            "Dates": dates,
            "PX_OPEN": np.linspace(3000, 3050, 10),
            "PX_LAST": np.linspace(3005, 3055, 10),
        }
    )


@pytest.fixture
def standardized_panel() -> pd.DataFrame:
    """Tiny standardized single-ticker panel resembling the pipeline output.

    For tests of downstream feature functions that operate on the
    standardized panel (split_chronologically, econometric functions).
    """
    rng = np.random.default_rng(seed=42)
    n = 200
    dates = pd.bdate_range("2023-01-02", periods=n)
    fx_ret = rng.standard_normal(n)
    gap = 0.2 * fx_ret + rng.standard_normal(n) * 0.9

    df = pd.DataFrame(
        {
            "Gap": gap,
            "FX_Ret": fx_ret,
            "FX_Vol": np.abs(rng.standard_normal(n)),
            "FX_Ret_2D": rng.standard_normal(n),
            "FX_Ret_5D": rng.standard_normal(n),
            "Gap_Lag1": np.concatenate([[0.0], gap[:-1]]),
            "Gap_Lag2": np.concatenate([[0.0, 0.0], gap[:-2]]),
            "Equity": "5711",
            "Target_Class": (gap > 0).astype(int),
        },
        index=dates,
    )
    # Add DoW dummies (same convention as pipeline).
    df["DoW_1"] = (df.index.dayofweek == 1).astype(int)
    df["DoW_2"] = (df.index.dayofweek == 2).astype(int)
    df["DoW_3"] = (df.index.dayofweek == 3).astype(int)
    df["DoW_4"] = (df.index.dayofweek == 4).astype(int)
    return df
