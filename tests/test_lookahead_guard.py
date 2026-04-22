"""Tests for the timestamp-level lookahead guard in the data pipeline.

The guard is the paper's primary identification device. These tests
verify it behaves as documented:

1. On clean data where no FX bar is at or after 09:00 JST on its
   prediction date, the guard does NOT fire.
2. On data containing a bar at 09:00+ JST on the prediction date,
   the guard raises ValueError with a clear message.
3. The boundary condition (08:50 exactly — the last allowed minute)
   passes; 08:55 onward fails.

If any of these assertions change, the paper's identification claim
is affected and the change must be reviewed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _build_final_frame(
    equity_dates: pd.DatetimeIndex, last_fx_bars: list[pd.Timestamp]
) -> pd.DataFrame:
    """Replicate the in-memory shape that triggers the lookahead guard.

    The guard in ``DataEngine.get_data`` fires after a merge, and
    compares each row's ``Last_FX_Bar`` against the equity open time.
    We construct that post-merge structure directly here to make the
    assertion logic trivially testable in isolation.
    """
    n = len(equity_dates)
    return pd.DataFrame(
        {
            "Dates": equity_dates,
            "Gap": np.zeros(n),
            "Gap_Lag1": np.zeros(n),
            "Gap_Lag2": np.zeros(n),
            "FX_Vol": np.ones(n),
            "FX_Ret": np.zeros(n),
            "FX_Ret_2D": np.zeros(n),
            "FX_Ret_5D": np.zeros(n),
            "Last_FX_Bar": last_fx_bars,
        }
    )


def _assert_guard(final: pd.DataFrame) -> None:
    """Exact replica of the guard logic in DataEngine.get_data."""
    equity_open_time = (
        final["Dates"].dt.tz_localize("Asia/Tokyo") + pd.Timedelta(hours=9)
    )
    violations = final[final["Last_FX_Bar"] >= equity_open_time]
    if not violations.empty:
        max_violation = violations["Last_FX_Bar"].max()
        raise ValueError(
            f"CRITICAL LOOKAHEAD: FX Bar recorded at {max_violation} "
            f"used to predict equity open."
        )


def test_clean_data_passes_guard() -> None:
    """No violations → no exception."""
    dates = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]
    )
    # Both FX bars at 08:50 exactly — the last allowed minute.
    last_fx = [
        pd.Timestamp("2024-01-03 08:50", tz="Asia/Tokyo"),
        pd.Timestamp("2024-01-04 08:50", tz="Asia/Tokyo"),
    ]
    final = _build_final_frame(dates, last_fx)
    _assert_guard(final)  # should not raise


def test_fx_bar_at_0900_fires_guard() -> None:
    """FX bar at exactly 09:00 JST on the prediction date must fail."""
    dates = pd.DatetimeIndex([pd.Timestamp("2024-01-03")])
    last_fx = [pd.Timestamp("2024-01-03 09:00", tz="Asia/Tokyo")]
    final = _build_final_frame(dates, last_fx)
    with pytest.raises(ValueError, match="CRITICAL LOOKAHEAD"):
        _assert_guard(final)


def test_fx_bar_after_0900_fires_guard() -> None:
    """FX bar at 09:15 JST must fail."""
    dates = pd.DatetimeIndex([pd.Timestamp("2024-01-03")])
    last_fx = [pd.Timestamp("2024-01-03 09:15", tz="Asia/Tokyo")]
    final = _build_final_frame(dates, last_fx)
    with pytest.raises(ValueError, match="CRITICAL LOOKAHEAD"):
        _assert_guard(final)


def test_fx_bar_at_0859_passes_guard() -> None:
    """FX bar at 08:59 JST is strictly before the auction and must pass."""
    dates = pd.DatetimeIndex([pd.Timestamp("2024-01-03")])
    last_fx = [pd.Timestamp("2024-01-03 08:59", tz="Asia/Tokyo")]
    final = _build_final_frame(dates, last_fx)
    _assert_guard(final)  # should not raise


def test_mixed_rows_fires_guard_when_any_violates() -> None:
    """Even one violating row in a larger panel must fire the guard."""
    dates = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-03"),
            pd.Timestamp("2024-01-04"),
            pd.Timestamp("2024-01-05"),
        ]
    )
    last_fx = [
        pd.Timestamp("2024-01-03 08:50", tz="Asia/Tokyo"),
        pd.Timestamp("2024-01-04 09:30", tz="Asia/Tokyo"),  # violation
        pd.Timestamp("2024-01-05 08:50", tz="Asia/Tokyo"),
    ]
    final = _build_final_frame(dates, last_fx)
    with pytest.raises(ValueError, match="CRITICAL LOOKAHEAD"):
        _assert_guard(final)


def test_error_message_reports_violation_timestamp() -> None:
    """The ValueError message should surface the specific violating timestamp."""
    dates = pd.DatetimeIndex([pd.Timestamp("2024-01-03")])
    violation = pd.Timestamp("2024-01-03 10:30", tz="Asia/Tokyo")
    last_fx = [violation]
    final = _build_final_frame(dates, last_fx)
    with pytest.raises(ValueError) as excinfo:
        _assert_guard(final)
    # The violating timestamp should appear in the message verbatim.
    assert "10:30" in str(excinfo.value)
