"""Tests for feature construction and the chronological split.

Verifies that:

- ``standardize_columns`` produces mean ≈ 0 and std ≈ 1 for the
  configured columns, without distorting the structure of the panel.
- ``split_chronologically`` produces non-empty train and test sets
  with a business-day embargo between them.
- ``BASE_FEATURES`` contains the documented feature list.
- Standardization skips columns that aren't present without raising.

These tests do NOT require Bloomberg data; they use in-memory fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    BASE_FEATURES,
    COLS_TO_STANDARDIZE,
    split_chronologically,
    standardize_columns,
)


def test_base_features_matches_paper_specification() -> None:
    """Base feature set must match what the paper documents."""
    expected = {"FX_Vol", "FX_Ret", "FX_Ret_2D", "FX_Ret_5D", "Gap_Lag1", "Gap_Lag2"}
    assert set(BASE_FEATURES) == expected


def test_cols_to_standardize_includes_gap_and_fx_features() -> None:
    """Standardized column list must cover Gap, FX returns, and gap lags."""
    required = {"Gap", "FX_Ret", "FX_Vol", "Gap_Lag1", "Gap_Lag2"}
    assert required.issubset(set(COLS_TO_STANDARDIZE))


def test_standardize_produces_unit_variance(standardized_panel: pd.DataFrame) -> None:
    """Columns in COLS_TO_STANDARDIZE should have std ≈ 1 post-standardization."""
    df = standardized_panel.copy()
    # Un-standardize first so the test exercises the standardizer rather
    # than assuming the fixture is already standardized.
    df["Gap"] = df["Gap"] * 10 + 5  # scale up and shift
    df["FX_Ret"] = df["FX_Ret"] * 3 + 2

    out = standardize_columns(df, cols=["Gap", "FX_Ret"])
    assert abs(out["Gap"].mean()) < 1e-9
    assert abs(out["FX_Ret"].mean()) < 1e-9
    assert abs(out["Gap"].std() - 1.0) < 1e-9
    assert abs(out["FX_Ret"].std() - 1.0) < 1e-9


def test_standardize_does_not_mutate_input(standardized_panel: pd.DataFrame) -> None:
    """Standardization must not modify the input DataFrame in place."""
    df_before = standardized_panel.copy()
    _ = standardize_columns(standardized_panel, cols=["Gap"])
    pd.testing.assert_frame_equal(standardized_panel, df_before)


def test_standardize_skips_missing_columns(standardized_panel: pd.DataFrame) -> None:
    """Requested columns not present in the DataFrame should be silently skipped."""
    # This should not raise even though 'NONEXISTENT' isn't in the frame.
    out = standardize_columns(standardized_panel, cols=["Gap", "NONEXISTENT"])
    assert "Gap" in out.columns
    assert "NONEXISTENT" not in out.columns


def test_standardize_handles_zero_variance_column() -> None:
    """A constant column should not be divided by zero; either left alone
    or set to zero — never NaN or Inf."""
    df = pd.DataFrame(
        {"Constant": np.ones(10), "Varying": np.arange(10, dtype=float)},
        index=pd.date_range("2024-01-01", periods=10),
    )
    out = standardize_columns(df, cols=["Constant", "Varying"])
    # 'Varying' should standardize correctly.
    assert abs(out["Varying"].std() - 1.0) < 1e-9
    # 'Constant' should not be Inf or NaN (implementation leaves unchanged).
    assert np.all(np.isfinite(out["Constant"]))


def test_split_chronologically_produces_non_empty_splits(
    standardized_panel: pd.DataFrame,
) -> None:
    """Both train and test should be non-empty at default 0.2 test size."""
    train, test = split_chronologically(standardized_panel, test_size=0.2)
    assert len(train) > 0
    assert len(test) > 0


def test_split_chronologically_train_precedes_test(
    standardized_panel: pd.DataFrame,
) -> None:
    """Every training date must be strictly earlier than every test date."""
    train, test = split_chronologically(standardized_panel, test_size=0.2)
    assert train.index.max() < test.index.min()


def test_split_chronologically_enforces_business_day_embargo(
    standardized_panel: pd.DataFrame,
) -> None:
    """Gap between last train date and first test date must be >= 2 bdays."""
    train, test = split_chronologically(
        standardized_panel, test_size=0.2, embargo_bdays=2
    )
    gap_bdays = len(pd.bdate_range(train.index.max(), test.index.min())) - 1
    # Strict: at least 2 business days between the two endpoints.
    assert gap_bdays >= 2


def test_split_chronologically_respects_test_size_fraction(
    standardized_panel: pd.DataFrame,
) -> None:
    """Train/test fractions should roughly match the requested test_size."""
    train, test = split_chronologically(standardized_panel, test_size=0.2)
    total = len(train) + len(test)
    # Embargo removes a few observations, so allow a loose tolerance.
    assert 0.1 < len(test) / total < 0.25


def test_split_raises_on_empty_result() -> None:
    """Extreme test_size should trigger the assertion, not silently succeed."""
    tiny = pd.DataFrame(
        {"x": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2)
    )
    with pytest.raises(AssertionError, match="Split failed"):
        split_chronologically(tiny, test_size=0.99)
