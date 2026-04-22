"""Tests for the data engine's parsing behavior.

These tests use hand-constructed DataFrames (not real Bloomberg data)
to verify the DataEngine's resilience to the quirks of Bloomberg's
export format: metadata rows at the top, mixed column orders, Excel
serial date formatting, and occasional High/Low column swaps.

If any of these parsers change behavior, a code review is required
— the paper's numerical results depend on them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import ResearchConfig
from src.data_engine import DataEngine


def test_parse_equity_sheet_finds_header_after_metadata_rows() -> None:
    """Bloomberg exports often have metadata before the actual table.

    The parser should locate the header row by searching for 'Dates'
    or 'PX_LAST' within the first 20 rows.
    """
    # Simulate a Bloomberg sheet with 3 metadata rows before the header.
    raw_rows = [
        ["Ticker", "5711 JT Equity", None, None],
        ["Company", "Mitsubishi Materials", None, None],
        [None, None, None, None],
        ["Dates", "PX_OPEN", "PX_LAST", "Extra"],
        [pd.Timestamp("2024-01-03"), 3000.0, 3010.0, "ignored"],
        [pd.Timestamp("2024-01-04"), 3005.0, 3015.0, "ignored"],
    ]
    raw_df = pd.DataFrame(raw_rows)

    cfg = ResearchConfig()
    engine = DataEngine(cfg)
    result = engine._parse_equity_sheet(raw_df, ticker="5711")

    assert result is not None
    assert "Dates" in result.columns
    assert "PX_OPEN" in result.columns
    assert "PX_LAST" in result.columns
    assert len(result) == 2


def test_parse_equity_sheet_returns_none_on_missing_required_columns() -> None:
    """If PX_OPEN or PX_LAST is missing, parser should return None."""
    raw_rows = [
        ["Dates", "SomeOtherColumn"],
        [pd.Timestamp("2024-01-03"), 3000.0],
    ]
    raw_df = pd.DataFrame(raw_rows)

    cfg = ResearchConfig()
    engine = DataEngine(cfg)
    result = engine._parse_equity_sheet(raw_df, ticker="5711")
    assert result is None


def test_parse_equity_sheet_coerces_numeric_columns() -> None:
    """Strings in PX_OPEN/PX_LAST should be coerced to numeric, NaN rows dropped."""
    raw_rows = [
        ["Dates", "PX_OPEN", "PX_LAST"],
        [pd.Timestamp("2024-01-03"), "3000.0", "3010.0"],
        [pd.Timestamp("2024-01-04"), "not_a_number", "3015.0"],  # should be dropped
        [pd.Timestamp("2024-01-05"), "3005.0", "3020.0"],
    ]
    raw_df = pd.DataFrame(raw_rows)

    cfg = ResearchConfig()
    engine = DataEngine(cfg)
    result = engine._parse_equity_sheet(raw_df, ticker="5711")

    assert result is not None
    assert len(result) == 2  # row with 'not_a_number' dropped
    assert pd.api.types.is_numeric_dtype(result["PX_OPEN"])
    assert pd.api.types.is_numeric_dtype(result["PX_LAST"])


def test_parse_equity_sheet_handles_excel_serial_dates() -> None:
    """Excel stores dates as numbers from 1899-12-30. Parser must detect this."""
    # Excel serial 45294 corresponds to 2024-01-03.
    raw_rows = [
        ["Dates", "PX_OPEN", "PX_LAST"],
        [45294, 3000.0, 3010.0],
        [45295, 3005.0, 3015.0],
    ]
    raw_df = pd.DataFrame(raw_rows)

    cfg = ResearchConfig()
    engine = DataEngine(cfg)
    result = engine._parse_equity_sheet(raw_df, ticker="5711")

    assert result is not None
    # Dates should be parsed to actual datetimes, not left as integers.
    assert pd.api.types.is_datetime64_any_dtype(result["Dates"])
    assert result["Dates"].iloc[0] == pd.Timestamp("2024-01-03")


def test_research_config_paths_resolve() -> None:
    """Default config should produce valid-looking paths, even without data present."""
    cfg = ResearchConfig()
    # Paths should be strings; they need not exist yet.
    assert isinstance(cfg.predictor_path, str)
    assert isinstance(cfg.equity_path, str)
    assert cfg.predictor_path.endswith("USDJPY.csv")
    assert cfg.equity_path.endswith("JPEquities.xlsx")


def test_research_config_respects_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """FX_GAP_DATA_DIR env var should override the default base_path."""
    custom = "/tmp/custom_fx_gap_data"
    monkeypatch.setenv("FX_GAP_DATA_DIR", custom)
    cfg = ResearchConfig()
    assert cfg.base_path == custom
    assert cfg.predictor_path == f"{custom}/USDJPY.csv"


def test_research_config_hyperparameters_match_paper() -> None:
    """Critical hyperparameters must match what the paper reports."""
    cfg = ResearchConfig()
    assert cfg.n_estimators == 500
    assert cfg.random_state == 42
    assert cfg.test_size == 0.2
    assert cfg.max_lag_analysis == 5
    assert set(cfg.equities.keys()) == {"5713", "5711", "5706"}


def test_find_sheet_for_ticker_matches_by_name() -> None:
    """Sheet lookup should match a ticker code embedded in the sheet name."""
    cfg = ResearchConfig()
    engine = DataEngine(cfg)
    # Inject a cached set of sheets.
    engine._excel_sheets = {
        "5711 JT Equity": pd.DataFrame({"col": [1, 2]}),
        "5713 JT Equity": pd.DataFrame({"col": [3, 4]}),
    }
    result = engine._find_sheet_for_ticker("5711")
    assert result is not None
    assert list(result["col"]) == [1, 2]


def test_find_sheet_for_ticker_returns_none_on_miss() -> None:
    """Returns None when no sheet matches the requested ticker."""
    cfg = ResearchConfig()
    engine = DataEngine(cfg)
    engine._excel_sheets = {
        "SomeOtherStock": pd.DataFrame({"Dates": [pd.Timestamp("2024-01-01")]}),
    }
    result = engine._find_sheet_for_ticker("5711")
    assert result is None
