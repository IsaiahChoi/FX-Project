"""Data ingestion for FX and equity inputs.

The DataEngine class loads 5-minute USD/JPY bars from CSV and daily
equity OHLC from XLSX (one sheet per ticker), then produces a cleaned,
timezone-aware, lookahead-safe feature set suitable for modeling.

Key methodological choices preserved from the original implementation:

- FX bars are mapped to the **next business day** if recorded at or
  after 15:00 Tokyo time (overnight window for the next open).
- FX bars after 08:50 Tokyo time on the prediction day are excluded so
  that no data at or after the 09:00 opening auction enters the feature
  set. A hard timestamp-level assertion enforces this.
- Opening gaps with absolute value below 1e-5 are truncated to zero to
  mitigate rounding artifacts. (This is flagged in CODE_REVIEW_NOTES.)
- Days with fewer than 50 valid FX bars are dropped.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from .config import ResearchConfig

logger = logging.getLogger(__name__)


class DataEngine:
    """Load, clean, and align FX and equity data.

    Parameters
    ----------
    cfg
        ResearchConfig controlling file paths and data parameters.
    """

    def __init__(self, cfg: ResearchConfig) -> None:
        self.cfg = cfg
        self.fx_daily: Optional[pd.DataFrame] = None
        self.equity_cache: dict[str, pd.DataFrame] = {}
        self._excel_sheets: Optional[dict[str, pd.DataFrame]] = None

    # ------------------------------------------------------------------
    # Equity file handling
    # ------------------------------------------------------------------

    def _load_excel_sheets(self) -> dict[str, pd.DataFrame]:
        """Lazy-load all equity Excel sheets into memory."""
        if self._excel_sheets is None:
            filepath = self.cfg.equity_path
            logger.info(f"Loading Excel file: {filepath}")
            self._excel_sheets = pd.read_excel(
                filepath, sheet_name=None, engine="openpyxl"
            )
        return self._excel_sheets

    def _find_sheet_for_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """Locate the sheet matching a ticker by name or by content scan."""
        sheets = self._load_excel_sheets()
        # First pass: match by sheet name.
        for sheet_name, df in sheets.items():
            if ticker in str(sheet_name):
                return df.copy()
        # Second pass: scan top-left of each sheet for the ticker string.
        for _, df in sheets.items():
            sample_grid = df.iloc[:15, :5].astype(str).values.flatten()
            if any(ticker in cell for cell in sample_grid):
                return df.copy()
        return None

    def _parse_equity_sheet(
        self, raw_df: pd.DataFrame, ticker: str
    ) -> Optional[pd.DataFrame]:
        """Extract a clean OHLC frame from a messy Bloomberg export sheet."""
        df = raw_df.copy()

        # Locate the header row by searching for a Dates/PX_LAST marker.
        header_idx = None
        for i in range(min(20, len(df))):
            row_vals = [str(v).strip() for v in df.iloc[i].values]
            if any("Dates" in v or "PX_LAST" in v for v in row_vals):
                header_idx = i
                break
        if header_idx is None:
            return None

        df.columns = [str(c).strip() for c in df.iloc[header_idx].values]
        df = df.iloc[header_idx + 1 :].reset_index(drop=True)
        df = df[
            [c for c in df.columns if str(c).lower() not in ("nan", "none", "nat", "")]
        ]

        if "Dates" not in df.columns:
            return None

        if not pd.api.types.is_datetime64_any_dtype(df["Dates"]):
            df_numeric = pd.to_numeric(df["Dates"], errors="coerce")
            if df_numeric.notna().mean() > 0.5:
                # Excel serial date origin.
                df["Dates"] = pd.to_datetime(df_numeric, unit="D", origin="1899-12-30")
            else:
                df["Dates"] = pd.to_datetime(df["Dates"], errors="coerce")

        df.dropna(subset=["Dates"], inplace=True)

        required = ["PX_LAST", "PX_OPEN"]
        for col in required:
            if col not in df.columns:
                return None
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=required, inplace=True)
        return df

    # ------------------------------------------------------------------
    # FX file handling
    # ------------------------------------------------------------------

    def _inspect_and_load_fx(self, filepath: str) -> Optional[pd.DataFrame]:
        """Load the 5-minute FX CSV, auto-detecting the header row."""
        try:
            with open(filepath, "r") as f:
                head_lines = [next(f) for _ in range(20)]
            header_idx = next(
                (i for i, line in enumerate(head_lines) if "Dates" in line), -1
            )
            if header_idx == -1:
                return None

            df = pd.read_csv(filepath, skiprows=header_idx)
            df.columns = [str(c).strip() for c in df.columns]

            if pd.api.types.is_numeric_dtype(df["Dates"]):
                df["Dates"] = pd.to_datetime(df["Dates"], unit="D", origin="1899-12-30")
            else:
                df["Dates"] = pd.to_datetime(df["Dates"], errors="coerce")
            df.dropna(subset=["Dates"], inplace=True)

            cols = ["Open", "Close", "High", "Low"]
            if "High" not in df.columns and len(df.columns) >= 5:
                df = df.iloc[:, :5]
                df.columns = ["Dates", "Open", "Close", "High", "Low"]
            for c in cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df.dropna(subset=["Close"], inplace=True)

            # Correct occasional High/Low swaps in Bloomberg exports.
            if "High" in df.columns and "Low" in df.columns:
                if (df["High"] < df["Low"]).mean() > 0.1:
                    df.rename(columns={"High": "Low", "Low": "High"}, inplace=True)
            return df
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to load FX file {filepath}: {exc}")
            return None

    def load_fx(self) -> Optional[pd.DataFrame]:
        """Load and aggregate FX bars to a daily panel with a lookahead guard.

        Returns a frame indexed by prediction date with columns:
        ``FX_Vol`` (std. dev. of 5-min log returns), ``FX_Ret`` (sum of
        log returns), ``FX_Ret_2D``, ``FX_Ret_5D``, ``DoW``, and
        ``Last_FX_Bar`` (timestamp of the latest bar used).
        """
        df_5min = self._inspect_and_load_fx(self.cfg.predictor_path)
        if df_5min is None:
            return None

        # Localize or convert to Tokyo time so the 15:00 / 08:50 cutoffs
        # are meaningful.
        if df_5min["Dates"].dt.tz is None:
            df_5min["Dates"] = df_5min["Dates"].dt.tz_localize("UTC").dt.tz_convert(
                "Asia/Tokyo"
            )
        else:
            df_5min["Dates"] = df_5min["Dates"].dt.tz_convert("Asia/Tokyo")

        df_5min.sort_values("Dates", inplace=True)

        df_5min["Base_Date"] = df_5min["Dates"].dt.normalize().dt.tz_localize(None)
        df_5min["Hour"] = df_5min["Dates"].dt.hour
        df_5min["Minute"] = df_5min["Dates"].dt.minute

        # Evening bars (>= 15:00) map to the next business day; morning
        # bars stay on the current date.
        df_5min["Pred_Date"] = df_5min["Base_Date"]
        df_5min.loc[df_5min["Hour"] >= 15, "Pred_Date"] = df_5min["Base_Date"] + BDay(1)
        df_5min["Pred_Date"] = df_5min["Pred_Date"] + BDay(0)

        # Restrict to the overnight window: 15:00 prior day through 08:50
        # of prediction day. Anything between 09:00 and 14:55 is excluded
        # because it belongs to the continuous trading session.
        is_evening = df_5min["Hour"] >= 15
        is_morning = (df_5min["Hour"] < 8) | (
            (df_5min["Hour"] == 8) & (df_5min["Minute"] < 50)
        )
        df_5min = df_5min[is_evening | is_morning].copy()

        df_5min["Log_Ret"] = df_5min.groupby("Pred_Date")["Close"].transform(
            lambda x: np.log(x / x.shift(1))
        )

        daily = df_5min.groupby("Pred_Date").agg(
            FX_Vol=("Log_Ret", "std"),
            FX_Ret=("Log_Ret", "sum"),
            Bars=("Log_Ret", "count"),
            Last_FX_Bar=("Dates", "max"),
        ).reset_index()

        daily.sort_values("Pred_Date", inplace=True)
        daily["FX_Ret_2D"] = daily["FX_Ret"] + daily["FX_Ret"].shift(1)
        daily["FX_Ret_5D"] = daily["FX_Ret"].rolling(5, min_periods=5).sum()

        daily["DoW"] = daily["Pred_Date"].dt.dayofweek
        daily = daily[daily["Bars"] > 50].dropna(subset=["FX_Ret_5D", "FX_Vol"])

        self.fx_daily = daily
        return daily

    # ------------------------------------------------------------------
    # Combined feature set
    # ------------------------------------------------------------------

    def get_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Build the aligned feature frame for a single ticker.

        Merges per-day FX features with daily equity opening gaps, adds
        day-of-week dummies, and enforces the strict lookahead guard.

        Raises
        ------
        ValueError
            If any FX bar used as a predictor is dated at or after the
            09:00 Tokyo equity opening auction.
        """
        if ticker in self.equity_cache:
            return self.equity_cache[ticker].copy()

        if self.fx_daily is None:
            self.load_fx()

        raw_df = self._find_sheet_for_ticker(ticker)
        if raw_df is None:
            return None

        df_target = self._parse_equity_sheet(raw_df, ticker)
        if df_target is None:
            return None

        df_target.sort_values("Dates", inplace=True)
        df_target["Prev_Close"] = df_target["PX_LAST"].shift(1)
        df_target["Gap"] = (
            (df_target["PX_OPEN"] - df_target["Prev_Close"]) / df_target["Prev_Close"]
        )

        # Rounding-artifact truncation; see CODE_REVIEW_NOTES.md.
        df_target["Gap"] = np.where(
            df_target["Gap"].abs() < 1e-5, 0.0, df_target["Gap"]
        )

        df_target["Gap_Lag1"] = df_target["Gap"].shift(1)
        df_target["Gap_Lag2"] = df_target["Gap"].shift(2)

        df_target.dropna(subset=["Gap", "Gap_Lag2"], inplace=True)

        final = pd.merge(
            df_target[["Dates", "Gap", "Gap_Lag1", "Gap_Lag2"]],
            self.fx_daily[
                ["Pred_Date", "FX_Vol", "FX_Ret", "FX_Ret_2D", "FX_Ret_5D",
                 "DoW", "Last_FX_Bar"]
            ],
            left_on="Dates",
            right_on="Pred_Date",
            how="inner",
        )

        final["DoW"] = pd.Categorical(final["DoW"], categories=[0, 1, 2, 3, 4])
        final = pd.get_dummies(
            final, columns=["DoW"], drop_first=True, prefix="DoW", dtype=int
        )
        final.drop_duplicates(subset=["Dates"], inplace=True)

        # Hard lookahead guard — abort if any FX bar timestamp is at or
        # after the 09:00 Tokyo auction.
        equity_open_time = (
            final["Dates"].dt.tz_localize("Asia/Tokyo") + pd.Timedelta(hours=9)
        )
        violations = final[final["Last_FX_Bar"] >= equity_open_time]
        if not violations.empty:
            max_violation = violations["Last_FX_Bar"].max()
            offending_date = violations.loc[
                violations["Last_FX_Bar"] == max_violation, "Dates"
            ].iloc[0]
            raise ValueError(
                f"CRITICAL LOOKAHEAD: FX bar recorded at {max_violation} "
                f"used to predict equity open on {offending_date.date()}"
            )

        final.drop(columns=["Pred_Date", "Last_FX_Bar"], inplace=True, errors="ignore")
        final["Equity"] = ticker
        final["Target_Class"] = (final["Gap"] > 0).astype(int)
        final.set_index("Dates", inplace=True)

        self.equity_cache[ticker] = final
        return final.copy()
