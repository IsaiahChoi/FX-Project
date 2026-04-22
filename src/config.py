"""Configuration for the FX gap research pipeline.

Defines file paths, model hyperparameters, and experimental parameters.
Paths can be overridden via environment variables or by constructing
ResearchConfig with custom arguments — no hardcoded user-specific paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_base_path() -> str:
    """Return the default data directory, overridable via FX_GAP_DATA_DIR env var."""
    env_override = os.environ.get("FX_GAP_DATA_DIR")
    if env_override:
        return env_override
    # Default: data/ directory at the repo root.
    return str(Path(__file__).resolve().parents[1] / "data")


@dataclass
class ResearchConfig:
    """Central configuration for the research pipeline.

    Attributes
    ----------
    base_path
        Directory containing the input data files. Defaults to ``data/``
        at the repository root but can be overridden via the
        ``FX_GAP_DATA_DIR`` environment variable or by passing a value
        explicitly.
    predictor_filename
        CSV filename containing 5-minute USD/JPY bars.
    equity_filename
        XLSX filename containing daily equity OHLC data across one sheet
        per ticker.
    cost_bps_grid
        Grid of transaction cost assumptions (basis points per round
        trip) used in the net-of-cost simulation.
    equities
        Mapping of TSE ticker code to firm name. The three firms studied
        in the paper are Sumitomo Metal Mining, Mitsubishi Materials,
        and Mitsui Mining & Smelting.
    test_size
        Fraction of observations held out for out-of-sample evaluation.
    n_estimators
        Number of trees in the Random Forest classifier.
    random_state
        Seed for Random Forest and any other stochastic component.
    trading_days
        Trading days per year used to annualize returns and volatility.
    max_lag_analysis
        Maximum lag order for CCF, Granger causality, and IRF analyses.
        Paper uses 5.
    figures_dir
        Output directory for generated figures.
    tables_dir
        Output directory for generated result tables.
    """

    base_path: str = field(default_factory=_default_base_path)
    predictor_filename: str = "USDJPY.csv"
    equity_filename: str = "JPEquities.xlsx"

    cost_bps_grid: list[int] = field(default_factory=lambda: [5, 10, 15, 20])

    equities: dict[str, str] = field(
        default_factory=lambda: {
            "5713": "Sumitomo Metal Mining",
            "5711": "Mitsubishi Materials",
            "5706": "Mitsui Mining & Smelting",
        }
    )

    test_size: float = 0.2
    n_estimators: int = 500
    random_state: int = 42
    trading_days: int = 245
    max_lag_analysis: int = 5

    figures_dir: str = field(
        default_factory=lambda: str(
            Path(__file__).resolve().parents[1] / "results" / "figures"
        )
    )
    tables_dir: str = field(
        default_factory=lambda: str(
            Path(__file__).resolve().parents[1] / "results" / "tables"
        )
    )

    @property
    def predictor_path(self) -> str:
        """Full filesystem path to the FX CSV file."""
        return os.path.join(self.base_path, self.predictor_filename)

    @property
    def equity_path(self) -> str:
        """Full filesystem path to the equities Excel file."""
        return os.path.join(self.base_path, self.equity_filename)
