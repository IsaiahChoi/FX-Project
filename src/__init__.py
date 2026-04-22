"""FX Gap Trading in Japanese Equities — research codebase.

A refactored version of the original single-file research code, split
into focused modules for clarity and maintainability. All numerical
choices, hyperparameters, and methodological decisions are preserved
exactly from the original implementation.

Public entry points
-------------------

>>> from src.evaluation import run_all_experiments
>>> from src.plotting import plot_all
>>> df_results, ts_cache, importances, cfg = run_all_experiments()
>>> plot_all(df_results, ts_cache, importances, cfg)

Or via the command line::

    python -m src.main reproduce-paper

Module organisation
-------------------

- ``config``: ResearchConfig dataclass (paths, hyperparameters).
- ``data_engine``: FX and equity data loading with lookahead guard.
- ``features``: panel assembly, standardization, chronological split.
- ``models``: HAC OLS fitting and Random Forest wrappers.
- ``econometrics``: CCF, Granger, VAR/IRF, rolling R², DM test, MI.
- ``evaluation``: experiment orchestration and results aggregation.
- ``plotting``: six paper figures.
- ``main``: CLI entry point.
"""

__version__ = "1.0.0"
