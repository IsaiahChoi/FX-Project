"""Command-line entry point for the FX gap research pipeline.

Usage examples::

    # Reproduce the paper end-to-end (table + all six figures).
    python -m src.main reproduce-paper

    # Just print the table; skip figure generation.
    python -m src.main run-econometrics --no-figures

    # Override data directory without modifying config defaults.
    FX_GAP_DATA_DIR=/path/to/data python -m src.main reproduce-paper
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe; prevents display backend errors on servers

from .config import ResearchConfig
from .evaluation import format_results_table, run_all_experiments
from .plotting import plot_all


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _save_results_table(df_results, cfg: ResearchConfig) -> str:
    """Save the results DataFrame as CSV in the tables directory."""
    os.makedirs(cfg.tables_dir, exist_ok=True)
    path = os.path.join(cfg.tables_dir, "table1_results.csv")
    df_results.to_csv(path, index=False)
    return path


def cmd_reproduce_paper(args: argparse.Namespace) -> int:
    """Run all experiments and produce every figure."""
    cfg = ResearchConfig(base_path=args.data_dir) if args.data_dir else ResearchConfig()

    df_results, ts_cache, importances, cfg = run_all_experiments(cfg=cfg)
    if df_results.empty:
        logging.error("No experiments succeeded; nothing to report.")
        return 1

    print("\n" + "=" * 140)
    print("ECONOMETRIC LEAD-LAG EVIDENCE: USD/JPY OVERNIGHT RETURNS vs NEXT-DAY EQUITY GAP")
    print("=" * 140 + "\n")
    print(format_results_table(df_results))
    print()

    table_path = _save_results_table(df_results, cfg)
    logging.info("Results table saved: %s", table_path)

    if not args.no_figures:
        plot_all(df_results, ts_cache, importances, cfg)

    return 0


def cmd_run_econometrics(args: argparse.Namespace) -> int:
    """Run the econometric pipeline only; skip figure generation unless requested."""
    cfg = ResearchConfig(base_path=args.data_dir) if args.data_dir else ResearchConfig()

    df_results, ts_cache, importances, cfg = run_all_experiments(cfg=cfg)
    if df_results.empty:
        logging.error("No experiments succeeded; nothing to report.")
        return 1

    print(format_results_table(df_results))
    table_path = _save_results_table(df_results, cfg)
    logging.info("Results table saved: %s", table_path)

    if args.figures:
        plot_all(df_results, ts_cache, importances, cfg)

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fx-gap",
        description="FX gap research pipeline: run HAC OLS, CCF, Granger, VAR/IRF, rolling R², and Random Forest experiments across seven configurations.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory (default: repo_root/data or $FX_GAP_DATA_DIR).",
    )

    sub = p.add_subparsers(dest="command", required=True)

    p_repro = sub.add_parser(
        "reproduce-paper", help="Run every experiment and generate all paper figures."
    )
    p_repro.add_argument(
        "--no-figures", action="store_true", help="Skip figure generation."
    )
    p_repro.set_defaults(func=cmd_reproduce_paper)

    p_econ = sub.add_parser(
        "run-econometrics", help="Run experiments; optionally produce figures."
    )
    p_econ.add_argument(
        "--figures", action="store_true", help="Also generate paper figures."
    )
    p_econ.set_defaults(func=cmd_run_econometrics)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
