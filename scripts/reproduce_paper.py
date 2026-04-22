#!/usr/bin/env python3
"""Convenience wrapper around ``src.main reproduce-paper``.

This script exists so the documented command in the README works
verbatim from a fresh clone. All it does is forward arguments to the
main CLI.

    python scripts/reproduce_paper.py               # default data dir
    python scripts/reproduce_paper.py --data-dir X  # override
    python scripts/reproduce_paper.py --no-figures  # skip plotting
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable so ``from src.main import main`` works
# regardless of where this script is invoked from.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.main import main  # noqa: E402 (imports after sys.path edit are intentional)


if __name__ == "__main__":
    # Inject 'reproduce-paper' as the subcommand so users don't have to
    # type it; any other flags they provide (--data-dir, --no-figures,
    # -v) are forwarded untouched.
    argv = ["reproduce-paper"] + sys.argv[1:]
    sys.exit(main(argv))
