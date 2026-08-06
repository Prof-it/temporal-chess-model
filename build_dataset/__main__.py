"""Entry point for ``python -m build_dataset``.

  python -m build_dataset --status            # state summary
  python -m build_dataset --end 2025-12       # build backward from 2025-12 until filled
  python -m build_dataset --balance-only      # rerun the balance pass
  python -m build_dataset --verify-only       # rerun verification
  python -m build_dataset --restart-month 2024-07

See build_dataset/cli.py for the full option set.
"""
import sys

from build_dataset.cli import main

if __name__ == "__main__":
    sys.exit(main())
