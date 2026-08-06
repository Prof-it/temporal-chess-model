"""Persistent build state — survives Ctrl-C, lets the orchestrator resume.

Schema (build_state.json):
    {
      "version": 1,
      "created": "ISO timestamp",
      "config": {month_range, target_per_bin, tc_targets, seed, workers, ...},
      "months": {
        "YYYY-MM": {
          "status": "pending|downloading|extracting|extracted|failed|unavailable",
          "url": "...",
          "started_at": "...", "finished_at": "...",
          "games_seen": int, "rows_kept": int,
          "stage_dir": "data/train/stage/YYYY-MM",
          "stage_shards": int,
          "yields": {"<elo_bin>_<tc_cat>": int, ...},
          "error": "optional error message",
        }
      },
      "balance_pass": {
        "status": "pending|in_progress|done|failed",
        "started_at": null, "finished_at": null,
        "final_counts": null,
      }
    }
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .config import STATE_FILE, BALANCED_DIR


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def empty_state() -> dict:
    return {
        "version": 1,
        "created": now_str(),
        "updated": now_str(),
        "config": None,
        "months": {},
        "balance_pass": {
            "status": "pending",
            "started_at": None,
            "finished_at": None,
            "final_counts": None,
        },
    }


def load_state(path: Path = STATE_FILE) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return empty_state()


def save_state(state: dict, path: Path = STATE_FILE) -> None:
    state["updated"] = now_str()
    BALANCED_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def get_month(state: dict, month: str) -> dict:
    """Return per-month state dict, creating empty entry if missing."""
    if month not in state["months"]:
        state["months"][month] = {
            "status": "pending",
            "url": None,
            "started_at": None,
            "finished_at": None,
            "games_seen": 0,
            "rows_kept": 0,
            "stage_dir": None,
            "stage_shards": 0,
            "yields": {},
            "error": None,
        }
    return state["months"][month]


def set_month_status(state: dict, month: str, status: str, **extra) -> None:
    m = get_month(state, month)
    m["status"] = status
    m.update(extra)


def month_is_complete(state: dict, month: str) -> bool:
    return state["months"].get(month, {}).get("status") == "extracted"


def reset_month(state: dict, month: str) -> None:
    """Wipe a month from state (used by --restart-month)."""
    state["months"].pop(month, None)
