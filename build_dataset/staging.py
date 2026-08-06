"""Per-month direct-to-final parquet writer.

Writes shards under data/train/train/YYYY-MM/ as rows arrive. Within
each shard, rows are shuffled before flushing. No second balance pass — caps
are tuned per-month to land at target across the whole run.

Each month writes into train/YYYY-MM.tmp/ first, then on success the dir is
atomically renamed to train/YYYY-MM/. _done.json sentinel inside marks commit.
"""
from __future__ import annotations

import gc
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import CHUNK_SIZE, DEFAULT_SEED, TRAIN_DIR
from .extractor import SCHEMA_COLUMNS


class MonthlyWriter:
    """Buffer rows and flush 250K-row shuffled parquet shards into train/YYYY-MM/."""

    def __init__(self, month: str, chunk_size: int = CHUNK_SIZE,
                 seed: int = DEFAULT_SEED):
        self.month = month
        self.chunk_size = chunk_size
        self.dir = TRAIN_DIR / month
        self.tmp_dir = TRAIN_DIR / f"{month}.tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        # Resume: continue numbering past any pre-existing shards in tmp_dir.
        existing = sorted(self.tmp_dir.glob("balanced_*.parquet"))
        self.chunk_idx = len(existing)
        self.buffer: List[dict] = []
        self.rows_written = 0
        self.shards_written = self.chunk_idx
        # Per-(active_elo, tc_cat) yields for diagnostic / state tracking.
        self.yields: Dict[Tuple[int, int], int] = defaultdict(int)
        self.games_seen = 0
        # Per-shard RNG so sequential runs are deterministic given seed.
        self._rng = np.random.default_rng(seed)

    def add_rows(self, rows: List[dict]) -> None:
        if not rows:
            return
        self.buffer.extend(rows)
        for row in rows:
            self.yields[(row["active_elo"], row["tc_cat"])] += 1
        while len(self.buffer) >= self.chunk_size:
            self._flush_chunk()

    def add_games_seen(self, n: int) -> None:
        self.games_seen += n

    def finalize(self, status: str = "extracted",
                 error: Optional[str] = None) -> None:
        """Flush remaining buffer and atomically promote .tmp/ → final/."""
        if self.buffer:
            self._flush_chunk()
        if status == "extracted":
            self._write_done()
            if self.dir.exists():
                # Stale dir from prior failed run.
                import shutil
                shutil.rmtree(self.dir)
            os.replace(self.tmp_dir, self.dir)
        else:
            failed = self.tmp_dir / "_failed.json"
            with open(failed, "w") as f:
                json.dump({"status": status, "error": error,
                           "rows_written": self.rows_written}, f, indent=2)

    def _flush_chunk(self) -> None:
        if not self.buffer:
            return
        write_rows = self.buffer[: self.chunk_size]
        self.buffer = self.buffer[self.chunk_size :]
        df = pd.DataFrame(write_rows, columns=SCHEMA_COLUMNS)
        table = pa.Table.from_pandas(df, preserve_index=False)
        # Within-shard shuffle so consecutive rows aren't from the same game.
        perm = self._rng.permutation(table.num_rows)
        table = table.take(pa.array(perm))
        out_path = self.tmp_dir / f"balanced_{self.chunk_idx:05d}.parquet"
        pq.write_table(table, out_path, compression="zstd")
        self.rows_written += len(write_rows)
        self.shards_written += 1
        self.chunk_idx += 1
        del df, table, write_rows
        gc.collect()

    def _write_done(self) -> None:
        yields_serializable = {
            f"{elo}_{tc}": cnt for (elo, tc), cnt in self.yields.items()
        }
        manifest = {
            "month": self.month,
            "rows_written": self.rows_written,
            "shards_written": self.shards_written,
            "games_seen": self.games_seen,
            "yields_by_active_elo_tc": yields_serializable,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.tmp_dir / "_done.json", "w") as f:
            json.dump(manifest, f, indent=2)
