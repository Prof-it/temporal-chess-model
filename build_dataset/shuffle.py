"""Global shuffle pass over all per-month shards.

Reads every shard under train/<month>/*.parquet, writes globally-shuffled output
shards under train/shuffled/. Optionally deletes the per-month input dirs after.

Two-pass approach (memory-bounded):
  Pass 1 — scatter: each input row gets a random partition id ∈ [0, K).
           Per-partition writers buffer rows and flush as parquet shards.
    Pass 2 — gather + shuffle: read partition subshards in bounded batches,
                     shuffle each gathered batch in memory, slice into CHUNK_SIZE final
                     shards, write, then delete the consumed partition files.

K is chosen so each partition stays reasonably sized on disk. Peak RAM is then
bounded by the pass-2 gather limit rather than total partition size. Total disk
peak during the pass: 2× input size (originals + partition files), then drops as
partitions are consumed.
"""
from __future__ import annotations

import gc
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import CHUNK_SIZE, DEFAULT_SEED, TRAIN_DIR


SHUFFLED_SUBDIR = "shuffled"
PARTITIONS_SUBDIR = "_shuffle_partitions"
PARTITION_FLUSH_ROWS = 200_000  # rows per partition flush
PASS1_READ_BATCH_ROWS = 65_536
PASS1_MAX_BUFFERED_ROWS = 1_000_000
PASS2_READ_BATCH_ROWS = 16_384
PASS2_MAX_ROWS_IN_MEMORY = 32_768


def _write_table_atomic(table: pa.Table, out_path: Path) -> None:
    """Write parquet to a temp file and atomically rename into place."""
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    pq.write_table(table, tmp_path, compression="zstd")
    os.replace(tmp_path, out_path)


def _input_shards(train_dir: Path) -> list:
    """Return committed input shards only.

    Excludes shuffle scratch/output dirs and failed per-month staging dirs that
    still end in .tmp.
    """
    excluded_dirs = {SHUFFLED_SUBDIR, PARTITIONS_SUBDIR}
    return sorted(
        s for s in train_dir.rglob("balanced_*.parquet")
        if not any(part in excluded_dirs or part.endswith(".tmp") for part in s.parts)
    )


def _estimate_partitions(input_shards: list, target_partition_rows: int = 5_000_000) -> int:
    """Pick K so each partition is ~target_partition_rows when populated."""
    total_rows = 0
    for s in input_shards[:10]:  # sample first 10 to estimate
        total_rows += pq.ParquetFile(s).metadata.num_rows
    if not input_shards:
        return 1
    avg_per_shard = total_rows / min(10, len(input_shards))
    estimated_total = int(avg_per_shard * len(input_shards))
    k = max(1, estimated_total // target_partition_rows)
    # Round to a reasonable number
    return min(max(k, 16), 256)


class _PartitionWriter:
    """Buffers rows for one partition, flushes when over PARTITION_FLUSH_ROWS."""

    def __init__(self, path_prefix: Path):
        self.prefix = path_prefix
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        self.shard_idx = 0
        self.buffered_tables: list = []
        self.buffered_rows = 0

    def add(self, table: pa.Table) -> None:
        self.buffered_tables.append(table)
        self.buffered_rows += table.num_rows
        if self.buffered_rows >= PARTITION_FLUSH_ROWS:
            self.flush()

    def flush(self) -> None:
        if not self.buffered_tables:
            return
        combined = pa.concat_tables(self.buffered_tables)
        out = self.prefix.parent / f"{self.prefix.name}_{self.shard_idx:05d}.parquet"
        _write_table_atomic(combined, out)
        self.shard_idx += 1
        self.buffered_tables = []
        self.buffered_rows = 0

    def finalize(self) -> None:
        self.flush()


def _flush_partition_buffers(
    writers: list[_PartitionWriter],
    max_total_rows: int,
) -> None:
    """Bound total in-memory partition buffering across all writers."""
    total_rows = sum(writer.buffered_rows for writer in writers)
    if total_rows <= max_total_rows:
        return

    for writer in sorted(writers, key=lambda item: item.buffered_rows, reverse=True):
        if total_rows <= max_total_rows:
            break
        if writer.buffered_rows == 0:
            continue
        total_rows -= writer.buffered_rows
        writer.flush()


def _scatter_table_to_partitions(
    table: pa.Table,
    writers: list[_PartitionWriter],
    num_partitions: int,
    rng: np.random.Generator,
) -> None:
    """Assign a table's rows to partitions without materializing K boolean masks."""
    if table.num_rows == 0:
        return

    partitions = rng.integers(0, num_partitions, table.num_rows)
    order = np.argsort(partitions, kind="stable")
    sorted_partitions = partitions[order]
    sorted_table = table.take(pa.array(order))

    change_points = np.flatnonzero(sorted_partitions[1:] != sorted_partitions[:-1]) + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [table.num_rows]))
    for start, end in zip(starts.tolist(), ends.tolist()):
        writers[int(sorted_partitions[start])].add(sorted_table.slice(start, end - start))


def _shuffle_tables(tables: list[pa.Table], rng: np.random.Generator) -> Optional[pa.Table]:
    """Shuffle one bounded in-memory batch of tables."""
    if not tables:
        return None
    if len(tables) == 1:
        batch = tables[0]
    else:
        batch = pa.concat_tables(tables)
    perm = rng.permutation(batch.num_rows)
    return batch.take(pa.array(perm))


def _release_arrow_memory() -> None:
    """Hint Arrow to return unused buffers after large batch operations."""
    gc.collect()
    try:
        pa.default_memory_pool().release_unused()
    except (AttributeError, NotImplementedError):
        pass


class _ShuffledOutputWriter:
    """Buffers shuffled tables and writes fixed-size final shards."""

    def __init__(self, out_dir: Path, chunk_size: int, start_idx: int = 0, total_rows: int = 0):
        self.out_dir = out_dir
        self.chunk_size = chunk_size
        self.final_idx = start_idx
        self.total_rows = total_rows
        self.buffered_tables: list[pa.Table] = []
        self.buffered_rows = 0

    def add(self, table: Optional[pa.Table]) -> None:
        if table is None or table.num_rows == 0:
            return
        self.buffered_tables.append(table)
        self.buffered_rows += table.num_rows
        if self.buffered_rows >= self.chunk_size:
            self._flush_full_chunks()

    def _flush_full_chunks(self) -> None:
        if self.buffered_rows < self.chunk_size:
            return

        combined = pa.concat_tables(self.buffered_tables)
        offset = 0
        while combined.num_rows - offset >= self.chunk_size:
            chunk = combined.slice(offset, self.chunk_size)
            out_path = self.out_dir / f"shuffled_{self.final_idx:05d}.parquet"
            _write_table_atomic(chunk, out_path)
            self.final_idx += 1
            self.total_rows += chunk.num_rows
            offset += self.chunk_size

        remainder = combined.slice(offset)
        self.buffered_tables = [remainder] if remainder.num_rows else []
        self.buffered_rows = remainder.num_rows
        del combined
        _release_arrow_memory()

    def finalize(self) -> None:
        if not self.buffered_tables:
            return
        combined = pa.concat_tables(self.buffered_tables)
        if combined.num_rows:
            out_path = self.out_dir / f"shuffled_{self.final_idx:05d}.parquet"
            _write_table_atomic(combined, out_path)
            self.final_idx += 1
            self.total_rows += combined.num_rows
        self.buffered_tables = []
        self.buffered_rows = 0
        del combined
        _release_arrow_memory()


def _existing_shuffled_state(out_dir: Path) -> tuple[int, int]:
    """Return next shard index and rows already written in shuffled/ for resume.

    Zero-byte or unreadable files are treated as interrupted writes and removed.
    """
    existing = sorted(out_dir.glob("shuffled_*.parquet"))
    if not existing:
        return 0, 0

    total_rows = 0
    max_idx = -1
    for shard in existing:
        if shard.stat().st_size == 0:
            shard.unlink()
            continue
        try:
            total_rows += pq.ParquetFile(shard).metadata.num_rows
        except Exception:
            shard.unlink()
            continue
        try:
            shard_idx = int(shard.stem.split("_")[-1])
        except ValueError:
            continue
        max_idx = max(max_idx, shard_idx)
    return max_idx + 1, total_rows


def _has_existing_partitions(part_dir: Path) -> bool:
    return any(part_dir.glob("p*.parquet"))


def run_shuffle(
    train_dir: Path = TRAIN_DIR,
    seed: int = DEFAULT_SEED,
    num_partitions: Optional[int] = None,
    delete_inputs: bool = False,
    chunk_size: int = CHUNK_SIZE,
    max_rows_in_memory: int = PASS2_MAX_ROWS_IN_MEMORY,
    scatter_batch_rows: int = PASS1_READ_BATCH_ROWS,
    scatter_max_buffered_rows: int = PASS1_MAX_BUFFERED_ROWS,
    gather_batch_rows: int = PASS2_READ_BATCH_ROWS,
) -> dict:
    """Shuffle all input shards into train/shuffled/.

    Returns a summary dict suitable for the manifest.
    """
    train_dir = Path(train_dir)
    inputs = _input_shards(train_dir)
    if not inputs:
        raise FileNotFoundError(f"No shards under {train_dir} (excluded shuffled/, partitions/)")

    if num_partitions is None:
        num_partitions = _estimate_partitions(inputs)

    out_dir = train_dir / SHUFFLED_SUBDIR
    part_dir = train_dir / PARTITIONS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    part_dir.mkdir(parents=True, exist_ok=True)

    next_output_idx, existing_output_rows = _existing_shuffled_state(out_dir)
    resume_pass2 = _has_existing_partitions(part_dir)

    print(f"[shuffle] inputs: {len(inputs)} shards, partitions: K={num_partitions}", flush=True)
    print(f"[shuffle] partition dir: {part_dir}")
    print(f"[shuffle] output dir:    {out_dir}")
    if resume_pass2:
        print("[shuffle] resuming from existing partition files; skipping pass1", flush=True)
    if next_output_idx:
        print(f"[shuffle] found {next_output_idx} existing shuffled shards ({existing_output_rows:,} rows)", flush=True)
    t0 = time.time()
    rng = np.random.default_rng(seed)

    # ── Pass 1: scatter rows into K partitions ────────────────────────────────
    total_in = 0
    if resume_pass2:
        pass1_elapsed = 0.0
        total_in = sum(pq.ParquetFile(shard).metadata.num_rows for shard in inputs)
    else:
        print(f"\n[shuffle/pass1] scattering rows into {num_partitions} partitions...", flush=True)
        writers = [_PartitionWriter(part_dir / f"p{i:03d}") for i in range(num_partitions)]
        for i, shard in enumerate(inputs):
            parquet_file = pq.ParquetFile(shard)
            for batch in parquet_file.iter_batches(batch_size=scatter_batch_rows):
                table = pa.Table.from_batches([batch])
                total_in += table.num_rows
                _scatter_table_to_partitions(table, writers, num_partitions, rng)
                _flush_partition_buffers(writers, scatter_max_buffered_rows)
            del parquet_file
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  pass1 {i+1}/{len(inputs)} rows={total_in:,} elapsed={elapsed:.0f}s",
                      flush=True)
                _release_arrow_memory()
        for w in writers:
            w.finalize()
        pass1_elapsed = time.time() - t0
        print(f"[shuffle/pass1] done. rows={total_in:,} elapsed={pass1_elapsed:.0f}s", flush=True)

    # ── Pass 2: gather each partition in bounded batches, shuffle, write ────
    print("\n[shuffle/pass2] gathering + shuffling partitions...", flush=True)
    pass2_t0 = time.time()
    output_writer = _ShuffledOutputWriter(
        out_dir,
        chunk_size,
        start_idx=next_output_idx,
        total_rows=existing_output_rows,
    )
    partition_order = list(range(num_partitions))
    rng.shuffle(partition_order)
    for order_idx, p in enumerate(partition_order, start=1):
        part_files = sorted(part_dir.glob(f"p{p:03d}_*.parquet"))
        if not part_files:
            continue
        rng.shuffle(part_files)

        batch_tables: list = []
        batch_rows = 0
        for f in part_files:
            parquet_file = pq.ParquetFile(f)
            for batch in parquet_file.iter_batches(batch_size=gather_batch_rows):
                table = pa.Table.from_batches([batch])
                batch_tables.append(table)
                batch_rows += table.num_rows
                if batch_rows >= max_rows_in_memory:
                    output_writer.add(_shuffle_tables(batch_tables, rng))
                    batch_tables = []
                    batch_rows = 0
                    _release_arrow_memory()
            del parquet_file
            f.unlink()

        if batch_tables:
            output_writer.add(_shuffle_tables(batch_tables, rng))
            batch_tables = []

        _release_arrow_memory()
        if order_idx % 10 == 0:
            elapsed = time.time() - pass2_t0
            print(f"  pass2 {order_idx}/{num_partitions} out_shards={output_writer.final_idx} elapsed={elapsed:.0f}s",
                  flush=True)
    output_writer.finalize()
    final_idx = output_writer.final_idx
    total_out = output_writer.total_rows
    # Remove empty partitions dir
    try:
        part_dir.rmdir()
    except OSError:
        pass

    pass2_elapsed = time.time() - pass2_t0
    print(f"[shuffle/pass2] done. shards={final_idx} rows={total_out:,} "
          f"elapsed={pass2_elapsed:.0f}s", flush=True)

    # ── Optional: remove per-month input dirs ────────────────────────────────
    if delete_inputs:
        print("\n[shuffle] deleting per-month input dirs...", flush=True)
        # Only remove dirs that contained only input shards (avoid blasting unknown stuff)
        for month_dir in train_dir.iterdir():
            if month_dir.is_dir() and month_dir.name not in (SHUFFLED_SUBDIR, PARTITIONS_SUBDIR):
                # Sanity: only delete if it has balanced_*.parquet inside
                if any(month_dir.glob("balanced_*.parquet")):
                    shutil.rmtree(month_dir)

    total_elapsed = time.time() - t0
    return {
        "shuffled_shards": final_idx,
        "total_rows_in": total_in,
        "total_rows_out": total_out,
        "num_partitions": num_partitions,
        "pass1_scatter_batch_rows": scatter_batch_rows,
        "pass1_max_buffered_rows": scatter_max_buffered_rows,
        "pass2_gather_batch_rows": gather_batch_rows,
        "pass2_max_rows_in_memory": max_rows_in_memory,
        "pass1_seconds": int(pass1_elapsed),
        "pass2_seconds": int(pass2_elapsed),
        "total_seconds": int(total_elapsed),
        "deleted_inputs": delete_inputs,
    }
