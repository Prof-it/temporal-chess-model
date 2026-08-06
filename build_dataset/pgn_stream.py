"""Split a stream of decompressed PGN bytes into per-game text blocks.

A Lichess PGN file is a sequence of:
    [Event "..."]
    [Site "..."]
    ...other headers...
    [TimeControl "..."]

    1. e4 e5 2. Nf3 ... 1-0


Game blocks are separated by blank lines after the move text. We split on the
boundary `\\n\\n[Event ` (with one blank line between games), accumulating
trailing bytes between chunks.
"""
from __future__ import annotations

from typing import Iterable, Iterator


def iter_game_blocks(byte_chunks: Iterable[bytes]) -> Iterator[str]:
    """Yield complete game text blocks, decoded as UTF-8 with replacement.

    Each yielded string starts with `[Event ` and ends just before the next
    `[Event ` (move list included). Blocks are stripped of leading/trailing
    blank lines but otherwise unchanged.
    """
    buf = bytearray()
    SEP = b"\n[Event "
    first_seen = False

    for chunk in byte_chunks:
        buf.extend(chunk)
        # Find as many complete game boundaries as possible in the buffer.
        while True:
            sep_idx = buf.find(SEP, 1)  # start at 1 so we don't re-match a buffer-start [Event
            if sep_idx < 0:
                break
            game_bytes = bytes(buf[: sep_idx + 1])  # include the trailing \n before next [Event
            # If we haven't seen the first [Event yet, strip prefix junk.
            if not first_seen:
                first = game_bytes.find(b"[Event ")
                if first < 0:
                    # Move on; entire prefix is non-game junk.
                    del buf[: sep_idx + 1]
                    continue
                game_bytes = game_bytes[first:]
                first_seen = True
            text = game_bytes.decode("utf-8", errors="replace").strip()
            if text:
                yield text
            # Drop the consumed bytes; keep the \n before next [Event
            del buf[: sep_idx + 1]

    # Tail: whatever is left after the last separator.
    if buf:
        first = buf.find(b"[Event ")
        if first >= 0:
            tail = bytes(buf[first:]).decode("utf-8", errors="replace").strip()
            if tail:
                yield tail
