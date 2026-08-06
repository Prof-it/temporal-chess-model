"""HTTP-stream a Lichess monthly .pgn.zst and yield decompressed bytes.

Never persists the .zst or .pgn to disk. On mid-stream failure (network drop,
zstd error), redownloads the entire month — there is no byte-aligned resume
point in a Zstandard frame stream relative to PGN game boundaries.
"""
from __future__ import annotations

import time
from typing import Iterator, Optional

import requests
import zstandard

from .config import (
    HTTP_BACKOFF_BASE,
    HTTP_RETRIES,
    HTTP_TIMEOUT_CONNECT,
    HTTP_TIMEOUT_READ,
    LICHESS_URL_TEMPLATE,
)


class MonthUnavailable(Exception):
    """Lichess returned 404 for this month (likely not yet published)."""


def url_for(month: str) -> str:
    return LICHESS_URL_TEMPLATE.format(month=month)


def head_size_bytes(month: str) -> Optional[int]:
    """Return Content-Length of the .zst, or None if HEAD doesn't return it."""
    url = url_for(month)
    try:
        resp = requests.head(url, timeout=(HTTP_TIMEOUT_CONNECT, 30), allow_redirects=True)
        if resp.status_code == 404:
            raise MonthUnavailable(month)
        resp.raise_for_status()
        cl = resp.headers.get("Content-Length")
        return int(cl) if cl else None
    except MonthUnavailable:
        raise
    except requests.RequestException:
        return None


def stream_pgn_bytes(
    month: str,
    chunk_size: int = 1 << 20,
    progress_cb=None,
) -> Iterator[bytes]:
    """Yield decompressed PGN bytes for the given month.

    progress_cb: optional callable(compressed_bytes_consumed: int) called per chunk.
    Raises MonthUnavailable on 404. Raises after HTTP_RETRIES failed attempts.
    """
    url = url_for(month)
    last_exc: Optional[Exception] = None
    for attempt in range(HTTP_RETRIES):
        try:
            session = requests.Session()
            with session.get(
                url, stream=True, timeout=(HTTP_TIMEOUT_CONNECT, HTTP_TIMEOUT_READ)
            ) as resp:
                if resp.status_code == 404:
                    raise MonthUnavailable(month)
                resp.raise_for_status()
                dctx = zstandard.ZstdDecompressor()
                # Wrap raw socket-backed file. resp.raw doesn't honor decode_content
                # by default, which is what we want here (we do zstd ourselves).
                resp.raw.decode_content = False
                with dctx.stream_reader(resp.raw, read_size=chunk_size) as decompressed:
                    while True:
                        chunk = decompressed.read(chunk_size)
                        if not chunk:
                            return
                        if progress_cb is not None:
                            progress_cb(len(chunk))
                        yield chunk
                # Successful end of stream
                return
        except MonthUnavailable:
            raise
        except (requests.RequestException, zstandard.ZstdError, ConnectionError) as e:
            last_exc = e
            if attempt == HTTP_RETRIES - 1:
                raise
            wait = HTTP_BACKOFF_BASE * (2 ** attempt)
            print(
                f"[http_zstd] attempt {attempt+1}/{HTTP_RETRIES} for {month} failed: "
                f"{type(e).__name__}: {e} — retrying in {wait:.0f}s",
                flush=True,
            )
            time.sleep(wait)
    # Unreachable, but keeps type-checkers happy.
    if last_exc is not None:
        raise last_exc
