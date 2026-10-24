"""X (Twitter) posts read from a prefetched cache file.

The pipeline makes no X API call. Posts are pulled outside Python (Grok Bot's
built-in X search, e.g. ``$NVDA lang:en -is:retweet``, 25 results per call) and
saved to ``<X_CACHE_DIR>/<TICKER>_<D>.json``, where D is the run's trade date
and ``X_CACHE_DIR`` defaults to ``/workspace/x_cache``.

X data for trade date D means posts from the US market close (16:00
America/New_York, DST-aware) on ``start_date`` up to the close on D. Nothing
after the close is used, so an evening pull for D carries no lookahead. File
format::

    {
      "pulled_at": "2026-10-24T02:36:00Z",     # when the search ran (UTC)
      "trade_date": "2026-10-23",
      "window_start": "2026-10-16T20:00:00Z",  # close(D) - 7 days
      "window_end": "2026-10-23T20:00:00Z",    # close(D)
      "query": "$NVDA lang:en -is:retweet",
      "posts": [
        {"handle": "someone", "created_at": "2026-10-24T02:05:19.000Z",
         "text": "...", "url": "https://x.com/someone/status/1", "likes": 0}
      ]
    }

Only ``posts`` is required, and a bare list of posts is also accepted. Posts
are trimmed to the close-to-close window; a missing or unreadable file yields an ``X data missing: <reason>``
placeholder. Nothing here invents posts: only file contents are returned.
"""

from __future__ import annotations

import contextlib
import html
import json
import logging
import os
from datetime import UTC, datetime, time
from pathlib import Path

from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = "/workspace/x_cache"
_NEW_YORK = ZoneInfo("America/New_York")
_CLOSE = time(16, 0)


def market_close_utc(day: str) -> datetime:
    """16:00 America/New_York on ``day`` (yyyy-mm-dd) as a UTC datetime, DST-aware."""
    d = datetime.strptime(day, "%Y-%m-%d").date()
    return datetime.combine(d, _CLOSE, tzinfo=_NEW_YORK).astimezone(UTC)


def _cache_path(ticker: str, end_date: str) -> Path:
    base = Path(os.environ.get("X_CACHE_DIR", DEFAULT_CACHE_DIR))
    return base / f"{ticker.strip().upper()}_{end_date}.json"


def _created_at(post) -> datetime | None:
    raw = post.get("created_at")
    if not raw:
        return None
    with contextlib.suppress(ValueError, TypeError):
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return None


def _in_window(created: datetime | None, start: datetime, end: datetime) -> bool:
    """Undated posts are dropped: nothing proves they precede the close."""
    if created is None:
        return False
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    return start < created <= end


def fetch_x_posts(
    ticker: str,
    start_date: str,
    end_date: str,
    limit: int = 20,
    screen=None,
) -> str:
    """Return prefetched X posts for ``ticker`` as a prompt block.

    The file is keyed by trade date ``end_date``; posts are kept when published
    in (close(start_date), close(end_date)], closes at 16:00 New York time.

    ``screen`` works as in the StockTwits and Reddit fetchers: it takes the post
    texts and returns a keep flag per post and a note line heading the block.
    """
    path = _cache_path(ticker, end_date)
    if not path.is_file():
        return f"X data missing: no prefetch file ({path.name})"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("X cache read failed for %s: %s", path, exc)
        return f"X data missing: unreadable prefetch file ({type(exc).__name__})"

    posts = data.get("posts") if isinstance(data, dict) else data
    if not isinstance(posts, list):
        return "X data missing: prefetch file has no posts list"
    pulled = f" (pulled {data['pulled_at']})" if isinstance(data, dict) and data.get("pulled_at") else ""

    start_dt = market_close_utc(start_date)
    end_dt = market_close_utc(end_date)
    window = f"{start_dt:%Y-%m-%dT%H:%MZ}..{end_dt:%Y-%m-%dT%H:%MZ}"
    posts = [
        p for p in posts
        if isinstance(p, dict) and p.get("text") and _in_window(_created_at(p), start_dt, end_dt)
    ]
    if not posts:
        return f"<no X posts for ${ticker.upper()} within {window}{pulled}>"

    note = ""
    if screen:
        keep, note = screen([p["text"] for p in posts])
        screened = len(posts)
        posts = [p for p, kept in zip(posts, keep, strict=True) if kept]
        if not posts:
            return f"{note}\n\n<none of the {screened} X posts is about ${ticker.upper()}>"

    lines = []
    for p in posts[:limit]:
        text = " ".join(html.unescape(str(p["text"])).split())
        if len(text) > 280:
            text = text[:280] + "…"
        likes = p.get("likes")
        meta = f"{p.get('created_at', '')} · @{p.get('handle', '?')}"
        if likes is not None:
            meta += f" · {likes} likes"
        url = f" {p['url']}" if p.get("url") else ""
        lines.append(f"[{meta}] {text}{url}")
    header = f"{len(lines)} of {len(posts)} X posts, {window}{pulled}"
    return (f"{note}\n\n" if note else "") + header + "\n\n" + "\n".join(lines)
