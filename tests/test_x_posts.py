"""X posts read from the prefetch cache: the close-to-close window (16:00
New York, DST-aware), screening, and the missing-data placeholder. The cache
directory is a tmp path, so no real file or network is touched."""

from __future__ import annotations

import json

import pytest

from tradingagents.dataflows.vendors import x_posts


def _write(tmp_path, monkeypatch, payload, name="NVDA_2026-10-23.json"):
    monkeypatch.setenv("X_CACHE_DIR", str(tmp_path))
    (tmp_path / name).write_text(json.dumps(payload) if not isinstance(payload, str) else payload)


POSTS = [
    {"handle": "a", "created_at": "2026-10-23T19:59:00.000Z", "text": "$NVDA ripping into the close", "url": "https://x.com/a/status/1", "likes": 3},
    {"handle": "b", "created_at": "2026-10-23T20:01:00.000Z", "text": "after the close", "url": "https://x.com/b/status/2", "likes": 0},
    {"handle": "c", "created_at": "2026-10-16T19:00:00.000Z", "text": "before the window", "url": "https://x.com/c/status/3", "likes": 0},
    {"handle": "d", "text": "undated", "url": "https://x.com/d/status/4"},
]


@pytest.mark.unit
class TestXPosts:
    @pytest.mark.parametrize(
        ("day", "hour"),
        [("2026-10-23", 20), ("2026-12-01", 21), ("2026-03-06", 21), ("2026-03-09", 20)],
    )
    def test_market_close_is_dst_aware(self, day, hour):
        close = x_posts.market_close_utc(day)
        assert (close.hour, close.minute, close.utcoffset().total_seconds()) == (hour, 0, 0)

    def test_missing_file_placeholder(self, tmp_path, monkeypatch):
        monkeypatch.setenv("X_CACHE_DIR", str(tmp_path))
        out = x_posts.fetch_x_posts("NVDA", "2026-10-16", "2026-10-23")
        assert out == "X data missing: no prefetch file (NVDA_2026-10-23.json)"

    def test_unreadable_file_placeholder(self, tmp_path, monkeypatch):
        _write(tmp_path, monkeypatch, "{not json")
        out = x_posts.fetch_x_posts("NVDA", "2026-10-16", "2026-10-23")
        assert out.startswith("X data missing: unreadable")

    def test_keeps_only_posts_up_to_the_close(self, tmp_path, monkeypatch):
        _write(tmp_path, monkeypatch, {"pulled_at": "2026-10-24T02:36:00Z", "posts": POSTS})
        out = x_posts.fetch_x_posts("nvda", "2026-10-16", "2026-10-23")
        assert "@a · 3 likes] $NVDA ripping into the close https://x.com/a/status/1" in out
        assert out.startswith("1 of 1 X posts, 2026-10-16T20:00Z..2026-10-23T20:00Z (pulled 2026-10-24T02:36:00Z)")
        for gone in ("after the close", "before the window", "undated"):
            assert gone not in out

    def test_bare_list_and_screen(self, tmp_path, monkeypatch):
        _write(tmp_path, monkeypatch, POSTS)
        out = x_posts.fetch_x_posts(
            "NVDA", "2026-10-16", "2026-10-23", screen=lambda texts: ([False] * len(texts), "Screened by Jev")
        )
        assert out.startswith("Screened by Jev") and "none of the 1 X posts" in out
