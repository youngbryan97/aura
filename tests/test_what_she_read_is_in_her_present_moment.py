"""What she read on the web in the last day is a line of her present moment, as the clock is.

Bryan, 17 November 2026: "aura shouldnt only talk about herself. there's a whole world out there to be discussed and
thought about". She had read Reuters' world page at 01:54 and "significant world events today" at 07:49, on her own,
and asked what was on her mind she had only her own state in front of her.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from core.brain import present_moment

pytestmark = pytest.mark.unit


def _store(tmp_path, monkeypatch, rows):
    folder = tmp_path / "search"
    folder.mkdir()
    (folder / "web_artifacts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    import core.config as config_module

    monkeypatch.setattr(config_module, "config", SimpleNamespace(paths=SimpleNamespace(data_dir=tmp_path)))
    present_moment._READ_TAIL.clear()


def _read(title, url, at, query="q"):
    return {"query": query, "created_at": at, "citations": [{"title": title, "url": url}]}


def test_the_last_days_reading_newest_first_each_page_once(tmp_path, monkeypatch):
    now = time.time()
    _store(tmp_path, monkeypatch, [
        _read("A story from two days ago", "https://old.example/x", now - 2 * 86400),
        _read("Global Timeline: World Events Today", "https://www.the-world-now.com/t", now - 3 * 3600),
        _read("Task Completion as an Internal Judgment", "https://dmnlabs.org/r", now - 2 * 3600),
        _read("Task Completion as an Internal Judgment", "https://dmnlabs.org/r", now - 3600),
        _read("World News | Reuters", "https://www.reuters.com/world/", now - 600),
    ])
    line = present_moment._read_lately_line(now)
    assert line.startswith("- Read on the web in the last day, newest first: World News | Reuters (www.reuters.com,")
    assert line.count("Task Completion") == 1
    assert "Global Timeline" in line and "two days ago" not in line


def test_at_most_four_pages(tmp_path, monkeypatch):
    now = time.time()
    _store(tmp_path, monkeypatch, [_read(f"Page {n}", f"https://site{n}.example/", now - n * 60) for n in range(1, 8)])
    line = present_moment._read_lately_line(now)
    assert line.count("Page ") == 4 and "Page 1 " in line and "Page 5" not in line


def test_no_reading_no_line(tmp_path, monkeypatch):
    import core.config as config_module

    monkeypatch.setattr(config_module, "config", SimpleNamespace(paths=SimpleNamespace(data_dir=tmp_path)))
    present_moment._READ_TAIL.clear()
    assert present_moment._read_lately_line(time.time()) == ""


def test_the_present_moment_carries_it(tmp_path, monkeypatch):
    now = time.time()
    _store(tmp_path, monkeypatch, [_read("World News | Reuters", "https://www.reuters.com/world/", now - 600)])
    block = present_moment.present_moment_block()
    assert "Read on the web in the last day" in block and "Reuters" in block
