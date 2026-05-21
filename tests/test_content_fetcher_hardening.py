import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import core.autonomy.content_fetcher as content_fetcher
from core.autonomy.content_fetcher import ContentFetcher, FetchedContent
from core.runtime.errors import get_degradation_tracker


def test_content_fetcher_uses_instance_cache_index(tmp_path):
    cache_dir = tmp_path / "cache"
    fetcher = ContentFetcher(cache_dir=cache_dir)
    key = fetcher._cache_key("web_html", "https://example.test", {})

    fetcher._cache_put(
        key,
        FetchedContent(
            method="web_html",
            priority_level=1,
            target="https://example.test",
            success=True,
            text="bounded content",
            sources=["https://example.test"],
            bytes_fetched=15,
        ),
    )

    assert (cache_dir / "index.json").exists()
    reloaded = ContentFetcher(cache_dir=cache_dir)
    cached = reloaded._get_cached(key)
    assert cached is not None
    assert cached["text"] == "bounded content"


@pytest.mark.asyncio
async def test_execute_attempt_records_method_failure(tmp_path, monkeypatch):
    get_degradation_tracker().reset()
    fetcher = ContentFetcher(cache_dir=tmp_path / "cache")

    async def broken_web_html(*_args, **_kwargs):
        broken_web_html.called = True
        raise RuntimeError("fetch path exploded")

    monkeypatch.setattr(fetcher, "_web_html", broken_web_html)
    attempt = SimpleNamespace(
        method="web_html",
        target="https://example.test",
        priority_level=2,
        args={},
    )

    result = await fetcher._execute_attempt(attempt)

    assert result.success is False
    assert "fetch path exploded" in (result.error or "")
    assert getattr(broken_web_html, "called", False) is True
    assert any(
        "method execution failed" in record.action
        for record in get_degradation_tracker().recent(subsystem="content_fetcher_attempt")
    )


@pytest.mark.asyncio
async def test_ytdlp_audio_transcribe_uses_wired_transcriber(tmp_path, monkeypatch):
    class Transcriber:
        def __init__(self):
            self.paths: list[str] = []

        def transcribe_file(self, audio_path: str) -> str:
            self.paths.append(audio_path)
            return "transcribed audio"

    async def fake_run_subprocess(cmd: list[str], timeout_seconds: int):
        output_template = Path(cmd[cmd.index("-o") + 1])
        audio_path = Path(str(output_template).replace("%(id)s", "clip").replace("%(ext)s", "mp3"))
        await asyncio.to_thread(audio_path.write_bytes, b"fake audio")
        return True, "", ""

    monkeypatch.setattr(content_fetcher.shutil, "which", lambda _name: "/usr/bin/yt-dlp")
    monkeypatch.setattr(content_fetcher, "_run_subprocess", fake_run_subprocess)

    transcriber = Transcriber()
    fetcher = ContentFetcher(cache_dir=tmp_path / "cache", whisper_transcriber=transcriber)
    result = await fetcher._ytdlp_audio_transcribe(
        "https://video.example.test/watch?v=1",
        {},
        priority=1,
    )

    assert result.success is True
    assert result.transcript == "transcribed audio"
    assert result.bytes_fetched == len(b"fake audio")
    assert transcriber.paths and transcriber.paths[0].endswith(".mp3")
