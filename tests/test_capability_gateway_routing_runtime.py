from __future__ import annotations

import asyncio
import base64
import tempfile
from pathlib import Path

import pytest

from core.capabilities.browser_controller import BrowserController
from core.capabilities.capability_discovery import CapabilityDiscovery, CapabilityReport
from core.capabilities.clipboard_manager import ClipboardManager
from core.capabilities.document_service import DocumentService
from core.capabilities.file_broker import SandboxedFileBroker
from core.capabilities.os_settings import OSSettingsAdapter
from core.capabilities.web_asset_handler import WebAssetHandler
from core.container import ServiceContainer
from core.perception.screen_perception import ScreenPerception
from core.self.mind_state_export import MindStateExporter
from core.voice.voice_session import VoiceSessionManager


class FakeNetworkGateway:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    async def request_async(self, method: str, url: str, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return {
            "ok": True,
            "status_code": 200,
            "headers": {"Content-Type": "image/png"},
            "content": self.content,
        }


class FakeFileWriteGateway:
    def __init__(self) -> None:
        self.text_writes: list[dict[str, object]] = []
        self.byte_writes: list[dict[str, object]] = []
        self.directory_writes: list[dict[str, object]] = []

    async def ensure_directory_async(self, path, *, source="unknown") -> str:
        target = Path(path)
        await asyncio.to_thread(target.mkdir, parents=True, exist_ok=True)
        self.directory_writes.append({"path": target, "source": source})
        return str(target)

    def write_text(self, path, text, *, encoding="utf-8", source="unknown", durable=True) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding=encoding)
        self.text_writes.append(
            {"path": target, "text": text, "encoding": encoding, "source": source, "durable": durable}
        )

    def write_bytes(self, path, payload, *, source="unknown") -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes(payload))
        self.byte_writes.append(
            {"path": target, "payload": bytes(payload), "source": source}
        )

    # Async lane delegators: production code now calls *_async; fakes
    # must mirror the gateway surface or every governed write breaks.
    async def write_text_async(self, *args, **kwargs):
        return self.write_text(*args, **kwargs)
    async def write_bytes_async(self, *args, **kwargs):
        return self.write_bytes(*args, **kwargs)


class FakeProcess:
    def __init__(self, stdout: bytes = b"", returncode: int = 0) -> None:
        self._stdout = stdout
        self.returncode = returncode

    async def communicate(self, input=None):
        return self._stdout, b""

    async def wait(self):
        return self.returncode


class FakeSubprocessGateway:
    def __init__(self, stdout: bytes = b"", returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.calls: list[dict[str, object]] = []

    async def spawn_async(self, argv, **kwargs):
        self.calls.append({"argv": list(argv), **kwargs})
        return FakeProcess(stdout=self.stdout, returncode=self.returncode)


@pytest.mark.asyncio
async def test_browser_search_fetch_uses_network_gateway(monkeypatch) -> None:
    from core.capabilities import browser_controller as module

    html = b'<a href="https://example.com/a" class="result-link">Example</a>'
    gateway = FakeNetworkGateway(html)
    async def request_public_http(method, url, **kwargs):
        return await gateway.request_async(method, url, **kwargs)

    monkeypatch.setattr(module, "request_public_http", request_public_http)

    results = await BrowserController()._fetch_search_results("climate news", count=1)

    assert results == [{"url": "https://example.com/a", "title": "Example"}]
    assert gateway.calls[0]["source"] == "browser_controller.search.duckduckgo"
    assert gateway.calls[0]["max_response_bytes"] == 2 * 1024 * 1024


@pytest.mark.asyncio
async def test_web_asset_download_uses_network_and_file_gateways(monkeypatch, tmp_path) -> None:
    from core.capabilities import web_asset_handler as module

    # A genuine 1x1 PNG, not a magic-byte stub. The old fixture was
    # b"\x89PNG\r\n\x1a\n" + 256 zero bytes, which is not a decodable image
    # — it only passed while download_image validated nothing but the
    # prefix. This test is about gateway ROUTING, so it needs a payload
    # that survives admission for the reason a real download would.
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAP0lEQVR4"
        "nO2WOQoAQAgDRxC2kPz/uz5isRtIL0iOKeCROzUTOFR7AF8UXYRBi1WB"
        "bToODk4mUkUEL0TH/LhgAbhlNp+Pd29NAAAAAElFTkSuQmCC"
    )
    network = FakeNetworkGateway(png)
    writer = FakeFileWriteGateway()
    monkeypatch.setattr(module, "get_network_gateway", lambda: network)
    monkeypatch.setattr(module, "get_file_write_gateway", lambda: writer)

    path = await WebAssetHandler().download_image(
        "https://example.com/image.png",
        save_dir=str(tmp_path),
    )

    assert path.endswith(".png")
    assert network.calls[0]["read_only"] is True
    assert network.calls[0]["source"] == "web_asset_handler.download_image"
    assert writer.directory_writes == [
        {"path": tmp_path, "source": "web_asset_handler.download_image"}
    ]
    assert writer.byte_writes[0]["source"] == "web_asset_handler.download_image"
    assert writer.byte_writes[0]["payload"] == png


@pytest.mark.asyncio
async def test_file_broker_write_uses_file_gateway(monkeypatch) -> None:
    from core.capabilities import file_broker as module

    writer = FakeFileWriteGateway()
    monkeypatch.setattr(module, "get_file_write_gateway", lambda: writer)
    broker = SandboxedFileBroker()
    await broker.start()
    target = Path(tempfile.gettempdir()) / "aura_gateway_test" / "note.txt"

    try:
        result = await broker.write_file(str(target), "hello")
    finally:
        ServiceContainer.clear()

    assert result["success"] is True
    assert writer.text_writes[0]["source"] == "file_broker.write_file"
    assert writer.text_writes[0]["text"] == "hello"


@pytest.mark.asyncio
async def test_document_service_create_text_uses_file_gateway(monkeypatch, tmp_path) -> None:
    from core.capabilities import document_service as module

    writer = FakeFileWriteGateway()
    monkeypatch.setattr(module, "get_file_write_gateway", lambda: writer)

    ok = await DocumentService().create_text(str(tmp_path / "doc.txt"), "body")

    assert ok is True
    assert writer.text_writes[0]["source"] == "document_service.create_text"
    assert writer.text_writes[0]["text"] == "body"


@pytest.mark.asyncio
async def test_os_settings_read_probe_uses_subprocess_gateway(monkeypatch) -> None:
    from core.capabilities import os_settings as module

    gateway = FakeSubprocessGateway(stdout=b"42\n")
    monkeypatch.setattr(module, "get_subprocess_gateway", lambda: gateway)

    volume = await OSSettingsAdapter().get_volume()

    assert volume == 42
    assert gateway.calls[0]["read_only"] is True
    assert gateway.calls[0]["source"] == "os_settings.get_volume"


@pytest.mark.asyncio
async def test_screen_perception_active_window_uses_subprocess_gateway(monkeypatch) -> None:
    from core.perception import screen_perception as module

    gateway = FakeSubprocessGateway(stdout=b"Google Chrome|News\n")
    monkeypatch.setattr(module, "get_subprocess_gateway", lambda: gateway)

    result = await ScreenPerception().get_active_window()

    assert result == {"app": "Google Chrome", "title": "News", "bounds": ""}
    assert gateway.calls[0]["read_only"] is True
    assert gateway.calls[0]["source"] == "screen_perception.active_window"


@pytest.mark.asyncio
async def test_screen_perception_registers_unreaped_timeout_child(monkeypatch) -> None:
    from core import reaper
    from core.perception import screen_perception as module

    class StuckProcess:
        pid = 4242
        returncode = None

        async def communicate(self):
            await asyncio.sleep(60.0)

        def kill(self):
            return None

        async def wait(self):
            await asyncio.sleep(60.0)

    class Gateway:
        async def spawn_async(self, *_args, **_kwargs):
            return StuckProcess()

    registered = []
    monkeypatch.setattr(module, "get_subprocess_gateway", lambda: Gateway())
    monkeypatch.setattr(reaper, "register_reaper_pid", registered.append)

    result = await ScreenPerception()._run_osascript(
        "return 1",
        source="screen_perception.timeout_test",
        timeout_s=0.01,
    )

    assert result == ""
    assert registered == [4242]


@pytest.mark.asyncio
async def test_voice_session_say_fallback_uses_subprocess_gateway(monkeypatch) -> None:
    from core.voice import voice_session as module

    ServiceContainer.clear()
    gateway = FakeSubprocessGateway()
    monkeypatch.setattr(module, "get_subprocess_gateway", lambda: gateway)

    try:
        await VoiceSessionManager().narrate("working")
    finally:
        ServiceContainer.clear()

    assert gateway.calls[0]["argv"] == ["say", "-v", "Samantha", "working"]
    assert gateway.calls[0]["source"] == "voice_session.say_fallback"


@pytest.mark.asyncio
async def test_capability_discovery_network_probe_uses_subprocess_gateway(monkeypatch) -> None:
    from core.capabilities import capability_discovery as module

    gateway = FakeSubprocessGateway(returncode=0)
    monkeypatch.setattr(module, "get_subprocess_gateway", lambda: gateway)
    report = CapabilityReport()

    await CapabilityDiscovery()._discover_network(report)

    assert report.has_network is True
    assert gateway.calls[0]["argv"] == ["ping", "-c", "1", "-W", "2", "8.8.8.8"]
    assert gateway.calls[0]["read_only"] is True
    assert gateway.calls[0]["source"] == "capability_discovery.network_probe"


@pytest.mark.asyncio
async def test_capability_discovery_writable_probe_uses_file_gateway(monkeypatch, tmp_path) -> None:
    from core.capabilities import capability_discovery as module

    writer = FakeFileWriteGateway()
    monkeypatch.setattr(module, "get_file_write_gateway", lambda: writer)
    monkeypatch.setattr(module.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    monkeypatch.setattr(module.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    report = CapabilityReport()

    await CapabilityDiscovery()._discover_writable_dirs(report)

    assert report.writable_directories
    assert writer.text_writes
    assert {write["source"] for write in writer.text_writes} == {
        "capability_discovery.writable_dir_probe"
    }
    # Probe files are worthless after a crash; a durable fsync here wedged
    # the live event loop for 20 minutes under disk thrash.
    assert {write["durable"] for write in writer.text_writes} == {False}


@pytest.mark.asyncio
async def test_capability_discovery_start_does_not_block_on_full_scan() -> None:
    from core.container import ServiceContainer

    class SlowDiscovery(CapabilityDiscovery):
        def __init__(self) -> None:
            super().__init__()
            self.release = asyncio.Event()

        async def discover(self) -> CapabilityReport:
            await self.release.wait()
            return CapabilityReport(has_network=True)

    ServiceContainer.clear()
    discovery = SlowDiscovery()
    try:
        await asyncio.wait_for(discovery.start(), timeout=0.2)
        assert discovery._started is True
        assert discovery._scan_task is not None
        assert discovery.get_report().has_network is False

        discovery.release.set()
        await asyncio.wait_for(discovery._scan_task, timeout=0.5)
        assert discovery.get_report().has_network is True
    finally:
        ServiceContainer.clear()


@pytest.mark.asyncio
async def test_clipboard_get_uses_subprocess_gateway(monkeypatch) -> None:
    from core.capabilities import clipboard_manager as module

    gateway = FakeSubprocessGateway(stdout=b"clipboard text")
    monkeypatch.setattr(module, "get_subprocess_gateway", lambda: gateway)

    text = await ClipboardManager().get()

    assert text == "clipboard text"
    assert gateway.calls[0]["argv"] == ["pbpaste"]
    assert gateway.calls[0]["read_only"] is True
    assert gateway.calls[0]["source"] == "clipboard_manager.get"


@pytest.mark.asyncio
async def test_clipboard_set_uses_subprocess_gateway(monkeypatch) -> None:
    from core.capabilities import clipboard_manager as module

    gateway = FakeSubprocessGateway()
    monkeypatch.setattr(module, "get_subprocess_gateway", lambda: gateway)

    ok = await ClipboardManager().set("new text")

    assert ok is True
    assert gateway.calls[0]["argv"] == ["pbpaste"]
    assert gateway.calls[0]["read_only"] is True
    assert gateway.calls[1]["argv"] == ["pbcopy"]
    assert gateway.calls[1]["source"] == "clipboard_manager.set"


@pytest.mark.asyncio
async def test_mind_state_export_uses_file_gateway(monkeypatch, tmp_path) -> None:
    from core.self import mind_state_export as module

    writer = FakeFileWriteGateway()
    monkeypatch.setattr(module, "get_file_write_gateway", lambda: writer)
    target = tmp_path / "aura.aura-mind"

    result = await MindStateExporter().export_mind(str(target))

    assert result["success"] is True
    assert target.exists()
    assert writer.byte_writes[0]["source"] == "mind_state_export.export_mind"
    assert writer.byte_writes[0]["path"] == target
