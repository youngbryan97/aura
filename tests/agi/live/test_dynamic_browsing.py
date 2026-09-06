import http.server
import pathlib
import socket
import threading
from pathlib import Path

import pytest

# The fixture is an http server on 127.0.0.1, which the browser's SSRF
# policy refuses by default and should. The opt-in is an environment
# flag the process owner sets, it covers LOOPBACK only — not the rest
# of the local network — and every use is recorded (CP126 8bf8d32e).
@pytest.fixture(autouse=True)
def _allow_loopback_fixture(monkeypatch):
    monkeypatch.setenv("AURA_BROWSER_ALLOW_LOOPBACK", "1")


def _playwright_browser_installed() -> bool:
    """Whether a browser binary Playwright can launch is actually here."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as engine:
            return bool(pathlib.Path(engine.chromium.executable_path).exists())
    except Exception:  # noqa: BLE001 - any launch-path failure means "not installed"
        return False


#: This drives a real browser. Without `playwright install` there is no
#: binary to launch, and the test failed with an executable-not-found error
#: that reads as a capability regression. It is a missing dependency, and
#: the skip says so — the same treatment this suite gives a model that is
#: not on the host.
pytestmark = pytest.mark.skipif(
    not _playwright_browser_installed(),
    reason="playwright browser binary is not installed; run `playwright install chromium`",
)

from tools.agi.run_dynamic_browsing_task import run_browsing_task


class LocalHTTPServer:
    """A simple threaded local HTTP server to host dynamic test fixtures."""

    def __init__(self, port: int, root_dir: Path):
        self.port = port
        self.root_dir = root_dir
        self.server = None
        self.thread = None

    def start(self):
        root_dir_str = str(self.root_dir)
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=root_dir_str, **kwargs)

            def log_message(self, format, *args):
                return None

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join(timeout=2.0)


def get_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def local_web_server(tmp_path):
    # Set up index.html and doc.html in a temp directory
    index_content = """
    <html>
        <head><title>Aura Home</title></head>
        <body>
            <h1>Welcome to Aura Main Gate</h1>
            <p>Here is the portal for research.</p>
            <a id="docs-link" href="/doc.html">Aura Docs Portal</a>
        </body>
    </html>
    """
    doc_content = """
    <html>
        <head><title>Aura Documentation</title></head>
        <body>
            <h1>Aura Live Architecture</h1>
            <p>Authentication credentials verification successfully completed.</p>
            <p>Verification Key: AURA-LIVE-AGI-9921</p>
        </body>
    </html>
    """
    (tmp_path / "index.html").write_text(index_content)
    (tmp_path / "doc.html").write_text(doc_content)

    port = get_free_port()
    server = LocalHTTPServer(port, tmp_path)
    server.start()
    
    yield f"http://127.0.0.1:{port}"
    
    server.stop()


@pytest.mark.asyncio
async def test_live_phantom_browser_dynamic_navigation(local_web_server):
    # Run the dynamic browsing task using PhantomBrowser
    res = await run_browsing_task(
        start_url=local_web_server,
        target_link_text="Aura Docs Portal",
        expected_content_keywords=["AURA-LIVE-AGI-9921", "Authentication credentials"],
    )

    assert res["ok"] is True
    assert res["verification"]["AURA-LIVE-AGI-9921"] is True
    assert res["verification"]["Authentication credentials"] is True
    assert "Aura Documentation" in res["content_snippet"]


@pytest.mark.asyncio
async def test_a_refused_read_is_not_reported_as_a_page_without_the_words(
    local_web_server,
):
    """The failure this task actually had, kept from coming back.

    Every browser call in the task named a principal except the read, and an
    unnamed principal is refused. A refused read comes back as an empty
    string — the same value a blank page gives — so the task reported that
    its keywords were missing from a page it had never been allowed to read.
    """
    from core.capabilities.phantom_browser import PhantomBrowser

    browser = PhantomBrowser(visible=False)
    try:
        await browser.browse(local_web_server, principal="test_reader")
        anonymous = await browser.read_content()
        assert anonymous == "", "an unnamed read is refused"
        verdict = browser.last_verdict
        assert verdict.get("allowed") is False
        assert "principal" in str(verdict.get("reason", ""))
        # And the same read, named, is not empty — so the empty one above was
        # the refusal and not the page.
        named = await browser.read_content(principal="test_reader")
        assert named.strip(), "the page has content when the read is allowed"
    finally:
        await browser.close()
