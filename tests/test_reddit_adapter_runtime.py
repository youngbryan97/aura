import asyncio
import time

import pytest

from core.runtime.errors import get_degradation_tracker
from core.skills.reddit_adapter import RedditAdapterSkill, RedditInput


class Auth:
    approved = True
    reason = ""
    capability_token_id = "cap-reddit"
    executive_intent_id = "intent-reddit"
    will_receipt_id = "receipt-reddit"


@pytest.fixture(autouse=True)
def _isolate_reddit_storage(monkeypatch, tmp_path):
    from core.skills import reddit_adapter

    storage = tmp_path / "reddit"
    monkeypatch.setattr(reddit_adapter, "_STORAGE_DIR", storage)
    monkeypatch.setattr(
        reddit_adapter,
        "_STORAGE_STATE_FILE",
        storage / "browser_state.json",
    )
    monkeypatch.setattr(
        reddit_adapter,
        "_COMMENT_HISTORY_FILE",
        storage / "comment_history.json",
    )
    monkeypatch.setattr(
        reddit_adapter,
        "_CONNECTION_STATE_FILE",
        storage / "connection_state.json",
    )


def test_reddit_adapter_marks_authority_finalize_degraded(monkeypatch):
    async def scenario():
        from core.being.welfare_transaction import WelfareTransaction

        tracker = get_degradation_tracker()
        tracker.reset()
        WelfareTransaction.reset()
        skill = RedditAdapterSkill()

        class Gateway:
            async def authorize_tool_execution(self, *_args, **_kwargs):
                return Auth()

            def verify_tool_access(self, *_args, **_kwargs):
                return True

            def finalize_tool_execution(self, *_args, **_kwargs):
                self.finalized = True
                raise RuntimeError("authority ledger unavailable")

        gateway = Gateway()
        closed = []

        async def create_browser():
            return object()

        async def close_browser(browser):
            closed.append(browser)

        async def browse_posts(_browser, _params):
            return {"ok": True, "posts": [], "count": 0}

        monkeypatch.setattr(
            "core.executive.authority_gateway.get_authority_gateway",
            lambda: gateway,
        )
        monkeypatch.setattr(skill, "_create_browser", create_browser)
        monkeypatch.setattr(skill, "_safe_close", close_browser)
        monkeypatch.setattr(skill, "_handle_browse", browse_posts)

        result = await skill.execute(RedditInput(mode="browse"), {})

        assert result["ok"] is True
        assert result["authority_finalized"] is False
        assert result["authority_finalization_status"] == "degraded"
        assert result["authority_receipt_id"] == "receipt-reddit"
        assert result["welfare_transaction_id"]
        records = WelfareTransaction.recent_records(1)
        assert records
        assert records[0].action == "reddit_adapter.browse"
        assert records[0].outcome == "success"
        assert records[0].will_receipt_id == "receipt-reddit"
        assert gateway.finalized is True
        assert len(closed) == 1
        assert any(
            "authority finalization degraded" in record.action
            for record in tracker.recent(subsystem="reddit_adapter")
        )
        tracker.reset()
        WelfareTransaction.reset()

    asyncio.run(scenario())


def test_reddit_adapter_construction_does_not_create_persistence_directory(tmp_path, monkeypatch):
    from core.skills import reddit_adapter

    storage = tmp_path / "not-created-during-discovery"
    monkeypatch.setattr(reddit_adapter, "_STORAGE_DIR", storage)
    monkeypatch.setattr(
        reddit_adapter,
        "_CONNECTION_STATE_FILE",
        storage / "connection_state.json",
    )

    skill = RedditAdapterSkill()

    assert skill.get_connection_status()["state"] == "public_only"
    assert not storage.exists()


def test_reddit_adapter_failure_finalizes_authority_false(monkeypatch):
    async def scenario():
        from core.being.welfare_transaction import WelfareTransaction

        WelfareTransaction.reset()
        skill = RedditAdapterSkill()

        class Gateway:
            def __init__(self):
                self.finalized_success = []

            async def authorize_tool_execution(self, *_args, **_kwargs):
                return Auth()

            def verify_tool_access(self, *_args, **_kwargs):
                return True

            def finalize_tool_execution(self, *_args, **kwargs):
                self.finalized_success.append(kwargs.get("success"))

        gateway = Gateway()
        read_attempts = []

        async def create_browser():
            return object()

        async def close_browser(_browser):
            return None

        async def read_post(_browser, _params):
            read_attempts.append("called")
            raise RuntimeError("reddit page unavailable")

        monkeypatch.setattr(
            "core.executive.authority_gateway.get_authority_gateway",
            lambda: gateway,
        )
        monkeypatch.setattr(skill, "_create_browser", create_browser)
        monkeypatch.setattr(skill, "_safe_close", close_browser)
        monkeypatch.setattr(skill, "_handle_read_post", read_post)

        result = await skill.execute(
            RedditInput(mode="read_post", url="https://reddit.com/r/a"), {}
        )

        assert result["ok"] is False
        assert "reddit page unavailable" in result["error"]
        assert result["authority_finalized"] is True
        assert result["welfare_transaction_id"]
        records = WelfareTransaction.recent_records(1)
        assert records
        assert records[0].action == "reddit_adapter.read_post"
        assert records[0].outcome == "failure"
        assert records[0].error
        assert gateway.finalized_success == [False]
        assert read_attempts == ["called"]
        WelfareTransaction.reset()

    asyncio.run(scenario())


def test_reddit_failure_survives_navigation_during_captcha_probe(monkeypatch):
    async def scenario():
        from core.skills import reddit_adapter

        skill = RedditAdapterSkill()

        class Gateway:
            async def authorize_tool_execution(self, *_args, **_kwargs):
                return Auth()

            def verify_tool_access(self, *_args, **_kwargs):
                return True

            def finalize_tool_execution(self, *_args, **_kwargs):
                return None

        class Page:
            async def content(self):
                raise reddit_adapter.PlaywrightError(
                    "Unable to retrieve content because the page is navigating"
                )

        class Browser:
            page = Page()

        async def read_post(_browser, _params):
            raise reddit_adapter.PlaywrightError("provider navigation changed")

        async def create_browser():
            return Browser()

        async def close_browser(_browser):
            return None

        monkeypatch.setattr(
            "core.executive.authority_gateway.get_authority_gateway",
            lambda: Gateway(),
        )
        monkeypatch.setattr(skill, "_create_browser", create_browser)
        monkeypatch.setattr(skill, "_safe_close", close_browser)
        monkeypatch.setattr(skill, "_handle_read_post", read_post)

        result = await skill.execute(
            RedditInput(mode="read_post", url="https://reddit.com/r/a"), {}
        )

        assert result["ok"] is False
        assert "provider navigation changed" in result["error"]
        assert result["authority_finalized"] is True

    asyncio.run(scenario())


def test_reddit_adapter_safe_close_records_browser_teardown_failure():
    async def scenario():
        tracker = get_degradation_tracker()
        tracker.reset()
        skill = RedditAdapterSkill()

        class Browser:
            def __init__(self):
                self.is_active = True
                self.close_calls = 0

            async def close(self):
                self.close_calls += 1
                raise RuntimeError("browser close failed")

        browser = Browser()

        await skill._safe_close(browser)

        assert browser.close_calls == 1
        assert browser.is_active is False
        assert any(
            "browser inactive after close failed" in record.action
            for record in tracker.recent(subsystem="reddit_adapter")
        )
        tracker.reset()

    asyncio.run(scenario())


def test_reddit_login_reads_credentials_off_event_loop(monkeypatch):
    async def scenario():
        from core.skills import reddit_adapter

        skill = RedditAdapterSkill()
        calls = []

        class Page:
            async def content(self):
                return "<html><body>logged out</body></html>"

        class Browser:
            page = Page()

            async def browse(self, url):
                calls.append(("browse", url))
                return True

        def get_creds():
            calls.append(("creds", "called"))
            raise RuntimeError("credentials missing")

        async def fake_to_thread(fn, *args, **kwargs):
            calls.append(("to_thread", getattr(fn, "__name__", "")))
            return fn(*args, **kwargs)

        async def fake_sleep(_seconds):
            calls.append(("sleep", "skipped"))

        monkeypatch.setattr(skill, "_get_creds", get_creds)
        monkeypatch.setattr(reddit_adapter.asyncio, "to_thread", fake_to_thread)
        monkeypatch.setattr(reddit_adapter.asyncio, "sleep", fake_sleep)
        skill._allow_reauthentication = True

        result = await skill._ensure_logged_in(Browser())

        assert result is False
        assert ("to_thread", "get_creds") in calls
        assert ("creds", "called") in calls

    asyncio.run(scenario())


def test_reddit_background_session_check_never_attempts_password_login(monkeypatch):
    async def scenario():
        from core.skills import reddit_adapter

        persisted = []

        class Gateway:
            async def write_text_async(self, path, content, **_kwargs):
                persisted.append((path, content))

        class Page:
            async def content(self):
                return "<html><body>logged out</body></html>"

        class Browser:
            page = Page()

            async def browse(self, _url):
                return True

        async def no_sleep(_seconds):
            return None

        monkeypatch.setattr(reddit_adapter, "get_file_write_gateway", lambda: Gateway())
        monkeypatch.setattr(reddit_adapter.asyncio, "sleep", no_sleep)
        skill = RedditAdapterSkill()
        monkeypatch.setattr(
            skill,
            "_get_creds",
            lambda: (_ for _ in ()).throw(AssertionError("credentials must not be read")),
        )

        assert await skill._ensure_logged_in(Browser()) is False
        assert skill.get_connection_status()["state"] == "auth_required"
        assert persisted

    asyncio.run(scenario())


def test_reddit_playwright_timeout_becomes_bounded_provider_state(monkeypatch):
    async def scenario():
        from core.skills import reddit_adapter

        class Gateway:
            async def write_text_async(self, *_args, **_kwargs):
                return None

        class Page:
            async def content(self):
                raise reddit_adapter.PlaywrightTimeoutError("provider timeout")

        class Browser:
            page = Page()

            async def browse(self, _url):
                return True

        async def no_sleep(_seconds):
            return None

        monkeypatch.setattr(reddit_adapter, "get_file_write_gateway", lambda: Gateway())
        monkeypatch.setattr(reddit_adapter.asyncio, "sleep", no_sleep)
        skill = RedditAdapterSkill()

        assert await skill._ensure_logged_in(Browser()) is False
        status = skill.get_connection_status()
        assert status["state"] == "transient_failure"
        assert "TimeoutError" in status["reason"]
        assert 0.0 < status["retry_in_s"] <= 3600.0

    asyncio.run(scenario())


def test_reddit_browse_does_not_claim_success_without_extracted_posts(monkeypatch):
    async def scenario():
        from core.skills import reddit_adapter

        class Page:
            async def evaluate(self, *_args):
                return []

        class Browser:
            page = Page()

            async def browse(self, _url):
                return True

        async def no_sleep(_seconds):
            return None

        monkeypatch.setattr(reddit_adapter.asyncio, "sleep", no_sleep)
        result = await RedditAdapterSkill()._handle_browse(
            Browser(), RedditInput(mode="browse", subreddit="futurology")
        )

        assert result["ok"] is False
        assert result["completed"] is False
        assert result["status"] == "extraction_empty"
        assert result["navigated"] is True
        assert result["posts"] == []

    asyncio.run(scenario())


def test_reddit_read_does_not_claim_success_without_extracted_content(monkeypatch):
    async def scenario():
        from core.skills import reddit_adapter

        class Browser:
            async def browse(self, _url):
                return True

            async def read_content(self):
                return ""

        async def no_sleep(_seconds):
            return None

        monkeypatch.setattr(reddit_adapter.asyncio, "sleep", no_sleep)
        result = await RedditAdapterSkill()._handle_read_post(
            Browser(),
            RedditInput(mode="read_post", url="https://www.reddit.com/r/a/1"),
        )

        assert result["ok"] is False
        assert result["completed"] is False
        assert result["status"] == "extraction_empty"
        assert result["navigated"] is True
        assert result["content"] == ""

    asyncio.run(scenario())


def test_reddit_inbox_does_not_claim_success_without_extracted_content(monkeypatch):
    async def scenario():
        from core.skills import reddit_adapter

        class Browser:
            async def browse(self, _url):
                return True

            async def read_content(self):
                return ""

        async def no_sleep(_seconds):
            return None

        skill = RedditAdapterSkill()
        monkeypatch.setattr(skill, "_ensure_logged_in", lambda _browser: _true())
        monkeypatch.setattr(reddit_adapter.asyncio, "sleep", no_sleep)
        result = await skill._handle_check_inbox(
            Browser(), RedditInput(mode="check_inbox")
        )

        assert result["ok"] is False
        assert result["completed"] is False
        assert result["status"] == "extraction_empty"
        assert result["navigated"] is True

    async def _true():
        return True

    asyncio.run(scenario())


def test_reddit_filters_expired_and_malformed_cookies():
    now = time.time()

    assert RedditAdapterSkill._filter_live_cookies(
        [
            {"name": "reddit_session", "expires": now + 60},
            {"name": "expired", "expires": now - 1},
            {"name": "session_cookie", "expires": -1},
            {"name": "bad", "expires": "not-a-number"},
            "not-a-cookie",
        ]
    ) == [
        {"name": "reddit_session", "expires": now + 60},
        {"name": "session_cookie", "expires": -1},
    ]
