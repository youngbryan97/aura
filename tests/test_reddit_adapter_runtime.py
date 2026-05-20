import asyncio

from core.runtime.errors import get_degradation_tracker
from core.skills.reddit_adapter import RedditAdapterSkill, RedditInput


class Auth:
    approved = True
    reason = ""
    capability_token_id = "cap-reddit"
    executive_intent_id = "intent-reddit"
    will_receipt_id = "receipt-reddit"


def test_reddit_adapter_marks_authority_finalize_degraded(monkeypatch):
    async def scenario():
        tracker = get_degradation_tracker()
        tracker.reset()
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
        assert gateway.finalized is True
        assert len(closed) == 1
        assert any(
            "authority finalization degraded" in record.action
            for record in tracker.recent(subsystem="reddit_adapter")
        )
        tracker.reset()

    asyncio.run(scenario())


def test_reddit_adapter_failure_finalizes_authority_false(monkeypatch):
    async def scenario():
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
        assert gateway.finalized_success == [False]
        assert read_attempts == ["called"]

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
