import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.runtime.content_integrity import paragraph_sha256s, text_sha256
from core.runtime.errors import get_degradation_tracker
from core.skills.computer_use import ComputerUseSkill


@pytest.fixture
def screen_capture_allowed(monkeypatch):
    """Pin the privacy gate so these tests measure result shaping, not the host.

    `inspect_screen` and `read_screen_text` both ask
    `evaluate_screen_capture_admission_async` first, and that answer depends on
    what is actually in the foreground of the machine running the suite. Left
    unpinned, these eight tests went green or red on whichever window happened
    to be frontmost — and, run together with other files, on whether some
    earlier test had already patched the gate. A test that measures the host is
    not measuring what its name says.

    The refusal path has its own test below, so pinning here removes an
    order dependence rather than coverage.
    """
    from core.security import screen_capture_policy as policy

    async def _allow():
        return policy.ScreenCaptureAdmission(allowed=True, context_known=True)

    monkeypatch.setattr(policy, "evaluate_screen_capture_admission_async", _allow)
    return _allow


@pytest.mark.asyncio
async def test_computer_use_inspect_screen_returns_structured_perception(screen_capture_allowed, monkeypatch):
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    class PerceptionDouble:
        async def capture(self, *, save_screenshot=False):
            # Reading the screen captures the screen. This asserted False
            # until 2026-08-04, pinning the defect that made the ordinary
            # read skip the screenshot and therefore OCR — so the answer
            # could only ever be the frontmost window's title. What these
            # tests exist to pin is that the read goes through structured
            # perception, which is unchanged.
            assert save_screenshot is True
            return SimpleNamespace(
                active_app="Google Chrome",
                window_title="Climate article",
                frontmost_window_bounds="0,25,1440,900",
                focused_role="AXTextArea",
                focused_name="Document body",
                focused_description="editing area",
                focused_value="",
                accessibility_text="Google Docs document body text",
                screen_text="",
                screenshot_path="",
                text_hash="abc123",
                has_modal=False,
                modal_text="",
                has_loading=False,
                timestamp=123.0,
            )

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)
    monkeypatch.setattr(
        "core.perception.screen_perception.get_screen_perception",
        lambda: PerceptionDouble(),
    )

    result = await skill.execute({"action": "inspect_screen", "target": ""}, {})

    assert result["ok"] is True
    assert result["source"] == "screen_perception"
    assert result["active_app"] == "Google Chrome"
    assert result["window_title"] == "Climate article"
    assert result["focused_role"] == "AXTextArea"
    assert result["text"] == "Google Docs document body text"
    assert result["text_hash"] == "abc123"


@pytest.mark.asyncio
async def test_computer_use_inspect_screen_falls_back_to_window_tree_on_permission_block(screen_capture_allowed, monkeypatch):
    skill = ComputerUseSkill()

    async def controlled_permission_denial(capability, *permission_names):
        return {"ok": False, "status": "denied", "error": "permission denied by test guard"}

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_denial)
    monkeypatch.setattr(
        "core.perception.screen_perception.get_screen_perception",
        lambda: SimpleNamespace(
            capture=lambda save_screenshot=True: (_ for _ in ()).throw(
                RuntimeError("screen perception unavailable")
            )
        ),
    )
    monkeypatch.setattr(skill, "_frontmost_app_name", lambda: "Notes")
    monkeypatch.setattr(
        skill,
        "_query_system_events_window_tree",
        lambda: "Process: Notes\n  Window: Aura note\n    Element [AXTextArea]: draft",
    )

    result = await skill.execute({"action": "inspect_screen", "target": ""}, {})

    assert result["ok"] is True
    assert result["status"] == "limited"
    assert result["source"] == "applescript_window_tree_fallback"
    assert result["active_app"] == "Notes"
    assert "AXTextArea" in result["text"]
    assert result["accessibility_blocked"] is True


@pytest.mark.asyncio
async def test_computer_use_inspect_screen_uses_ocr_before_tree_on_permission_block(screen_capture_allowed, monkeypatch, tmp_path):
    skill = ComputerUseSkill()

    async def controlled_permission_denial(capability, *permission_names):
        return {"ok": False, "status": "denied", "error": "permission denied by test guard"}

    class PerceptionDouble:
        async def capture(self, *, save_screenshot=False):
            assert save_screenshot is True
            return SimpleNamespace(
                active_app="Google Chrome",
                window_title="ChatGPT",
                frontmost_window_bounds="0,25,1440,900",
                focused_role="",
                focused_name="",
                focused_description="",
                focused_value="",
                accessibility_text="",
                screen_text="ChatGPT\nAsk anything\nAura can read the visible screen.",
                screenshot_path=str(tmp_path / "screen.png"),
                text_hash="ocrhash",
                has_modal=False,
                modal_text="",
                has_loading=False,
                timestamp=125.0,
            )

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_denial)
    monkeypatch.setattr(
        "core.perception.screen_perception.get_screen_perception",
        lambda: PerceptionDouble(),
    )
    monkeypatch.setattr(
        skill,
        "_query_system_events_window_tree",
        lambda: (_ for _ in ()).throw(AssertionError("tree fallback should not run")),
    )

    result = await skill.execute({"action": "inspect_screen", "target": ""}, {})

    assert result["ok"] is True
    assert result["source"] == "screen_perception_permission_fallback"
    assert result["accessibility_blocked"] is True
    assert "Aura can read the visible screen" in result["text"]


@pytest.mark.asyncio
async def test_computer_use_read_screen_text_uses_structured_perception(screen_capture_allowed, monkeypatch):
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    class PerceptionDouble:
        async def capture(self, *, save_screenshot=False):
            # Reading the screen captures the screen. This asserted False
            # until 2026-08-04, pinning the defect that made the ordinary
            # read skip the screenshot and therefore OCR — so the answer
            # could only ever be the frontmost window's title. What these
            # tests exist to pin is that the read goes through structured
            # perception, which is unchanged.
            assert save_screenshot is True
            return SimpleNamespace(
                active_app="Google Chrome",
                window_title="Google Docs - Aura Journal",
                frontmost_window_bounds="0,25,1440,900",
                focused_role="AXTextArea",
                focused_name="Document body",
                focused_description="editable text",
                focused_value="",
                accessibility_text="Aura is typing in the document body.",
                screen_text="",
                screenshot_path="",
                text_hash="screenabc",
                has_modal=False,
                modal_text="",
                has_loading=False,
                timestamp=123.0,
            )

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)
    monkeypatch.setattr(
        "core.perception.screen_perception.get_screen_perception",
        lambda: PerceptionDouble(),
    )

    result = await skill.execute({"action": "read_screen_text", "target": ""}, {})

    assert result["ok"] is True
    assert result["source"] == "screen_perception"
    assert result["active_app"] == "Google Chrome"
    assert result["window_title"] == "Google Docs - Aura Journal"
    assert result["focused_role"] == "AXTextArea"
    # The reading is present; the answer now also carries which app and
    # window it came from, because "what is on my screen" was being answered
    # with one window while several were visible.
    assert "Aura is typing in the document body." in result["text"]
    assert "Google Docs - Aura Journal" in result["text"]
    assert result["text_hash"] == "screenabc"


@pytest.mark.asyncio
async def test_computer_use_read_screen_text_can_return_limited_window_context(screen_capture_allowed, monkeypatch):
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    class PerceptionDouble:
        async def capture(self, *, save_screenshot=False):
            return SimpleNamespace(
                active_app="Notes",
                window_title="Aura note",
                frontmost_window_bounds="0,25,900,700",
                focused_role="AXTextArea",
                focused_name="body",
                focused_description="editor",
                focused_value="",
                accessibility_text="",
                screen_text="",
                screenshot_path="",
                text_hash="",
                has_modal=False,
                modal_text="",
                has_loading=False,
                timestamp=124.0,
            )

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)
    monkeypatch.setattr(
        "core.perception.screen_perception.get_screen_perception",
        lambda: PerceptionDouble(),
    )

    result = await skill.execute({"action": "read_screen_text", "target": ""}, {})

    assert result["ok"] is True
    assert result["status"] == "limited"
    assert result["source"] == "screen_perception"
    assert "Active app: Notes" in result["text"]
    assert "Focused element: AXTextArea | body | editor" in result["text"]


@pytest.mark.asyncio
async def test_computer_use_read_screen_text_fallback_on_permission_block(screen_capture_allowed, monkeypatch):
    skill = ComputerUseSkill()

    async def controlled_permission_denial(capability, *permission_names):
        return {"ok": False, "status": "denied", "error": "permission denied by test guard"}

    called_tree = False

    def controlled_window_tree():
        nonlocal called_tree
        called_tree = True
        return "Process: Finder\n  Window: Desktop\n    Element [AXButton]: Close"

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_denial)
    monkeypatch.setattr(
        "core.perception.screen_perception.get_screen_perception",
        lambda: SimpleNamespace(
            capture=lambda save_screenshot=True: (_ for _ in ()).throw(
                RuntimeError("screen perception unavailable")
            )
        ),
    )
    monkeypatch.setattr(skill, "_query_system_events_window_tree", controlled_window_tree)

    result = await skill.execute({"action": "read_screen_text", "target": ""}, {})
    assert result["ok"] is True
    assert result["source"] == "applescript_window_tree_fallback"
    assert "Finder" in result["text"]
    assert called_tree is True


@pytest.mark.asyncio
async def test_computer_use_read_screen_text_uses_ocr_before_tree_on_permission_block(screen_capture_allowed, monkeypatch, tmp_path):
    skill = ComputerUseSkill()

    async def controlled_permission_denial(capability, *permission_names):
        return {"ok": False, "status": "denied", "error": "permission denied by test guard"}

    class PerceptionDouble:
        async def capture(self, *, save_screenshot=False):
            assert save_screenshot is True
            return SimpleNamespace(
                active_app="Google Chrome",
                window_title="ChatGPT",
                frontmost_window_bounds="0,25,1440,900",
                focused_role="",
                focused_name="",
                focused_description="",
                focused_value="",
                accessibility_text="",
                screen_text="ChatGPT\nAsk anything\nAura can read visible browser replies.",
                screenshot_path=str(tmp_path / "screen.png"),
                text_hash="ocrhash",
                has_modal=False,
                modal_text="",
                has_loading=False,
                timestamp=125.0,
            )

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_denial)
    monkeypatch.setattr(
        "core.perception.screen_perception.get_screen_perception",
        lambda: PerceptionDouble(),
    )
    monkeypatch.setattr(
        skill,
        "_query_system_events_window_tree",
        lambda: (_ for _ in ()).throw(AssertionError("tree fallback should not run")),
    )

    result = await skill.execute({"action": "read_screen_text", "target": ""}, {})

    assert result["ok"] is True
    assert result["source"] == "screen_perception_permission_fallback"
    assert result["accessibility_blocked"] is True
    assert "browser replies" in result["text"]


@pytest.mark.asyncio
async def test_computer_use_inspect_browser_page_reads_dom_text_and_links(monkeypatch):
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)
    monkeypatch.setattr(skill, "_frontmost_app_name", lambda: "Google Chrome")
    monkeypatch.setattr(
        skill,
        "_active_browser_location",
        lambda browser: ("https://example.com/article", "Example Article"),
    )

    def fake_applescript(script, *, timeout=10):
        assert "execute javascript" in script
        return json.dumps(
            {
                "ok": True,
                "url": "https://example.com/article",
                "title": "Example Article",
                "text": "A robust article body Aura can inspect before deciding where to navigate.",
                "links": [{"text": "Next source", "href": "https://example.com/next"}],
                "editable_count": 1,
            }
        )

    monkeypatch.setattr(skill, "_run_applescript", fake_applescript)

    result = await skill.execute({"action": "inspect_browser_page", "target": ""}, {})

    assert result["ok"] is True
    assert result["source"] == "browser_dom"
    assert "robust article body" in result["text"]
    assert result["links"][0]["href"] == "https://example.com/next"


@pytest.mark.asyncio
async def test_computer_use_inspect_browser_page_blocks_private_source(monkeypatch):
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)
    monkeypatch.setattr(skill, "_frontmost_app_name", lambda: "Google Chrome")
    monkeypatch.setattr(
        skill,
        "_active_browser_location",
        lambda browser: ("https://chatgpt.com/", "ChatGPT"),
    )
    monkeypatch.setattr(
        skill,
        "_run_applescript",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("source should be blocked")),
    )

    result = await skill.execute(
        {
            "action": "inspect_browser_page",
            "target": json.dumps({"browser": "Google Chrome", "mode": "source"}),
        },
        {},
    )

    assert result["ok"] is False
    assert result["status"] == "private_source_blocked"


@pytest.mark.asyncio
async def test_computer_use_dismiss_popup_uses_screen_perception_effect_evidence(monkeypatch):
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    class PerceptionDouble:
        def __init__(self):
            self.calls = 0

        async def capture(self, *, save_screenshot=False):
            self.calls += 1
            return SimpleNamespace(
                active_app="Google Chrome",
                window_title="Permission Alert" if self.calls == 1 else "Article",
                frontmost_window_bounds="",
                focused_role="",
                focused_name="",
                focused_description="",
                focused_value="",
                accessibility_text="Allow access?" if self.calls == 1 else "Article body",
                screen_text="",
                screenshot_path="",
                text_hash=f"h{self.calls}",
                has_modal=self.calls == 1,
                modal_text="Allow access?" if self.calls == 1 else "",
                has_loading=False,
                timestamp=125.0 + self.calls,
            )

    perception = PerceptionDouble()
    scripts = []

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)
    monkeypatch.setattr(
        "core.perception.screen_perception.get_screen_perception",
        lambda: perception,
    )

    async def controlled_sleep(_secs):
        return None

    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)

    def fake_applescript(script, *, timeout=10):
        scripts.append(script)
        return ""

    monkeypatch.setattr(skill, "_run_applescript", fake_applescript)

    result = await skill.execute({"action": "dismiss_popup", "target": ""}, {})

    assert result["ok"] is True
    assert result["effect_verified"] is True
    assert result["modal_before"] is True
    assert result["modal_after"] is False
    assert any("key code 53" in script for script in scripts)


@pytest.mark.asyncio
async def test_computer_use_direct_execution_records_welfare_transaction(monkeypatch):
    skill = ComputerUseSkill()

    async def controlled_permission_denial(capability, *permission_names):
        return {"ok": False, "status": "denied", "error": "permission denied by test guard"}

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_denial)

    result = await skill.execute({"action": "read_menu_clock", "target": ""}, {})

    assert result["ok"] is True
    assert result["welfare_transaction_id"]
    assert result["welfare_transaction_outcome"] == "success"


@pytest.mark.asyncio
async def test_read_screen_text_does_not_retry_the_same_broken_permission(
    screen_capture_allowed, monkeypatch
):
    """An accessibility failure must NOT fall through to the AppleScript tree.

    This test used to assert the opposite, and the code changed underneath it
    for a good reason: the System Events window tree runs through the SAME
    accessibility permission that just failed, so trying it spends a
    subprocess to fail identically. Reporting unavailable immediately is the
    honest answer.
    """
    skill = ComputerUseSkill()
    called_tree = False

    async def controlled_permission_pass(capability, *permission_names):
        return None

    def controlled_window_tree():
        nonlocal called_tree
        called_tree = True
        return "Fallback Process tree"

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)
    monkeypatch.setattr(
        "core.perception.screen_perception.get_screen_perception",
        lambda: SimpleNamespace(
            capture=lambda save_screenshot=False: (_ for _ in ()).throw(
                RuntimeError("perception unavailable")
            )
        ),
    )
    monkeypatch.setattr(
        skill, "_read_screen_text_macos", lambda: "[accessibility error or ui unresponsive]"
    )
    monkeypatch.setattr(skill, "_query_system_events_window_tree", controlled_window_tree)

    result = await skill.execute({"action": "read_screen_text", "target": ""}, {})

    assert result["ok"] is False
    assert result["status"] == "unavailable"
    assert called_tree is False, "the tree needs the permission that just failed"


@pytest.mark.asyncio
async def test_read_screen_text_falls_back_when_the_failure_is_not_accessibility(
    screen_capture_allowed, monkeypatch
):
    """A non-accessibility read failure SHOULD try the window tree."""
    skill = ComputerUseSkill()
    called_tree = False

    async def controlled_permission_pass(capability, *permission_names):
        return None

    def controlled_window_tree():
        nonlocal called_tree
        called_tree = True
        return "Fallback Process tree"

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)
    monkeypatch.setattr(
        "core.perception.screen_perception.get_screen_perception",
        lambda: SimpleNamespace(
            capture=lambda save_screenshot=False: (_ for _ in ()).throw(
                RuntimeError("perception unavailable")
            )
        ),
    )
    monkeypatch.setattr(skill, "_read_screen_text_macos", lambda: "[read_screen_text failed]")
    monkeypatch.setattr(skill, "_query_system_events_window_tree", controlled_window_tree)

    result = await skill.execute({"action": "read_screen_text", "target": ""}, {})

    assert result["ok"] is True
    assert result["source"] == "applescript_window_tree_fallback"
    assert "Fallback Process tree" in result["text"]
    assert called_tree is True


async def test_computer_use_click_retry_success(monkeypatch):
    skill = ComputerUseSkill()

    class TestPyAutoGUI:
        def __init__(self):
            self.clicks = 0

        def click(self, x, y):
            self.clicks += 1

    pyautogui_double = TestPyAutoGUI()
    monkeypatch.setattr("core.skills.computer_use.get_pyautogui", lambda: (pyautogui_double, None))

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)

    state_counter = 0

    def controlled_screen_text():
        nonlocal state_counter
        state_counter += 1
        if state_counter <= 2:
            return "State A"
        return "State B"

    monkeypatch.setattr(skill, "_read_screen_text_macos", controlled_screen_text)

    # Fast forward sleep
    sleep_calls = []

    async def controlled_sleep(secs):
        sleep_calls.append(secs)

    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)

    result = await skill.execute({"action": "click", "x": 100, "y": 200}, {})
    assert result["ok"] is True
    assert result["attempts"] == 2
    assert result["verification"] == "State shifted."
    assert pyautogui_double.clicks == 2
    assert sleep_calls


@pytest.mark.asyncio
async def test_computer_use_type_pre_clicks_and_retries(monkeypatch):
    skill = ComputerUseSkill()

    class TestPyAutoGUI:
        def __init__(self):
            self.clicks = 0
            self.typed = ""

        def click(self, x, y):
            self.clicks += 1

        def typewrite(self, text, interval):
            self.typed = text

    pyautogui_double = TestPyAutoGUI()
    monkeypatch.setattr("core.skills.computer_use.get_pyautogui", lambda: (pyautogui_double, None))

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)

    def controlled_screen_text():
        return "Hello World! output"

    monkeypatch.setattr(skill, "_read_screen_text_macos", controlled_screen_text)

    # Fast forward sleep
    sleep_calls = []

    async def controlled_sleep(secs):
        sleep_calls.append(secs)

    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)

    result = await skill.execute({"action": "type", "target": "Hello World!", "x": 50, "y": 60}, {})
    assert result["ok"] is True
    assert result["attempts"] == 1
    assert result["verification"] == "Text confirmed on screen or state shifted."
    assert pyautogui_double.clicks == 1
    assert pyautogui_double.typed == "Hello World!"
    assert sleep_calls


@pytest.mark.asyncio
async def test_computer_use_click_without_effect_evidence_is_not_success(monkeypatch):
    skill = ComputerUseSkill()

    class TestPyAutoGUI:
        def click(self, x, y):
            return None

    monkeypatch.setattr("core.skills.computer_use.get_pyautogui", lambda: (TestPyAutoGUI(), None))
    monkeypatch.setattr(skill, "_read_screen_text_macos", lambda: "unchanged")

    async def controlled_permission_pass(capability, *permission_names):
        return None

    async def controlled_sleep(_secs):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)
    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)

    result = await skill.execute({"action": "click", "x": 100, "y": 200}, {})

    assert result["ok"] is False
    assert result["effect_verified"] is False
    assert result["verification"] == "No obvious state shift detected after retries."


@pytest.mark.asyncio
async def test_named_click_resolves_a_fresh_anchor_and_ignores_stale_coordinates(monkeypatch):
    from core.perception.element_inventory import ScreenElement, inventory_from_elements

    skill = ComputerUseSkill()
    send = ScreenElement(
        element_id="e-send",
        role="button",
        name="Send",
        x=100,
        y=200,
        width=80,
        height=24,
        source="accessibility",
        app="Mail",
    )
    inventories = iter(
        (
            inventory_from_elements([send], app="Mail", window="Draft"),
            inventory_from_elements([], app="Mail", window="Inbox"),
        )
    )
    monkeypatch.setattr(
        "core.perception.element_inventory.build_inventory",
        lambda _app: next(inventories),
    )
    monkeypatch.setattr(skill, "_frontmost_app_name", lambda: "Mail")
    monkeypatch.setattr(skill, "_read_screen_text_macos", lambda: "unchanged")

    clicked: list[tuple[int, int]] = []

    class TestPyAutoGUI:
        def click(self, x, y):
            clicked.append((x, y))

    monkeypatch.setattr(
        "core.skills.computer_use.get_pyautogui", lambda: (TestPyAutoGUI(), None)
    )

    async def allow_permissions(*_args, **_kwargs):
        return None

    async def no_sleep(_secs):
        return None

    monkeypatch.setattr(skill, "_require_permissions", allow_permissions)
    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    result = await skill.execute(
        {"action": "click", "target": "Send button", "x": 1, "y": 1},
        {},
    )

    assert result["ok"] is True
    assert clicked == [(140, 212)]
    assert result["target_anchor"]["element_id"] == "e-send"
    assert result["planned_coordinates"] == [1, 1]
    assert result["actual_coordinates"] == [140, 212]
    assert result["target_anchor_disappeared"] is True


@pytest.mark.asyncio
async def test_named_click_refuses_when_the_target_is_not_observed(monkeypatch):
    from core.perception.element_inventory import inventory_from_elements

    skill = ComputerUseSkill()
    monkeypatch.setattr(skill, "_frontmost_app_name", lambda: "Mail")
    monkeypatch.setattr(
        "core.perception.element_inventory.build_inventory",
        lambda _app: inventory_from_elements([], app="Mail"),
    )
    clicked: list[tuple[int, int]] = []

    class TestPyAutoGUI:
        def click(self, x, y):
            clicked.append((x, y))

    monkeypatch.setattr(
        "core.skills.computer_use.get_pyautogui", lambda: (TestPyAutoGUI(), None)
    )

    # A granting permission guard. Without one registered the skill fails
    # CLOSED at the permissions stage with status "unavailable" — correct
    # behaviour, and it means this test never reached the resolution path it
    # names. Registering the guard is what puts the test back on its subject.
    from core.container import ServiceContainer

    class _GrantingGuard:
        async def check_permission(self, *_args, **_kwargs):
            return {"granted": True, "status": "active", "guidance": ""}

        async def check_permission_direct(self, *_args, **_kwargs):
            return {"granted": True, "status": "active", "guidance": ""}

        def get_guidance(self, *_args, **_kwargs):
            return ""

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _GrantingGuard()
            if name == "permission_guard"
            else default
        ),
    )

    result = await skill.execute(
        {"action": "click", "target": "Publish", "x": 900, "y": 600},
        {},
    )

    assert result["ok"] is False
    assert result["status"] == "click_target_unresolved"
    assert clicked == []


@pytest.mark.asyncio
async def test_computer_use_type_without_effect_evidence_is_not_success(monkeypatch):
    skill = ComputerUseSkill()

    class TestPyAutoGUI:
        def typewrite(self, text, interval):
            return None

    monkeypatch.setattr("core.skills.computer_use.get_pyautogui", lambda: (TestPyAutoGUI(), None))
    monkeypatch.setattr(skill, "_read_screen_text_macos", lambda: "unchanged")

    async def controlled_permission_pass(capability, *permission_names):
        return None

    async def controlled_sleep(_secs):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)
    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)

    result = await skill.execute({"action": "type", "target": "invisible"}, {})

    assert result["ok"] is False
    assert result["effect_verified"] is False
    assert result["verification"] == "Typed but could not verify visibility."


@pytest.mark.asyncio
async def test_computer_use_run_command_intercepts(monkeypatch, tmp_path):
    skill = ComputerUseSkill()

    # Let's create a couple of files to list
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file1.txt").write_text("hello")
    (tmp_path / "file2.py").write_text("print(1)")

    # 1. Test tree command intercept
    result = await skill.execute({"action": "run_command", "target": f"tree {tmp_path}"}, {})
    assert result["ok"] is True
    assert "subdir/" in result["output"]
    assert "file2.py" in result["output"]
    assert "file1.txt" in result["output"]

    # 2. Test recursive ls command intercept
    result = await skill.execute({"action": "run_command", "target": f"ls -R {tmp_path}"}, {})
    assert result["ok"] is True
    assert "subdir/" in result["output"]
    assert "file2.py" in result["output"]

    # 3. Test find command auto-constraining depth
    run_args = None

    class FakeSubprocessGateway:
        def run(self, args, **kwargs):
            nonlocal run_args
            run_args = args
            return SimpleNamespace(returncode=0, stdout="find output", stderr="")

    monkeypatch.setattr(
        "core.skills.computer_use.get_subprocess_gateway",
        lambda: FakeSubprocessGateway(),
    )
    async def confirmed_frontmost(_expected):
        return True, "Google Chrome"

    monkeypatch.setattr(
        skill,
        "_wait_for_frontmost_app",
        confirmed_frontmost,
    )

    result = await skill.execute({"action": "run_command", "target": "find . -name '*.py'"}, {})
    assert result["ok"] is True
    assert "-maxdepth" in run_args
    assert "4" in run_args


@pytest.mark.asyncio
async def test_computer_use_run_command_nonzero_exit_is_failure(monkeypatch):
    skill = ComputerUseSkill()

    class FakeSubprocessGateway:
        def run(self, args, **kwargs):
            return SimpleNamespace(returncode=7, stdout="", stderr="fatal: not a repository")

    monkeypatch.setattr(
        "core.skills.computer_use.get_subprocess_gateway",
        lambda: FakeSubprocessGateway(),
    )

    result = await skill.execute({"action": "run_command", "target": "git status"}, {})

    assert result["ok"] is False
    assert result["exit_code"] == 7
    assert result["error"] == "fatal: not a repository"


@pytest.mark.asyncio
async def test_computer_use_missing_permission_guard_fails_closed(monkeypatch):
    from core.container import ServiceContainer

    tracker = get_degradation_tracker()
    tracker.reset()
    skill = ComputerUseSkill()
    monkeypatch.setattr(ServiceContainer, "get", lambda *_args, **_kwargs: None)

    result = await skill._require_permissions("desktop control", "ACCESSIBILITY")

    assert result["ok"] is False
    assert result["permission"] == "guard"
    assert any(
        "permission guard was not registered" in record.action
        for record in tracker.recent(subsystem="computer_use")
    )
    tracker.reset()


@pytest.mark.asyncio
async def test_computer_use_direct_permission_denial_overrides_cached_assumption(monkeypatch):
    """Visible desktop actions must trust the direct macOS probe, not an
    environment/cached permission assertion."""
    from core.container import ServiceContainer

    direct_calls = []
    cached_calls = []
    subprocess_calls = []

    class EnvAssumingGuard:
        async def check_permission(self, ptype, force=False):
            cached_calls.append((ptype.name, force))
            return {"granted": True, "status": "active", "guidance": "env asserted"}

        async def check_permission_direct(self, ptype):
            direct_calls.append(ptype.name)
            return {
                "granted": False,
                "status": "denied",
                "guidance": "Enable Accessibility in System Settings.",
                "detail": "direct macOS probe denied",
            }

        def get_guidance(self, *_args, **_kwargs):
            return "Enable Accessibility in System Settings."

    class SubprocessShouldNotRun:
        def run(self, *args, **kwargs):
            subprocess_calls.append((args, kwargs))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: EnvAssumingGuard()
            if name == "permission_guard"
            else default
        ),
    )
    monkeypatch.setattr(
        "core.skills.computer_use.get_subprocess_gateway",
        lambda: SubprocessShouldNotRun(),
    )

    result = await ComputerUseSkill().execute({"action": "open_app", "target": "Notes"}, {})

    assert result["ok"] is False
    assert result["permission"] == "accessibility"
    assert result["permission_source"] == "direct"
    assert "Accessibility permission is required" in result["error"]
    assert direct_calls == ["ACCESSIBILITY"]
    assert cached_calls == []
    assert subprocess_calls == []


@pytest.mark.asyncio
async def test_computer_use_direct_permission_timeout_blocks_visible_dispatch(monkeypatch):
    from core.container import ServiceContainer

    dispatched = []

    class SlowDirectGuard:
        async def check_permission(self, *_args, **_kwargs):
            return {"granted": True, "status": "active", "guidance": "env asserted"}

        async def check_permission_direct(self, *_args, **_kwargs):
            await asyncio.sleep(0.1)
            return {"granted": True, "status": "active", "guidance": ""}

        def get_guidance(self, *_args, **_kwargs):
            return "permission guidance"

    class SubprocessShouldNotRun:
        def run(self, *args, **kwargs):
            dispatched.append((args, kwargs))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: SlowDirectGuard()
            if name == "permission_guard"
            else default
        ),
    )
    monkeypatch.setattr(
        "core.skills.computer_use.get_subprocess_gateway",
        lambda: SubprocessShouldNotRun(),
    )

    skill = ComputerUseSkill()
    skill.PERMISSION_CHECK_TIMEOUT_S = 0.01
    result = await skill.execute(
        {
            "action": "open_url",
            "target": json.dumps({"url": "https://example.com", "browser": "Google Chrome"}),
        },
        {},
    )

    assert result["ok"] is False
    assert result["status"] == "timeout"
    assert result["permission_source"] == "direct"
    assert result["permission"] == "accessibility"
    assert dispatched == []


@pytest.mark.asyncio
async def test_computer_use_create_folder_uses_allowed_artifact_roots(monkeypatch, tmp_path):
    skill = ComputerUseSkill()
    monkeypatch.setattr(skill, "_allowed_desktop_roots", lambda: [tmp_path])

    target = tmp_path / "Aura Journal"
    result = await skill.execute(
        {
            "action": "create_folder",
            "target": json.dumps({"path": str(target)}),
        },
        {},
    )

    assert result["ok"] is True
    assert result["action"] == "create_folder"
    assert result["effect_verified"] is True
    assert Path(result["path"]) == target
    assert target.is_dir()


@pytest.mark.asyncio
async def test_computer_use_clock_does_not_require_desktop_permissions(monkeypatch):
    from core.container import ServiceContainer

    tracker = get_degradation_tracker()
    tracker.reset()
    skill = ComputerUseSkill()
    skill.PERMISSION_CHECK_TIMEOUT_S = 0.01

    class SlowPermissionGuard:
        async def check_permission(self, *_args, **_kwargs):
            raise AssertionError("system clock must not request desktop permissions")

        def get_guidance(self, *_args, **_kwargs):
            return "permission guidance"

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: SlowPermissionGuard()
        if name == "permission_guard"
        else default,
    )

    monkeypatch.setattr(skill, "_read_menu_clock_macos", lambda: "Aug 21, 2026 at 7:26 AM")

    result = await skill.execute({"action": "read_menu_clock", "target": ""}, context={})

    assert result["ok"] is True
    assert result["source"] == "macos_system_clock"
    assert result["clock_text"] == "Aug 21, 2026 at 7:26 AM"
    assert tracker.recent(subsystem="computer_use") == []
    tracker.reset()


def test_computer_use_applescript_runner_uses_bounded_desktop_gateway_by_default(monkeypatch):
    from core.runtime.action_executor import ActionExecutor

    skill = ComputerUseSkill()
    run_call = {}

    def request_desktop_transport(*, script, source, timeout_s):
        run_call.update(
            {
                "script": script,
                "source": source,
                "timeout": timeout_s,
            }
        )
        return {"ok": True, "stdout": "menu clock", "stderr": "", "exit_code": 0}

    monkeypatch.delenv("AURA_COMPUTER_USE_NATIVE_APPLESCRIPT", raising=False)
    monkeypatch.setattr(
        ActionExecutor,
        "request_desktop_transport",
        request_desktop_transport,
    )

    assert skill._run_applescript('return "menu clock"', timeout=6) == "menu clock"
    assert run_call["script"] == 'return "menu clock"'
    assert run_call["source"] == "computer_use"
    assert run_call["timeout"] == 6


@pytest.mark.asyncio
async def test_computer_use_clipboard_actions_use_system_clipboard(monkeypatch):
    skill = ComputerUseSkill()
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "pbpaste":
            return SimpleNamespace(returncode=0, stdout="copied text", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("core.skills.computer_use.subprocess.run", fake_run)

    set_result = await skill.execute({"action": "set_clipboard", "target": "copied text"}, {})
    get_result = await skill.execute({"action": "get_clipboard", "target": ""}, {})

    assert set_result["ok"] is True
    assert set_result["action"] == "set_clipboard"
    assert set_result["chars"] == 11
    assert set_result["effect_verified"] is True
    assert len(set_result["sha256"]) == 64
    assert set_result["welfare_transaction_id"]
    assert set_result["welfare_transaction_outcome"] == "success"
    assert get_result["ok"] is True
    assert get_result["text"] == "copied text"
    assert get_result["welfare_transaction_id"]
    assert get_result["welfare_transaction_outcome"] == "success"
    assert calls[0][0] == ["pbcopy"]
    assert calls[1][0] == ["pbpaste"]


@pytest.mark.asyncio
async def test_computer_use_run_applescript_requires_permissions_and_blocks_shell(monkeypatch):
    skill = ComputerUseSkill()

    async def allow_permissions(*_args, **_kwargs):
        return None

    monkeypatch.setattr(skill, "_require_permissions", allow_permissions)
    monkeypatch.setattr(skill, "_run_applescript", lambda *_args, **_kwargs: "done")
    monkeypatch.setattr(skill, "_frontmost_app_name", lambda: "Notes")

    ok = await skill.execute(
        {"action": "run_applescript", "target": 'tell application "Notes" to activate'},
        {},
    )
    unverifiable = await skill.execute(
        {"action": "run_applescript", "target": 'return "done"'},
        {},
    )
    blocked_target = 'do shell script "rm -rf ' + "/".join(["", "tmp", "demo"]) + '"'
    blocked = await skill.execute({"action": "run_applescript", "target": blocked_target}, {})

    assert ok["ok"] is True
    assert ok["output"] == "done"
    assert ok["effect_verified"] is True
    assert ok["verification_results"][0]["strong"] is True
    assert unverifiable["ok"] is False
    assert unverifiable["status"] == "applescript_effect_contract_required"
    assert blocked["ok"] is False
    assert "blocked desktop operation" in blocked["error"]


@pytest.mark.asyncio
async def test_computer_use_desktop_file_pdf_and_move_receipts(monkeypatch, tmp_path):
    skill = ComputerUseSkill()
    monkeypatch.setattr(skill, "_allowed_desktop_roots", lambda: [tmp_path])

    source_pdf = tmp_path / "note.pdf"
    moved_pdf = tmp_path / "proof" / "moved-note.pdf"
    receipt_file = tmp_path / "proof" / "receipt.txt"

    pdf_payload = {
        "path": str(source_pdf),
        "title": "Aura Desktop Proof",
        "body": "Equation: 2 + 3 = 5\nCreated by Aura's governed desktop skill.",
    }
    move_payload = {"source": str(source_pdf), "destination": str(moved_pdf)}
    text_payload = {"path": str(receipt_file), "content": "moved PDF into proof folder"}

    rendered = await skill.execute(
        {"action": "render_text_pdf", "target": json.dumps(pdf_payload)},
        {},
    )
    moved = await skill.execute(
        {"action": "move_file", "target": json.dumps(move_payload)},
        {},
    )
    written = await skill.execute(
        {"action": "write_text_file", "target": json.dumps(text_payload)},
        {},
    )

    assert rendered["ok"] is True
    assert rendered["bytes"] > 100
    assert rendered["source_body_sha256"] == text_sha256(pdf_payload["body"])
    assert rendered["source_paragraph_sha256s"] == list(
        paragraph_sha256s(pdf_payload["body"])
    )
    assert not source_pdf.exists()
    assert moved["ok"] is True
    assert moved["effect_verified"] is True
    assert len(moved["sha256"]) == 64
    assert moved_pdf.exists()
    assert moved_pdf.read_bytes().startswith(b"%PDF")
    assert written["ok"] is True
    assert written["effect_verified"] is True
    assert len(written["sha256"]) == 64
    assert receipt_file.read_text() == "moved PDF into proof folder"


def test_desktop_artifact_verifier_rejects_unproven_mutations():
    from core.skills.desktop_task import DesktopTaskSkill, DesktopTaskStep

    cases = (
        (
            DesktopTaskStep(action="create_folder", target={"path": "Aura Proof"}),
            {"ok": True, "path": "Aura Proof"},
        ),
        (
            DesktopTaskStep(
                action="write_text_file",
                target={"path": "Aura Proof/note.txt", "content": "hello"},
            ),
            {"ok": True, "path": "Aura Proof/note.txt", "bytes": 5},
        ),
        (
            DesktopTaskStep(action="set_clipboard", target="hello"),
            {"ok": True, "chars": 5},
        ),
        (
            DesktopTaskStep(
                action="move_file",
                target={"source": "a.txt", "destination": "b.txt"},
            ),
            {"ok": True, "destination": "b.txt", "bytes": 5},
        ),
    )

    verifier = DesktopTaskSkill()
    for step, result in cases:
        verified, _evidence = verifier._verify_step_effect(step, result)
        assert verified is False


@pytest.mark.asyncio
async def test_computer_use_clock_falls_back_when_native_clock_fails(monkeypatch):
    tracker = get_degradation_tracker()
    tracker.reset()
    skill = ComputerUseSkill()

    def fail_native_clock():
        raise RuntimeError("native clock unavailable")

    monkeypatch.setattr(skill, "_read_menu_clock_macos", fail_native_clock)

    result = await skill.execute({"action": "read_menu_clock", "target": ""}, context={})

    assert result["ok"] is True
    assert result["status"] == "limited"
    assert result["source"] == "system_clock_fallback"
    assert "native clock unavailable" in result["error"]
    assert any(
        "native system clock" in record.action
        for record in tracker.recent(subsystem="computer_use")
    )
    tracker.reset()


@pytest.mark.asyncio
async def test_computer_use_click_failure_returns_payload_and_receipt(monkeypatch):
    tracker = get_degradation_tracker()
    tracker.reset()
    skill = ComputerUseSkill()

    class DesktopController:
        def __init__(self):
            self.clicked = False

        def click(self, x, y):
            self.clicked = True
            raise RuntimeError(f"desktop rejected click at {x},{y}")

    controller = DesktopController()
    monkeypatch.setattr("core.skills.computer_use.get_pyautogui", lambda: (controller, None))
    monkeypatch.setattr(skill, "_read_screen_text_macos", lambda: "before")

    async def permissions_available(*_args, **_kwargs):
        return None

    monkeypatch.setattr(skill, "_require_permissions", permissions_available)

    result = await skill.execute({"action": "click", "x": 10, "y": 20}, {})

    assert result["ok"] is False
    assert controller.clicked is True
    assert "desktop rejected click" in result["error"]
    assert any(
        "explicit computer-use failure payload" in record.action
        for record in tracker.recent(subsystem="computer_use")
    )
    tracker.reset()


@pytest.mark.asyncio
async def test_computer_use_mycelial_pulse_failure_does_not_block_action(monkeypatch):
    import core.skills.computer_use as computer_use
    from core.container import ServiceContainer

    tracker = get_degradation_tracker()
    tracker.reset()
    skill = ComputerUseSkill()
    container_failures = []
    original_container_get = ServiceContainer.get

    def unavailable_container(*_args, **_kwargs):
        if _args and _args[0] == "mycelial_network":
            container_failures.append("called")
            raise RuntimeError("container unavailable")
        return original_container_get(*_args, **_kwargs)

    monkeypatch.setattr(ServiceContainer, "get", unavailable_container)

    def run_echo(args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="hello\n", stderr="")

    monkeypatch.setattr(computer_use.subprocess, "run", run_echo)

    result = await skill.execute({"action": "run_command", "target": "echo hello"}, {})

    assert result["ok"] is True
    assert result["output"] == "hello"
    assert container_failures == ["called"]
    assert any(
        "mycelial telemetry pulse failed" in record.action
        for record in tracker.recent(subsystem="computer_use")
    )
    tracker.reset()


def test_write_text_file_versions_instead_of_refusing(tmp_path, monkeypatch):
    """Live failure: second run of the same desktop request died on
    'Refusing to overwrite existing file' and the whole chain failed.
    Repeats must version like Finder ('name (2).txt'), never clobber,
    never fail."""
    import json

    from core.skills.computer_use import ComputerUseSkill

    skill = ComputerUseSkill()
    monkeypatch.setattr(
        skill, "_resolve_allowed_desktop_path", lambda p, must_exist=False: tmp_path / Path(str(p)).name
    )

    first = skill._write_text_file(json.dumps({"path": "note.txt", "content": "one"}))
    second = skill._write_text_file(json.dumps({"path": "note.txt", "content": "two"}))

    assert first["ok"] and second["ok"]
    assert first["path"].endswith("note.txt")
    assert second["versioned"] is True
    assert second["path"].endswith("note (2).txt")
    assert (tmp_path / "note.txt").read_text() == "one"
    assert (tmp_path / "note (2).txt").read_text() == "two"


def test_write_text_file_overwrite_flag_still_overwrites(tmp_path, monkeypatch):
    import json

    from core.skills.computer_use import ComputerUseSkill

    skill = ComputerUseSkill()
    monkeypatch.setattr(
        skill, "_resolve_allowed_desktop_path", lambda p, must_exist=False: tmp_path / Path(str(p)).name
    )
    skill._write_text_file(json.dumps({"path": "note.txt", "content": "one"}))
    result = skill._write_text_file(
        json.dumps({"path": "note.txt", "content": "two", "overwrite": True})
    )
    assert result["ok"] and result["versioned"] is False
    assert (tmp_path / "note.txt").read_text() == "two"


@pytest.mark.asyncio
async def test_hotkey_dispatch_failure_carries_real_error(monkeypatch):
    """Visible-demo regression: a refused keystroke surfaced as 'unknown'
    because the failure receipt had no error text. Dispatch now goes
    through System Events and refusals carry the real message."""
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)

    scripts_seen = []

    def refusing_applescript(script, *, timeout=10):
        scripts_seen.append(script)
        raise RuntimeError("osascript is not allowed to send keystrokes")

    monkeypatch.setattr(skill, "_run_applescript", refusing_applescript)

    result = await skill.execute({"action": "hotkey", "target": "command+n"}, {})
    assert result["ok"] is False
    assert "not allowed to send keystrokes" in result["error"]
    assert scripts_seen  # the keystroke dispatch was attempted before refusal


@pytest.mark.asyncio
async def test_native_hotkey_timeout_recovers_with_pyautogui_fallback(monkeypatch):
    """Notes/TextEdit shortcuts must not depend on a single AX keystroke lane.

    System Events can stall even when permissions are granted and the native
    app is foreground. In that case Aura may use the secondary keyboard adapter,
    then keep the normal receipt/effect checks.
    """
    import core.skills.computer_use as computer_use_module

    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    class PyAutoGUIDouble:
        calls = []

        def hotkey(self, *keys, interval=0.0):
            self.calls.append((keys, interval))

    pyautogui_double = PyAutoGUIDouble()
    monkeypatch.setattr(
        computer_use_module,
        "get_pyautogui",
        lambda: (pyautogui_double, None),
    )
    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)
    monkeypatch.setattr(skill, "_frontmost_app_name", lambda: "Notes")
    monkeypatch.setattr(skill, "_read_screen_text_macos", lambda: "")

    scripts_seen = []

    def timing_out_applescript(script, *, timeout=10):
        scripts_seen.append(script)
        if "System Events" in script and "keystroke" in script:
            raise TimeoutError("AppleScript timed out after 8s.")
        return ""

    monkeypatch.setattr(skill, "_run_applescript", timing_out_applescript)

    async def controlled_sleep(secs):
        return None

    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)

    result = await skill.execute(
        {"action": "hotkey", "target": "command+n"},
        {"desktop_task_expected_frontmost_app": "Notes"},
    )

    assert result["ok"] is True
    assert result["effect_verified"] is True
    assert result["frontmost_app_before"] == "Notes"
    assert result["write_target_app_verified"] is True
    assert result["dispatch"].startswith("system_events_timeout:")
    assert "pyautogui:command+n" in result["dispatch"]
    assert pyautogui_double.calls == [(("command", "n"), 0.05)]
    assert any('keystroke "n" using {command down}' in s for s in scripts_seen)


@pytest.mark.asyncio
async def test_native_paste_accepts_prior_verified_frontmost_when_probe_unavailable(monkeypatch):
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)
    monkeypatch.setattr(
        skill,
        "_frontmost_app_name",
        lambda: (_ for _ in ()).throw(AssertionError("frontmost probe should be skipped")),
    )
    monkeypatch.setattr(skill, "_read_screen_text_macos", lambda: "")
    monkeypatch.setattr(skill, "_send_hotkey_system_events", lambda keys: f"system_events:{'+'.join(keys)}")
    activated = []

    async def controlled_activate(app_name):
        activated.append(app_name)
        return ""

    monkeypatch.setattr(skill, "_activate_app", controlled_activate)

    async def controlled_sleep(secs):
        return None

    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)

    result = await skill.execute(
        {"action": "hotkey", "target": "command+v"},
        {
            "desktop_task_expected_frontmost_app": "Notes",
            "desktop_task_prior_verified_frontmost_app": "Notes",
            "desktop_task_allow_unavailable_frontmost_from_prior": True,
        },
    )

    assert result["ok"] is True
    assert result["effect_verified"] is True
    assert result["frontmost_app_before"] == "Notes"
    assert result["frontmost_app_from_prior_receipt"] is True
    assert result["write_target_app_verified"] is True
    assert result["dispatch"] == "system_events:command+v"
    assert activated == ["Notes"]


@pytest.mark.asyncio
async def test_hotkey_dispatch_without_focused_control_readback_is_rejected(monkeypatch):
    """A clean dispatch is not proof that a browser editor accepted it."""
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)

    scripts = []

    def recording_applescript(script, *, timeout=10):
        scripts.append(script)
        if "frontmost is true" in script:
            return "Google Chrome"
        if "System Events" in script and "keystroke" in script:
            return ""
        return ""

    monkeypatch.setattr(skill, "_run_applescript", recording_applescript)
    monkeypatch.setattr(skill, "_focused_element_snapshot", lambda: "")

    async def controlled_sleep(secs):
        return None

    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)

    result = await skill.execute({"action": "hotkey", "target": "command+v"}, {})
    assert result["ok"] is False
    assert result["effect_verified"] is False
    assert result["dispatch"].startswith("system_events:")
    assert "verification was unavailable" in result["verification"].lower()
    assert any('keystroke "v" using {command down}' in s for s in scripts)

    from core.skills.desktop_task import DesktopTaskSkill, DesktopTaskStep

    step = DesktopTaskStep(
        action="hotkey",
        target="command+v",
        reason="paste staged body",
        expect="The focused writing surface accepts the paste shortcut.",
    )
    verified, evidence = DesktopTaskSkill()._verify_step_effect(step, result)
    assert verified is False
    assert "verification was unavailable" in evidence.lower()


@pytest.mark.asyncio
async def test_browser_paste_refuses_location_bar_focus(monkeypatch):
    """A Google Docs paste must not be counted when focus is still in the URL bar."""
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)
    monkeypatch.setattr(skill, "_frontmost_app_name", lambda: "Google Chrome")
    monkeypatch.setattr(
        skill,
        "_focused_element_snapshot",
        lambda: "AXTextField\thttps://docs.google.com/document/u/0/create",
    )

    result = await skill.execute({"action": "hotkey", "target": "command+v"}, {})

    assert result["ok"] is False
    assert result["effect_verified"] is False
    assert result["verification"] == "browser_location_bar_focused"
    assert "address/search field" in result["error"]


@pytest.mark.asyncio
async def test_browser_paste_refuses_address_bar_by_accessibility_description(monkeypatch):
    """Chrome can expose arbitrary selected text in the omnibox, not a URL."""
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)
    monkeypatch.setattr(skill, "_frontmost_app_name", lambda: "Google Chrome")
    monkeypatch.setattr(
        skill,
        "_focused_element_snapshot",
        lambda: "AXTextField\tAura paragraph body\t\tAddress and search bar\t",
    )

    result = await skill.execute({"action": "hotkey", "target": "command+v"}, {})

    assert result["ok"] is False
    assert result["verification"] == "browser_location_bar_focused"


@pytest.mark.asyncio
async def test_browser_paste_refuses_generic_text_field_when_doc_focus_required(monkeypatch):
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)
    monkeypatch.setattr(skill, "_frontmost_app_name", lambda: "Google Chrome")
    monkeypatch.setattr(
        skill,
        "_focused_element_snapshot",
        lambda: "AXTextField\tAura paragraph body\t\t\t",
    )

    result = await skill.execute(
        {"action": "hotkey", "target": "command+v"},
        {"desktop_task_requires_editable_focus": True},
    )

    assert result["ok"] is False
    assert result["verification"] == "browser_location_bar_focused"


@pytest.mark.asyncio
async def test_browser_paste_uses_prior_editor_focus_when_ax_unavailable(monkeypatch):
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    staged = "Aura writes the article summary into the verified document body."
    staged_sha = hashlib.sha256(staged.encode("utf-8")).hexdigest()

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)
    monkeypatch.setattr(skill, "_frontmost_app_name", lambda: "Google Chrome")
    monkeypatch.setattr(skill, "_focused_element_snapshot", lambda: "")
    monkeypatch.setattr(skill, "_send_hotkey_system_events", lambda keys: "ok")
    monkeypatch.setattr(skill, "_get_clipboard", lambda: {"ok": True, "text": staged})

    async def controlled_sleep(_secs):
        return None

    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)

    result = await skill.execute(
        {"action": "hotkey", "target": "command+v"},
        {
            "desktop_task_expected_frontmost_app": "Google Chrome",
            "desktop_task_requires_editable_focus": True,
            "desktop_task_editor_focus_verified": True,
            "desktop_task_verified_editor_url": "https://docs.google.com/document/d/abc/edit",
            "desktop_task_expected_clipboard_sha256": staged_sha,
            "desktop_task_expected_clipboard_chars": len(staged),
        },
    )

    assert result["ok"] is True
    assert result["effect_verified"] is True
    assert result["is_paste"] is True
    assert result["write_target_app_verified"] is True
    assert result["clipboard_payload_verification"]["verified"] is True
    assert "previously verified browser editor" in result["verification"]


@pytest.mark.asyncio
async def test_browser_type_refuses_generic_text_field_when_doc_focus_required(monkeypatch):
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)
    monkeypatch.setattr(skill, "_frontmost_app_name", lambda: "Google Chrome")
    monkeypatch.setattr(
        skill,
        "_focused_element_snapshot",
        lambda: "AXTextField\tAura paragraph body\t\t\t",
    )

    result = await skill.execute(
        {"action": "type", "target": "This belongs in the document body."},
        {"desktop_task_requires_editable_focus": True},
    )

    assert result["ok"] is False
    assert result["verification"] == "browser_text_control_focused"


@pytest.mark.asyncio
async def test_web_editor_focus_rejects_generic_browser_text_field(monkeypatch):
    skill = ComputerUseSkill()

    class PyAutoGUIDouble:
        def size(self):
            return (1440, 900)

        def click(self, x, y):
            return None

    snapshots = iter(
        (
            "AXTextField\tquery text\t\tSearch or enter address\t",
            "AXTextField\tarticle draft\t\t\t",
            "AXTextField\tarticle draft\t\t\t",
        )
    )
    monkeypatch.setattr(
        skill,
        "_focused_element_snapshot",
        lambda: next(snapshots),
    )
    monkeypatch.setattr(skill, "_send_hotkey_system_events", lambda keys: "ok")

    async def controlled_sleep(_secs):
        return None

    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)

    focused, snapshot, reason = await skill._focus_web_editor_surface(PyAutoGUIDouble())

    assert focused is False
    assert "AXTextField" in snapshot
    assert reason == "generic_browser_text_field_focused"


@pytest.mark.asyncio
async def test_web_editor_focus_accepts_editor_like_surface(monkeypatch):
    skill = ComputerUseSkill()

    class PyAutoGUIDouble:
        def size(self):
            return (1440, 900)

        def click(self, x, y):
            return None

    snapshots = iter(("AXWebArea\tGoogle Docs document editor body\t\t\t",))
    monkeypatch.setattr(
        skill,
        "_focused_element_snapshot",
        lambda: next(snapshots),
    )
    monkeypatch.setattr(skill, "_send_hotkey_system_events", lambda keys: "ok")

    async def controlled_sleep(_secs):
        return None

    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)

    focused, snapshot, reason = await skill._focus_web_editor_surface(PyAutoGUIDouble())

    assert focused is True
    assert "Google Docs" in snapshot
    assert reason == "editable_focus_verified"


@pytest.mark.asyncio
async def test_web_editor_focus_accepts_canvas_surface_when_editor_url_still_active(
    monkeypatch,
):
    skill = ComputerUseSkill()

    class PyAutoGUIDouble:
        def size(self):
            return (1440, 900)

        def click(self, x, y):
            return None

    monkeypatch.setattr(
        skill,
        "_focused_element_snapshot",
        lambda: "AXWebArea\t\t\t\t",
    )
    monkeypatch.setattr(skill, "_send_hotkey_system_events", lambda keys: "ok")
    monkeypatch.setattr(
        skill,
        "_active_browser_location",
        lambda browser: ("https://docs.google.com/document/d/abc/edit", "Aura proof"),
    )

    async def controlled_sleep(_secs):
        return None

    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)

    focused, snapshot, reason = await skill._focus_web_editor_surface(
        PyAutoGUIDouble(),
        browser="Google Chrome",
        target_url="https://docs.google.com/document/u/0/create",
    )

    assert focused is True
    assert snapshot.startswith("AXWebArea")
    assert reason == "editable_focus_verified_canvas_url"


@pytest.mark.asyncio
async def test_web_editor_focus_accepts_no_ax_focus_when_editor_url_still_active(
    monkeypatch,
):
    skill = ComputerUseSkill()

    class PyAutoGUIDouble:
        def size(self):
            return (1440, 900)

        def click(self, x, y):
            return None

    monkeypatch.setattr(skill, "_focused_element_snapshot", lambda: "")
    monkeypatch.setattr(skill, "_send_hotkey_system_events", lambda keys: "ok")
    monkeypatch.setattr(
        skill,
        "_active_browser_location",
        lambda browser: ("https://docs.google.com/document/d/abc/edit", "Aura proof"),
    )

    async def controlled_sleep(_secs):
        return None

    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)

    focused, snapshot, reason = await skill._focus_web_editor_surface(
        PyAutoGUIDouble(),
        browser="Google Chrome",
        target_url="https://docs.google.com/document/u/0/create",
    )

    assert focused is True
    assert snapshot == ""
    assert reason == "editable_focus_verified_canvas_no_ax_focus"


@pytest.mark.asyncio
async def test_web_editor_focus_rejects_canvas_surface_when_editor_url_not_active(
    monkeypatch,
):
    skill = ComputerUseSkill()

    class PyAutoGUIDouble:
        def size(self):
            return (1440, 900)

        def click(self, x, y):
            return None

    monkeypatch.setattr(
        skill,
        "_focused_element_snapshot",
        lambda: "AXWebArea\t\t\t\t",
    )
    monkeypatch.setattr(skill, "_send_hotkey_system_events", lambda keys: "ok")
    monkeypatch.setattr(
        skill,
        "_active_browser_location",
        lambda browser: ("https://example.com/not-a-doc", "Example"),
    )

    async def controlled_sleep(_secs):
        return None

    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)

    focused, snapshot, reason = await skill._focus_web_editor_surface(
        PyAutoGUIDouble(),
        browser="Google Chrome",
        target_url="https://docs.google.com/document/u/0/create",
    )

    assert focused is False
    assert snapshot.startswith("AXWebArea")
    assert reason == "editable_focus_unverified"


@pytest.mark.asyncio
async def test_open_url_waits_for_google_docs_create_to_resolve_before_focus(
    monkeypatch,
):
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)

    argv_seen = []

    class FakeSubprocessGateway:
        def run(self, argv, **kwargs):
            argv_seen.append(argv)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "core.skills.computer_use.get_subprocess_gateway",
        lambda: FakeSubprocessGateway(),
    )
    async def wait_for_frontmost_app(expected):
        return True, expected

    monkeypatch.setattr(skill, "_wait_for_frontmost_app", wait_for_frontmost_app)
    locations = iter(
        (
            ("https://docs.google.com/document/u/0/create", "New document"),
            ("https://docs.google.com/document/d/abc/edit", "Aura proof"),
        )
    )
    monkeypatch.setattr(skill, "_active_browser_location", lambda browser: next(locations))

    async def focus_surface(pyautogui, *, browser="", target_url=""):
        assert browser == "Google Chrome"
        assert target_url == "https://docs.google.com/document/u/0/create"
        return True, "AXWebArea\t\t\t\t", "editable_focus_verified_canvas_url"

    monkeypatch.setattr(skill, "_focus_web_editor_surface", focus_surface)
    monkeypatch.setattr("core.skills.computer_use.get_pyautogui", lambda: (object(), ""))

    async def controlled_sleep(_secs):
        return None

    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)

    target = json.dumps(
        {
            "url": "https://docs.google.com/document/u/0/create",
            "browser": "Google Chrome",
            "requires_editable_focus": True,
        }
    )
    result = await skill.execute({"action": "open_url", "target": target}, {})

    assert result["ok"] is True
    assert result["doc_focused"] is True
    assert result["active_url"] == "https://docs.google.com/document/d/abc/edit"
    assert result["focus_error"] == "editable_focus_verified_canvas_url"
    assert argv_seen == [
        ["open", "-a", "Google Chrome", "https://docs.google.com/document/u/0/create"]
    ]


@pytest.mark.asyncio
async def test_hotkey_screen_shift_still_required_when_screen_readable(monkeypatch):
    """With a readable screen and NO state shift, the step stays red —
    the dispatch-receipt fallback must not soften real verification."""
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)

    def stable_screen_applescript(script, *, timeout=10):
        if "System Events" in script and "keystroke" in script:
            return ""
        return "Notes: same visible text"

    monkeypatch.setattr(skill, "_run_applescript", stable_screen_applescript)

    async def controlled_sleep(secs):
        return None

    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)

    result = await skill.execute({"action": "hotkey", "target": "command+n"}, {})
    assert result["ok"] is False
    assert "no visible state shift" in result["error"]


@pytest.mark.asyncio
async def test_hotkey_skips_screen_reads_on_browser_surface(monkeypatch):
    """Part-2 round 2: a loading Google Docs tab held the 'entire contents'
    accessibility walk busy long enough that the keystroke itself timed
    out after 8s. When a browser is frontmost the screen-text reads are
    skipped entirely and the governed dispatch receipt is the evidence."""
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)

    scripts = []

    def recording_applescript(script, *, timeout=10):
        scripts.append(script)
        if "frontmost is true" in script:
            return "Google Chrome"
        if "System Events" in script and "keystroke" in script:
            return ""
        # If the entire-contents walk is ever reached on a browser, fail
        # loudly so the regression is caught — it must NOT be called.
        raise AssertionError("screen-text walk must be skipped on browser surfaces")

    monkeypatch.setattr(skill, "_run_applescript", recording_applescript)
    focused_snapshots = iter(("AXTextArea\tbefore", "AXTextArea\tbefore and after"))
    monkeypatch.setattr(
        skill,
        "_focused_element_snapshot",
        lambda: next(focused_snapshots),
    )

    async def controlled_sleep(secs):
        return None

    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)

    result = await skill.execute({"action": "hotkey", "target": "command+v"}, {})
    assert result["ok"] is True
    assert result["effect_verified"] is True
    assert result["verification"] == "Focused element changed."
    assert result["dispatch"].startswith("system_events:")
    assert not any("entire contents" in s for s in scripts)


@pytest.mark.asyncio
async def test_paste_receipt_verifies_expected_clipboard_payload(monkeypatch):
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)
    monkeypatch.setattr(skill, "_frontmost_app_name", lambda: "Google Chrome")
    focused_snapshots = iter(
        ("AXTextArea\tGoogle Docs editor", "AXTextArea\tGoogle Docs editor changed")
    )
    monkeypatch.setattr(skill, "_focused_element_snapshot", lambda: next(focused_snapshots))
    monkeypatch.setattr(
        skill,
        "_send_hotkey_system_events",
        lambda keys: f"system_events:{'+'.join(keys)}",
    )
    staged = "Aura should paste this into the document body."
    monkeypatch.setattr(
        skill,
        "_get_clipboard",
        lambda: {"ok": True, "text": staged, "chars": len(staged)},
    )

    async def controlled_sleep(secs):
        return None

    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)

    result = await skill.execute(
        {"action": "hotkey", "target": "command+v"},
        {
            "desktop_task_expected_frontmost_app": "Google Chrome",
            "desktop_task_expected_clipboard_sha256": hashlib.sha256(
                staged.encode("utf-8")
            ).hexdigest(),
            "desktop_task_expected_clipboard_chars": len(staged),
        },
    )

    assert result["ok"] is True
    assert result["is_paste"] is True
    assert result["write_target_app_verified"] is True
    assert result["clipboard_payload_verification"]["verified"] is True
    assert result["visible_state_changed"] is True


@pytest.mark.asyncio
async def test_open_url_targets_named_browser(monkeypatch):
    """Bryan's Google session lives in Chrome: open_url honors a browser
    in the step target via 'open -a', bounded to a known browser set."""
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)

    argv_seen = []

    class FakeSubprocessGateway:
        def run(self, argv, **kwargs):
            argv_seen.append(argv)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "core.skills.computer_use.get_subprocess_gateway",
        lambda: FakeSubprocessGateway(),
    )
    monkeypatch.setattr(
        skill,
        "_wait_for_frontmost_app",
        lambda expected: asyncio.sleep(0, result=(True, expected)),
    )
    monkeypatch.setattr(
        skill,
        "_active_browser_location",
        lambda browser: ("https://docs.google.com/document/u/0/create", "New document"),
    )

    target = json.dumps(
        {"url": "https://docs.google.com/document/u/0/create", "browser": "Google Chrome"}
    )
    result = await skill.execute({"action": "open_url", "target": target}, {})
    assert result["ok"] is True
    assert result["browser"] == "Google Chrome"
    assert argv_seen == [
        ["open", "-a", "Google Chrome", "https://docs.google.com/document/u/0/create"]
    ]
    assert result["effect_verified"] is True
    assert result["frontmost_app"] == "Google Chrome"
    assert result["active_url"] == "https://docs.google.com/document/u/0/create"


@pytest.mark.asyncio
async def test_open_url_repairs_frontmost_browser_with_wrong_active_tab(monkeypatch):
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)

    argv_seen = []

    class FakeSubprocessGateway:
        def run(self, argv, **kwargs):
            argv_seen.append(argv)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "core.skills.computer_use.get_subprocess_gateway",
        lambda: FakeSubprocessGateway(),
    )
    monkeypatch.setattr(
        skill,
        "_wait_for_frontmost_app",
        lambda expected: asyncio.sleep(0, result=(True, expected)),
    )
    locations = iter(
        (
            ("https://example.test/wrong-tab", "Wrong tab"),
            ("https://docs.google.com/document/u/0/create", "New document"),
        )
    )
    monkeypatch.setattr(skill, "_active_browser_location", lambda browser: next(locations))
    forced = []
    monkeypatch.setattr(
        skill,
        "_force_browser_tab_url",
        lambda browser, url: forced.append((browser, url)) or "",
    )

    target = json.dumps({"url": "https://docs.google.com/document/u/0/create", "browser": "Google Chrome"})
    result = await skill.execute({"action": "open_url", "target": target}, {})

    assert result["ok"] is True
    assert result["effect_verified"] is True
    assert result["frontmost_app"] == "Google Chrome"
    assert result["active_url"] == "https://docs.google.com/document/u/0/create"
    assert result["forced_navigation"] is True
    assert forced == [("Google Chrome", "https://docs.google.com/document/u/0/create")]
    assert argv_seen == [
        ["open", "-a", "Google Chrome", "https://docs.google.com/document/u/0/create"]
    ]


@pytest.mark.asyncio
async def test_open_app_canonicalizes_user_wording_at_execution_boundary(monkeypatch):
    from core.runtime.app_target_resolution import InstalledApp

    skill = ComputerUseSkill()
    argv_seen = []
    activated = []

    async def permission_pass(*_args, **_kwargs):
        return None

    async def activate(app):
        activated.append(app)

    class Gateway:
        def run(self, argv, **_kwargs):
            argv_seen.append(argv)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(skill, "_require_permissions", permission_pass)
    monkeypatch.setattr(skill, "_activate_app", activate)
    monkeypatch.setattr(
        skill,
        "_wait_for_frontmost_app",
        lambda expected: asyncio.sleep(0, result=(True, expected)),
    )
    monkeypatch.setattr(
        "core.skills.computer_use.get_subprocess_gateway",
        lambda: Gateway(),
    )
    monkeypatch.setattr(
        "core.runtime.app_target_resolution.installed_app_inventory",
        lambda **_kwargs: (
            InstalledApp("Notes", "/System/Applications/Notes.app"),
        ),
    )

    result = await skill.execute({"action": "open_app", "target": "Note app"}, {})

    assert result["ok"] is True
    assert result["opened"] == "Notes"
    assert argv_seen == [["open", "/System/Applications/Notes.app"]]
    assert activated == ["Notes"]
    assert result["app_resolution"]["method"] == "installed_exact"
    assert result["app_resolution"]["corrected"] is True


@pytest.mark.asyncio
async def test_open_app_refreshes_a_stale_bundle_path_before_failing(monkeypatch):
    from core.runtime.app_target_resolution import AppTargetResolution

    skill = ComputerUseSkill()
    resolutions = iter(
        (
            AppTargetResolution(
                requested="Notes",
                canonical="Notes",
                resolved="Notes",
                app_path="/Applications/Old Notes.app",
                method="installed_exact",
                inventory_available=True,
            ),
            AppTargetResolution(
                requested="Notes",
                canonical="Notes",
                resolved="Notes",
                app_path="/System/Applications/Notes.app",
                method="installed_exact",
                inventory_available=True,
            ),
        )
    )
    resolve_calls: list[bool] = []

    def resolve(_target, *, refresh=False):
        resolve_calls.append(refresh)
        return next(resolutions)

    class Gateway:
        def __init__(self):
            self.argv: list[list[str]] = []

        def run(self, argv, **_kwargs):
            self.argv.append(argv)
            if "Old Notes.app" in argv[-1]:
                return SimpleNamespace(returncode=1, stdout="", stderr="stale path")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    gateway = Gateway()

    async def permission_pass(*_args, **_kwargs):
        return None

    async def activate(_app):
        return None

    monkeypatch.setattr(skill, "_require_permissions", permission_pass)
    monkeypatch.setattr(skill, "_activate_app", activate)
    monkeypatch.setattr(
        skill,
        "_wait_for_frontmost_app",
        lambda expected: asyncio.sleep(0, result=(True, expected)),
    )
    monkeypatch.setattr(
        "core.skills.computer_use.resolve_installed_app_target",
        resolve,
    )
    monkeypatch.setattr(
        "core.skills.computer_use.get_subprocess_gateway",
        lambda: gateway,
    )

    result = await skill.execute({"action": "open_app", "target": "Notes"}, {})

    assert result["ok"] is True
    assert resolve_calls == [False, True]
    assert gateway.argv == [
        ["open", "/Applications/Old Notes.app"],
        ["open", "/System/Applications/Notes.app"],
    ]
    assert result["app_resolution"]["app_path"] == "/System/Applications/Notes.app"


@pytest.mark.asyncio
async def test_open_url_repairs_default_browser_when_readback_is_empty(monkeypatch):
    """Default-browser opens still need a proven active tab.

    Live proof exposed the failure: macOS accepted `open <url>`, Safari came
    frontmost, but active URL readback was empty, so the governed chain stopped
    before the writing step. The repair must not depend on an explicitly named
    browser because default-browser dispatch is a normal user-lane action.
    """

    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)

    argv_seen = []

    class FakeSubprocessGateway:
        def run(self, argv, **kwargs):
            argv_seen.append(argv)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "core.skills.computer_use.get_subprocess_gateway",
        lambda: FakeSubprocessGateway(),
    )
    monkeypatch.setattr(skill, "_frontmost_app_name", lambda: "Safari")
    monkeypatch.setattr(
        skill,
        "_wait_for_frontmost_app",
        lambda expected: asyncio.sleep(0, result=(True, expected)),
    )
    locations = iter(
        (
            ("", ""),
            ("https://duckduckgo.com/?q=robot&iax=images&ia=images", "robot images"),
        )
    )
    monkeypatch.setattr(skill, "_active_browser_location", lambda browser: next(locations))
    forced = []
    monkeypatch.setattr(
        skill,
        "_force_browser_tab_url",
        lambda browser, url: forced.append((browser, url)) or "",
    )

    target = "https://duckduckgo.com/?q=robot&iax=images&ia=images"
    result = await skill.execute({"action": "open_url", "target": target}, {})

    assert result["ok"] is True
    assert result["browser"] == ""
    assert result["frontmost_app"] == "Safari"
    assert result["active_url"] == target
    assert result["forced_navigation"] is True
    assert forced == [("Safari", target)]
    assert argv_seen == [["open", target]]


@pytest.mark.asyncio
async def test_open_url_still_rejects_after_forced_navigation_mismatch(monkeypatch):
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)

    class FakeSubprocessGateway:
        def run(self, argv, **kwargs):
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "core.skills.computer_use.get_subprocess_gateway",
        lambda: FakeSubprocessGateway(),
    )
    monkeypatch.setattr(
        skill,
        "_wait_for_frontmost_app",
        lambda expected: asyncio.sleep(0, result=(True, expected)),
    )
    monkeypatch.setattr(
        skill,
        "_active_browser_location",
        lambda browser: ("https://example.test/wrong-tab", "Wrong tab"),
    )
    monkeypatch.setattr(skill, "_force_browser_tab_url", lambda browser, url: "")

    target = json.dumps({"url": "https://docs.google.com/document/u/0/create", "browser": "Google Chrome"})
    result = await skill.execute({"action": "open_url", "target": target}, {})

    assert result["ok"] is False
    assert result["effect_verified"] is False
    assert result["forced_navigation"] is True
    assert result["active_url"] == "https://example.test/wrong-tab"
    assert "could not be semantically confirmed" in result["error"]


@pytest.mark.asyncio
async def test_open_app_reports_a_lost_focus_race_without_failing(monkeypatch):
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)

    class FakeSubprocessGateway:
        def run(self, argv, **kwargs):
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "core.skills.computer_use.get_subprocess_gateway",
        lambda: FakeSubprocessGateway(),
    )
    async def mismatched_frontmost(_expected):
        return False, "Finder"

    monkeypatch.setattr(
        skill,
        "_wait_for_frontmost_app",
        mismatched_frontmost,
    )

    result = await skill.execute({"action": "open_app", "target": "Notes"}, {})

    # LAUNCHING IS THE ACTION. BEING FRONTMOST IS A WISH.
    #
    # This asserted ok is False, which meant losing the focus race killed the
    # step — and losing it is not something she controls, because the person is
    # typing in something while she works. Measured 2026-07-29:
    #
    #   open_app failed: Application launch command succeeded, but the
    #   requested app did not become frontmost (observed=Claude).
    #   Completed 0/2 steps.
    #
    # "Claude" was the window Bryan happened to be reading. Notes had launched
    # perfectly well and the whole task died at step zero over which window had
    # the highlight — pre-empting both of the mechanisms that handle this:
    # hold_focus re-asserts the front for steps that need it, and the scripting
    # dictionary does not need it at all.
    assert result["ok"] is True, "the app opened, which is what was asked"
    assert result["frontmost_app"] == "Finder"
    assert result["is_frontmost"] is False, "and that is reported, not hidden"
    assert "another app holds the front" in result["verification"]
    assert "error" not in result


@pytest.mark.asyncio
async def test_open_app_activates_before_frontmost_verification(monkeypatch):
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)

    class FakeSubprocessGateway:
        def run(self, argv, **kwargs):
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "core.skills.computer_use.get_subprocess_gateway",
        lambda: FakeSubprocessGateway(),
    )

    calls = []

    async def controlled_activate(app_name):
        calls.append(("activate", app_name))

    async def controlled_frontmost(expected):
        calls.append(("wait", expected))
        return True, expected

    monkeypatch.setattr(skill, "_activate_app", controlled_activate)
    monkeypatch.setattr(skill, "_wait_for_frontmost_app", controlled_frontmost)

    result = await skill.execute({"action": "open_app", "target": "Notes"}, {})

    assert result["ok"] is True
    assert result["effect_verified"] is True
    assert calls == [("activate", "Notes"), ("wait", "Notes")]


def test_frontmost_app_match_accepts_bundle_suffix_not_wrong_app():
    skill = ComputerUseSkill()

    assert skill._frontmost_app_matches("Calculator", "Calculator.app") is True
    assert skill._frontmost_app_matches("Notes", "Notes app") is True
    assert skill._frontmost_app_matches("Finder", "Notes.app") is False


@pytest.mark.asyncio
async def test_wait_for_frontmost_app_actively_raises_expected_app(monkeypatch):
    skill = ComputerUseSkill()

    seen = iter(("Notes", "Google Chrome"))
    activations = []

    monkeypatch.setattr(skill, "_frontmost_app_name", lambda: next(seen))

    async def controlled_activate(app_name):
        activations.append(app_name)

    async def controlled_sleep(_secs):
        return None

    monkeypatch.setattr(skill, "_activate_app", controlled_activate)
    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)

    ok, last_seen = await skill._wait_for_frontmost_app("Google Chrome")

    assert ok is True
    assert last_seen == "Google Chrome"
    assert activations == ["Google Chrome"]


@pytest.mark.asyncio
async def test_open_url_refuses_unknown_browser(monkeypatch):
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)

    target = json.dumps({"url": "https://example.com", "browser": "Calculator"})
    result = await skill.execute({"action": "open_url", "target": target}, {})
    assert result["ok"] is False
    assert "not in the allowed browser set" in result["error"]


class _FakeOSSettingsAdapter:
    """Stand-in for the canonical OSSettingsAdapter — records delegated
    set calls and serves a goal-state read-back."""

    def __init__(self, wallpaper="/Library/Desktop Pictures/Sonoma.heic",
                 appearance="light", volume=10, drop_first_readback=False):
        self._wallpaper = wallpaper
        self._appearance = appearance
        self._volume = volume
        self.calls = []
        self._set_done = False
        self._readbacks_after_set = 0
        self._drop_first = drop_first_readback

    async def get_wallpaper(self):
        # Simulate the modern-macOS race: the first read-back AFTER a set
        # returns `missing value` (empty) before the store propagates.
        if self._set_done:
            self._readbacks_after_set += 1
            if self._drop_first and self._readbacks_after_set == 1:
                return ""
        return self._wallpaper

    async def set_wallpaper(self, path):
        self.calls.append(("set_wallpaper", path))
        self._wallpaper = str(path)
        self._set_done = True
        return SimpleNamespace(success=True)

    async def get_appearance_mode(self):
        return self._appearance

    async def set_appearance_mode(self, mode):
        self.calls.append(("set_appearance_mode", mode))
        self._appearance = mode
        return True

    async def get_volume(self):
        return self._volume

    async def set_volume(self, level):
        self.calls.append(("set_volume", level))
        self._volume = int(level)
        return True


def _patch_adapter(monkeypatch, adapter):
    from core.container import ServiceContainer

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: adapter if name == "os_settings" else default),
    )


@pytest.mark.asyncio
async def test_system_control_wallpaper_delegates_to_adapter(monkeypatch, tmp_path):
    """system_control routes execution through the canonical
    OSSettingsAdapter (no parallel AppleScript), confirms the goal-state
    via the adapter's getter, and survives the modern-macOS read-back race."""
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)
    monkeypatch.setattr(ComputerUseSkill, "_SETTING_READBACK_INTERVAL_S", 0.0)

    img = tmp_path / "squid_wallpaper.png"
    img.write_bytes(b"\x89PNG fake")
    monkeypatch.setattr(
        skill, "_resolve_allowed_desktop_path", lambda raw, must_exist=False: img
    )
    adapter = _FakeOSSettingsAdapter(drop_first_readback=True)
    _patch_adapter(monkeypatch, adapter)

    result = await skill.execute(
        {"action": "system_control", "target": json.dumps({"domain": "wallpaper", "value": str(img)})},
        {},
    )
    assert result["ok"] is True
    assert result["effect_verified"] is True
    assert result["domain"] == "wallpaper"
    assert ("set_wallpaper", str(img)) in adapter.calls
    assert result["previous"].endswith("Sonoma.heic")

    from core.skills.desktop_task import DesktopTaskSkill, DesktopTaskStep

    step = DesktopTaskStep(
        action="system_control",
        target={"domain": "wallpaper", "value": str(img)},
        reason="set wallpaper",
        expect="Read-back confirms the wallpaper goal-state.",
    )
    verified, evidence = DesktopTaskSkill()._verify_step_effect(step, result)
    assert verified, evidence


@pytest.mark.asyncio
async def test_system_control_dark_mode_and_volume_delegate(monkeypatch):
    """Same general action drives dark mode and volume — proof the
    registry is general, not wallpaper-shaped."""
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)
    monkeypatch.setattr(ComputerUseSkill, "_SETTING_READBACK_INTERVAL_S", 0.0)

    adapter = _FakeOSSettingsAdapter()
    _patch_adapter(monkeypatch, adapter)

    dark = await skill.execute(
        {"action": "system_control", "target": json.dumps({"domain": "dark_mode", "value": "true"})}, {}
    )
    assert dark["ok"] is True
    assert ("set_appearance_mode", "dark") in adapter.calls

    vol = await skill.execute(
        {"action": "system_control", "target": json.dumps({"domain": "volume", "value": "30"})}, {}
    )
    assert vol["ok"] is True
    assert ("set_volume", 30) in adapter.calls


@pytest.mark.asyncio
async def test_system_control_readback_mismatch_fails_closed(monkeypatch, tmp_path):
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)
    monkeypatch.setattr(ComputerUseSkill, "_SETTING_READBACK_INTERVAL_S", 0.0)

    img = tmp_path / "squid_wallpaper.png"
    img.write_bytes(b"\x89PNG fake")
    monkeypatch.setattr(
        skill, "_resolve_allowed_desktop_path", lambda raw, must_exist=False: img
    )

    class StubbornAdapter(_FakeOSSettingsAdapter):
        async def set_wallpaper(self, path):  # set silently no-ops
            self.calls.append(("set_wallpaper", path))
            return SimpleNamespace(success=False)

    adapter = StubbornAdapter()  # get_wallpaper keeps returning the old one
    _patch_adapter(monkeypatch, adapter)

    result = await skill.execute(
        {"action": "system_control", "target": json.dumps({"domain": "wallpaper", "value": str(img)})}, {}
    )
    assert result["ok"] is False
    assert "does not confirm" in result["error"]


@pytest.mark.asyncio
async def test_system_control_unavailable_adapter_fails_closed(monkeypatch):
    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)
    _patch_adapter(monkeypatch, None)

    result = await skill.execute(
        {"action": "system_control", "target": json.dumps({"domain": "dark_mode", "value": "true"})}, {}
    )
    assert result["ok"] is False
    assert "unavailable" in result["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["inspect_screen", "read_screen_text"])
async def test_a_refused_screen_capture_returns_the_refusal_not_a_reading(
    monkeypatch, action
):
    """The privacy gate is what the pinned fixture above stops depending on the
    host for, so the refusal it produces needs a test of its own — and this one
    controls the answer instead of hoping the machine gives it."""
    from core.security import screen_capture_policy as policy

    async def _refuse():
        return policy.ScreenCaptureAdmission(
            allowed=False,
            reason=policy.ScreenCaptureDenial.PRIVATE_FOREGROUND,
            context_known=True,
        )

    monkeypatch.setattr(policy, "evaluate_screen_capture_admission_async", _refuse)

    skill = ComputerUseSkill()

    async def controlled_permission_pass(capability, *permission_names):
        return None

    monkeypatch.setattr(skill, "_require_permissions", controlled_permission_pass)

    def _must_not_capture():
        raise AssertionError("a refused capture still read the screen")

    monkeypatch.setattr(
        "core.perception.screen_perception.get_screen_perception", _must_not_capture
    )

    result = await skill.execute({"action": action, "target": ""}, {})

    assert result["ok"] is False
    assert result["status"] == "screen_capture_refused"
    assert result["text"] == ""
    # The reason reaches the caller; the private window title does not.
    assert result["capture_admission"]["reason"] == "private_foreground"
    # Assert the STRUCTURED refusal, not its prose. This asserted the exact
    # sentence "foreground is private" and broke when the message was reworded
    # to "screen capture refused because private content is visible" — the
    # behaviour never changed. capture_admission.reason above is the contract;
    # the message only has to tell the person it was a privacy refusal.
    assert "private" in str(result["error"]).lower()
    assert "refused" in str(result["error"]).lower()
