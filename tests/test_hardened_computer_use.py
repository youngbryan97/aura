import asyncio
import pytest
from types import SimpleNamespace
from core.skills.computer_use import ComputerUseSkill, ComputerUseParams

@pytest.mark.asyncio
async def test_computer_use_read_screen_text_fallback_on_permission_block(monkeypatch):
    skill = ComputerUseSkill()

    # Mock permissions to return blocked for ACCESSIBILITY/AUTOMATION
    async def mock_require_permissions(capability, *permission_names):
        return {"ok": False, "status": "denied", "error": "mock block"}

    called_tree = False
    def mock_query_window_tree():
        nonlocal called_tree
        called_tree = True
        return "Process: Finder\n  Window: Desktop\n    Element [AXButton]: Close"

    monkeypatch.setattr(skill, "_require_permissions", mock_require_permissions)
    monkeypatch.setattr(skill, "_query_system_events_window_tree", mock_query_window_tree)

    result = await skill.execute({"action": "read_screen_text", "target": ""}, {})
    assert result["ok"] is True
    assert result["source"] == "applescript_window_tree_fallback"
    assert "Finder" in result["text"]
    assert called_tree is True

@pytest.mark.asyncio
async def test_computer_use_read_screen_text_fallback_on_unavailable(monkeypatch):
    skill = ComputerUseSkill()

    # Mock permissions to pass
    async def mock_require_permissions(capability, *permission_names):
        return None

    # Mock screen text to return unavailable error string
    def mock_read_screen_text_macos():
        return "[accessibility error or ui unresponsive]"

    called_tree = False
    def mock_query_window_tree():
        nonlocal called_tree
        called_tree = True
        return "Fallback Process tree"

    monkeypatch.setattr(skill, "_require_permissions", mock_require_permissions)
    monkeypatch.setattr(skill, "_read_screen_text_macos", mock_read_screen_text_macos)
    monkeypatch.setattr(skill, "_query_system_events_window_tree", mock_query_window_tree)

    result = await skill.execute({"action": "read_screen_text", "target": ""}, {})
    assert result["ok"] is True
    assert result["source"] == "applescript_window_tree_fallback"
    assert "Fallback Process tree" in result["text"]
    assert called_tree is True

@pytest.mark.asyncio
async def test_computer_use_click_retry_success(monkeypatch):
    skill = ComputerUseSkill()

    # Mock get_pyautogui
    class MockPyAutoGUI:
        def __init__(self):
            self.clicks = 0
        def click(self, x, y):
            self.clicks += 1

    mock_pyautogui = MockPyAutoGUI()
    monkeypatch.setattr("core.skills.computer_use.get_pyautogui", lambda: (mock_pyautogui, None))

    # Mock permissions
    async def mock_require_permissions(capability, *permission_names):
        return None
    monkeypatch.setattr(skill, "_require_permissions", mock_require_permissions)

    # Mock _read_screen_text_macos to simulate state change only on 2nd attempt
    state_counter = 0
    def mock_read_screen():
        nonlocal state_counter
        state_counter += 1
        if state_counter <= 2:
            return "State A"
        return "State B"
    monkeypatch.setattr(skill, "_read_screen_text_macos", mock_read_screen)

    # Fast forward sleep
    async def mock_sleep(secs):
        pass
    monkeypatch.setattr(asyncio, "sleep", mock_sleep)

    result = await skill.execute({"action": "click", "x": 100, "y": 200}, {})
    assert result["ok"] is True
    assert result["attempts"] == 2
    assert result["verification"] == "State shifted."
    assert mock_pyautogui.clicks == 2

@pytest.mark.asyncio
async def test_computer_use_type_pre_clicks_and_retries(monkeypatch):
    skill = ComputerUseSkill()

    # Mock get_pyautogui
    class MockPyAutoGUI:
        def __init__(self):
            self.clicks = 0
            self.typed = ""
        def click(self, x, y):
            self.clicks += 1
        def typewrite(self, text, interval):
            self.typed = text

    mock_pyautogui = MockPyAutoGUI()
    monkeypatch.setattr("core.skills.computer_use.get_pyautogui", lambda: (mock_pyautogui, None))

    # Mock permissions
    async def mock_require_permissions(capability, *permission_names):
        return None
    monkeypatch.setattr(skill, "_require_permissions", mock_require_permissions)

    # Mock _read_screen_text_macos to contain the typed text
    def mock_read_screen():
        return "Hello World! output"
    monkeypatch.setattr(skill, "_read_screen_text_macos", mock_read_screen)

    # Fast forward sleep
    async def mock_sleep(secs):
        pass
    monkeypatch.setattr(asyncio, "sleep", mock_sleep)

    result = await skill.execute({"action": "type", "target": "Hello World!", "x": 50, "y": 60}, {})
    assert result["ok"] is True
    assert result["attempts"] == 1
    assert result["verification"] == "Text confirmed on screen or state shifted."
    assert mock_pyautogui.clicks == 1
    assert mock_pyautogui.typed == "Hello World!"

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
    def mock_run(args, capture_output, text, timeout):
        nonlocal run_args
        run_args = args
        return SimpleNamespace(returncode=0, stdout="find output", stderr="")
    monkeypatch.setattr("core.skills.computer_use.subprocess.run", mock_run)

    result = await skill.execute({"action": "run_command", "target": "find . -name '*.py'"}, {})
    assert result["ok"] is True
    assert "-maxdepth" in run_args
    assert "4" in run_args
