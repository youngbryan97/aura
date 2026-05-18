from types import SimpleNamespace

import pytest

from core import external_chat


def test_terminal_chat_rejects_unsafe_window_id():
    with pytest.raises(ValueError):
        external_chat.TerminalChatWindow("chat;bad", SimpleNamespace())


def test_terminal_script_stores_initial_message_outside_shell_body():
    window = external_chat.TerminalChatWindow("chat_safe", SimpleNamespace())
    initial = 'hello; touch "not-executed"'

    script_path = window._create_chat_script(initial)
    script = script_path.read_text(encoding="utf-8")
    message_file = script_path.with_name("chat_safe_initial.txt")

    assert initial not in script
    assert message_file.read_text(encoding="utf-8") == initial
    assert "INITIAL_MESSAGE_FILE=" in script

    window.close()


def test_linux_terminal_launch_uses_argument_vector(monkeypatch):
    launched = []
    window = external_chat.TerminalChatWindow("chat_launch", SimpleNamespace())

    monkeypatch.setattr(external_chat.platform, "system", lambda: "Linux")
    monkeypatch.setattr(external_chat.shutil, "which", lambda name: f"/usr/bin/{name}" if name == "xterm" else None)
    monkeypatch.setattr(external_chat, "_spawn_detached", lambda command: launched.append(command) or 1234)
    monkeypatch.setattr(window, "_start_message_handler", lambda: None)

    window.open("hello")

    assert window.active is True
    assert window.process == 1234
    assert launched[0][0:3] == ["xterm", "-e", "bash"]
    assert launched[0][3].endswith("chat_launch.sh")

    window.close()


def test_pipe_helpers_read_and_write_regular_files(tmp_path):
    inbound = tmp_path / "in"
    inbound.write_text("first\n\nsecond\n", encoding="utf-8")

    assert external_chat.TerminalChatWindow._read_pipe_messages(inbound) == ["first", "second"]

    outbound = tmp_path / "out"
    outbound.touch()
    external_chat.TerminalChatWindow._write_pipe_message(outbound, "aura says hi")

    assert outbound.read_text(encoding="utf-8") == "aura says hi\n"


def test_terminal_user_message_reaches_orchestrator_history():
    calls = []
    orchestrator = SimpleNamespace(
        conversation_history=[],
        enqueue_from_thread=lambda message, origin: calls.append((message, origin)),
    )
    window = external_chat.TerminalChatWindow("chat_history", orchestrator)

    window._process_user_message("please inspect status")

    assert calls == [("please inspect status", "external_window_chat_history")]
    assert orchestrator.conversation_history[-1]["message"] == "please inspect status"
