"""External Chat Window System
Aura can open chat windows to communicate even when running in background.

CRITICAL CAPABILITIES:
- Open terminal or GUI chat windows
- Aura initiates conversation (not just responds)
- All conversations retained by core model
- Full request execution through external windows
- Multiple simultaneous windows

This allows Aura to "tap you on the shoulder" when she wants to talk.
"""

import asyncio
import errno
import logging
import os
import platform
import queue
import re
import shlex
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("ExternalChat")

_RECOVERABLE_EXTERNAL_CHAT_ERRORS = (
    AttributeError,
    FileNotFoundError,
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    queue.Empty,
)
_WINDOW_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


def _spawn_detached(command: list[str]) -> int:
    if not command:
        raise ValueError("external chat launch command cannot be empty")
    if os.name == "posix":
        return os.posix_spawnp(command[0], command, os.environ.copy())
    return os.spawnvp(os.P_NOWAIT, command[0], command)


def _escape_applescript_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


@dataclass
class ChatMessage:
    """A message in external chat"""

    speaker: str  # "aura" or "user"
    text: str
    timestamp: float
    window_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "text": self.text,
            "timestamp": self.timestamp,
            "window_id": self.window_id,
        }


class TerminalChatWindow:
    """Terminal-based chat window.

    Fast, simple, works on any system.
    Opens in a new terminal window.
    """

    def __init__(self, window_id: str, orchestrator):
        if not _WINDOW_ID_RE.fullmatch(window_id):
            raise ValueError("external chat window_id contains unsupported characters")
        self.window_id = window_id
        self.orchestrator = orchestrator

        # Communication queues
        self.incoming_queue = queue.Queue()  # User → Aura
        self.outgoing_queue = queue.Queue()  # Aura → User

        # State
        self.active = False
        self.process: int | None = None
        self.handler_task = None

        logger.info("✓ Terminal Chat Window created: %s", window_id)

    def open(self, initial_message: str | None = None) -> None:
        """Open terminal chat window.

        Args:
            initial_message: What Aura says when opening the window

        """
        logger.info("📟 Opening terminal chat window: %s", self.window_id)

        script_path = self._create_chat_script(initial_message)
        system = platform.system()

        try:
            if system == "Linux":
                for term_cmd in self._linux_terminal_commands(script_path):
                    if shutil.which(term_cmd[0]):
                        self.process = _spawn_detached(term_cmd)
                        break
                if self.process is None:
                    raise FileNotFoundError("no supported Linux terminal emulator found")

            elif system == "Darwin":  # macOS
                script_for_apple = _escape_applescript_string(str(script_path))
                apple_script = f"""
tell application "Terminal"
    do script "bash {script_for_apple}"
    activate
end tell
"""
                self.process = _spawn_detached(["osascript", "-e", apple_script])

            elif system == "Windows":
                self.process = _spawn_detached(
                    ["cmd", "/c", "start", "cmd", "/k", "bash", str(script_path)]
                )
            else:
                raise RuntimeError(f"unsupported terminal platform: {system}")

            self.active = True
            self._start_message_handler()
            logger.info("✅ Terminal window opened: %s", self.window_id)

        except _RECOVERABLE_EXTERNAL_CHAT_ERRORS as exc:
            record_degradation("external_chat", exc)
            logger.error("Failed to open terminal: %s", exc)

    @staticmethod
    def _linux_terminal_commands(script_path: Path) -> list[list[str]]:
        return [
            ["gnome-terminal", "--", "bash", str(script_path)],
            ["xterm", "-e", "bash", str(script_path)],
            ["konsole", "-e", "bash", str(script_path)],
            ["xfce4-terminal", "-e", "bash", str(script_path)],
        ]

    def _create_chat_script(self, initial_message: str | None) -> Path:
        """Create bash script for chat interface"""
        chat_dir = Path(tempfile.gettempdir()) / "aura_chat"
        chat_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        pipe_in = chat_dir / f"{self.window_id}_in"
        pipe_out = chat_dir / f"{self.window_id}_out"
        message_file = chat_dir / f"{self.window_id}_initial.txt"
        if initial_message:
            message_file.write_text(initial_message, encoding="utf-8")
        else:
            message_file.write_text("", encoding="utf-8")

        script = f'''#!/bin/bash
set -u

# Chat window for Aura
WINDOW_ID={shlex.quote(self.window_id)}
PIPE_IN={shlex.quote(str(pipe_in))}
PIPE_OUT={shlex.quote(str(pipe_out))}
INITIAL_MESSAGE_FILE={shlex.quote(str(message_file))}

# Create named pipes
mkfifo "$PIPE_IN" 2>/dev/null || true
mkfifo "$PIPE_OUT" 2>/dev/null || true

# Display header
clear
echo "=================================="
echo "    AURA CHAT - {self.window_id}"
echo "=================================="
echo ""

# Initial message from Aura
if [ -s "$INITIAL_MESSAGE_FILE" ]; then
    printf "AURA: "
    cat "$INITIAL_MESSAGE_FILE"
    echo ""
    echo ""
fi

# Chat loop
while true; do
    # Check for Aura's messages
    if [ -p "$PIPE_OUT" ]; then
        if read -t 0.1 line < "$PIPE_OUT"; then
            echo "AURA: $line"
        fi
    fi

    # Prompt for user input
    read -t 0.1 -p "YOU: " user_input
    if [ $? -eq 0 ]; then
        # Send to Aura
        if [ -n "$user_input" ]; then
            echo "$user_input" > "$PIPE_IN"
        fi

        # Check for exit
        if [ "$user_input" = "exit" ] || [ "$user_input" = "quit" ]; then
            echo "Closing chat window..."
            break
        fi
    fi
done

# Cleanup
rm -f "$PIPE_IN" "$PIPE_OUT" "$INITIAL_MESSAGE_FILE"
'''

        script_path = chat_dir / f"{self.window_id}.sh"
        script_path.write_text(script, encoding="utf-8")
        script_path.chmod(0o700)

        return script_path

    def _start_message_handler(self) -> None:
        """Start background task to handle messages."""
        try:
            loop = asyncio.get_running_loop()
            self.handler_task = loop.create_task(self._message_handler_loop())
        except RuntimeError:
            # No running loop — use ensure_future to schedule for when one starts
            self.handler_task = get_task_tracker().track(self._message_handler_loop())

    async def _message_handler_loop(self):
        """Handle bidirectional communication"""
        chat_dir = Path(tempfile.gettempdir()) / "aura_chat"
        pipe_in = chat_dir / f"{self.window_id}_in"
        pipe_out = chat_dir / f"{self.window_id}_out"

        while self.active:
            try:
                for user_msg in await asyncio.to_thread(self._read_pipe_messages, pipe_in):
                    self.incoming_queue.put(
                        ChatMessage(
                            speaker="user",
                            text=user_msg,
                            timestamp=time.time(),
                            window_id=self.window_id,
                        )
                    )
                    self._process_user_message(user_msg)

                if not self.outgoing_queue.empty():
                    aura_msg = self.outgoing_queue.get()
                    if await asyncio.to_thread(pipe_out.exists):
                        await asyncio.to_thread(self._write_pipe_message, pipe_out, aura_msg)

                await asyncio.sleep(0.1)

            except _RECOVERABLE_EXTERNAL_CHAT_ERRORS as exc:
                record_degradation("external_chat", exc)
                logger.error("Message handler error: %s", exc)
                await asyncio.sleep(1)

    @staticmethod
    def _read_pipe_messages(pipe_in: Path) -> list[str]:
        if not pipe_in.exists():
            return []
        try:
            fd = os.open(pipe_in, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as exc:
            if exc.errno in {errno.ENXIO, errno.ENOENT}:
                return []
            raise
        try:
            data = os.read(fd, 65536)
        finally:
            os.close(fd)
        if not data:
            return []
        return [
            line.strip()
            for line in data.decode("utf-8", errors="replace").splitlines()
            if line.strip()
        ]

    @staticmethod
    def _write_pipe_message(pipe_out: Path, message: str) -> None:
        try:
            fd = os.open(pipe_out, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as exc:
            if exc.errno in {errno.ENXIO, errno.ENOENT}:
                return
            raise
        try:
            os.write(fd, f"{message}\n".encode("utf-8", errors="replace"))
        finally:
            os.close(fd)

    def _process_user_message(self, message: str) -> None:
        """Process user message through orchestrator.

        This is CRITICAL - allows full request execution through external window.
        """
        try:
            # Add to orchestrator's message queue
            # Add to orchestrator's message queue via threadsafe method
            if hasattr(self.orchestrator, "enqueue_from_thread"):
                self.orchestrator.enqueue_from_thread(
                    message, origin=f"external_window_{self.window_id}"
                )

            # Store in orchestrator's conversation history
            if hasattr(self.orchestrator, "conversation_history"):
                self.orchestrator.conversation_history.append(
                    {
                        "timestamp": time.time(),
                        "source": f"external_window_{self.window_id}",
                        "speaker": "user",
                        "message": message,
                    }
                )

        except _RECOVERABLE_EXTERNAL_CHAT_ERRORS as exc:
            record_degradation("external_chat", exc)
            logger.error("Failed to process message through orchestrator: %s", exc)

    def send_message(self, text: str) -> None:
        """Send message from Aura to user in this window.

        Args:
            text: What Aura wants to say

        """
        self.outgoing_queue.put(text)

        # Store in conversation history
        msg = ChatMessage(
            speaker="aura",
            text=text,
            timestamp=time.time(),
            window_id=self.window_id,
        )

        # Add to orchestrator's history
        if hasattr(self.orchestrator, "conversation_history"):
            self.orchestrator.conversation_history.append(msg.to_dict())

    def close(self) -> None:
        """Close the chat window"""
        self.active = False

        # Clean up pipes
        chat_dir = Path(tempfile.gettempdir()) / "aura_chat"
        for pipe in [
            chat_dir / f"{self.window_id}_in",
            chat_dir / f"{self.window_id}_out",
            chat_dir / f"{self.window_id}_initial.txt",
            chat_dir / f"{self.window_id}.sh",
        ]:
            if pipe.exists():
                try:
                    pipe.unlink()
                except _RECOVERABLE_EXTERNAL_CHAT_ERRORS as exc:
                    record_degradation("external_chat", exc)
                    logger.debug("External chat cleanup skipped %s: %s", pipe, exc)
        logger.info("✅ Terminal window closed: %s", self.window_id)


class GUIChatWindow:
    """Simple GUI chat window using tkinter.

    Better UX than terminal, still simple and fast.
    """

    def __init__(self, window_id: str, orchestrator):
        if not _WINDOW_ID_RE.fullmatch(window_id):
            raise ValueError("external chat window_id contains unsupported characters")
        self.window_id = window_id
        self.orchestrator = orchestrator

        # Communication
        self.incoming_queue = queue.Queue()
        self.outgoing_queue = queue.Queue()

        # State
        self.active = False
        self.window = None

        logger.info("✓ GUI Chat Window created: %s", window_id)

    def open(self, initial_message: str | None = None) -> None:
        """Open GUI chat window"""
        logger.info("🪟 Opening GUI chat window: %s", self.window_id)

        # Start GUI in separate thread
        thread = threading.Thread(
            target=self._create_gui,
            args=(initial_message,),
            daemon=True,
        )
        thread.start()

        self.active = True

    def _create_gui(self, initial_message: str | None) -> None:
        """Create tkinter GUI"""
        try:
            import tkinter as tk
            from tkinter import scrolledtext

            # Create window
            root = tk.Tk()
            root.title(f"Aura Chat - {self.window_id}")
            root.geometry("500x600")

            # Chat display
            chat_display = scrolledtext.ScrolledText(
                root,
                wrap=tk.WORD,
                width=60,
                height=30,
                font=("Arial", 10),
            )
            chat_display.pack(padx=10, pady=10)
            chat_display.config(state=tk.DISABLED)

            # Show initial message
            if initial_message:
                chat_display.config(state=tk.NORMAL)
                chat_display.insert(tk.END, f"AURA: {initial_message}\n\n")
                chat_display.config(state=tk.DISABLED)

            # Input field
            input_frame = tk.Frame(root)
            input_frame.pack(padx=10, pady=5, fill=tk.X)

            input_field = tk.Entry(input_frame, font=("Arial", 10))
            input_field.pack(side=tk.LEFT, fill=tk.X, expand=True)

            def send_message() -> None:
                """Send user message"""
                text = input_field.get().strip()
                if text:
                    # Display in chat
                    chat_display.config(state=tk.NORMAL)
                    chat_display.insert(tk.END, f"YOU: {text}\n")
                    chat_display.config(state=tk.DISABLED)
                    chat_display.see(tk.END)

                    # Clear input
                    input_field.delete(0, tk.END)

                    # Process message
                    msg = ChatMessage(
                        speaker="user",
                        text=text,
                        timestamp=time.time(),
                        window_id=self.window_id,
                    )

                    self.incoming_queue.put(msg)
                    self._process_user_message(text)

            # Send button
            send_btn = tk.Button(
                input_frame,
                text="Send",
                command=send_message,
                font=("Arial", 10),
            )
            send_btn.pack(side=tk.RIGHT, padx=5)

            # Bind Enter key
            input_field.bind("<Return>", lambda e: send_message())

            # Check for Aura's messages
            def check_outgoing() -> None:
                """Check for messages from Aura"""
                try:
                    drained = 0
                    while drained < 32:
                        aura_text = self.outgoing_queue.get_nowait()
                        drained += 1

                        chat_display.config(state=tk.NORMAL)
                        chat_display.insert(tk.END, f"AURA: {aura_text}\n\n")
                        chat_display.config(state=tk.DISABLED)
                        chat_display.see(tk.END)
                except queue.Empty:
                    logger.debug("GUI outgoing queue is empty")
                except _RECOVERABLE_EXTERNAL_CHAT_ERRORS as exc:
                    record_degradation("external_chat", exc)
                    logger.debug("GUI outgoing pump failed: %s", exc, exc_info=True)

                # Schedule next check
                if self.active:
                    root.after(25 if not self.outgoing_queue.empty() else 100, check_outgoing)

            # Start checking for messages
            root.after(100, check_outgoing)

            # Store window reference
            self.window = root

            # Run GUI
            root.mainloop()

            # Cleanup when closed
            self.active = False

        except _RECOVERABLE_EXTERNAL_CHAT_ERRORS as exc:
            record_degradation("external_chat", exc)
            logger.error("GUI creation failed: %s", exc)
            logger.info("Falling back to terminal window")
            # Could fallback to terminal window here

    def _process_user_message(self, message: str) -> None:
        """Process user message through orchestrator"""
        try:
            # Same as terminal window - full integration
            # Add to orchestrator's message queue via threadsafe method
            if hasattr(self.orchestrator, "enqueue_from_thread"):
                self.orchestrator.enqueue_from_thread(
                    message, origin=f"external_window_{self.window_id}"
                )

            if hasattr(self.orchestrator, "conversation_history"):
                self.orchestrator.conversation_history.append(
                    {
                        "timestamp": time.time(),
                        "source": f"external_window_{self.window_id}",
                        "speaker": "user",
                        "message": message,
                    }
                )

        except _RECOVERABLE_EXTERNAL_CHAT_ERRORS as exc:
            record_degradation("external_chat", exc)
            logger.error("Failed to process message: %s", exc)

    def send_message(self, text: str) -> None:
        """Send message from Aura to user"""
        self.outgoing_queue.put(text)

        # Store in history
        if hasattr(self.orchestrator, "conversation_history"):
            self.orchestrator.conversation_history.append(
                {
                    "timestamp": time.time(),
                    "source": f"external_window_{self.window_id}",
                    "speaker": "aura",
                    "message": text,
                }
            )

    def close(self) -> None:
        """Close GUI window"""
        self.active = False
        if self.window:
            try:
                self.window.after(0, self.window.destroy)
            except _RECOVERABLE_EXTERNAL_CHAT_ERRORS as exc:
                record_degradation("external_chat", exc)
                logger.debug("GUI close scheduling failed: %s", exc)


class ExternalChatManager:
    """Manages all external chat windows.

    Aura uses this to initiate conversations with the user.
    """

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

        # Track windows
        self.windows: dict[str, Any] = {}
        self.next_window_id = 1

        # Preferences
        self.preferred_window_type = "gui"  # "gui" or "terminal"

        logger.info("✓ External Chat Manager initialized")

    def open_chat_window(
        self,
        message: str | None = None,
        window_type: str | None = None,
    ) -> str:
        """Open a new chat window.

        Args:
            message: Initial message from Aura
            window_type: "gui" or "terminal"

        Returns:
            Window ID

        """
        window_id = f"chat_{self.next_window_id}"
        self.next_window_id += 1

        window_type = window_type or self.preferred_window_type

        logger.info("🪟 Opening %s chat window: %s", window_type, window_id)

        # Create window
        if window_type == "gui":
            window = GUIChatWindow(window_id, self.orchestrator)
        else:
            window = TerminalChatWindow(window_id, self.orchestrator)

        # Open it
        window.open(message)

        # Store reference
        self.windows[window_id] = window

        return window_id

    def send_to_window(self, window_id: str, message: str):
        """Send message to specific window"""
        if window_id in self.windows:
            self.windows[window_id].send_message(message)
        else:
            logger.error("Window %s not found", window_id)

    def broadcast(self, message: str):
        """Send message to all open windows"""
        for window in self.windows.values():
            if window.active:
                window.send_message(message)

    def close_window(self, window_id: str):
        """Close specific window"""
        if window_id in self.windows:
            self.windows[window_id].close()
            del self.windows[window_id]

    def close_all_windows(self):
        """Close all external windows"""
        for window_id in list(self.windows.keys()):
            self.close_window(window_id)

    def get_active_windows(self) -> list:
        """Get list of active window IDs"""
        return [wid for wid, w in self.windows.items() if w.active]


def integrate_external_chat(orchestrator):
    """Integrate external chat capability into orchestrator.

    After this, Aura can:
    - Open chat windows from background
    - Initiate conversations with user
    - Process requests through external windows
    """
    # Initialize chat manager
    orchestrator.external_chat = ExternalChatManager(orchestrator)

    # Add conversation history if not present
    if not hasattr(orchestrator, "conversation_history"):
        orchestrator.conversation_history = []

    # Hook response delivery to also send to external windows
    # Check if there's a method to hook into
    # orchestrator usually just prints or returns.
    # We might need to monkey patch or rely on orchestrator calling this explicitly.

    logger.info("✅ External chat integrated")
    logger.info("   Aura can now open chat windows and initiate conversations")
