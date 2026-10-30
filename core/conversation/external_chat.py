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
import concurrent.futures
import logging
import os
import platform
import queue
import re
import secrets
import shlex
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.conversation.external_chat_ipc import (
    DurableChannelSpool,
    ExternalChatAuthenticationError,
    ExternalChatLaunchError,
    FrameCodec,
    channel_secret,
    create_private_channel_directory,
    terminal_client_source,
)
from core.conversation.session_scope import (
    LOCAL_OWNER_CONVERSATION_ID,
    conversation_session_scope,
)
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.runtime.wall_clock import monotonic_time
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
    concurrent.futures.CancelledError,
    queue.Empty,
)
_WINDOW_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
_EXTERNAL_CHAT_ORIGIN = "external"


async def _run_external_turn(orchestrator: Any, message: str) -> str | None:
    """Run one local external-surface turn and return its owned reply."""

    processor = getattr(orchestrator, "process_user_input_priority", None)
    if not callable(processor):
        raise ExternalChatLaunchError(
            "external chat orchestrator has no foreground processing entry point"
        )
    with conversation_session_scope(LOCAL_OWNER_CONVERSATION_ID):
        response = await processor(message, origin=_EXTERNAL_CHAT_ORIGIN)
    if response == "":
        raise RuntimeError("external chat foreground processing returned an empty failure")
    if response is None:
        return None
    return str(response)


def _spawn_detached(
    command: list[str],
    *,
    source: str = "core.conversation.external_chat.spawn_detached",
) -> int:
    if not command:
        raise ValueError("external chat launch command cannot be empty")
    process = get_subprocess_gateway().spawn(
        command,
        env=os.environ.copy(),
        start_new_session=True,
        source=source,
        accelerator_capability="auto",
    )
    return int(process.pid)


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

    def __init__(self, window_id: str, orchestrator, *, runtime_root: Path | None = None):
        if not _WINDOW_ID_RE.fullmatch(window_id):
            raise ValueError("external chat window_id contains unsupported characters")
        self.window_id = window_id
        self.orchestrator = orchestrator
        self.channel_dir = create_private_channel_directory(runtime_root)
        self._secret = channel_secret()
        self._codec = FrameCodec(channel_id=window_id, secret=self._secret)
        self._spool = DurableChannelSpool(self.channel_dir, self._codec)

        # Communication queues
        self.incoming_queue = queue.Queue()  # User → Aura

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

        try:
            script_path = self._create_chat_script()
            system = platform.system()
            if system == "Linux":
                for term_cmd in self._linux_terminal_commands(script_path):
                    if shutil.which(term_cmd[0]):
                        self.process = _spawn_detached(
                            term_cmd,
                            source="core.conversation.external_chat.terminal_linux_launch",
                        )
                        break
                if self.process is None:
                    raise FileNotFoundError("no supported Linux terminal emulator found")

            elif system == "Darwin":  # macOS
                command = shlex.join([sys.executable, str(script_path)])
                script_for_apple = _escape_applescript_string(command)
                apple_script = f"""
tell application "Terminal"
    do script "{script_for_apple}"
    activate
end tell
"""
                self.process = _spawn_detached(
                    ["osascript", "-e", apple_script],
                    source="core.conversation.external_chat.terminal_macos_launch",
                )

            elif system == "Windows":
                self.process = _spawn_detached(
                    ["cmd", "/c", "start", "", sys.executable, str(script_path)],
                    source="core.conversation.external_chat.terminal_windows_launch",
                )
            else:
                raise RuntimeError(f"unsupported terminal platform: {system}")

            if not self._wait_for_launch_ack(timeout_s=10.0):
                raise ExternalChatLaunchError(
                    f"external chat client did not acknowledge launch: {self.window_id}"
                )
            self.active = True
            self._start_message_handler()
            if initial_message:
                self.send_message(initial_message)
            logger.info("✅ Terminal window opened: %s", self.window_id)
        except _RECOVERABLE_EXTERNAL_CHAT_ERRORS as exc:
            record_degradation("external_chat", exc)
            logger.error("Failed to open terminal: %s", exc)
            self.close()
            if isinstance(exc, ExternalChatLaunchError):
                raise
            raise ExternalChatLaunchError(
                f"failed to launch terminal chat {self.window_id}: {exc}"
            ) from exc

    @staticmethod
    def _linux_terminal_commands(script_path: Path) -> list[list[str]]:
        return [
            ["gnome-terminal", "--", sys.executable, str(script_path)],
            ["xterm", "-e", sys.executable, str(script_path)],
            ["konsole", "-e", sys.executable, str(script_path)],
            ["xfce4-terminal", "-e", sys.executable, str(script_path)],
        ]

    def _create_chat_script(self) -> Path:
        """Create the private authenticated terminal client."""

        file_gateway = get_file_write_gateway()
        script_path = self.channel_dir / "client.py"
        file_gateway.write_text(
            script_path,
            terminal_client_source(
                channel_id=self.window_id,
                secret=self._secret,
                channel_dir=self.channel_dir,
            ),
            encoding="utf-8",
            source="core.conversation.external_chat.chat_script",
        )
        script_path.chmod(0o700)

        return script_path

    def _wait_for_launch_ack(self, *, timeout_s: float) -> bool:
        deadline = monotonic_time() + max(0.1, timeout_s)
        while monotonic_time() < deadline:
            try:
                if self._spool.client_ready():
                    return True
            except (ExternalChatAuthenticationError, OSError, ValueError) as exc:
                record_degradation("external_chat.launch_ack", exc)
                return False
            time.sleep(0.05)
        return False

    def _start_message_handler(self) -> None:
        """Start background task to handle messages."""
        handler = self._message_handler_loop()
        try:
            self.handler_task = get_task_tracker().create_task(
                handler,
                name=f"external_chat.message_handler.{self.window_id}",
            )
        except _RECOVERABLE_EXTERNAL_CHAT_ERRORS:
            handler.close()
            raise
        if self.handler_task is None:
            handler.close()
            raise ExternalChatLaunchError(
                f"external chat handler was not admitted: {self.window_id}"
            )

    async def _message_handler_loop(self):
        """Handle signed inbound frames and durable outbound acknowledgements."""
        while self.active:
            try:
                await asyncio.to_thread(self._spool.acknowledge_outbound)
                inbound = await asyncio.to_thread(self._spool.pending_inbound)
                for frame in inbound:
                    if await asyncio.to_thread(
                        self._spool.inbound_completed,
                        frame.message_id,
                    ):
                        await asyncio.to_thread(
                            self._spool.acknowledge_inbound,
                            frame.message_id,
                        )
                        continue
                    response = await _run_external_turn(self.orchestrator, frame.text)
                    self.incoming_queue.put(
                        ChatMessage(
                            speaker="user",
                            text=frame.text,
                            timestamp=time.time(),
                            window_id=self.window_id,
                        )
                    )
                    await asyncio.to_thread(
                        self._spool.complete_inbound,
                        frame.message_id,
                        response_text=response,
                    )

                await asyncio.sleep(0.1)

            except _RECOVERABLE_EXTERNAL_CHAT_ERRORS as exc:
                record_degradation("external_chat", exc)
                logger.error("Message handler error: %s", exc)
                await asyncio.sleep(1)

    def send_message(self, text: str, *, message_id: str = "") -> str:
        """Send message from Aura to user in this window.

        Args:
            text: What Aura wants to say

        """
        if not self.active:
            raise ExternalChatLaunchError(f"external chat window is not active: {self.window_id}")
        message_id = self._spool.enqueue_outbound(text, message_id=message_id)

        # Store in conversation history
        msg = ChatMessage(
            speaker="aura",
            text=text,
            timestamp=time.time(),
            window_id=self.window_id,
        )

        # Add to orchestrator's history
        if hasattr(self.orchestrator, "conversation_history"):
            history = msg.to_dict()
            history.update({"message_id": message_id, "delivery_state": "pending_ack"})
            self.orchestrator.conversation_history.append(history)
        return message_id

    def close(self) -> None:
        """Close the chat window"""
        self.active = False
        handler = self.handler_task
        if handler is not None and not handler.done():
            handler.cancel()

        try:
            if self.channel_dir.exists():
                shutil.rmtree(self.channel_dir)
        except _RECOVERABLE_EXTERNAL_CHAT_ERRORS as exc:
            record_degradation("external_chat", exc)
            logger.debug("External chat cleanup skipped %s: %s", self.channel_dir, exc)
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
        self._launch_ready = threading.Event()
        self._launch_error: BaseException | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None

        logger.info("✓ GUI Chat Window created: %s", window_id)

    def open(self, initial_message: str | None = None) -> None:
        """Open GUI chat window"""
        logger.info("🪟 Opening GUI chat window: %s", self.window_id)

        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            candidate = getattr(self.orchestrator, "loop", None)
            if candidate is None or not candidate.is_running():
                raise ExternalChatLaunchError(
                    f"GUI chat has no live orchestrator event loop: {self.window_id}"
                ) from None
            self._event_loop = candidate
        self.active = True
        thread = threading.Thread(
            target=self._create_gui,
            args=(initial_message,),
            daemon=True,
        )
        thread.start()
        if not self._launch_ready.wait(timeout=5.0):
            self.active = False
            raise ExternalChatLaunchError(
                f"GUI chat did not acknowledge launch: {self.window_id}"
            )
        if self._launch_error is not None:
            self.active = False
            raise ExternalChatLaunchError(
                f"GUI chat failed to launch: {self.window_id}: {self._launch_error}"
            ) from self._launch_error

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
                    self._schedule_user_message(text)

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
                aura_text: str | None = None
                try:
                    drained = 0
                    while drained < 32:
                        aura_text = self.outgoing_queue.get_nowait()
                        drained += 1

                        chat_display.config(state=tk.NORMAL)
                        chat_display.insert(tk.END, f"AURA: {aura_text}\n\n")
                        chat_display.config(state=tk.DISABLED)
                        chat_display.see(tk.END)
                        aura_text = None
                except queue.Empty:
                    logger.debug("GUI outgoing queue is empty")
                except _RECOVERABLE_EXTERNAL_CHAT_ERRORS as exc:
                    if aura_text is not None:
                        self.outgoing_queue.put(aura_text)
                    record_degradation("external_chat", exc)
                    logger.debug("GUI outgoing pump failed: %s", exc, exc_info=True)

                # Schedule next check
                if self.active:
                    root.after(25 if not self.outgoing_queue.empty() else 100, check_outgoing)

            # Start checking for messages
            root.after(100, check_outgoing)

            # Store window reference
            self.window = root
            self._launch_ready.set()

            # Run GUI
            root.mainloop()

            # Cleanup when closed
            self.active = False

        except _RECOVERABLE_EXTERNAL_CHAT_ERRORS as exc:
            self._launch_error = exc
            self._launch_ready.set()
            self.active = False
            record_degradation("external_chat", exc)
            logger.error("GUI creation failed: %s", exc)

    def _schedule_user_message(self, message: str) -> None:
        """Submit a GUI turn to the owner loop and route its exact reply back."""

        loop = self._event_loop
        if loop is None or not loop.is_running():
            raise ExternalChatLaunchError(
                f"external chat owner loop is unavailable: {self.window_id}"
            )
        future = asyncio.run_coroutine_threadsafe(
            _run_external_turn(self.orchestrator, message),
            loop,
        )

        def _deliver_reply(completed) -> None:
            try:
                response = completed.result()
                if response:
                    self.send_message(response)
            except _RECOVERABLE_EXTERNAL_CHAT_ERRORS as exc:
                record_degradation("external_chat", exc)
                logger.error("External GUI turn failed: %s", exc)

        future.add_done_callback(_deliver_reply)

    def send_message(self, text: str) -> str:
        """Send message from Aura to user"""
        if not self.active:
            raise ExternalChatLaunchError(f"external chat window is not active: {self.window_id}")
        message_id = secrets.token_hex(16)
        self.outgoing_queue.put(text)

        # Store in history
        if hasattr(self.orchestrator, "conversation_history"):
            self.orchestrator.conversation_history.append(
                {
                    "timestamp": time.time(),
                    "source": f"external_window_{self.window_id}",
                    "speaker": "aura",
                    "message": text,
                    "message_id": message_id,
                    "delivery_state": "queued_for_gui",
                }
            )
        return message_id

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

    def __init__(self, orchestrator, *, runtime_root: Path | None = None):
        self.orchestrator = orchestrator
        self.runtime_root = create_private_channel_directory(runtime_root)

        # Track windows
        self.windows: dict[str, Any] = {}
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
        window_type = window_type or self.preferred_window_type
        if window_type not in {"gui", "terminal"}:
            raise ValueError(f"unsupported external chat window type: {window_type}")
        window_id = f"chat_{secrets.token_hex(12)}"

        logger.info("🪟 Opening %s chat window: %s", window_type, window_id)

        # Create window
        if window_type == "gui":
            window = GUIChatWindow(window_id, self.orchestrator)
        else:
            window = TerminalChatWindow(
                window_id,
                self.orchestrator,
                runtime_root=self.runtime_root,
            )

        try:
            window.open(message)
        except ExternalChatLaunchError:
            window.close()
            if window_type != "gui":
                raise
            logger.warning("GUI external chat failed; trying authenticated terminal surface")
            window = TerminalChatWindow(
                window_id,
                self.orchestrator,
                runtime_root=self.runtime_root,
            )
            window.open(message)

        if not window.active:
            window.close()
            raise ExternalChatLaunchError(
                f"external chat launch returned without an active surface: {window_id}"
            )
        self.windows[window_id] = window

        return window_id

    def send_to_window(self, window_id: str, message: str) -> str:
        """Send message to specific window"""
        window = self.windows.get(window_id)
        if window is None or not window.active:
            raise KeyError(f"active external chat window not found: {window_id}")
        result = window.send_message(message)
        return str(result or "")

    def broadcast(self, message: str) -> dict[str, str]:
        """Send message to all open windows"""
        receipts: dict[str, str] = {}
        for window in self.windows.values():
            if window.active:
                receipts[window.window_id] = str(window.send_message(message) or "")
        return receipts

    def close_window(self, window_id: str):
        """Close specific window"""
        if window_id in self.windows:
            self.windows[window_id].close()
            del self.windows[window_id]

    def close_all_windows(self):
        """Close all external windows"""
        for window_id in list(self.windows.keys()):
            self.close_window(window_id)

    def shutdown(self) -> None:
        """Canonical synchronous shutdown hook for every owned surface."""

        self.close_all_windows()
        try:
            if self.runtime_root.exists():
                shutil.rmtree(self.runtime_root)
        except _RECOVERABLE_EXTERNAL_CHAT_ERRORS as exc:
            record_degradation("external_chat.shutdown", exc)
            logger.warning("External chat runtime cleanup failed: %s", exc)

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
