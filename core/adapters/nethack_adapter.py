"""core/adapters/nethack_adapter.py — the live NetHack session behind the skill.

This is the adapter ``aura_main`` registers as ``nethack_adapter`` and
``core/skills/nethack.py`` drives, so every keystroke Aura sends into a game
arrives through here.

CP126 found fourteen findings in 144 lines, and they shared one shape: the
adapter acted on the world and then described the action rather than the
outcome.

  * It answered "Destroy old game?" with ``y``, automatically, on startup. It
    had no way of knowing whose save that was. Now the save area is Aura's
    own, the player name is constrained, and the prompt is a REFUSAL unless
    the caller has explicitly asked for the save to be destroyed.
  * It wrote ``~/.nethackrc_aura`` into the human's home directory on every
    start. The session now lives entirely inside its own directory under the
    data root, and nothing in ``$HOME`` is touched.
  * It composed the executable path and player name into one shell-ish
    string. Now: a resolved, verified executable and an argv list.
  * ``_update_screen`` read once, so an observation could be a stale frame
    while the terminal still had bytes waiting. Now it drains to quiet under
    a deadline, and every observation carries the frame sequence it was read
    at.
  * Terminal EOF was logged at debug and otherwise ignored, so a dead game
    kept answering questions about its last screen. EOF now moves the
    adapter to ``dead`` and every observation says so.
  * ``stop`` went straight to ``terminate(force=True)`` without asking the
    game to save, without reaping, and without saying which happened.

Every method here blocks — pexpect reads and terminal settling are real
waits. The public API is therefore the ``*_async`` pair, which runs the
blocking body on a worker thread; the synchronous methods remain for the
challenge harness and for callers that are already off the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.NetHackAdapter")

#: Optional game dependencies. They used to be imported at module scope, so a
#: host without them could not even enumerate the adapter — a missing game
#: became a broken capability registry. Absence is now a reportable state.
try:  # pragma: no cover - exercised by whichever host lacks the package
    import pexpect
except ImportError as _exc:  # pragma: no cover
    pexpect = None  # type: ignore[assignment]
    _PEXPECT_IMPORT_ERROR: str | None = str(_exc)
else:
    _PEXPECT_IMPORT_ERROR = None

try:  # pragma: no cover
    import pyte
except ImportError as _exc:  # pragma: no cover
    pyte = None  # type: ignore[assignment]
    _PYTE_IMPORT_ERROR: str | None = str(_exc)
else:
    _PYTE_IMPORT_ERROR = None


class NetHackAdapterError(RuntimeError):
    """Base for every refusal this adapter makes."""


class NetHackUnavailable(NetHackAdapterError):
    """The game or its terminal dependencies are not installed here."""


class NetHackSessionError(NetHackAdapterError):
    """The session is in the wrong state for the request."""


class NetHackExistingSaveError(NetHackAdapterError):
    """A saved game exists and destroying it was not authorized."""


class NetHackActionRefused(NetHackAdapterError):
    """The requested keystroke is not in the action grammar."""


#: A player name becomes an argv element and a save-file name. Anything
#: outside this is refused rather than escaped.
_PLAYER_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,30}$")

#: Single keys NetHack actually reads. An action outside this set is refused,
#: because "send whatever the model produced to a live process" is not an
#: interface — it is the absence of one.
_MOVEMENT_KEYS = frozenset("hjklyubn" "HJKLYUBN")
_COMMAND_KEYS = frozenset(
    "abcdefgimnopqrstuvwxzADEFIPQRSTWXZ"  # apply, open, eat, ... (NetHack letters)
    ",.:;<>?@^_`~$*)[=\"!/&+-#"
)
_CONTROL_KEYS = frozenset({"\x1b", "\n", "\r", " ", "\t", "\x04", "\x14", "\x18"})

#: Extended ``#`` commands that may be typed as a word. Everything else is
#: refused: #quit and #wizard-anything must not be reachable by accident.
_EXTENDED_COMMANDS = frozenset(
    {
        "adjust", "chat", "dip", "enhance", "force", "invoke", "jump", "loot",
        "monster", "name", "offer", "pray", "ride", "rub", "sit", "terrain",
        "turn", "twoweapon", "untrap", "wipe",
    }
)

#: How the screen is read: keep pulling until the terminal goes quiet, so an
#: observation is the settled frame rather than the first chunk of it.
_READ_CHUNK = 10_000
_QUIET_TIMEOUT_S = 0.1
_DRAIN_DEADLINE_S = 2.0
_MAX_DRAIN_READS = 64

_SCREEN_COLUMNS = 80
_SCREEN_ROWS = 24


def support_status() -> dict[str, Any]:
    """Whether this host can run a session, and what is missing if not."""
    missing: list[str] = []
    if pexpect is None:
        missing.append(f"pexpect ({_PEXPECT_IMPORT_ERROR})")
    if pyte is None:
        missing.append(f"pyte ({_PYTE_IMPORT_ERROR})")
    return {
        "available": not missing,
        "missing": tuple(missing),
    }


@dataclass(frozen=True)
class ActionReceipt:
    """What a keystroke actually did, not what it was asked to do."""

    action_id: str
    session_id: str
    key: str
    accepted: bool
    frame_seq_before: int
    frame_seq_after: int
    screen_changed: bool
    process_state: str
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "session_id": self.session_id,
            "key": self.key,
            "accepted": self.accepted,
            "frame_seq_before": self.frame_seq_before,
            "frame_seq_after": self.frame_seq_after,
            "screen_changed": self.screen_changed,
            "process_state": self.process_state,
            "detail": self.detail,
        }


@dataclass
class _Session:
    """Everything one game owns, so stopping it releases all of it."""

    session_id: str
    directory: Path
    player: str
    child: Any
    screen: Any
    stream: Any
    frame_seq: int = 0
    last_action_id: str = ""
    state: str = "running"
    last_output: str = ""
    exit_disposition: str = ""
    started_at_unix: float = field(default_factory=lambda: time.time())


class NetHackAdapter:
    """A single, confined NetHack session.

    The adapter owns one child process at a time. Every mutation of the
    terminal state happens under ``self._lock``: the screen, the stream, the
    frame counter and the child handle were previously written from the
    lifecycle, action and observation paths with nothing between them.
    """

    def __init__(
        self,
        nethack_path: str = "/opt/homebrew/bin/nethack",
        *,
        state_root: Path | str | None = None,
    ) -> None:
        self.nethack_path = str(nethack_path)
        self._state_root = Path(state_root) if state_root is not None else None
        self._lock = checked_lock("nethack_adapter")
        self._session: _Session | None = None

    # ── availability ────────────────────────────────────────────────────

    def _resolved_executable(self) -> Path:
        """The game binary, verified before anything is spawned.

        The path arrives from a caller and used to be interpolated straight
        into a command string. A path that is missing, is a directory, or is
        not executable is a refusal here rather than an opaque pexpect
        failure three lines later.
        """
        candidate = Path(self.nethack_path).expanduser()
        if not candidate.is_absolute():
            found = shutil.which(str(candidate))
            if not found:
                raise NetHackUnavailable(f"nethack executable not found: {candidate}")
            candidate = Path(found)
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise NetHackUnavailable(f"nethack executable is not a file: {candidate}")
        if not os.access(candidate, os.X_OK):
            raise NetHackUnavailable(f"nethack executable is not executable: {candidate}")
        return candidate

    def _session_root(self) -> Path:
        if self._state_root is not None:
            root = self._state_root
        else:
            from core.utils.engine_support import data_root

            root = data_root("nethack_sessions")
        # Through the gateway, like every other consequential filesystem
        # mutation: a raw mkdir here is a private directory nobody governs.
        with local_internal_governed_scope(
            "adapters.nethack_adapter.session_root", domain="state_mutation"
        ):
            get_file_write_gateway().ensure_directory(
                root, source="adapters.nethack_adapter.session_root"
            )
        return root

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(
        self,
        name: str = "Aura",
        *,
        destroy_existing_save: bool = False,
        replace_running_session: bool = False,
    ) -> dict[str, Any]:
        """Begin a session, refusing rather than guessing.

        ``destroy_existing_save`` has to be asked for. The old code answered
        the "Destroy old game?" prompt with ``y`` on its own, which is an
        irreversible action taken on a file it could not attribute. The save
        area is now Aura's own directory, so the only save that can be there
        is one of hers — and even then, destroying it is the caller's
        decision to make and to record.
        """
        status = support_status()
        if not status["available"]:
            raise NetHackUnavailable(
                "nethack session support unavailable: " + ", ".join(status["missing"])
            )
        if not _PLAYER_NAME_RE.match(str(name)):
            raise NetHackSessionError(
                f"player name must match {_PLAYER_NAME_RE.pattern!r}, got {name!r}"
            )

        executable = self._resolved_executable()

        with self._lock:
            existing = self._session
            if existing is not None and self._session_alive(existing):
                if not replace_running_session:
                    # The old start() overwrote self.child and orphaned the
                    # previous process, its pty and its save.
                    raise NetHackSessionError(
                        f"session {existing.session_id} is still running; "
                        "stop it or pass replace_running_session=True"
                    )
                self._stop_locked(existing, reason="replaced_by_new_session")

            session = self._spawn_locked(executable, str(name))
            self._session = session

            self._drain_locked(session)
            screen_text = self._screen_text_locked(session)
            if "Destroy old game?" in screen_text:
                if not destroy_existing_save:
                    # save=False deliberately. The clean-quit ladder answers
                    # its own confirmation prompt with "y", and at THIS prompt
                    # "y" means destroy the save — the refusal path would have
                    # done the exact thing it is refusing to do.
                    self._stop_locked(
                        session, reason="refused_to_destroy_save", save=False
                    )
                    self._session = None
                    raise NetHackExistingSaveError(
                        f"a saved game exists for player {name!r} in "
                        f"{session.directory}; pass destroy_existing_save=True "
                        "to discard it"
                    )
                logger.info(
                    "Destroying saved game for %s at explicit request (session %s)",
                    name,
                    session.session_id,
                )
                self._send_key_locked(session, "y")
                self._drain_locked(session)
                screen_text = self._screen_text_locked(session)

            return {
                "session_id": session.session_id,
                "player": session.player,
                "directory": str(session.directory),
                "process_state": session.state,
                "frame_seq": session.frame_seq,
                "destroyed_existing_save": destroy_existing_save,
            }

    def _spawn_locked(self, executable: Path, name: str) -> _Session:
        session_id = uuid.uuid4().hex[:12]
        directory = self._session_root() / f"session-{session_id}"
        rc_path = directory / "nethackrc"
        with local_internal_governed_scope(
            "adapters.nethack_adapter.rc", domain="state_mutation"
        ):
            get_file_write_gateway().ensure_directory(
                directory, source="adapters.nethack_adapter.session"
            )
            get_file_write_gateway().write_text(
                rc_path,
                "\n".join(
                    [
                        "OPTIONS=color,autoquiver,autopickup,hitpointbar,"
                        "showexp,time,statuslines:2",
                        "OPTIONS=pettype:none",
                        "OPTIONS=pickup_types:$",
                    ]
                )
                + "\n",
                source="adapters.nethack_adapter.rc",
            )

        env = os.environ.copy()
        env["TERM"] = "xterm-256color"
        env["NETHACKOPTIONS"] = f"@{rc_path}"
        # The session owns its whole world. Nothing under the human's HOME is
        # read or written, so no save of theirs can be the one at the
        # "Destroy old game?" prompt.
        env["HOME"] = str(directory)

        logger.info("Spawning nethack %s -u %s (session %s)", executable, name, session_id)
        child = pexpect.spawn(
            str(executable),
            ["-u", name],
            env=env,
            encoding="utf-8",
            timeout=_DRAIN_DEADLINE_S,
            cwd=str(directory),
        )
        child.setwinsize(_SCREEN_ROWS, _SCREEN_COLUMNS)

        screen = pyte.Screen(_SCREEN_COLUMNS, _SCREEN_ROWS)
        return _Session(
            session_id=session_id,
            directory=directory,
            player=name,
            child=child,
            screen=screen,
            stream=pyte.Stream(screen),
        )

    @staticmethod
    def _session_alive(session: _Session) -> bool:
        child = session.child
        if child is None:
            return False
        try:
            return bool(child.isalive())
        except (OSError, ValueError):
            return False

    def is_alive(self) -> bool:
        """Whether a live child is attached.

        Returns a real bool. It used to be ``self.child and
        self.child.isalive()``, which hands back ``None`` when there is no
        child — callers annotated for bool were reading truthiness instead of
        a contract.
        """
        with self._lock:
            session = self._session
            return bool(session is not None and self._session_alive(session))

    # ── terminal reading ────────────────────────────────────────────────

    def _drain_locked(self, session: _Session) -> int:
        """Read until the terminal goes quiet, and say how many frames landed.

        A single bounded read returns the first chunk and leaves the rest in
        the pty, so a caller could act on half a frame and an observation
        could describe a screen the game had already replaced.
        """
        deadline = time.monotonic() + _DRAIN_DEADLINE_S
        reads = 0
        while reads < _MAX_DRAIN_READS and time.monotonic() < deadline:
            try:
                chunk = session.child.read_nonblocking(
                    size=_READ_CHUNK, timeout=_QUIET_TIMEOUT_S
                )
            except pexpect.TIMEOUT:
                break
            except pexpect.EOF as exc:
                self._mark_dead_locked(session, "terminal_eof", exc)
                break
            except (OSError, ValueError) as exc:
                self._mark_dead_locked(session, "terminal_read_failed", exc)
                break
            if not chunk:
                break
            session.last_output = chunk
            session.stream.feed(chunk)
            session.frame_seq += 1
            reads += 1
        return reads

    def _mark_dead_locked(self, session: _Session, reason: str, exc: BaseException) -> None:
        """EOF is the game ending, not a quiet moment.

        It used to be logged at debug and swallowed, so the adapter kept
        answering questions about a screen belonging to a process that no
        longer existed.
        """
        if session.state == "dead":
            return
        session.state = "dead"
        session.exit_disposition = reason
        record_degradation(
            "nethack_adapter",
            exc if isinstance(exc, Exception) else RuntimeError(reason),
            action=f"session {session.session_id} marked dead ({reason})",
        )
        logger.info("NetHack session %s is dead: %s", session.session_id, reason)

    @staticmethod
    def _screen_text_locked(session: _Session) -> str:
        return "\n".join(session.screen.display)

    def get_screen_text(self) -> str:
        with self._lock:
            session = self._require_session_locked()
            self._drain_locked(session)
            return self._screen_text_locked(session)

    def _require_session_locked(self) -> _Session:
        session = self._session
        if session is None:
            raise NetHackSessionError("no NetHack session has been started")
        return session

    def get_observation(self) -> dict:
        """The current screen, with enough receipt to place it in a causal chain.

        It used to carry text, a timestamp and the string "nethack" — nothing
        that said which session produced it, which frame it was, whether the
        process was still alive, or which action preceded it. Two identical
        observations from different games were indistinguishable.
        """
        with self._lock:
            session = self._require_session_locked()
            self._drain_locked(session)
            text = self._screen_text_locked(session)
            return {
                "text": text,
                "metadata": {
                    "timestamp": time.time(),
                    "source": "nethack",
                    "session_id": session.session_id,
                    "player": session.player,
                    "frame_seq": session.frame_seq,
                    "last_action_id": session.last_action_id,
                    "process_state": session.state
                    if self._session_alive(session)
                    else "dead",
                    "screen_columns": _SCREEN_COLUMNS,
                    "screen_rows": _SCREEN_ROWS,
                },
            }

    # ── actions ─────────────────────────────────────────────────────────

    @staticmethod
    def resolve_action(action: str) -> str:
        """Turn a requested action into the exact bytes to send, or refuse.

        ``send_action`` used to forward whatever string it was handed to a
        live process. The grammar is narrow on purpose: one key NetHack
        actually reads, or one extended ``#`` command from a known set.
        """
        if not isinstance(action, str) or not action:
            raise NetHackActionRefused("action must be a non-empty string")
        if action.startswith("#"):
            word = action[1:].strip().lower()
            if word not in _EXTENDED_COMMANDS:
                raise NetHackActionRefused(f"extended command not permitted: {action!r}")
            return f"#{word}\n"
        if len(action) != 1:
            raise NetHackActionRefused(
                f"action must be one key or a #command, got {action!r}"
            )
        if action in _CONTROL_KEYS or action in _MOVEMENT_KEYS or action in _COMMAND_KEYS:
            return action
        raise NetHackActionRefused(f"key not in the NetHack action grammar: {action!r}")

    def _send_key_locked(self, session: _Session, payload: str) -> None:
        session.child.send(payload)

    def send_action(self, action: str) -> dict[str, Any]:
        """Send one keystroke and return what it did.

        Returns a receipt rather than ``None``: the caller can see whether
        the key was accepted, which frames bracket it, and whether the screen
        actually moved.
        """
        payload = self.resolve_action(action)
        action_id = uuid.uuid4().hex[:12]

        with self._lock:
            session = self._require_session_locked()
            if not self._session_alive(session):
                self._mark_dead_locked(
                    session,
                    "process_not_alive",
                    NetHackSessionError("child exited before the action was sent"),
                )
                return ActionReceipt(
                    action_id=action_id,
                    session_id=session.session_id,
                    key=action,
                    accepted=False,
                    frame_seq_before=session.frame_seq,
                    frame_seq_after=session.frame_seq,
                    screen_changed=False,
                    process_state="dead",
                    detail="the game process is not running",
                ).as_dict()

            before_seq = session.frame_seq
            before_text = self._screen_text_locked(session)
            session.last_action_id = action_id

            try:
                self._send_key_locked(session, payload)
            except (OSError, ValueError) as exc:
                self._mark_dead_locked(session, "terminal_write_failed", exc)
                return ActionReceipt(
                    action_id=action_id,
                    session_id=session.session_id,
                    key=action,
                    accepted=False,
                    frame_seq_before=before_seq,
                    frame_seq_after=session.frame_seq,
                    screen_changed=False,
                    process_state="dead",
                    detail=f"write failed: {type(exc).__name__}",
                ).as_dict()

            self._drain_locked(session)
            screen = self._screen_text_locked(session)

            # A blocking prompt is cleared through the same drain path. The
            # old code called read_nonblocking directly here, outside any
            # TIMEOUT/EOF handling, so the ordinary "no further output" case
            # raised out of an action that had already succeeded.
            if any(
                marker in screen
                for marker in ("--More--", "Hit return to continue", "Press return")
            ):
                logger.debug("Clearing NetHack prompt in session %s", session.session_id)
                try:
                    session.child.sendline("")
                except (OSError, ValueError) as exc:
                    self._mark_dead_locked(session, "prompt_clear_failed", exc)
                else:
                    self._drain_locked(session)
                    screen = self._screen_text_locked(session)

            return ActionReceipt(
                action_id=action_id,
                session_id=session.session_id,
                key=action,
                accepted=True,
                frame_seq_before=before_seq,
                frame_seq_after=session.frame_seq,
                screen_changed=screen != before_text,
                process_state=session.state,
                detail="",
            ).as_dict()

    # ── shutdown ────────────────────────────────────────────────────────

    def stop(self, *, save: bool = True, deadline_s: float = 5.0) -> dict[str, Any]:
        """End the session and say what happened to the game.

        The old stop() called ``terminate(force=True)`` immediately, never
        reaped, never checked the process had actually died, and never told
        anyone whether the game had been saved or left mid-write.
        """
        with self._lock:
            session = self._session
            if session is None:
                return {"stopped": False, "detail": "no session", "disposition": "absent"}
            result = self._stop_locked(session, reason="requested", save=save, deadline_s=deadline_s)
            self._session = None
            return result

    def _stop_locked(
        self,
        session: _Session,
        *,
        reason: str,
        save: bool = True,
        deadline_s: float = 5.0,
    ) -> dict[str, Any]:
        disposition = "already_exited"
        if self._session_alive(session):
            deadline = time.monotonic() + max(0.5, deadline_s)
            if save:
                # #quit, then confirm. NetHack writes the save on its way out;
                # killing it here is what leaves a half-written file behind.
                for payload in ("\x1b", "#quit\n", "y", "\n"):
                    try:
                        session.child.send(payload)
                    except (OSError, ValueError):
                        break
                    self._drain_locked(session)
                    if not self._session_alive(session):
                        disposition = "saved_and_exited"
                        break
            if self._session_alive(session) and time.monotonic() < deadline:
                try:
                    session.child.terminate(force=False)
                except (OSError, ValueError) as exc:
                    logger.debug("SIGTERM to session %s failed: %s", session.session_id, exc)
                disposition = "terminated"
            while self._session_alive(session) and time.monotonic() < deadline:
                time.sleep(0.05)
            if self._session_alive(session):
                try:
                    session.child.terminate(force=True)
                    disposition = "killed_state_unknown"
                except (OSError, ValueError) as exc:
                    logger.error(
                        "Could not force-terminate session %s: %s", session.session_id, exc
                    )
                    disposition = "termination_failed"

        try:
            session.child.close(force=False)
        except (OSError, ValueError) as exc:
            logger.debug("Closing session %s pty failed: %s", session.session_id, exc)

        reaped = not self._session_alive(session)
        if not reaped:
            record_degradation(
                "nethack_adapter",
                NetHackSessionError(f"session {session.session_id} would not die"),
                action="nethack child left running after stop",
            )
        session.state = "dead" if reaped else "unreaped"
        session.exit_disposition = disposition

        logger.info(
            "NetHack session %s stopped (%s, %s, reaped=%s)",
            session.session_id,
            reason,
            disposition,
            reaped,
        )
        return {
            "stopped": True,
            "session_id": session.session_id,
            "reason": reason,
            "disposition": disposition,
            "reaped": reaped,
            "exit_status": getattr(session.child, "exitstatus", None),
            "directory": str(session.directory),
        }

    def shutdown(self) -> dict[str, Any]:
        """Alias for stop() to support different cleanup protocols."""
        return self.stop()

    def close(self) -> dict[str, Any]:
        """Legacy close method."""
        return self.stop()

    # ── async surface ───────────────────────────────────────────────────
    #
    # Everything above blocks: pexpect reads wait, terminals settle, a child
    # takes time to die. The skill that drives this adapter runs on the
    # conversation loop, where a one-second sleep is a one-second stall for
    # every other turn in flight.

    async def start_async(self, name: str = "Aura", **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(lambda: self.start(name, **kwargs))

    async def send_action_async(self, action: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.send_action, action)

    async def get_observation_async(self) -> dict:
        return await asyncio.to_thread(self.get_observation)

    async def get_screen_text_async(self) -> str:
        return await asyncio.to_thread(self.get_screen_text)

    async def stop_async(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(lambda: self.stop(**kwargs))
