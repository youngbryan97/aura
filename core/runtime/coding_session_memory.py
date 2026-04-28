from __future__ import annotations
from core.runtime.errors import record_degradation


import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from core.config import config
from core.runtime.turn_analysis import TurnAnalysis, analyze_turn, canonical_turn_text
from core.utils.file_utils import atomic_write_json

logger = logging.getLogger("Aura.CodingSessionMemory")

_MAX_EXCHANGES = 4
_MAX_EVENTS = 8
_MAX_FILES = 8
_MAX_COMMANDS = 6
_MAX_PLAN_STEPS = 5
_MAX_SUMMARY_CHARS = 220
_FILE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?:\.{0,2}/|/)?"
    r"[A-Za-z0-9_.~/-]+\.(?:"
    r"py|md|json|toml|ya?ml|txt|ini|cfg|js|jsx|ts|tsx|sh|bash|zsh|swift|java|rs|go|c|cc|cpp|h|hpp"
    r")"
    r"(?::\d+)?"
)
_CODING_MARKERS = (
    "code",
    "coding",
    "debug",
    "bug",
    "traceback",
    "stack trace",
    "stacktrace",
    "pytest",
    "test",
    "tests",
    "compile",
    "build",
    "refactor",
    "implement",
    "function",
    "module",
    "repo",
    "repository",
    "file",
    "files",
    "path",
    "patch",
    "git",
    "terminal",
    "shell",
    "command",
    "exception",
    "error",
    "llm",
    "mlx",
    "model",
)
_TEST_MARKERS = ("pytest", "unittest", "nose", "test", "tests")
_CONTINUATION_MARKERS = (
    "keep going",
    "keep it going",
    "continue",
    "resume",
    "let's do it",
    "lets do it",
    "do it",
    "go ahead",
    "finish it",
    "fix it",
    "patch it",
)


def _normalize_text(text: Any, *, max_len: int = _MAX_SUMMARY_CHARS) -> str:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return ""
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 16].rstrip() + "...[truncated]"


def _first_signal_line(*values: Any) -> str:
    for value in values:
        text = str(value or "")
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line in {"...", "Traceback (most recent call last):"}:
                continue
            return _normalize_text(line, max_len=160)
    return ""


def _is_probably_coding_text(text: str, analysis: TurnAnalysis | None = None) -> bool:
    normalized = canonical_turn_text(text)
    if not normalized:
        return False
    current_analysis = analysis or analyze_turn(normalized)
    lowered = normalized.lower()
    if current_analysis.semantic_mode == "technical":
        return True
    if current_analysis.intent_type == "TASK" and any(marker in lowered for marker in _CODING_MARKERS):
        return True
    return any(marker in lowered for marker in _CODING_MARKERS)


def _extract_file_candidates(value: Any) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    def _push(candidate: Any) -> None:
        text = str(candidate or "").strip()
        if not text or "://" in text:
            return
        normalized = text.split(":", 1)[0] if re.search(r":[0-9]+$", text) else text
        normalized = normalized.strip()
        if (
            not normalized
            or normalized in seen
            or normalized.startswith("-")
            or normalized.lower() in {"true", "false", "none"}
        ):
            return
        seen.add(normalized)
        found.append(normalized)

    def _walk(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, dict):
            for key, nested in item.items():
                if key in {"path", "paths", "file", "files", "target", "cwd", "new_cwd"}:
                    _walk(nested)
                elif key in {"stdout", "stderr", "summary", "message", "error", "content", "result", "command"}:
                    _walk(nested)
            return
        if isinstance(item, (list, tuple, set)):
            for nested in item:
                _walk(nested)
            return
        text = str(item or "")
        for match in _FILE_PATTERN.findall(text):
            _push(match)

    _walk(value)
    return found[:_MAX_FILES]


def _looks_like_test_command(command: str) -> bool:
    lowered = str(command or "").lower()
    return any(marker in lowered for marker in _TEST_MARKERS)


def _looks_like_continuation_text(text: str) -> bool:
    lowered = str(text or "").lower()
    if not lowered:
        return False
    return any(marker in lowered for marker in _CONTINUATION_MARKERS)


@dataclass
class CodingExchange:
    objective: str
    user_summary: str
    assistant_summary: str
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CodingExchange":
        return cls(
            objective=str(payload.get("objective", "") or ""),
            user_summary=str(payload.get("user_summary", "") or ""),
            assistant_summary=str(payload.get("assistant_summary", "") or ""),
            timestamp=float(payload.get("timestamp", time.time()) or time.time()),
        )


@dataclass
class ToolEvent:
    tool_name: str
    summary: str
    success: bool
    origin: str = ""
    command: str = ""
    files: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToolEvent":
        return cls(
            tool_name=str(payload.get("tool_name", "") or ""),
            summary=str(payload.get("summary", "") or ""),
            success=bool(payload.get("success", False)),
            origin=str(payload.get("origin", "") or ""),
            command=str(payload.get("command", "") or ""),
            files=list(payload.get("files") or []),
            timestamp=float(payload.get("timestamp", time.time()) or time.time()),
        )


@dataclass
class ExecutionLoopState:
    plan_id: str = ""
    goal: str = ""
    phase: str = ""
    active_step: str = ""
    plan_steps: list[str] = field(default_factory=list)
    verification_summary: str = ""
    repair_summary: str = ""
    last_result_summary: str = ""
    steps_completed: int = 0
    steps_total: int = 0
    verification_failures: int = 0
    repair_count: int = 0
    updated_at: float = field(default_factory=time.time)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExecutionLoopState":
        return cls(
            plan_id=str(payload.get("plan_id", "") or ""),
            goal=str(payload.get("goal", "") or ""),
            phase=str(payload.get("phase", "") or ""),
            active_step=str(payload.get("active_step", "") or ""),
            plan_steps=[str(item) for item in list(payload.get("plan_steps") or [])[:_MAX_PLAN_STEPS]],
            verification_summary=str(payload.get("verification_summary", "") or ""),
            repair_summary=str(payload.get("repair_summary", "") or ""),
            last_result_summary=str(payload.get("last_result_summary", "") or ""),
            steps_completed=int(payload.get("steps_completed", 0) or 0),
            steps_total=int(payload.get("steps_total", 0) or 0),
            verification_failures=int(payload.get("verification_failures", 0) or 0),
            repair_count=int(payload.get("repair_count", 0) or 0),
            updated_at=float(payload.get("updated_at", time.time()) or time.time()),
        )


class CodingSessionMemory:
    """Compact, coding-specific continuity for technical turns."""

    def __init__(self, persist_path: Path | None = None):
        base_dir = config.paths.data_dir / "runtime"
        self.persist_path = Path(persist_path or (base_dir / "coding_session_memory.json"))
        self._lock = RLock()
        self._exchanges: list[CodingExchange] = []
        self._events: list[ToolEvent] = []
        self._recent_files: list[str] = []
        self._recent_commands: list[str] = []
        self._last_objective: str = ""
        self._last_test_failure: str = ""
        self._last_runtime_error: str = ""
        self._execution = ExecutionLoopState()
        self._updated_at: float = 0.0
        self._load()

    @staticmethod
    def is_coding_request(text: str, analysis: TurnAnalysis | None = None) -> bool:
        return _is_probably_coding_text(text, analysis)

    def clear(self) -> None:
        with self._lock:
            self._exchanges = []
            self._events = []
            self._recent_files = []
            self._recent_commands = []
            self._last_objective = ""
            self._last_test_failure = ""
            self._last_runtime_error = ""
            self._execution = ExecutionLoopState(updated_at=time.time())
            self._updated_at = time.time()
            self._save_locked()

    def record_conversation_turn(
        self,
        user_input: str,
        aura_response: str,
        *,
        analysis: TurnAnalysis | None = None,
    ) -> None:
        current_analysis = analysis or analyze_turn(user_input)
        if not self.is_coding_request(user_input, current_analysis):
            return

        user_summary = _normalize_text(canonical_turn_text(user_input))
        assistant_summary = _normalize_text(aura_response)
        if not user_summary or not assistant_summary:
            return

        exchange = CodingExchange(
            objective=user_summary,
            user_summary=user_summary,
            assistant_summary=assistant_summary,
        )

        with self._lock:
            self._exchanges.append(exchange)
            self._exchanges = self._exchanges[-_MAX_EXCHANGES:]
            self._last_objective = user_summary
            self._updated_at = time.time()
            self._save_locked()

    def record_tool_event(
        self,
        *,
        tool_name: str,
        args: dict[str, Any] | None,
        result: Any,
        objective: str = "",
        origin: str = "",
        success: bool | None = None,
        error: str = "",
    ) -> None:
        payload = result if isinstance(result, dict) else {"result": result}
        tool = str(tool_name or "").strip()
        command = _normalize_text((args or {}).get("command", ""), max_len=140)
        current_objective = _normalize_text(objective, max_len=180)
        files = _extract_file_candidates(args or {})
        for path in _extract_file_candidates(payload):
            if path not in files:
                files.append(path)
        files = files[:_MAX_FILES]

        tool_is_coding = tool in {
            "sovereign_terminal",
            "run_code",
            "active_coding",
            "file_operation",
            "browser",
            "web_search",
            "read_file",
            "write_file",
            "edit_file",
        }
        relevant = bool(files or command or tool_is_coding or self.is_coding_request(current_objective))
        if not relevant:
            return

        event_success = bool(payload.get("ok", success if success is not None else True))
        summary = self._build_event_summary(tool, command, payload, event_success, error=error)
        if not summary:
            return

        event = ToolEvent(
            tool_name=tool,
            summary=summary,
            success=event_success,
            origin=str(origin or ""),
            command=command,
            files=files,
        )

        with self._lock:
            self._events.append(event)
            self._events = self._events[-_MAX_EVENTS:]
            self._remember_recent_items(self._recent_files, files, max_items=_MAX_FILES)
            if command:
                self._remember_recent_items(self._recent_commands, [command], max_items=_MAX_COMMANDS)
            if current_objective:
                self._last_objective = current_objective
            self._update_failure_state(command, payload, event_success, error=error)
            self._updated_at = time.time()
            self._save_locked()

    def record_execution_plan(
        self,
        *,
        goal: str,
        steps: list[Any],
        plan_id: str = "",
        objective: str = "",
    ) -> None:
        normalized_goal = _normalize_text(goal, max_len=180)
        if not self.is_coding_request(normalized_goal):
            return

        rendered_steps: list[str] = []
        for step in list(steps or [])[:_MAX_PLAN_STEPS]:
            if isinstance(step, str):
                rendered = _normalize_text(step, max_len=140)
            else:
                rendered = _normalize_text(getattr(step, "description", "") or step, max_len=140)
            if rendered:
                rendered_steps.append(rendered)

        with self._lock:
            self._execution = ExecutionLoopState(
                plan_id=str(plan_id or ""),
                goal=normalized_goal,
                phase="planning",
                active_step=rendered_steps[0] if rendered_steps else "",
                plan_steps=rendered_steps,
                steps_completed=0,
                steps_total=len(list(steps or [])),
                updated_at=time.time(),
            )
            if objective:
                self._last_objective = _normalize_text(objective, max_len=180)
            elif normalized_goal:
                self._last_objective = normalized_goal
            self._updated_at = time.time()
            self._save_locked()

    def record_execution_step(
        self,
        *,
        step_description: str,
        tool_name: str = "",
        status: str,
        attempt: int = 0,
        result_summary: str = "",
        error: str = "",
        success_criterion: str = "",
        steps_completed: int | None = None,
        steps_total: int | None = None,
    ) -> None:
        rendered_step = _normalize_text(step_description, max_len=160)
        if not rendered_step:
            return

        detail = _first_signal_line(result_summary, error, success_criterion)
        with self._lock:
            if not self._execution.goal:
                self._execution.goal = self._last_objective or rendered_step
            self._execution.active_step = rendered_step
            if steps_total is not None:
                self._execution.steps_total = max(0, int(steps_total or 0))
            if steps_completed is not None:
                self._execution.steps_completed = max(0, int(steps_completed or 0))

            status_text = str(status or "").strip().lower()
            if status_text in {"running", "executing"}:
                self._execution.phase = "executing"
                if detail:
                    self._execution.last_result_summary = _normalize_text(
                        f"{tool_name or 'step'}: {detail}",
                        max_len=180,
                    )
            elif status_text in {"verifying"}:
                self._execution.phase = "verifying"
                criterion = _normalize_text(success_criterion, max_len=160)
                if criterion:
                    self._execution.verification_summary = criterion
            elif status_text in {"verified", "succeeded", "completed"}:
                self._execution.phase = "executing"
                self._execution.verification_summary = ""
                if detail:
                    self._execution.last_result_summary = _normalize_text(detail, max_len=180)
            elif status_text in {"verification_failed", "failed", "timeout"}:
                self._execution.phase = "repairing"
                self._execution.verification_failures += 1
                failure = rendered_step
                if detail:
                    failure = f"{failure} -> {detail}"
                if attempt:
                    failure = f"{failure} (attempt {attempt})"
                self._execution.verification_summary = _normalize_text(failure, max_len=200)

            self._execution.updated_at = time.time()
            self._updated_at = time.time()
            self._save_locked()

    def record_execution_repair(
        self,
        *,
        step_description: str,
        reason: str = "",
        new_args: dict[str, Any] | None = None,
    ) -> None:
        rendered_step = _normalize_text(step_description, max_len=160)
        detail = _first_signal_line(reason, new_args)
        with self._lock:
            self._execution.phase = "repairing"
            self._execution.active_step = rendered_step or self._execution.active_step
            self._execution.repair_count += 1
            summary = rendered_step or self._execution.active_step or "repair loop"
            if detail:
                summary = f"{summary}: {detail}"
            self._execution.repair_summary = _normalize_text(summary, max_len=200)
            self._execution.updated_at = time.time()
            self._updated_at = time.time()
            self._save_locked()

    def record_execution_result(
        self,
        *,
        summary: str,
        succeeded: bool,
        steps_completed: int = 0,
        steps_total: int = 0,
    ) -> None:
        with self._lock:
            self._execution.phase = "completed" if succeeded else "failed"
            self._execution.active_step = ""
            self._execution.last_result_summary = _normalize_text(summary, max_len=200)
            self._execution.steps_completed = max(0, int(steps_completed or self._execution.steps_completed))
            self._execution.steps_total = max(0, int(steps_total or self._execution.steps_total))
            self._execution.updated_at = time.time()
            self._updated_at = time.time()
            self._save_locked()

    def build_context_block(self, objective: str = "") -> str:
        current_objective = _normalize_text(objective, max_len=180)
        with self._lock:
            if not self.is_coding_request(current_objective):
                if not (self._last_objective and _looks_like_continuation_text(current_objective)):
                    return ""
            lines = ["## CODING WORKING SET"]
            if current_objective:
                lines.append(f"- Active technical objective: {current_objective}")
            if self._last_objective and self._last_objective != current_objective:
                lines.append(f"- Recent coding thread: {self._last_objective}")

            latest_exchange = self._exchanges[-1] if self._exchanges else None
            if latest_exchange is not None:
                lines.append(f"- Recent user ask: {latest_exchange.user_summary}")
                lines.append(f"- Recent assistant direction: {latest_exchange.assistant_summary}")

            if self._recent_files:
                lines.append(f"- Files in play: {', '.join(self._recent_files[-5:])}")
            if self._recent_commands:
                lines.append(f"- Recent commands: {' | '.join(self._recent_commands[-3:])}")
            if self._execution.goal:
                progress = ""
                if self._execution.steps_total:
                    progress = f" ({self._execution.steps_completed}/{self._execution.steps_total} steps)"
                lines.append(
                    f"- Execution loop: phase={self._execution.phase or 'idle'}{progress} — {self._execution.goal}"
                )
                if self._execution.plan_steps:
                    lines.append(f"- Plan spine: {' | '.join(self._execution.plan_steps[:_MAX_PLAN_STEPS])}")
                if self._execution.active_step:
                    lines.append(f"- Current step: {self._execution.active_step}")
                if self._execution.verification_summary:
                    lines.append(f"- Verification pressure: {self._execution.verification_summary}")
                if self._execution.repair_summary:
                    lines.append(f"- Repair loop: {self._execution.repair_summary}")
                if self._execution.last_result_summary:
                    lines.append(f"- Latest execution result: {self._execution.last_result_summary}")
            if self._last_test_failure:
                lines.append(f"- Last failing test/run: {self._last_test_failure}")
            elif self._last_runtime_error:
                lines.append(f"- Last runtime error: {self._last_runtime_error}")

            recent_events = [event.summary for event in self._events[-3:] if event.summary]
            for summary in recent_events:
                lines.append(f"- Tool result: {summary}")

            lines.append(
                "Use this working set to continue the engineering thread without re-deriving already known context."
            )
            return "\n".join(lines)

    def get_route_hints(self, objective: str = "") -> dict[str, Any]:
        current_objective = _normalize_text(objective, max_len=180)
        with self._lock:
            continuation = bool(self._last_objective and _looks_like_continuation_text(current_objective))
            return {
                "coding_request": self.is_coding_request(current_objective) or continuation,
                "active_coding_thread": bool(self._last_objective or self._events or self._exchanges),
                "recent_file_count": len(self._recent_files),
                "recent_command_count": len(self._recent_commands),
                "has_test_failure": bool(self._last_test_failure),
                "has_runtime_error": bool(self._last_runtime_error),
                "has_active_plan": bool(self._execution.goal and self._execution.phase not in {"completed", "failed", ""}),
                "has_verification_failure": bool(self._execution.verification_summary),
                "repair_attempts": int(self._execution.repair_count or 0),
                "execution_phase": self._execution.phase,
                "last_objective": self._last_objective,
            }

    def _load(self) -> None:
        try:
            if not self.persist_path.exists():
                return
            payload = json.loads(self.persist_path.read_text(encoding="utf-8"))
        except Exception as exc:
            record_degradation('coding_session_memory', exc)
            logger.debug("Coding session memory load skipped: %s", exc)
            return

        with self._lock:
            self._exchanges = [CodingExchange.from_dict(item) for item in list(payload.get("exchanges") or [])[-_MAX_EXCHANGES:]]
            self._events = [ToolEvent.from_dict(item) for item in list(payload.get("events") or [])[-_MAX_EVENTS:]]
            self._recent_files = list(payload.get("recent_files") or [])[-_MAX_FILES:]
            self._recent_commands = list(payload.get("recent_commands") or [])[-_MAX_COMMANDS:]
            self._last_objective = str(payload.get("last_objective", "") or "")
            self._last_test_failure = str(payload.get("last_test_failure", "") or "")
            self._last_runtime_error = str(payload.get("last_runtime_error", "") or "")
            self._execution = ExecutionLoopState.from_dict(payload.get("execution") or {})
            self._updated_at = float(payload.get("updated_at", 0.0) or 0.0)

    def _save_locked(self) -> None:
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                str(self.persist_path),
                {
                    "exchanges": [asdict(item) for item in self._exchanges],
                    "events": [asdict(item) for item in self._events],
                    "recent_files": list(self._recent_files),
                    "recent_commands": list(self._recent_commands),
                    "last_objective": self._last_objective,
                    "last_test_failure": self._last_test_failure,
                    "last_runtime_error": self._last_runtime_error,
                    "execution": asdict(self._execution),
                    "updated_at": self._updated_at,
                },
            )
        except Exception as exc:
            record_degradation('coding_session_memory', exc)
            logger.debug("Coding session memory save skipped: %s", exc)

    @staticmethod
    def _remember_recent_items(target: list[str], values: list[str], *, max_items: int) -> None:
        for value in values:
            normalized = _normalize_text(value, max_len=160)
            if not normalized:
                continue
            if normalized in target:
                target.remove(normalized)
            target.append(normalized)
            if len(target) > max_items:
                del target[0 : len(target) - max_items]

    def _build_event_summary(
        self,
        tool_name: str,
        command: str,
        payload: dict[str, Any],
        success: bool,
        *,
        error: str = "",
    ) -> str:
        status = "ok" if success else "failed"
        if command:
            signal = _first_signal_line(payload.get("stderr"), payload.get("stdout"), payload.get("error"), error)
            base = f"{tool_name}: {command} -> {status}"
            if signal:
                base = f"{base} ({signal})"
            return _normalize_text(base, max_len=200)

        explicit = _first_signal_line(
            payload.get("summary"),
            payload.get("message"),
            payload.get("error"),
            error,
            payload.get("title"),
            payload.get("result"),
            payload.get("content"),
            payload.get("stderr"),
            payload.get("stdout"),
        )
        if explicit:
            return _normalize_text(f"{tool_name}: {explicit}", max_len=200)
        return _normalize_text(f"{tool_name}: {status}", max_len=200)

    def _update_failure_state(
        self,
        command: str,
        payload: dict[str, Any],
        success: bool,
        *,
        error: str = "",
    ) -> None:
        signal = _first_signal_line(
            payload.get("error"),
            error,
            payload.get("stderr"),
            payload.get("stdout"),
        )
        if command and _looks_like_test_command(command):
            if success:
                self._last_test_failure = ""
            elif signal:
                self._last_test_failure = _normalize_text(f"{command} -> {signal}", max_len=220)
            return
        if success:
            return
        if signal:
            self._last_runtime_error = _normalize_text(signal, max_len=220)


_INSTANCE: CodingSessionMemory | None = None
_INSTANCE_LOCK = RLock()


def get_coding_session_memory() -> CodingSessionMemory:
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = CodingSessionMemory()
        return _INSTANCE


def build_coding_context_block(objective: str = "") -> str:
    return get_coding_session_memory().build_context_block(objective)


def get_coding_route_hints(objective: str = "") -> dict[str, Any]:
    return get_coding_session_memory().get_route_hints(objective)


def reset_coding_session_memory_for_tests() -> None:
    global _INSTANCE
    with _INSTANCE_LOCK:
        _INSTANCE = None
