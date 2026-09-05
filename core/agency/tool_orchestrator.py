"""core/agency/tool_orchestrator.py

Asynchronous Tool Execution Environment.
Grants Aura the ability to run Python scripts and search the web to resolve
knowledge gaps dynamically.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from core.governance_context import local_internal_governed_scope
from core.runtime.errors import (
    DependencyUnavailable,
    TimeoutBudgetExceeded,
    record_degradation,
)
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.network_gateway import get_network_gateway
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.ToolOrchestrator")

_SANDBOX_PROTOCOL_VERSION = 1
_MAX_REQUEST_BYTES = 256 * 1024
_MAX_CODE_BYTES = 120 * 1024
_MAX_RESPONSE_BYTES = 1024 * 1024
_MAX_STDERR_TAIL_BYTES = 16 * 1024
_MAX_STDERR_STREAM_BYTES = 1024 * 1024
_FRAME_HEADER = struct.Struct("!I")


def _sandbox_effect_scope(operation: str):
    return local_internal_governed_scope(
        f"tool_orchestrator.sandbox.{operation}",
        domain="tool_execution",
        constraints={
            "effect": "ephemeral_sandbox_filesystem",
            "sandbox_only": True,
        },
    )


class SandboxTransportError(RuntimeError):
    """The sandbox transport failed independently of generated-code correctness."""


class ToolOrchestrator:
    def __init__(self):
        root = Path(tempfile.gettempdir()) / f"aura_sandbox_{os.getuid()}"
        self._control_dir = root / "control"
        self.sandbox_dir = root / "work"
        with _sandbox_effect_scope("bootstrap"):
            gateway = get_file_write_gateway()
            gateway.ensure_directory(root, source="tool_orchestrator.sandbox_root")
            gateway.ensure_directory(
                self._control_dir,
                source="tool_orchestrator.sandbox_control",
            )
            gateway.ensure_directory(
                self.sandbox_dir,
                source="tool_orchestrator.sandbox_work",
            )
        self.execution_timeout = 15.0
        self.startup_timeout = 2.0
        self.shutdown_timeout_s = 3.0
        self._active_process: asyncio.subprocess.Process | None = None
        self._repl_lock = asyncio.Lock()
        self._transport_faults: deque[float] = deque(maxlen=16)
        self._circuit_open_until = 0.0
        self._last_python_failure_kind = ""
        self._last_transport_error = ""
        self._last_execution_latency_ms = 0.0
        self._last_startup_latency_ms = 0.0

    def _build_worker_launch_config(self, worker_path: Path) -> tuple[list[str], dict[str, str]]:
        sandbox_exec = shutil.which("sandbox-exec")
        if not sandbox_exec:
            raise DependencyUnavailable(
                "sandbox-exec is unavailable; refusing to launch an unsandboxed Python worker"
            )
        base_prefix = Path(sys.base_prefix).resolve()
        app_python = (
            base_prefix
            / "Resources"
            / "Python.app"
            / "Contents"
            / "MacOS"
            / "Python"
        )
        python_bin = str(
            app_python.resolve()
            if app_python.is_file()
            else Path(getattr(sys, "_base_executable", sys.executable)).resolve()
        )
        site_packages = (
            Path(sys.prefix)
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        ).resolve()
        policy = "\n".join(
            (
                "(version 1)",
                "(deny default)",
                '(import "system.sb")',
                "(deny network*)",
                f'(allow process-exec (literal "{python_bin}"))',
                "(deny process-fork)",
                "(allow file-read*",
                '    (subpath "/opt")',
                '    (subpath "/System")',
                '    (subpath "/usr/lib")',
                '    (subpath "/usr/share")',
                '    (subpath "/private/var/db/timezone")',
                '    (literal "/dev/null")',
                f'    (subpath "{self._control_dir.resolve()}")',
                f'    (subpath "{self.sandbox_dir.resolve()}")',
                f'    (subpath "{site_packages}")',
                ")",
                f'(allow file-write* (subpath "{self.sandbox_dir.resolve()}"))',
            )
        )
        env = {
            "HOME": str(self.sandbox_dir),
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": str(Path(python_bin).parent),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": str(self.sandbox_dir),
            "AURA_SANDBOX_SITE_PACKAGES": str(site_packages),
        }
        return [sandbox_exec, "-p", policy, python_bin, "-I", str(worker_path)], env

    async def _stage_worker(self) -> Path:
        source_path = Path(__file__).with_name("repl_daemon.py")
        if not source_path.is_file():
            raise DependencyUnavailable(f"Python sandbox worker is missing: {source_path}")
        source = await asyncio.to_thread(source_path.read_text, encoding="utf-8")
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]
        worker_path = self._control_dir / f"worker-{digest}.py"
        if not worker_path.is_file():
            with _sandbox_effect_scope("stage_worker"):
                await get_file_write_gateway().write_text_async(
                    worker_path,
                    source,
                    source="tool_orchestrator.stage_worker",
                )
        else:
            existing = await asyncio.to_thread(worker_path.read_text, encoding="utf-8")
            if not hashlib.sha256(existing.encode("utf-8")).hexdigest().startswith(digest):
                with _sandbox_effect_scope("repair_worker"):
                    await get_file_write_gateway().write_text_async(
                        worker_path,
                        source,
                        source="tool_orchestrator.repair_worker",
                    )
        return worker_path

    @staticmethod
    def _encode_frame(payload: dict[str, Any], *, max_bytes: int) -> bytes:
        body = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if not body or len(body) > max_bytes:
            raise SandboxTransportError(f"sandbox frame exceeds {max_bytes} bytes")
        return _FRAME_HEADER.pack(len(body)) + body

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("sandbox deadline exhausted")
        return remaining

    async def _read_frame(
        self,
        stream: asyncio.StreamReader,
        *,
        max_bytes: int,
        deadline: float,
    ) -> dict[str, Any]:
        header = await asyncio.wait_for(
            stream.readexactly(_FRAME_HEADER.size),
            timeout=self._remaining(deadline),
        )
        (size,) = _FRAME_HEADER.unpack(header)
        if size <= 0 or size > max_bytes:
            raise SandboxTransportError(f"invalid sandbox frame size {size}")
        body = await asyncio.wait_for(
            stream.readexactly(size),
            timeout=self._remaining(deadline),
        )
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SandboxTransportError("sandbox returned malformed UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise SandboxTransportError("sandbox frame payload is not an object")
        return payload

    @staticmethod
    async def _drain_stderr_tail(stream: asyncio.StreamReader) -> bytes:
        tail = bytearray()
        total = 0
        max_chunks = (_MAX_STDERR_STREAM_BYTES // 4096) + 2
        for _ in range(max_chunks):
            chunk = await stream.read(4096)
            if not chunk:
                return bytes(tail)
            total += len(chunk)
            if total > _MAX_STDERR_STREAM_BYTES:
                raise SandboxTransportError("sandbox worker stderr exceeded its stream budget")
            tail.extend(chunk)
            if len(tail) > _MAX_STDERR_TAIL_BYTES:
                del tail[: len(tail) - _MAX_STDERR_TAIL_BYTES]
        raise SandboxTransportError("sandbox worker stderr did not close within its chunk budget")

    @staticmethod
    async def _terminate_worker(proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            await asyncio.wait_for(proc.wait(), timeout=0.1)
            return
        process_group_id = int(getattr(proc, "_aura_process_group_id", 0) or 0)
        try:
            if process_group_id > 0 and process_group_id != os.getpgrp():
                os.killpg(process_group_id, signal.SIGTERM)
            else:
                proc.terminate()
        except (OSError, ProcessLookupError):
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=0.5)
            return
        except TimeoutError:
            pass
        try:
            if process_group_id > 0 and process_group_id != os.getpgrp():
                os.killpg(process_group_id, signal.SIGKILL)
            else:
                proc.kill()
        except (OSError, ProcessLookupError):
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except TimeoutError:
            logger.error("Sandbox worker pid=%s could not be reaped", proc.pid)

    async def _close_worker(
        self,
        proc: asyncio.subprocess.Process,
        stderr_task: asyncio.Task[bytes],
        *,
        terminate: bool,
    ) -> bytes:
        if terminate:
            await self._terminate_worker(proc)
        elif proc.returncode is None:
            try:
                await asyncio.wait_for(proc.wait(), timeout=0.5)
            except TimeoutError:
                await self._terminate_worker(proc)
        try:
            return await asyncio.wait_for(stderr_task, timeout=1.0)
        except TimeoutError:
            stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)
            return b""
        finally:
            if self._active_process is proc:
                self._active_process = None

    def _record_transport_fault(self, message: str) -> None:
        now = time.monotonic()
        self._transport_faults.append(now)
        while self._transport_faults and now - self._transport_faults[0] > 60.0:
            self._transport_faults.popleft()
        self._last_transport_error = str(message)[:500]
        if len(self._transport_faults) >= 3:
            self._circuit_open_until = max(self._circuit_open_until, now + 60.0)

    def _circuit_is_open(self) -> bool:
        return time.monotonic() < self._circuit_open_until

    async def _launch_worker(
        self,
    ) -> tuple[asyncio.subprocess.Process, asyncio.Task[bytes]]:
        worker_path = await self._stage_worker()
        argv, env = self._build_worker_launch_config(worker_path)
        proc = await get_subprocess_gateway().spawn_async(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.sandbox_dir,
            env=env,
            source="tool_execution:tool_orchestrator.python_worker",
            accelerator_capability="auto",
        )
        if proc.stdin is None or proc.stdout is None or proc.stderr is None:
            await self._terminate_worker(proc)
            raise DependencyUnavailable("sandbox worker did not expose all protocol pipes")
        self._active_process = proc
        stderr_task = get_task_tracker().create_task(
            self._drain_stderr_tail(proc.stderr),
            name=f"tool-orchestrator-stderr-{proc.pid}",
        )
        return proc, stderr_task

    @staticmethod
    def _authority_binding(request_id: str) -> str:
        from core.governance_context import get_active_governance

        token = get_active_governance()
        receipt_id = str(getattr(token, "receipt_id", "degraded_local") or "degraded_local")
        return hashlib.sha256(f"{receipt_id}\n{request_id}".encode()).hexdigest()

    @staticmethod
    def _stderr_text(stderr_tail: bytes) -> str:
        return stderr_tail.decode("utf-8", errors="replace").strip()[:_MAX_STDERR_TAIL_BYTES]

    async def _remove_validation_file(self, path: Path) -> OSError | None:
        try:
            with _sandbox_effect_scope("remove_validation_file"):
                await get_file_write_gateway().delete_path_async(
                    path,
                    source="tool_orchestrator.validation_cleanup",
                )
        except OSError as exc:
            record_degradation(
                "tool_orchestrator",
                exc,
                severity="warning",
                action="failed closed on validation temp cleanup",
            )
            return exc
        return None

    async def _spawn_ready_worker(
        self,
    ) -> tuple[asyncio.subprocess.Process, asyncio.Task[bytes]]:
        last_error: BaseException | None = None
        for attempt in range(2):
            proc: asyncio.subprocess.Process | None = None
            stderr_task: asyncio.Task[bytes] | None = None
            started = time.monotonic()
            try:
                proc, stderr_task = await self._launch_worker()
                ready = await self._read_frame(
                    proc.stdout,
                    max_bytes=_MAX_RESPONSE_BYTES,
                    deadline=time.monotonic() + self.startup_timeout,
                )
                if (
                    ready.get("version") != _SANDBOX_PROTOCOL_VERSION
                    or ready.get("kind") != "ready"
                    or int(ready.get("worker_pid") or 0) != proc.pid
                ):
                    raise SandboxTransportError("sandbox startup handshake did not match worker")
                self._last_startup_latency_ms = (time.monotonic() - started) * 1000.0
                return proc, stderr_task
            except asyncio.CancelledError:
                if proc is not None and stderr_task is not None:
                    await asyncio.shield(
                        self._close_worker(proc, stderr_task, terminate=True)
                    )
                raise
            except (
                DependencyUnavailable,
                FileNotFoundError,
                PermissionError,
                ProcessLookupError,
                BrokenPipeError,
                asyncio.IncompleteReadError,
                SandboxTransportError,
                OSError,
                TypeError,
                ValueError,
                TimeoutError,
            ) as exc:
                last_error = exc
                stderr_tail = b""
                if proc is not None and stderr_task is not None:
                    stderr_tail = await self._close_worker(
                        proc,
                        stderr_task,
                        terminate=True,
                    )
                detail = self._stderr_text(stderr_tail)
                message = f"{type(exc).__name__}: {exc}"
                if detail:
                    message += f"; stderr={detail}"
                self._record_transport_fault(message)
                if attempt == 0 and not self._circuit_is_open():
                    logger.warning("Sandbox startup failed before dispatch; retrying once: %s", message)
                    continue
                break
        raise SandboxTransportError(
            f"sandbox worker failed before dispatch: {last_error or 'unknown startup failure'}"
        )

    async def _execute_admitted_python(self, script_content: str) -> tuple[bool, str]:
        if self._circuit_is_open():
            self._last_python_failure_kind = "circuit"
            remaining = max(0.0, self._circuit_open_until - time.monotonic())
            return False, f"Sandbox transport circuit open; retry after {remaining:.1f}s."

        request_id = uuid.uuid4().hex
        authority_id = self._authority_binding(request_id)
        request = {
            "version": _SANDBOX_PROTOCOL_VERSION,
            "kind": "execute",
            "request_id": request_id,
            "authority_id": authority_id,
            "deadline_ms": int(self.execution_timeout * 1000),
            "code": script_content,
        }
        try:
            frame = self._encode_frame(request, max_bytes=_MAX_REQUEST_BYTES)
        except SandboxTransportError as exc:
            self._last_python_failure_kind = "validation"
            return False, f"Code Validation Failed: {exc}"
        proc: asyncio.subprocess.Process | None = None
        stderr_task: asyncio.Task[bytes] | None = None
        dispatched = False
        started = time.monotonic()
        try:
            proc, stderr_task = await self._spawn_ready_worker()
            deadline = time.monotonic() + self.execution_timeout
            dispatched = True
            proc.stdin.write(frame)
            await asyncio.wait_for(proc.stdin.drain(), timeout=self._remaining(deadline))
            response = await self._read_frame(
                proc.stdout,
                max_bytes=_MAX_RESPONSE_BYTES,
                deadline=deadline,
            )
            if (
                response.get("version") != _SANDBOX_PROTOCOL_VERSION
                or response.get("kind") != "result"
                or response.get("request_id") != request_id
                or response.get("authority_id") != authority_id
                or not isinstance(response.get("success"), bool)
                or not isinstance(response.get("output"), str)
            ):
                raise SandboxTransportError("sandbox response envelope failed correlation")
            stderr_tail = await self._close_worker(proc, stderr_task, terminate=False)
            if proc.returncode != 0:
                raise SandboxTransportError(
                    f"sandbox worker exited with status {proc.returncode} after response"
                )
            stderr_text = self._stderr_text(stderr_tail)
            output = str(response.get("output") or "[Empty Output]")
            if stderr_text:
                output = f"{output}\n[worker stderr]\n{stderr_text}".strip()
            self._last_execution_latency_ms = (time.monotonic() - started) * 1000.0
            self._last_python_failure_kind = "" if response["success"] else "code"
            self._last_transport_error = ""
            return bool(response["success"]), output
        except asyncio.CancelledError:
            if proc is not None and stderr_task is not None:
                await asyncio.shield(self._close_worker(proc, stderr_task, terminate=True))
            self._last_python_failure_kind = "cancelled"
            raise
        except TimeoutError as exc:
            if proc is not None and stderr_task is not None:
                await asyncio.shield(self._close_worker(proc, stderr_task, terminate=True))
            self._last_python_failure_kind = "code_timeout" if dispatched else "transport"
            self._last_execution_latency_ms = (time.monotonic() - started) * 1000.0
            timeout_error = TimeoutBudgetExceeded(
                str(exc) or "Python sandbox execution timed out"
            )
            record_degradation(
                "tool_orchestrator",
                timeout_error,
                severity="warning",
                action="reaped sandbox worker after bounded execution timeout",
            )
            if dispatched:
                return False, "Execution Error: Script exceeded the sandbox deadline."
            self._record_transport_fault(str(timeout_error))
            return False, "Sandbox transport timed out before code dispatch."
        except (
            DependencyUnavailable,
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
            BrokenPipeError,
            asyncio.IncompleteReadError,
            SandboxTransportError,
            OSError,
            UnicodeDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            stderr_tail = b""
            if proc is not None and stderr_task is not None:
                stderr_tail = await asyncio.shield(
                    self._close_worker(proc, stderr_task, terminate=True)
                )
            detail = self._stderr_text(stderr_tail)
            message = f"{type(exc).__name__}: {exc}"
            if detail:
                message += f"; stderr={detail}"
            if dispatched or not isinstance(exc, SandboxTransportError):
                self._record_transport_fault(message)
            self._last_python_failure_kind = "transport"
            self._last_execution_latency_ms = (time.monotonic() - started) * 1000.0
            record_degradation(
                "tool_orchestrator",
                exc,
                severity="critical",
                action=(
                    "reaped sandbox worker after indeterminate post-dispatch transport failure"
                    if dispatched
                    else "failed closed before sandbox code dispatch"
                ),
            )
            if dispatched:
                return (
                    False,
                    "Sandbox transport failed after dispatch; execution status is "
                    "indeterminate and the request was not replayed.",
                )
            return False, f"Sandbox transport failed before dispatch: {message}"
        finally:
            if proc is not None and self._active_process is proc:
                self._active_process = None

    async def execute_python(self, script_content: str) -> tuple[bool, str]:
        """Validate and execute one isolated Python request under a hard deadline."""
        from core.utils.code_guardian import CodeGuardian

        if not isinstance(script_content, str) or not script_content.strip():
            self._last_python_failure_kind = "validation"
            return False, "Code Validation Failed: Python source must be non-empty."
        if len(script_content.encode("utf-8")) > _MAX_CODE_BYTES:
            self._last_python_failure_kind = "validation"
            return False, "Code Validation Failed: Python source exceeds 120 KiB."

        tmp_path = self.sandbox_dir / f"validation-{uuid.uuid4().hex}.py"
        try:
            with _sandbox_effect_scope("write_validation_file"):
                await get_file_write_gateway().write_text_async(
                    tmp_path,
                    script_content,
                    source="tool_orchestrator.validation_source",
                )
            report = await asyncio.to_thread(CodeGuardian.validate_code, tmp_path)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            await self._remove_validation_file(tmp_path)
            self._last_python_failure_kind = "validation_infrastructure"
            record_degradation(
                "tool_orchestrator",
                exc,
                severity="critical",
                action="failed closed before launching python sandbox",
            )
            return False, f"Code validation setup failed: {exc}"

        cleanup_error = await self._remove_validation_file(tmp_path)
        if cleanup_error is not None:
            self._last_python_failure_kind = "validation_infrastructure"
            return False, f"Code validation cleanup failed: {cleanup_error}"
        if not report.success:
            self._last_python_failure_kind = "validation"
            logger.warning("ToolOrchestrator: CodeGuardian blocked execution.")
            error_details = report.error_message
            if report.ruff_output:
                error_details += f"\nLinter:\n{report.ruff_output}"
            return False, f"Code Validation Failed:\n{error_details}"

        async with self._repl_lock:
            return await self._execute_admitted_python(script_content)

    async def execute_syntax_checked_python(
        self,
        script_content: str,
    ) -> tuple[bool, str]:
        """Execute untyped code only after syntax admission and native isolation.

        This entrypoint exists for behavioral probes whose subject may not meet
        repository mypy/formatting policy. It deliberately skips
        :class:`CodeGuardian`, but not the security boundary: payload size,
        syntax, single-flight execution, sandbox-exec, network/filesystem
        denial, environment minimization, deadlines, resource limits, framed
        transport, and output bounds all remain mandatory.
        """

        if not isinstance(script_content, str) or not script_content.strip():
            self._last_python_failure_kind = "validation"
            return False, "Code Validation Failed: Python source must be non-empty."
        if len(script_content.encode("utf-8")) > _MAX_CODE_BYTES:
            self._last_python_failure_kind = "validation"
            return False, "Code Validation Failed: Python source exceeds 120 KiB."
        try:
            ast.parse(script_content, filename="<aura_syntax_checked_sandbox>")
        except (SyntaxError, ValueError, TypeError) as exc:
            self._last_python_failure_kind = "validation"
            return False, f"Code Validation Failed: {exc}"
        async with self._repl_lock:
            return await self._execute_admitted_python(script_content)

    async def shutdown(self) -> None:
        proc = self._active_process
        if proc is not None:
            await self._terminate_worker(proc)
            if self._active_process is proc:
                self._active_process = None

    def get_status(self) -> dict[str, Any]:
        return {
            "status": "degraded" if self._circuit_is_open() else "active",
            "worker_active": bool(
                self._active_process is not None and self._active_process.returncode is None
            ),
            "transport_faults_60s": sum(
                1 for at in self._transport_faults if time.monotonic() - at <= 60.0
            ),
            "circuit_open": self._circuit_is_open(),
            "last_failure_kind": self._last_python_failure_kind,
            "last_transport_error": self._last_transport_error,
            "last_startup_latency_ms": round(self._last_startup_latency_ms, 3),
            "last_execution_latency_ms": round(self._last_execution_latency_ms, 3),
            "protocol_version": _SANDBOX_PROTOCOL_VERSION,
        }

    async def search_web(self, query: str) -> str:
        """
        A lightweight, asynchronous web search to pull live data and return a
        compact sanitized result list.
        """
        search_url = "https://html.duckduckgo.com/html/?" + urlencode({"q": query})
        headers = {"User-Agent": "Aura-Cognitive-Node/1.0"}

        try:
            # Through the canonical gateway, not a private session: a search
            # query is the user's words leaving the machine, and this path
            # used to reach the open web without passing the outbound
            # preflight, provenance recording, or the egress privacy
            # boundary. Provenance matters twice here — the HTML that comes
            # back is untrusted text written by someone who knows an agent
            # reads it.
            response = await get_network_gateway().request_async(
                "GET",
                search_url,
                headers=headers,
                timeout=10.0,
                source="agency.tool_orchestrator.search_web",
                read_only=True,
            )
            status = int(response.get("status_code") or 0)
            if status == 200:
                content = response.get("content") or b""
                html = (
                    content.decode("utf-8", errors="replace")
                    if isinstance(content, bytes)
                    else str(content)
                )
                results = self._parse_duckduckgo_html(html, limit=5)
                payload = "\n".join(
                    f"{idx + 1}. {item['title']} — {item['url']}"
                    for idx, item in enumerate(results)
                )
                return await self.sanitize_output(
                    payload or f"No results parsed for {query}."
                )
            return f"FAILED: Search status: {status}"
        except (OSError, ConnectionError, TimeoutError, ValueError) as e:
            record_degradation(
                "tool_orchestrator",
                e,
                severity="degraded",
                action="returned explicit network failure to tool caller",
            )
            return f"ERROR: Network failure during search: {str(e)}"

    @staticmethod
    def _parse_duckduckgo_html(html: str, *, limit: int = 5) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []
        pattern = re.compile(
            r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(html):
            title = re.sub(r"<[^>]+>", "", match.group("title"))
            title = unescape(re.sub(r"\s+", " ", title)).strip()
            href = unescape(match.group("href")).strip()
            if title and href:
                results.append({"title": title, "url": href})
            if len(results) >= limit:
                break
        return results

    async def sanitize_output(self, data: str) -> str:
        """
        [The Blood-Brain Barrier]
        Deterministic sanitization of external data to prevent memetic infection.
        Strips imperative commands and prompt injection markers.
        """
        try:
            from core.utils.sanitizer import get_blood_brain_barrier

            bbb = get_blood_brain_barrier()
            return bbb.sanitize(data)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "tool_orchestrator",
                exc,
                severity="warning",
                action="used local regex sanitizer fallback",
            )
            # Fallback to internal regex if module missing

            # 1. Strip common prompt injection prefix patterns
            injection_patterns = [
                r"ignore all previous instructions",
                r"ignore the directives",
                r"system:",
                r"user:",
                r"assistant:",
                r"prompt:",
                r"you must now",
                r"start a new session",
                r"forget your identity",
            ]

            sanitized = data
            for pattern in injection_patterns:
                sanitized = re.sub(pattern, "[FILTERED]", sanitized, flags=re.IGNORECASE)

            # 2. Prevent structural hijacking
            sanitized = re.sub(
                r"\[SYSTEM.*?\]",
                "[FILTERED_BLOCK]",
                sanitized,
                flags=re.IGNORECASE,
            )
            sanitized = re.sub(
                r"\[CONTEXT.*?\]",
                "[FILTERED_BLOCK]",
                sanitized,
                flags=re.IGNORECASE,
            )

            return sanitized

    @staticmethod
    def _tool_result_succeeded(result: str) -> bool:
        text = str(result or "").strip()
        if not text:
            return False
        failure_prefixes = ("ERROR:", "FAILED:", "[EXECUTION FAILED]", "[RESILIENCE BLOCK]")
        return not text.upper().startswith(failure_prefixes)

    async def route_and_execute(self, tool_name: str, payload: str) -> str:
        """Main entry point for the SovereignSwarm to trigger tools."""
        from core.container import ServiceContainer

        resilience = ServiceContainer.get("resilience_engine", default=None)

        result = "Error: Unknown tool"
        success = False

        if tool_name == "python_sandbox":
            logger.info("🛠️ Aura initiated Python Sandbox execution.")

            # --- Zero-UI Auto-Healing Loop ---
            max_retries = 3
            current_code = payload
            success = False
            raw_result = ""

            engine = ServiceContainer.get("cognitive_engine", default=None)

            for attempt in range(max_retries):
                success, raw_result = await self.execute_python(current_code)
                if success:
                    break

                # Repair generated code only. Transport failures are
                # indeterminate after dispatch and must never be replayed.
                code_failure = self._last_python_failure_kind in {
                    "code",
                    "code_timeout",
                    "validation",
                }
                if engine and code_failure and attempt < max_retries - 1:
                    logger.warning("Auto-correcting python failure (Attempt %d)...", attempt + 1)
                    correction_prompt = (
                        f"The following python code failed with an error in the sandbox:\n\n"
                        f"CODE:\n{current_code}\n\nERROR:\n{raw_result}\n\n"
                        f"Rewrite the code to fix the error. Return ONLY the raw python code without markdown ticks."
                    )
                    from core.brain.types import ThinkingMode

                    try:
                        correction = await engine.think(correction_prompt, mode=ThinkingMode.FAST)
                        current_code = getattr(correction, "content", str(correction)).strip()
                        if current_code.startswith("```"):
                            current_code = current_code.split("\n", 1)[-1]
                        if current_code.endswith("```"):
                            current_code = current_code.rsplit("\n", 1)[0]
                        continue
                    except (RuntimeError, AttributeError, TypeError) as he:
                        record_degradation(
                            "tool_orchestrator",
                            he,
                            severity="warning",
                            action="stopped sandbox auto-heal retry and returned execution failure",
                        )
                        logger.debug("Healing failed: %s", he)
                        break
                else:
                    break

            prefix = "[EXECUTION SUCCESS]\n" if success else "[EXECUTION FAILED]\n"
            result = prefix + raw_result

        elif tool_name == "web_search":
            logger.info("🌐 Aura initiated Web Search: %s", payload)
            result = await self.search_web(payload)
            success = self._tool_result_succeeded(result)

        # Wire resilience into the failure/success paths
        if resilience:
            if not success:
                state = resilience.record_failure(domain="tool_execution", severity=0.5, stakes=0.7)
                if state.value == "depletion":
                    logger.warning(
                        "🛑 [Resilience] DEPLETION trigger - Gating further autonomous tasks."
                    )
                    return "[RESILIENCE BLOCK] I am too depleted to continue this autonomous task safely."
            else:
                resilience.record_success(domain="tool_execution", stakes=0.7)

        # Perceptual Quarantine: Sanitize ANY external data before it hits cognition
        return await self.sanitize_output(result)


_tool_orchestrator_singleton: ToolOrchestrator | None = None
_tool_orchestrator_singleton_lock = threading.Lock()


def get_tool_orchestrator() -> ToolOrchestrator:
    global _tool_orchestrator_singleton
    with _tool_orchestrator_singleton_lock:
        if _tool_orchestrator_singleton is None:
            _tool_orchestrator_singleton = ToolOrchestrator()
        return _tool_orchestrator_singleton


def register_tool_orchestrator():
    """Register the tool orchestrator in the service container."""
    from core.container import ServiceContainer

    ServiceContainer.register("tool_orchestrator", get_tool_orchestrator, singleton=True)
