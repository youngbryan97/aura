"""core/agency/tool_orchestrator.py

Asynchronous Tool Execution Environment.
Grants Aura the ability to run Python scripts and search the web to resolve
knowledge gaps dynamically.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from html import unescape
from pathlib import Path
from urllib.parse import urlencode

import aiohttp

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import (
    DependencyUnavailable,
    TimeoutBudgetExceeded,
    record_degradation,
)

logger = logging.getLogger("Aura.ToolOrchestrator")


class ToolOrchestrator:
    def __init__(self):
        # We use a temporary directory to sandbox her generated scripts
        self.sandbox_dir = Path(tempfile.gettempdir()) / "aura_sandbox"
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.execution_timeout = 15.0  # Kill runaway loops after 15 seconds
        self._repl_process = None
        self._repl_lock = asyncio.Lock()

    def _build_repl_launch_config(self) -> tuple[str, str, str, str]:
        sandbox_exec = shutil.which("sandbox-exec")
        if not sandbox_exec:
            raise DependencyUnavailable(
                "sandbox-exec is unavailable; refusing to launch an unsandboxed Python daemon"
            )

        python_bin = str(Path(sys.executable).resolve())
        daemon_path = Path(__file__).with_name("repl_daemon.py")
        if not daemon_path.exists():
            raise DependencyUnavailable(f"Python sandbox daemon is missing: {daemon_path}")

        policy = f"""
        (version 1)
        (allow default)
        (deny network*)
        (deny file-read* (subpath "/Users"))
        (deny file-read* (subpath "/Volumes"))
        (allow file-read* (subpath "{self.sandbox_dir}"))
        (allow file-read* (subpath "{os.path.dirname(__file__)}"))
        (deny file-write* (subpath "/"))
        (allow file-write* (subpath "{self.sandbox_dir}"))
        """
        return sandbox_exec, python_bin, str(daemon_path), policy

    async def _ensure_repl(self):
        if self._repl_process and self._repl_process.returncode is None:
            return

        sandbox_exec, python_bin, daemon_path, policy = self._build_repl_launch_config()
        self._repl_process = await asyncio.create_subprocess_exec(
            sandbox_exec,
            "-p",
            policy,
            python_bin,
            daemon_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _terminate_repl(self) -> None:
        proc = self._repl_process
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except (ProcessLookupError, OSError) as exc:
                logger.debug(
                    "Sandbox daemon termination already complete or unavailable: %s",
                    exc,
                )
        self._repl_process = None

    async def _remove_validation_file(self, path: Path) -> OSError | None:
        try:
            await asyncio.to_thread(lambda: path.unlink(missing_ok=True))
        except OSError as exc:
            record_degradation(
                "tool_orchestrator",
                exc,
                severity="warning",
                action="failed closed on validation temp cleanup",
            )
            return exc
        return None

    async def execute_python(self, script_content: str) -> tuple[bool, str]:
        """
        Asynchronously executes a Python script in a STATEFUL sandbox daemon.
        Variables persist across calls!
        """
        # Autonomous Code Validation (Anti-NameError/TypeError)
        from core.utils.code_guardian import CodeGuardian

        tmp_path = self.sandbox_dir / "temp_validation.py"
        try:
            await asyncio.to_thread(lambda: atomic_write_text(tmp_path, script_content))
            report = CodeGuardian.validate_code(tmp_path)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            await self._remove_validation_file(tmp_path)
            record_degradation(
                "tool_orchestrator",
                exc,
                severity="critical",
                action="failed closed before launching python sandbox",
            )
            return False, f"Code validation setup failed: {exc}"

        cleanup_error = await self._remove_validation_file(tmp_path)
        if cleanup_error is not None:
            return False, f"Code validation cleanup failed: {cleanup_error}"

        if not report.success:
            logger.warning("🛡️ ToolOrchestrator: CodeGuardian BLOCKED execution.")
            error_details = report.error_message
            if report.ruff_output:
                error_details += f"\nLinter:\n{report.ruff_output}"
            return False, f"Code Validation Failed:\n{error_details}"

        async with self._repl_lock:
            try:
                await self._ensure_repl()
                proc = self._repl_process
                if proc is None or proc.stdin is None or proc.stdout is None:
                    raise DependencyUnavailable("sandbox daemon did not expose stdin/stdout pipes")

                code_bytes = script_content.encode("utf-8")
                proc.stdin.write(f"{len(code_bytes)}\n".encode())
                proc.stdin.write(code_bytes)
                proc.stdin.write(b"\n")
                await proc.stdin.drain()

                # Daemon returns length followed by JSON payload
                len_line = await asyncio.wait_for(
                    proc.stdout.readline(),
                    timeout=self.execution_timeout,
                )
                if not len_line:
                    return False, "Daemon crashed or closed stream."

                payload_len = int(len_line.strip())
                payload_bytes = await proc.stdout.readexactly(payload_len)
                # clear trailing newline
                await proc.stdout.readline()

                data = json.loads(payload_bytes.decode("utf-8"))
                return data["success"], data["output"] or "[Empty Output]"

            except TimeoutError as exc:
                record_degradation(
                    "tool_orchestrator",
                    TimeoutBudgetExceeded(str(exc) or "Python sandbox execution timed out"),
                    severity="warning",
                    action="killed sandbox daemon and returned timeout failure",
                )
                self._terminate_repl()
                return False, "Execution Error: Script timed out (Infinite loop or heavy resource)."
            except (
                DependencyUnavailable,
                FileNotFoundError,
                PermissionError,
                ProcessLookupError,
                BrokenPipeError,
                asyncio.IncompleteReadError,
                json.JSONDecodeError,
                OSError,
                UnicodeDecodeError,
                TypeError,
                ValueError,
            ) as exc:
                record_degradation(
                    "tool_orchestrator",
                    exc,
                    severity="critical",
                    action="terminated python sandbox and returned explicit tool failure",
                )
                self._terminate_repl()
                return False, f"Daemon protocol error: {exc}"

    async def search_web(self, query: str) -> str:
        """
        A lightweight, asynchronous web search to pull live data and return a
        compact sanitized result list.
        """
        search_url = "https://html.duckduckgo.com/html/?" + urlencode({"q": query})
        headers = {"User-Agent": "Aura-Cognitive-Node/1.0"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        html = await resp.text()
                        results = self._parse_duckduckgo_html(html, limit=5)
                        payload = "\n".join(
                            f"{idx + 1}. {item['title']} — {item['url']}"
                            for idx, item in enumerate(results)
                        )
                        return await self.sanitize_output(
                            payload or f"No results parsed for {query}."
                        )
                    return f"FAILED: Search status: {resp.status}"
        except (OSError, ConnectionError, TimeoutError) as e:
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

                # Auto-heal on failure
                if engine and attempt < max_retries - 1:
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


def register_tool_orchestrator():
    """Register the tool orchestrator in the service container."""
    from core.container import ServiceContainer

    ServiceContainer.register("tool_orchestrator", lambda: ToolOrchestrator(), singleton=True)
