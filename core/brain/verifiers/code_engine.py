"""Code truth engine — compile, AST-safety, lint, and (when the candidate
carries executable claims) sandboxed execution.

Wraps the existing :class:`core.resilience.code_verifier.CodeVerifier` (isolated
``py_compile`` + AST safety) and adds an optional ``ruff`` static pass through the
governed subprocess gateway as a *read-only probe*.

Checked-semantics contract (Verifier Foundry finding, 2026-07-14): a candidate
whose blocks contain MODULE-LEVEL executable claims (asserts that would run on
import) is making a semantic claim statics cannot adjudicate. Such blocks are
executed in the :mod:`core.brain.symbolic_sandbox` (vetted, isolated CPython,
governed gateway). When execution is impossible — unsafe block, sandbox
unavailable, timeout — a PASSING static verdict demotes to ``checked=False``:
"compiles clean" must never masquerade as "verified correct". Provable static
failures still fail hard regardless.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import re
from dataclasses import dataclass
from typing import Any

from core.conversation.request_coverage import markdown_fences
from core.runtime.atomic_writer import async_atomic_write_text
from core.runtime.errors import record_degradation

from .base import VerificationResult

# A line that looks like Python even without a fence (heuristic for inline code answers).
_PY_HINT_RE = re.compile(r"^\s*(?:def |class |import |from \w+ import |async def )", re.MULTILINE)

_OUTPUT_CLAIM_LABEL_RE = re.compile(
    r"(?P<label>(?:^|\n)[ \t]*(?:#{1,6}[ \t]+)?"
    r"(?:(?:typical|example|sample|expected|possible|actual|observed)[ \t]+)?"
    r"(?:program[ \t]+)?(?:output|stdout|console[ \t]+output|prints?)"
    r"(?:[ \t]*\([^\n)]*\))?[ \t]*:\s*)\Z",
    re.IGNORECASE,
)
_PYTHON_FENCE_LANGUAGES = frozenset({"python", "python3", "py"})
_OUTPUT_FENCE_LANGUAGES = frozenset({"", "console", "output", "plaintext", "stdout", "text"})
_MAX_OUTPUT_CLAIMS = 3
_OUTPUT_GROUNDING_DEADLINE_S = 15.0
_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


@dataclass(frozen=True)
class ExecutableOutputGrounding:
    """Visible reply plus receipts for Python-output claims checked at egress."""

    text: str
    receipts: tuple[dict[str, Any], ...] = ()

    @property
    def changed(self) -> bool:
        return any(bool(item.get("visible_text_changed")) for item in self.receipts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.executable_output_grounding.v1",
            "claim_count": len(self.receipts),
            "checked_count": sum(bool(item.get("checked")) for item in self.receipts),
            "grounded_count": sum(
                str(item.get("status") or "") in {"verified_match", "grounded_to_observation"}
                for item in self.receipts
            ),
            "visible_text_changed": self.changed,
            "claims": [dict(item) for item in self.receipts],
        }


def _observed_output_block(stdout: str) -> str:
    visible = str(stdout or "").rstrip("\n") or "(no stdout)"
    fence = "```"
    while fence in visible:
        fence += "`"
    return f"\n\nOne observed run:\n\n{fence}text\n{visible}\n{fence}"


def _normalize_captured_output(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _ANSI_ESCAPE_RE.sub("", text)
    text = "".join(char for char in text if char in {"\n", "\t"} or ord(char) >= 32)
    return text.rstrip("\n")


async def ground_python_output_claims(
    candidate: str,
    *,
    sandbox: Any | None = None,
) -> ExecutableOutputGrounding:
    """Ground explicit Python-output claims in isolated execution.

    Only adjacent pairs of an explicitly labelled Python fence and a prose-
    labelled output fence qualify. A successful run replaces a mismatched
    claim with captured stdout and labels it as one observation. If execution
    cannot establish an output, the unsupported output section is removed
    without discarding the surrounding explanation or source.
    """

    text = str(candidate or "")
    fences = list(markdown_fences(text))
    if len(fences) < 2:
        return ExecutableOutputGrounding(text=text)

    executor = sandbox
    if executor is None:
        try:
            from core.brain.symbolic_sandbox import get_symbolic_sandbox

            executor = get_symbolic_sandbox()
        except (ImportError, RuntimeError) as exc:
            record_degradation(
                "executable_output_grounding",
                exc,
                severity="warning",
                action="removed unsupported output claims because the sandbox was unavailable",
            )

    replacements: list[tuple[int, int, str]] = []
    receipts: list[dict[str, Any]] = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _OUTPUT_GROUNDING_DEADLINE_S
    for index, code_fence in enumerate(fences[:-1]):
        if len(receipts) >= _MAX_OUTPUT_CLAIMS:
            break
        if not code_fence.closed or code_fence.language not in _PYTHON_FENCE_LANGUAGES:
            continue
        output_fence = fences[index + 1]
        if not output_fence.closed or output_fence.language not in _OUTPUT_FENCE_LANGUAGES:
            continue
        between = text[code_fence.end : output_fence.start]
        label_match = _OUTPUT_CLAIM_LABEL_RE.search(between)
        if label_match is None:
            continue

        claim_start = code_fence.end + label_match.start("label")
        claim_end = output_fence.end
        claimed = _normalize_captured_output(output_fence.body)
        receipt: dict[str, Any] = {
            "claim_index": len(receipts),
            "code_sha256": hashlib.sha256(code_fence.body.encode("utf-8")).hexdigest(),
            "claimed_output_sha256": hashlib.sha256(claimed.encode("utf-8")).hexdigest(),
            "checked": False,
            "status": "sandbox_unavailable",
            "visible_text_changed": True,
        }
        observed = ""
        result = None
        if executor is not None:
            try:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    receipt["status"] = "shared_deadline_exhausted"
                else:
                    result = await asyncio.wait_for(
                        executor.run(code_fence.body),
                        timeout=remaining,
                    )
            except TimeoutError:
                receipt["status"] = "execution_timed_out"
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    "executable_output_grounding",
                    exc,
                    severity="warning",
                    action="removed an output claim whose execution check failed",
                )
        if result is not None and bool(getattr(result, "ok", False)):
            isolation = dict(getattr(result, "isolation", {}) or {})
            if bool(isolation.get("sandboxed")):
                observed = _normalize_captured_output(getattr(result, "stdout", ""))
                receipt.update(
                    {
                        "checked": True,
                        "observed_output_sha256": hashlib.sha256(
                            observed.encode("utf-8")
                        ).hexdigest(),
                        "isolation_level": str(isolation.get("isolation_level") or ""),
                        "comparison_normalization": "crlf_ansi_controls_terminal_newline_v1",
                    }
                )
                if claimed == observed:
                    receipt.update(
                        {
                            "status": "verified_match",
                            "visible_text_changed": False,
                        }
                    )
                else:
                    receipt["status"] = "grounded_to_observation"
                    replacements.append(
                        (claim_start, claim_end, _observed_output_block(observed))
                    )
            else:
                receipt["status"] = "isolation_unproven"
        elif result is not None:
            if bool(getattr(result, "timed_out", False)):
                receipt["status"] = "execution_timed_out"
            elif bool(getattr(result, "refused", False)):
                receipt["status"] = "execution_refused"
            else:
                receipt["status"] = "execution_failed"

        if receipt["visible_text_changed"] and receipt["status"] != "grounded_to_observation":
            replacements.append(
                (
                    claim_start,
                    claim_end,
                    "\n\nThe isolated execution check did not verify an output, so no output is claimed here.",
                )
            )
        receipts.append(receipt)

    grounded = text
    for start, end, replacement in reversed(replacements):
        grounded = grounded[:start] + replacement + grounded[end:]
    return ExecutableOutputGrounding(text=grounded, receipts=tuple(receipts))


def extract_code_blocks(text: str) -> list[str]:
    """Pull fenced python blocks; fall back to the whole text if it parses as code."""
    blocks = [
        fence.body.strip()
        for fence in markdown_fences(text)
        if fence.closed
        and fence.language in _PYTHON_FENCE_LANGUAGES | {""}
        and fence.body.strip()
    ]
    if blocks:
        return blocks
    body = str(text or "").strip()
    if body and _PY_HINT_RE.search(body):
        return [body]
    return []


_DOCTEST_RE = re.compile(r"^\s*>>> ", re.MULTILINE)


def has_doctest_examples(code: str) -> bool:
    """True when the block states what it does, in runnable form.

    A doctest is the most common way a Python answer says what its code
    returns, and it is exactly the executable self-claim this engine exists to
    adjudicate — the same standing as a module-level assert. Skipping them
    meant an answer whose own examples were WRONG still scored as verified,
    because "compiles clean" was the whole check.
    """
    return bool(_DOCTEST_RE.search(str(code or "")))


def doctest_harness(code: str) -> str:
    """The block, plus the two lines that run its own examples.

    Appended rather than executed here so the sandbox stays the only place
    candidate code ever runs.
    """
    # doctest.testmod() tests the module named __main__ — whatever happens to
    # be running the code, which is not the candidate — and DocTestFinder
    # additionally skips any object whose __module__ does not match. Both make
    # a wrong example count as zero examples and pass. The finder is given the
    # objects themselves instead.
    return (
        f"{code.rstrip()}\n\n"
        "import doctest as _aura_doctest\n"
        "_aura_finder = _aura_doctest.DocTestFinder()\n"
        "_aura_runner = _aura_doctest.DocTestRunner(verbose=False)\n"
        "_aura_globals = dict(globals())\n"
        "for _aura_name, _aura_obj in list(_aura_globals.items()):\n"
        "    if _aura_name.startswith('_aura'):\n"
        "        continue\n"
        "    if not (callable(_aura_obj) or isinstance(_aura_obj, type)):\n"
        "        continue\n"
        "    if not getattr(_aura_obj, '__doc__', None):\n"
        "        continue\n"
        "    for _aura_test in _aura_finder.find(\n"
        "        _aura_obj, name=_aura_name, globs=dict(_aura_globals)\n"
        "    ):\n"
        "        _aura_runner.run(_aura_test)\n"
        "assert _aura_runner.failures == 0, (\n"
        '    f"{_aura_runner.failures} of {_aura_runner.tries} "\n'
        '    "doctest examples failed"\n'
        ")\n"
    )


def has_module_level_asserts(code: str) -> bool:
    """True when the block contains asserts that would EXECUTE on a plain run:
    statements at module level (including inside module-level if/try/loop
    bodies, e.g. an ``if __name__ == "__main__"`` harness), but not asserts
    tucked inside function/class definitions that nothing calls."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    def _walk(stmts: list[ast.stmt]) -> bool:
        for node in stmts:
            if isinstance(node, ast.Assert):
                return True
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue  # not executed on import unless called
            for field_name in ("body", "orelse", "finalbody"):
                child = getattr(node, field_name, None)
                if child and _walk(child):
                    return True
            for handler in getattr(node, "handlers", []) or []:
                if _walk(handler.body):
                    return True
        return False

    return _walk(tree.body)


class CodeTruthEngine:
    name = "code"
    domains = ("code", "code_audit", "code_patch", "debug")

    def __init__(self, *, run_ruff: bool = True, sandbox: Any | None = None) -> None:
        self._run_ruff = run_ruff
        self._sandbox = sandbox

    def handles(self, task_type: str) -> bool:
        return task_type in self.domains

    def _resolve_sandbox(self) -> Any | None:
        if self._sandbox is not None:
            return self._sandbox
        try:
            from core.brain.symbolic_sandbox import get_symbolic_sandbox

            self._sandbox = get_symbolic_sandbox()
        except (ImportError, RuntimeError) as exc:
            record_degradation("code_truth_engine", exc, severity="warning",
                               action="sandbox unavailable; executable claims will demote checked")
            self._sandbox = None
        return self._sandbox

    async def verify(self, candidate: str, *, context: dict[str, Any] | None = None) -> VerificationResult:
        blocks = extract_code_blocks(candidate)
        if not blocks:
            # Nothing to verify — benign, and NOT an infrastructure failure.
            return VerificationResult(domain="code", ok=True, checked=False, engine=self.name)

        issues: list[str] = []
        evidence: list[str] = []
        compiled_ok = 0
        try:
            from core.resilience.code_verifier import CodeVerifier
        except (ImportError, RuntimeError) as exc:  # pragma: no cover - import guard
            record_degradation("code_truth_engine", exc)
            # There IS code here and the verifier could not load. That is
            # unverifiable, not unnecessary — the distinction a gate needs.
            return VerificationResult(
                domain="code",
                ok=True,
                checked=False,
                engine=self.name,
                infrastructure_failed=True,
                issues=[f"code_verifier_unavailable:{type(exc).__name__}"],
            )

        executed_ok = 0
        unverified_claims = 0
        for idx, block in enumerate(blocks):
            report = CodeVerifier.verify_importability_report(block, module_name=f"candidate_{idx}")
            if not report.syntax_ok:
                issues.append(f"block#{idx}: syntax error")
                continue
            if not report.ok:
                stderr = (report.stderr or report.error or "compile failed").strip().splitlines()
                issues.append(f"block#{idx}: {stderr[-1] if stderr else 'compile failed'}")
            else:
                compiled_ok += 1
                evidence.append(f"block#{idx}: compiles clean")
            if report.warnings:
                issues.extend(f"block#{idx} unsafe: {w}" for w in report.warnings)
            if self._run_ruff:
                ruff_issues = await self._ruff(block)
                issues.extend(f"block#{idx} lint: {m}" for m in ruff_issues[:3])

            # Executable self-claims (module-level asserts) demand execution:
            # "compiles clean" is not a verdict on what the block CLAIMS.
            if report.syntax_ok and report.ok and not report.warnings:
                runnable, claim = "", ""
                if has_module_level_asserts(block):
                    runnable, claim = block, "module-level asserts"
                elif has_doctest_examples(block):
                    runnable, claim = doctest_harness(block), "doctest examples"
                if runnable:
                    outcome, issue = await self._execute_claims(idx, runnable)
                    if outcome == "pass":
                        executed_ok += 1
                        evidence.append(f"block#{idx}: {claim} passed in sandbox")
                    elif outcome == "fail":
                        issues.append(issue or f"block#{idx}: runtime failure")
                    else:  # "unavailable"
                        unverified_claims += 1

        hard_fail_markers = ("syntax", "compile", "unsafe", "runtime failure")
        ok = not any(any(m in i for m in hard_fail_markers) for i in issues)

        # Demotion: statics passed but the candidate's executable claims could
        # not be run — the engine did NOT meaningfully check this candidate.
        if ok and unverified_claims:
            return VerificationResult(
                domain="code",
                ok=True,
                checked=False,
                score=0.5,
                engine=self.name,
                issues=issues,
                evidence=evidence + [
                    f"{unverified_claims} block(s) carry executable claims that "
                    "could not be executed — static pass is not a semantic verdict"
                ],
                detail={"blocks": len(blocks), "compiled_ok": compiled_ok,
                        "unverified_executable_claims": unverified_claims},
            )

        # Score rewards clean-compiling blocks, penalises lint noise, and
        # credits genuine execution evidence.
        score = (compiled_ok / max(1, len(blocks))) * (0.9 if not issues else 0.6)
        if executed_ok:
            score = min(0.98, score + 0.05 * executed_ok)
        return VerificationResult(
            domain="code",
            ok=ok,
            checked=True,
            score=round(min(0.98, max(0.05, score)), 4),
            engine=self.name,
            issues=issues,
            evidence=evidence,
            detail={"blocks": len(blocks), "compiled_ok": compiled_ok,
                    "executed_ok": executed_ok},
        )

    async def _execute_claims(self, idx: int, block: str) -> tuple[str, str]:
        """Run an assert-bearing block in the symbolic sandbox.

        Returns ("pass", ""), ("fail", issue) for a proven runtime failure, or
        ("unavailable", "") when no trustworthy execution could happen
        (sandbox missing, block refused, timeout, infra error)."""
        sandbox = self._resolve_sandbox()
        if sandbox is None:
            return "unavailable", ""
        try:
            result = await sandbox.run(block)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            record_degradation("code_truth_engine", exc, severity="warning",
                               action="sandbox execution errored; claims unverified")
            return "unavailable", ""
        if getattr(result, "refused", False) or getattr(result, "timed_out", False):
            return "unavailable", ""
        if getattr(result, "ok", False):
            return "pass", ""
        tb = (getattr(result, "traceback", "") or getattr(result, "stderr", "")
              or "nonzero exit").strip().splitlines()
        return "fail", f"block#{idx}: runtime failure: {tb[-1] if tb else 'nonzero exit'}"

    async def _ruff(self, code: str) -> list[str]:
        """Static lint via ruff as a governed read-only probe; best-effort."""
        try:
            import shutil
            import tempfile
            from pathlib import Path

            from core.runtime.subprocess_gateway import get_subprocess_gateway

            if shutil.which("ruff") is None:
                return []
            with tempfile.TemporaryDirectory(prefix="aura_ruff_") as td:
                path = Path(td) / "candidate.py"
                await async_atomic_write_text(path, code, encoding="utf-8")
                res = await get_subprocess_gateway().run_async(
                    ("ruff", "check", "--quiet", "--no-cache", str(path)),
                    timeout=15.0,
                    read_only=True,
                    source="reasoning_verifier:code_ruff",
                    accelerator_capability="none",
                )
            out = (res.stdout or "") + (res.stderr or "")
            return [ln.strip() for ln in out.splitlines() if ln.strip() and ":" in ln][:5]
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            record_degradation("code_truth_engine_ruff", exc)
            return []
