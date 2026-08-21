"""core/capabilities/self_taught_builder.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Aura teaches herself to build things that actually WORK.

The difference between a code generator and an intelligence is metacognition:
recognizing a gap, going to learn (concepts AND real reference code), applying
it, TESTING that the result meets a standard, persisting until it does, and
RETAINING what was learned in a general form so the next hard task compounds.

The loop, general over any buildable artifact:

    recall prior learnings  →  research the task (corpus + web, incl. code)
      →  generate with everything injected  →  FUNCTIONALLY test it
      →  if it fails: feed the exact failure back + research that failure
      →  persist (bounded)  →  retain the general lesson

Code synthesis uses the un-steered local code model. Functional testing runs a
real headless DOM (tools/appcheck/test_game.js) so "it works" is verified by
simulating actual play, not assumed from structure. Every run — success or
partial — writes a general, reusable lesson to memory (recall feeds the next
build), so she genuinely learns and applies.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import async_atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.SelfTaughtBuilder")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TESTER = _REPO_ROOT / "tools" / "appcheck" / "test_game.js"
_LESSON_FAMILY = "learned_build_lesson"


@dataclass
class VerifiedBuildResult:
    ok: bool
    spec: str
    path: str = ""
    code: str = ""
    playable: bool = False
    iterations: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    research_used: list[str] = field(default_factory=list)
    recalled_lessons: list[str] = field(default_factory=list)
    lesson_retained: str = ""
    status: str = "ok"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload["code"] = self.code[:20000]
        return payload


# ── research (learn concepts AND real reference code) ─────────────────────

async def _research(topic: str, *, want_code: bool = True, max_notes: int = 6) -> list[str]:
    notes: list[str] = []
    # 1) her offline reference corpus (fast, always available): concepts/rules
    try:
        from core.knowledge.local_corpus import get_local_corpus_store

        for hit in get_local_corpus_store().search(topic, limit=3):
            snippet = f"{hit.title}: {hit.snippet}".strip()
            if snippet:
                notes.append("[corpus] " + snippet[:400])
    except (ImportError, RuntimeError, OSError, TypeError, ValueError) as exc:
        logger.debug("corpus research skipped: %s", exc)
    # 2) the live web (implementation patterns + literal reference code)
    if want_code:
        try:
            from core.skills.web_search import EnhancedWebSearchSkill

            res = await EnhancedWebSearchSkill().safe_execute(
                {"query": topic, "max_results": 3},
                {"origin": "self_taught_builder"},
            )
            for item in (res.get("results") or [])[:3]:
                text = str(item.get("snippet") or item.get("content") or item.get("title") or "")
                if text.strip():
                    notes.append("[web] " + text.strip()[:400])
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError, KeyError) as exc:
            logger.debug("web research skipped: %s", exc)
    return notes[:max_notes]


# ── retain + recall (general, cumulative learning) ────────────────────────

async def _retain(spec: str, outcome: str, lesson: str) -> str:
    text = (
        f"Build lesson ({outcome}) for '{spec[:80]}': {lesson}"
    )
    try:
        from core.memory.memory_write_gateway import get_memory_write_gateway
        from core.runtime.gateways import MemoryWriteRequest

        await get_memory_write_gateway().write(
            MemoryWriteRequest(
                content=text,
                metadata={
                    "family": _LESSON_FAMILY,
                    "source": "self_taught_builder",
                    "outcome": outcome,
                    "domain": _domain_of(spec),
                    "explicit_observational_memory_write": True,
                },
                cause="self_taught_builder.retain",
            )
        )
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
        record_degradation(
            "self_taught_builder.retain", exc, severity="warning",
            action="kept the built artifact after lesson retention failed",
        )
    return text


async def _recall(spec: str, *, limit: int = 4) -> list[str]:
    """Pull prior general lessons that apply to this build (cumulative learning)."""
    domain = _domain_of(spec)
    lessons: list[str] = []
    try:
        from core.container import ServiceContainer
        from core.service_names import ServiceNames

        memory = ServiceContainer.get(ServiceNames.MEMORY_FACADE, default=None)
        hits = (
            memory.search_sync(f"build lesson {domain}", limit=limit)
            if memory is not None and hasattr(memory, "search_sync")
            else []
        )
        for h in hits or []:
            content = str(getattr(h, "content", "") or (h.get("content") if isinstance(h, dict) else ""))
            if "Build lesson" in content:
                lessons.append(content[:400])
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
        pass
    return lessons[:limit]


def _domain_of(spec: str) -> str:
    s = spec.lower()
    for kw in ("chess", "checkers", "draughts", "tic", "snake", "pong", "calculator", "game"):
        if kw in s:
            return "web_game" if kw not in {"calculator"} else "web_tool"
    return "web_app"


# ── generation (un-steered code model, everything injected) ───────────────

def _build_prompt(spec: str, research: list[str], lessons: list[str], prior: str, failure: str) -> str:
    parts = [
        "Build a COMPLETE, self-contained, single-file web app (one HTML document; CSS in "
        "<style>, JS in <script>; no external files/CDNs/network). It must ACTUALLY WORK when "
        "opened — real, wired interactivity, not a mockup. Output ONLY the HTML.\n",
        f"App to build: {spec}\n",
    ]
    if lessons:
        parts.append("What you learned from earlier builds (apply it):\n- " + "\n- ".join(lessons))
    if research:
        parts.append("Reference knowledge you researched (use it):\n- " + "\n- ".join(research))
    if prior and failure:
        parts.append(
            "Your PREVIOUS attempt FAILED a functional test. Fix the exact problem:\n"
            f"{failure}\n\nPrevious code (repair it, keep what worked):\n{prior[:6000]}"
        )
    return "\n\n".join(parts)


async def _generate(prompt: str, *, max_tokens: int) -> str:
    try:
        from core.brain.llm.local_code_model import get_local_code_model

        model = get_local_code_model()
        if model is not None:
            generated = str(
                await model.generate(
                    prompt,
                    system_prompt=(
                        "You are a meticulous front-end engineer. You output ONE complete HTML "
                        "document and nothing else. You wire every interaction and test your logic "
                        "mentally before finalizing. Standard browser APIs only."
                    ),
                    max_tokens=max_tokens,
                    temperature=0.2,
                )
            )
            logger.info(
                "🎓 build: local code model returned %d chars (budget %d).",
                len(generated),
                max_tokens,
            )
            if generated.strip():
                return generated
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        # At debug this was invisible, so a build that produced no code three
        # times in a row said nothing about which generator had failed or why
        # — and the turn reported "the construction process didn't generate
        # any code" with no way to find out.
        logger.warning(
            "🎓 build: local code model unavailable (%s: %s); trying the fallback generator.",
            type(exc).__name__,
            str(exc)[:160],
        )
    # The model that is already loaded.
    #
    # LIVE, 2026-08-21: "local code model unavailable
    # (ModelLaneControlError: in_process_model_admission_refused:
    # lane_budget_exceeded:cortex request 21.5GB + committed 25.3GB > budget)"
    # — a second code model cannot fit beside the resident cortex on this
    # host, so build_app depended on something that could never load. The
    # cortex writes HTML perfectly well; asked directly in a chat turn it
    # produced this same page in 33 seconds.
    try:
        from core.container import ServiceContainer

        gate = ServiceContainer.get("inference_gate", default=None)
        if gate is not None and hasattr(gate, "think"):
            resident = str(
                await gate.think(
                    prompt,
                    system_prompt=(
                        "You are a meticulous front-end engineer. You output ONE complete HTML "
                        "document and nothing else. Standard browser APIs only."
                    ),
                    max_tokens=max_tokens,
                    temperature=0.2,
                    origin="self_taught_builder",
                )
                or ""
            )
            logger.info("🎓 build: resident cortex returned %d chars.", len(resident))
            if resident.strip():
                return resident
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
        logger.warning(
            "🎓 build: resident cortex unavailable (%s: %s).",
            type(exc).__name__,
            str(exc)[:160],
        )

    try:
        from core.brain.llm.code_generator import LLMCodeGenerator

        fallback = str(
            await LLMCodeGenerator(max_tokens=max_tokens, temperature=0.2).generate_async(
                prompt, context={"origin": "self_taught_builder"}
            )
        )
        logger.info("🎓 build: fallback generator returned %d chars.", len(fallback))
        return fallback
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
        logger.warning(
            "🎓 build: fallback generator failed (%s: %s).",
            type(exc).__name__,
            str(exc)[:160],
        )
        record_degradation("self_taught_builder.generate", exc, severity="warning",
                           action="build failed because no code model was available")
        return ""


def _extract_html(raw: str) -> str:
    text = str(raw or "").strip()
    fence = re.search(r"```(?:html|xml)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    low = text.lower()
    start = low.find("<!doctype")
    if start == -1:
        start = low.find("<html")
    if start > 0:
        text = text[start:]
    end = text.lower().rfind("</html>")
    if end != -1:
        text = text[: end + len("</html>")]
    return text.strip()


# ── functional test (real headless DOM: does it actually play?) ───────────

async def _functional_test(html_path: str) -> dict[str, Any]:
    if not _TESTER.exists():
        return {"playable": None, "reason": "no functional tester available"}
    proc = None
    try:
        proc = await get_subprocess_gateway().spawn_async(
            ["node", str(_TESTER), str(html_path)],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            read_only=True,
            source="self_taught_builder.functional_test",
            accelerator_capability="none",
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=45)
    except (FileNotFoundError, TimeoutError, OSError, RuntimeError) as exc:
        if proc is not None and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except TimeoutError:
                proc.kill()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except TimeoutError:
                    logger.warning("SIGKILLed tester pid=%s not reaped in 5s", proc.pid)
        return {"playable": None, "reason": f"functional tester could not run: {exc}"}
    line = (out or b"").decode(errors="replace").strip().splitlines()
    for candidate in reversed(line):
        candidate = candidate.strip()
        if candidate.startswith("{"):
            try:
                payload = json.loads(candidate)
                if isinstance(payload, dict):
                    return {str(key): value for key, value in payload.items()}
            except json.JSONDecodeError:
                continue
    return {"playable": None, "reason": "functional tester produced no verdict", "stderr": (err or b"").decode(errors="replace")[:300]}


# ── the loop ──────────────────────────────────────────────────────────────

def _prepare_output_directory(out_dir: str | Path) -> Path:
    path = Path(out_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


async def build_app_verified(
    spec: str,
    *,
    out_dir: str | Path = "artifacts/live_apps",
    max_iters: int = 4,
    max_tokens: int = 9000,
) -> VerifiedBuildResult:
    spec = str(spec or "").strip()
    result = VerifiedBuildResult(ok=False, spec=spec)
    if not spec:
        result.status = "no_spec"
        return result

    result.recalled_lessons = await _recall(spec)
    research = await _research(spec)
    result.research_used = list(research)

    out_path = await asyncio.to_thread(_prepare_output_directory, out_dir)
    tmp = out_path / f"_wip_{int(time.time())}.html"

    code = ""
    failure = ""
    best_code = ""
    for i in range(1, max_iters + 1):
        result.iterations = i
        prompt = _build_prompt(spec, research, result.recalled_lessons, code, failure)
        raw = await _generate(prompt, max_tokens=max_tokens)
        code = _extract_html(raw)
        if not code:
            result.history.append({"iter": i, "playable": False, "reason": "no code generated"})
            continue
        best_code = code
        await async_atomic_write_text(tmp, code)
        test = await _functional_test(str(tmp))
        entry = {"iter": i, "playable": test.get("playable"), "reason": test.get("reason"),
                 "console_errors": test.get("console_errors", [])}
        result.history.append(entry)
        logger.info("🎓 build iter %d: playable=%s reason=%s", i, test.get("playable"), test.get("reason"))
        if test.get("playable") is True:
            result.playable = True
            break
        # What the test measured, and nothing else.
        #
        # This carried three sentences of checkers-specific advice — "the
        # click handler reads data-row/data-col from the clicked target…
        # resolve to the enclosing square" — appended to EVERY failure in
        # every domain. Building a sitting timer, that is noise about a board
        # game, and steering a repair with a hint from somewhere else is
        # worse than saying only what happened. The research call below is
        # the general mechanism for finding a fix, and it already takes the
        # domain from the spec.
        failure = (
            f"Functional test FAILED: {test.get('reason')}. "
            f"Console errors: {test.get('console_errors')}."
        )
        research += await _research(f"fix {_domain_of(spec)}: {test.get('reason')} {test.get('console_errors')}")

    final = out_path / f"{_slug(spec)}_{int(time.time())}.html"
    await async_atomic_write_text(final if best_code else tmp, best_code or code)
    result.code = best_code or code
    result.path = str(final)
    result.ok = result.playable
    result.status = "playable" if result.playable else "built_not_verified_playable"

    lesson = _distill_lesson(spec, result)
    result.lesson_retained = await _retain(spec, "SUCCESS" if result.playable else "PARTIAL", lesson)
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    return result


def _distill_lesson(spec: str, result: VerifiedBuildResult) -> str:
    if result.playable:
        return (
            f"A working single-file {_domain_of(spec)} needs: a data-model board as source of "
            "truth; every square carrying data-row/data-col; ONE delegated click listener that "
            "resolves the clicked target to its square via closest('[data-row]'); select-then-move "
            "with move validation; re-render after each move. Verified by simulated play in "
            f"{result.iterations} iteration(s)."
        )
    reasons = "; ".join(str(h.get("reason") or "") for h in result.history[-2:])
    return (
        f"Building a {_domain_of(spec)} still failed functional play after {result.iterations} "
        f"iterations (last: {reasons}). Next time, resolve clicks to the square element before "
        "reading coordinates, and keep the board data-model and DOM in sync on every move."
    )


def _slug(spec: str) -> str:
    return (re.sub(r"[^a-z0-9]+", "_", spec.lower()).strip("_") or "app")[:48]


__all__ = ["VerifiedBuildResult", "build_app_verified"]
