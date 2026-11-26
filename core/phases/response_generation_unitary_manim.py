"""How an answer that holds a manim scene is rendered, off the turn.

Lifted whole out of `response_generation_unitary`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import asyncio


def _manim_source_for(response_text: str) -> tuple[str, str]:
    """Manim source for this answer, and the scene class inside it.

    Returns ``("", "")`` when nothing usable came back, which is not a failure
    worth telling anyone about: the answer stands on its own and the animation
    was never promised.
    """
    from .response_generation_unitary import (
        _A_SCENE_CLASS,
        logger,
    )

    try:
        from core.brain.llm.code_generator import LLMCodeGenerator
    except ImportError:
        return "", ""

    generator = LLMCodeGenerator(prefer_tier="primary", max_tokens=1400, temperature=0.2)
    try:
        code = generator.generate(
            "Write a Manim scene that draws the working in this answer, as one "
            "class inheriting from Scene with a construct method and no other "
            f"top-level code:\n\n{response_text[:1200]}",
            {"is_background": True, "language": "python"},
        )
    except (RuntimeError, TimeoutError, TypeError, ValueError) as exc:
        logger.debug("no Manim source for this answer: %s", exc)
        return "", ""

    found = _A_SCENE_CLASS.search(str(code or ""))
    if not found:
        # A scene the renderer cannot name is a scene it cannot render, and
        # handing it over would fail inside the subprocess instead of here.
        return "", ""
    return str(code), found.group(1)


def _render_manim_in_background(response_text: str) -> None:
    """Render a Manim animation for this answer, on its own thread.

    Lifted out of `UnitaryResponsePhase.execute`, where it was a closure over
    a single string. It runs on a plain thread with its own event loop and
    touches no phase state, which is exactly why it did not need to be
    nested — and being nested inside a 3,000-line method is how a
    self-contained side quest becomes part of the response path's apparent
    complexity.

    Owns the release of `_MANIM_RENDER_LOCK`: the caller acquires it before
    starting the thread, so this function must release it on every path or
    the next render never starts.
    """
    from .response_generation_unitary import (
        _MANIM_RENDER_LOCK,
        _RESPONSE_RECOVERABLE_ERRORS,
        _manim_source_for,
        _record_response_degradation,
        logger,
    )

    try:
        from core.skills.manim_renderer import ManimInput, ManimRendererSkill

        skill = ManimRendererSkill()
        # The renderer takes Manim source and the name of the scene in it.
        # This passed `task=` and `timeout_seconds=`, which are not fields on
        # its input at all, so every autonomous render since it was written
        # died on validation — while the answer it belonged to told the person
        # an animation was on its way. LIVE, 2026-09-10: "2 validation errors
        # for ManimInput" on a percentage question.
        source, scene = _manim_source_for(response_text)
        if not source or not scene:
            return
        params = ManimInput(python_code=source, scene_name=scene, quality="l")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            res = loop.run_until_complete(skill.safe_execute(params))
        finally:
            asyncio.set_event_loop(None)
            loop.close()
        if isinstance(res, dict) and res.get("ok"):
            logger.info(
                "Autonomous Manim generation complete: %s", res.get("file_path")
            )
            try:
                from core.thought_stream import get_emitter

                get_emitter().emit(
                    "Pedagogy",
                    f"Visual render complete: {res.get('file_path')}",
                    level="success",
                    category="Media",
                )
            except _RESPONSE_RECOVERABLE_ERRORS as emit_exc:
                _record_response_degradation(
                    emit_exc,
                    "UnitaryResponse: Manim completion emission skipped: %s",
                )
    except _RESPONSE_RECOVERABLE_ERRORS as exc:
        _record_response_degradation(
            exc, "UnitaryResponse: autonomous Manim failed: %s"
        )
    finally:
        _MANIM_RENDER_LOCK.release()


