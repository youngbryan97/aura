"""One question at a time: where she stands on it, what she says, and that she is still moving.

Lifted out of `sovereign_browser_understanding`, which had reached the size a
new module may not exceed. Everything here is used by that module and by the
pursuit loop; nothing here imports either of them at import time.
"""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable, Mapping
from contextvars import ContextVar
from typing import Any

from core.runtime.errors import record_degradation

__all__ = [
    "SAYING_IT_MOVES",
    "measure_the_screen",
    "note_the_size_of_her_mind",
    "while_she_writes",
    "SCREENS_MEASURED",
    "HER_MIND_THIS_PURSUIT",
]

#: How a pursuit running here says it is still getting somewhere, for the
#: model calls made inside it. Set by `_handle_pursue`; None outside one.
SAYING_IT_MOVES: ContextVar[Callable[[str], None] | None] = ContextVar(
    "aura_pursuit_says_it_moves", default=None
)

#: The screens a pursuit has measured whole, by the questions on them. Set by
#: `_handle_pursue`; outside one nothing is kept. See `measure_the_screen`.
SCREENS_MEASURED: ContextVar[dict[tuple[Any, ...], list[dict[str, Any]]] | None] = ContextVar(
    "aura_pursuit_screens_measured", default=None
)

#: Her assembled mind for the pursuit in progress, built once. Set by
#: `_handle_pursue`; outside one each call builds its own.
#:
#: Each build draws a fresh nonce for the blocks it fences as data, and the
#: first of those sits about 1,470 tokens into the prompt. Rebuilt every round,
#: LIVE 2026-10-02, every page decision of a psych run diverged from the cached
#: prompt there and read the other 75-82% of it again: 4,000 to 7,000 tokens a
#: decision, a minute and a half each on her 27B.
HER_MIND_THIS_PURSUIT: ContextVar[dict[str, str] | None] = ContextVar(
    "aura_pursuit_her_mind", default=None
)

#: How long her assembled mind was the last time it was built, in characters.
#: Part of what a page's own text has to fit beside. See `_room_for_page_text`.
_LAST_MIND_CHARS = 0


def _room_for_page_text(rest_chars: int, answer_tokens: int) -> int:
    """Characters of a page's own text that fit beside everything else she reads.

    Her lane's window, less her assembled mind, the rest of the rendered page,
    and the room her answer and its private channel need — all measured. A
    fixed 900 characters used to stand here, and LIVE 2026-10-01 the line her
    result was in ("your personality type INTJ") sat past it: she said the last
    scale was below the fold and pressed "more" three times. Zero when it
    cannot be measured, and the caller falls back.
    """
    try:
        from core.brain.llm.model_registry import PRIMARY_ENDPOINT, get_lane_context_window
        from core.brain.llm.thinking_reserve import reserve_tokens
        from core.brain.llm.token_budget_evidence import chars_per_token

        ratio = chars_per_token()
        window = ratio.tokens_to_chars(int(get_lane_context_window(PRIMARY_ENDPOINT)))
        answer = ratio.tokens_to_chars(int(answer_tokens) + int(reserve_tokens()))
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("sovereign_browser.page_room", exc, severity="info")
        return 0
    return max(0, window - _LAST_MIND_CHARS - int(rest_chars) - answer)


#: How often her model's worker reports reading or writing, at least
#: (`_should_emit_generation_progress` in the worker). Looking more often
#: finds nothing new; looking less often only notices progress later.
_HER_MODEL_SPEAKS_UP_EVERY_S = 1.5


def _when_her_model_last_moved() -> float:
    """The latest moment any of her model processes read or wrote a token."""
    try:
        from core.brain.llm.mlx_client import clients_snapshot
    except ImportError:
        return 0.0
    latest = 0.0
    for _key, client in clients_snapshot():
        try:
            status = client.get_lane_status()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
        for stamp in ("last_token_progress_at", "last_prefill_progress_at"):
            try:
                latest = max(latest, float(status.get(stamp) or 0.0))
            except (TypeError, ValueError):
                continue
    return latest


@contextlib.asynccontextmanager
async def while_she_writes(what: str) -> AsyncIterator[None]:
    """Say the pursuit is moving each time her model reads or writes, while this runs.

    The executor ends an action that is silent for its ceiling, and one model
    call is not silent while it decodes. LIVE 2026-10-01: her forecast took
    599 seconds at 3.3 tokens a second, the pursuit reported nothing for the
    whole of it, and at 600 the run was stopped before the first question.
    Only real reading and writing counts: a generation that stops moving is
    still silence, and still ends the run.
    """
    say = SAYING_IT_MOVES.get()
    if say is None:
        yield
        return

    async def _watch() -> None:
        seen = _when_her_model_last_moved()
        while True:
            await asyncio.sleep(_HER_MODEL_SPEAKS_UP_EVERY_S)
            moved = _when_her_model_last_moved()
            if moved > seen:
                seen = moved
                say(what)

    watcher = asyncio.create_task(_watch())
    try:
        yield
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher

#: A decision round must never take the browser down with it. The loop can
#: always report a failed round and stop; it can never leave a live lease and a
#: half-driven page behind because the model call raised.


def note_the_size_of_her_mind(mind: str) -> None:
    """Record how long her assembled mind came out, for `_room_for_page_text`."""
    global _LAST_MIND_CHARS
    _LAST_MIND_CHARS = len(mind or "")


def _thinking_for_one_answer(
    skill: Any,
    goal: str,
    theme: list[dict[str, Any]],
    mind: str,
    item: dict[str, Any],
    resolved: dict[str, Any],
    decision: dict[str, Any],
    *,
    on_progress: Callable[[str], None] | None = None,
) -> Callable[[], Any]:
    """Her thinking about one answer, made when that answer is about to be given.

    Returns a coroutine function. Awaited, it asks her about this item with
    its theme in view, writes what she said into ``resolved`` and into the
    decision's record, and returns the line to say before the click.
    """

    async def _think() -> str:
        if resolved.get("said"):
            return str(resolved["said"])
        try:
            said = await skill._her_thinking_about(goal, theme, mind, about=item)
        except (RuntimeError, ValueError, TypeError, KeyError, OSError, TimeoutError) as exc:
            # The placement stands without its sentence: the answer falls
            # back to what in her decided it.
            record_degradation(
                "sovereign_browser.theme",
                exc,
                severity="warning",
                action=f"placed question {item['group']} without her words for it",
            )
            said = {}
        hers = str(said.get(item["group"]) or "")
        options = item["options"]
        index = item["index"]
        lean = item["lean"]
        # Where her own sentence puts her, read by entailment, where it can be.
        # See `core.self.how_her_words_stand`: her considered words are the
        # better evidence, and the record's place is kept where they say
        # nothing that can be read.
        placed = (
            await asyncio.to_thread(_where_her_words_place_her, skill, item, hers)
            if hers
            else None
        )
        if placed is not None and placed != index and options[placed].get("selector"):
            index = placed
            resolved["selector"] = str(options[index]["selector"])
        leaning = item["second"] if lean.toward > 0 else item["first"]
        # An answer is never said bare.
        #
        # Her sentence first; the thing in her that decided it where she
        # said nothing; and where even that is empty — a pass that failed
        # outright — the placement itself, in words. LIVE 2026-09-29:
        # "works best in groups … works best alone — 3 of 5, between
        # "works best in groups" and "works best alone"." and nothing
        # after it, which reads as an answer with no reason behind it.
        why = (
            hers
            or next(iter(lean.because), "")
            or f'this sits nearer "{leaning}" for me than the other side'
        )
        # Her own words held against the place her record gave. The place
        # stands, because it is the measurement; a sentence that leans the
        # other way is noticed and reported beside it. LIVE 2026-09-28:
        # "3 of 5 ... I genuinely hold a strong preference for
        # externalized structure", and nothing noticed.
        disagrees = (
            skill._the_choice_disagrees_with_its_reason(options, index, hers)
            if hers
            else ""
        )
        words = skill._an_answer_in_words(options, index, why)
        asks, picked = _the_question_and_the_answer(skill, options, index)
        resolved["said"] = words
        resolved["why"] = why
        resolved["because"] = list(lean.because)
        decision["answered"].append(words)
        if disagrees:
            decision["noticed"].append(disagrees)
        decision["why"] = "; ".join(
            dict.fromkeys(
                str(done.get("why") or "")
                for done in decision["resolved_actions"]
                if done.get("why")
            )
        )[:400]
        if on_progress is not None:
            on_progress("a question thought about")
        return (
            words,
            {"asks": asks, "chose": picked, "said": " ".join(why.split())},
            str(resolved.get("selector") or ""),
        )

    return _think


def _where_her_words_place_her(skill: Any, item: Mapping[str, Any], hers: str) -> int | None:
    """The position her own sentence puts her at on this question, or None."""
    from core.self import how_her_words_stand as reading
    from core.self.where_i_stand import Lean

    options = item["options"]
    lean = item["lean"]
    if getattr(lean, "facing", 0.0):
        shape = skill._a_statement_on_a_named_scale(options)
        borne = None if shape is None else reading.whether_her_words_bear_it_out(hers, shape[0])
        toward = None if borne is None else borne * lean.facing
    else:
        sides = skill._the_two_sides(options)
        toward = None if sides is None else reading.where_her_words_put_her(hers, *sides)
    if toward is None:
        return None
    return Lean(toward=toward, first=0.0, second=0.0, measured=True).position_in(len(options))


def _the_question_and_the_answer(
    skill: Any, options: list[Mapping[str, Any]], index: int
) -> tuple[str, str]:
    """The question and the answer, apart: the two halves of `_an_answer_in_words`."""
    joined = skill._an_answer_in_words(options, index, "")
    question, _dash, picked = joined.rpartition(" \u2014 ")
    return (question, picked) if question else ("", joined)


async def measure_the_screen(
    skill: Any,
    questions: list[tuple[Any, list[dict[str, Any]]]],
    *,
    among: list[tuple[Any, list[dict[str, Any]]]] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Where her record puts her on each of ``questions``, measured among ``among``.

    A question is placed against the others on its page: the strongest lean
    there is as far as she goes, and a statement in a grid is placed by how
    it stands beside the rest of the grid. So what it is measured among is
    every question on the screen, answered or not. Measuring each round's
    batch on its own gave a statement different neighbours from one round to
    the next. LIVE 2026-10-02, "I study how to hold on to my money" read
    +0.99 against the whole page and was answered Disagree, measured among
    the last four statements left open.

    The whole screen is measured once in a pursuit and its questions answered
    from that, however many rounds it takes. Measured on her record, one
    screen of thirty-two questions is 17 s.
    """
    whole = list(among or [])
    listed = {str(group) for group, _options in whole}
    whole += [(group, options) for group, options in questions if str(group) not in listed]
    screen = tuple(
        (str(group), str((options[0] if options else {}).get("asks") or ""), len(options))
        for group, options in whole
    )
    kept = SCREENS_MEASURED.get()
    everything = kept.get(screen) if kept is not None else None
    if everything is None:
        from core.self.where_i_stand import one_measurement

        with one_measurement():
            everything = await _every_question_measured(skill, whole, on_progress=on_progress)
        if kept is not None:
            kept[screen] = everything
    current = {str(group): options for group, options in questions}
    return [
        dict(item, options=current[item["group"]])
        for item in everything
        if item["group"] in current
    ]


async def _every_question_measured(
    skill: Any,
    questions: list[tuple[Any, list[dict[str, Any]]]],
    *,
    on_progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Every question placed from her record, against the others; see `measure_the_screen`."""
    asked_now = list(questions)
    # A grid of statements is measured as a grid, because that is what it is.
    #
    # One statement on an agree scale has no second thing to be weighed
    # against, so the per-question reader can make nothing of it; the whole
    # column of them supplies the contrast, and the scale's direction is a
    # property of the page rather than of any one row. Measured together
    # before anything else, then merged back in the page's own order.
    on_a_grid = await asyncio.to_thread(skill._a_grid_of_statements, asked_now)
    measured: list[dict[str, Any]] = []
    for group, options in asked_now:
        # One question her record cannot be read against is one question
        # left open, and said so. It used to be the whole screen: the
        # first raise out of this loop discarded every answer before it.
        if group in on_a_grid:
            reading = on_a_grid[group]
        else:
            try:
                reading = await asyncio.to_thread(skill._measure_where_she_stands, options)
            except (RuntimeError, ValueError, TypeError, KeyError, IndexError, OSError) as exc:
                record_degradation(
                    "sovereign_browser.question",
                    exc,
                    severity="warning",
                    action=f"left question {group} open and answered the rest",
                )
                continue
        if reading is None:
            continue
        index, lean, first, second = reading
        measured.append(
            {
                "group": str(group),
                "options": options,
                "index": index,
                "count": len(options),
                "lean": lean,
                "first": first,
                "second": second,
            }
        )
        if on_progress is not None:
            on_progress("a question measured")
    if measured:
        # Placed against the strongest of them, not each on its own.
        #
        # How consistently her record points one way saturates: twenty
        # things all a hair closer to one side read the same as twenty
        # decisively closer, and a page came out with thirty-four of sixty
        # items at the far end, which is not a person answering a
        # questionnaire. The questions on a screen are all asked of the
        # same record, so the widest gap among them is what "as far as she
        # goes" means here and the rest are placed in proportion.
        try:
            from dataclasses import replace

            from core.self.where_i_stand import against_the_rest

            shares = against_the_rest([item["lean"] for item in measured])
            for item, share in zip(measured, shares, strict=False):
                item["lean"] = replace(item["lean"], toward=share)
                placed = item["lean"].position_in(item["count"])
                if placed is not None:
                    item["index"] = placed
        except ImportError as exc:
            record_degradation("sovereign_browser.against", exc, severity="debug")
    return measured
