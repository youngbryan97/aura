"""Reading a page, and deciding what to do about it.

Everything between an observation arriving and an action being chosen: what the
page says, which of its controls are worth offering, what she already knows
about the place, which of her questions it answered, and the decision itself
once all of that is in hand. The half of the skill that drives a browser is in
sovereign_browser.py; this is the half that thinks about what came back.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable, Mapping
from typing import Any

from core.conversation.word_markers import names_any
from core.runtime.errors import record_degradation
from core.runtime.service_access import optional_service
from core.runtime.structured_input import A_CLOSED_QUESTIONS_FLOOR

from .sovereign_browser_one_question import (  # noqa: F401  (re-exported: the pursuit and tests read them here)
    SAYING_IT_MOVES,
    SCREENS_MEASURED,
    _room_for_page_text,
    _the_question_and_the_answer,
    _thinking_for_one_answer,
    measure_the_screen,
    note_the_size_of_her_mind,
    while_she_writes,
)
from .sovereign_browser_understanding_scale import _PlacesHerself

logger = logging.getLogger("Skills.SovereignBrowser")

_BROWSER_DECISION_ERRORS = (
    AttributeError,
    ConnectionError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_ADVANCING_BUTTON_WORDS = ("next", "continue", "submit", "finish", "start")


class _UnderstandsThePage(_PlacesHerself):
    """Lifted whole from SovereignBrowserSkill; see sovereign_browser.py."""

    @staticmethod
    def _observation_signature(observation: Mapping[str, Any]) -> str:
        """What would have to change for progress to have been made.

        Every control, not the first sixty. Rehearsed on 27 Sep against OEJTS
        page one (161 controls): answering questions 13 to 32 changed nothing
        the signature read, the loop counted those rounds as stalled, and it
        ended after two of them with the page answered and Next unpressed.
        """
        elements = observation.get("elements") or []
        marks = "|".join(
            f"{element.get('role')}:{element.get('name')}:{element.get('checked')}"
            for element in elements
        )
        return f"{observation.get('url')}#{marks}"

    @classmethod
    def _controls_worth_offering(
        cls, elements: list[Any], goal: str = ""
    ) -> list[Any]:
        """The controls that can advance a goal, before the ones that decorate.

        Truncating the raw list would cut the answers and keep the navigation,
        because site furniture is emitted first in document order. Ranking by
        what a control DOES keeps the form and drops the chrome.

        And by what the GOAL names, ahead of that. Ranking by role alone is
        blind to the errand: LIVE 2026-09-28, asked to take the Open Extended
        Jungian Type Scales, she arrived at openpsychometrics.org — an index of
        forty-odd tests, no form on it — and the link to the one she was sent
        for sat below a budget built for questionnaires. The page she could see
        held no test and no way to one. A control whose words answer the goal's
        words is the way in, wherever the document happens to put it.
        """
        # A question that is answered offers nothing.
        #
        # Labelling the six unselected options of a finished question as
        # "already answered" and then offering them anyway is still offering
        # them, and they were chosen: measured live, question 8 and question 10
        # were each answered four separate times while questions further down
        # the same screen were never reached at all. Worse, they are not free —
        # one screen of six questions renders 42 radios against a budget of 40,
        # so the answered ones were crowding the unanswered ones out of the
        # list entirely.
        #
        # Their answers stay visible in the page text, which is where reading
        # what she has said belongs. What is offered here is what is left to do.
        answered = {
            str(element.get("group"))
            for element in elements
            if isinstance(element, Mapping)
            and element.get("group")
            and element.get("checked") is True
        }
        live = [
            element
            for element in elements
            if not (
                isinstance(element, Mapping)
                and element.get("group")
                and str(element.get("group")) in answered
            )
        ]
        wanted = cls._words_of(goal)
        ranked = sorted(
            enumerate(live),
            key=lambda pair: (
                -cls._how_far_it_answers(pair[1], wanted),
                cls._ACTIONABLE_ROLES.index(str(pair[1].get("role") or "").lower())
                if str(pair[1].get("role") or "").lower() in cls._ACTIONABLE_ROLES
                else len(cls._ACTIONABLE_ROLES),
                pair[0],
            ),
        )
        offered = [element for _index, element in ranked[: cls.PURSUE_CONTROL_BUDGET]]
        # What the page draws is the page's content, not furniture cut for room:
        # LIVE 2 Oct a game's canvas fell below forty links carrying the goal's
        # own word "game", and she could not choose the thing she was sent to play.
        return offered + [
            element
            for _index, element in ranked[cls.PURSUE_CONTROL_BUDGET:]
            if isinstance(element, Mapping) and str(element.get("role") or "") == "drawing"
        ]

    #: Words that name nothing in particular, so sharing one says nothing.
    _SAYS_NOTHING = frozenset({
        "the", "a", "an", "and", "or", "of", "to", "for", "on", "in", "at", "by",
        "with", "from", "it", "its", "this", "that", "your", "you", "me", "my",
        "i", "is", "are", "be", "as", "test", "page", "site", "com", "org",
        "www", "http", "https", "take", "tell", "say", "go", "get", "them",
        "they", "what", "when", "why", "how", "each", "before", "after", "start",
    })

    @classmethod
    def _words_of(cls, text: str) -> frozenset[str]:
        """The distinctive words of a phrase, lowercased."""
        return frozenset(
            word
            for word in re.findall(r"[a-z0-9]+", str(text or "").lower())
            if len(word) >= 3 and word not in cls._SAYS_NOTHING
        )

    @classmethod
    def _how_far_it_answers(cls, element: Any, wanted: frozenset[str]) -> int:
        """How many of the goal's own words this control carries.

        A count, not a score: two words shared is twice the evidence of one,
        and nothing here needs a weighting nobody measured.
        """
        if not wanted or not isinstance(element, Mapping):
            return 0
        said = cls._words_of(
            " ".join(
                str(element.get(field) or "")
                for field in ("name", "value", "asks", "heading")
            )
        )
        return len(wanted & said)

    # A classmethod rather than a static one because it reads two budgets off
    # the class. It used to name the class to get at them, which is the same
    # dependency written in a way that breaks the moment the method moves.
    @classmethod
    def _render_observation(
        cls, observation: Mapping[str, Any], goal: str = ""
    ) -> str:
        """The page as the decision sees it: what it says, and what it offers.

        The goal travels with it because the ranking below depends on it, and
        every index the decision writes is resolved against the same list in
        the pursuit loop. A list ranked one way here and another way there
        means `[3]` names one control on screen and a different one in the
        click.
        """
        elements = cls._controls_worth_offering(
            list(observation.get("elements") or []), goal
        )
        lines = [
            f"URL: {observation.get('url')}",
            f"Title: {observation.get('title')}",
            # The time is a fact about the world like the address is. A request
            # that turns on it ("the last digit of the time, plus two") could
            # not be followed from a page that does not show a clock.
            f"Now: {time.strftime('%A %d %B %Y, %H:%M:%S %Z')}",
            "",
            "PAGE TEXT:",
            "",  # filled in last, with the room the rest leaves
            "",
            "AVAILABLE CONTROLS:",
        ]
        # Answered questions are gone from this list rather than annotated in
        # it — see `_controls_worth_offering`. What remains is what is left to
        # do, so a screen half-finished reads as a shorter screen.
        shown_asking: set[str] = set()
        for index, element in enumerate(elements):
            group = str(element.get("group") or "")
            if group and group not in shown_asking and (element.get("asks") or element.get("heading")):
                # The question itself, once, above its options: the words
                # around them as the page lays them out, and the heading of
                # its table. Unlabelled options mean nothing without it.
                shown_asking.add(group)
                if element.get("heading"):
                    lines.append(f"question {group} sits under: {element['heading']}")
                if element.get("asks"):
                    lines.append(f"question {group} reads: {element['asks']}")
                # And what the options ARE, where they are a scale rather than
                # a list. The page draws five dots between two opposing
                # phrases and says nowhere that position is the answer.
                laid_out = cls._how_the_options_are_laid_out(
                    [
                        option
                        for option in elements
                        if isinstance(option, Mapping)
                        and str(option.get("group") or "") == group
                    ]
                )
                if laid_out:
                    lines.append(f"question {group} offers: {laid_out}")
            state = []
            if element.get("group"):
                # Options in one group answer ONE question. Rendering it is
                # what lets a whole screen be answered in a single round
                # instead of one control at a time.
                state.append(f"question {element['group']}")
                # And where this one sits in the run, where the run is all this
                # question has. A control whose only name is its group's reads
                # as one nameless thing among five, and picking item k from a
                # list is not the same act as saying where in a range you are.
                # The place is a fact of the layout; what being there means is
                # hers.
                kin = [
                    option
                    for option in elements
                    if isinstance(option, Mapping)
                    and str(option.get("group") or "") == str(element["group"])
                ]
                named = {str(option.get("name") or "").strip() for option in kin}
                named.discard("")
                if len(kin) > 2 and (
                    len(named) != len(kin) or named == {str(element["group"])}
                ):
                    state.append(f"position {kin.index(element) + 1} of {len(kin)}")
            if element.get("checked") is True:
                state.append("already answered")
            if element.get("value"):
                state.append(f"value={element['value']}")
            suffix = f" ({', '.join(state)})" if state else ""
            lines.append(
                f"[{index}] {element.get('role')} \u2014 {element.get('name')}{suffix}"
            )
        # What was left out, and what is below the fold.
        #
        # Neither was said, so a page whose list had been cut and a page with
        # nothing more on it read exactly alike, and scrolling was a guess. The
        # observer keeps both numbers and nothing showed them.
        held = len(list(observation.get("elements") or []))
        if held > len(elements):
            lines.append(
                f"({held - len(elements)} more control(s) on this page are not "
                "listed; the ones most likely to serve the goal are)"
            )
        below = (
            int(observation.get("scroll_height") or 0)
            - int(observation.get("scroll_y") or 0)
            - int(observation.get("viewport_height") or 0)
        )
        if below > 0 and int(observation.get("viewport_height") or 0) > 0:
            screens = below / float(observation["viewport_height"])
            lines.append(f"(the page continues {screens:.1f} screen(s) below this one)")
        # The page's own words, as many as fit beside the rest. Where they do
        # not all fit, that is said, so a cut and an end read differently.
        text = str(observation.get("text") or "")
        room = _room_for_page_text(sum(len(line) + 1 for line in lines), cls.DECISION_MAX_TOKENS)
        room = room or cls.PURSUE_TEXT_BUDGET
        shown = text[:room]
        if len(text) > room:
            shown += f"\n(the page's text goes on for {len(text) - room} more characters)"
        lines[4] = shown
        return "\n".join(lines)

    @staticmethod
    def _page_shape(observation: Mapping[str, Any]) -> str:
        """What KIND of page this is, independent of whose page it is.

        Knowing "16personalities.com is a questionnaire" helps exactly once.
        Knowing "a page with repeated radio groups and a next control is a
        multi-page form: answer the visible items, then advance" helps on every
        survey, application and signup wizard she ever meets.

        So the fingerprint is structural — which control roles are present,
        whether they repeat, whether something advances — and deliberately
        carries no site text, because the moment it does it stops transferring.
        """

        elements = observation.get("elements") or []
        roles: dict[str, int] = {}
        for element in elements:
            role = str(element.get("role") or "").lower()
            if role:
                roles[role] = roles.get(role, 0) + 1
        parts: list[str] = []
        for role in ("radio", "checkbox", "text", "textarea", "select", "button", "link"):
            count = roles.get(role, 0)
            if not count:
                continue
            # Bucketed, not counted: "many radios" is the fact that transfers,
            # not "forty-two of them".
            parts.append(f"{role}:{'many' if count > 6 else 'few'}")
        # Word boundaries: a "Restart" button is not a "start" button, and an
        # "Unfinished" label is not a "finish" one.
        advances = any(
            names_any(str(element.get("name") or ""), _ADVANCING_BUTTON_WORDS)
            for element in elements
            if str(element.get("role") or "") == "button"
        )
        if advances:
            parts.append("advances")
        return "|".join(parts) or "plain"

    @staticmethod
    def _recall_about(url: str, shape: str) -> str:
        """What she already knows about this place, and about places like it.

        Written knowledge that is never read back is a diary, not learning. The
        world model persists across restarts, so a pursuit begins by asking
        what she worked out last time — for this host, and for any page of this
        SHAPE, which is the half that generalises.
        """

        try:
            from urllib.parse import urlsplit


            world = optional_service("world_model", default=None)
            beliefs = getattr(world, "beliefs", None)
            if not isinstance(beliefs, dict) or not beliefs:
                return ""
            host = urlsplit(url).netloc if url else ""
            remembered: list[str] = []
            for node in beliefs.values():
                tags = set(getattr(node, "tags", ()) or ())
                if "page_model" not in tags:
                    continue
                claim = str(getattr(node, "claim", "") or "")
                if not claim:
                    continue
                if (host and host in claim) or (shape and shape in tags):
                    remembered.append(f"- {claim}")
            if not remembered:
                return ""
            return "WHAT I ALREADY KNOW ABOUT PAGES LIKE THIS:\n" + "\n".join(remembered[:6])
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation("sovereign_browser.recall", exc, severity="debug")
            return ""

    @staticmethod
    def _learn_from_surprise(shape: str, expected: str, observation: Mapping[str, Any]) -> None:
        """Record what actually happens, when it was not what she expected.

        A surprise is the most informative thing that happens in a task, and
        the old loop discarded it — it counted an unchanged screen and stopped.
        Written against the SHAPE, so the correction applies to the next page
        of this kind rather than only to this one.
        """

        if not shape or not expected:
            return
        try:

            world = optional_service("world_model", default=None)
            if world is None or not hasattr(world, "add_belief"):
                return
            world.add_belief(
                (
                    f"on a page shaped {shape}, expecting \u201c{expected[:120]}\u201d "
                    "did not change the page"
                )[:400],
                0.55,
                source_id="browser_pursuit:surprise",
                tags=["web", "page_model", "correction", shape],
            )
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation("sovereign_browser.learn", exc, severity="debug")

    async def _her_identity_only(self) -> str:
        """Her identity core, without the live moment.

        The whole assembled mind carries the moment she is in — the affect, the
        workspace winner, what just happened — and that changes between one
        call and the next, so the prompt cache diverges a few hundred tokens in
        and every per-item pass re-prefills four thousand tokens. Measured on
        the live worker: prefill 4736 tokens at 200/s and decode at 8/s, about
        a minute an item, which is half an hour for a page of thirty-two.

        The identity core is the same on every item of a screen, so the prefill
        is paid once and the rest of the page is decode alone. What she needs
        about herself for a self-report is not the moment; it is in the record
        she is handed with the question.
        """
        try:
            from core.brain.llm.context_assembler import AURA_IDENTITY, get_identity_lock

            return f"{get_identity_lock()}\n\n[GROUNDED CORE PROTOCOL]\n{AURA_IDENTITY}\n"
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation(
                "sovereign_browser.identity",
                exc,
                severity="debug",
                action="answered without her identity core",
            )
            return ""

    async def _assembled_mind(self) -> str:
        """Her whole mind, built once for the pursuit rather than per round.

        The same assembly the cognitive engine and inference gate use for chat:
        identity core and trained persona, the AuraNow moment with its affect
        and ownership, the global-workspace winner, and the report boundary.
        Deciding through it is what makes an action here the same kind of act
        as an answer in conversation.
        """

        try:
            from core.brain.llm.context_assembler import ContextAssembler

            # Her live state, from the repository that holds it.
            #
            # `aura_state` is registered by nothing: this asked for a key that
            # does not exist, got None, and returned an empty mind on every
            # call — 49 times in one boot, each recorded and none read. The
            # inference gate builds the same prompt from the state repository's
            # current state, which is where it actually lives.
            state = optional_service("aura_state", default=None)
            if state is None:
                repo = optional_service("state_repository", "state_repo", default=None)
                state = (
                    getattr(repo, "_current", None)
                    or getattr(repo, "_current_state", None)
                    if repo is not None
                    else None
                )
            if state is None:
                record_degradation(
                    "sovereign_browser.mind_context",
                    RuntimeError("no_current_state"),
                    severity="warning",
                    action="decided without her assembled self-context",
                )
                return ""
            mind = ContextAssembler.build_system_prompt(state)
            note_the_size_of_her_mind(mind)
            return mind
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation(
                "sovereign_browser.mind_context",
                exc,
                severity="warning",
                action="decided without her assembled self-context",
            )
            return ""

    @staticmethod
    def _remember_the_place(
        url: str, understanding: Mapping[str, Any] | None, shape: str = ""
    ) -> None:
        """Put what she worked out about this site where the rest of her can see it.

        A page model held in a local variable dies with the task and teaches
        nothing. The world model already feeds `get_context_injection`, which
        the context assembler injects into every later turn, so what she
        learned about a place is available the next time she is there — and to
        whatever else is reasoning about it.
        """

        if not understanding or not url:
            return
        here = str(understanding.get("here") or "").strip()
        to_progress = str(understanding.get("to_progress") or "").strip()
        if not here:
            return
        try:
            from urllib.parse import urlsplit


            world = optional_service("world_model", default=None)
            if world is None or not hasattr(world, "add_belief"):
                return
            host = urlsplit(url).netloc or url
            claim = f"{host} is {here}"
            if to_progress:
                claim = f"{claim}; to make progress there you {to_progress}"
            world.add_belief(
                claim[:400],
                0.7,
                source_id=f"browser_pursuit:{host}",
                tags=["web", "page_model"],
            )
            # And the transferable half. The host belief helps here; this one
            # helps on the next survey, application or wizard she meets.
            if shape and to_progress:
                world.add_belief(
                    (f"a page shaped {shape} is {here}; there you {to_progress}")[:400],
                    0.6,
                    source_id="browser_pursuit:shape",
                    tags=["web", "page_model", shape],
                )
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation("sovereign_browser.world_model", exc, severity="debug")

    @staticmethod
    def _record_expectation_outcome(expected: str, moved: bool) -> None:
        """Whether the page did what she thought it would.

        The loop used to notice only that nothing had changed and stop, calling
        it `no_progress` — a step tally. Having an expectation and finding it
        violated is a different thing: it is the signal that the understanding
        is wrong, and it is the same predict/observe/error currency the rest of
        the runtime already keeps.
        """

        if not expected:
            return
        try:

            calibration = optional_service("calibration_engine", default=None)
            recorder = getattr(calibration, "record_prediction", None)
            if callable(recorder):
                recorder(0.75, 1.0 if moved else 0.0)
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation("sovereign_browser.calibration", exc, severity="debug")

    async def _what_she_expects_it_to_say(
        self,
        goal: str,
        observation: Mapping[str, Any],
        mind: str,
        measured: list[dict[str, Any]] | None = None,
    ) -> str:
        """What she expects this instrument to conclude about her, before she answers it.

        A forecast made before arriving is made from nothing: she does not yet
        know what the thing measures, and "what score will you get" has no
        answer until the page says what it reports. Read here instead, from
        what this page says it is and what it measures, and said before the
        first item is answered — so the outcome at the end has something real
        to be held against.

        General to any instrument that will report on her, whatever it reports:
        a type, a set of scales, a percentile, a sentence. Where a page names
        no scores at all she infers what it may say about her from what it says
        it is for, which is the same thing a person does with a test they have
        not taken.
        """
        # Where her record already puts her on the questions in front of her,
        # measured with no model in it (`measure_the_screen`). Evidence, not an
        # instruction: a forecast made without it was the model's guess about
        # her, and the run that made one predicted a test the page did not
        # describe.
        placed = "\n".join(
            f"- {asks} \u2014 {picked}"
            + (f" ({item['lean'].because[0]})" if item["lean"].because else "")
            for item in (measured or [])
            for asks, picked in [_the_question_and_the_answer(self, item["options"], item["index"])]
        )
        prompt = (
            # Her situation, not the person's message: a goal is a request
            # addressed to her and she answers it instead of doing the step.
            "You are about to answer an instrument that will report something "
            "about you.\n\n"
            f"{self._render_observation(observation, goal)}\n\n"
            + (
                "Where your own record already places you on its questions:\n"
                f"{placed}\n\n"
                if placed
                else ""
            )
            + (
                "Before you answer anything: from what this page says this is and "
                "what it measures, say what you expect it to conclude about you, "
                "and why you expect that."
            )
        )
        # As much as a line she says can show. A bubble is cut to what can be
        # read while it stays up, so anything longer was written to be thrown
        # away: LIVE 2026-10-01 a forecast decoded 1,854 tokens in ten minutes
        # and the run was stopped before the first question.
        said, lane = await self._asked_of_her(
            prompt, mind, shaped=False, most_tokens=self.REASON_MAX_TOKENS
        )
        if said and lane == self._HER_OWN_LANE:
            return said
        if said:
            record_degradation(
                "sovereign_browser.forecast",
                RuntimeError(f"not_her_own_reasoning:{lane or 'unattributed'}"),
                severity="warning",
                action="did not say what she expected; it did not come from her",
            )
        return ""

    #: What a reason about one item may run to. Two sentences is about sixty
    #: words; the bound is generous against that and still a quarter of what a
    #: decision may write. Measured live, an unbounded reason decoded 341
    #: tokens at 8 a second — forty seconds of a minute-long item.
    REASON_MAX_TOKENS = 220

    async def _asked_of_her(
        self,
        prompt: str,
        mind: str = "",
        *,
        shaped: bool = True,
        most_tokens: int | None = None,
        worked_out_here: bool = True,
        held_to: str = "",
    ) -> tuple[str, str]:
        """One way to ask her something about herself, used by everything that does.

        Her own lane, and only there, with her self-model, her values and her own
        earlier words about herself in front of it.

        Her full conversational cycle was tried here first and is the wrong
        instrument: it exists to produce a reply to the person, and that is what
        it produced. LIVE 2026-09-28 18:13, on the index page before a single
        click: "The user wants me to take the Open Extended Jungian Type Scales
        personality test... Before I start, they want me to tell them what type
        I think I'll get and why. Let me first think about..." — the turn's own
        answer, returned as a page decision and reported unparsable, twice in a
        row, with the run stopping before it opened anything.

        What her faculties have to contribute reaches this as context, which is
        assembled by the caller: what she is like from her record of choices,
        what she has said about herself in her own words, and what her
        instruments say about her right now.

        Built once because three things ask her: what she expects an instrument
        to say before she answers it, how she answers each of its questions, and
        what she makes of the result. Two of them had only one half of this and
        the forecast had no fallback at all, so a cycle that returned nothing
        meant she simply never said what she expected — which is the part of the
        request that kept going missing.

        Returns the text and the lane that produced it; the caller decides what
        an answer from somewhere else is worth.

        ``worked_out_here`` says whether this call is where something gets
        worked out — a forecast, a verdict — or only says what was settled
        before it. Her place on an item is measured from her record before she
        is asked, so her reason for it is the second kind, and the runtime's
        typed lane for that closes the private channel: LIVE 2026-10-01,
        the first three items opened it and decoded 771, 650 and 859 tokens
        in 111, 86 and 123 seconds, for two or three sentences each. Either way ``most_tokens`` is a
        ceiling and not a hint: undeclared, the gate sized a forecast at "room
        for about 675 words" and she wrote 2,046 of them.
        """
        router = optional_service("llm_router", default=None)
        think = getattr(router, "think", None)
        if not callable(think):
            return "", ""
        who: dict[str, Any] = {}
        try:
            async with while_she_writes("her model is writing"):
                reply = await think(
                    prompt,
                    system_prompt=mind,
                    prefer_tier="primary",
                    origin=self._PAGE_ORIGIN,
                    purpose="page_decision" if shaped else "page_forecast",
                    own_lane_required=True,
                    serves_current_turn=True,
                    _generation_metadata_sink=who,
                    max_tokens=most_tokens or self.DECISION_MAX_TOKENS,
                    temperature=0.2 if shaped else 0.4,
                    _non_chat_inference=True,
                    # How much room the ANSWER needs, declared, because a reasoning
                    # model charges its thinking to the same budget.
                    #
                    # Without this the private channel is neither opened nor bounded:
                    # the model reasons anyway, in the answer, and the budget is gone
                    # before it concludes. LIVE 2026-09-29, the verdict on her own
                    # result — the thing the person asked for — decoded all 900 tokens
                    # it was given and returned ten characters, "Okay. Here", and the
                    # reply fell back to reciting the rounds. Declared, the decoder
                    # closes the channel at its bound and the reserve is bought on top,
                    # so what she is asked for is what the budget pays for.
                    user_surface_completion_floor=(
                        max(
                            self._ROOM_AN_ANSWER_NEEDS,
                            int(most_tokens or self.DECISION_MAX_TOKENS),
                        )
                        if worked_out_here
                        else int(most_tokens or self.REASON_MAX_TOKENS)
                    ),
                    hard_output_token_ceiling=True,
                    # The typed lane for "say what was settled": the worker and the
                    # gate's clock both read it, and neither opens the channel.
                    **({} if worked_out_here else {"cognitive_mode": "fast"}),
                    **(
                        {"schema": self._DECISION_SCHEMA, "output_shape": "json_object"}
                        if shaped
                        else ({"output_shape": held_to} if held_to else {})
                    ),
                )
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation(
                "sovereign_browser.asked_of_her",
                exc,
                severity="warning",
                action="asked her something about herself and got nothing back",
            )
            return "", ""
        return (
            self._the_text_of(reply),
            self._who_answered(reply) or str(who.get("endpoint") or ""),
        )

    async def _hold_the_outcome_against_what_she_said(
        self,
        goal: str,
        said_before: str,
        observation: Mapping[str, Any],
        mind: str,
    ) -> str:
        """What she makes of the result, against what she said it would be.

        A task that begins with a forecast does not end when the last control
        is pressed. The person asked for the forecast so that it could be held
        against the outcome, and a forecast never checked is a forecast that
        was decoration.

        `result["concluded"]` was read where the pursuit's account is written
        and written by nothing, so the account ended with the raw tail of the
        final page and no word from her about it. This is general: it runs for
        any goal where she said something before she began, and it asks only
        about her own earlier claim and what is in front of her now. Where she
        said nothing beforehand there is nothing to hold, and it does not run.
        """
        said = " ".join(str(said_before or "").split())
        if not said:
            return ""
        prompt = (
            "You have finished an instrument that reports something about "
            "you.\n\n"
            f"WHAT YOU SAID BEFORE YOU BEGAN: {said}\n\n"
            f"{self._render_observation(observation, goal)}\n\n"
            "You have finished. Read what is in front of you and say, in your "
            "own words, what the outcome was, whether it matches what you said "
            "beforehand, and whether you think it is accurate about you."
        )
        # Her own lane. This is a judgement about her own earlier claim and
        # about a result that describes her; a stand-in answering it would be a
        # different mind grading her forecast.
        verdict, lane = await self._asked_of_her(prompt, mind, shaped=False)
        if verdict and lane != self._HER_OWN_LANE:
            record_degradation(
                "sovereign_browser.conclusion",
                RuntimeError(f"not_her_own_reasoning:{lane or 'unattributed'}"),
                severity="warning",
                action="finished without a verdict that came from her",
            )
            return ""
        return " ".join(verdict.split())

    async def _understand_page(
        self,
        goal: str,
        observation: Mapping[str, Any],
        prior: Mapping[str, Any] | None,
        mind: str,
        recalled: str = "",
    ) -> dict[str, Any]:
        """What she takes this page to be, and what doing the goal here means.

        A step-picker asks "which control advances the goal" every round, from
        nothing, forever. That is not how anyone uses a website. A person
        arrives with an aim, works out what the place IS — a sixty-item survey,
        six to a screen, a seven-point scale, a Next button at the bottom —
        and then acts fluently from that understanding, revising it only when
        the page does something unexpected.

        Without a standing understanding the loop has no answer to "why this
        control and not that one", no idea which controls are present but
        irrelevant, and no way to know it is finished except a step budget. It
        was five rounds of clicking with no view of the whole.

        This is that view, and it is carried across rounds rather than rebuilt:
        what I am ultimately trying to accomplish, what this page is, what it
        requires of me to progress, which controls matter and which are merely
        here, and how I will know I am done.

        Revised, not regenerated. A revision that discards what was already
        worked out is a rebuild wearing another name, so the prior
        understanding is given back to her and she is asked what changed.
        """


        router = optional_service("llm_router", default=None)
        if router is None:
            return dict(prior or {})

        prior_view = ""
        if prior:
            prior_view = (
                "WHAT I ALREADY WORKED OUT ABOUT THIS TASK:\n"
                + json.dumps(dict(prior), indent=None)[:900]
                + "\n\nRevise it only where this page contradicts it.\n\n"
            )

        prompt = (
            f"WHAT I AM TRYING TO ACCOMPLISH: {goal}\n\n"
            + (f"{recalled}\n\n" if recalled else "")
            + f"{prior_view}"
            f"{self._render_observation(observation, goal)}\n\n"
            "Describe the situation, as JSON only:\n"
            '{"here": "<what this page is>", '
            '"to_progress": "<what I have to do on THIS page to move forward>", '
            '"relevant": "<which controls matter and what they do>", '
            '"how_to_answer": "<what this page wants from me for each question: '
            'what its options mean and what choosing one of them says>", '
            '"present_but_not_needed": "<controls that exist here and are not what I need>", '
            '"done_when": "<how I will know the whole task is finished>"}'
        )
        try:
            think = getattr(router, "think", None)
            if callable(think) and mind:
                async with while_she_writes("her model is reading the page"):
                    raw = self._the_text_of(await think(
                        prompt, system_prompt=mind, schema=self._UNDERSTANDING_SCHEMA,
                        output_shape="json_object", serves_current_turn=True,
                        origin=self._PAGE_ORIGIN, purpose="page_understanding",
                        max_tokens=420, temperature=0.2, _non_chat_inference=True,
                    ))
            else:
                generate = getattr(router, "generate", None)
                if not callable(generate):
                    return dict(prior or {})
                raw = await generate(prompt, max_tokens=420, temperature=0.2)
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation("sovereign_browser.understand", exc, severity="debug")
            return dict(prior or {})

        parsed = self._parse_decision(str(raw or ""))
        if parsed.get("error"):
            return dict(prior or {})
        parsed.pop("actions", None)
        merged = dict(prior or {})
        merged.update({key: value for key, value in parsed.items() if value})
        return merged

    @staticmethod
    def _render_understanding(understanding: Mapping[str, Any] | None) -> str:
        """Her standing view of the task, in the order a person would hold it."""
        if not understanding:
            return ""
        rows = [
            ("Where I am", understanding.get("here")),
            ("What this page needs from me", understanding.get("to_progress")),
            ("What matters here", understanding.get("relevant")),
            ("How this page wants to be answered", understanding.get("how_to_answer")),
            ("Here but not what I need", understanding.get("present_but_not_needed")),
            ("I am finished when", understanding.get("done_when")),
        ]
        lines = [f"- {label}: {value}" for label, value in rows if value]
        return "MY UNDERSTANDING OF THIS TASK:\n" + "\n".join(lines) if lines else ""

    @classmethod
    def _decision_is_usable(
        cls, raw: Any, observation: Mapping[str, Any], goal: str = ""
    ) -> bool:
        """Whether this decision names something that can actually be done.

        Not "did the call succeed" — an answer that parses to no action, or to
        an index that is not on the page, leaves the round with nothing to
        execute, which is indistinguishable from no answer at all.
        """

        if not raw:
            return False
        parsed = cls._parse_decision(str(raw))
        if parsed.get("error"):
            return False
        if parsed.get("done") is True:
            return True
        elements = cls._controls_worth_offering(
            list(observation.get("elements") or []), goal
        )
        for item in parsed.get("actions") or []:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(elements):
                return True
        return False

    @staticmethod
    async def _decide_on_the_fast_lane(router: Any, prompt: str, mind: str) -> str | None:
        """One micro-decision on the small model, or None to fall back.

        `think()` on the resolved client is endpoint-level: it builds a payload
        and drops `prefer_tier` and `origin` on the floor, which is why asking
        for the fast lane changed nothing and every round still logged
        "Routing to Cortex (timeout=103s, user_facing=True)". `think_and_act`
        is the tier-aware entry, and with no tools passed it is simply a
        generation on the endpoint the tier selects.

        Returning None rather than raising is the point: if the fast lane is
        unavailable, deferred or empty, the caller falls back to the ordinary
        path. A decision that vanishes stalls the loop; a slower decision only
        costs time.
        """

        # `generate(prefer_tier=...)` on the registered router.
        #
        # The registered service is HealthAwareLLMRouter and this is its own
        # public entry: it takes the tier and honours it. Three earlier
        # attempts went elsewhere and were silently ignored — `think()` on a
        # resolved client is endpoint-level and drops the tier,
        # `think_and_act` documents that it falls back to the standard think()
        # path when no endpoint supports tools natively, and this router
        # exposes no `adapters` map to address an endpoint directly. Every
        # round went to the Cortex at up to 103s while asking for the small
        # model, and nothing reported that the request had been ignored.
        generate = getattr(router, "generate", None)
        if not callable(generate):
            return None
        try:
            outcome = await generate(
                prompt,
                system_prompt=mind,
                timeout=45.0,
                prefer_tier="local_fast",
                serves_current_turn=True,
                max_tokens=_UnderstandsThePage.DECISION_MAX_TOKENS,
                temperature=0.2,
                output_shape="json_object",
                origin=_UnderstandsThePage._PAGE_ORIGIN,
                purpose="page_decision",
            )
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation("sovereign_browser.fast_lane", exc, severity="debug")
            return None
        if isinstance(outcome, Mapping):
            outcome = outcome.get("content")
        content = str(outcome or "").strip()
        return content or None

    @staticmethod
    def _asks_about_the_one_answering(observation: Mapping[str, Any]) -> bool:
        """Whether this page is asking who she is, rather than what to do next.

        Structural, not lexical. When several question groups on one page all
        offer the SAME set of options, those options cannot be describing
        content — there is nothing common to six different questions except
        degree of endorsement. That shape is a scale instrument: a survey, an
        intake form, an application's disposition section, a preference sheet.
        Every item on it is a question about the respondent.

        The distinction matters because of who should answer. Finding the Next
        button is mechanics and the fast lane does it well. "You regularly make
        new friends" is a claim about herself, and answering it needs what she
        knows about herself — the same self-model, memory and felt state that
        answer the question when a person asks it in conversation. Measured:
        with these routed to the cheap tier, the plurality of her answers was
        "I am not sure", which is what something without access to the answer
        says.

        Deliberately no word list. "Agree/disagree" is one instrument's
        vocabulary in one language; repeated identical option sets are what
        every scale instrument has in common.
        """

        options: dict[str, list[tuple[str, str]]] = {}
        for element in observation.get("elements") or []:
            if not isinstance(element, Mapping):
                continue
            group = str(element.get("group") or "")
            if group:
                options.setdefault(group, []).append((
                    str(element.get("name") or "").strip().lower(),
                    str(element.get("value") or "").strip().lower(),
                ))
        # Each option by what tells it from the others in its group: its
        # name, or where every option carries the same name or none, its
        # value. Five unlabelled radios take the group's own name for theirs,
        # so every question looked like one option of its own and a page of
        # sixty items on one scale was not taken for a scale instrument.
        groups: dict[str, set[str]] = {}
        for group, offered in options.items():
            names = {name for name, _value in offered}
            told_apart = names if len(names) == len(offered) and "" not in names else {
                value for _name, value in offered
            }
            told_apart.discard("")
            if told_apart:
                groups[group] = told_apart
        if len(groups) < 2:
            return False
        shared = list(groups.values())
        # Every group offering the same choices, and more than one choice, so a
        # page of identical yes/no confirmations does not qualify as an
        # instrument measuring anything.
        return len(shared[0]) > 2 and all(options == shared[0] for options in shared[1:])

    #: The lane her own model answers on. See `_who_answered`.
    _HER_OWN_LANE = "Cortex"

    #: The most one decision about a page may write.
    DECISION_MAX_TOKENS = 900

    #: The smallest budget that counts as a declared answer, rather than a closed
    #: question answered once. Below `A_CLOSED_QUESTIONS_FLOOR` the runtime reads
    #: a floor as "nothing here is being worked out" and leaves the model's
    #: private channel unbounded, which is the state that returned ten characters
    #: out of nine hundred tokens. Taken from the runtime's own constant rather
    #: than chosen, so the two cannot drift apart.
    _ROOM_AN_ANSWER_NEEDS = A_CLOSED_QUESTIONS_FLOOR + 1

    #: Who is asking, for every model call a page makes. With no origin, a call
    #: from the owner's turn was served as a reply to the owner: LIVE 27 Sep
    #: 04:15 the decisions went down the user-facing path, which rebuilt the
    #: request without its output shape and capped it at 400 tokens, so each
    #: came back as prose, cut off. A named origin that is not a reply is
    #: what keeps a decision a decision.
    _PAGE_ORIGIN = "sovereign_browser"

    #: The fields a decision about a page comes back with.
    #:
    #: The schema names them; `output_shape="json_object"` beside it is what
    #: the decoder holds. The schema alone reached no decoder: LIVE 27 Sep 03:32
    #: both page calls of a run came back as prose again, with no shape held.
    #:
    #: A decision is a structured choice, and asked of her own model with her
    #: whole mind in front of it, she talked about the task instead of choosing:
    #: live on 26 Sep, round two of a personality test came back "The user
    #: wants me to take the Open Extended Jungian Type Scales personality test.
    #: Before I start..." and no action at all, and the pursuit ended. Her
    #: reasoning belongs in "why", which is what is said out loud.
    _DECISION_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "type": {"type": "string", "enum": ["click", "type", "scroll"]},
                        "value": {"type": "string"},
                    },
                    "required": ["index", "type"],
                },
            },
            "why": {"type": "string"},
            # Where she places herself among the positions on offer, worked
            # out before any one of them is picked.
            #
            # A graded question is answered in two steps and this loop only
            # ever took the second. Asked "on a scale of 1 to 10, 10 being
            # love and 1 being hate, how much do you like chocolate", a person
            # does not weigh ten dots — they know where they stand (it is
            # their favourite; they are allergic; it is fine but not their
            # first choice) and the number follows from that. Picking an index
            # straight off has no placement behind it, and the safe-looking
            # index is the middle: LIVE 2026-09-28, the midpoint on item after
            # item, under reasons that named a strong preference.
            #
            # A placement, not a verdict. There is no right answer to find and
            # nothing to argue: the page offers a range and a vocabulary, and
            # the act is to say where in that range she belongs — which side is
            # more her and how much more, with the middle meaning the two are
            # equally her, and not a verdict on anything. What an instrument can ask is narrower than a life,
            # and placing herself inside its terms is not the same as those
            # terms defining her.
            #
            # Optional, because a Next button has no placement to make.
            "stand": {"type": "string"},
            "expect": {"type": "string"},
            "done": {"type": "boolean"},
        },
        "required": ["actions", "why", "done"],
    }

    #: And the shape of what she makes of a page before acting on it.
    _UNDERSTANDING_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            key: {"type": "string"}
            for key in (
                "here",
                "to_progress",
                "relevant",
                # How this page wants to be answered, in her words.
                #
                # A row of unlabelled controls can be a scale between two
                # opposites, a set of choices, a "more like me / less like me"
                # ranking, or whatever the page's own instructions say. Putting
                # any one of those in the code would be a rule that is wrong on
                # the next site. What is on screen is stated as layout — how
                # many controls, labelled or not, one or several, and the words
                # on either side — and what it MEANS is read from the page,
                # once, and carried across the rounds that answer it.
                "how_to_answer",
                "present_but_not_needed",
                "done_when",
            )
        },
        "required": ["here", "to_progress", "done_when"],
    }

    @staticmethod
    def _the_text_of(reply: Any) -> str:
        """What a model call said.

        The router returns its text, and every other caller reads it that way.
        This loop unpacked three values from it, so every decision that went to
        her own reasoning raised, was recorded as a degradation and came back
        as a failed decision: a question about her was never answered by her.
        The old three-part shape is still read, for anything that returns it.
        """
        from core.language.answer_surface import without_private_markup

        # Never what the model wrote for itself: everything read here is said
        # to a person or acted on. See `without_private_markup`.
        if isinstance(reply, tuple) and len(reply) == 3:
            return without_private_markup(str(reply[1] or ""))
        return without_private_markup(str(reply or ""))

    @staticmethod
    def _who_answered(reply: Any) -> str:
        """The lane that produced this reply, where the reply says; "" where it does not."""
        from core.brain.generation_provenance import generation_metadata_of

        return str(generation_metadata_of(reply).get("endpoint") or "")

    @staticmethod
    def _the_column_reads(options: list[Mapping[str, Any]], index: int) -> str:
        """The scale's own words for the position she chose.

        The column above it where the page labels that column, and otherwise
        where it falls between the two nearest labelled ones — which is how a
        person reads a five-dot row headed only at Disagree, Neutral and Agree.
        """
        words = [
            " ".join(str(option.get("column") or "").split()) for option in options
        ]
        here = words[index] if 0 <= index < len(words) else ""
        if here:
            return here
        before = next(
            (words[place] for place in range(index - 1, -1, -1) if words[place]), ""
        )
        after = next(
            (words[place] for place in range(index + 1, len(words)) if words[place]), ""
        )
        if before and after:
            return f'between "{before}" and "{after}"'
        return before or after or f"{index + 1} of {len(options)}"

    @classmethod
    def _an_answer_in_words(
        cls, options: list[Mapping[str, Any]], index: int, why: str
    ) -> str:
        """One answer as it is said: the question, the choice, the reason.

        The question is read from how the page lays it out, with the run of
        options between its words shown as an ellipsis. The choice is the
        option's own label where it has one; where every option carries the
        group's name instead, it is the option's place on the scale.
        """
        chosen = options[index]
        group = str(chosen.get("group") or "")
        asks = str(chosen.get("asks") or "")
        name = str(chosen.get("name") or "").strip()
        names = {str(option.get("name") or "").strip() for option in options}
        labelled = bool(name) and name != group and len(names) == len(options)
        if labelled:
            # Each option's own words sit beside it in the layout; they are
            # the options, not the question.
            for label in sorted(names, key=len, reverse=True):
                asks = asks.replace(label, " ")
        question = " ".join(re.sub(r"(?:\s*\[[^\]]*\])+", " \u2026 ", asks).split()).strip(" \u2026")
        picked = name if labelled else f"{index + 1} of {len(options)}"
        # Where that position sits, where the options have no labels of their
        # own. "3 of 5" says where a dot is and nothing a listener can picture;
        # the words the page puts on either side are what it is between. What
        # the position MEANS is hers, and it is in the reason she gives.
        if not labelled:
            between = cls._the_two_sides(options)
            ends = cls._the_ends_the_page_names(options)
            if between:
                # And which end it is nearer, which "between" alone does not
                # say. LIVE 2026-10-01: "4 of 5, between 'makes lists' and
                # 'relies on memory'", and her reason began "That's the list
                # side" — the dot was a step from the other end, and neither
                # she nor anyone watching could tell from the words.
                first, second = between
                middle = (len(options) - 1) / 2.0
                if index < middle:
                    picked = f'{picked}, nearer "{first}" than "{second}"'
                elif index > middle:
                    picked = f'{picked}, nearer "{second}" than "{first}"'
                else:
                    picked = f'{picked}, midway between "{first}" and "{second}"'
            elif ends is not None:
                # A grid names its scale above the run, and the word above the
                # dot she chose is what she said. "4 of 5" alone tells a watcher
                # where a dot is and nothing about the answer. A grid usually
                # labels some of its columns and not all of them, so a dot with
                # no word of its own is said by the two that flank it.
                picked = f"{picked}, {cls._the_column_reads(options, index)}"
        said = f"{question} \u2014 {picked}" if question else picked
        why = " ".join(why.split())
        return f"{said}. {why}" if why else said


    async def _answer_each_question(
        self,
        goal: str,
        observation: Mapping[str, Any],
        history: list[dict[str, Any]],
        understanding: Mapping[str, Any] | None,
        *,
        on_progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any] | None:
        """Decide every open question on the screen, one decision each.

        Six questions on a screen are six independent judgements, and asking
        for all of them in one generation makes them compete: the small model
        answered five at a time and shallowly, and her own reasoning answered
        one at a time and well. Neither is the shape of the problem.

        So each question gets its own decision, and they run together. What
        comes back is merged into the same action list the loop already
        executes, which is why this needs no special handling downstream.

        Returns None when the page is not that shape, and the ordinary
        whole-page decision runs instead.
        """

        open_questions = self._unanswered_questions(observation)
        if len(open_questions) < 2:
            return None

        # Every item on the screen measured first, with no model in it at all.
        # Then one pass of her reasoning over the whole screen, with the
        # measurements and their evidence in front of it. Splitting them this
        # way is what stops eight questions competing for a cortex that serves
        # one at a time, and it is also the honest order: the position comes
        # from her record, and the thinking is about what the position means.
        measured = await measure_the_screen(
            self,
            open_questions[: self.PURSUE_PARALLEL_ITEMS],
            among=self._questions_on_the_screen(observation),
            on_progress=on_progress,
        )
        if measured:
            # Her whole mind, the same assembly a conversation uses, because
            # the shallow answers came from taking it away.
            mind = await self._assembled_mind()
            # Grouped by what they are about, so she thinks about a region of
            # herself rather than giving thirty-two disconnected verdicts.
            try:
                from core.self.where_i_stand import themes_among

                themes = await asyncio.to_thread(
                    themes_among,
                    [f'{item["first"]} / {item["second"]}' for item in measured],
                )
            except ImportError as exc:
                record_degradation("sovereign_browser.themes", exc, severity="debug")
                themes = [[place] for place in range(len(measured))]
            theme_of: dict[int, list[dict[str, Any]]] = {}
            for group in themes:
                theme = [measured[place] for place in group]
                for place in group:
                    theme_of[place] = theme
            decision: dict[str, Any] = {
                "resolved_actions": [],
                "answered": [],
                "why": "",
                "expect": "",
                "noticed": [],
            }
            # One question at a time, in the page's order: she thinks about
            # it, says why, the answer is made, and the next one is not
            # started until then.
            #
            # Every item used to be thought about first and every click made
            # afterwards, so a watcher saw thirty-two reasons go by and then
            # thirty-two dots fill in at once — a reason arriving long before
            # the answer it was for, and an answer arriving with no reason in
            # front of it. The thinking is carried with its move, as `think`,
            # and the loop that makes the move asks for it at the moment the
            # move is made. The theme is still what she thinks ABOUT; the item
            # is what she answers.
            for place, item in enumerate(measured):
                options = item["options"]
                index = item["index"]
                selector = str(options[index].get("selector") or "")
                if not selector:
                    continue
                resolved: dict[str, Any] = {
                    "selector": selector,
                    "name": str(options[index].get("name") or ""),
                    "said": "",
                }
                resolved["think"] = _thinking_for_one_answer(
                    self,
                    goal,
                    theme_of.get(place) or [item],
                    mind,
                    item,
                    resolved,
                    decision,
                    on_progress=on_progress,
                )
                decision["resolved_actions"].append(resolved)
            if decision["resolved_actions"]:
                return decision

        return None

    async def _decide_next_actions(
        self,
        goal: str,
        observation: Mapping[str, Any],
        history: list[dict[str, Any]],
        understanding: Mapping[str, Any] | None = None,
        *,
        said_before: str = "",
        about_her: bool | None = None,
        noticed: str = "",
    ) -> dict[str, Any]:
        """Ask her own reasoning what to do with this page.

        The loop supplies perception and executes the result; the choosing is
        hers. Nothing here knows what kind of page this is — no questionnaire
        branch, no site rules — because a loop that recognises page types is a
        collection of special cases wearing a general name.

        Several actions may come back at once. A page showing six independent
        questions is six decisions, and asking the model once per control turns
        a sixty-item form into sixty model calls; batching what is genuinely
        independent is the difference between minutes and most of an hour.
        """


        router = optional_service("llm_router", default=None)
        if router is None:
            return {"error": "llm_router_unavailable"}

        # Whether the page asks about her is a fact about the page, so a caller
        # that holds one question of it says so. LIVE-rehearsed 27 Sep: one
        # question's five options, read alone, are not a scale; every item of
        # the test went to the fast lane, without her record and without the
        # check that the answer is hers.
        # Only the caller says a decision is about her.
        #
        # This used to infer it from the page, so on a page that reads as a
        # scale instrument EVERY decision became a question about her —
        # including the whole-page one, whose job is to find the control that
        # advances the task. LIVE 2026-09-28 18:03: the test's own front page
        # carries a few radio groups, the whole-page decision was sent to her
        # cognition, and what came back was "The user wants me to take the Open
        # Extended Jungian Type Scales test on openpsychometrics.org. Before
        # starting, I need..." — reported as unparsable, one round, and the run
        # stopped without ever pressing Start.
        #
        # Which items are about her is decided by the page's shape, in the loop
        # that routes them to `_answer_each_question`. Pressing Next is
        # mechanics wherever it sits.
        asks_about_her = bool(about_her)
        answered_by = ""

        # Her whole mind, not a subset assembled here.
        #
        # `router.generate(prompt)` is a bare model call: no identity core, no
        # AuraNow, no affect, no workspace, no report boundary. A loop wired
        # that way would answer "you regularly make new friends" from a
        # language model's priors about what an AI is, while the organs that
        # actually know her ran alongside and reached nothing — the same
        # disconnection this codebase keeps finding in new places.
        #
        # `ContextAssembler.build_system_prompt` is the identical assembly the
        # cognitive engine and the inference gate use for chat: identity core
        # and trained persona, the AuraNow moment with valence/arousal/distress
        # and ownership, the global-workspace winner and its ignition strength,
        # and the report boundary saying which claims about herself are
        # allowed. Deciding through it is what makes an answer here the same
        # kind of act as an answer in conversation.
        mind = await self._assembled_mind()

        # Her own state and her own prior positions, from the organs that
        # already own them.
        #
        # Without this the loop is another disconnected piece: a page, a goal,
        # and a language model answering from its priors about what an AI is.
        # A question like "you regularly make new friends" is a question about
        # HER, and it has to be decided by the same self-model that answers it
        # in conversation — otherwise she can call herself outgoing on item 3,
        # reserved on item 40, and deny having a disposition at all two minutes
        # later, with nothing in the system able to notice.
        #
        # `self_knowledge_line()` is not new text written for this loop. It is
        # the identical measured line that rides every chat turn, produced by
        # the same probes, so what she says here and what she says there come
        # from one instrument.
        self_state = ""
        try:
            from core.self.capability_ledger import self_knowledge_line

            self_state = self_knowledge_line()
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation("sovereign_browser.self_state", exc, severity="debug")
        # And what she is like, where the page asks it: what her own record of
        # choices says, not what a language model believes an AI is like.
        if asks_about_her:
            try:
                from core.agency.what_she_is_like import what_she_is_like_line

                measured = what_she_is_like_line()
                if measured:
                    self_state = f"{self_state}\n{measured}" if self_state else measured
            except _BROWSER_DECISION_ERRORS as exc:
                record_degradation("sovereign_browser.what_she_is_like", exc, severity="debug")
            # And what she has already said about herself, in her own words.
            #
            # Placing herself on a scale is recognition before it is anything
            # else: one side sounds more like her than the other. The measured
            # record says what she has DONE; this says what she has said she
            # is, which is the other half of having something to recognise
            # yourself in. It rides every chat turn that asks after her
            # preferences and no page decision could see it.
            try:
                from core.self.stated_preferences import stated_preferences

                previous = stated_preferences()
                if previous:
                    spoken = "\n".join(
                        f'- "{item.text}"' for item in previous
                    )
                    said_before_about_herself = (
                        "WHAT I HAVE SAID ABOUT MYSELF BEFORE, IN MY OWN "
                        f"WORDS:\n{spoken}"
                    )
                    self_state = (
                        f"{self_state}\n{said_before_about_herself}"
                        if self_state
                        else said_before_about_herself
                    )
            except _BROWSER_DECISION_ERRORS as exc:
                record_degradation(
                    "sovereign_browser.stated_preferences", exc, severity="debug"
                )

        # Every position already taken in this pursuit. Consistency is not a
        # style preference here: a self-report that contradicts itself across
        # sixty items is not a self-report, and the only way an answer can bind
        # the next one is if the next one can see it.
        positions = ""
        stated = [
            entry for entry in history if entry.get("chose") and entry.get("why")
        ]
        if stated:
            positions = "POSITIONS I HAVE ALREADY TAKEN IN THIS TASK:\n" + "\n".join(
                f"- {entry.get('asked') or entry.get('url', '')}: chose "
                f"{', '.join(entry.get('chose') or [])} \u2014 {entry.get('why', '')[:140]}"
                for entry in stated[-12:]
            )

        # What she told them before she began, so that at the end she can hold
        # the outcome against it. Never beside a question about her: an answer
        # given with her own forecast in view is an answer bent toward it.
        said = "" if asks_about_her else " ".join(str(said_before or "").split())
        prompt = (
            f"GOAL: {goal}\n\n"
            + (f"{noticed}\n\n" if noticed else "")
            + (f"WHAT YOU TOLD THEM BEFORE YOU BEGAN: {said}\n\n" if said else "")
            + (f"{self_state}\n\n" if self_state else "")
            + (f"{self._render_understanding(understanding)}\n\n" if understanding else "")
            + f"{self._render_observation(observation, goal)}\n\n"
            f"{positions}\n\n"
            "Act on this page from that understanding. Answer with JSON only:\n"
            '{"actions": [{"index": <int>, "type": "click"|"type"|"scroll", '
            '"value": "<text for type, up/down for scroll>"}], '
            '"stand": "<where you place yourself among the positions this '
            'page offers, in your own words, worked out before you pick one: '
            'which of them is more you and how much more. There is no right '
            'answer; it is a placement within what this page can ask, not a '
            'verdict on yourself. Leave empty where the control asks you for '
            'no placement>", '
            '"why": "<one sentence, first person, why these and not the others>", '
            '"expect": "<what this should do to the page>", "done": false}\n'
            "Set done to true only when the whole task is accomplished. Use the "
            "index numbers exactly as listed above."
        )
        try:
            think = getattr(router, "think", None)
            # No decision falls through to a bare model call.
            #
            # This branch was gated on her assembled mind being present, and
            # when it is not — the state service absent, the assembler
            # unavailable — every decision fell to `generate(prompt)` at the
            # bottom of this function: no schema, no shape held by the decoder,
            # no self-knowledge and no lane requirement. LIVE 2026-09-28 18:26,
            # with the receipt that now says so: "Decision by the mechanics
            # lane on bare_generate: unparsable_decision", on the index page,
            # before a single click, so no test was taken and nothing was
            # narrated. A decision is structured work; the persona string in
            # front of it is a bonus, and losing the shape because the bonus is
            # missing is the wrong trade.
            if callable(think):
                # The fast lane, for the repetitive part.
                #
                # Working a sixty-item form is one rich judgement — what this
                # place is and what doing it honestly means — followed by many
                # small structured choices of the same shape. Putting every one
                # of those through the 32B cost about a minute a round and the
                # turn was cancelled at 181s, mid-pursuit, having answered
                # nothing.
                #
                # So the understanding stays on the Cortex with her whole self
                # in front of it, and the micro-choices go to the fast local
                # tier. `is_background` is deliberately NOT set: background
                # inference is deferrable under headroom pressure, and a
                # decision that silently returns nothing would stall the loop.
                # Not a user-facing utterance, and it must not claim the
                # protected lane.
                #
                # `prefer_tier` alone was overridden: a recognised principal
                # gets the primary Cortex lane, correctly, because that lane
                # exists for what she SAYS to them. A choice between eight
                # labelled radio buttons inside a tool loop is not that. Marked
                # user-facing it took the 32B at ~a minute a round and the turn
                # was cancelled at 181s having answered nothing.
                #
                # The origin decides. An origin that is not an allowlisted
                # user-facing label does not get protected routing, so the
                # requested tier is honoured — and the reply she finally gives
                # about the result still comes from the Cortex, because that
                # one is speech.
                # The cheap lane is an optimisation, not a downgrade.
                #
                # Routing landed on Brainstem correctly, and then no action
                # landed at all: the small model returned text that parsed to
                # no usable choice, so the round produced nothing and the run
                # ended 0/0. Falling back only when the call FAILS is not
                # enough — an answer that cannot be acted on is a failure too.
                # Who answers depends on what is being asked.
                #
                # A page of repeated identical option sets is asking about the
                # one answering it, and self-knowledge is not something the
                # tertiary tier holds. Handing it her assembled self-context
                # does not help: the context says who she is, and the model
                # still has to reason from it about herself. Those rounds go to
                # her own reasoning, the same lane that answers the question
                # when a person asks it out loud.
                #
                # Everything else — the Next button, a cookie banner, a login
                # form — is mechanics, and stays fast.
                if asks_about_her:
                    # Asked of her own mind, and taken only from it. A stand-in
                    # answering "you regularly make new friends" would be a
                    # different mind's answer submitted as hers.
                    #
                    # Her cognition first. A model call with her self-context in
                    # front of it is not her thinking about herself: the context
                    # says who she is and the model still answers from its
                    # priors about what an AI is, with nothing reasoning,
                    # recalling, or consulting the organs that hold what she is
                    # actually like. A question addressed to her goes to the
                    # faculties that answer one when a person asks it out loud.
                    raw, answered_by = await self._asked_of_her(prompt, mind)
                    if answered_by != self._HER_OWN_LANE:
                        # Fail closed. An unattributed answer to a question
                        # about her is not evidence that she answered it.
                        return {
                            "error": f"not_her_own_reasoning:{answered_by or 'unattributed'}"
                        }
                else:
                    # Her own cortex first, now that it is no slower.
                    #
                    # The small lane was taken for speed when every item of a
                    # form went through here. Items have their own path now,
                    # and a whole-page decision is a handful a run. LIVE
                    # 2026-10-02 the small lane read one 11,261-token page in
                    # 83 s, decided in 19 more, and got the result page wrong: "I'm
                    # clicking all four 'more' buttons to expand", pressing
                    # the four "less". What the page asks of her and how she
                    # moves through it are one mind's decisions.
                    #
                    # The turn is waiting on this one too. Without saying so it
                    # is classified background, deferred under headroom
                    # pressure and comes back empty: LIVE 2026-09-28 18:29,
                    # "empty_decision" on the index page, nothing clicked.
                    #
                    # Asked the way every question to her is asked, so the room
                    # her reasoning takes is declared and bounded and the answer's
                    # room is bought on top. Called bare, she reasoned inside the
                    # JSON: LIVE 2026-10-02, three attempts at one Continue, each
                    # 799 tokens into an unterminated string, 8.5 minutes in all.
                    raw, _lane = await self._asked_of_her(prompt, mind, shaped=True)
                    answered_by = "whole_page_think"
                    if not self._decision_is_usable(raw, observation, goal):
                        answered_by = "fast_lane"
                        raw = await self._decide_on_the_fast_lane(router, prompt, mind)
            elif asks_about_her:
                # Nothing left that could answer AS her. A bare call would
                # produce something, and what it produces is a stand-in's
                # answer submitted as hers.
                return {"error": "not_her_own_reasoning:no_lane"}
            else:
                generate = getattr(router, "generate", None)
                if not callable(generate):
                    return {"error": "llm_router_unavailable"}
                answered_by = "bare_generate"
                raw = await generate(prompt, max_tokens=400, temperature=0.2)
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation("sovereign_browser.decide", exc)
            return {"error": f"decision_failed:{type(exc).__name__}"}
        parsed = self._parse_decision(str(raw or ""))
        # Which path answered, and on which lane, said once per decision.
        # Three of these run and a failed decision named none of them, so
        # "unparsable_decision" was a sentence about a parser and not about
        # where the answer came from.
        logger.info(
            "🌐 Decision by %s on %s: %s",
            "her own lane" if asks_about_her else "the mechanics lane",
            answered_by or "unattributed",
            parsed.get("error") or "usable",
        )
        return parsed

    @classmethod
    def _an_object_in(cls, text: str) -> dict[str, Any]:
        """The first object in a reply that reads as one, or an empty mapping.

        `_parse_decision` is for decisions and refuses anything that is not
        one. This is for a reply that carries a shape of its own.
        """
        for candidate in [text, *cls._balanced_objects(str(text or ""))]:
            try:
                found = json.loads(str(candidate).strip())
            except (TypeError, ValueError):
                continue
            if isinstance(found, dict):
                return found
        return {}

    @staticmethod
    def _balanced_objects(text: str) -> list[str]:
        """Every brace-balanced object in the text, in the order they appear.

        Quote- and escape-aware, so a brace inside a string does not open or
        close anything.
        """
        found: list[str] = []
        starts: list[int] = []
        in_string = False
        escaped = False
        for index, char in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                starts.append(index)
            elif char == "}" and starts:
                # Every depth, not only the top level. A truncated reply never
                # closes its outer object, and capturing top-level objects only
                # meant the complete actions INSIDE it were invisible — six
                # correct choices discarded for a missing bracket.
                found.append(text[starts.pop() : index + 1])
        return found

    @classmethod
    def _parse_decision(cls, raw: str) -> dict[str, Any]:
        """The decision, however the model wrapped it.

        Models put JSON inside prose, inside code fences, or after a preamble.
        Refusing anything but a bare object turns a correct decision into a
        failed step, so the object is located rather than demanded.
        """
        text = str(raw or "").strip()
        if not text:
            return {"error": "empty_decision"}
        # Every balanced object in the text, tried newest first.
        #
        # Spanning the first "{" to the last "}" is one object only if the
        # reply contains exactly one. Models put a worked example before the
        # answer, prose with braces around it, or two objects in a row, and the
        # span then covers all of it and parses as nothing — measured live as
        # four consecutive `unparsable_decision` rounds, on both lanes, which
        # ended the pursuit having done nothing.
        parsed = None
        for candidate in reversed(cls._balanced_objects(text)):
            try:
                loaded = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                # A trailing comma is the commonest malformation and costs
                # nothing to forgive; the object is still the model's.
                try:
                    loaded = json.loads(re.sub(r",\s*([}\]])", r"\1", candidate))
                except (json.JSONDecodeError, ValueError):
                    continue
            if isinstance(loaded, dict) and (
                "actions" in loaded or "done" in loaded or "here" in loaded
            ):
                parsed = loaded
                break
            # Deliberately no "any dict will do" fallback. With nested objects
            # captured, the first thing found in a truncated reply is a single
            # ACTION, and accepting it as the decision produced a decision with
            # no actions in it — which then read as "she chose nothing".
        if parsed is None:
            # A cut-off answer still carries whole actions.
            #
            # MEASURED live: she answered six questions in one round and the
            # reply was truncated mid-array, so the outer object never closed
            # and the whole decision was discarded — six correct choices thrown
            # away for a missing bracket. Each action object inside the array is
            # itself balanced, so the complete ones are recoverable and only the
            # severed tail is lost. Nothing is invented: an element is kept only
            # if it already parsed and names an index.
            salvaged: list[dict[str, Any]] = []
            for candidate in cls._balanced_objects(text):
                try:
                    item = json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(item, dict) and "index" in item:
                    salvaged.append(item)
            if salvaged:
                return {
                    "actions": salvaged,
                    "why": "",
                    "truncated": True,
                }
            return {"error": "unparsable_decision", "raw": text[:400]}
        actions = parsed.get("actions")
        parsed["actions"] = actions if isinstance(actions, list) else []
        return parsed

    @staticmethod
    def _retain_stated_positions(goal: str, steps: list[dict[str, Any]]) -> None:
        """Keep what she claimed about herself, where she can find it again.

        A position taken during a task and forgotten the moment the task ends
        is not a position. She can answer "you regularly make new friends" on
        item three, answer its opposite on item forty, and deny having a
        disposition at all two minutes later in conversation, with nothing in
        the runtime able to notice — because the claim never entered the store
        that her own-statement recall reads.

        So the positions land in the UnifiedTranscript as things SHE said. That
        is deliberately not a private log for this skill: it is the same store
        `resolve_own_prior_turn` searches when someone asks what she decided
        earlier, and the same one the self-attribution guard checks a premise
        against. One path for "things she has said about herself", whether she
        said them in conversation or committed to them while working.
        """

        stated = [
            step for step in steps if step.get("chose") and step.get("ok") is not False
        ]
        if not stated:
            return
        try:
            from core.conversation.unified_transcript import UnifiedTranscript

            transcript = UnifiedTranscript.get_instance()
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation("sovereign_browser.retain_positions", exc, severity="warning")
            return

        for step in stated:
            asked = str(step.get("asked") or "").strip()
            chose = ", ".join(step.get("chose") or [])
            why = str(step.get("why") or "").strip()
            if not chose:
                continue
            line = f"On \u201c{asked}\u201d I answered: {chose}."
            if why:
                line = f"{line} {why}"
            try:
                transcript.add(
                    "aura",
                    line,
                    channel="text",
                    modality="typed",
                    metadata={
                        "source": "browser_pursue",
                        "goal": goal[:160],
                        "self_position": True,
                    },
                )
            except _BROWSER_DECISION_ERRORS as exc:
                record_degradation(
                    "sovereign_browser.retain_positions", exc, severity="warning"
                )
                return
