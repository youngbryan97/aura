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
import re
from collections.abc import Mapping
from typing import Any

from core.conversation.word_markers import names_any
from core.runtime.errors import record_degradation

#: A decision round must never take the browser down with it. The loop can
#: always report a failed round and stop; it can never leave a live lease and a
#: half-driven page behind because the model call raised.
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


class _UnderstandsThePage:
    """Lifted whole from SovereignBrowserSkill; see sovereign_browser.py."""

    @staticmethod
    def _observation_signature(observation: Mapping[str, Any]) -> str:
        """What would have to change for progress to have been made."""
        elements = observation.get("elements") or []
        marks = "|".join(
            f"{element.get('role')}:{element.get('name')}:{element.get('checked')}"
            for element in elements[:60]
        )
        return f"{observation.get('url')}#{marks}"

    @classmethod
    def _controls_worth_offering(cls, elements: list[Any]) -> list[Any]:
        """The controls that can advance a goal, before the ones that decorate.

        Truncating the raw list would cut the answers and keep the navigation,
        because site furniture is emitted first in document order. Ranking by
        what a control DOES keeps the form and drops the chrome.
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
        ranked = sorted(
            enumerate(live),
            key=lambda pair: (
                cls._ACTIONABLE_ROLES.index(str(pair[1].get("role") or "").lower())
                if str(pair[1].get("role") or "").lower() in cls._ACTIONABLE_ROLES
                else len(cls._ACTIONABLE_ROLES),
                pair[0],
            ),
        )
        return [element for _index, element in ranked[: cls.PURSUE_CONTROL_BUDGET]]

    # A classmethod rather than a static one because it reads two budgets off
    # the class. It used to name the class to get at them, which is the same
    # dependency written in a way that breaks the moment the method moves.
    @classmethod
    def _render_observation(cls, observation: Mapping[str, Any]) -> str:
        """The page as the decision sees it: what it says, and what it offers."""
        elements = cls._controls_worth_offering(
            list(observation.get("elements") or [])
        )
        lines = [
            f"URL: {observation.get('url')}",
            f"Title: {observation.get('title')}",
            "",
            "PAGE TEXT:",
            str(observation.get("text") or "")[: cls.PURSUE_TEXT_BUDGET],
            "",
            "AVAILABLE CONTROLS:",
        ]
        # Answered questions are gone from this list rather than annotated in
        # it — see `_controls_worth_offering`. What remains is what is left to
        # do, so a screen half-finished reads as a shorter screen.
        for index, element in enumerate(elements):
            state = []
            if element.get("group"):
                # Options in one group answer ONE question. Rendering it is
                # what lets a whole screen be answered in a single round
                # instead of one control at a time.
                state.append(f"question {element['group']}")
            if element.get("checked") is True:
                state.append("already answered")
            if element.get("value"):
                state.append(f"value={element['value']}")
            suffix = f" ({', '.join(state)})" if state else ""
            lines.append(
                f"[{index}] {element.get('role')} \u2014 {element.get('name')}{suffix}"
            )
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

            from core.container import ServiceContainer

            world = ServiceContainer.get("world_model", default=None)
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
            from core.container import ServiceContainer

            world = ServiceContainer.get("world_model", default=None)
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
            from core.container import ServiceContainer

            state = ServiceContainer.get("aura_state", default=None)
            if state is None:
                return ""
            return ContextAssembler.build_system_prompt(state)
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

            from core.container import ServiceContainer

            world = ServiceContainer.get("world_model", default=None)
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
            from core.container import ServiceContainer

            calibration = ServiceContainer.get("calibration_engine", default=None)
            recorder = getattr(calibration, "record_prediction", None)
            if callable(recorder):
                recorder(0.75, 1.0 if moved else 0.0)
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation("sovereign_browser.calibration", exc, severity="debug")

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

        from core.container import ServiceContainer

        router = ServiceContainer.get("llm_router", default=None)
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
            f"{self._render_observation(observation)}\n\n"
            "Describe the situation, as JSON only:\n"
            '{"here": "<what this page is>", '
            '"to_progress": "<what I have to do on THIS page to move forward>", '
            '"relevant": "<which controls matter and what they do>", '
            '"present_but_not_needed": "<controls that exist here and are not what I need>", '
            '"done_when": "<how I will know the whole task is finished>"}'
        )
        try:
            think = getattr(router, "think", None)
            if callable(think) and mind:
                _ok, raw, _meta = await think(
                    prompt, system_prompt=mind, max_tokens=420, temperature=0.2
                )
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
            ("Here but not what I need", understanding.get("present_but_not_needed")),
            ("I am finished when", understanding.get("done_when")),
        ]
        lines = [f"- {label}: {value}" for label, value in rows if value]
        return "MY UNDERSTANDING OF THIS TASK:\n" + "\n".join(lines) if lines else ""

    @classmethod
    def _decision_is_usable(cls, raw: Any, observation: Mapping[str, Any]) -> bool:
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
        elements = cls._controls_worth_offering(list(observation.get("elements") or []))
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
                max_tokens=900,
                temperature=0.2,
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

        groups: dict[str, set[str]] = {}
        for element in observation.get("elements") or []:
            if not isinstance(element, Mapping):
                continue
            group = str(element.get("group") or "")
            name = str(element.get("name") or "").strip().lower()
            if group and name:
                groups.setdefault(group, set()).add(name)
        if len(groups) < 2:
            return False
        shared = list(groups.values())
        # Every group offering the same choices, and more than one choice, so a
        # page of identical yes/no confirmations does not qualify as an
        # instrument measuring anything.
        return len(shared[0]) > 2 and all(options == shared[0] for options in shared[1:])

    @staticmethod
    def _unanswered_questions(
        observation: Mapping[str, Any]
    ) -> list[tuple[str, list[Mapping[str, Any]]]]:
        """The question groups still open, each with its own options."""
        groups: dict[str, list[Mapping[str, Any]]] = {}
        for element in observation.get("elements") or []:
            if not isinstance(element, Mapping):
                continue
            group = str(element.get("group") or "")
            if group:
                groups.setdefault(group, []).append(element)
        return [
            (group, options)
            for group, options in groups.items()
            if not any(option.get("checked") is True for option in options)
        ]

    async def _answer_each_question(
        self,
        goal: str,
        observation: Mapping[str, Any],
        history: list[dict[str, Any]],
        understanding: Mapping[str, Any] | None,
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

        async def decide_one(options: list[Mapping[str, Any]]) -> dict[str, Any] | None:
            # The page, cut down to one question. Everything else about the
            # observation is unchanged, so what she sees is this item in its
            # real context — the same URL, the same page text.
            single = {**observation, "elements": list(options)}
            decision = await self._decide_next_actions(goal, single, history, understanding)
            if decision.get("error"):
                return None
            for item in decision.get("actions") or []:
                if not isinstance(item, dict):
                    continue
                try:
                    index = int(item.get("index"))
                except (TypeError, ValueError):
                    continue
                if not 0 <= index < len(options):
                    continue
                selector = str(options[index].get("selector") or "")
                if not selector:
                    continue
                # Resolved here, against the list this decision was shown.
                # Handing an index back to the caller would resolve it against
                # the whole page, which is how a loop ends up pressing whatever
                # moved into slot four.
                return {
                    "selector": selector,
                    "name": str(options[index].get("name") or ""),
                    "why": str(decision.get("why") or ""),
                    "expect": str(decision.get("expect") or ""),
                }
            return None

        chosen = await asyncio.gather(
            *(decide_one(options) for _group, options in open_questions[: self.PURSUE_PARALLEL_ITEMS]),
            return_exceptions=True,
        )
        answers = [item for item in chosen if isinstance(item, dict)]
        for outcome in chosen:
            if isinstance(outcome, BaseException):
                record_degradation(
                    "sovereign_browser.answer_item",
                    outcome,
                    severity="debug",
                    action="one question of a screen went unanswered",
                )
        if not answers:
            return None
        return {
            "resolved_actions": [
                {"selector": answer["selector"], "name": answer["name"]} for answer in answers
            ],
            "why": "; ".join(dict.fromkeys(a["why"] for a in answers if a["why"]))[:400],
            "expect": next((a["expect"] for a in answers if a["expect"]), ""),
        }

    async def _decide_next_actions(
        self,
        goal: str,
        observation: Mapping[str, Any],
        history: list[dict[str, Any]],
        understanding: Mapping[str, Any] | None = None,
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

        from core.container import ServiceContainer

        router = ServiceContainer.get("llm_router", default=None)
        if router is None:
            return {"error": "llm_router_unavailable"}

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

        prompt = (
            f"GOAL: {goal}\n\n"
            + (f"{self_state}\n\n" if self_state else "")
            + (f"{self._render_understanding(understanding)}\n\n" if understanding else "")
            + f"{self._render_observation(observation)}\n\n"
            f"{positions}\n\n"
            "Act on this page from that understanding. Answer with JSON only:\n"
            '{"actions": [{"index": <int>, "type": "click"|"type"|"scroll", '
            '"value": "<text for type, up/down for scroll>"}], '
            '"why": "<one sentence, first person, why these and not the others>", '
            '"expect": "<what this should do to the page>", "done": false}\n'
            "Set done to true only when the whole task is accomplished. Use the "
            "index numbers exactly as listed above."
        )
        try:
            think = getattr(router, "think", None)
            if callable(think) and mind:
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
                if self._asks_about_the_one_answering(observation):
                    _ok, raw, _meta = await think(
                        prompt, system_prompt=mind, max_tokens=900, temperature=0.2
                    )
                else:
                    raw = await self._decide_on_the_fast_lane(router, prompt, mind)
                    if not self._decision_is_usable(raw, observation):
                        _ok, raw, _meta = await think(
                            prompt, system_prompt=mind, max_tokens=900, temperature=0.2
                        )
            else:
                generate = getattr(router, "generate", None)
                if not callable(generate):
                    return {"error": "llm_router_unavailable"}
                raw = await generate(prompt, max_tokens=400, temperature=0.2)
        except _BROWSER_DECISION_ERRORS as exc:
            record_degradation("sovereign_browser.decide", exc)
            return {"error": f"decision_failed:{type(exc).__name__}"}
        return self._parse_decision(str(raw or ""))

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
