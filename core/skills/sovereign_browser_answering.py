"""What the page asks of her, and her answer to each question on it.

Lifted whole out of `sovereign_browser_understanding`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Mapping
from typing import Any


class _AnswersWhatThePageAsks:
    """Lifted whole out of _UnderstandsThePage; see sovereign_browser_understanding.py."""

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
        from .sovereign_browser_understanding import (
            _the_question_and_the_answer,
            record_degradation,
        )

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
        from .sovereign_browser_understanding import (
            _BROWSER_DECISION_ERRORS,
            optional_service,
            record_degradation,
            while_she_writes,
        )

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
                    # A token count, as every reader of it takes it: the gate
                    # does not raise max_tokens past a ceiling it is given.
                    hard_output_token_ceiling=int(most_tokens or self.DECISION_MAX_TOKENS),
                    # The typed lane for "say what was settled": the worker and the
                    # gate's clock both read it, and neither opens the channel.
                    **({} if worked_out_here else {"cognitive_mode": "fast"}),
                    # How long she reasons before choosing, bounded: a choice on a page is a step, and its thinking
                    # at 15 tokens a second was the most of each round's time (LIVE 7 November 2026, 450 tokens of
                    # thinking, 61 seconds, for one click).
                    **({"private_channel_budget": self.DECISION_THINKS_AT_MOST} if worked_out_here else {}),
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
        from .sovereign_browser_understanding import (
            _thinking_for_one_answer,
            measure_the_screen,
            record_degradation,
        )


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

