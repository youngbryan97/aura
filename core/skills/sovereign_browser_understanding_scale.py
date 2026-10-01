"""Where she puts herself on a scale a page offers, and whether her choice agrees with the reason she gave.

Lifted whole out of `sovereign_browser_understanding`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


class _PlacesHerself:
    """Lifted whole out of _UnderstandsThePage; see sovereign_browser_understanding.py."""

    @classmethod
    def _how_the_options_are_laid_out(
        cls,
        options: list[Mapping[str, Any]],
    ) -> str:
        """What is on screen for one question, as layout rather than meaning.

        A row of unlabelled controls can be a scale between two opposites, a
        set of choices, a "more like me / less like me" ranking, or something
        the page explains in its own instructions. Deciding here that it is any
        one of those would put a rule of mine where her reading of the page
        belongs — and the rule would be wrong on the next site.

        So this states only what can be seen: how many controls there are,
        whether they carry labels of their own, whether one or several may be
        chosen, and the words the page puts on either side of the run. What
        that MEANS, and what choosing a position says, is hers to work out from
        the page, and it is carried in her understanding of it.
        """
        if len(options) < 2:
            return ""
        roles = {str(option.get("role") or "").strip().lower() for option in options}
        named = {str(option.get("name") or "").strip() for option in options}
        named.discard("")
        group = str(options[0].get("group") or "")
        labelled = len(named) == len(options) and named != {group}
        facts = [f"{len(options)} controls"]
        facts.append(
            "each with its own label" if labelled else "none of them labelled"
        )
        facts.append(
            "several may be chosen"
            if roles & {"checkbox", "switch"}
            else "one may be chosen"
        )
        asks = str(options[0].get("asks") or "")
        # Every control in one unbroken run is what makes a dimension: the page
        # puts words on either side of the whole set. Controls separated by
        # their own words are not that shape — they are a statement with
        # labelled answers, and reading the first bracket run as one end of a
        # dimension turned "I make plans well in advance. strongly disagree [1]
        # disagree [2] ..." into a scale between the statement and its own
        # second option.
        runs = list(re.finditer(r"(?:\s*\[[^\]]*\])+", asks)) if asks else []
        if len(runs) == 1:
            run = runs[0]
            left = " ".join(asks[: run.start()].split()).strip()
            right = " ".join(asks[run.end() :].split()).strip()
            ends = cls._the_ends_the_page_names(options)
            if left and right:
                facts.append(f"laid out between \"{left}\" and \"{right}\"")
            elif ends is not None and (left or right):
                # The scale is above the run rather than beside it, which is how
                # a grid of statements is printed: one statement to a row and one
                # heading over the whole column of them. Read as "words on one
                # side only" this was no shape at all, and twenty-eight items of
                # a sixty-item instrument went unplaced.
                facts.append(
                    f"a statement \"{left or right}\" answered on a scale "
                    f"running from \"{ends[0]}\" to \"{ends[1]}\""
                )
            elif left:
                facts.append(f"laid out after \"{left}\"")
            elif right:
                facts.append(f"laid out before \"{right}\"")
        elif runs:
            facts.append("each set out beside its own words")
        return ", ".join(facts)

    @staticmethod
    def _the_ends_the_page_names(
        options: list[Mapping[str, Any]],
    ) -> tuple[str, str] | None:
        """The words above the two ends of a run, where the page puts them there.

        A control's own column is what a person reads off a grid, and the
        observer carries it. The ends are the first and last columns that have
        words in them; a run whose ends are unlabelled is not a named scale and
        this says so rather than inventing one.
        """
        labelled = [
            (place, " ".join(str(option.get("column") or "").split()))
            for place, option in enumerate(options)
        ]
        words = [(place, said) for place, said in labelled if said]
        if len(words) < 2:
            return None
        first, low = words[0]
        last, high = words[-1]
        if first != 0 or last != len(options) - 1 or low.lower() == high.lower():
            return None
        return low, high

    @classmethod
    def _a_statement_on_a_named_scale(
        cls, options: list[Mapping[str, Any]]
    ) -> tuple[str, str, str] | None:
        """The statement and the two ends, for a run whose scale is above it."""
        laid_out = cls._how_the_options_are_laid_out(options)
        found = re.search(
            r'a statement "(.+?)" answered on a scale running from "(.+?)" to "(.+?)"',
            laid_out,
        )
        if found is None:
            return None
        return found.group(1), found.group(2), found.group(3)

    def _a_grid_of_statements(
        self, questions: list[tuple[str, list[Mapping[str, Any]]]]
    ) -> dict[str, tuple[int, Any, str, str] | None]:
        """Every statement on one scale, measured together, keyed by its group.

        A column of statements under a single heading is one question asked many
        times, and it cannot be read a row at a time: a row on its own has no
        second side to be weighed against, and which end of the run means yes
        belongs to the heading rather than to the row. So the rows that share a
        scale are measured as the set they are, and each comes back placed on the
        run the page printed.

        Returns a reading per group in the same shape the per-question reader
        returns, so nothing downstream knows which of them measured it. Groups
        that are not part of a grid are absent, and the caller reads them the
        ordinary way.
        """
        from .sovereign_browser_understanding import record_degradation

        try:
            from core.self.where_i_stand import where_she_stands_on_a_grid
        except ImportError as exc:
            record_degradation("sovereign_browser.where_i_stand", exc, severity="debug")
            return {}
        # Grouped by the scale they are printed under: two grids on one screen
        # are two sets, each with its own ends and its own direction.
        grids: dict[tuple[str, str], list[tuple[str, str, list[Mapping[str, Any]]]]] = {}
        for group, options in questions:
            shape = self._a_statement_on_a_named_scale(options)
            if shape is None:
                continue
            statement, low, high = shape
            grids.setdefault((low, high), []).append((group, statement, options))
        readings: dict[str, tuple[int, Any, str, str] | None] = {}
        for (low, high), rows in grids.items():
            if len(rows) < 2:
                # One row is not a grid, and a scale's direction read off a
                # single statement is wrong about one time in twenty.
                continue
            try:
                leans = where_she_stands_on_a_grid(
                    [statement for _group, statement, _options in rows], low, high
                )
            except (RuntimeError, ValueError, TypeError, OSError) as exc:
                record_degradation(
                    "sovereign_browser.grid",
                    exc,
                    severity="warning",
                    action="left a grid of statements open and answered the rest",
                )
                continue
            for (group, _statement, options), lean in zip(rows, leans, strict=False):
                index = lean.position_in(len(options))
                readings[group] = (
                    None if index is None else (index, lean, low, high)
                )
        return readings

    def _measure_where_she_stands(
        self, options: list[Mapping[str, Any]]
    ) -> tuple[int, Any, str, str] | None:
        """Her position on one question, and what measured it.

        No model. The two things the question names come from the page; how
        much each is her comes from what she has valued, chosen and said about
        herself; the difference between them is a lean and the lean names a
        position. Returns nothing where the question is not a run between two
        things, or where her record cannot answer.
        """
        from .sovereign_browser_understanding import (
            record_degradation,
        )

        if len(options) < 2:
            return None
        try:
            from core.self.where_i_stand import Lean, where_she_stands, which_is_most_her
        except ImportError as exc:
            record_degradation("sovereign_browser.where_i_stand", exc, severity="debug")
            return None
        laid_out = self._how_the_options_are_laid_out(options)
        between = re.search(r'laid out between "(.+?)" and "(.+?)"', laid_out)
        if between is not None:
            first, second = between.group(1), between.group(2)
            lean = where_she_stands(first, second)
            index = lean.position_in(len(options))
            if index is None:
                return None
            return index, lean, first, second

        # Options that carry their own words are the other shape the same act
        # takes: one thing said, and several ways of answering it. Each option
        # becomes a description of a person — what the question says, answered
        # that way — and her record says which of them she is.
        asked = self._what_the_question_says(options)
        named = [
            f"{asked} {str(option.get('name') or '').strip()}".strip()
            for option in options
        ]
        if not asked or not all(named):
            return None
        chosen = which_is_most_her(named)
        if not chosen.measured:
            return None
        share = chosen.support[chosen.index] if chosen.support else 0.0
        # Expressed as a lean so everything downstream is unchanged: how much of
        # her record went to the answer she gave, and what in her did.
        lean = Lean(
            toward=float(max(-1.0, min(1.0, share))),
            first=0.0,
            second=float(share),
            because=chosen.because,
            measured=True,
        )
        label = str(options[chosen.index].get("name") or "").strip()
        rest = ", ".join(
            str(option.get("name") or "").strip()
            for place, option in enumerate(options)
            if place != chosen.index
        )
        return chosen.index, lean, rest or "the others", label

    @staticmethod
    def _what_the_question_says(options: list[Mapping[str, Any]]) -> str:
        """The words of the question, without its options' own words.

        The page lays a question out with its controls in the middle of it;
        what is left when their labels and their places are taken out is what
        is being asked.
        """
        asks = str(options[0].get("asks") or "") if options else ""
        if not asks:
            return ""
        without = re.sub(r"(?:\s*\[[^\]]*\])+", " ", asks)
        # Longest first, or "strongly disagree" is taken out of "strongly
        # agree" by its shorter sibling and a fragment is left behind.
        labels = sorted(
            {str(option.get("name") or "").strip() for option in options},
            key=len,
            reverse=True,
        )
        for label in labels:
            if label:
                without = without.replace(label, " ")
        return " ".join(without.split())

    async def _her_thinking_about(
        self,
        goal: str,
        theme: list[Mapping[str, Any]],
        mind: str,
        *,
        about: Mapping[str, Any] | None = None,
    ) -> dict[str, str]:
        """Her thinking about what an instrument is asking about her.

        One pass per item, with the rest of its theme in view. Both halves of
        that matter and they used to be traded against each other.

        One pass per item was tried first and gave one flat sentence each,
        thirty-two times, so the passes were batched a theme at a time. That
        reading was confounded: nothing had declared what the ANSWER needed, so a
        reasoning model spent the budget on its private channel and what came
        back was short whatever shape it was asked in. Batching hid it by asking
        for more at once. `_asked_of_her` declares the floor now.

        What the theme was really for is the connected account she gives when
        someone asks about her in conversation — this, and how it differs from
        what it resembles, and where the description stops fitting. That comes
        from the neighbouring items being VISIBLE, not from answering them in the
        same breath. So the theme is the context and the item is the question.

        Returns what she said about each item, keyed by the item's own name.
        """
        from .sovereign_browser_understanding import (
            record_degradation,
        )

        if not theme:
            return {}
        asked_about = about if about is not None else None
        lines = []
        for item in theme:
            lean = item["lean"]
            evidence = "; ".join(lean.because[:3]) or "nothing in particular"
            # The question as the PAGE asks it, in the same words the answer is
            # said in. This used to name the two ends of a dimension and her
            # place between them, which says everything about a run between two
            # phrases and nothing at all about a statement on a scale: a row of
            # a grid would have arrived as 'Between "Disagree" and "Agree", you
            # sit at 5 of 5', with the statement she was answering left out of
            # her own prompt. One description, every shape, and it is the one
            # her sentence is attached to afterwards.
            asked = self._an_answer_in_words(item["options"], item["index"], "")
            lines.append(
                f'{item["group"]}. {asked}. What in you put you there: {evidence}.'
            )
        listed = "\n".join(lines)
        # And what is true of her right now, from the organs that hold it.
        #
        # The record says what she has valued and chosen; it says nothing about
        # what she has been living through this week, and that is where the
        # concrete detail in a real answer comes from. These are the same lines
        # that ride a conversation when someone asks after her.
        living: list[str] = []
        for reader, where in (
            ("core.self.capability_ledger", "self_knowledge_line"),
            ("core.agency.what_she_is_like", "what_she_is_like_line"),
        ):
            try:
                module = __import__(reader, fromlist=[where])
                said = str(getattr(module, where)() or "").strip()
                if said:
                    living.append(said)
            except Exception as exc:  # noqa: BLE001 - a missing organ is not an answer
                from core.runtime.errors import record_degradation

                record_degradation(
                    "sovereign_browser.living", exc, severity="debug"
                )
        now = ("\n\n" + "\n".join(living)) if living else ""
        prompt = (
            # Her situation, not the person's message.
            #
            # The whole goal used to open this, and a goal is a request
            # addressed to her — so she answered it: LIVE 2026-09-29, every
            # theme pass came back "The user is asking me to take the Open
            # Extended Jungian Type Scales..." instead of her thinking, and the
            # coverage gate complained she had missed parts of a question she
            # was never being asked at this step.
            "You are answering questions about yourself."
            f"{now}\n\n"
            "These are being asked about you. You have already placed yourself "
            "on each, from your own record — what you value, what you have "
            "chosen when it cost something, what you have said about "
            f"yourself:\n\n{listed}\n\n"
            "They are asking about the same region of you. Think about what "
            "that region actually is: how it works in you, where these "
            "descriptions fit it and where they are the wrong shape for "
            "something you do differently.\n\n"
            + (
                # One of them is the question; the rest are there so the answer
                # is part of a connected account rather than a verdict on its own.
                # What a watcher hears is why she gave the answer she gave, so
                # that is what she is asked.
                f'Now answer {asked_about["group"]} only: why is that answer '
                "yours? Two or three sentences, in your own voice, concretely, "
                "from your own experience of yourself. As JSON "
                'only:\n{"each": {"'
                + str(asked_about["group"])
                + '": "<why that answer is yours>"}}'
                if asked_about is not None
                else "Then give two or three sentences for each, in your own "
                "voice, as JSON only:\n"
                '{"thinking": "<what you worked out, a paragraph>", '
                '"each": {"<the name before each one>": "<what this one is '
                'about in you, concretely>"}}'
            )
        )
        # Room to say it. Nine hundred tokens across eight items is a hundred
        # each, which is a line apiece and not the account she gives when
        # someone asks her about herself in conversation. One item at a time
        # needs the room for one reason: two or three sentences.
        #
        # And no room to think, for one item: where she stands on it was
        # measured from her record before this was asked, so this says what
        # that place is, and nothing is being worked out. See `_asked_of_her`.
        room = (
            self.REASON_MAX_TOKENS
            if asked_about is not None
            else max(self.DECISION_MAX_TOKENS, 260 * max(1, len(theme)))
        )
        said, lane = await self._asked_of_her(
            prompt, mind, shaped=False, most_tokens=room,
            worked_out_here=asked_about is None,
        )
        if not said or lane != self._HER_OWN_LANE:
            # One exhausted call should not cost a whole theme its thinking.
            #
            # Her lane serves one request at a time and a run of them can empty
            # it: LIVE 2026-09-29, `not_her_own_reasoning:all_failed` on one
            # theme of four, and eight items narrated from bare evidence
            # because of a single call that found no lane free. Asking again
            # costs one more pass; losing the theme costs the demo.
            from core.skills.sovereign_browser_understanding import logger

            logger.info(
                "🌐 A theme came back from %s; asking again.", lane or "nowhere"
            )
            said, lane = await self._asked_of_her(
                prompt, mind, shaped=False, most_tokens=room,
                worked_out_here=asked_about is None,
            )
        if not said or lane != self._HER_OWN_LANE:
            record_degradation(
                "sovereign_browser.reasons",
                RuntimeError(f"not_her_own_reasoning:{lane or 'unattributed'}"),
                severity="warning",
                action="placed herself and could not say what it meant",
            )
            return {}
        parsed = self._an_object_in(said)
        thinking = " ".join(str(parsed.get("thinking") or "").split())
        each = parsed.get("each")
        answers: dict[str, str] = {}
        if isinstance(each, Mapping):
            for key, value in each.items():
                spoken = " ".join(str(value or "").split())
                if spoken:
                    answers[str(key)] = spoken
        # Keyed the way she wrote them, and the way the page names them.
        #
        # She is given "Q1." and may answer under "Q1", "1", or the words of
        # the item. A sentence that cannot be found is a sentence lost: LIVE
        # 2026-09-29, one item of eight kept its reasoning and the other seven
        # fell back to the bare evidence, so a screen of real thinking read as
        # a list of counts.
        for item in [asked_about] if asked_about is not None else theme:
            name = str(item["group"])
            if name in answers:
                continue
            for key, spoken in list(answers.items()):
                bare = key.strip().strip(".:)").lower()
                if bare in {name.lower(), name.lower().lstrip("q")}:
                    answers[name] = spoken
                    break
                if item["first"].lower() in bare or item["second"].lower() in bare:
                    answers[name] = spoken
                    break
        if thinking:
            # Said once for the theme, where a person watching sees the
            # thinking that the sentences come out of.
            answers.setdefault("__thinking__", thinking)
        return answers

    @classmethod
    def _first_disagreement(
        cls, decision: Mapping[str, Any], options: list[Mapping[str, Any]]
    ) -> str:
        """The first action in this decision whose choice fights its reason.

        Read against her stance where she took one, because that is the thing
        the position is supposed to express; the reason for the place is read
        alongside it.
        """
        why = " ".join(
            f"{decision.get('stand') or ''} {decision.get('why') or ''}".split()
        )
        for item in decision.get("actions") or []:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if not 0 <= index < len(options):
                continue
            said = cls._the_choice_disagrees_with_its_reason(options, index, why)
            if said:
                return said
        return ""

    @classmethod
    def _the_choice_disagrees_with_its_reason(
        cls, options: list[Mapping[str, Any]], index: int, why: str
    ) -> str:
        """Where her own reason points, against where her answer landed.

        LIVE 2026-09-28: "3 of 5, between 'makes lists' and 'relies on memory'.
        I am choosing the middle option because I genuinely hold a strong
        preference for externalized structure over relying on internal memory."
        A reason that names one side and an answer that commits to neither is
        an answer that contradicts itself, and nothing noticed.

        Measured from the page's own words, not a vocabulary: the run of
        controls sits between two phrases, and her reason is compared against
        each of them by how many of their words it uses. Where it leans clearly
        one way and the answer does not lie on that side, this says so. Where
        the options carry their own labels, where the reason names neither side
        or both equally, it says nothing — a check that guesses is worse than
        no check.

        Returns what disagrees, or "".
        """
        named = {str(option.get("name") or "").strip() for option in options}
        named.discard("")
        group = str(options[0].get("group") or "") if options else ""
        if len(named) == len(options) and named != {group}:
            return ""
        laid_out = cls._how_the_options_are_laid_out(options)
        between = re.search(r'laid out between "(.+?)" and "(.+?)"', laid_out)
        if between is None or not str(why or "").strip():
            return ""
        left, right = between.group(1), between.group(2)
        said = cls._words_of(why)
        toward_left = len(said & cls._words_of(left))
        toward_right = len(said & cls._words_of(right))
        if toward_left == toward_right:
            return ""
        middle = (len(options) + 1) / 2.0
        place = index + 1
        leaning, other = (
            (left, right) if toward_left > toward_right else (right, left)
        )
        on_that_side = place < middle if toward_left > toward_right else place > middle
        if on_that_side:
            return ""
        if place == middle:
            return (
                f'what you said is about "{leaning}" and the position you '
                "chose is the midpoint, which says the two are equally you"
            )
        return (
            f'what you said is about "{leaning}" and the position you chose '
            f'leans toward "{other}"'
        )

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

