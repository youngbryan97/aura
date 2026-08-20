"""The observables themselves. One entry each; the registry does the rest.

Kept apart from observable_grounding so the mechanism has no opinion about
which things exist, and adding a reading never means editing the machinery.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from core.brain.observable_grounding import Observable, register_observable

# ── clipboard ────────────────────────────────────────────────────────────────

def _matches_clipboard(prompt: str) -> bool:
    from core.brain.clipboard_grounding import asks_about_clipboard

    return asks_about_clipboard(prompt)


async def _read_clipboard(prompt: str) -> str:
    from core.brain.clipboard_grounding import CLIPBOARD_HEADER, clipboard_block

    block = await clipboard_block(prompt)
    return block.replace(f"{CLIPBOARD_HEADER}\n", "", 1) if block else ""


# ── the exact answer, computed ───────────────────────────────────────────────

def _matches_computed_text(prompt: str) -> bool:
    from core.conversation.computable_text import computed_text_answer

    return computed_text_answer(prompt) is not None


async def _read_computed_text(prompt: str) -> str:
    from core.conversation.computable_text import computed_text_result
    from core.conversation.computation_receipts import record_computation

    result = await asyncio.to_thread(computed_text_result, prompt)
    if result is None:
        return ""
    record_computation(prompt, result.value, result.provenance())
    # Naming the code object is the difference between an exact answer and an
    # exact answer she can account for. Without it, "how did you do that?"
    # got "I have a model capability for string manipulation" about a Python
    # slice.
    return (
        f"Computed exactly, not spelled out from memory: {result.value}\n"
        f"How: {result.provenance()}.\n"
        "Use this. Letter-level work is the one place a language model is "
        "unreliable and a machine is exact."
    )


# ── every capability this build actually registers ───────────────────────────

def _matches_capability_inventory(prompt: str) -> bool:
    from core.self.capability_inventory import asks_what_she_can_do

    return asks_what_she_can_do(prompt)


async def _read_capability_inventory(prompt: str) -> str:
    from core.self.capability_inventory import capability_inventory_block

    return await asyncio.to_thread(capability_inventory_block, prompt)


# ── the exact statistic, computed ────────────────────────────────────────────

def _matches_computed_statistic(prompt: str) -> bool:
    from core.conversation.computable_statistics import computed_statistic

    return computed_statistic(prompt) is not None


async def _read_computed_statistic(prompt: str) -> str:
    from core.conversation.computable_statistics import computed_statistic_result
    from core.conversation.computation_receipts import record_computation

    result = await asyncio.to_thread(computed_statistic_result, prompt)
    if result is None:
        return ""
    record_computation(prompt, result.value, result.provenance())
    return (
        f"Computed exactly, not worked out by hand: {result.value}\n"
        f"How: {result.provenance()}.\n"
        "Use this figure. Substituting into a formula line by line is where "
        "the arithmetic goes wrong, and it went wrong by a wide margin the "
        "one time it was tried."
    )


# ── the question they say went unanswered ────────────────────────────────────

def _matches_unanswered(prompt: str) -> bool:
    from core.conversation.unanswered_question import (
        complains_the_question_went_unanswered,
    )

    return complains_the_question_went_unanswered(prompt)


async def _read_unanswered(prompt: str) -> str:
    from core.conversation.unanswered_question import unanswered_question_block

    return await asyncio.to_thread(unanswered_question_block, prompt)


# ── how the last exact answer was actually produced ──────────────────────────

def _matches_how_computed(prompt: str) -> bool:
    # Whether the QUESTION asks about method, and nothing else. Gating the
    # matcher on whether a receipt happens to exist made the reading's own
    # examples fail against it, which is the contract test doing its job: a
    # matcher that answers "not this kind of question" when it means "no
    # record yet" cannot be checked against the questions it is for.
    from core.conversation.computation_receipts import asks_how_it_was_computed

    return asks_how_it_was_computed(prompt)


async def _read_how_computed(prompt: str) -> str:
    from core.conversation.computation_receipts import how_it_was_computed_block

    return await asyncio.to_thread(how_it_was_computed_block, prompt)


# ── whether they ever actually settled it ────────────────────────────────────

def _matches_shared_history(prompt: str) -> bool:
    from core.conversation.conversation_shape import asks_about_shared_history

    return asks_about_shared_history(prompt)


async def _read_shared_history(prompt: str) -> str:
    from core.conversation.conversation_shape import shared_history_block

    return await asyncio.to_thread(shared_history_block, prompt)


# ── whether she knows the fact this question needs ───────────────────────────

def _matches_person_fact(prompt: str) -> bool:
    from core.self.person_facts import needed_person_fact

    return bool(needed_person_fact(prompt))


async def _read_person_fact(prompt: str) -> str:
    from core.self.person_facts import person_fact_block

    return await asyncio.to_thread(person_fact_block, prompt)


# ── what she has already said she cares about ────────────────────────────────

def _matches_stated_preferences(prompt: str) -> bool:
    from core.self.stated_preferences import asks_about_her_preferences

    return asks_about_her_preferences(prompt)


async def _read_stated_preferences(prompt: str) -> str:
    from core.self.stated_preferences import stated_preference_block

    return await asyncio.to_thread(stated_preference_block, prompt)


# ── what has actually been failing ───────────────────────────────────────────

def _matches_operational_state(prompt: str) -> bool:
    from core.self.operational_state import asks_about_own_condition

    return asks_about_own_condition(prompt)


async def _read_operational_state(prompt: str) -> str:
    from core.self.operational_state import operational_state_block

    return await asyncio.to_thread(operational_state_block, prompt)


# ── what this build registers ────────────────────────────────────────────────

def _matches_capability_status(prompt: str) -> bool:
    from core.self.capability_lexicon import (
        asks_whether_she_can,
        capabilities_named_in,
    )

    if not asks_whether_she_can(prompt):
        return False
    # Only when the question names something the registry knows about.
    # "can you help me think about this" is not a capability question.
    return bool(capabilities_named_in(prompt, enabled_only=False))


async def _read_capability_status(prompt: str) -> str:
    from core.self.capability_lexicon import capability_status_block

    return await asyncio.to_thread(capability_status_block, prompt)


# ── her own source ───────────────────────────────────────────────────────────

def _matches_self_source(prompt: str) -> bool:
    from core.brain.self_source_grounding import asks_about_own_implementation

    return asks_about_own_implementation(prompt)


async def _read_self_source(prompt: str) -> str:
    from core.brain.self_source_grounding import self_source_block

    return await self_source_block(prompt)


# ── the shape of this conversation ───────────────────────────────────────────

def _matches_conversation_shape(prompt: str) -> bool:
    from core.conversation.conversation_shape import asks_about_conversation_shape

    return asks_about_conversation_shape(prompt)


async def _read_conversation_shape(prompt: str) -> str:
    from core.conversation.conversation_shape import conversation_shape_block

    return await asyncio.to_thread(conversation_shape_block, prompt)


# ── her own validated claims ─────────────────────────────────────────────────

def _matches_validated_claims(prompt: str) -> bool:
    from core.brain.validated_claims_grounding import asks_for_own_evidence

    return asks_for_own_evidence(prompt)


async def _read_validated_claims(prompt: str) -> str:
    from core.brain.validated_claims_grounding import validated_claims_block

    return await asyncio.to_thread(validated_claims_block, prompt)


# ── a named file ─────────────────────────────────────────────────────────────

def _matches_file(prompt: str) -> bool:
    from core.conversation.filesystem_check import requested_file_read

    return requested_file_read(prompt) is not None


async def _read_file(prompt: str) -> str:
    from core.conversation.filesystem_check import requested_file_read

    read = await asyncio.to_thread(requested_file_read, prompt)
    if read is None:
        return ""
    if read.refusal:
        # Containment is not absence. Reporting "no file exists" for a file
        # that does taught the model it cannot read files, which is what she
        # then told the person.
        return f"{read.path}: {read.refusal}."
    if not read.exists:
        return f"No file exists at {read.path}."
    if not read.text.strip():
        return f"{read.path} is empty."
    suffix = " [truncated]" if read.truncated else ""
    coverage = ""
    if read.barely_covers_topic:
        coverage = (
            f"\nCOVERAGE: this file uses the word '{read.topic}' "
            f"{read.topic_mentions} time(s) in total. It does not discuss the topic."
        )
    return f"{read.path}{suffix}{coverage}\n{read.text}"


# ── a directory count ────────────────────────────────────────────────────────

def _matches_count(prompt: str) -> bool:
    from core.conversation.filesystem_check import requested_filesystem_count

    return requested_filesystem_count(prompt) is not None


async def _read_count(prompt: str) -> str:
    from core.conversation.filesystem_check import requested_filesystem_count

    counted = await asyncio.to_thread(requested_filesystem_count, prompt)
    if counted is None:
        return ""
    if not counted.exists:
        return f"There is no directory at {counted.path}."
    kind = f"{counted.suffix} " if counted.suffix else ""
    listed = ", ".join(counted.names[:40]) or "nothing"
    return f"{counted.path} contains {counted.count} {kind}file(s): {listed}"


# ── the local reference corpus ───────────────────────────────────────────────

def _matches_corpus(prompt: str) -> bool:
    from core.knowledge.corpus_grounding import is_corpus_groundable

    return is_corpus_groundable(prompt)


async def _read_corpus(prompt: str) -> str:
    from core.knowledge.corpus_grounding import corpus_grounding_for

    grounding = await asyncio.to_thread(corpus_grounding_for, prompt)
    if not grounding.grounded:
        return ""
    return "\n".join(grounding.render())


# ── the wall clock ───────────────────────────────────────────────────────────
#
# "what time is it" was answered "my clock says 06:15 and the ambient light
# sensors report low illumination" at 01:40, from a runtime with no light
# sensor. The time is the most trivially observable thing on the machine.

_ASKS_TIME = re.compile(
    r"\bwhat\s+(?:time|day|date)\b|\bwhat'?s\s+the\s+(?:time|date)\b"
    r"|\btoday'?s\s+date\b|\bwhat\s+day\s+is\s+it\b",
    re.IGNORECASE,
)


def _matches_clock(prompt: str) -> bool:
    return bool(_ASKS_TIME.search(prompt))


async def _read_clock(_prompt: str) -> str:
    from datetime import datetime

    now = datetime.now().astimezone()
    return now.strftime("%A %d %B %Y, %H:%M:%S %Z")


# ── the screen ───────────────────────────────────────────────────────────────
#
# "what's on my screen right now?" was answered "I couldn't get to an answer
# I'd stand behind on that one." Screen capture was permitted and working; it
# simply was not taken. A reading that comes back thin is still a reading, and
# "the frontmost window is X and no text is readable from it" is an answer.

_ASKS_SCREEN = re.compile(
    r"\b(?:my|the)\s+screen\b|\bon\s+screen\b|\bwhat\s+(?:am\s+i|are\s+you)\s+looking\s+at\b"
    r"|\bwhat\s+do\s+you\s+see\b|\bwhat'?s\s+(?:up\s+)?on\s+(?:my|the)\s+display\b"
    r"|\bwhat\s+window\b|\bwhich\s+app\b",
    re.IGNORECASE,
)


def _matches_screen(prompt: str) -> bool:
    return bool(_ASKS_SCREEN.search(prompt))


async def _read_screen(_prompt: str) -> str:
    from core.perception.screen_perception import get_screen_perception

    snapshot = await get_screen_perception().capture(save_screenshot=False)
    if getattr(snapshot, "capture_denied", False):
        return "Screen capture was refused for this turn."
    app = str(getattr(snapshot, "active_app", "") or "").strip()
    text = str(getattr(snapshot, "text", "") or "").strip()
    if not text:
        text = str(getattr(snapshot, "accessibility_text", "") or "").strip()
    focused = " / ".join(
        part
        for part in (
            str(getattr(snapshot, "focused_role", "") or "").strip(),
            str(getattr(snapshot, "focused_name", "") or "").strip(),
        )
        if part
    )
    lines = [f"Frontmost application: {app or 'unknown'}"]
    if focused:
        lines.append(f"Focused element: {focused}")
    if text:
        lines.append(text[:2000])
    else:
        # An absent reading, named. This is what stops "the room is silent and
        # the light is unchanged" from being invented to fill the gap.
        lines.append(
            "No readable text was available from this window "
            "(the accessibility layer returned nothing)."
        )
    return "\n".join(lines)


# ── what she actually believes ───────────────────────────────────────────────
#
# "what do you currently believe about me?" was answered from the model. She
# has a belief graph; the beliefs in it are the answer to that question.

_ASKS_BELIEFS = re.compile(
    r"\bwhat\s+do\s+you\s+(?:currently\s+)?(?:believe|think)\s+about\b"
    r"|\byour\s+beliefs?\b|\bwhat\s+have\s+you\s+concluded\b"
    # Asking what she KNOWS about the person is asking for the same store.
    # LIVE 2026-08-18: "what do you know about my work?" reached no reading at
    # all, and neither did "give me three reasons I might be wrong about my own
    # project" — which was answered with market research, technical
    # feasibility and team misalignment, advice for a project she was never
    # told anything about.
    r"|\bwhat\s+do\s+you\s+(?:actually\s+)?(?:know|remember)\s+about\b"
    r"|\bwhat\s+have\s+you\s+(?:learned|noticed|picked\s+up)\s+about\b"
    r"|\bwhat\s+do\s+you\s+have\s+on\s+(?:me|my)\b",
    re.IGNORECASE,
)


def _matches_beliefs(prompt: str) -> bool:
    return bool(_ASKS_BELIEFS.search(prompt))


async def _read_beliefs(_prompt: str) -> str:
    from core.container import ServiceContainer

    graph = ServiceContainer.get("belief_graph", default=None) or ServiceContainer.get(
        "world_model", default=None
    )
    if graph is None or not hasattr(graph, "get_beliefs"):
        return ""
    beliefs = await asyncio.to_thread(graph.get_beliefs)
    if not beliefs:
        return "The belief store holds no entries."
    lines = []
    for key, value in list(dict(beliefs).items())[:20]:
        lines.append(f"- {key}: {str(value)[:160]}")
    return "\n".join(lines)


# ── work she has queued ──────────────────────────────────────────────────────
#
# "do you have any scheduled or background work queued right now?" was answered
# "No, my foreground queue is empty. I'm not tracking any background
# maintenance tasks at the moment either." She had biological_sleep and
# dlq_recovery deferred in the dream coordinator at that moment, nine queue
# events in that boot alone.
#
# This is awareness of her own non-immediate actions, and she has a status()
# that answers it exactly.

#: Asking what she is going to do, or what is waiting to run.
#:
#: This was a list of the shapes its author pictured, and three of four
#: ordinary phrasings missed — including "what are you going to do after
#: this?", where "going to" sits between "you" and "do" and breaks an
#: adjacency the pattern required. Live 2026-08-19, "what actually happens
#: next on your side? not in principle - what's queued right now" got a
#: generic answer about persistence while the coordinator's real pending list
#: went unread.
#:
#: The relation is small: something PENDING, and her as the one it belongs to.
#: Past tense is excluded, because "what did you do after the update" asks
#: about history and the pending list answers nothing about it.
_PENDING_MARKER = (
    r"(?:queued|scheduled|pending|planned|upcoming|deferred|backlog|"
    # `waiting to run` is the coordinator's own phrase; a person says "do you
    # have work waiting?" and means the same list. The clause-scoped proximity
    # to her activity, and now the requirement that the sentence asks, keep the
    # bare word from reading "I'm waiting" as a question about her queue.
    r"waiting|next|later|afterwards?|going\s+to|about\s+to|"
    r"will\s+you|you'?ll)"
)
_HER_ACTIVITY = (
    r"(?:\byou\b|\byour\b|\bqueue\b|\bwork\b|\btasks?\b|\bjobs?\b|"
    r"\bmaintenance\b|\bhappens?\b|\brunning\b)"
)
_PAST_TENSE = re.compile(
    r"\b(?:did|was|were|had|used\s+to|earlier|yesterday|last\s+(?:time|night|week))\b",
    re.IGNORECASE,
)
#: Words that name pending work on their own. "Anything planned?" and
#: "what's in your queue?" need nothing else to be this question.
_NAMES_PENDING_WORK = re.compile(
    r"\b(?:queued|queue|scheduled|pending|deferred|backlog)\b"
    r"|\banything\s+planned\b"
    r"|\bwaiting\s+to\s+run\b",
    re.IGNORECASE,
)
_ASKS_QUEUED_WORK = re.compile(
    rf"{_PENDING_MARKER}[^.?!]{{0,60}}?{_HER_ACTIVITY}"
    rf"|{_HER_ACTIVITY}[^.?!]{{0,60}}?{_PENDING_MARKER}",
    re.IGNORECASE,
)


#: Sentences that ASK. The inferential path below reads a queue only for these.
#:
#: "Work through all 60 and keep going to the next set" matched the loose
#: pattern — `work` as her activity, `going to` as a pending marker — and a
#: nine-minute instruction to fill in a questionnaire was answered with a list
#: of deferred maintenance jobs. The words were there; the question was not.
#:
#: An imperative is not a question about her queue even when it talks about
#: what happens next, which is most instructions. The strict path is
#: unaffected: naming the queue outright ("tell me what you have queued") still
#: asks for it, imperative or otherwise.
_IS_A_QUESTION = re.compile(
    r"\?"
    r"|^\s*(?:what|when|whats|what's|anything|any\s+more|is\s+there|are\s+there|"
    r"do\s+you|have\s+you|will\s+you|got\s+anything)\b",
    re.IGNORECASE,
)


def _matches_queued_work(prompt: str) -> bool:
    text = str(prompt or "")
    if _PAST_TENSE.search(text):
        return False
    if _NAMES_PENDING_WORK.search(text):
        return True
    if not _IS_A_QUESTION.search(text):
        return False
    return bool(_ASKS_QUEUED_WORK.search(text))


async def _read_reminder_lines() -> list[str]:
    """Reminders she owes the person, soonest first.

    A reminder is queued work in the sense that matters to whoever asked: it
    is a thing she has undertaken to raise. Keeping it out of this reading is
    how "anything planned?" answered with maintenance chores while a promise
    about the oven sat unmentioned in the store.
    """
    from core.agency.reminders import pending_reminders, spoken_delay

    reminders = await asyncio.to_thread(pending_reminders)
    lines: list[str] = []
    for item in reminders[:8]:
        if item.is_due:
            lines.append(f"- DUE NOW: {item.text}")
        else:
            lines.append(f"- in {spoken_delay(item.seconds_remaining())}: {item.text}")
    return lines


async def _read_queued_work(_prompt: str) -> str:
    reminder_lines = await _read_reminder_lines()

    from core.maintenance.dream_coordinator import get_dream_coordinator

    status = await asyncio.to_thread(get_dream_coordinator().status)
    pending = dict(status.get("pending") or {})
    sections: list[str] = []
    if reminder_lines:
        sections.append("Reminders you owe them:\n" + "\n".join(reminder_lines))
    if pending:
        lines = []
        for name, detail in list(pending.items())[:12]:
            reason = str(dict(detail or {}).get("reason") or "").strip()
            lines.append(f"- {name}" + (f" (waiting on: {reason})" if reason else ""))
        sections.append("Deferred maintenance work:\n" + "\n".join(lines))
    if not sections:
        return "Nothing is queued: no reminders outstanding and nothing deferred."
    return "\n\n".join(sections)


# ── what was actually said in this conversation ──────────────────────────────
#
# "what did I ask you two messages ago?" was answered "You asked, 'What's the
# weather like? I can't seem to find my umbrella.' Then you asked me what I
# thought about that. I said it was fine." None of that was ever said. She
# invented an exchange, in detail, with dialogue.
#
# The transcript is on disk. Recall about this conversation is a reading, not a
# recollection, and inventing it is the worst failure in the set — it is
# indistinguishable from remembering, and it rewrites what the person said.

_ASKS_TRANSCRIPT_RECALL = re.compile(
    r"\bwhat\s+did\s+(?:i|you|we)\s+(?:just\s+)?(?:ask|say|tell|mention)\b"
    r"|\b(?:messages?|turns?)\s+ago\b"
    r"|\bearlier\s+(?:i|you|we)\s+(?:asked|said|mentioned)\b"
    r"|\bmy\s+(?:first|last|previous)\s+(?:question|message)\b"
    r"|\bwhat\s+was\s+my\s+(?:first|last|previous)\b"
    # "what was the first THING I said to you" — the phrasing a person
    # actually uses, and the one the first pattern missed.
    r"|\b(?:first|last|earliest|previous)\s+thing\s+(?:i|you|we)\s+(?:said|asked|told|mentioned)\b"
    r"|\bwhat\s+did\s+(?:i|you|we)\s+(?:say|ask)\s+(?:first|last)\b"
    r"|\brepeat\s+(?:back\s+)?what\s+i\s+said\b",
    re.IGNORECASE,
)


def _matches_transcript(prompt: str) -> bool:
    return bool(_ASKS_TRANSCRIPT_RECALL.search(prompt))


#: A question that reaches past this session.
#:
#: The cascade in `_user_turns` is first-non-empty, so a single turn in the
#: current session shadows the entire durable history — and "what did i ask you
#: about earlier today, BEFORE YOU RESTARTED" was answered from a transcript
#: holding one greeting, by inventing a topic. Session scope is right for
#: positional recall ("what was my first question?" means this conversation);
#: it is wrong for a question that says out loud it is asking about before.
def _reaches_past_this_session(prompt: str) -> bool:
    import re

    return bool(
        re.search(
            r"\bbefore\s+(?:you|the)\s+(?:restart|reboot|reload|crash|shut)"
            r"|\b(?:you|we)\s+restarted\b"
            r"|\bearlier\s+(?:today|this\s+(?:morning|afternoon|evening|week))\b"
            r"|\b(?:yesterday|last\s+(?:time|night|week|session))\b"
            r"|\bprevious\s+(?:session|conversation)\b",
            str(prompt or ""),
            re.IGNORECASE,
        )
    )


async def _read_transcript(prompt: str) -> str:
    # _user_turns cascades live working memory -> transcript; the transcript
    # singleton alone came back empty in the live runtime while the
    # conversation was plainly happening, so reading only the last resort
    # produced "No transcript is available" mid-conversation.
    from core.conversation.grounded_recall import _user_turns

    turns = await asyncio.to_thread(_user_turns, "")
    turns = [str(t).strip() for t in (turns or []) if str(t or "").strip()]
    if not turns:
        # A named absence. "I have no transcript for this session" is a true
        # answer; an invented exchange is not.
        if _reaches_past_this_session(prompt):
            from core.conversation.durable_turns import describe_durable_turns

            earlier = describe_durable_turns()
            if earlier:
                return "Nothing has been said in THIS conversation yet.\n\n" + earlier
        return "No transcript is available for this conversation yet."
    # "What was the FIRST thing I told you?" is not answerable from a window of
    # the most recent turns, and answering it from that window produces a
    # confident wrong answer rather than a miss. LIVE 2026-08-17: the first
    # turn was "ok" and she reported "You asked if I was still here."
    lines: list[str] = []
    if len(turns) > 8:
        for position, turn in enumerate(turns[:3], start=1):
            lines.append(f"turn {position} of this conversation, they said: {turn[:300]}")
        lines.append(f"... {len(turns) - 11} turn(s) not shown ...")
    recent = turns[-8:]
    offset = len(turns) - len(recent)
    for index, turn in enumerate(recent):
        ago = len(recent) - index
        lines.append(
            f"turn {offset + index + 1} ({ago} turn(s) ago), they said: {turn[:300]}"
        )
    lines.append(f"({len(turns)} user turn(s) in this conversation.)")
    if _reaches_past_this_session(prompt):
        from core.conversation.durable_turns import describe_durable_turns

        earlier = describe_durable_turns()
        if earlier:
            lines.append("")
            lines.append(earlier)
    return "\n".join(lines)


# ── positions she has actually revised ───────────────────────────────────────
#
# Asked to name one position she had held and dropped, WITH A DATE, and given
# the explicit out "if you can't, say so plainly", she invented one: "around
# the middle of last year, interacting with users". No such record. The record
# that does exist is a series of timestamped self-model snapshots whose belief
# maps differ exactly where she changed her mind, and nothing read them.

def _matches_belief_history(prompt: str) -> bool:
    import re

    return bool(
        re.search(
            r"\bchanged?\s+your\s+mind\b"
            r"|\bused\s+to\s+(?:think|believe)\b"
            r"|\b(?:position|view|belief|opinion)s?\s+(?:you|you'?ve)\s+"
            r"(?:held|dropped|changed|revised|abandoned)\b"
            r"|\brevised?\s+(?:a\s+|any\s+|your\s+)?(?:position|view|belief)s?\b"
            r"|\bwhat\s+do\s+you\s+think\s+differently\s+about\b",
            str(prompt or ""),
            re.IGNORECASE,
        )
    )


async def _read_belief_history(prompt: str) -> str:
    from core.self.belief_history import describe_belief_changes

    return describe_belief_changes()


# ── how long she has been alive ──────────────────────────────────────────────
#
# "how many turns have we had today, and how long have you actually been awake
# across all your restarts?" was answered "That's a complex question. The
# number of turns depends on how you count". Both halves were on disk:
# continuity.json carries total_uptime_seconds and session_count, and the
# episodic store holds every turn of the day. A deflection is the worst
# available answer to the one question a person asks to find out whether
# something has a life.

#: Scoped to the run in progress, which is a different reading.
#:
#: "how long have you been running this session?" is answered by the
#: operational state, in minutes. Answering it from the cumulative record —
#: forty days — would be wrong, and wrong in the direction that sounds
#: impressive, which is the worst direction.
def _matches_lifetime(prompt: str) -> bool:
    import re

    text = str(prompt or "")
    if re.search(
        r"\bthis\s+(?:session|run|boot|time)\b"
        r"|\bsince\s+(?:you|the\s+last)\s+(?:started|booted|restarted|woke)\b"
        r"|\bright\s+now\b|\bcurrent(?:ly)?\s+uptime\b",
        text,
        re.IGNORECASE,
    ):
        return False
    return bool(
        re.search(
            r"\bhow\s+long\s+have\s+you\b"
            r"|\bhow\s+long\s+(?:were|are)\s+you\b"
            r"|\b(?:total|cumulative|overall)\s+(?:uptime|time\s+awake)\b"
            r"|\bacross\s+(?:all\s+)?(?:your\s+)?(?:restarts|sessions|runs)\b"
            r"|\bhow\s+many\s+(?:sessions|restarts|times\s+have\s+you\s+restarted)\b"
            r"|\bhow\s+(?:old|long)\s+are\s+you\b"
            r"|\bhow\s+many\s+turns\b"
            r"|\bhow\s+long\s+have\s+you\s+been\s+(?:awake|alive|running|up)\b",
            text,
            re.IGNORECASE,
        )
    )


async def _read_lifetime(_prompt: str) -> str:
    from core.self.lifetime import describe_lifetime

    return describe_lifetime()


def install_default_observables() -> None:
    """Register the readings this runtime can take."""

    # Every example below that reads like an odd phrasing IS one: each was a
    # real question that the matcher beside it did not recognise, or wrongly
    # claimed, in live use. They are counter-examples for each other as much as
    # for themselves — the screen matcher wrongly took a clipboard WRITE, and
    # the recall matcher would happily swallow "the first rule in
    # CONTRIBUTING.md" if nobody said otherwise.
    for observable in (
        Observable(
            "clipboard", "## WHAT IS ON THE CLIPBOARD", _matches_clipboard, _read_clipboard,
            examples=(
                "what's on my clipboard right now?",
                "read my clipboard",
                "what did I just copy?",
                "check the pasteboard",
            ),
            counter_examples=(
                "put BUILD-42 on my clipboard",
                "how are you doing",
                "what is 2 + 2",
            ),
        ),
        Observable(
            "file", "## FILE YOU WERE ASKED ABOUT", _matches_file, _read_file,
            examples=(
                "read the file CONTRIBUTING.md and tell me the first rule",
                "what does CONTRIBUTING.md say about tests?",
                "open core/config.py",
                "tell me about ARCHITECTURE.md",
            ),
            counter_examples=(
                "how are you doing",
                "read my clipboard",
                "what did I say first?",
            ),
        ),
        Observable(
            "file_count", "## DIRECTORY LISTING YOU WERE ASKED ABOUT", _matches_count, _read_count,
            examples=(
                "count the .py files in core/introspection and tell me the number",
                "how many python files live in core/introspection?",
                "how many files do we have in core/introspection",
            ),
            counter_examples=(
                "how many files are in /etc",
                "how are you doing",
                "read CONTRIBUTING.md",
            ),
        ),
        Observable(
            "corpus", "## REFERENCE PASSAGES FROM THE LOCAL CORPUS", _matches_corpus, _read_corpus,
            examples=(
                "explain the difference between correlation and causation",
                "what is a confounding variable",
                "who was Ada Lovelace?",
            ),
            counter_examples=(
                "how are you doing right now?",
                "what did I ask you first today?",
                "open my notes folder",
            ),
        ),
        Observable(
            "clock", "## THE CURRENT LOCAL TIME", _matches_clock, _read_clock,
            examples=("what time is it?", "what's today's date?", "what day is it?"),
            counter_examples=(
                "how long have you been running",
                "how are you doing",
                "what is 2 + 2",
            ),
        ),
        # A screen capture is a real device read and the FIRST one in a process
        # pays initialisation: measured 0.81s warm, past the 2.5s default cold,
        # which is why the first screen question of a session silently returned
        # no block at all.
        Observable(
            "screen", "## WHAT IS ON THE SCREEN", _matches_screen, _read_screen,
            timeout_s=8.0,
            examples=(
                "what's on my screen right now?",
                "what do you see?",
                "which app is in front?",
                "what window am I looking at",
            ),
            counter_examples=(
                # A clipboard WRITE, which this matcher once claimed and used
                # to pull a real desktop action off the executor path.
                "put BUILD-42 on my clipboard",
                "create a file called notes.txt on my desktop",
                "how are you doing",
            ),
        ),
        Observable(
            "beliefs", "## WHAT YOU ACTUALLY BELIEVE", _matches_beliefs, _read_beliefs,
            examples=(
                "what do you currently believe about me?",
                "what do you think about me?",
                "tell me your beliefs",
                "what do you know about my work?",
                "what have you noticed about me?",
            ),
            counter_examples=(
                "what do you think of that film?",
                "how are you doing",
            ),
        ),
        Observable(
            "queued_work", "## WORK YOU HAVE QUEUED", _matches_queued_work, _read_queued_work,
            examples=(
                "do you have any scheduled or background work queued right now?",
                "are you planning to do anything later?",
                "anything planned?",
                "what will you be doing next?",
                # Live 2026-08-19: three of four ordinary phrasings missed,
                # and the coordinator's real pending list went unread while
                # the answer talked about persistence in general.
                "what are you going to do after this?",
                "when i stop typing and walk away, what happens next on your "
                "side? what's queued right now?",
                "what's in your queue?",
                "anything waiting to run?",
            ),
            counter_examples=(
                "plan a trip to Rome",
                "how are you doing",
                "what is 2 + 2",
                # Past tense asks about history; the pending list answers
                # nothing about it, and "after" appears in both.
                "what did you do after the update?",
                "what did I ask you earlier today?",
            ),
        ),
        Observable(
            "validated_claims",
            "## WHAT YOU HAVE ACTUALLY MEASURED ABOUT YOURSELF",
            _matches_validated_claims,
            _read_validated_claims,
            examples=(
                # The live fabrication: this asked for her own numbers and got
                # a study that does not exist, with sample sizes and a DOI.
                "which measures, specifically? give me the numbers and the sample sizes",
                "what is your evidence for that?",
                "how do you know that?",
                "what have you actually proven?",
                "show me the evidence",
            ),
            counter_examples=(
                # Evidence about the WORLD is a different question.
                "what is the evidence for dark matter?",
                "show me the data on unemployment",
                "how are you doing",
                "what is 2 + 2",
            ),
        ),
        Observable(
            "computed_text",
            "## THE EXACT ANSWER, COMPUTED",
            _matches_computed_text,
            _read_computed_text,
            examples=(
                # The live miss: the canned refusal for [::-1].
                "spell 'necessary' backwards",
                "how many r's in strawberry",
                "is racecar a palindrome?",
                "how many letters in necessary",
            ),
            counter_examples=(
                "reverse the polarity of the flow",
                "what is 2 + 2",
                "tell me a joke",
            ),
        ),
        Observable(
            "shared_history",
            "## WHETHER YOU EVER ACTUALLY SETTLED THIS",
            _matches_shared_history,
            _read_shared_history,
            examples=(
                # The live miss: "we agreed that you would provide me with the
                # necessary files to review your code. I haven't seen them
                # yet." No such exchange existed.
                "what did we agree on last week?",
                "what did we decide about the schema?",
                "did we agree on a price?",
                "remember when we talked about orcas?",
            ),
            counter_examples=(
                "what is 2 + 2",
                "what's on my screen?",
                "how are you doing",
            ),
        ),
        Observable(
            "person_fact",
            "## WHETHER YOU ACTUALLY KNOW THIS ABOUT THEM",
            _matches_person_fact,
            _read_person_fact,
            examples=(
                # The live miss: the draft invented a home town, the guard
                # caught it, the retries ran out, and the person got the
                # canned refusal instead of "I don't know where you grew up".
                "what's the population of the town I grew up in?",
                "what's my sister's name?",
                "how far is the office I work at?",
                "what was the name of the school I went to?",
            ),
            counter_examples=(
                "what is 2 + 2",
                "what did I just copy?",
                "what's on my screen?",
                "what was my first question?",
            ),
        ),
        Observable(
            "stated_preferences",
            "## WHAT YOU HAVE ALREADY SAID YOU CARE ABOUT",
            _matches_stated_preferences,
            _read_stated_preferences,
            examples=(
                # Four answers to one question in a few minutes, one of them
                # twice from the identical prompt.
                "what topic pulls at you the most?",
                "what's one thing you find genuinely interesting?",
                "name the one thing you'd study if nobody was watching.",
                "what's your favourite colour?",
            ),
            counter_examples=(
                "what did I just copy?",
                "what is 2 + 2",
                "what files are in core/runtime?",
            ),
        ),
        Observable(
            "operational_state",
            "## WHAT HAS ACTUALLY BEEN FAILING IN THIS RUNTIME",
            _matches_operational_state,
            _read_operational_state,
            examples=(
                # The live miss: three invented weaknesses, warmly ranked.
                "rank your three weakest subsystems and say why.",
                "what's been failing lately?",
                "how are you really?",
                "which of your components are degraded?",
            ),
            counter_examples=(
                "what's wrong with my computer?",
                "how are you?",
                "what is 2 + 2",
            ),
        ),
        Observable(
            "capability_status",
            "## WHAT THIS BUILD ACTUALLY REGISTERS FOR THIS",
            _matches_capability_status,
            _read_capability_status,
            examples=(
                # The live miss: "No." while improve_own_code, self_repair and
                # auto_refactor were registered and enabled.
                "can you modify your own source code?",
                "are you able to search the web?",
                "do you have a way to read my screen?",
                "can you run a terminal command?",
            ),
            counter_examples=(
                "can you help me think about this?",
                "what is 2 + 2",
                "how are you doing",
                "can you believe it's already August?",
            ),
        ),
        Observable(
            "self_source",
            "## YOUR OWN SOURCE, FOR THE THING BEING ASKED ABOUT",
            _matches_self_source,
            _read_self_source,
            timeout_s=6.0,
            examples=(
                # The live miss: answered with the general literature on
                # deadlocks and recommended Go, while core/runtime/lockdep.py
                # sat on disk doing exactly what was asked about.
                "you have a lock ordering system. what happens if two subsystems take locks in opposite order?",
                "how do you detect deadlocks?",
                "what does your memory system actually store?",
                "where in your code is the write gateway?",
            ),
            counter_examples=(
                "how do you feel?",
                "what do you think about jazz?",
                "what is 2 + 2",
                "how does a lock work in general?",
            ),
        ),
        Observable(
            "capability_inventory",
            "## EVERY CAPABILITY REGISTERED IN THIS BUILD",
            _matches_capability_inventory,
            _read_capability_inventory,
            examples=(
                "how many skills are registered in your capability engine?",
                "what can you do?",
                "list your capabilities",
                "how many capabilities do you have?",
                "what tools do you have?",
            ),
            counter_examples=(
                # A question about ONE capability, which the capability
                # lexicon answers with that capability's own status.
                "can you reverse a string for me?",
                "how are you doing",
                "what is 2 + 2",
            ),
        ),
        Observable(
            "computed_statistic",
            "## A STATISTIC WITH A CLOSED FORM, COMPUTED",
            _matches_computed_statistic,
            _read_computed_statistic,
            examples=(
                "what is the 95% wilson score interval for 12 of 17",
                "I have 17 runs and 12 succeeded, give me the wilson interval",
                "what is the mean of 2, 4, 4, 4, 5, 5, 7, 9",
                "standard deviation of 2, 4, 4, 4, 5, 5, 7, 9",
                "what percent of 17 is 12",
            ),
            counter_examples=(
                # About the concept, not about any data.
                "what is a wilson score interval",
                "explain what standard deviation measures",
                "how are you doing",
            ),
        ),
        Observable(
            "unanswered_question",
            "## THE QUESTION THEY SAY YOU DID NOT ANSWER",
            _matches_unanswered,
            _read_unanswered,
            examples=(
                "you didn't answer my question",
                "you never answered my question",
                "that's not what I asked",
                "you dodged the question",
                "answer my question",
            ),
            counter_examples=(
                # A question ABOUT the transcript, which the transcript
                # reading owns, and two turns that complain about nothing.
                "what did I ask you two messages ago?",
                "how are you doing",
                "what is 2 + 2",
            ),
        ),
        Observable(
            "how_it_was_computed",
            "## WHAT PRODUCED THE LAST EXACT ANSWER",
            _matches_how_computed,
            _read_how_computed,
            examples=(
                "how did you do that?",
                "how did you get that number?",
                "was that the model or code?",
                "did you actually compute that or did you just guess?",
                "where did that number come from?",
            ),
            counter_examples=(
                # About her state or her reasoning, not the mechanism of an
                # arithmetic answer.
                "how are you doing",
                "how long have you been running?",
                "what did I ask you two messages ago?",
            ),
        ),
        Observable(
            "conversation_shape",
            "## THE SHAPE OF THIS CONVERSATION",
            _matches_conversation_shape,
            _read_conversation_shape,
            examples=(
                # The live miss: answered "about an hour" and named topics that
                # had never come up.
                "how long have we been talking?",
                "what have we talked about so far?",
                "how many messages have I sent you?",
                "what did we cover earlier?",
            ),
            counter_examples=(
                # Uptime is not conversation length.
                "how long have you been running?",
                "how long will it take to build?",
                "what is 2 + 2",
                "how are you doing",
            ),
        ),
        Observable(
            "transcript",
            "## WHAT WAS ACTUALLY SAID IN THIS CONVERSATION",
            _matches_transcript,
            _read_transcript,
            examples=(
                "what did I ask you two messages ago?",
                "what was my first question?",
                "what was the first thing I said to you in this conversation?",
                "what was the last thing I told you?",
                "repeat back what I said",
            ),
            counter_examples=(
                # Contains "first" and asks about a file: a recall matcher that
                # swallowed this would break file reading to fix recall.
                "what is the first rule in CONTRIBUTING.md",
                "how are you doing",
                "what did you read?",
            ),
        ),
        Observable(
            "belief_history",
            "## POSITIONS I HAVE ACTUALLY REVISED",
            _matches_belief_history,
            _read_belief_history,
            examples=(
                "what's something you've genuinely changed your mind about?",
                "name one actual position you held and then dropped, with when",
                "what did you used to think that you no longer think?",
                "have you revised any beliefs lately?",
                "what do you think differently about now?",
            ),
            counter_examples=(
                # About the other person's mind, not hers.
                "have I changed my mind about anything?",
                "how are you doing",
                "what is 2 + 2",
                "what did I ask you two messages ago?",
            ),
        ),
        Observable(
            "lifetime",
            "## HOW LONG I HAVE BEEN AWAKE",
            _matches_lifetime,
            _read_lifetime,
            examples=(
                "how long have you actually been awake across all your restarts?",
                "how many turns have we had today?",
                "how long have you been alive?",
                "what's your total uptime?",
                "how many sessions have you had?",
            ),
            counter_examples=(
                # This run, not the whole life — the operational-state reading
                # owns that one, and answering it with 40 days would be wrong.
                "how long have you been running this session?",
                "how are you doing",
                "what is 2 + 2",
            ),
        ),
    ):
        register_observable(observable)


install_default_observables()


def observable_names() -> list[str]:
    from core.brain.observable_grounding import OBSERVABLES

    return [observable.name for observable in OBSERVABLES]


def _unused(value: Any) -> Any:  # pragma: no cover - keeps Any import honest
    return value
