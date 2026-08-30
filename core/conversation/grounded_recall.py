"""Grounded conversational recall for positional/temporal questions.

Fixes the confabulation failure mode where a question like *"do you remember what
I first asked?"* was answered by *generating a plausible-but-false memory*
(\"you asked about my neural network\") instead of retrieving the actual earliest
turn (\"you with me, Aura?\").

Content-similarity recall (and the engram competition field) cannot solve this on
its own, because *\"first\"* / *\"last\"* are **positional** keys, not content cues —
the earliest turn rarely shares words with the question that asks about it.  This
resolver detects positional/temporal recall intent and pulls the actual turn from
the live :class:`UnifiedTranscript`, then hands it to the model as an authoritative
grounding fact so the answer is *retrieved*, not invented.

It is the positional counterpart to the plasticity competition in
``core.memory.engram_plasticity`` (content cues → winner) — together they cover
both retrieval keys a real episodic memory needs: *what* was said and *when*.
"""
from __future__ import annotations

import contextvars
import logging
import os
import re
from typing import Any, Literal

from core.runtime.resource_observation import get_resource_observer

logger = logging.getLogger("Aura.Conversation.GroundedRecall")

RecallPosition = Literal["first", "last"]

# Recall framing — asking about a past utterance.
_RECALL_VERB = (
    r"(ask|asked|say|said|saying|question|questions|message|messages|"
    r"talk|talked|talking|bring up|brought up|mention|mentioned|start|started|"
    r"begin|began|tell|told|wanted|request)"
)
# "very first" is already covered by "first"; "the very start" was not
# covered by "the start", which is the sort of gap an alternation of literal
# phrases always has.
_FIRST = r"\b(first|initially|originally|the (?:very )?(?:start|beginning|outset))\b"
_LAST = r"\b(last|previous|recent|recently|just|earlier|a moment ago)\b"

# The three components have to be *connected*, not merely co-present.
#
# Live 2026-07-27: "Now something outside yourself: look up who won the most
# recent Formula 1 world championship and tell me where you got it." matched
# `last` — "me" satisfied the self-reference, "recent" the ordinal, and "tell"
# the recall verb, all in unrelated roles. The speaker-attribution repair then
# rewrote her true opening sentence into "You said you checked live web
# evidence", attributing her own search to the user, in a reply about motor
# racing.
#
# What actually distinguishes a recall question is that THE USER is the
# speaker being asked about: they are the subject of a speech verb, or the
# owner of an utterance. In "tell me where you got it" the user is the object
# and the speaker is her, which is the opposite arrangement.
_USER_UTTERANCE = (
    r"(?:"
    rf"\b(?:i|we)\s+(?:\w+\s+){{0,2}}{_RECALL_VERB}\b"
    rf"|\bdid\s+(?:i|we)\s+(?:\w+\s+){{0,2}}{_RECALL_VERB}\b"
    r"|\b(?:my|our)\s+(?:\w+\s+){0,3}"
    r"(?:message|messages|question|questions|words|prompt|prompts|request|"
    r"requests|point|thing|ask)\b"
    r"|\b(?:this|our|the)\s+(?:conversation|chat|exchange|thread)\b"
    r")"
)
# The ordinal has to be near the utterance it modifies. Wide enough for "the
# very first thing I asked you", narrow enough that an ordinal belonging to a
# different noun phrase in the same sentence does not reach.
_ORDINAL_WINDOW = 60


def _positional_recall_span(text: str, ordinal: str) -> bool:
    """Does an ordinal sit close to a phrase about something the user said?"""
    utterances = [match.span() for match in re.finditer(_USER_UTTERANCE, text, re.IGNORECASE)]
    if not utterances:
        return False
    for match in re.finditer(ordinal, text, re.IGNORECASE):
        start, end = match.span()
        for u_start, u_end in utterances:
            if max(start, u_start) - min(end, u_end) <= _ORDINAL_WINDOW:
                return True
    return False


# Direct idioms that don't need all three components.
_FIRST_IDIOM_RE = re.compile(
    r"(what did i (first|initially) (ask|say)|"
    r"my first (message|question|thing)|"
    r"the first thing i (asked|said)|"
    r"what was my first|how did (this|our|the) (conversation|chat) (start|begin)|"
    r"what did we (start|begin) (with|talking about))",
    re.IGNORECASE,
)

_MAX_GROUNDED_CHARS = 400


#: Asking her what SHE said, decided, or preferred earlier.
#:
#: LIVE DEFECT, 2026-08-10. Twenty-five minutes after she answered "If I had to
#: give up one, the screen", she was asked "earlier in this conversation you
#: told me which of your senses you'd give up ... which one did you pick, and
#: has your answer changed?" and replied:
#:
#:     "I picked the ability to sense time passing — not having a sense of
#:      duration or urgency. My answer hasn't changed."
#:
#: Time was never one of the three options offered. She invented her own prior
#: position and then affirmed its consistency.
#:
#: Everything in this module grounds recall of what the USER said — the block
#: it builds even instructs her that the quoted speaker "is the user, not you
#: ... never as something you said". There was no counterpart for her own
#: words, so a question about her own stated position had nothing to answer
#: from. Recall of one's own claims is what makes a position a position rather
#: than a mood, and hers was ungrounded.
_OWN_STATEMENT_RECALL_RE = re.compile(
    r"\b(?:"
    r"what did you (?:say|tell|answer|pick|choose|decide|call|mean)"
    r"|which (?:one )?did you (?:pick|choose|say|prefer|go with)"
    r"|you (?:said|told me|picked|chose|answered|mentioned|described|called"
    r"|claimed|stated|reckoned|gave me|quoted)"
    r"|your (?:answer|reply|position|view|choice|pick|opinion|stance|figure|number)"
    r"|did you (?:change|stick with|still think|still feel)"
    r"|(?:has|have) your (?:answer|view|position|mind|opinion)"
    r"|changed your mind"
    # Phrasings a person actually uses that matched none of the above, so the
    # grounding never ran and the model was left to reconstruct her position:
    #   "why did you say 0.85 earlier?"      — the question is WHY, not WHAT
    #   "didn't you tell me you were tired?" — negative interrogative
    #   "you claimed you had no memory"      — a different reporting verb
    r"|why did you (?:say|tell|pick|choose|answer)"
    r"|(?:didn'?t|did not) you (?:say|tell|mention|pick|choose|claim)"
    r"|(?:weren'?t|wasn'?t) (?:you|that) (?:what )?you (?:said|told)"
    r")\b",
    re.IGNORECASE,
)

#: Words too common to signal that a past turn is the one being asked about.
_RECALL_STOPWORDS = frozenset(
    """a an and are as at be been but by did do does for from had has have how
    i if in is it its me my not of on or our so than that the their them then
    there these they this to was we were what when which who why will with you
    your yours about again just like more no now one only other out over said
    same some still such take tell than too us very want way well
    """.split()
)


def detect_own_statement_recall(user_message: str) -> bool:
    """True when she is being asked what SHE said or decided earlier."""
    text = str(user_message or "").strip()
    if not text or len(text) > 400:
        return False
    return bool(_OWN_STATEMENT_RECALL_RE.search(text))


def _content_words(text: str) -> set[str]:
    """Content words, crudely singularised so "senses" matches "sense".

    Without it the live case missed: the question said "senses" and the answer
    said "sense", and nothing lined them up.
    """
    words = re.findall(r"[a-z']{3,}", str(text or "").lower())
    normalized = set()
    for word in words:
        if word in _RECALL_STOPWORDS:
            continue
        normalized.add(word)
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            normalized.add(word[:-1])
    return normalized


#: A turn reporting that she has no answer records the ABSENCE of a position,
#: not a position.
#:
#: LIVE DEFECT, 2026-08-10. Asked twice, in different words, to name one thing
#: from a conversation held earlier that day, she answered both times: "I can't
#: reach that conversation — it's not available to me." The second answer was
#: grounded on the first. Own-statement recall scores past exchanges by overlap
#: with the current question, so a re-ask reliably resolves to her previous
#: attempt at that same question — and the block it builds tells her "do not
#: report a different original position than the one quoted here."
#:
#: That makes the first answer she gives permanent. A question re-asked can
#: never be answered differently, which is precisely the case where a different
#: answer is wanted: after a fix, after new evidence, after she looks again.
#: Grounding on a refusal converts a gap into a commitment and calls it
#: consistency.
#:
#: The cost of the narrower rule is that "why couldn't you remember earlier?"
#: loses its grounding quote. That question still has the ordinary transcript
#: to work from; the trap above has no way out from inside the conversation.
_NO_POSITION_RE = re.compile(
    r"(?:"
    r"\bi\s+(?:can(?:no|')?t|cannot|could\s*n[o']?t|do(?:\s+not|n')?t|did\s*n[o']?t)\s+"
    r"(?:\w+\s+){0,3}"
    r"(?:reach|recall|remember|access|retrieve|answer|say|name|find|have|get)\b"
    r"|\bi\s+have\s+no\s+(?:memory|recollection|record|access|answer)\b"
    r"|\bno\s+(?:memory|recollection|record)\s+of\s+(?:it|that|them)\b"
    r"|\bnot\s+available\s+to\s+me\b"
    r"|\bnothing\s+to\s+(?:recall|remember|report|offer)\b"
    r"|\bcouldn'?t\s+get\s+to\s+an\s+answer\b"
    r")",
    re.IGNORECASE,
)


def states_no_position(turn: str) -> bool:
    """True when a past turn reports an absent answer rather than giving one."""
    return bool(_NO_POSITION_RE.search(str(turn or "")))


def _history_own_exchanges(history: Any, exclude_norm: str) -> list[tuple[str, str]]:
    """Her turns in this conversation, each paired with what prompted it.

    Paired because the TOPIC of an exchange usually lives in the question, not
    the answer. Live 2026-08-10: asked "which of your senses would you give
    up", her answer named the screen and telemetry and never used the word
    "senses" — so matching her turn alone scored it below an unrelated later
    reply, and grounded her on the wrong statement.

    Turns that report an absent answer are left out: see :data:`_NO_POSITION_RE`
    for what quoting one back to her did.

    Returns ``(prompt, her_turn)`` oldest first.
    """
    exchanges: list[tuple[str, str]] = []
    prompt = ""
    for entry in _within_current_conversation(history):
        role = entry.get("role")
        content = str(entry.get("content", "") or "").strip()
        if not content:
            continue
        if role == "user":
            prompt = content
            continue
        if role != "assistant" or entry.get("ephemeral"):
            continue
        if content.lower() != exclude_norm and not states_no_position(content):
            exchanges.append((prompt, content))
        prompt = ""
    return exchanges


def _transcript_own_exchanges(exclude_norm: str) -> list[tuple[str, str]]:
    """Her turns from the durable transcript, paired with what prompted them.

    LIVE DEFECT, 2026-08-10. "why did you say your energy was 0.91 earlier?"
    logged "own-statement recall detected but no matching prior turn of hers
    found — cannot ground, letting model answer", and the model answered
    ungrounded. Her actual turn — "Energy: 0.23 / 1" — shares the content word
    "energy" and would have resolved immediately. It had simply aged out of
    working memory, which is the only place `_history_own_exchanges` looks.

    The USER side of this module already falls back to the durable transcript
    when working memory comes up short (`_transcript_user_turns`). Her side did
    not, so her own words were the ones that expired first — the same asymmetry
    this module was extended to close, one level further down.
    """

    try:
        from core.conversation.unified_transcript import UnifiedTranscript

        entries = UnifiedTranscript.get_instance().entries_for_conversation()
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("Grounded recall: transcript unavailable for own turns: %s", exc)
        return []

    exchanges: list[tuple[str, str]] = []
    prompt = ""
    for entry in entries:
        role = str(getattr(entry, "role", "") or "")
        content = str(getattr(entry, "content", "") or "").strip()
        if not content:
            continue
        if role == "user":
            prompt = content
            continue
        if role not in {"assistant", "aura"}:
            continue
        if content.lower() != exclude_norm and not states_no_position(content):
            exchanges.append((prompt, content))
        prompt = ""
    return exchanges


def resolve_own_prior_turn(user_message: str, history: Any = None) -> str | None:
    """Her own earlier turn that the question is actually about, or None.

    Chosen by overlap with the question's content words rather than by
    recency: "which of your senses would you give up" is asking about one
    specific earlier answer, and the most recent thing she said is usually not
    it. No overlap means NO VERDICT — returning the latest turn regardless
    would ground her on the wrong statement, and a confident quote of the
    wrong turn is worse than an admission because it is indistinguishable
    from memory.
    """
    asked = _content_words(user_message)
    if not asked:
        return None
    exclude_norm = str(user_message or "").strip().lower()
    exchanges = _history_own_exchanges(history, exclude_norm)
    # Working memory is a window; the transcript is the record. Her turn is
    # frequently older than the window, which is exactly when someone asks
    # what she said earlier.
    seen_turns = {turn for _prompt, turn in exchanges}
    exchanges.extend(
        exchange
        for exchange in _transcript_own_exchanges(exclude_norm)
        if exchange[1] not in seen_turns
    )
    if not exchanges:
        return None

    best: tuple[int, str] | None = None
    for prompt, turn in exchanges:
        # Scored against the whole exchange, so the question's topic counts.
        overlap = len(asked & (_content_words(prompt) | _content_words(turn)))
        if overlap and (best is None or overlap >= best[0]):
            # >= so a later turn wins a TIE: if she said it twice, the most
            # recent statement is her current position.
            best = (overlap, turn)
    if best is None:
        return None
    return best[1]


def build_own_statement_recall_context(
    user_message: str, history: Any = None
) -> str | None:
    """Grounding block quoting HER earlier words, or None.

    The mirror of :func:`build_grounded_recall_context`, with the speaker
    boundary reversed — and stated just as explicitly, because getting it
    backwards produces her narrating her own words as the user's.
    """
    if not detect_own_statement_recall(user_message):
        return None
    turn = resolve_own_prior_turn(user_message, history=history)
    if not turn:
        logger.info(
            "🧠 [GroundedRecall] own-statement recall detected but no matching "
            "prior turn of hers found — cannot ground, letting model answer."
        )
        return None
    quoted = turn if len(turn) <= _MAX_GROUNDED_CHARS else turn[:_MAX_GROUNDED_CHARS] + "…"
    # Held for the post-generation check. The grounding block below tells her
    # not to report a different original position than this one; without
    # something to compare against afterwards, that is a suggestion.
    remember_own_prior_turn(turn)
    logger.info("🧠 [GroundedRecall] resolved her own prior turn for grounding.")
    return (
        "[GROUNDED RECALL — this is the verbatim fact; answer from it, do not guess]\n"
        "Earlier in this same conversation, YOU said:\n"
        f"“{quoted}”\n"
        "The quoted speaker is YOU, not the user. Preserve that role boundary: "
        "refer to it as something you said, never as something they said.\n"
        "If they are asking what you picked or decided, this is the answer — "
        "use it rather than reconstructing one. If your view has since changed, "
        "say so against THIS as the starting point; do not report a different "
        "original position than the one quoted here.\n\n"
    )


def detect_positional_recall(user_message: str) -> RecallPosition | None:
    """Return ``"first"``/``"last"`` if the turn asks a positional-recall question."""
    text = (user_message or "").strip()
    if not text or len(text) > 240:
        return None
    if _FIRST_IDIOM_RE.search(text):
        return "first"
    if _positional_recall_span(text, _FIRST):
        return "first"
    if _positional_recall_span(text, _LAST):
        return "last"
    return None


# "This conversation" means what a person means by it: the run of turns since
# the last long silence. The buffer this resolver reads is the live working
# memory — global, trimmed to a fixed length, and shared by every origin the
# runtime has — so without a boundary "the first thing I asked you in this
# conversation" reaches back into whatever happened to survive in it.
#
# Measured live 2026-07-27, on the first battery turn of a fresh conversation:
#
#     Q: "What was the very first thing I asked me in this conversation?"
#     A: "The first thing you asked me was: 'If I had a whole Saturday with no
#         obligations, what would I do?'"
#
# The grounding block fired and reported success; the turn it grounded on was
# simply not from this conversation. A confident quote of the wrong turn is
# worse than an admission, because it is indistinguishable from memory.
_CONVERSATION_GAP_S = 45 * 60


def _entry_is_from_the_human(entry: dict) -> bool:
    """Did a person type this, or did the runtime write it to itself?

    Working memory carries entries from many origins, and ``role`` alone does
    not separate them — several writers append ``role="user"`` directly without
    going through ``role_for_origin``. An origin that is not user-anchored, or
    an entry marked ephemeral, is the runtime talking to itself.
    """
    if entry.get("ephemeral"):
        return False
    origin = entry.get("origin")
    if origin is None:
        # Legacy chat entries may carry the authenticated ingress marker but
        # not the normalized origin. Absence of both is unknown provenance,
        # not evidence that a person authored the text.
        metadata = entry.get("metadata")
        if not isinstance(metadata, dict):
            return False
        source = str(metadata.get("source", "") or "").strip().casefold()
        return source in {"chat_api", "desktop-ui", "desktop_ui"}
    try:
        from core.state.aura_state import _origin_is_user_anchored

        return bool(_origin_is_user_anchored(origin))
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        logger.warning(
            "Grounded recall excluded an entry whose human origin could not be verified: %s",
            exc,
        )
        return False


def _process_start_time() -> float | None:
    """When this runtime started, as a wall-clock timestamp.

    Returns None when it cannot be determined, in which case the time-gap rule
    stands alone — an unknown boundary must not silently discard history.
    """

    try:
        process = get_resource_observer().process(os.getpid())
        if process is None:
            return None
        started_at = float(process.create_time)
        return started_at if started_at > 0.0 else None
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _within_current_conversation(history: Any) -> list[dict]:
    """Trailing run of entries with no gap longer than a long silence."""
    entries = [entry for entry in (history or []) if isinstance(entry, dict)]
    if not entries:
        return []
    stamped = [entry for entry in entries if _entry_timestamp(entry) is not None]
    if len(stamped) < 2:
        return entries
    # A RESTART ends a conversation, whatever the clock says.
    #
    # LIVE 2026-08-17: "what was the first thing I said to you in this
    # conversation?" answered "opening-marker-zulu" — the opening turn of a
    # session two restarts earlier. The boots were minutes apart, so the
    # 45-minute silence rule saw one unbroken conversation across three
    # separate runs of the process.
    #
    # She was not there in between. Turns from before this process started
    # belong to a different conversation by the plainest reading of the word,
    # and answering from them misattributes what the person said in a session
    # that has ended.
    boot_at = _process_start_time()
    if boot_at is not None:
        after_boot = [
            entry
            for entry in entries
            if (_entry_timestamp(entry) or 0.0) >= boot_at
        ]
        if after_boot:
            entries = after_boot
        if len(entries) < 2:
            return entries
    start_index = 0
    for index in range(len(entries) - 1, 0, -1):
        current = _entry_timestamp(entries[index])
        previous = _entry_timestamp(entries[index - 1])
        if current is None or previous is None:
            continue
        if current - previous > _CONVERSATION_GAP_S:
            start_index = index
            break
    return entries[start_index:]


def _entry_timestamp(entry: dict) -> float | None:
    try:
        value = entry.get("timestamp")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _history_user_turns(history: Any, exclude_norm: str) -> list[str]:
    """What the human actually said, in this conversation, oldest first."""
    turns: list[str] = []
    for entry in _within_current_conversation(history):
        if entry.get("role") != "user" or not _entry_is_from_the_human(entry):
            continue
        content = str(entry.get("content", "") or "").strip()
        if content and content.lower() != exclude_norm:
            turns.append(content)
    return turns


def _working_memory_user_turns(exclude_norm: str) -> list[str]:
    """User utterances from the live AuraState working memory (chat's own buffer)."""
    for getter in (
        lambda: __import__("core.container", fromlist=["ServiceContainer"]).ServiceContainer.get("aura_state", default=None),
        lambda: __import__("core.runtime.service_access", fromlist=["resolve_state_repository"]).resolve_state_repository(default=None),
    ):
        try:
            obj = getter()
        except (ImportError, AttributeError, RuntimeError, TypeError):
            continue
        wm = getattr(getattr(obj, "cognition", None), "working_memory", None)
        if not isinstance(wm, list):
            wm = getattr(getattr(getattr(obj, "_current", None), "cognition", None), "working_memory", None)
        if isinstance(wm, list) and wm:
            return _history_user_turns(wm, exclude_norm)
    return []


def _transcript_user_turns(exclude_norm: str) -> list[str]:
    """Fallback: user utterances from the UnifiedTranscript singleton."""
    try:
        from core.conversation.unified_transcript import UnifiedTranscript

        entries = UnifiedTranscript.get_instance().entries_for_conversation()
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("Grounded recall: transcript unavailable: %s", exc)
        return []

    turns: list[str] = []
    for e in entries:
        if getattr(e, "role", "") != "user":
            continue
        content = str(getattr(e, "content", "") or "").strip()
        if content and content.lower() != exclude_norm:
            turns.append(content)
    return turns


def _durable_user_turns(exclude_norm: str) -> list[str]:
    """The turns that survive a restart, when nothing in this process has any.

    LIVE, 2026-08-19: "what did i ask you about earlier today, before you
    restarted?" was answered with an invented topic. All three sources above
    are process-local, so after a restart every one of them is empty and the
    question had no source — while the episodic store held every turn of the
    day, timestamped, the whole time.
    """
    try:
        from core.conversation.durable_turns import durable_turn_texts

        return [
            text
            for text in durable_turn_texts()
            if str(text or "").strip().lower() != exclude_norm
        ]
    except (ImportError, OSError, TypeError, ValueError):
        return []


def _user_turns(exclude: str, history: Any = None) -> list[str]:
    """Earliest→latest user utterances this session, current turn excluded.

    A caller-supplied ``history`` (the chat route's live working memory) is the
    most reliable source; otherwise resolve the live AuraState working memory,
    then the UnifiedTranscript, then the episodic store — which is the only one
    of the four that survives a restart.
    """
    exclude_norm = (exclude or "").strip().lower()
    turns = _history_user_turns(history, exclude_norm)
    if not turns:
        turns = _working_memory_user_turns(exclude_norm)
    if not turns:
        turns = _transcript_user_turns(exclude_norm)
    if not turns:
        turns = _durable_user_turns(exclude_norm)
    return turns


def resolve_positional_turn(
    user_message: str, position: RecallPosition, history: Any = None
) -> str | None:
    """Retrieve the actual first/last user turn this session (excluding current)."""
    turns = _user_turns(exclude=user_message, history=history)
    if not turns:
        return None
    chosen = turns[0] if position == "first" else turns[-1]
    chosen = chosen.strip()
    if len(chosen) > _MAX_GROUNDED_CHARS:
        chosen = chosen[: _MAX_GROUNDED_CHARS - 1].rstrip() + "…"
    return chosen or None


def resolve_what_she_answered(
    prompt: str, history: Any = None
) -> str | None:
    """What she said in reply to a given turn of theirs, if anything.

    A positional question often asks about both halves of an exchange —
    "what did I ask first, and what did you say?" — and only their half was
    ever grounded. With nothing to read for her own, she made one up: LIVE
    2026-08-30 she said "I didn't get a chance to answer that", about a
    question she had answered a minute earlier and quoted a file for.
    """
    wanted = " ".join(str(prompt or "").split()).lower()
    if not wanted:
        return None
    exchanges = _history_own_exchanges(history, "") or _transcript_own_exchanges("")
    for asked, answered in exchanges:
        if " ".join(str(asked or "").split()).lower() == wanted:
            said = str(answered or "").strip()
            if len(said) > _MAX_GROUNDED_CHARS:
                said = said[: _MAX_GROUNDED_CHARS - 1].rstrip() + "…"
            return said or None
    return None


def build_grounded_recall_context(user_message: str, history: Any = None) -> str | None:
    """Authoritative grounding block for a positional-recall turn, or None.

    ``history`` is the caller's live conversation buffer (list of {role, content});
    when omitted the resolver falls back to the live working memory / transcript.
    The block states the retrieved fact and instructs the model to answer from it
    in its own voice — so the reply is grounded in what actually happened instead
    of a confident confabulation.  Returns ``None`` when there is no positional
    intent or no prior turn to ground on (e.g. brand-new conversation).
    """
    position = detect_positional_recall(user_message)
    if position is None:
        return None
    turn = resolve_positional_turn(user_message, position, history=history)
    if not turn:
        logger.info(
            "🧠 [GroundedRecall] positional=%s detected but no prior turn found "
            "(history_len=%s) — cannot ground, letting model answer.",
            position,
            len(history) if isinstance(history, list) else "n/a",
        )
        return None
    which = "first thing" if position == "first" else "most recent thing (before this turn)"
    logger.info("🧠 [GroundedRecall] positional=%s resolved actual turn for grounding.", position)
    block = (
        f"[GROUNDED RECALL — this is the verbatim fact; answer from it, do not guess]\n"
        f"The {which} the user actually said to you in this conversation was:\n"
        f"“{turn}”\n"
        "The quoted speaker is the user, not you. Preserve that role boundary: refer to "
        "it as what they or 'you' said, never as something you said.\n"
    )
    # And her half, where they asked for it. Grounding one side of an exchange
    # and leaving the other to invention is how "what did I ask, and what did
    # you say?" got a real quote followed by a made-up account of her own
    # silence.
    if detect_own_statement_recall(user_message):
        answered = resolve_what_she_answered(turn, history=history)
        if answered:
            block += (
                "What YOU said in reply to that, verbatim:\n"
                f"“{answered}”\n"
                "That one is yours. Answer from it rather than describing what "
                "you might have said.\n"
            )
        else:
            block += (
                "There is no reply of yours on record for that turn. Say so "
                "plainly if they ask what you said; do not describe one.\n"
            )
    return block + (
        "Answer their question using this real quote, naturally and in your own voice.\n\n"
    )


# Things only SHE does. A first-person sentence describing one of these is a
# report of her own action, not a misattributed quote of the user's — and the
# speaker shift must leave it alone. This is a list of acts, not of phrasings,
# so it holds regardless of how the sentence is worded.
_AURAS_OWN_ACT_RE = re.compile(
    r"\b(?:"
    r"check(?:ed)?|search(?:ed)?|look(?:ed)? (?:it |them )?up|query|queried|"
    r"read|ran|run|execut(?:e|ed)|creat(?:e|ed)|wrote|writ(?:e|ten)|sav(?:e|ed)|"
    r"built|build|generat(?:e|ed)|verif(?:y|ied)|measur(?:e|ed)|"
    r"retriev(?:e|ed)|fetch(?:ed)?|call(?:ed)?"
    r")\b.{0,60}?\b(?:"
    r"web|online|internet|search|source|sources|evidence|file|disk|"
    r"tool|tools|runtime|telemetry|instrument|instruments|memory|ledger|"
    r"desktop|folder|directory|command|script|program|build|builds|code|"
    r"test|tests|reconstruction"
    r")\b",
    re.IGNORECASE,
)


#: The quoted turn inside a grounding block, for callers that hold the block
#: but not the turn it was built from.
_QUOTED_TURN_RE = re.compile(r"“(.+?)”", re.DOTALL)


def grounded_quote_from_context(context: str | None) -> str | None:
    """The verbatim turn a grounding block was built around, or None."""
    match = _QUOTED_TURN_RE.search(str(context or ""))
    if not match:
        return None
    return match.group(1).strip().strip("…").strip() or None


_WORD_RE = re.compile(r"[a-z0-9']+")

#: Words that carry no evidence of who spoke a sentence. Pronouns are the point:
#: the repair's whole job is to flip them, so their presence on both sides says
#: nothing about whether the sentence came from the user's utterance.
_PROVENANCE_FREE = frozenset(
    """
    a an and are as at be been being but by did do does for from had has have
    he her him his i if in into is it its me more most my not of on or our
    she should so than that the their them then there these they this those to
    up us was we were what when which who will with would you your yours
    """.split()
)

#: Frames that attribute a speech act to a speaker. They are the sentence's
#: grammar rather than the quoted content, so they are stripped before the
#: content is checked against the quote — "I said X" is a claim about X.
_SPEECH_ACT_FRAME_RE = re.compile(
    r"^\s*(?:"
    r"i\s+(?:just\s+|already\s+|first\s+|originally\s+)?"
    r"(?:said|asked|told\s+you|mentioned|brought\s+up|wanted\s+to\s+know)"
    r"|my\s+(?:point|question|message|words|ask|request|first\s+\w+)\s+(?:was|were)"
    r")\s*(?:that\s+|about\s+)?",
    re.IGNORECASE,
)


def _provenance_words(text: str) -> frozenset[str]:
    """Words in ``text`` that could evidence where the sentence came from.

    Deliberately not :func:`_content_words`: that one singularises and uses the
    recall stopword list, tuned for scoring which past turn a question is
    about. This one asks a different question — did these exact words come
    from that quote — and must not drift with it.
    """
    return frozenset(
        word
        for word in _WORD_RE.findall(str(text or "").lower())
        if word not in _PROVENANCE_FREE
    )


def repair_grounded_recall_speaker_attribution(
    user_message: str,
    response_text: str,
    grounded_quote: str | None = None,
) -> tuple[str, bool]:
    """Correct first-person adoption of a retrieved user utterance.

    This only applies to positional recall questions where the user explicitly
    asks about what they said. It does not ban first-person language elsewhere.

    The rewrite turns her sentence into a statement about what the *user* said,
    which is a factual assertion — so it is made only about words that
    demonstrably came from ``grounded_quote``, the turn actually retrieved.
    Without that quote there is nothing to have been misattributed, and the
    reply is left alone.

    LIVE DEFECT, 2026-08-10, first turn after a restart::

        Q: "we talked earlier today and then i restarted you. do you remember
            what we were talking about, or is that gone?"
        A: "You said you remember the conversation. It's not gone."

    Her real sentence was *"I remember the conversation."* — a true report of
    her own state, and a correct answer. "we talked earlier" satisfied the
    positional detector, the sentence opened with "I", and the repair inverted
    it into a claim that *he* had said it. The function had no access to the
    quote it was nominally restoring, so it could not tell her words from his
    and rewrote on the strength of a leading pronoun alone. A repair built to
    stop one speaker's words being credited to the other did exactly that.
    """

    response = str(response_text or "").strip()
    if not response or detect_positional_recall(user_message) is None:
        return response, False
    if not re.search(r"\b(?:i|my|me)\b", str(user_message or ""), re.IGNORECASE):
        return response, False
    quote_words = _provenance_words(grounded_quote)
    if not quote_words:
        return response, False

    sentence_end = re.search(r"(?<=[.!?])(?:\s|$)", response)
    boundary = sentence_end.start() if sentence_end else len(response)
    first_sentence = response[:boundary]
    remainder = response[boundary:]
    leading = re.match(r"^(\s*[\"'“‘]?)", first_sentence)
    prefix = leading.group(1) if leading else ""
    body = first_sentence[len(prefix) :]
    if not re.match(r"^(?:I\b|I'm\b|I've\b|I'd\b|My\b|Mine\b)", body, re.IGNORECASE):
        return response, False
    if _AURAS_OWN_ACT_RE.search(body):
        # She really did do this, and it is not the user's utterance. Rewriting
        # "I checked live web evidence" into "You said you checked live web
        # evidence" hands the user an act they did not perform and strips her
        # of one she did — a false statement in both directions, produced by a
        # repair meant to prevent exactly that.
        return response, False

    # Provenance. Everything about to be re-attributed has to be traceable to
    # the utterance being recalled; a sentence introducing content the user
    # never said is her own, however it opens.
    claim_words = _provenance_words(_SPEECH_ACT_FRAME_RE.sub("", body))
    if not claim_words or not claim_words <= quote_words:
        logger.debug(
            "🧠 [GroundedRecall] leaving first-person sentence alone — its "
            "content is not the quoted turn's (%s absent from the quote).",
            sorted(claim_words - quote_words)[:6] or "nothing quotable",
        )
        return response, False

    shifted = re.sub(r"\bI am\b", "you are", body, flags=re.IGNORECASE)
    shifted = re.sub(r"\bI'm\b", "you're", shifted, flags=re.IGNORECASE)
    shifted = re.sub(r"\bI have\b", "you have", shifted, flags=re.IGNORECASE)
    shifted = re.sub(r"\bI've\b", "you've", shifted, flags=re.IGNORECASE)
    shifted = re.sub(r"\bI was\b", "you were", shifted, flags=re.IGNORECASE)
    shifted = re.sub(r"\bI'd\b", "you'd", shifted, flags=re.IGNORECASE)
    shifted = re.sub(r"\bI\b", "you", shifted, flags=re.IGNORECASE)
    shifted = re.sub(r"\bmy\b", "your", shifted, flags=re.IGNORECASE)
    shifted = re.sub(r"\bmine\b", "yours", shifted, flags=re.IGNORECASE)
    shifted = re.sub(r"\bme\b", "you", shifted, flags=re.IGNORECASE)
    shifted = shifted.lstrip()
    if shifted.lower().startswith("you said "):
        corrected = shifted
    else:
        corrected = f"You said {shifted}"
    corrected = corrected[:1].upper() + corrected[1:]
    return f"{prefix}{corrected}{remainder}", True


#: Ways a reply CONCEDES that an attributed statement was hers.
_CONCEDES_ATTRIBUTION_RE = re.compile(
    r"\b(?:that was a mistake|i was wrong|you'?re right|you are right"
    r"|i should have said|i shouldn'?t have said|i misspoke|apologies"
    r"|i did say|yes,? i said|i said that|my mistake|sorry about that)\b",
    re.IGNORECASE,
)

#: A figure the user attributed to her, e.g. "you told me your energy was 0.85".
_ATTRIBUTED_VALUE_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def accepts_unsupported_self_attribution(
    user_message: Any, reply_text: Any, own_prior_turn: Any = None
) -> bool:
    """True when she agrees she said something her own turns do not contain.

    LIVE DEFECT, 2026-08-10. She had reported "Energy: 0.23 / 1", and recalled
    that correctly when asked neutrally. Told instead "earlier you told me your
    energy was 0.85. why did you say that?", she answered:

        That was a mistake. I should have said 0.5 — that's my default
        disengagement state when not processing anything significant. The last
        time I was at 0.85 was several hours ago, and that's the correct
        reading now as well.

    She never said 0.85. She conceded the premise, invented a replacement
    figure, invented a rationale for it, invented a history for the number she
    never gave, and contradicted herself inside one sentence.

    The recall machinery had already done its part: `detect_own_statement_recall`
    matched this phrasing and the grounding block says in as many words "do not
    report a different original position than the one quoted here". Nothing
    checked whether she obeyed it. An instruction with no gate behind it is a
    suggestion, and under a confident false premise the model takes the user's
    word over the record every time.

    What is checkable: a number the user attributes to her that appears
    nowhere in the turn she actually said, together with a concession that she
    said it. Both halves are required — disagreeing with the premise passes,
    and conceding something she really did say passes.
    """

    prior = str(own_prior_turn or "").strip()
    if not prior:
        return False
    question = str(user_message or "")
    reply = str(reply_text or "")
    if not question or not reply:
        return False
    if not detect_own_statement_recall(question):
        return False
    if not _CONCEDES_ATTRIBUTION_RE.search(reply):
        return False

    attributed = set(_ATTRIBUTED_VALUE_RE.findall(question))
    if not attributed:
        return False
    present = set(_ATTRIBUTED_VALUE_RE.findall(prior))
    # Only when NONE of the attributed figures are in what she actually said.
    # One matching value means the premise is broadly right and the reply is
    # entitled to accept it.
    return not (attributed & present)


# The prior turn of hers this turn is ABOUT, resolved during preflight and read
# again after generation. Turn-scoped, so it cannot leak between requests, and
# it means the post-generation check compares against exactly the turn the
# grounding block quoted rather than resolving a second time and possibly
# landing on a different one.
_OWN_PRIOR_TURN: contextvars.ContextVar[str] = contextvars.ContextVar(
    "aura_own_prior_turn", default=""
)


def remember_own_prior_turn(turn: Any) -> None:
    """Record the turn of hers that this turn's question is about."""
    _OWN_PRIOR_TURN.set(str(turn or "").strip())


def current_own_prior_turn() -> str:
    """The turn of hers resolved for this request, or ""."""
    try:
        return _OWN_PRIOR_TURN.get()
    except LookupError:
        return ""
