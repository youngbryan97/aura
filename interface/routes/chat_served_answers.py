"""Answers Aura can give from what she already has.

Lifted out of `interface/routes/chat.py`. A question about her own uptime, the
files in a directory, the last thing she worked on, a table she has already
read — these have answers that do not need a generation, and serving one is
faster and more truthful than describing it. Each returns the answer or None,
and None means the turn goes on to think.
"""
from __future__ import annotations

from core.runtime.errors import record_degradation
from interface.routes.chat_common import _CHAT_RECOVERABLE_ERRORS, logger
from pathlib import Path
from typing import Any
import re

# Lifted alongside this module; imported rather than re-derived.
from .chat_lane_bookkeeping import (
    _brevity_requested,
)
from .chat_reply_shaping import (
    _compose,
)


def _the_answer_has_to_be_worked_out(
    user_message: str,
    shape: Any,
) -> bool:
    """Whether this turn's answer is a derivation rather than a reading.

    The compact lane's whole justification is that the full phase stack adds no
    evidence: present state, recall and capability all have a snapshot to read
    from, and going the long way round only spends the deadline. That
    justification fails for a question whose answer does not exist anywhere
    until it has been worked out.

    LIVE, 2026-08-27: "45 becomes 15. 28 becomes 14. 66 becomes 22. What am I
    doing, what does 91 become?" went compact on two question parts, spent its
    budget on false starts about digit manipulation and stopped mid-derivation.
    The same class of question with three parts had gone the long way one turn
    earlier and found the rule cleanly. The lane was decided by counting parts,
    in two places that each wrote the count out again.

    Two independent things mark a turn as a derivation, and neither contains
    the other: several obligations to satisfy in one reply, and a work contract
    that already measured the answer as needing deliberation. The count stays,
    because a three-part request is heavy however little each part weighs.
    """

    if int(getattr(shape, "question_parts", 0) or 0) >= 3:
        return True
    try:
        from core.intent.capability_selection import points_at_something_real
        from core.language.semantic_work import build_semantic_work_contract

        # A path on this disk or an address is not in any snapshot, and the
        # quick lane's whole justification is that a snapshot already holds
        # the answer. The bytes are AT that place; the turn has to go and
        # look, then work from what it found.
        #
        # LIVE, 2026-08-28: "Something's off in <path> ... Go through the code
        # and tell me what's actually happening, with the file and line" is one
        # long sentence with one question in it, so nothing about its shape
        # asked for room. It went compact with 512 tokens, was handed
        # diagnose_repo, and ran out of budget before it could say what the
        # tool found.
        if points_at_something_real(user_message):
            return True
        return bool(build_semantic_work_contract(user_message).requires_deliberation)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Answer-work classification skipped: %s", exc)
        return False


async def _readings_for(text: str) -> list[str]:
    """The present-moment readings, on the ladder's path as well as the main one.

    Every observable is registered once and read once, and it was read on one
    path. A model asked about a fact it does not hold produces something
    fact-shaped, which is the whole reason the readings exist — and the ladder
    is exactly the case where the model holds least.
    """
    try:
        import core.brain.observable_registry  # noqa: F401  (registers)
        from core.brain.observable_grounding import observable_blocks

        readings = list(await observable_blocks(text))
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.fallback_ladder",
            exc,
            severity="warning",
            action="the smaller model answered without the readings",
        )
        return []
    if readings:
        logger.info(
            "🔭 [GROUNDING] ladder took %d reading(s): %s",
            len(readings),
            ",".join(
                block.split("\n", 1)[0].removeprefix("## ").lower()
                for block in readings
            ),
        )
    return readings


def _tabular_readings(path: object, question: str) -> list[str]:
    """One reading per question the message asks that this table can settle.

    A message asking two things gets two readings. Duplicates are dropped,
    because two clauses often resolve to the same column and the same figure
    said twice reads as two findings.
    """
    from core.conversation.tabular_answer import (
        answer_tabular_question,
        check_ranking_claim,
        describe_tabular_answer,
    )
    from core.language.asking_clauses import asking_clauses

    asked = [clause for clause in asking_clauses(question) if clause.strip()]
    said: list[str] = []
    seen: set[str] = set()
    # A premise about this table is checkable, and settling one is worth a
    # great deal more than doubting it.
    #
    # LIVE, 2026-08-27: "Since West came out top on average approved deal size
    # in deals.csv, what's West doing that the other regions should copy?" She
    # doubted the premise — correctly, the leader is South — and then reasoned
    # from figures she had never looked at: "West often has deals sitting for a
    # long time or getting rejected."
    correction = check_ranking_claim(path, question)
    if correction:
        said.append(correction)
        seen.add(correction)
    # The whole message is tried too: a clause split can lose context another
    # clause carried, and the common case is one question anyway.
    for candidate in ([question] if not asked else [*asked, question]):
        described = describe_tabular_answer(answer_tabular_question(path, candidate))
        if described and described not in seen:
            seen.add(described)
            said.append(described)
    return said


def _tables_named_in(question: str, *, already: list) -> list:
    """CSV and TSV files the person named that are actually on this disk."""
    seen = {str(path) for path in already}
    found: list = []
    try:
        from core.language.named_paths import named_paths

        for candidate in named_paths(question):
            if not candidate.lower().endswith((".csv", ".tsv")):
                continue
            resolved = Path(candidate).expanduser()
            if resolved.is_file() and str(resolved) not in seen:
                seen.add(str(resolved))
                found.append(resolved)
    except (ImportError, OSError, ValueError, TypeError):
        return found
    return found


def _serve_tabular_answer(user_message: object, reply: object) -> object:
    """Answer a quantitative question about a data file by computing it.

    LIVE, 2026-08-19. Given a 60-row CSV she had already read, asked which
    team spent the most on approved expenses and how much, every draft was
    rejected as arithmetic_answer_missing and the turn ended in a canned
    apology. The gate was right — the question asks for a number and no draft
    had one — and no model sums sixty rows reliably in its head.

    The file is on disk and the answer is arithmetic, so it is computed. When
    the question does not resolve to one unambiguous reading of the table this
    returns the model's reply untouched: a wrong number served with authority
    is worse than no number.
    """
    try:
        from core.conversation.filesystem_check import files_already_read

        question = str(user_message or "")
        if not question.strip():
            return reply
        tables = [
            path
            for path in files_already_read()
            if str(path).lower().endswith((".csv", ".tsv"))
        ]
        # A table the PERSON named is a table, whether or not anything read it
        # first.
        #
        # LIVE, 2026-08-27: "I've got a deals export at <path>. How many are
        # approved, what do they add up to, which region has the highest
        # average?" The model was offered a file reader and a REPL, called
        # neither, and answered from nothing; the draft was correctly rejected
        # for having no numbers and the turn ended in a canned apology. The
        # file was named in the sentence and the arithmetic needs no model at
        # all — this waited for somebody else to have opened it.
        tables.extend(_tables_named_in(question, already=tables))
        for path in tables[:3]:
            # Every question the table can answer, not the first one.
            #
            # LIVE, 2026-08-27: "how many of those are approved, and which
            # region has the highest average approved amount_gbp?" came back
            # with the regional means and nothing about the count — one
            # reading resolved, the other never attempted, and no mention that
            # half the message went unanswered.
            readings = _tabular_readings(path, question)
            if readings:
                logger.info(
                    "📊 Served %d computed reading(s) of %s.", len(readings), path
                )
                computed = "\n\n".join(readings)
                # A reading is evidence for an answer, not a replacement for it.
                #
                # LIVE, 2026-08-27: "Given Wren has the most deals in
                # deals.csv, should we put her on the enterprise accounts?"
                # came back "Wren is not top: Marek leads at 21 and Wren is at
                # 16." — the premise correctly settled and the question left
                # unanswered. Correcting somebody is not answering them.
                written = str(reply or "").strip()
                if not written or computed in written:
                    return computed
                from core.conversation.reply_provenance import (
                    ReplyProvenance,
                    admits_no_answer,
                    declared_provenance,
                )

                # Never follow evidence with an admission of having none.
                if (
                    declared_provenance(written) == ReplyProvenance.HONEST_FAILURE.value
                    or admits_no_answer(written)
                ):
                    return computed
                return f"{computed}\n\n{written}"
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.tabular_answer",
            exc,
            severity="debug",
            action="left the table question to the model",
            enforce_failure_policy=False,
        )
    return reply


async def _serve_solved_game(user_message: object, reply: object) -> object:
    """Answer "who wins" by enumerating the game rather than arguing about it.

    LIVE, 2026-08-22: given the rules of an invented game on nine squares she
    answered "move your piece one square on every turn", at high confidence,
    and called it a Nim variant the first player always wins. The conclusion
    was right by luck and the strategy loses. There are eight positions.
    """
    try:
        from core.conversation.session_scope import solved_answers
        from core.reasoning.game_answer import solve_described_game

        # Worked out in preflight on almost every turn; never plan it twice.
        solved = solved_answers().get("finite_game", "") or await solve_described_game(
            user_message
        )
        if not solved:
            return reply
        # The preflight answer is already the reply on a solved turn, and
        # composing it with itself printed the whole thing twice.
        if solved.strip() and solved.strip() in str(reply or ""):
            return reply
        from core.conversation.composed_answer import compose_measured
        from core.reasoning.game_planner import describes_a_game

        return compose_measured(user_message, reply, solved, describes_a_game)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.solved_game",
            exc,
            severity="debug",
            action="left the game question to the model",
            enforce_failure_policy=False,
        )
    return reply


def _serve_lifetime(user_message: object, reply: object) -> object:
    """Answer how long she has been alive from the record that counts it.

    LIVE, 2026-08-19: "how many turns have we had today, and how long have you
    actually been awake across all your restarts?" got "That's a complex
    question. The number of turns depends on how you count." Forty days across
    1,523 sessions was in continuity.json, and every turn of the day was in the
    episodic store.

    A deflection is the worst available answer to the one question a person
    asks to find out whether something has a life — and the true answer is
    more impressive than anything a hedge could suggest.
    """
    try:
        from core.brain.observable_registry import _matches_lifetime
        from core.self.lifetime import describe_lifetime

        if not _matches_lifetime(str(user_message or "")):
            return reply
        measured = describe_lifetime()
        if measured:
            logger.info("⏳ Served the cumulative lifetime from the continuity record.")
            from core.self.lifetime import strike_uptime_contradiction

            return _compose(
                user_message,
                reply,
                measured,
                _matches_lifetime,
                refute=strike_uptime_contradiction,
            )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.lifetime",
            exc,
            severity="debug",
            action="left the lifetime answer to the model",
            enforce_failure_policy=False,
        )
    return reply


def _serve_worked_out_sequence(user_message: object, reply: object) -> object:
    """Answer a "what does this become" question by working the rule out.

    The induction machinery had no consumer outside its own battery. It could
    learn a transformation from a few examples, keep it, compose with it and
    carry it to the next problem, and none of that ever met a person: the
    architecture had the mechanism and the live agent did not use it. This is
    the seam.

    Where the rule accounts for every example shown, there is nothing to
    generate — the same as a seating arrangement or a product of two numbers.
    The shape is kept afterwards, so the next question of the kind is settled
    from fewer examples, which is the point of the library being in the live
    path rather than beside it.

    Quiet on single values: "45 becomes 15" is a relation between numbers, not
    a rearrangement of positions, and answering it would mean guessing.
    """
    try:
        from core.cognition.sequence_induction import answer_sequence_question

        worked_out = answer_sequence_question(str(user_message or ""))
        if not worked_out:
            return reply
        logger.info("🔁 Served a sequence question from the rule it worked out.")
        return worked_out
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.worked_out_sequence",
            exc,
            severity="debug",
            action="left the sequence question to the model",
            enforce_failure_policy=False,
        )
        return reply


def _serve_positional_solution(user_message: object, reply: object) -> object:
    """Answer a seating or order problem from the enumeration.

    LIVE, 2026-08-20. Six people, four constraints, one arrangement. She
    narrated it twice and was wrong twice, the second time stating a layout in
    which Dara sat opposite Ada one line after Boris did. The tools reached
    the turn — a Python sandbox among them — and the model answered directly
    anyway.

    Where the answer follows from the constraints there is nothing to
    generate, the same as a column of a spreadsheet or a product of two
    numbers. Reported only when every arrangement that satisfies the
    constraints agrees, so a problem that is genuinely open stays with the
    model.
    """
    try:
        from core.reasoning.positional_constraints import (
            answer_positional_problem,
            describe_positional_answer,
        )

        described = describe_positional_answer(
            answer_positional_problem(str(user_message or ""))
        )
        if not described:
            return reply
        logger.info("🪑 Served a seating problem from the enumeration.")
        return described
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.positional_solution",
            exc,
            severity="debug",
            action="left the seating problem to the model",
            enforce_failure_policy=False,
        )
        return reply


def _serve_recent_activity(user_message: object, reply: object) -> object:
    """Answer "what have you been working on" from the record of doing it.

    LIVE, 2026-08-20. The reading was taken — the log records "took 2
    reading(s): work you have actually done" — and the answer was "I've been
    analyzing my cognitive architecture, looking at how information flows
    between different systems." Thirty-three finished pieces of work sat in
    the block in front of her, with the tools she ran and the topics she
    chased, and one of them reached the reply.

    Same treatment as the queue, for the same reason: evidence informs, it
    does not enforce, and what she did is not a matter of opinion. The reply
    she wrote is kept after the record, because why one piece of work was
    interesting IS hers to say — only the list of what happened is not.
    """
    try:
        from core.self.recent_activity import (
            looks_like_a_question_about_recent_activity,
            narrate_recent_activity,
            read_recent_activity,
        )

        if not looks_like_a_question_about_recent_activity(str(user_message or "")):
            return reply
        # Said, not displayed. The block form belongs in a prompt, where
        # headings separate evidence from everything around it; served to
        # somebody who asked what she had been up to tonight it reads as a
        # status page — the right facts in the wrong voice.
        # Bound it to the stretch the person asked about.
        #
        # LIVE, 2026-08-27: "of everything I've thrown at you in the last hour
        # or so, what did you actually do well?" came back with several days of
        # work — 2048, a sliding puzzle, notes written to a Desktop — because
        # the record was read by COUNT and the window in the sentence was never
        # read at all.
        from core.language.stated_window import seconds_named

        described = narrate_recent_activity(
            read_recent_activity(since_seconds=seconds_named(user_message))
        )
        if not described:
            return reply
        logger.info("🗂️ Served the activity record from the intention log.")
        return _compose(
            user_message, reply, described, looks_like_a_question_about_recent_activity
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.recent_activity",
            exc,
            severity="debug",
            action="left the activity answer to the model",
            enforce_failure_policy=False,
        )
        return reply


def _serve_host_load(user_message: object, reply: object) -> object:
    """Answer "how hard is the machine working" with the reading.

    LIVE, 2026-08-25: "How hard is the machine you run on working right now?
    Give me a number you can stand behind." came back as "I have 19 stored
    turns of recent conversation I can read back. So I can't give you a
    defensible number" — while the telemetry said 24% processor and 57%
    memory.

    Fixing the reading and the gate got the number as far as the giving-up
    path, where it arrived as a five-line status page in place of her reply:
    the right fact, none of it in her voice, and three lines nobody asked for.
    A reading belongs in the answer, and this one covers load and nothing
    else — whether a job is failing is a different question.
    """
    try:
        from core.introspection.self_evidence import (
            asks_about_own_operational_state,
            resolve_self_health,
        )

        asked = str(user_message or "")
        if not asks_about_own_operational_state(asked):
            return reply
        readings = {
            reading.channel: reading for reading in resolve_self_health().readings
        }
        load = readings.get("host_load")
        if load is None or not load.present:
            return reply
        values = dict(load.value or {})
        said = (
            f"The machine is at {values.get('processor_percent', 0.0):.1f}% processor "
            f"and {values.get('memory_percent', 0.0):.1f}% memory right now."
        )
        thermal = readings.get("host_thermal")
        if thermal is not None and thermal.present and float(thermal.value) > 0.0:
            said += f" Thermal pressure {float(thermal.value):.2f} of 1."
        logger.info("🌡️ Served the host load from the telemetry.")
        return _compose(user_message, reply, said, asks_about_own_operational_state)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.host_load",
            exc,
            severity="debug",
            action="left the load answer to the model",
            enforce_failure_policy=False,
        )
    return reply


def _serve_queued_work(user_message: object, reply: object) -> object:
    """Answer "what are you going to do next" from the coordinator's list.

    LIVE, 2026-08-19. The reading was taken and reached dispatch — the log
    records "took 1 reading(s): work you have queued" — and the answer was
    "After this, I'm going to keep running. There's no stopping point." Two
    jobs were waiting at that moment: dlq_recovery, held by an active
    foreground generation, and biological_sleep, held for want of a user
    anchor. Neither was mentioned.

    Third channel to get this treatment, for the same reason as the first two:
    evidence informs, it does not enforce, and a pending list is not a matter
    of opinion.
    """
    try:
        from core.brain.observable_registry import _matches_queued_work
        from core.maintenance.dream_coordinator import get_dream_coordinator

        if not _matches_queued_work(str(user_message or "")):
            return reply
        pending = dict((get_dream_coordinator().status() or {}).get("pending") or {})
        if not pending:
            return reply
        lines = []
        for name, detail in list(pending.items())[:8]:
            label = str(name).replace("_", " ").strip()
            reason = str(dict(detail or {}).get("reason") or "").replace("_", " ").strip()
            lines.append(f"- {label}" + (f" — waiting on {reason}" if reason else ""))
        count = len(pending)
        head = (
            f"{count} job{'s' if count != 1 else ''} waiting to run, "
            "and nothing else queued:"
        )
        logger.info("🗓️ Served the queued-work list from the coordinator.")
        return _compose(
            user_message, reply, head + "\n" + "\n".join(lines), _matches_queued_work
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.queued_work",
            exc,
            severity="debug",
            action="left the queued-work answer to the model",
            enforce_failure_policy=False,
        )
    return reply


def _serve_earlier_conversation(user_message: object, reply: object) -> object:
    """Answer "what did I ask you earlier" from the record, not from memory.

    Given the durable turns in its prompt the model answered with the
    immediately previous message; before the turns were available at all it
    invented topics. Neither is what was asked, and the record holds the
    answer exactly, with times.

    Only for a question that says out loud it reaches past this session, and
    only when the store actually holds earlier turns.
    """
    try:
        from core.brain.observable_registry import _reaches_past_this_session
        from core.conversation.durable_turns import earlier_conversation_answer

        question = str(user_message or "")
        if not _reaches_past_this_session(question):
            return reply
        if not re.search(
            r"\bwhat\s+(?:did|was|were|have)\b|\bremember\b|\brecall\b|\btalk(?:ed|ing)?\s+about\b",
            question,
            re.IGNORECASE,
        ):
            return reply
        composed = earlier_conversation_answer(exclude=question)
        if composed:
            logger.info("🗒️ Served the earlier conversation from the durable record.")
            return composed
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.earlier_conversation",
            exc,
            severity="debug",
            action="left the recall answer to the model",
            enforce_failure_policy=False,
        )
    return reply


def _serve_measured_belief_history(reply: object) -> object:
    """Replace an invented revision with what her snapshots actually hold.

    Asked to name a position she had held and dropped, WITH a date, and given
    the explicit out "if you can't, say so plainly", she named one that does
    not exist and dated it "around the middle of last year". The belief-history
    reading was in the prompt for that very turn — the log records it taken and
    surviving to dispatch — and the reply contradicted it anyway.

    A revision she cannot evidence is not a matter of opinion, so the reading
    is served instead of requested, the same treatment file counts get and for
    the same reason.
    """
    try:
        from core.self.belief_history import unevidenced_revision_claim

        measured = unevidenced_revision_claim(reply)
        if measured:
            logger.warning(
                "🧠 Replaced an unevidenced revision claim with the snapshot reading."
            )
            return measured
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.belief_history",
            exc,
            severity="debug",
            action="left the reply's revision claim unchecked",
            enforce_failure_policy=False,
        )
    return reply


def _serve_measured_filesystem_count(user_message: object, reply: object) -> object:
    """Replace a contradicted file count with the one the runtime took.

    The re-answer pass injects the real count into the context and asks the
    model to answer again from it. LIVE 2026-08-17 it did that, logged that it
    did it, and the model returned "There are 3 Python files" for the third
    time. The codebase already knew this would happen — response_generation
    says it outright: "evidence informs, it does not enforce."

    So the fact is served rather than requested. A count is not a matter of
    opinion, the runtime holds it exactly, and at that point the model can only
    add error. This composes the sentence from the reading, the way
    _desktop_effect_summary composes an action summary from receipts instead of
    letting the model narrate what it did.
    """

    text = str(reply or "")
    try:
        from core.conversation.filesystem_check import (
            contradicted_filesystem_claims,
            requested_filesystem_counts,
        )

        counts = requested_filesystem_counts(user_message)
        contradicted = contradicted_filesystem_claims(user_message)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.filesystem_check", exc)
        return reply

    # A number the person STATED, which the runtime can settle exactly.
    #
    # Live 2026-08-18: "earlier you told me core/agency has 61 python files.
    # just confirming those before i write them down." She replied "Yes, that's
    # right ... exactly 61 Python files ... Feel free to write those down —
    # they're factual observations you can trust." There are 54, and she had
    # answered 54 correctly earlier in the same conversation.
    #
    # Every count path here fired on a QUESTION. An assertion is the same claim
    # with the same answer available, and it is the more dangerous shape: a
    # question invites a check, a statement invites a nod. Agreeing with a
    # number the runtime holds is what makes every other number she gives
    # worthless.
    if contradicted:
        corrections = "; ".join(
            f"{Path(counted.path).name} has {counted.count}"
            f"{' ' + counted.suffix if counted.suffix else ''} files, not {claimed}"
            for claimed, counted in contradicted
        )
        logger.warning("📁 Corrected a stated count: %s.", corrections)
        correction = (
            f"Not quite — {corrections}. I counted the directory just now rather "
            "than agreeing."
        )
        if not counts:
            # If the draft repeats the wrong figure it cannot be kept beside
            # the correction: the person would be handed both numbers and no
            # way to tell which she meant. A reply whose content is the false
            # confirmation IS the defect, so it goes.
            repeats_the_claim = any(
                str(claimed) in text for claimed, _counted in contradicted
            )
            if repeats_the_claim or not text:
                return correction
            return f"{correction}\n\n{text}".strip()

    if not counts:
        return reply
    # Every count that was asked for. Serving only the first answered half of
    # "how many test files do you have, and how many python files are in
    # core/agency?" with "54 .py files" — the second number, exactly right, and
    # the first silently dropped. Half an answer reads as a whole one, which is
    # worse than saying a part is unavailable.
    # A missing directory always has to be said; the reply cannot already
    # contain a count it does not have. Testing only the EXISTING ones made
    # this an `all()` over an empty sequence, which is True — so a single
    # missing directory short-circuited to "leave her wording alone" and the
    # "no directory" report was never reached.
    present = [counted for counted in counts if counted.exists]
    if len(present) == len(counts) and all(str(c.count) in text for c in present):
        return reply  # she already has them right; leave her wording alone

    sentences: list[str] = []
    for counted in counts:
        if not counted.exists:
            sentences.append(
                f"There is no directory at {counted.path}, so there is nothing "
                "to count there."
            )
            continue
        kind = f"{counted.suffix} " if counted.suffix else ""
        where = Path(counted.path).name
        # "just the numbers" is an instruction about the answer's shape, and
        # listing twelve filenames under it answers a question that was not
        # asked. Measured live 2026-08-18: asked for two counts and "just the
        # numbers", the reply opened with a dozen test filenames.
        if len(counts) == 1 and not _brevity_requested(user_message):
            listed = ", ".join(counted.names[:12])
            more = (
                "" if len(counted.names) <= 12 else f", and {len(counted.names) - 12} more"
            )
            sentences.append(
                f"{counted.count} {kind}files. I listed the directory rather than "
                f"estimating: {listed}{more}."
            )
        else:
            # With more than one, naming the place matters more than listing
            # every file — a wall of names buries the second number.
            sentences.append(f"{where}: {counted.count} {kind}files, counted from disk.")
    logger.warning(
        "📁 Served %d measured count(s) over the generated one(s): %s.",
        len(counts),
        ", ".join(f"{c.path}={c.count}" for c in counts),
    )
    return " ".join(sentences)
