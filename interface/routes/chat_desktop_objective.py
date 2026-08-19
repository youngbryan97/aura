"""Running a desktop objective, and deciding whether to.

Executing something on the machine is consequential, so a turn clears two
separate questions before anything happens: is this actually an execution
request, and does it carry enough on its own to run without the cognitive
lane filling in the blanks.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from interface.routes.chat_common import (  # noqa: E402
    _CHAT_BLOCKING_PREFLIGHT_TIMEOUT_S,  # noqa: F401
    _CHAT_RECOVERABLE_ERRORS,  # noqa: F401
    _CHAT_REQUEST_PRINCIPAL,  # noqa: F401
    _CHAT_REQUEST_SURFACE,  # noqa: F401
    _MAX_CONVERSATION_LOG_EXCHANGES,  # noqa: F401
    _conversation_log,  # noqa: F401
    _locks,  # noqa: F401
    logger,  # noqa: F401
)
from interface.routes.chat_common import (  # noqa: E402
    _EXPLICIT_NON_EXECUTION_RE,  # noqa: F401
    _INCOMPLETE_TAIL_WORDS,  # noqa: F401
    _INTERNAL_STATE_PATTERNS,  # noqa: F401
    _LOCAL_CHOICE_REFERENCE_RE,  # noqa: F401
    _ORGAN_INERT_STREAKS,  # noqa: F401
)
from interface.routes import chat_capability_inventory as _chat_capability_inventory
from interface.routes import chat_desktop_repair as _chat_desktop_repair
from interface.routes import chat_preflight as _chat_preflight
import re
from core.runtime.errors import describe_error, record_degradation


def _blocks_consequential_desktop_execution(user_message: str) -> bool:
    """True when the user asked for planning/explanation, not live desktop effects."""
    text = str(user_message or "").strip()
    if not text:
        return False
    return bool(
        _EXPLICIT_NON_EXECUTION_RE.search(text)
        or _chat_desktop_repair._is_bounded_nonexecuting_planning_request(text)
    )


def _verified_desktop_task_result(result: dict[str, Any]) -> tuple[bool, str]:
    """Require step-level effect proof before a desktop result can be claimed.

    A chat bridge must not accept a bare ``ok=True`` from any executor. The
    desktop task may involve Notes, Docs, browser tabs, files, PDFs, settings,
    or future OS actions, but the invariant is the same: every requested step
    needs a verified receipt with observable effect evidence.
    """
    if not bool(result.get("ok")):
        return False, "desktop_task_result_not_ok"

    requested = result.get("steps_requested")
    completed = result.get("steps_completed")
    if not isinstance(requested, int) or requested <= 0:
        return False, "missing_positive_steps_requested"
    if not isinstance(completed, int):
        return False, "missing_steps_completed"
    # A task that completed nothing has not been done, whatever else the
    # result says. LIVE 2026-08-18: "make a file on my desktop called
    # aura-live-check.txt" was answered "Done — the desktop steps completed
    # and their effects verified" in 2ms, and no file existed. The count was
    # read only to check it was an integer.
    # A non-critical step is allowed to fail, so this is not "completed ==
    # requested" — the loop below already refuses any CRITICAL step that is
    # not ok and verified. What was missing is the floor: a task that
    # completed nothing has not been done, whatever else the result says.
    if completed <= 0:
        return False, f"no_steps_completed:0/{requested}"

    receipts = result.get("receipts")
    if not isinstance(receipts, list) or len(receipts) < requested:
        return False, "missing_step_receipts"

    for index, receipt in enumerate(receipts[:requested], start=1):
        if not isinstance(receipt, dict):
            return False, f"step_{index}_receipt_not_structured"
        if not bool(receipt.get("ok")):
            if bool(receipt.get("critical", True)):
                return False, f"step_{index}_not_ok"
            continue
        if receipt.get("effect_verified") is not True:
            return False, f"step_{index}_effect_unverified"
        evidence = str(receipt.get("effect_evidence") or "").strip()
        if not evidence:
            return False, f"step_{index}_missing_effect_evidence"
        if evidence.startswith("receipt_id="):
            return False, f"step_{index}_audit_receipt_without_effect"
        missing = _effect_claim_contradicted_by_disk(evidence)
        if missing:
            return False, f"step_{index}_{missing}"
    return True, "verified"


def _effect_claim_contradicted_by_disk(evidence: str) -> str:
    """Check a receipt's own claim against the filesystem, or "" if it holds.

    A file receipt states "path=X;bytes=N", which is a claim anyone can check
    — so this bridge checks it rather than taking the executor's word, on the
    same reasoning that already stops it accepting a bare ok=True. The cost is
    one stat() and it is the difference between "it is done" and "the executor
    believes it is done".
    """
    fields = dict(
        part.split("=", 1)
        for part in str(evidence or "").split(";")
        if "=" in part
    )
    claimed_path = fields.get("path", "").strip()
    if not claimed_path:
        return ""
    try:
        target = Path(claimed_path).expanduser()
        if not target.is_file():
            return "claimed_file_is_not_on_disk"
        claimed_bytes = fields.get("bytes", "").strip()
        if claimed_bytes.isdigit() and target.stat().st_size != int(claimed_bytes):
            return "claimed_file_size_does_not_match"
    except (OSError, ValueError):
        # An unreadable path is not proof of absence; the receipt stands.
        return ""
    return ""


def _desktop_task_action_expectation(objective: str) -> dict[str, Any]:
    return {
        "objective": str(objective or "")[:500],
        "acceptance_criteria": ["steps_requested", "steps_completed"],
        "required_evidence": ["receipts"],
        "repair_hint": "rerun_desktop_task_with_effect_receipts",
        "allow_partial": True,
    }


async def _execute_desktop_objective_from_chat(
    user_message: str,
    *,
    cognitive_reply: str,
) -> dict[str, Any] | None:
    """Execute a desktop objective through the generic desktop_task skill.

    This is the live desktop counterpart to proof runners: the UI request is
    first answered/planned by CognitiveEngine, then the actual consequential
    work is performed through Authority/Capability/desktop_task/computer_use.
    """
    if _blocks_consequential_desktop_execution(user_message):
        logger.info(
            "Desktop objective execution blocked by explicit non-execution/planning-only request."
        )
        return None
    if not _chat_preflight._looks_like_desktop_objective(user_message):
        return None

    objective = str(user_message or "").strip()
    # The visible desktop lane is already consuming the foreground Cortex turn.
    # A second hidden model synthesis inside desktop_task can starve the
    # generation gate and prevent any governed receipts from being emitted.
    # Research documents therefore use source-grounded synthesis by default;
    # explicit callers can still opt into model synthesis by invoking
    # desktop_task directly with allow_desktop_task_model_synthesis=True.
    allow_research_synthesis = False
    action_expectation = _desktop_task_action_expectation(objective)
    desktop_params = {
        "objective": objective,
        "steps": [],
        "desktop_execution_contract": True,
        "allow_heuristic_desktop_plan": True,
        "disable_outer_skill_retry": True,
        "foreground_request": True,
        "user_requested_action": True,
        "user_explicitly_authorized": True,
        "user_visible_desktop_action": True,
        "local_desktop_action": True,
        "verification_required": True,
        "predicted_outcome": "The requested visible desktop/file effect is verified after execution.",
        "action_expectation": action_expectation,
    }
    result = await _chat_capability_inventory._execute_governed_live_skill(
        "desktop_task",
        desktop_params,
        objective=objective,
        extra_context={
            "origin": "desktop_ui",
            "source": "desktop_ui",
            "route": "chat.desktop_objective",
            "desktop_execution_contract": True,
            "allow_heuristic_desktop_plan": True,
            "disable_outer_skill_retry": True,
            "user_visible_desktop_action": True,
            "local_desktop_action": True,
            "verification_required": True,
            "allow_desktop_task_model_synthesis": allow_research_synthesis,
            "desktop_task_document_body": str(cognitive_reply or "").strip(),
            "cognitive_reply": str(cognitive_reply or "").strip(),
            "action_expectation": action_expectation,
        },
    )
    if not isinstance(result, dict):
        return {"ok": bool(result), "result": result, "status": "desktop_objective_unknown"}

    if result.get("ok"):
        verified, verification_reason = _verified_desktop_task_result(result)
        if not verified:
            result = dict(result)
            result["ok"] = False
            result["status"] = "desktop_task_effect_evidence_missing"
            result["error"] = verification_reason

    governed_status = str(result.get("status") or "").strip()
    if governed_status in {"approval_required", "require_fresh_user_auth"}:
        approval = result.get("approval") if isinstance(result.get("approval"), dict) else {}
        return {
            "ok": False,
            "status": "approval_required",
            "response": (
                "This desktop action needs a fresh confirmation. Confirm it to retry "
                "the same request; all standing authority and governance checks still apply."
            ),
            "approval": approval,
            "result": result,
        }

    status = "desktop_objective_completed" if result.get("ok") else "desktop_objective_failed"
    completed = int(result.get("steps_completed") or 0)
    requested = int(result.get("steps_requested") or 0)
    summary = str(result.get("summary") or "").strip()
    observation = _desktop_task_observation(result)
    research_response = _desktop_task_research_response(
        result,
        completed=completed,
        requested=requested,
    )
    if result.get("ok") and research_response:
        response = research_response
    elif result.get("ok"):
        # Lead with WHAT SHE SAW, not with the step count.
        #
        # Live 2026-07-27, "read my screen and tell me what you actually see"
        # succeeded — 1/1 governed steps — and the person was handed
        # "Desktop task completed 1/1 governed computer-use steps through
        # heuristic_compat planning." The observation was in the receipt the
        # whole time. A step count is what the machine did; the answer is
        # what it found, and the question was the second one.
        if observation and _perception_needs_her_own_answer(result, objective):
            # A SPECIFIC question about the screen is not answered by a
            # description of the screen.
            #
            # Live 2026-08-04: "what was that repo you saw on my screen?"
            # came back as "Aura is in front. Behind it, partly visible:
            # Claude and Google Chrome…" — a correct description, and not
            # the answer. The repo name was sitting inside the evidence.
            #
            # The read has happened and is retained, so returning no reply
            # here is not a loss: the turn continues into cognition, which
            # receives the perception and answers the question that was
            # actually asked. Only a plain "what's on my screen" is served
            # by the description, and that one is served natively.
            response = ""
        elif observation:
            # A perception answers with what was SEEN and stops there. The
            # step count is bookkeeping about the machinery; appending it to
            # "Chrome is in front, showing …" turns an answer back into a
            # progress report, and nobody asked how many steps it took to
            # look at their own screen. Non-perception observations keep it.
            response = (
                observation
                if result.get("observation_meta")
                else f"{observation} (Completed {completed}/{requested} governed desktop steps.)"
            )
        else:
            # What was PRODUCED, not how many steps produced it.
            #
            # Live 2026-07-30 demo: "open the Notes app and write a note where
            # you write a paragraph describing yourself" was answered with
            # "Desktop task completed 2/2 governed computer-use steps through
            # heuristic_compat planning. Completed 2/2 governed desktop steps."
            # The note WAS written. Nothing in the reply showed it, so Bryan
            # asked again — the task had succeeded and the answer made it look
            # like it had not.
            #
            # The same lesson is already written three branches up for
            # perceptions: nobody asked how many steps it took to look at
            # their own screen. Nobody asked how many steps it took to write a
            # paragraph either. They asked for the paragraph.
            produced = _desktop_deliverable_text(result)
            step_note = f"Completed {completed}/{requested} governed desktop steps."
            # Whether the effects were actually proven, so a completed action is
            # never dropped for lack of a quotable deliverable.
            verified_effects, _verification_reason = _verified_desktop_task_result(result)
            if produced:
                # A summary that only counts steps is machinery, and prepending
                # it to a real deliverable puts internal vocabulary in front of
                # the answer: "Desktop task completed 2/2 governed computer-use
                # steps through heuristic_compat planning. Here is what I
                # wrote: ..." — the planner's identifier, in a sentence to a
                # person. The deliverable is the answer; when the summary says
                # nothing about the world, it leads with nothing.
                if _is_step_bookkeeping_only(summary):
                    response = produced
                else:
                    response = f"{summary or 'Done.'} Here is what I wrote:\n\n{produced}"
            elif _is_step_bookkeeping_only(summary):
                # Live 2026-08-10: "can you read text that is only pixels?
                # answer yes or no, then tell me how you know" was answered
                # with "Desktop task completed 1/1 governed computer-use steps
                # through heuristic_compat planning. Completed 1/1 governed
                # desktop steps." — the step count TWICE, in the branch whose
                # own comment says to report what was produced instead of it.
                #
                # The bookkeeping is still not an answer. But deferring by
                # returning "" was too blunt and regressed a real case the same
                # day: a desktop task that COMPLETED and verified its effects,
                # with no text deliverable to quote ("open Notes and write a
                # note saying Hello"), produced an empty reply — and an empty
                # reply is falsy at the caller, so a successful, receipt-verified
                # action fell through to cognition as though nothing had
                # happened. That both wasted a foreground model pass and put the
                # turn back into the lane whose failures this branch exists to
                # avoid.
                #
                # So: defer only when there is genuinely nothing to report.
                # When the effects were verified, say plainly that the thing was
                # done — once, in her own voice, without the step count.
                if verified_effects:
                    # Lead with the plain fact, and keep the executor's summary
                    # behind it as the evidence that it happened. When there is
                    # no text deliverable to quote, "it is done" IS the answer,
                    # and the receipt line is what makes it checkable rather
                    # than a claim.
                    # LIVE, 2026-08-10: "Done — Desktop task completed 1/1
                    # governed computer-use steps through heuristic_compat
                    # planning." The receipt line was meant to make the claim
                    # checkable, and a step count with a planner identifier in
                    # it checks nothing a person can use — it just puts the
                    # engineering log in the sentence. What makes "it is done"
                    # checkable is WHAT was done: the file that now exists.
                    effect_line = _desktop_effect_summary(result)
                    # An ANSWER is not an effect receipt.
                    #
                    # LIVE, 2026-08-10: "Look at what's on my screen right now
                    # and tell me what the paper is about. What is the actual
                    # mechanism they use?" — a question about a bioRxiv preprint
                    # — was answered "Done — the desktop steps completed and
                    # their effects verified."
                    #
                    # The lane took a reading, verified that the reading
                    # happened, and reported the verification. "Did you do it"
                    # and "what is it" are different questions, and a receipt
                    # only answers the first. Deferring here sends the turn to
                    # cognition, which has the reading and can answer from it.
                    if _asks_for_information(user_message):
                        response = ""
                    else:
                        response = (
                            f"Done — {effect_line}"
                            if effect_line
                            else "Done — the desktop steps completed and their effects verified."
                        )
                else:
                    response = ""
            else:
                response = (
                    f"{summary or 'I completed the requested desktop task through governed desktop control.'} "
                    f"{step_note}"
                )
    else:
        error = str(result.get("error") or result.get("status") or "desktop task failed").strip()
        response = (
            "I routed this through CognitiveEngine and the governed desktop task lane, "
            f"but it did not complete: {error}. Completed {completed}/{requested} steps. "
            "I am not claiming the desktop action finished."
        )
        # A partial task can still hold the answer, and withholding it is its
        # own failure.
        #
        # LIVE, 2026-08-10: "count how many .py files are in <dir> ... Tell me
        # the number" completed 2/2 steps, read 9, wrote 9 to the right file,
        # and replied only "semantic completion incomplete:
        # requested_source_count_found" — a checker correctly reporting that
        # the number was missing from the reply, while the number sat in a
        # verified receipt one function away. The person was told the task
        # failed and never told the answer it had found.
        partial = _desktop_deliverable_text(result)
        if partial:
            response = (
                f"{partial}\n\n"
                f"That much is verified. The rest did not complete: {error} "
                f"({completed}/{requested} steps)."
            )
    # Every effect the reply claims, checked against the receipts of this turn.
    #
    # The file check verifies writes against the filesystem. "I opened Notes",
    # "I created that folder", "I moved it to your Desktop" and "I put it on
    # the clipboard" are claims about the world too, and nothing verified any
    # of them — the honesty guarantee stopped exactly where the file API
    # stopped. The receipts already record every governed effect, so the
    # general form is to name the effect a reply claims and ask whether one
    # like it verified.
    try:
        from core.conversation.claimed_effect import unverified_effect_correction

        effect_correction = str(
            unverified_effect_correction(response, result.get("receipts")) or ""
        ).strip()
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        effect_correction = ""
    if effect_correction:
        response = f"{str(response or '').rstrip()}\n\n{effect_correction}"
    return {
        "ok": bool(result.get("ok")),
        "status": status,
        "response": response,
        "result": result,
    }


_STEP_BOOKKEEPING_RE = re.compile(
    r"^\s*(?:desktop task\s+)?(?:completed|finished|executed)\s+\d+\s*/\s*\d+\s+"
    r"governed[\w\s-]*steps?\b[^.]*\.?\s*$",
    re.IGNORECASE,
)


def _is_step_bookkeeping_only(summary: str) -> bool:
    """True when a desktop summary reports step counts and nothing else.

    The desktop lane's own summary string is frequently a step count. Pasting
    it in front of the step sentence produced replies that stated the count
    twice and answered nothing — see the call site.
    """
    text = str(summary or "").strip()
    if not text:
        return True
    return bool(_STEP_BOOKKEEPING_RE.match(text))


def _desktop_task_research_response(
    result: dict[str, Any],
    *,
    completed: int,
    requested: int,
) -> str:
    research = result.get("research")
    if not isinstance(research, dict) or research.get("error"):
        return ""
    sources = [s for s in (research.get("sources") or []) if isinstance(s, dict)]
    synthesis = str(research.get("synthesis") or research.get("summary") or "").strip()
    if not synthesis and not sources:
        return ""
    query = str(research.get("query") or "the requested topic").strip()
    # Shape the answer before deciding which citations it still carries.
    # A synthesis may cite all sources after the clipping boundary; checking
    # the unshaped text would then suppress the only citations the user sees.
    body = _clip_reply_to_sentence(synthesis, 1200)
    source_bits: list[str] = []
    for source in sources[:3]:
        title = str(source.get("title") or source.get("url") or "").strip()
        url = str(source.get("url") or "").strip()
        if title and url and title != url:
            source_bits.append(f"{title} ({url})")
        elif title or url:
            source_bits.append(title or url)
    # The synthesis usually ends with its own "Sources opened or consulted"
    # list, and this function appended a second one regardless. Live
    # 2026-07-30 00:33 Bryan's reply carried all three sources twice, the
    # second copy introduced by a sentence fragment. Only add the sentence when
    # the sources are not already in the text — checked against the URLs
    # themselves rather than a heading, so a reworded heading cannot
    # reintroduce the duplicate.
    missing_source_bits = []
    for bit in source_bits:
        url = bit.rpartition("(")[2].rstrip(")")
        if not url:
            url = bit
        if url not in body:
            missing_source_bits.append(bit)
    if missing_source_bits:
        source_sentence = " Sources: " + "; ".join(missing_source_bits) + "."
    elif source_bits:
        source_sentence = ""
    else:
        source_sentence = " No source URL was available in the receipt."
    step_sentence = f" Completed {completed}/{requested} governed desktop steps."
    # Clipped at a sentence, not a character count. The 1200-char cut landed
    # mid-clause — "where they differ I should" — and then ran a "Sources:"
    # fragment onto the stump.
    return (
        f"I completed the research-backed desktop task for {query}. "
        f"{body}"
        f"{source_sentence}"
        f"{step_sentence}"
    )


_DESKTOP_DELIVERABLE_MAX_CHARS = 1200

_ASKS_FOR_INFORMATION_RE = re.compile(
    r"\b(?:what|who|when|where|why|which|how)\b"
    r"|\b(?:tell|show|explain|describe|summari[sz]e|read)\s+(?:me|it|this|that|the)\b"
    r"|\bwhat(?:'s| is)\s+(?:it|this|that|on)\b"
    r"|\byour\s+(?:opinion|take|thoughts?|view)\b"
    r"|\bdo\s+you\s+(?:think|agree|reckon)\b",
    re.IGNORECASE,
)

_ASKS_FOR_INFORMATION_EXCLUSION_RE = re.compile(
    r"\b(?:put|copy|paste|save|write|create|make|move|rename|delete|open|quit|"
    r"close|set|install)\b",
    re.IGNORECASE,
)


def _asks_for_information(user_message: object) -> bool:
    """True when the person asked a question the reply has to answer."""

    text = str(user_message or "").strip()
    if not text or not _ASKS_FOR_INFORMATION_RE.search(text):
        return False
    return not _ASKS_FOR_INFORMATION_EXCLUSION_RE.search(text)


def _desktop_effect_summary(result: Any) -> str:
    """What a verified task actually changed, in words a person can check.

    Reads the receipts for effects with a name — a path written, a folder
    created, an app opened — rather than repeating how many steps ran.
    """

    if not isinstance(result, dict):
        return ""
    # Built from typed effect claims, not assembled by hand here.
    #
    # The hand-assembled version knew four actions and read receipt fields
    # directly, so it could be made to overstate by any receipt shape it did
    # not anticipate — and it was a fifth place that had to be taught what an
    # effect is. render_effect_claims cannot overstate: a completed claim
    # without a receipt raises at construction, so the sentence has no way to
    # contain one.
    try:
        from core.conversation.effect_claim import render_effect_claims

        return render_effect_claims(result.get("receipts"))
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        return ""


def _pursuit_account(result: dict) -> list[str]:
    """What she did on the page, in her words, and what the page says now.

    The narration is per-round: the item she was reading, what she picked, and
    why she picked it. That is the deliverable of working a page, in the same
    way a written paragraph is the deliverable of writing one.
    """

    lines: list[str] = []
    narration = result.get("narration")
    if isinstance(narration, list):
        for entry in narration:
            if not isinstance(entry, dict):
                continue
            chose = [str(choice) for choice in (entry.get("chose") or []) if choice]
            if not chose:
                continue
            asked = str(entry.get("asked") or "").strip()
            why = str(entry.get("why") or "").strip()
            line = f"{asked} — {', '.join(chose)}" if asked else ", ".join(chose)
            if why:
                line += f" ({why})"
            lines.append(line)
    ending = str(result.get("result_text") or "").strip()
    if ending:
        # The tail, not the head: a result page repeats its navigation before
        # it says anything, and what a page concludes with is what it is for.
        lines.append("The page ends with:\n" + ending[-600:].strip())
    return lines


def _desktop_deliverable_text(result: Any) -> str:
    """The text a desktop task actually wrote, if it wrote any.

    A task that produced content has a deliverable, and the deliverable is
    the answer. Step counts describe the machinery that produced it, which is
    the thing the person is least interested in once it worked.

    Reads the per-step receipts rather than the summary, because the summary
    is written by the planner and describes its own execution.
    """
    if not isinstance(result, dict):
        return ""
    receipts = result.get("receipts")
    if not isinstance(receipts, list):
        return ""
    written: list[str] = []
    for receipt in receipts:
        if not isinstance(receipt, dict):
            continue
        # A read produces a FINDING, and the finding is the answer. "Count how
        # many .py files are in <dir>, then write that ... Tell me the number"
        # completed 2/2 steps with the right count in the right file, and the
        # reply said only that it had run — so the semantic verifier correctly
        # reported "requested_source_count_found" incomplete. The number was in
        # the receipt the whole time.
        if str(receipt.get("action") or "").strip() == "list_directory":
            if not receipt.get("ok"):
                continue
            inner = receipt.get("result")
            payload = inner if isinstance(inner, dict) else receipt
            count = payload.get("count")
            names = payload.get("names")
            if type(count) is not int:
                continue
            line = f"{count} file(s) matching {payload.get('pattern') or '*'}"
            path = str(payload.get("path") or "").strip()
            if path:
                line += f" in {path}"
            if isinstance(names, list) and names:
                line += ":\n" + "\n".join(f"  {name}" for name in names)
            written.append(line)
            continue
        if str(receipt.get("action") or "").strip() != "type":
            continue
        if not receipt.get("ok"):
            # A step that did not verify did not write anything, and quoting
            # its intended text as "what I wrote" would be the false-success
            # claim wearing a friendlier face.
            continue
        # The executor payload is nested under "result"; the top level is the
        # step record. Check both so a shape change does not silently return
        # nothing and fall back to the step count.
        inner = receipt.get("result")
        typed = None
        if isinstance(inner, dict):
            typed = inner.get("typed")
        if not isinstance(typed, str):
            typed = receipt.get("typed")
        if not isinstance(typed, str):
            continue
        text = typed.strip()
        # Keystroke-level fragments are not a deliverable; a paragraph is.
        if len(text) >= 40:
            written.append(text)
    if not written and any(
        isinstance(receipt, dict)
        and str(receipt.get("action") or "").strip() == "browse_pursue"
        for receipt in receipts
    ):
        # Working a page produces an account, not a file.
        #
        # This function knew how to quote a paragraph she typed and how to
        # report a directory listing, and nothing about a pursuit — so a run
        # that answered most of a sixty-item questionnaire had no deliverable
        # and the reply fell back to counting steps. What she chose, and why,
        # is what happened; the page's own words are what it said back.
        written.extend(_pursuit_account(result))
    if not written:
        return ""
    body = "\n\n".join(written).strip()
    if len(body) > _DESKTOP_DELIVERABLE_MAX_CHARS:
        body = _clip_reply_to_sentence(body, _DESKTOP_DELIVERABLE_MAX_CHARS)
    return body


def _clip_reply_to_sentence(text: str, limit: int) -> str:
    """Clip to a sentence boundary, falling back to a word boundary."""
    try:
        from core.skills.desktop_task import DesktopTaskSkill

        return DesktopTaskSkill._clip_to_sentence(text, limit)
    except _CHAT_RECOVERABLE_ERRORS:
        body = " ".join(str(text or "").split())
        if len(body) <= limit:
            return body
        window = body[:limit]
        cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
        if cut >= limit // 2:
            return window[: cut + 1].strip()
        cut = window.rfind(" ")
        return (window[:cut] if cut >= limit // 2 else window).rstrip(" ,;:-—") + "…"


def _perception_needs_her_own_answer(result: dict[str, Any], objective: str) -> bool:
    """Whether this screen question wants an answer rather than a description.

    "What's on my screen?" wants the description, and gets it natively — a
    screen is read by the OS, and spending a model generation to narrate
    text the accessibility API already returned buys nothing.

    "What was that repo?", "is the build passing?", "read me the exact
    wording" all want something FOUND in the reading. Handing those a tour
    of the window stack answers a question nobody asked.
    """
    if not result.get("observation_meta"):
        return False
    try:
        from core.perception.observation_evidence import AnswerShape, answer_shape_for

        return answer_shape_for(objective) is not AnswerShape.DESCRIBE
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat",
            exc,
            action="served the screen description because answer shape was undecidable",
        )
        return False


def _desktop_task_observation(result: dict[str, Any]) -> str:
    """The content a desktop task OBSERVED, as opposed to what it did.

    A read-the-screen task returns its finding inside the step receipts,
    while the top-level `summary` describes the mechanism ("completed 1/1
    governed computer-use steps through heuristic_compat planning"). Handing
    someone the mechanism when they asked what is on their screen is the
    receipt standing in for the answer.
    """
    # A PERCEPTION is DESCRIBED, never pasted.
    #
    # Measured live 2026-08-04. The scraping below reaches into the receipts
    # for the first string it can find, and for a screen read that string is
    # `result.text` — the whole accessibility dump. It was handed back as the
    # reply verbatim: "E / 0:21 / Claude / File / Edit / View / Window /
    # Help / *", forty lines of window furniture, in answer to "can you tell
    # me what you see on the screen?".
    #
    # The scrape was itself a fix (2026-08-03) for the opposite failure, a
    # step count standing in for the answer. Both are the same mistake in
    # different directions: the receipt standing in for what was seen. What
    # a person asked for is neither the buffer nor the bookkeeping — it is
    # the reading, said plainly, which desktop_task now builds natively from
    # the capture and carries here.
    described = str(result.get("observation_description") or "").strip()
    if described:
        return described
    if result.get("observation_meta"):
        # A perception happened but produced no description. There is
        # nothing honest to scrape — the capture is the one thing that must
        # not be pasted — so say nothing and let the caller stay factual.
        return ""

    observation_keys = (
        "observation",
        "screen_text",
        "accessibility_text",
        "text",
        "output",
        "content",
    )

    def _first_observation(entry: dict[str, Any], depth: int = 0) -> str:
        for key in observation_keys:
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        # The reading lives one level down. desktop_task returns `receipts`,
        # each carrying the executor's own payload under `result` — and a
        # screen read fills `text` there while leaving the receipt's own
        # `screen_text` empty. Checking only for a string at the top meant the
        # dict was skipped and the observation was never found: live on
        # 2026-08-03 a completed read of a Chrome window was reported as
        # "Desktop task completed 1/1 governed computer-use steps".
        nested = entry.get("result")
        if depth < 2 and isinstance(nested, dict):
            return _first_observation(nested, depth + 1)
        return ""

    candidates: list[str] = []
    steps: list[Any] = []
    for container in ("steps", "step_results", "receipts"):
        value = result.get(container)
        if isinstance(value, list):
            steps.extend(value)
    for step in steps:
        if not isinstance(step, dict):
            continue
        found = _first_observation(step)
        if found:
            candidates.append(found)
    for key in observation_keys:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    for candidate in candidates:
        # Skip the executor's own bookkeeping phrasing.
        lowered = candidate.casefold()
        if lowered.startswith(("desktop task completed", "completed ")):
            continue
        if len(candidate) < 3:
            continue
        return candidate[:1200]
    return ""
