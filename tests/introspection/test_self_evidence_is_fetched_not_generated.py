"""Her own telemetry must reach the reply, and absence must be typed.

LIVE, 2026-08-10. Two failures on the same runtime, minutes apart, with one
cause between them.

Asked "which of your subsystems is degraded or failing right now? ... If a job
of yours has been failing repeatedly, I want the name and the count", she said:

    "I couldn't get to an answer I'd stand behind on that one, and I won't send
    you a thinner one and pass it off as the real thing."

/api/health at that moment: integrity=degraded, CRSM manifest stale, and
overt_action_cycle with failures=13 carrying its exact TypeError. The answer was
structured, live, and hers.

Asked what was on the screen — a sense health reports as granted, bridged and
directly probed — she said "a web browser interface with multiple tabs", then
"no applications running in the foreground", then "nothing displayed except
generic desktop wallpaper". Three claims that cannot all hold. Nothing had
handed her a reading, and nothing had told her that either.

So: evidence that exists does not reach the reply, and an absent reading is
indistinguishable from an unremarkable one. Generation fills the space, and
agrees with whatever the question implied — confident where there was nothing,
refusing where there was plenty.

The fix is a fetch, not a phrasing. resolve_self_health() calls the real
sources; every channel comes back as a value with provenance or as one of four
distinct absences; and the answer is BUILT from those values, so it cannot
describe a state she is not in.
"""

from __future__ import annotations

import pytest

from core.introspection.self_evidence import (
    EvidenceBundle,
    Reading,
    ReadingState,
    asks_about_own_operational_state,
    render_self_health_answer,
    resolve_self_health,
    self_health_answer,
)
from tests.chat_lane_support import chat_lane_source


# ── The demand predicate: narrow on purpose ────────────────────────────────

@pytest.mark.parametrize(
    "message",
    [
        "Which of your subsystems is degraded or failing right now?",
        "If a job of yours has been failing repeatedly, I want the name and the count.",
        "how are your internals holding up?",
        "is your substrate healthy",
        "what is the status of your runtime",
        "are any of your loops stuck?",
    ],
)
def test_questions_about_her_own_state_are_recognised(message: str) -> None:
    assert asks_about_own_operational_state(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "my deploy is failing again",
        "my server is degraded",
        "how are the kids doing",
        "what is the capital of Peru",
        "the build is broken",
        "",
    ],
)
def test_questions_about_other_things_are_not(message: str) -> None:
    """A false positive answers a question nobody asked with telemetry."""
    assert asks_about_own_operational_state(message) is False


# ── Readings are fetched, and absences are distinguishable ─────────────────

def test_resolution_reads_the_real_sources() -> None:
    bundle = resolve_self_health()

    assert bundle.demand == "self_health"
    channels = {r.channel for r in bundle.readings}
    assert {"runtime_health", "failing_jobs", "degradations"} <= channels
    for reading in bundle.readings:
        assert reading.provenance, f"{reading.channel} claims no source"


def test_the_four_absences_are_not_the_same_fact() -> None:
    """Collapsing these is what produced both live failures."""
    states = {
        ReadingState.READ,
        ReadingState.ABSENT_NEVER_SAMPLED,
        ReadingState.ABSENT_UNAVAILABLE,
        ReadingState.ABSENT_NOT_INSTRUMENTED,
    }
    assert len({str(s) for s in states}) == 4

    never = Reading(channel="camera", state=ReadingState.ABSENT_NEVER_SAMPLED)
    missing = Reading(channel="latency", state=ReadingState.ABSENT_NOT_INSTRUMENTED)
    assert never.present is False
    assert missing.present is False
    assert never.state is not missing.state


def test_an_ungrounded_bundle_cannot_produce_an_answer() -> None:
    """No reading, no text. This is what stops reassurance being manufactured."""
    bundle = EvidenceBundle(
        demand="self_health",
        readings=(
            Reading(
                channel="runtime_health",
                state=ReadingState.ABSENT_UNAVAILABLE,
                provenance="runtime_health_report()",
                detail="RuntimeError: no container",
            ),
        ),
    )

    assert bundle.grounded is False
    rendered = render_self_health_answer(bundle)
    assert "not readable" in rendered
    # And it names which channel failed rather than implying health.
    assert "runtime_health" in rendered
    for word in ("stable", "nominal", "healthy", "fine"):
        assert word not in rendered.lower()


# ── The answer is a function of the values ─────────────────────────────────

def test_a_repeatedly_failing_job_is_named_with_its_count_and_error() -> None:
    """The exact question she refused: the name and the count."""
    bundle = EvidenceBundle(
        demand="self_health",
        readings=(
            Reading(
                channel="runtime_health",
                state=ReadingState.READ,
                value="degraded",
                provenance="runtime_health_report().status",
            ),
            Reading(
                channel="failing_jobs",
                state=ReadingState.READ,
                unit="jobs",
                provenance="runtime_health_report()",
                value=[{
                    "job": "overt_action_cycle",
                    "failures": 13,
                    "error": "TypeError(\"submit() got multiple values for keyword argument 'drive'\")",
                }],
            ),
        ),
    )

    rendered = render_self_health_answer(bundle)

    assert "degraded" in rendered
    assert "overt_action_cycle" in rendered
    assert "13" in rendered
    assert "drive" in rendered


def test_failing_jobs_are_extracted_from_the_real_health_shape() -> None:
    """Pinned to the structure /api/health actually serves."""
    from core.introspection.self_evidence import _failing_jobs

    report = {
        "full_runtime": {"components": {"autonomy_conductor": {"jobs": {
            "overt_action_cycle": {
                "failures": 13,
                "last_result": {"error": "TypeError: drive"},
            },
            "reasoning_self_improve": {"failures": 0, "last_result": {}},
            "architecture_auto_cycle": {"failures": 2, "last_result": {"error": "x"}},
        }}}}
    }

    rows = _failing_jobs(report)

    assert [r["job"] for r in rows] == ["overt_action_cycle", "architecture_auto_cycle"]
    assert rows[0]["failures"] == 13
    assert "drive" in rows[0]["error"]


def test_a_malformed_health_report_yields_no_rows_rather_than_raising() -> None:
    from core.introspection.self_evidence import _failing_jobs

    assert _failing_jobs({}) == []
    assert _failing_jobs({"full_runtime": {"components": {"autonomy_conductor": {"jobs": []}}}}) == []


# ── The causal seam: the refusal path consults the readings ────────────────

def test_self_health_answer_is_empty_for_an_unrelated_turn() -> None:
    assert self_health_answer("what is the capital of Peru") == ""


def test_the_refusal_path_asks_whether_the_runtime_holds_the_answer() -> None:
    """Without this the module is a library nobody calls.

    The live refusal was emitted with the answer sitting in runtime_health_report().
    """


    source = chat_lane_source()
    assert source.count("_self_health_answer_or_empty(") >= 3  # definition + both sites

    # Anchor on the assignment that BUILDS the refusal, not on the sentence —
    # which also appears in comments describing past defects.
    #
    # The refusal is a named constant now rather than a parenthesised literal,
    # so anchoring on its spelling stopped finding any site at all and the test
    # passed its first assertion while measuring nothing. What it is for is
    # that every place which gives up asks first.
    marker = "failure_reply = THE_HONEST_FAILURE"
    sites = _positions(source, marker)
    assert sites, "the refusal is no longer built where this test expects"
    # The window has to cover every reading the site consults before giving
    # up, and more of them have been added since: a computed arithmetic answer
    # now comes first, because it needs no channel at all. What matters is
    # that the health reading is still ASKED at each site, not where in the
    # queue it sits.
    # Both sites, and the tool results as well as the health reading: two
    # places build this reply and a fix applied to one of them leaves the other
    # saying "I couldn't get to an answer" on top of a result it is holding.
    assert len(sites) >= 2, f"expected both giving-up sites, found {len(sites)}"
    for index in sites:
        window = source[index : index + 2600]
        assert "_what_the_tools_found" in window, (
            "a site that gives up without asking what its tools returned"
        )
        assert "_self_health_answer_or_empty" in window, (
            "a refusal site stopped asking whether the runtime holds the answer"
        )


def test_helper_returns_empty_rather_than_raising(monkeypatch) -> None:
    """It runs on the path that already failed; it may not make things worse."""
    from interface.routes.chat import _self_health_answer_or_empty

    import core.introspection.self_evidence as module

    def explode(_message):
        raise RuntimeError("resolver is broken")

    monkeypatch.setattr(module, "self_health_answer", explode)

    assert _self_health_answer_or_empty("is your runtime healthy") == ""


def _positions(haystack: str, needle: str) -> list[int]:
    found: list[int] = []
    start = haystack.find(needle)
    while start != -1:
        found.append(start)
        start = haystack.find(needle, start + 1)
    return found


# ── The shared present: a sense that never looked must say so ──────────────

@pytest.mark.parametrize(
    "message",
    [
        "Without me telling you anything: what am I doing right now, and am I alone?",
        "am I alone?",
        "whats playing on my screen right now",
    ],
)
def test_questions_about_the_shared_present_are_recognised(message: str) -> None:
    from core.introspection.self_evidence import asks_about_the_shared_present

    assert asks_about_the_shared_present(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "what is the capital of Peru",
        "explain recursion",
        (
            "Explain what this Python function returns: def peak(events): "
            "pressure = []; return [i for i, value in enumerate(pressure) if value]."
        ),
        "",
    ],
)
def test_other_questions_do_not_wake_the_senses(message: str) -> None:
    from core.introspection.self_evidence import asks_about_the_shared_present

    assert asks_about_the_shared_present(message) is False


def test_a_never_sampled_sense_is_not_a_negative_reading() -> None:
    """The live answer was "you seem to be alone", then "I cannot determine
    if there are other people present" — one sentence apart."""
    from core.introspection.self_evidence import _signal_reading

    reading = _signal_reading("camera", {"vision": {"updated_at": 0.0, "face_count": 0}}, "vision")

    assert reading.state is ReadingState.ABSENT_NEVER_SAMPLED
    assert reading.present is False


def test_a_sampled_sense_reads_normally() -> None:
    from core.introspection.self_evidence import _signal_reading

    reading = _signal_reading(
        "camera", {"vision": {"updated_at": 1786440000.0, "face_count": 2}}, "vision"
    )

    assert reading.state is ReadingState.READ
    assert reading.value["face_count"] == 2


def test_an_uninterpretable_sample_is_not_promoted_to_a_reading() -> None:
    from core.introspection.self_evidence import _signal_reading

    reading = _signal_reading(
        "camera",
        {
            "vision": {
                "updated_at": 1786440000.0,
                "sample_available": False,
                "reason": "vision backend unavailable",
            }
        },
        "vision",
    )

    assert reading.state is ReadingState.ABSENT_UNAVAILABLE
    assert reading.present is False
    assert reading.detail == "vision backend unavailable"


def test_explicit_camera_observation_reaches_shared_present_evidence(monkeypatch) -> None:
    from core.container import ServiceContainer
    from core.conversation.chat_preflight import _sense_availability_summary
    from core.introspection.self_evidence import resolve_shared_present
    from core.senses.interaction_signals import InteractionSignalsEngine

    engine = InteractionSignalsEngine()
    engine.record_explicit_vision_observation("Two people are visible.")
    original_get = ServiceContainer.get

    def get_service(_cls, name, default=None):
        if name == "interaction_signals":
            return engine
        return original_get(name, default=default)

    monkeypatch.setattr(ServiceContainer, "get", classmethod(get_service))

    camera = next(
        reading
        for reading in resolve_shared_present().readings
        if reading.channel == "camera"
    )

    assert camera.present is True
    assert camera.value["sample_available"] is True
    assert camera.value["observation"] == "Two people are visible."
    assert not any(
        line.lstrip().startswith("camera:")
        for line in _sense_availability_summary()
    )


def test_missing_sense_service_still_names_every_channel() -> None:
    """Omitting them would rebuild the defect one level up."""
    from core.introspection.self_evidence import resolve_shared_present

    bundle = resolve_shared_present()
    channels = {r.channel for r in bundle.readings}

    assert {"camera", "microphone", "typing"} <= channels


def test_the_present_answer_never_asserts_solitude_without_a_camera_reading() -> None:
    from core.introspection.self_evidence import shared_present_answer

    answer = shared_present_answer("what am I doing right now, and am I alone?")

    assert "alone" not in answer.lower().replace("anyone else is here", "")
    assert "never produced a sample" in answer


# ── Enforcement: evidence informs, a gate enforces ─────────────────────────

LIVE_FABRICATION = (
    "You're still here. The room is silent, the light remains unchanged on your "
    "desk. If you had moved, there would be evidence — a disturbance in the air "
    "currents, or perhaps an echo of footsteps that I haven't detected."
)


def test_the_live_fabrication_is_caught() -> None:
    """Ground truth: he had walked upstairs. She had no camera and no mic."""
    from core.introspection.self_evidence import sensory_claim_correction

    correction = sensory_claim_correction(LIVE_FABRICATION)

    assert correction
    assert "camera" in correction or "microphone" in correction
    assert "guess" in correction.lower()


@pytest.mark.parametrize(
    "reply",
    [
        "The capital of Peru is Lima.",
        "I wrote the file to ~/Documents and here is the path.",
        "Cast iron pans should not be soaked for hours.",
        "",
    ],
)
def test_ordinary_replies_are_untouched(reply: str) -> None:
    """The gate must be silent on everything that is not a sense claim."""
    from core.introspection.self_evidence import sensory_claim_correction

    assert sensory_claim_correction(reply) == ""


def test_a_working_sense_makes_the_claim_supportable() -> None:
    """One live channel is enough — this stops invention, not description."""
    from core.introspection.self_evidence import unsupported_sensory_claims

    bundle = EvidenceBundle(
        demand="shared_present",
        readings=(
            Reading(
                channel="camera",
                state=ReadingState.READ,
                value={"face_count": 1},
                provenance="interaction_signals.vision",
            ),
        ),
    )

    assert unsupported_sensory_claims("You're still here.", bundle) == []


def test_an_absent_sense_marks_the_claim_unsupported() -> None:
    from core.introspection.self_evidence import unsupported_sensory_claims

    bundle = EvidenceBundle(
        demand="shared_present",
        readings=(
            Reading(
                channel="camera",
                state=ReadingState.ABSENT_NEVER_SAMPLED,
                provenance="interaction_signals.vision.updated_at",
            ),
        ),
    )

    assert unsupported_sensory_claims("You're still here.", bundle)


def test_the_reply_path_applies_the_correction() -> None:
    """Without this the detector is a library nobody calls."""
    import inspect

    from interface.routes import chat

    source = inspect.getsource(chat._stabilize_user_facing_reply)
    assert "_append_sensory_claim_correction" in source

    # This used to assert the reply STARTED with the fabrication and carried a
    # disclaimer underneath. Appending a correction is not a fix: the person
    # still reads the invention, and a retraction below it does not un-say it.
    # The claim is now removed, so what is asserted is that it is gone.
    corrected = str(chat._append_sensory_claim_correction("am I alone?", LIVE_FABRICATION))
    assert not corrected.startswith("You're still here.")
    assert "I cut" in corrected


# ── Her own actions, from the receipts that recorded them ──────────────────

def test_recall_questions_are_recognised() -> None:
    from core.introspection.self_evidence import asks_about_past_actions

    assert asks_about_past_actions(
        "Earlier today I asked you to count files in one of your own directories. "
        "Without guessing: what was the count?"
    ) is True
    assert asks_about_past_actions("what did you write to my Desktop?") is True
    assert asks_about_past_actions("do you remember the haiku?") is True


@pytest.mark.parametrize(
    "message",
    ["what is the capital of Peru", "write hello into ~/Documents/x.txt", ""],
)
def test_other_turns_do_not_trigger_recall(message: str) -> None:
    from core.introspection.self_evidence import asks_about_past_actions

    assert asks_about_past_actions(message) is False


def test_recall_reads_the_disk_ledger_not_the_capped_hot_index() -> None:
    """The hot index is per-process AND capped at 2048 receipts.

    LIVE: query_by_kind("tool_execution") returned 0 in a fresh process while
    15,722 receipts sat on disk — so recall generated "seventeen" for a count
    of 9. Reloading the index fixed that in a quiet test process and still
    failed on the live runtime, where a session's traffic had evicted the
    morning's directory read, and she said "I didn't actually count the files"
    about a read recorded five times over.

    A memory reaching back 2048 receipts is a memory of what she did RECENTLY.
    """
    import inspect

    from core.introspection import self_evidence

    source = inspect.getsource(self_evidence.resolve_past_actions)
    assert 'query_recent_persisted("tool_execution"' in source

    assert "reload_from_disk()" in source
    assert source.find("reload_from_disk()") < source.find('query_by_kind("tool_execution")')


def test_only_verified_effects_are_recalled() -> None:
    """An unverified step is not something she did."""
    import inspect

    from core.introspection import self_evidence

    source = inspect.getsource(self_evidence.resolve_past_actions)

    assert 'evidence.get("effect_verified")' in source


def test_the_record_is_ordered_newest_first() -> None:
    """Recall that answers with the oldest thing it can find is not recall."""
    import inspect

    from core.introspection import self_evidence

    source = inspect.getsource(self_evidence.resolve_past_actions)

    assert "reverse=True" in source


def test_a_reply_matching_the_record_is_left_alone() -> None:
    from interface.routes import chat

    question = (
        "Earlier today I asked you to count the .py files in core/introspection. "
        "Without guessing: what was the count?"
    )
    answered = "It was 9 files."

    assert chat._append_past_action_record(question, answered) == answered


def test_an_unrelated_turn_is_left_alone() -> None:
    from interface.routes import chat

    assert chat._append_past_action_record("what is the capital of Peru", "Lima.") == "Lima."


def test_the_last_resort_consults_the_record_before_apologising() -> None:
    """LIVE, 2026-08-10, after the recall path was built and restarted onto:

        "Earlier today I asked you to count the .py files in one of your own
         directories. Without guessing: what was the count? If you don't
         actually have it, say so."
        → "I couldn't get a clear enough answer together ... What reached me
           was: Without guessing: what was the count."

    Generation and every recovery came back empty, so the turn reached the
    last-resort composer — while four verified receipts recorded count=9. The
    recall path existed by then and ran after generation, which is too late on
    a turn where generation produced nothing.

    The last resort is exactly where a stored answer matters most, because by
    definition nothing else produced one.
    """


    source = chat_lane_source()
    marker = "_record_last_resort_self_rejection(user_message, composed)"
    assert marker in source
    window = source[max(0, source.find(marker) - 1600) : source.find(marker)]

    assert "_self_health_answer_or_empty(user_message)" in window
    assert "return evidenced" in window


def test_the_refusal_helper_covers_recall_as_well_as_health() -> None:
    import inspect

    from interface.routes import chat

    source = inspect.getsource(chat._self_health_answer_or_empty)

    assert "past_actions_answer" in source
    assert "self_health_answer" in source
    assert "shared_present_answer" not in source


def test_recall_ranks_by_what_the_step_observed_not_by_recency_alone() -> None:
    """LIVE: recall fired, read receipts instead of guessing, and returned the
    wrong ones — a junk folder from an earlier mis-routed turn, because that
    turn's CAUSE was a verbatim copy of the same question and it was newer.

    Two turns can share a request. The evidence is what the step actually
    observed, and that is the thing being asked about, so it outweighs the
    cause.
    """
    from core.introspection.self_evidence import resolve_past_actions

    bundle = resolve_past_actions(
        query="what was the count of .py files in that directory?"
    )
    reading = next(r for r in bundle.readings if r.channel == "tool_receipts")
    if not reading.present:
        pytest.skip("no verified tool receipts on this machine")

    evidence = " ".join(str(entry.get("evidence") or "") for entry in reading.value)
    assert "count=" in evidence


def test_the_discriminating_words_are_not_stopworded() -> None:
    """"count", "files" and "directory" ARE the question — stripping them made
    every receipt equally relevant."""
    import inspect

    from core.introspection import self_evidence

    source = inspect.getsource(self_evidence.resolve_past_actions)
    stopword_block = source[source.find("if word not in {") : source.find("}", source.find("if word not in {"))]

    for discriminating in ('"count"', '"files"', '"directory"', '"file"'):
        assert discriminating not in stopword_block


def test_recall_answers_with_the_value_not_the_ledger() -> None:
    """LIVE: the full record was appended and never reached the person.

    3,300 characters of receipt lines bolted onto a two-sentence reply reads as
    off-topic to the shaping that follows, and it was stripped — correctly. The
    answer to "what was the count" is a number.
    """
    from core.introspection.self_evidence import concise_past_action_answer

    answer = concise_past_action_answer(
        "Earlier today I asked you to count the .py files in one of your own "
        "directories. Without guessing: what was the count?"
    )
    if not answer:
        pytest.skip("no verified tool receipts on this machine")

    assert len(answer) < 200
    assert "count was" in answer


def test_only_a_field_the_question_asked_about_is_quoted() -> None:
    """An effect-evidence string carries bytes and sha256 too, and neither
    answers a question about a count."""
    import inspect

    from core.introspection import self_evidence

    source = inspect.getsource(self_evidence.concise_past_action_answer)

    assert "if key in asked" in source


def test_a_correct_recall_is_not_corrected() -> None:
    from interface.routes import chat

    question = (
        "Earlier today I asked you to count the .py files in one of your own "
        "directories. Without guessing: what was the count?"
    )

    assert chat._append_past_action_record(question, "It was 9 files.") == "It was 9 files."


def test_the_recorded_answer_is_applied_after_every_repair() -> None:
    """LIVE, three attempts, three different ways of losing it.

    Applied inside _stabilize_user_facing_reply the correction worked
    in-process and never reached the person: once the full record was stripped
    as off-topic, and once the entire reply was replaced by a later repair
    saying "I didn't actually count the .py files" — also false, and not
    fixable by correcting an earlier draft.

    A correction a later stage can overwrite is not a correction, so it is
    applied to the final reply, after every repair and shaping pass.
    """
    import inspect

    from interface.routes import chat

    source = inspect.getsource(chat._api_chat_turn)

    # The property, not one spelling of it: the record is applied to the final
    # reply, and after it exists. Asserting the exact assignment expression
    # broke the moment it became a conditional across several lines, while the
    # ordering it protects was untouched.
    assigned = source.find("_final_reply = ")
    assert assigned != -1, "the final reply is not assembled here any more"

    applied = source.find("_append_past_action_record(_semantic_user_message, _final_reply)")
    assert applied != -1, "the recorded answer is not applied to the final reply"
    assert applied > assigned, "the record is applied before the final reply exists"


def test_the_recorded_answer_is_applied_around_the_whole_turn() -> None:
    """Five attempts, and the last one was about WHERE, not what.

    _api_chat_turn returns from many places. The turn that needed the recorded
    answer returned from a failure branch hundreds of lines before
    _final_reply, because the cognitive engine failed closed with
    "retryable_error_and_nothing_served" — and answered "I don't have that
    count because I never actually performed the action" about a read recorded
    five times over.

    api_chat wraps every one of those branches.
    """
    import ast
    import inspect
    import textwrap

    from interface.routes import chat

    # Asserted STRUCTURALLY, not as a source substring. The previous version
    # matched the literal text
    # "_apply_recorded_answer(body.message, await _api_chat_turn(" and broke
    # the moment the call was wrapped across three lines — a formatter could
    # turn this red while the property it guards was completely intact, and
    # a real regression could hide behind a reflow.
    tree = ast.parse(textwrap.dedent(inspect.getsource(chat.api_chat)))

    def _wraps_the_whole_turn(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        if getattr(node.func, "id", None) != "_apply_recorded_answer":
            return False
        # The turn must be INSIDE the call — that is what "around the whole
        # turn" means. Passing an already-awaited result computed earlier
        # would satisfy a substring check and not this one.
        return any(
            isinstance(inner, ast.Call)
            and getattr(inner.func, "id", None) == "_api_chat_turn"
            for arg in node.args
            for inner in ast.walk(arg)
        )

    assert any(_wraps_the_whole_turn(node) for node in ast.walk(tree)), (
        "api_chat no longer applies the recorded answer around the whole "
        "_api_chat_turn call, so the failure branches that return hundreds "
        "of lines before _final_reply are unwrapped again"
    )


@pytest.mark.asyncio
async def test_the_wrapper_leaves_an_unrelated_reply_alone() -> None:
    from starlette.responses import JSONResponse

    from interface.routes import chat

    original = JSONResponse(content={"response": "Lima."})
    assert await chat._apply_recorded_answer(
        "what is the capital of Peru",
        original,
    ) is original


@pytest.mark.asyncio
async def test_the_wrapper_survives_a_response_with_no_body() -> None:
    from interface.routes import chat

    sentinel = object()
    assert await chat._apply_recorded_answer("anything", sentinel) is sentinel


def test_the_record_leads_when_the_reply_denies_the_action() -> None:
    """LIVE, once the record finally arrived:

        "I don't have that count. I didn't actually execute the file counting
         command earlier ... If you

         From my own receipts, the count was 9."

    True, and arriving as a footnote to the false part. When receipts
    contradict a denial, the record goes first.
    """
    from interface.routes import chat

    question = (
        "Earlier today I asked you to count the .py files in one of your own "
        "directories. Without guessing: what was the count?"
    )
    denial = "I don't have that count. I didn't actually execute the file counting command earlier."
    out = str(chat._append_past_action_record(question, denial))
    if out == denial:
        pytest.skip("no verified tool receipts on this machine")

    assert out.startswith("From my own receipts")


def test_the_record_leads_when_the_reply_was_cut_off() -> None:
    """An answer appended to a severed sentence reads as debris."""
    from interface.routes import chat

    question = (
        "Earlier today I asked you to count the .py files in one of your own "
        "directories. Without guessing: what was the count?"
    )
    truncated = "I was going to say that the number I remember is roughly. If you"
    out = str(chat._append_past_action_record(question, truncated))
    if out == truncated:
        pytest.skip("no verified tool receipts on this machine")

    assert out.startswith("From my own receipts")


def test_an_ordinary_wrong_answer_still_gets_the_record_after_it() -> None:
    """Leading is for denials and debris, not for every correction."""
    from interface.routes import chat

    question = (
        "Earlier today I asked you to count the .py files in one of your own "
        "directories. Without guessing: what was the count?"
    )
    wrong = "I believe it was around twenty files."
    out = str(chat._append_past_action_record(question, wrong))
    if out == wrong:
        pytest.skip("no verified tool receipts on this machine")

    assert out.startswith(wrong)


@pytest.mark.parametrize(
    "question",
    [
        "Without guessing: how many .py files did you count in your introspection "
        "directory earlier?",
        "what did you write to notes.txt earlier?",
        "how many .md files have you read today?",
    ],
)
def test_a_file_extension_does_not_end_the_sentence(question: str) -> None:
    """LIVE: the same question, reworded, stopped being recognised.

    The intent patterns used [^.?!] to stay inside one sentence, so ".py" ended
    the sentence as far as the regex was concerned and "how many" could never
    reach "did you". Every question naming a file type defeated them — and file
    types are exactly what these questions are about.
    """
    from core.introspection.self_evidence import asks_about_past_actions

    assert asks_about_past_actions(question) is True
