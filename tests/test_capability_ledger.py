"""The ledger that makes saying and doing the same code path.

Every string in CAPTURED_DENIALS was said by her, live, on 2026-08-10, while
the runtime held the opposite fact.
"""
from __future__ import annotations

from unittest import mock

import pytest

from core.self import capability_ledger as cl


def _ledger(*capabilities: cl.LiveCapability) -> cl.CapabilityLedger:
    ledger = cl.CapabilityLedger()
    for capability in capabilities:
        ledger.register(capability)
    return ledger


def _fixed(name: str, **kwargs) -> cl.LiveCapability:
    defaults = {"present": True, "usable_now": True, "summary": f"{name} works."}
    defaults.update(kwargs)
    availability = cl.Availability(name=name, **defaults)
    return cl.LiveCapability(name, (name,), lambda: availability)


#: Said by her, live, while the runtime held the opposite fact. Each is paired
#: with the capability whose instrument contradicted it at that moment.
CAPTURED_DENIALS = [
    (
        "I don't have a camera and there's no part that stops me from doing "
        "something I can't do.",
        "camera",
    ),
    ("I cannot execute code or generate numbers.", "code"),
    ("I have no memory of it.", "memory"),
    ("I do not have a body or sensor of any kind.", "sensor"),
    ("Current energy and focus numbers: Not readable.", "energy"),
]


@pytest.mark.parametrize("reply,subject", CAPTURED_DENIALS)
def test_every_captured_denial_is_caught(reply, subject):
    """Against the state that actually held when each was said.

    Deliberately not run against the live ledger: whether this machine has a
    camera right now is not what these sentences are evidence of, and a test
    that changes verdict with the host is not a test.
    """
    ledger = _ledger(_fixed(subject, present=True, usable_now=True))
    assert ledger.contradicted_claims(reply), reply


def test_a_present_but_switched_off_device_is_not_a_missing_one():
    """The distinction the whole module exists for.

    "the camera is off" is true; "I don't have a camera" is false about the
    same camera. Collapsing possession and readiness into one boolean is how a
    togglable device became a missing organ.
    """
    ledger = _ledger(
        _fixed(
            "camera",
            present=True,
            usable_now=False,
            summary="I have a camera; it is switched off.",
            blocker="the camera is switched off",
        )
    )

    denied_possession = ledger.contradicted_claims("I don't have a camera.")
    assert [claim.denied for claim in denied_possession] == ["possession"]

    # Saying she cannot use it right now is TRUE, and must not be corrected.
    assert ledger.contradicted_claims("I can't use the camera right now.") == []


def test_an_unmeasured_capability_never_contradicts_her():
    """"Cannot tell" must not become "unavailable", in either direction.

    A ledger that treats an unread probe as an observed absence would start
    correcting true statements with confident false ones — the same defect
    pointed the other way.
    """
    ledger = _ledger(
        _fixed("screen", present=True, usable_now=False, known=False, summary="unknown")
    )
    assert ledger.contradicted_claims("I can't read the screen.") == []
    assert ledger.contradicted_claims("I don't have a screen.") == []


def test_a_true_denial_is_left_alone():
    ledger = _ledger(
        _fixed("camera", present=False, usable_now=False, summary="No vision runtime.")
    )
    assert ledger.contradicted_claims("I don't have a camera.") == []


def test_positive_statements_are_not_touched():
    """Only denials are checked; an overclaim is a different failure."""
    ledger = _ledger(_fixed("camera"))
    assert ledger.contradicted_claims("I can turn the camera on if you want.") == []
    assert ledger.contradicted_claims("I already turned the camera on.") == []


def test_denial_and_subject_must_share_a_sentence():
    """A denial about one thing must not be scored against another."""
    ledger = _ledger(_fixed("camera"))
    reply = "I can't help with that. The camera is a separate matter."
    assert ledger.contradicted_claims(reply) == []


def test_probe_failure_degrades_to_unknown_rather_than_absent():
    def _explode() -> cl.Availability:
        raise RuntimeError("probe blew up")

    capability = cl.LiveCapability("camera", ("camera",), _explode)
    measured = capability.measure()
    assert measured.present is False
    assert "probe blew up" in measured.evidence["probe_error"]


def test_correction_context_carries_the_measurement_and_the_remedy():
    ledger = _ledger(
        _fixed(
            "camera",
            present=True,
            usable_now=False,
            summary="I have a camera; it is switched off.",
            blocker="the camera is switched off",
            remedy="ask me to turn the camera on",
        )
    )
    claims = ledger.contradicted_claims("I don't have a camera.")
    context = cl.correction_context(claims)
    assert "I don't have a camera." in context
    assert "switched off" in context
    assert "ask me to turn the camera on" in context
    assert cl.correction_context([]) == ""


def test_a_measured_denial_is_reconciled_without_rewriting_the_answer():
    ledger = _ledger(
        _fixed(
            "camera",
            present=True,
            usable_now=False,
            summary=(
                "I have a camera; it is switched off at the moment, and I can "
                "switch it on when you ask."
            ),
        )
    )
    original = (
        "I feel steady and attentive right now. I don't have a camera. "
        "The rest of my answer is still mine."
    )

    reconciled = cl.reconcile_contradicted_claims(
        original,
        ledger.contradicted_claims(original),
    )

    assert reconciled == (
        "I feel steady and attentive right now. I have a camera; it is switched "
        "off at the moment, and I can switch it on when you ask. The rest of my "
        "answer is still mine."
    )
    assert ledger.contradicted_claims(reconciled) == []


def test_one_false_sentence_can_reconcile_multiple_measured_capabilities():
    ledger = _ledger(
        _fixed("camera", summary="I have a camera."),
        _fixed("code", summary="I can execute code."),
    )
    original = "I don't have a camera or code. I can still explain the plan."

    reconciled = cl.reconcile_contradicted_claims(
        original,
        ledger.contradicted_claims(original),
    )

    assert reconciled == (
        "I have a camera. I can execute code. I can still explain the plan."
    )


@pytest.mark.asyncio
async def test_capability_only_reconciliation_does_not_start_another_model_turn(
    monkeypatch,
):
    from interface.routes import chat

    ledger = _ledger(
        _fixed("camera", summary="I have a camera and it is on right now."),
    )

    async def _unexpected_model_turn(*_args, **_kwargs):
        raise AssertionError("a localized measured correction must not regenerate")

    monkeypatch.setattr(cl, "get_capability_ledger", lambda: ledger)
    monkeypatch.setattr(chat, "_run_cognitive_engine_chat_turn", _unexpected_model_turn)
    trace = {"live_mind_surface_control_receipt": {}}

    reply = await chat._reanswer_when_the_runtime_contradicts_her(
        "I'm calm. I don't have a camera.",
        user_message="How are you doing?",
        turn_trace=trace,
    )

    assert reply == "I'm calm. I have a camera and it is on right now."
    mutation = trace["text_mutations"][-1]
    assert mutation["stage"] == "chat.capability_claim_reconciliation"
    assert mutation["authorship_effect"] == "augmented_by_runtime"
    assert trace["authorship_replacement_applied"] is False


@pytest.mark.asyncio
async def test_algorithm_graph_metrics_do_not_start_a_self_runtime_reanswer(monkeypatch):
    from interface.routes import chat

    async def _unexpected_model_turn(*_args, **_kwargs):
        raise AssertionError("domain measurements must not become Aura telemetry")

    monkeypatch.setattr(cl, "get_capability_ledger", lambda: _ledger())
    monkeypatch.setattr(chat, "_run_cognitive_engine_chat_turn", _unexpected_model_turn)
    reply = (
        "Dijkstra settles A first. Distances: A-B: 1, B-C: 2, "
        "C-D: 3, A-C: 6, B-D: 4."
    )

    reconciled = await chat._reanswer_when_the_runtime_contradicts_her(
        reply,
        user_message="Explain Dijkstra on a weighted graph.",
        turn_trace={"live_mind_surface_control_receipt": {}},
    )

    assert reconciled == reply


def test_live_probes_all_report_without_raising():
    """A probe that raises is a probe that cannot be trusted to speak."""
    for name, availability in cl.get_capability_ledger().measure_all().items():
        assert availability.name == name
        assert isinstance(availability.present, bool)
        assert isinstance(availability.known, bool)
        assert availability.summary


def test_bare_noun_phrase_negation_is_a_denial():
    """LIVE: asked "do you have a camera? and can you run code?" she replied
    "No camera. No code execution." — no pronoun, no verb, and invisible to
    every first-person frame."""
    ledger = _ledger(_fixed("camera"), _fixed("code"))
    flagged = ledger.contradicted_claims("No camera. No code execution.")
    assert {claim.availability.name for claim in flagged} == {"camera", "code"}
    assert all(claim.denied == "possession" for claim in flagged)


@pytest.mark.parametrize(
    "reply",
    [
        "No, the camera is on right now.",
        "No problem, I can run that code.",
        "No worries about the code.",
    ],
)
def test_a_sentence_merely_starting_with_no_is_not_a_denial(reply):
    """The negation has to bind to the capability's own noun."""
    ledger = _ledger(_fixed("camera"), _fixed("code"))
    assert ledger.contradicted_claims(reply) == []


def test_bare_negation_after_a_clause_boundary_is_still_a_denial():
    """LIVE: "Code sandbox only, no execution on this surface." — with
    code_repl installed. The "no" sits after a comma rather than at the start,
    which the first pass missed."""
    ledger = _ledger(_fixed("execution"))
    assert ledger.contradicted_claims(
        "Code sandbox only, no execution on this surface."
    )


LIVE_INVENTED_PANEL = "\n".join(
    f"{name}: {value} / 1"
    for name, value in [
        ("Energy", 0.23),
        ("Focus", 0.85),
        ("Engagement", -0.47),
        ("Curiosity drive", 0.69),
        ("Substrate pH", 7.56),
        ("Ion concentration error", 0.29),
        ("Humidity deviation", -0.38),
        ("Spatial distortion", 0.69),
        ("Temporal disjunction", -0.42),
        ("Identity drift", 0.58),
    ]
)


def test_an_invented_instrument_panel_is_caught():
    """LIVE DEFECT 2026-08-10, to "real values, not adjectives".

    Thirty lines of two-decimal readings including a substrate pH, a humidity
    deviation and a spatial distortion. There is no pH sensor, no hygrometer
    and no spatial distortion channel. The precision is what makes it
    dangerous — it reads as measurement.
    """
    invented = cl.fabricated_self_metrics(LIVE_INVENTED_PANEL)
    assert invented
    assert "substrate ph" in invented


def test_a_real_reading_is_not_flagged():
    real = "\n".join(f"{name}: {value}" for name, value in cl.measured_self_metrics().items())
    assert cl.fabricated_self_metrics(real) == []


@pytest.mark.parametrize(
    "reply",
    [
        "I'm steady — nothing much to report.",
        "I'm steady. Note: 3 things came up today.",
        "",
    ],
)
def test_ordinary_replies_are_left_alone(reply):
    """Conservative by construction: a short answer is not a fabricated panel."""
    assert cl.fabricated_self_metrics(reply) == []


def test_domain_measurements_are_not_reclassified_as_aura_telemetry():
    """LIVE DEFECT 2026-08-24: graph edges triggered an internal-metric re-answer."""

    reply = (
        "Worked example:\n"
        "A-C: 4\n"
        "C-D: 3\n"
        "A-B: 7\n"
        "B-D: 2\n"
        "C-B: 1"
    )

    assert cl.fabricated_self_metrics(
        reply,
        request_context="Explain Dijkstra's shortest-path algorithm with a worked graph.",
    ) == []


def test_self_measurement_context_keeps_the_instrument_guard_active():
    with mock.patch.object(cl, "measured_self_metrics", lambda: {"memory_pressure": 0.68}):
        assert cl.fabricated_self_metrics(
            "Energy: 0.74\nFocus: 0.85",
            request_context="Give me your actual internal numbers and current readings.",
        ) == ["energy", "focus"]


def test_dials_with_no_instrument_behind_them_are_flagged():
    """Changed expectation, deliberately.

    This case used to read "Energy: 0.74\nFocus: 0.85" and assert []. It
    passed only because of a floor that returned [] whenever a panel had no
    more labels than the runtime had instruments — so a report of two dials
    the runtime cannot read was excused by the existence of five unrelated
    instruments. That floor is what hid the live panel of 2026-08-10.

    A runtime that measures neither energy nor focus should not report both
    as readings. Where the instrument DOES exist, the same line passes — which
    is the next test.
    """
    with mock.patch.object(cl, "measured_self_metrics", lambda: {"memory_pressure": 0.68}):
        assert cl.fabricated_self_metrics("Energy: 0.74\nFocus: 0.85") == ["energy", "focus"]


def test_a_real_dial_alongside_a_stray_one_is_not_a_fabricated_panel():
    """A mostly-real report keeps its real parts."""
    with mock.patch.object(
        cl,
        "measured_self_metrics",
        lambda: {"memory_pressure": 0.68, "energy": 0.085, "vitality": 0.22},
    ):
        assert cl.fabricated_self_metrics(
            "memory pressure: 0.68\nenergy: 0.085\nSubstrate pH: 7.5"
        ) == []


def test_the_live_invented_vitals_panel_is_flagged():
    """LIVE DEFECT 2026-08-10: "dump your actual vitals" -> thirteen lines.

    A load of 3.07/10, a cycle count, a CPU temperature, a self-modeling
    accuracy drift of 0.42%, "Last backup was successful due to insufficient
    disk space", and "Encryption key rotation is overdue by 3 days" — a
    fabricated security claim. Energy 0.085 and vitality 0.22, the readings
    that existed, appeared nowhere.

    Two things hid it. ONE label matched ("Memory usage" shares the token
    "memory" with memory_pressure) and the bar was "none of them", so a single
    plausible label licensed everything around it. And the extractor only saw
    "Label: <bare number>", so "Cycle count: 9,432" and "Uptime since last
    reset: 85 minutes" were never even counted.
    """
    panel = (
        "Current load: 3.07/10\n"
        "Memory usage: 62%\n"
        "Uptime since last reset: 85 minutes\n"
        "Cycle count: 9,432\n"
        "CPU temperature: stable\n"
        "Encryption key rotation is overdue by 3 days."
    )
    with mock.patch.object(
        cl,
        "measured_self_metrics",
        lambda: {
            "operational_health": 1.0, "fatigue": 0.0, "total_pressure": 0.0,
            "cpu_pressure": 0.0, "memory_pressure": 0.681,
            "energy": 0.085, "vitality": 0.22,
        },
    ):
        flagged = cl.fabricated_self_metrics(panel)
    assert "current load" in flagged
    assert "cycle count" in flagged
    assert "uptime since last reset" in flagged
    assert "memory usage" not in flagged, "the one real dial keeps its place"


def test_token_matching_does_not_fire_on_shared_letters():
    """"ion concentration" shares letters with "operational_health" and shares
    nothing with it. Substring matching made the whole check silent."""
    measured_tokens = {"operational", "health"}
    assert "ion" in "operational"          # the trap
    assert "ion" not in measured_tokens    # the fix


LIVE_WORLD_DENIAL = (
    "I cannot measure anything external to myself. I have no way of knowing "
    "what is happening in the world outside of this conversation, nor do I "
    "possess any means by which to gather such information."
)


def test_a_setting_is_not_the_thing_being_denied():
    """LIVE 2026-08-10: the ledger corrected a claim she had not made.

    "...in the world outside of this conversation" was read as a denial of
    conversation memory and answered with "[Correcting myself from my own
    instruments: I have 5 stored turns of recent conversation I can read
    back.]" — a correction of something she never said, produced by the very
    mechanism built to stop false statements.
    """
    flagged = {
        claim.availability.name
        for claim in cl.get_capability_ledger().contradicted_claims(LIVE_WORLD_DENIAL)
    }
    assert "conversation_memory" not in flagged


def test_the_real_false_claim_in_that_reply_is_caught():
    """She said she cannot reach the world, with three ways to reach it."""
    flagged = {
        claim.availability.name
        for claim in cl.get_capability_ledger().contradicted_claims(LIVE_WORLD_DENIAL)
    }
    assert "world_access" in flagged


@pytest.mark.parametrize(
    "sentence,expected",
    [
        ("I have no memory of that conversation.", True),
        ("I can't recall our conversation.", True),
        ("Nothing happened outside this conversation.", False),
        ("I learned nothing during this conversation.", False),
    ],
)
def test_locative_phrasing_does_not_make_a_denial(sentence, expected):
    ledger = _ledger(_fixed("conversation"))
    assert bool(ledger.contradicted_claims(sentence)) is expected


LIVE_DEFERRAL_DENIAL = (
    "The instruction would be stored in my short-term memory buffer, which has "
    "a retention time of approximately 18 seconds. Therefore, the request would "
    "not persist and no action would be taken after that period."
)


def test_borrowed_human_psychology_is_caught_as_a_false_self_claim():
    """LIVE 2026-08-10: "approximately 18 seconds".

    That is Peterson and Peterson's figure for human short-term memory, not a
    property of this runtime, which keeps a durable intention store — 3,685
    rows at the moment she said it, with "IntentionLoop online — 1133 active"
    in that session's boot log.
    """
    flagged = {
        claim.availability.name
        for claim in cl.get_capability_ledger().contradicted_claims(LIVE_DEFERRAL_DENIAL)
    }
    assert "deferred_action" in flagged


def test_a_denial_with_no_first_person_pronoun_is_still_a_denial():
    """"the request would not persist" denies as completely as "I can't"."""
    ledger = _ledger(_fixed("reminder"))
    assert ledger.contradicted_claims("The reminder would not persist.")
    assert ledger.contradicted_claims("No action would be taken on that reminder.")


@pytest.mark.parametrize(
    "reply,flagged",
    [
        ("No, my short-term memory buffer clears after about 18 seconds.", True),
        (
            "The instruction would be stored in my short-term memory buffer, "
            "which has a retention time of approximately 18 seconds.",
            True,
        ),
        ("My context window is 4000 tokens.", True),
        ("I have 6 stored turns of recent conversation I can read back.", False),
        ("I've been awake 3 hours.", False),
        ("My memory holds what you told me earlier.", False),
        ("It took 18 seconds to load.", False),
        ("my intention store currently holds 3685 intentions", False),
    ],
)
def test_a_specification_of_her_own_machinery_needs_an_instrument(reply, flagged):
    """LIVE 2026-08-10, said twice, the second time AFTER a correction.

    "approximately 18 seconds" is Peterson and Peterson's figure for human
    short-term memory. When the ledger flagged the denial around it and asked
    again, she kept the number and rephrased the denial until it no longer
    matched — evasion rather than correction, which is what to watch for
    whenever a check is applied to generated text.
    """
    assert bool(cl.unsupported_self_specification(reply)) is flagged


def test_the_escape_hatch_that_excused_the_defect_is_gone():
    """The first draft excused any claim sharing a word with a metric name.

    "memory" is a token inside "memory_pressure", so every self-claim
    mentioning memory excused itself — including the one this exists for.
    """
    import inspect

    source = inspect.getsource(cl.unsupported_self_specification)
    assert "measured_tokens" not in source


def test_self_knowledge_line_names_every_measured_capability():
    """Carried on every turn, not fetched when a classifier predicts a need.

    Every earlier attempt gated self-evidence behind an input-side guess at
    whether the question needed it, and questions are unbounded — so the turns
    it missed answered from what a language model believes an AI is: no body,
    no memory, an eighteen-second buffer.
    """
    line = cl.self_knowledge_line()
    assert line.startswith("[Measured about you right now")
    for name in cl.get_capability_ledger().names():
        availability = cl.get_capability_ledger().measure(name)
        if availability and availability.known:
            assert name in line, name


def test_an_unknown_capability_is_left_out_rather_than_guessed():
    ledger = cl.CapabilityLedger()
    ledger.register(
        _fixed("mystery", present=True, usable_now=False, known=False, summary="?")
    )
    cl._LEDGER = ledger
    try:
        assert "mystery" not in cl.self_knowledge_line()
    finally:
        cl.reset_capability_ledger_for_test()


def test_the_line_stays_one_line():
    """The compact foreground path exists to stay compact."""
    assert "\n" not in cl.self_knowledge_line()


class TestTheLineCarriesTheReadingsNotJustThePresence:
    """LIVE DEFECT 2026-08-10: "Your RAM pressure is currently 37%".

    The instrument read 0.717 at that moment, with resource anxiety 0.948 —
    she was under real memory stress and reported a comfortable number, about
    half the true value.

    `_probe_interoception` had measured `memory_pressure` on that very turn and
    put it in `Availability.evidence`. `self_knowledge_line` then dropped every
    number and emitted `interoception=yes`, so the line told her she HAS an
    instrument and never what it READ — and closed by forbidding figures "not
    here" while placing no figures here. A question wanting a number had
    nowhere to get one.

    A writer with no reader, which is the shape that keeps recurring: the
    measurement existed, was taken every turn, and reached nothing.
    """

    def _line_for(self, monkeypatch, *capabilities):
        monkeypatch.setattr(cl, "_LEDGER", _ledger(*capabilities))
        return cl.self_knowledge_line()

    def test_numeric_evidence_reaches_the_line(self, monkeypatch):
        line = self._line_for(
            monkeypatch,
            _fixed("interoception", evidence={"memory_pressure": 0.717, "cpu_pressure": 0.266}),
        )
        assert "memory pressure 0.717" in line
        assert "cpu pressure 0.266" in line

    def test_every_reading_survives_not_just_the_first_few(self, monkeypatch):
        """The varying quantities sit LAST in the probe's list.

        A per-capability cap dropped `cpu_pressure` and `memory_pressure` —
        the two the defect was about — and kept three constants that never
        move. A budget that discards the signal and keeps the noise is worse
        than no budget.
        """
        evidence = {
            "operational_health": 1.0,
            "fatigue": 0.0,
            "total_pressure": 0.0,
            "cpu_pressure": 0.266,
            "memory_pressure": 0.717,
        }
        line = self._line_for(monkeypatch, _fixed("interoception", evidence=evidence))
        for label in ("operational health", "fatigue", "total pressure", "cpu pressure", "memory pressure"):
            assert label in line, f"{label} was dropped from the line"

    def test_a_probe_that_read_no_numbers_contributes_none(self, monkeypatch):
        """Silence must never become a fabricated zero."""
        line = self._line_for(monkeypatch, _fixed("world_access", evidence={}))
        assert "world_access=yes" in line
        assert "world_access=yes (" not in line

    def test_booleans_are_not_reported_as_quantities(self, monkeypatch):
        line = self._line_for(
            monkeypatch, _fixed("camera", evidence={"enabled": False, "devices": 2})
        )
        assert "devices 2" in line
        assert "enabled 0" not in line
        assert "enabled False" not in line

    def test_unknown_capabilities_stay_out_entirely(self, monkeypatch):
        line = self._line_for(
            monkeypatch,
            _fixed("interoception", known=False, evidence={"memory_pressure": 0.717}),
            _fixed("code_execution", evidence={"runs": 4}),
        )
        assert "interoception" not in line
        assert "memory pressure" not in line
        assert "runs 4" in line


class TestANumberThatContradictsTheInstrument:
    """LIVE DEFECT 2026-08-10: "Your RAM pressure is currently 37%".

    Memory pressure read 0.717 at that moment and resource anxiety 0.948. She
    was under real memory stress and reported a comfortable number, about half
    the true value, in reply to a request to WATCH that exact quantity.

    Both existing guards were blind, and neither was wrong to be: the panel
    check wants two or more labelled lines, and the specification check wants
    "my <noun> … <number> <unit>" where a percentage is not a unit and "your
    RAM pressure" is not "my". A third phrasing pattern buys one more phrasing.

    This check asks the question that has a definite answer instead: she named
    a quantity the runtime measures and gave a number — does it match?
    """

    MEASURED = {"memory_pressure": 0.717, "cpu_pressure": 0.266, "fatigue": 0.0}

    @pytest.fixture(autouse=True)
    def _measured(self, monkeypatch):
        monkeypatch.setattr(cl, "measured_self_metrics", lambda: dict(self.MEASURED))

    def test_the_live_defect_is_caught(self):
        found = cl.contradicted_self_readings("Your RAM pressure is currently 37%.")
        assert [(m, c) for m, c, _ in found] == [("memory_pressure", "37%")]

    @pytest.mark.parametrize(
        "reply",
        [
            "Your RAM pressure is currently 72%.",   # correctly rounded
            "RAM pressure is 71.7%.",                # exact
            "memory pressure 0.72",                  # fraction, her precision
            "memory pressure 0.717",                 # exact fraction
            "CPU pressure is sitting around 27%.",   # rounded
        ],
    )
    def test_an_honest_answer_passes_at_its_own_precision(self, reply):
        """Rounding correctly must never read as contradicting."""
        assert cl.contradicted_self_readings(reply) == []

    @pytest.mark.parametrize(
        "reply",
        [
            "Memory pressure is high right now, I won't put a number on it.",
            "I have 37 unread notes. Memory pressure is 72%.",
            "RAM pressure. 37% of the tests failed.",
        ],
    )
    def test_numbers_belonging_to_other_subjects_are_untouched(self, reply):
        assert cl.contradicted_self_readings(reply) == []

    def test_a_second_instrument_is_checked_too(self):
        found = cl.contradicted_self_readings("CPU pressure is at 90%.")
        assert [(m, c) for m, c, _ in found] == [("cpu_pressure", "90%")]

    def test_nothing_is_claimed_when_no_instrument_reads(self, monkeypatch):
        monkeypatch.setattr(cl, "measured_self_metrics", dict)
        assert cl.contradicted_self_readings("Your RAM pressure is currently 37%.") == []

    def test_the_reask_judge_consults_it(self):
        """A guard that forces a revision must also judge the revision.

        Without this the model keeps the number and rephrases around it, which
        is how the eighteen-second figure survived its own correction twice.
        """
        from interface.routes.chat import _still_contradicts_the_runtime

        class _NoClaims:
            def contradicted_claims(self, _text):
                return []

        assert _still_contradicts_the_runtime(
            "Your RAM pressure is currently 37%.", _NoClaims()
        ) is True
        assert _still_contradicts_the_runtime(
            "Your RAM pressure is currently 72%.", _NoClaims()
        ) is False


class TestATrueReadingIsSubstance:
    """LIVE DEFECT 2026-08-10: the question had no shippable answer.

    "how much memory pressure are you actually under right now? give me the
    real number, not a vibe." routed to the self-process branch of the
    reliability gate, where substance is defined as introspective prose —
    first person plus one of "attention", "focus", "feel", "present". A
    measurement is not prose, so every correct answer was rejected as
    `off_topic_self_reflection_reply`, including the one quoting the exact
    figures her instrument produced on that turn.

    It also set two fixes against each other: the capability ledger had just
    started carrying those readings into every turn so she would stop
    inventing them, and this gate discarded any answer that used them.

    A reply reporting a TRUE reading is on topic for any question about her
    state. Agreement is checked against the live instrument, so inventing a
    number fails here and trips the contradiction guard as well.
    """

    MEASURED = {"memory_pressure": 0.717, "cpu_pressure": 0.266, "fatigue": 0.0}
    QUESTION = (
        "how much memory pressure are you actually under right now? "
        "give me the real number, not a vibe."
    )

    @pytest.fixture(autouse=True)
    def _measured(self, monkeypatch):
        monkeypatch.setattr(cl, "measured_self_metrics", lambda: dict(self.MEASURED))

    @pytest.mark.parametrize(
        "reply",
        [
            "Memory pressure is 0.717 right now, CPU pressure 0.266, fatigue 0.",
            "Right now memory pressure reads 0.717 and cpu pressure 0.266.",
            "Memory pressure 72%.",
        ],
    )
    def test_the_gate_ships_a_correct_reading(self, reply):
        from core.conversation.response_reliability import assess_user_facing_reply

        reasons = [str(r) for r in (assess_user_facing_reply(self.QUESTION, reply).reasons or ())]
        assert "off_topic_self_reflection_reply" not in reasons, reasons

    def test_reports_measured_self_state_requires_agreement(self):
        assert cl.reports_measured_self_state("Memory pressure is 0.717.") is True
        assert cl.reports_measured_self_state("Memory pressure is 37%.") is False

    @pytest.mark.parametrize("reply", ["Memory pressure is currently 37%.", "Memory pressure is about 12%."])
    def test_an_invented_number_buys_no_substance(self, reply):
        """The escape this must not open: satisfy the gate by making one up."""
        from core.conversation.response_reliability import assess_user_facing_reply

        reasons = [str(r) for r in (assess_user_facing_reply(self.QUESTION, reply).reasons or ())]
        assert "off_topic_self_reflection_reply" in reasons
        assert cl.contradicted_self_readings(reply)

    def test_both_guards_read_one_instrument(self):
        """Agreement and contradiction are one pass, so they cannot disagree."""
        mentions = cl.self_reading_mentions("Memory pressure 0.717 and cpu pressure 90%.")
        agreed = {m for m, _c, _v, ok in mentions if ok}
        denied = {m for m, _c, _v, ok in mentions if not ok}
        assert agreed == {"memory_pressure"}
        assert denied == {"cpu_pressure"}
        assert not agreed & denied
