"""Asked about the present, she must be given one.

Three failures from the same live conversation (2026-07-27), and one cause.

    "The sun's up but I'm not sure it will be warm today — there are clouds
     gathering in the east."                                     — at 00:30 AM

    "I processed a user request to summarize a 45-page PDF on neuromorphic
     computing."                    — asked for a real event from her telemetry

    web_search("...your current uptime and how much memory are you holding")
    -> headless browser -> windowsforum.com -> 302 seconds -> no answer
                              — asked to read her uptime from her own runtime

None of these is a lie she chose. The date and hour appeared nowhere in the
prompt path; no channel carried her recent activity; and a question about her
own uptime was classified as a live factual lookup, which is what the web is
for. Given no present, a language model writes a plausible one.

So: give her the clock (present_moment), give her the instruments
(self_state_report), and stop sending introspection to the internet
(self_state_intent). The honesty clause is the small part — grounding is the fix.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from core.brain.present_moment import present_moment_block
from core.brain.self_state_report import SELF_STATE_HEADER, runtime_self_report
from core.runtime.self_state_intent import asks_about_own_runtime

GATE = Path("core/brain/inference_gate.py")
CONTRACT = Path("core/phases/response_contract.py")


# ── The clock she never had ────────────────────────────────────────────────

def test_the_block_states_the_actual_date_and_hour() -> None:
    block = present_moment_block(now=datetime(2026, 7, 27, 0, 30))
    assert "Monday 27 July 2026, 00:30" in block


@pytest.mark.parametrize(
    ("hour", "expected"),
    [(0, "middle of the night"), (6, "early morning"), (10, "morning"),
     (14, "afternoon"), (19, "evening"), (23, "late evening")],
)
def test_part_of_day_tracks_the_clock(hour: int, expected: str) -> None:
    block = present_moment_block(now=datetime(2026, 7, 27, hour, 0))
    assert expected in block


def test_the_clock_is_never_dressed_up_as_a_window() -> None:
    """A clock says what time it is; it does not say whether it is sunny."""
    block = present_moment_block(now=datetime(2026, 7, 27, 12, 0))
    assert "by the clock" in block
    assert "no window, camera, thermometer or weather feed" in block


def test_it_is_small_enough_for_every_turn() -> None:
    block = present_moment_block(now=datetime(2026, 7, 27, 0, 30))
    assert len(block) < 900, "grounding that costs a turn its budget will get cut"


# ── The instruments ────────────────────────────────────────────────────────

def test_the_report_never_invents_a_reading() -> None:
    """Unavailable lines are omitted; the heading never appears alone."""
    report = runtime_self_report()
    if report:
        assert report.startswith(SELF_STATE_HEADER)
        assert report.count("\n") >= 2, "a bare heading invites her to fill it in"


def test_the_report_warns_that_rss_understates_her() -> None:
    """On Apple Silicon the model's weights are invisible to RSS.

    Reporting only the resident figure would be true and misleading — this is
    the same measurement trap that made a 16GB worker look like it belonged to
    someone else.
    """
    report = runtime_self_report()
    if "resident" in report:
        assert "wired GPU memory" in report


# ── Introspection is not a web query ───────────────────────────────────────

@pytest.mark.parametrize(
    "question",
    [
        "What's your current uptime and how much memory are you holding? "
        "Read it from your own runtime, don't estimate.",
        "How long have you been running?",
        "Which model are you running?",
        "What happened in your runtime in the last hour?",
        "show me your recent errors",
    ],
)
def test_questions_about_her_machine_are_introspection(question: str) -> None:
    assert asks_about_own_runtime(question)


@pytest.mark.parametrize(
    "question",
    [
        "How are you feeling today?",
        "What's the weather in Paris?",
        "Who won the most recent F1 championship?",
        "Can you look up your memory of our last chat?",
        "Tell me about your favourite book.",
    ],
)
def test_ordinary_questions_are_left_alone(question: str) -> None:
    """Over-claiming introspection would starve real lookups of the web."""
    assert not asks_about_own_runtime(question)


def test_the_contract_stops_searching_for_her_own_readings() -> None:
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    contract = build_response_contract(
        AuraState.default(),
        "What's your current uptime and how much memory are you holding? "
        "Read it from your own runtime, don't estimate.",
        is_user_facing=True,
    )
    assert not contract.requires_search, "her uptime is not on the internet"


def test_an_ordinary_lookup_still_searches() -> None:
    """The suppression must be narrow or she stops grounding anything."""
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    contract = build_response_contract(
        AuraState.default(),
        "Search the web and tell me who won the most recent F1 world championship.",
        is_user_facing=True,
    )
    assert contract.requires_search


# ── Both blocks reach the prompt, and survive it ───────────────────────────

def test_the_gate_injects_both_blocks() -> None:
    src = GATE.read_text(encoding="utf-8")
    assert "from core.brain.present_moment import present_moment_block" in src
    assert "from core.brain.self_state_report import runtime_self_report" in src
    # The gate narrowed this to asks_about_own_capabilities: this path only
    # ADDS an instrument reading, so a false positive costs a few prompt lines,
    # while asks_about_own_runtime also suppresses web search, where a false
    # positive costs the lookup the person asked for. The predicate changed on
    # purpose; asserting the old name pinned the test to a decision that was
    # deliberately reversed.
    assert "if asks_about_own_capabilities(visible_user_prompt):" in src


def test_grounding_survives_prompt_compaction() -> None:
    """Grounding is worth most exactly when the prompt is tight.

    Asked of the behaviour rather than of the source. These two assertions
    used to grep inference_gate.py for a literal, and the literal was
    refactored away — leaving a test that failed while the thing it was
    protecting still worked, which teaches the reader to distrust the suite.
    """
    from core.brain.inference_gate import InferenceGate

    kept = set(InferenceGate.CRITICAL_FOREGROUND_HEADERS)
    assert "## PRESENT MOMENT" in kept
    assert "## YOUR OWN INSTRUMENTS" in kept


def test_grounding_sorts_after_the_cacheable_prefix() -> None:
    """A per-minute timestamp early in the prompt would bust the KV cache.

    Emission is sorted by how turn-volatile each section is, so what changes
    every turn lands last and the stable prefix stays cacheable. What matters
    is that the clock and the instrument readings are ranked MORE volatile
    than the identity that never moves — not how that ranking is spelled.
    """
    from core.brain.inference_gate import InferenceGate as Gate

    def rank(header: str) -> int:
        return Gate._foreground_section_volatility(f"{header}\nsomething")

    assert rank("## PRESENT MOMENT") > rank("## IDENTITY")
    assert rank("## YOUR OWN INSTRUMENTS") > rank("## IDENTITY")


def test_and_the_order_it_produces_puts_them_last() -> None:
    """The property that actually protects the cache, checked end to end."""
    from core.brain.inference_gate import InferenceGate as Gate

    sections = [
        "## PRESENT MOMENT\nit is 00:53",
        "## IDENTITY\nwho she is",
        "## YOUR OWN INSTRUMENTS\nreadings",
    ]
    emitted = sorted(sections, key=Gate._foreground_section_volatility)
    assert emitted[0].startswith("## IDENTITY")
    assert {emitted[1][:5], emitted[2][:5]} == {"## PR", "## YO"}


# ── Both prompt builders, not just the one I found first ───────────────────

ENGINE = Path("core/brain/cognitive_engine.py")


def test_the_desktop_conversation_lane_is_grounded_too() -> None:
    """There are two system-prompt builders, and people only meet one of them.

    The first fix wired grounding into inference_gate. After it landed and the
    runtime restarted, "what's it actually like in there right now?" still
    answered "the sun's up ... clouds gathering in the east" at 00:53 — word for
    word the same sentence as before. The desktop conversation lane
    (mode=compact_foreground_prebuilt, origin=desktop_quick_user) assembles its
    own prompt and never saw it. That lane is the one every real conversation
    goes through.
    """
    src = ENGINE.read_text(encoding="utf-8")
    assert "from core.brain.present_moment import present_moment_block" in src
    # asks_about_own_runtime -> asks_about_own_capabilities, deliberately: this
    # path only ADDS a reading, so a false positive costs a few prompt lines,
    # while asks_about_own_runtime also suppresses web search, where a false
    # positive costs the lookup the person asked for.
    assert "asks_about_own_capabilities" in src
    assert "from core.brain.self_state_report import runtime_self_report" in src


def test_grounding_is_added_before_the_style_contract() -> None:
    """Order matters only in that both must survive to the same prompt."""
    src = ENGINE.read_text(encoding="utf-8")
    # The style contract is appended as a PERSONA CONTRACT block now; the
    # assertion pinned an exact f-string that no longer exists, so it failed on
    # a rename rather than on a regression. What matters is unchanged: the
    # grounding is assembled before the contract is appended.
    assert src.index("present_moment_block()") < src.index(
        'system_prompt = f"{system_prompt}\\n[PERSONA CONTRACT]'
    )


def test_the_terse_inventory_contract_is_left_alone() -> None:
    """That contract requires exactly four sentences under 80 words."""
    src = ENGINE.read_text(encoding="utf-8")
    block = src[src.index("if not capability_inventory_contract:") :]
    assert "present_moment_block" in block[:1200]


# ── "What's your uptime?" must never come back without a number ───────────

def test_uptime_is_always_readable() -> None:
    """The orchestrator lookup was the only source, and it returned nothing.

    Asked for uptime and memory read from her own runtime, the instruments
    block arrived with no uptime line at all — so the honest answer was
    unavailable to her, which is the precondition for inventing one. The
    process knows when it started, always.
    """
    from core.brain.self_state_report import _uptime_line

    line = _uptime_line()
    assert line.startswith("- Uptime:")
    assert any(char.isdigit() for char in line)


def test_the_gpu_claim_is_only_made_when_the_numbers_support_it() -> None:
    """RSS understates her — but only where it actually does."""
    from core.brain.self_state_report import _memory_lines

    lines = _memory_lines()
    gpu = [line for line in lines if "GPU memory" in line]
    assert gpu, "the accelerator must always be accounted for, even as an absence"
    if "bulk of what you are actually holding" in gpu[0]:
        assert "active" in gpu[0]


def test_the_instrument_panel_carries_the_cognitive_cycle_count(monkeypatch):
    """A number she can read must be in front of her, or she invents one.

    Measured live. Asked "read your own runtime and tell me three real numbers —
    uptime, memory, and how many cognitive cycles you've run — read them, don't
    estimate", she got uptime and memory right off this panel and then said:

        "Cognitive cycles since last awakening: I can't read this directly,
         but it's more than a few billion"

    The true figure was 3,502, and it sits in her own health payload. The panel
    had no cycle line, and the instruction above it tells her not to supplement
    what is missing — so the absence produced both a false claim about her own
    self-access and a guess wrong by six orders of magnitude.
    """
    from types import SimpleNamespace

    import core.brain.self_state_report as self_state_report
    from core.runtime import service_registry

    original = service_registry.get_runtime_service

    def _with_orchestrator(name, default=None):
        if name == "orchestrator":
            return SimpleNamespace(status=SimpleNamespace(cycle_count=3502))
        if name == "episodic_memory":
            return SimpleNamespace(episode_count=1229)
        return original(name, default=default)

    monkeypatch.setattr(service_registry, "get_runtime_service", _with_orchestrator)

    line = self_state_report._cognition_line()
    assert "3,502" in line, f"the readable cycle count must be shown: {line!r}"
    assert "1,229" in line

    report = self_state_report.runtime_self_report()
    assert "3,502" in report, "the panel must carry the cycle count she was asked for"


def test_an_unreadable_cycle_count_says_so_instead_of_going_silent(monkeypatch):
    """Silence is what she fills in. A stated absence is an answer she can give."""
    from core.runtime import service_registry

    import core.brain.self_state_report as self_state_report

    monkeypatch.setattr(
        service_registry, "get_runtime_service", lambda name, default=None: None
    )

    line = self_state_report._cognition_line()
    assert line, "an unreadable instrument must still produce a line"
    lowered = line.lower()
    assert "not readable" in lowered
    assert "do not estimate" in lowered, (
        "the line has to forbid the guess that actually happened"
    )
