"""Every writing rule fires on the example that motivated it, and not on prose.

A rule that cannot match is worse than a missing rule: the gate reports green and
the pattern it names spreads anyway. The triad rule shipped documented in
docs/WRITING_RULES.md and unenforced in the linter, and nothing noticed until
somebody counted the rules on each side.

So each entry below is a CAPTURED string. The positives come from the articles
the ruleset was built from and from prose actually found in this tree; the
negatives are real sentences from this repo that a sloppier regex ate.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "lint_ai_writing", ROOT / "tools" / "lint_ai_writing.py"
)
assert _spec and _spec.loader
lint = importlib.util.module_from_spec(_spec)
sys.modules["lint_ai_writing"] = lint
_spec.loader.exec_module(lint)


def _hits(text: str) -> set[str]:
    """Rule names that fire on a chunk of prose.

    Both extractors blank code spans and quoted spans before the rules run, so
    this does too. Testing the raw string would let a rule pass here and behave
    differently on a real file, which is the whole failure this suite exists for.
    """
    text = lint.INLINE_CODE.sub(lambda m: " " * len(m.group(0)), text)
    fired = set()
    for name, rx, _why, keep in lint.RULES:
        for m in rx.finditer(text):
            if keep is None or keep(m):
                fired.add(name)
    return fired


#: (rule name, prose that must trip it). One per rule, no exceptions: a rule
#: added without a line here is a rule nobody has seen work.
POSITIVES: list[tuple[str, str]] = [
    ("negation-flip", "That's not compliance. That's stalling."),
    ("negation-flip-inline", "This is not just a data structure, it's a claim."),
    ("negation-flip-inline", "The gate is not only slow but also wrong."),
    ("negation-flip-inline", "It runs not on the kernel, but rather on the host."),
    ("stapled-fragments", "We shipped it. Fast. Simple. Nothing else changed."),
    ("twin-images", "The verifier is less a hammer, more a scalpel."),
    ("reflexive-triad", "The new lane is faster, cheaper, and smarter."),
    ("reflexive-triad", "It felt refreshing, rejuvenating, invigorating."),
    ("self-applause", "The loop retries twice. And that matters."),
    ("borrowed-analogy", "It's the Excel of AI agents."),
    ("throat-clearing", "Here's the thing. The cache never hit."),
    ("hedged-range", "The shader compile takes 5 to 10 minutes."),
    ("recap-ending", "In short, the ratchet only goes down."),
    ("recap-ending", "At the end of the day, the cache never warmed."),
    ("participial-tail", "The system retries twice, ensuring reliability."),
    ("participial-tail", "Latency fell by half, underscoring the win."),
    ("disclaimer-hedge", "It is important to note that the gate is advisory."),
    ("vague-attribution", "Many argue that recurrence damages reasoning."),
    ("vague-attribution", "Studies show the cache never warms."),
    ("rhetorical-question", "The result? Nothing rendered at all."),
    ("false-collaboration", "Let's dive into the scheduler."),
    ("cliche-opener", "In today's fast-paced world, agents must adapt."),
    ("inflated-diction", "We utilize the gateway to write state."),
    ("inflated-diction", "Restart the organ in order to clear the fence."),
    ("chatbot-remnant", "As a large language model, I cannot verify that."),
    ("chatbot-remnant", "Hope this helps! Let me know if you would like more."),
]

#: Prose from this repository that earlier drafts of these rules flagged. Each
#: line is a real enumeration, a real measurement, or a real technical word.
NEGATIVES: list[str] = [
    # Three genuinely distinct states, not a rhythm.
    "The organ reports warming, recovering, or handshaking.",
    "The verdict covers containment, integrity, and authority.",
    "Findings are capped, expired, or cleared on the next tick.",
    "The prover handles logical, mathematical, and physical claims.",
    "Memory grows across personality, phenomenology, and goal.",
    # `underscore` is the character in a Python tree, never the verb.
    "The name must be letters, digits, dot, dash, or underscore.",
    "Public name; the underscore version is kept as an alias.",
    # A harness is a fixture here.
    "The test harness boots a stub container.",
    # An ISO date followed by a unit word is not a range. The hedged-range
    # rule read "2026-04-27 second reduction" as "04-27 second" and flagged a
    # dated engineering note as an unmeasured estimate. Real ranges — "took
    # 4-27 seconds", "a 58-82s first token" — still match.
    "Alpha fell to 5 on 2026-04-27, the second reduction that month.",
    "The 2026-04-27 second pass shipped without the steering vectors.",
    "Recorded 2026-01-30 days after the first attempt.",
    # Quoted material keeps its wording, per docs/WRITING_RULES.md.
    'Persona displacement: the reply speaking "as a large language model".',
    # Correlatives and comparisons are arguments, not twin images.
    "Ordered by typical availability (less popular = more capacity).",
    "The less was known about a moment, the more likely it was to train.",
    "A genome that keeps the system alive on less energy is more fit.",
    # A word shouted for emphasis is not a borrowed product name.
    "This is different from sustained high — it's the SURPRISE of load rising.",
    # `prior` the noun. Rewriting this to "a reasonable before" was a real edit
    # this suite now prevents.
    "The tracker calibrates over time, but we need a reasonable prior to start with.",
    "The flat prior to start from is uniform over all outcomes.",
    # `short` the adjective, not the discourse marker.
    "Wait in short slices so dead workers fail fast.",
    "Processes queued writes in short micro-batches with retry backoff.",
    "Check for deep keywords even in short messages.",
    # "worth reading" about a statistic, not about the prose.
    "How many episodes a rule must fire on before its precision is worth reading.",
    # Ordinary prose that must stay quiet.
    "The gate refuses rather than gambles when free memory is low.",
    "Set AURA_LOG_DIR so tests never write into the live log directory.",
]


@pytest.mark.parametrize("rule,text", POSITIVES, ids=[f"{r}:{t[:28]}" for r, t in POSITIVES])
def test_rule_fires_on_its_example(rule: str, text: str) -> None:
    assert rule in _hits(text), f"{rule} did not fire on its own example: {text!r}"


@pytest.mark.parametrize("text", NEGATIVES, ids=[t[:36] for t in NEGATIVES])
def test_clean_prose_stays_clean(text: str) -> None:
    assert _hits(text) == set(), f"false positive on repo prose: {text!r}"


def test_every_rule_has_a_positive_example() -> None:
    """No rule ships without a line in POSITIVES proving it can match."""
    covered = {rule for rule, _ in POSITIVES}
    declared = {name for name, _, _, _ in lint.RULES}
    assert declared == covered, (
        f"rules with no worked example: {sorted(declared - covered)}; "
        f"examples for rules that no longer exist: {sorted(covered - declared)}"
    )


def test_docstring_findings_report_the_docstring_line(tmp_path: Path) -> None:
    """A finding points at the prose, not at the `def` above it."""
    src = tmp_path / "sample.py"
    src.write_text(
        "def probe():\n"
        '    """Summary line.\n'
        "\n"
        "    We utilize the gateway here.\n"
        '    """\n'
        "    return 1\n",
        encoding="utf-8",
    )
    lines = dict(lint.python_prose_lines(src.read_text(encoding="utf-8")))
    assert "utilize" in lines[4], f"expected the utilize line at 4, got {lines}"


def test_code_is_never_linted_as_prose(tmp_path: Path) -> None:
    """Identifiers and string literals are code; only docstrings and comments count."""
    src = tmp_path / "sample.py"
    src.write_text(
        "BANNER = 'In short, we utilize everything.'\n"
        "def in_order_to_flush():\n"
        "    return 'let us dive into it'\n",
        encoding="utf-8",
    )
    assert lint.python_prose_lines(src.read_text(encoding="utf-8")) == []


def test_directive_comments_are_not_prose(tmp_path: Path) -> None:
    src = tmp_path / "sample.py"
    src.write_text("x = 1  # noqa: E501\ny = 2  # type: ignore\n", encoding="utf-8")
    assert lint.python_prose_lines(src.read_text(encoding="utf-8")) == []
