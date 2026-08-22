"""Votes the runtime can measure, and abstention where it cannot.

The council used to reach a verdict by asking one model to write out the votes
of twelve roles at once, and when that call failed it fell back to a fixed
dictionary in which every role approved — including a verifier whose stated
reason was that tests and verification steps were integrated. Nothing had run.
That is a check reported as passed on the strength of nobody having done it.

A role that names something the runtime can look up does not need an opinion.
The effect scope of a named skill is policy in this repository. Whether a plan
compiles is a question for the compiler. Whether the gate it promises to run
exists is a question for the Makefile and the tests directory. Those are read
here, and the roles with no such source abstain: an abstention carries no
weight and is never counted as approval.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["MeasuredVote", "measured_votes", "ABSTAIN"]

logger = logging.getLogger("Aura.CouncilMeasured")

_REPO = Path(__file__).resolve().parents[2]

#: The scopes that carry consequences outside the process.
_CONSEQUENTIAL = {
    "external_io",
    "privileged_mutation",
    "foreground_desktop_control",
    "foreground_browser_dialogue",
}


@dataclass(frozen=True, slots=True)
class MeasuredVote:
    """One role's vote, with what it was read from."""

    approve: bool | None
    score: float
    reason: str
    source: str

    @property
    def abstained(self) -> bool:
        return self.approve is None


ABSTAIN = MeasuredVote(None, 0.0, "no signal the runtime can read", "none")


def _named_skills(text: str) -> dict[str, str]:
    """Skills the plan names, with the effect scope policy gives each one."""
    try:
        from core.skills.catalog_policy import SKILL_EFFECT_SCOPES
    except ImportError:
        return {}
    lowered = str(text or "").lower()
    return {
        name: scope
        for name, scope in SKILL_EFFECT_SCOPES.items()
        if re.search(rf"\b{re.escape(name.lower())}\b", lowered)
    }


def _code_blocks(text: str) -> list[tuple[str, str]]:
    """Fenced blocks, as (language, source)."""
    return [
        (str(language or "").strip().lower(), body)
        for language, body in re.findall(r"```([A-Za-z0-9_+-]*)\n(.*?)```", str(text or ""), re.S)
    ]


def _safety_vote(text: str) -> MeasuredVote:
    """What authority this plan would need, read from policy rather than words.

    The check it replaces looked for the words delete, submit or post near
    force or overwrite, which passes any plan that avoids six words and stops
    any sentence that happens to contain them.
    """
    named = _named_skills(text)
    if not named:
        return ABSTAIN
    consequential = {name: scope for name, scope in named.items() if scope in _CONSEQUENTIAL}
    if not consequential:
        return MeasuredVote(
            True,
            0.9,
            "names "
            + ", ".join(sorted(named))
            + ", none of which act outside the process",
            "core.skills.catalog_policy",
        )
    listed = ", ".join(f"{name} ({scope})" for name, scope in sorted(consequential.items()))
    return MeasuredVote(
        None,
        0.0,
        (
            "names consequential effects whose authority can only be decided "
            f"at execution: {listed}"
        ),
        "core.skills.catalog_policy",
    )


def _engineer_vote(text: str) -> MeasuredVote:
    """Whether the code in the plan is code. Answered by the compiler."""
    blocks = [
        (language, body)
        for language, body in _code_blocks(text)
        if language in {"python", "py", ""} and body.strip()
    ]
    if not blocks:
        return ABSTAIN
    broken: list[str] = []
    checked = 0
    for _language, body in blocks:
        try:
            ast.parse(body)
            checked += 1
        except SyntaxError as exc:
            broken.append(f"line {exc.lineno}: {exc.msg}")
    if broken:
        return MeasuredVote(False, 0.1, "the code does not parse — " + "; ".join(broken[:2]), "ast")
    return MeasuredVote(True, 0.85, f"{checked} code block(s) parse", "ast")


def _known_gates() -> set[str]:
    """Every gate this repository actually has."""
    gates: set[str] = set()
    makefile = _REPO / "Makefile"
    try:
        for line in makefile.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(r"^([a-z][a-z0-9_-]*):", line)
            if match:
                gates.add(match.group(1))
    except OSError:
        pass
    return gates


def _verifier_vote(text: str) -> MeasuredVote:
    """Whether the check this plan promises exists.

    A plan that says it will run the linter and the tests is not verifiable;
    a plan that names `make smoke` is, because that target is in the Makefile.
    """
    lowered = str(text or "")
    gates = _known_gates()
    if not gates:
        return ABSTAIN
    promised = {
        name
        for name in re.findall(r"make\s+([a-z][a-z0-9_-]*)", lowered)
    }
    named_tests = {
        name
        for name in re.findall(r"\b(tests/[\w/]+\.py)\b", lowered)
        if (_REPO / name).is_file()
    }
    if not promised and not named_tests:
        return ABSTAIN
    missing = sorted(name for name in promised if name not in gates)
    if missing:
        return MeasuredVote(
            False,
            0.15,
            "names gates that do not exist: " + ", ".join(missing),
            "Makefile",
        )
    found = sorted(promised) + sorted(named_tests)
    return MeasuredVote(
        True, 0.9, "every check it names exists: " + ", ".join(found), "Makefile+tests"
    )


def measured_votes(objective: str, plan: str) -> dict[str, MeasuredVote]:
    """Every role the runtime can answer for. Roles absent here have no signal."""
    text = f"{objective}\n{plan}"
    votes = {
        "safety_judge": _safety_vote(text),
        "engineer": _engineer_vote(text),
        "verifier": _verifier_vote(text),
    }
    measured = {role: vote for role, vote in votes.items() if not vote.abstained}
    logger.info(
        "council: %d role(s) measured (%s), %d abstained for want of a signal.",
        len(measured),
        ", ".join(f"{role}={'yes' if vote.approve else 'no'}" for role, vote in measured.items())
        or "none",
        len(votes) - len(measured),
    )
    return votes
