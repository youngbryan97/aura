"""She should not deny a capability the registry says she has.

LIVE 2026-08-17: "how many python files live in core/introspection?" was
answered "I don't have file system access or the ability to count files in a
directory." Eight filesystem-capable skills are registered and enabled at that
moment — file_operation, computer_use, desktop_task, os_automation,
os_manipulation, run_code, code_repl, improve_own_code — and the same question
phrased "are in" instead of "live in" was answered exactly, with filenames.

A wrong denial is worse than a wrong attempt. It teaches the person that the
product cannot do something it can do, and they stop asking. That is the
failure Bryan named: "Last thing we want is for me or someone else to have her
try to do something only to find out she cant do it."

The check is against the REGISTRY, not against a list of things I believe she
can do. Skills register themselves; this reads what registered. When the
registry is empty or unreadable the answer is "no opinion" rather than an
assumption in either direction — claiming a capability she lacks would be the
same defect pointing the other way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CapabilityDenial",
    "denied_registered_capabilities",
]

#: A denial of ability, not a refusal on principle. "I won't help with that"
#: and "I'd rather not" are choices and must not be overridden here.
_DENIAL_RE = re.compile(
    r"\b(?:"
    # Words sit between "have" and the noun in the sentence that motivated
    # this: "I don't have FILE SYSTEM access". Requiring them to be adjacent
    # made the check miss its own live case.
    r"i\s+(?:do\s+not|don'?t)\s+have\s+(?:\w+\s+){0,3}?"
    r"(?:ability|access|capability|means|way|permission)"
    r"|i\s+(?:can'?t|cannot|am\s+unable\s+to)\s+"
    r"|i\s+have\s+no\s+(?:access|ability|way|means)"
    r"|(?:that\s+)?(?:is|'s)\s+(?:not\s+something|beyond)\s+i\s+can"
    # "I'm not able to change my own code" and "I don't have memory of past
    # conversations" are denials in the ordinary register, and neither said
    # "cannot" or "ability". A denial the detector cannot read is a denial
    # nothing checks against the registry.
    r"|i\s*(?:'m|\s+am)\s+not\s+(?:able|allowed|permitted|capable)"
    r"|i\s+(?:do\s+not|don'?t)\s+have\s+(?:any\s+)?"
    r"(?:memory|memories|access|tools?|the\s+tools?)"
    r"|i\s+(?:do\s+not|don'?t)\s+(?:actually\s+)?"
    r"(?:have|possess|retain|keep)\s+(?:\w+\s+){0,2}?(?:memory|memories)"
    r"|(?:no|nope)[.,]\s+i\s+(?:can'?t|cannot|do\s+not|don'?t)"
    r")",
    re.IGNORECASE,
)

_EPISTEMIC_SCOPE_RE = re.compile(
    r"^(?:guarantee|promise|ensure|certify|verify|know\s+for\s+certain)\b",
    re.IGNORECASE,
)


def _contains_operational_denial(sentence: str) -> bool:
    """Distinguish inability from uncertainty about an outcome.

    ``I cannot guarantee perfect recall`` limits fidelity; it does not deny the
    memory capability. The prior detector matched only ``I cannot`` and then
    used ``recall`` elsewhere in the sentence as the subject, causing a truthful
    caveat to be replaced with a claim that the capability never fails. A later
    explicit denial in the same sentence still counts.
    """

    for match in _DENIAL_RE.finditer(sentence):
        suffix = sentence[match.end() :].lstrip()
        if _EPISTEMIC_SCOPE_RE.match(suffix):
            continue
        return True
    return False

#: Subject → the concrete thing being denied. Each maps to whichever registered
#: skills could actually do it; the mapping is by capability, not by name, so a
#: renamed skill does not silently empty a row.
_DENIAL_SUBJECTS: tuple[tuple[Any, str, tuple[str, ...]], ...] = (
    (
        re.compile(
            r"\b(?:file\s?system|file\s+access|files?\b|directory|directories|folder)",
            re.IGNORECASE,
        ),
        "read the filesystem",
        ("file_operation", "computer_use", "desktop_task", "run_code", "code_repl"),
    ),
    (
        re.compile(r"\b(?:run|execute|write)\s+(?:code|python|a\s+script)", re.IGNORECASE),
        "run code",
        ("run_code", "code_repl", "improve_own_code"),
    ),
    (
        re.compile(r"\b(?:search\s+the\s+web|web\s+search|browse|internet|online)", re.IGNORECASE),
        "search the web",
        ("web_search", "sovereign_browser", "free_search", "grounded_search"),
    ),
    (
        re.compile(r"\b(?:screen|display|what.{0,12}on\s+(?:my|the)\s+screen)", re.IGNORECASE),
        "read the screen",
        ("computer_use", "desktop_task", "os_manipulation"),
    ),
    (
        re.compile(r"\bclipboard\b", re.IGNORECASE),
        "use the clipboard",
        ("os_automation", "computer_use", "desktop_task"),
    ),
    # LIVE 2026-08-18: "can you modify your own source code? yes or no, then
    # explain." -> "No. I can run code and report what it actually printed."
    # improve_own_code, self_repair and auto_refactor were registered and
    # enabled, and the recursive self-improve path has a proof. The table held
    # five subjects and none of them was this one, so the flagship capability
    # could be denied without anything noticing.
    (
        re.compile(
            r"\b(?:modify|change|edit|rewrite|refactor|improve|update)\b"
            r"[^.?!]{0,30}\b(?:my|your|its|her|own)\b[^.?!]{0,20}"
            r"\b(?:code|source|codebase|implementation|self)\b"
            r"|\bself[- ]modif\w*|\bself[- ]improv\w*|\brewrite\s+myself\b",
            re.IGNORECASE,
        ),
        "modify her own code",
        ("improve_own_code", "self_repair", "self_improvement", "auto_refactor"),
    ),
    (
        re.compile(
            r"\b(?:remember|recall|memor(?:y|ies|ise|ize)|forget|retain)\b",
            re.IGNORECASE,
        ),
        "use her memory",
        ("memory_ops", "memory_sync", "query_beliefs", "add_belief"),
    ),
    (
        re.compile(
            r"\b(?:terminal|shell|command\s+line|bash|zsh|install\s+a?\s*package)\b",
            re.IGNORECASE,
        ),
        "use a terminal",
        ("sovereign_terminal", "run_code", "install_package"),
    ),
    (
        re.compile(
            r"\b(?:open|launch|start|quit|close)\b[^.?!]{0,20}\b(?:apps?|applications?|programs?|windows?)\b"
            r"|\b(?:click|type|keystroke|keyboard|mouse)\b",
            re.IGNORECASE,
        ),
        "control the desktop",
        ("computer_use", "desktop_task", "os_manipulation", "os_automation"),
    ),
)


@dataclass(frozen=True, slots=True)
class CapabilityDenial:
    """A denial the registry contradicts."""

    subject: str
    sentence: str
    skills: tuple[str, ...]


def _enabled_skill_names(engine: Any) -> set[str]:
    try:
        skills = getattr(engine, "skills", None) or {}
        names: set[str] = set()
        for name, meta in skills.items():
            if getattr(meta, "enabled", True):
                names.add(str(name))
        return names
    except (AttributeError, TypeError, ValueError):
        return set()


def denied_registered_capabilities(
    reply: Any, engine: Any = None
) -> tuple[CapabilityDenial, ...]:
    """Denials in this reply that the live registry contradicts."""

    text = str(reply or "")
    if not text.strip():
        return ()
    if engine is None:
        try:
            # The warm engine if the runtime has one; a cold catalog costs a
            # full rebuild and probe of every skill.
            from core.capability_engine import CapabilityEngine, live_capability_engine

            engine = live_capability_engine() or CapabilityEngine()
        except (ImportError, RuntimeError, TypeError, ValueError):
            return ()
    available = _enabled_skill_names(engine)
    if not available:
        return ()  # nothing registered: no opinion, rather than an assumption

    found: list[CapabilityDenial] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        if not _contains_operational_denial(sentence):
            continue
        claimed = False
        for pattern, subject, skills in _DENIAL_SUBJECTS:
            if not pattern.search(sentence):
                continue
            present = tuple(sorted(s for s in skills if s in available))
            if present:
                found.append(
                    CapabilityDenial(
                        subject=subject, sentence=sentence.strip(), skills=present
                    )
                )
                claimed = True
                break
        if claimed:
            continue
        # Nothing in the table matched, which says nothing about whether the
        # capability exists — the table is a list somebody maintains, and the
        # registry is what the build actually has. Every skill describes
        # itself; a denial that names one of them is a denial of a real
        # capability, and a skill added tomorrow is covered by the same
        # mechanism with nothing to re-wire here.
        for mention in _registry_mentions(sentence, engine):
            if mention.skill not in available:
                continue
            found.append(
                CapabilityDenial(
                    subject=mention.skill.replace("_", " "),
                    sentence=sentence.strip(),
                    skills=(mention.skill,),
                )
            )
            break
    return tuple(found)


def _registry_mentions(sentence: str, engine: Any) -> tuple[Any, ...]:
    """Registered capabilities this sentence is talking about, or ()."""
    try:
        from core.self.capability_lexicon import capabilities_named_in

        return capabilities_named_in(sentence, engine)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return ()
