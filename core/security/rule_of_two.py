"""core/security/rule_of_two.py — the Rule of Two.

Clean-room adoption of Chromium's security "Rule of Two".

Chromium's rule is stated as a constraint on any piece of code, and its
force comes from being a *rule* rather than a judgement call: you may pick
at most two of

  1. handling **untrustworthy input**,
  2. written in an **unsafe implementation** (one where a bug becomes a
     capability rather than an exception),
  3. running **without a sandbox** (with the caller's full authority).

All three together is forbidden, and the fix is always to give one up:
validate the input into a trusted form, use a safe implementation, or drop
authority. The rule is valuable precisely because it does not ask anyone
to estimate exploitability — it asks three yes/no questions.

Aura's version of leg 2 is not memory safety; Python has that. It is
**capability**: code that can execute, spawn, write outside a sandbox,
mutate its own source, or call the network turns a parsing bug into an
action rather than a traceback. Aura has every ingredient — it reads web
pages, receives tool output and user files, writes its own code, spawns
subprocesses, and drives the desktop — and it had no single place where
the combination was named.

This module makes the combination declarable and checkable:

    handler = declare_handler(
        "web_fetch_summarizer",
        input_trust=InputTrust.UNTRUSTED,
        capability=Capability.PARSE_ONLY,
        isolation=Isolation.IN_PROCESS,
        owner="core/perception/web.py",
    )

Two of three is fine and passes silently. Three of three raises at
declaration — before the handler exists, not after it has run. Existing
handlers that cannot be fixed today are recorded as accepted risk *with a
reason and an owner*, which is a very different artifact from silence.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

logger = logging.getLogger("Aura.RuleOfTwo")


class InputTrust(IntEnum):
    """Where the bytes came from."""

    #: Produced by Aura or by validated internal state.
    TRUSTED = 0
    #: From the owner: intentional, but not necessarily well-formed.
    OWNER = 1
    #: Web pages, tool output, third-party files, other agents. Assume
    #: it is adversarial, because sometimes it is.
    UNTRUSTED = 2


class Capability(IntEnum):
    """What a bug in this code could become."""

    #: Reads and returns data. A bug is a wrong answer.
    PARSE_ONLY = 0
    #: Writes state or memory. A bug is corruption.
    MUTATES_STATE = 1
    #: Executes code, spawns processes, drives the desktop, calls the
    #: network, or edits Aura's own source. A bug is an action.
    EXECUTES = 2


class Isolation(IntEnum):
    """How much authority the code runs with."""

    #: Full runtime authority.
    IN_PROCESS = 0
    #: Separate process, restricted environment, no ambient credentials.
    SUBPROCESS = 1
    #: Sandboxed: no filesystem outside a scratch dir, no network, bounded
    #: CPU and memory, results validated on the way back.
    SANDBOXED = 2


@dataclass(frozen=True)
class HandlerSpec:
    name: str
    input_trust: InputTrust
    capability: Capability
    isolation: Isolation
    owner: str
    description: str = ""
    #: Set only by accept_risk(); carries who accepted it and why.
    accepted_risk: str = ""

    @property
    def legs(self) -> tuple[bool, bool, bool]:
        """(untrustworthy input, unsafe capability, no sandbox)."""
        return (
            self.input_trust is InputTrust.UNTRUSTED,
            self.capability is Capability.EXECUTES,
            self.isolation is Isolation.IN_PROCESS,
        )

    @property
    def violates(self) -> bool:
        return all(self.legs)

    @property
    def leg_count(self) -> int:
        return sum(self.legs)

    def legs_now(self) -> tuple[bool, bool, bool]:
        """The three legs as they stand for THIS turn, not on average.

        `input_trust` is declared once, at import. Two shipped surfaces declare
        TRUSTED on the grounds that their input is "model-generated" or
        "internally-formed intent" — self_modification_apply and
        desktop_automation, both of which EXECUTE and both of which run
        IN_PROCESS. That declaration is correct exactly when Aura has read
        nothing untrusted, and wrong the rest of the time, because indirect
        prompt injection does not make untrusted text act: it makes untrusted
        text persuade something trusted to act.

        So the first leg is asked of the live turn. A surface at two legs at
        rest is at three the moment a web page is in the context, which is
        precisely when the rule says to give one up.
        """
        from core.security.content_provenance import effective_input_trust

        return (
            effective_input_trust(self.input_trust) is InputTrust.UNTRUSTED,
            self.capability is Capability.EXECUTES,
            self.isolation is Isolation.IN_PROCESS,
        )

    def violates_now(self) -> bool:
        """Does this surface hold all three legs right now?"""
        return all(self.legs_now())

    def remedies(self) -> list[str]:
        """The three ways out, phrased as what to actually do."""
        return [
            (
                "validate the input into a trusted form first — parse it with a "
                "restrictive schema and pass the parsed object on, never the raw bytes"
            ),
            (
                "drop the capability — split parsing from execution so the part that "
                "touches untrusted bytes cannot act on them"
            ),
            (
                "drop the authority — run it under core/sandbox with no network, no "
                "filesystem outside a scratch dir, and validate what comes back"
            ),
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input_trust": self.input_trust.name,
            "capability": self.capability.name,
            "isolation": self.isolation.name,
            "owner": self.owner,
            "description": self.description,
            "legs": self.leg_count,
            "violates": self.violates,
            "accepted_risk": self.accepted_risk,
        }


class RuleOfTwoViolation(RuntimeError):
    """Raised at declaration when all three legs are present."""


class RuleOfTwoRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._handlers: dict[str, HandlerSpec] = {}
        self._accepted: dict[str, str] = {}
        self._declared_at: dict[str, float] = {}

    def accept_risk(self, name: str, *, reason: str, accepted_by: str) -> None:
        """Record that a known violation is being carried deliberately.

        This must be called BEFORE the handler is declared. It does not
        make the handler safe; it makes the risk an artifact with a name
        on it instead of an absence.
        """
        if not reason.strip() or not accepted_by.strip():
            raise ValueError("accepted risk needs a reason and someone accepting it")
        with self._lock:
            self._accepted[name] = f"{accepted_by}: {reason}"

    def declare(self, spec: HandlerSpec) -> HandlerSpec:
        with self._lock:
            accepted = self._accepted.get(spec.name, "")
            existing = self._handlers.get(spec.name)

        if spec.violates and not accepted:
            remedies = "\n  - ".join(spec.remedies())
            raise RuleOfTwoViolation(
                f"{spec.name!r} takes untrustworthy input, can execute or act, and "
                f"runs with full runtime authority. All three at once is forbidden.\n"
                f"Give up one:\n  - {remedies}\n"
                f"If this genuinely must ship as-is, call accept_risk({spec.name!r}, "
                "reason=..., accepted_by=...) first, so the risk is an artifact "
                "rather than an absence."
            )

        final = spec if not accepted else HandlerSpec(**{**spec.__dict__, "accepted_risk": accepted})
        with self._lock:
            if existing is not None and existing != final:
                logger.info(
                    "rule-of-two: %s re-declared with a different posture (%s → %s)",
                    spec.name,
                    existing.leg_count,
                    final.leg_count,
                )
            self._handlers[spec.name] = final
            self._declared_at.setdefault(spec.name, time.time())

        if final.violates:
            logger.warning(
                "rule-of-two violation carried as accepted risk: %s — %s",
                final.name,
                accepted,
            )
            from core.runtime.taint import TaintFlag, taint

            taint(
                TaintFlag.GATE_BYPASSED,
                f"rule-of-two violation accepted for {final.name}: {accepted}",
                subsystem="rule_of_two",
            )
        elif final.leg_count == 2:
            logger.debug(
                "rule-of-two: %s holds two legs (%s) — at the limit, not over it",
                final.name,
                final.to_dict(),
            )
        return final

    def get(self, name: str) -> HandlerSpec | None:
        with self._lock:
            return self._handlers.get(name)

    def violations(self) -> list[HandlerSpec]:
        with self._lock:
            return [h for h in self._handlers.values() if h.violates]

    def at_the_limit(self) -> list[HandlerSpec]:
        """Two legs: not forbidden, but the next change could make it three."""
        with self._lock:
            return [h for h in self._handlers.values() if h.leg_count == 2 and not h.violates]

    def report(self) -> dict[str, Any]:
        with self._lock:
            handlers = list(self._handlers.values())
        return {
            "count": len(handlers),
            "handlers": [h.to_dict() for h in sorted(handlers, key=lambda h: -h.leg_count)],
            "violations": [h.name for h in handlers if h.violates],
            "at_the_limit": [h.name for h in handlers if h.leg_count == 2 and not h.violates],
            "untrusted_input_handlers": [
                h.name for h in handlers if h.input_trust is InputTrust.UNTRUSTED
            ],
            "executing_handlers": [
                h.name for h in handlers if h.capability is Capability.EXECUTES
            ],
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._handlers.clear()
            self._accepted.clear()
            self._declared_at.clear()


_REGISTRY = RuleOfTwoRegistry()


def get_rule_of_two_registry() -> RuleOfTwoRegistry:
    return _REGISTRY


def declare_handler(
    name: str,
    *,
    input_trust: InputTrust,
    capability: Capability,
    isolation: Isolation,
    owner: str,
    description: str = "",
) -> HandlerSpec:
    """Declare a handler's security posture. Raises on three legs."""
    return _REGISTRY.declare(
        HandlerSpec(
            name=name,
            input_trust=input_trust,
            capability=capability,
            isolation=isolation,
            owner=owner,
            description=description,
        )
    )


def accept_risk(name: str, *, reason: str, accepted_by: str) -> None:
    _REGISTRY.accept_risk(name, reason=reason, accepted_by=accepted_by)


def rule_of_two_report() -> dict[str, Any]:
    return _REGISTRY.report()


def install_known_handlers() -> list[str]:
    """Declare the postures of the surfaces that already exist.

    Each of these is a real seam where outside bytes meet the runtime.
    Declaring them means a future change that adds a leg fails loudly at
    declaration rather than quietly in production.
    """
    declarations = (
        dict(
            name="web_content_ingest",
            input_trust=InputTrust.UNTRUSTED,
            capability=Capability.PARSE_ONLY,
            isolation=Isolation.IN_PROCESS,
            owner="core/runtime/network_gateway.py",
            description="fetched web pages parsed into text; never executed",
        ),
        dict(
            name="tool_result_ingest",
            input_trust=InputTrust.UNTRUSTED,
            capability=Capability.PARSE_ONLY,
            isolation=Isolation.IN_PROCESS,
            owner="core/runtime/tool_result_contracts.py",
            description="tool output validated against a contract before use",
        ),
        dict(
            name="self_modification_apply",
            input_trust=InputTrust.TRUSTED,
            capability=Capability.EXECUTES,
            isolation=Isolation.IN_PROCESS,
            owner="core/self_modification/",
            description="edits Aura's own source; input is model-generated and gated",
        ),
        dict(
            name="dynamic_code_execution",
            input_trust=InputTrust.OWNER,
            capability=Capability.EXECUTES,
            isolation=Isolation.SANDBOXED,
            owner="core/runtime/dynamic_execution_gateway.py",
            description="generated code executed under the sandbox",
        ),
        dict(
            name="desktop_automation",
            input_trust=InputTrust.TRUSTED,
            capability=Capability.EXECUTES,
            isolation=Isolation.IN_PROCESS,
            owner="core/runtime/desktop_action_gateway.py",
            description="drives the desktop from internally-formed intent only",
        ),
        dict(
            name="detached_subprocess",
            input_trust=InputTrust.TRUSTED,
            capability=Capability.EXECUTES,
            isolation=Isolation.SUBPROCESS,
            owner="core/runtime/detached_subprocess_broker.py",
            description="training and analysis workers, separate process",
        ),
        dict(
            name="user_file_ingest",
            input_trust=InputTrust.OWNER,
            capability=Capability.PARSE_ONLY,
            isolation=Isolation.IN_PROCESS,
            owner="core/runtime/file_read_gateway.py",
            description="owner-supplied files read as data",
        ),
    )
    declared: list[str] = []
    for declaration in declarations:
        try:
            declare_handler(**declaration)  # type: ignore[arg-type]
            declared.append(str(declaration["name"]))
        except RuleOfTwoViolation as exc:
            # A shipped surface that violates is exactly what this is for:
            # surface it loudly rather than skip it.
            logger.error("rule-of-two: %s", exc)
    return declared


def reset_rule_of_two_for_test() -> None:
    _REGISTRY.reset_for_test()


__all__ = [
    "Capability",
    "HandlerSpec",
    "InputTrust",
    "Isolation",
    "RuleOfTwoRegistry",
    "RuleOfTwoViolation",
    "accept_risk",
    "declare_handler",
    "get_rule_of_two_registry",
    "install_known_handlers",
    "reset_rule_of_two_for_test",
    "rule_of_two_report",
]
