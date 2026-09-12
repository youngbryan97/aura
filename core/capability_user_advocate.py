"""Whether the person would want this done, asked before it is done.

Authority to run a tool and the person wanting it run are different questions,
and the second one is this. Irreversibility, whether the request actually came
from them, what benefit the action claims, and the standing authority a
confirmed answer leaves behind. Every refusal is a denial payload naming the
chain it broke, because "not authorised" with nothing after it is a dead end
for whoever reads the log.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation only; that module imports this one
    from .capability_engine import SkillMetadata

import os
from typing import Any

from core.executive.execution_policy import (  # noqa: E402
    canonical_authority_arguments,
    classify_execution_risk,
)
from core.governance.capability_chain import (  # noqa: E402
    CapabilityDenial,
    CapabilityViolation,
    capability_enforcement_mode,
    capability_from_context,
    enforce_capability,
    get_capability_verifier,
)
from core.runtime.errors import record_degradation


class _AsksWhetherThePersonWouldWantThis:
    """Lifted whole from CapabilityEngine; see capability_engine.py."""

    def _capability_chain_denial(
        self,
        ctx: dict[str, Any],
        skill_name: str,
        params: Any,
        constitutional_runtime_live: bool,
    ) -> dict[str, Any] | None:
        """Authenticate the Will's grant at the moment of execution.

        This is the point where the constitutional chain actually closes: the
        capability presented here must be a signature over *this* decision, for
        *this* action, with *these* parameters — not merely a token that exists.

        Returns a denial payload to abort the execution, or None to proceed.
        Enforcement follows ``AURA_CAPABILITY_ENFORCEMENT``:

            strict  refuse execution without a verified capability (default once
                    the runtime is constitutionally live)
            warn    record the violation and proceed — migration only
            off     skip entirely — for tests and pre-runtime boot paths

        The default is strict whenever the constitutional runtime is live,
        because a governance check that can be quietly skipped is the failure
        this whole change exists to remove.
        """
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .capability_engine import (
            _record_capability_degradation,
        )

        mode = capability_enforcement_mode(default="strict" if constitutional_runtime_live else "off")
        if mode == "off":
            return None

        try:
            enforce_capability(
                ctx,
                sink=f"capability_engine.execute_skill:{skill_name}",
                # String rather than ActionDomain: importing core.will here would
                # cycle (the Will imports the engine's container). The verifier
                # normalizes enum and str identically.
                domain="tool_execution",
                action=skill_name,
                payload=canonical_authority_arguments(skill_name, params),
            )
            return None
        except CapabilityViolation as exc:
            _record_capability_degradation(
                exc,
                action=(
                    f"{'refused' if mode == 'strict' else 'permitted (warn mode)'} "
                    f"'{skill_name}': {exc.denial.value}"
                ),
                severity="degraded" if mode == "strict" else "warning",
                enforce_failure_policy=False,
            )
            if mode != "strict":
                self.logger.warning(
                    "⚠️  CapabilityEngine: '%s' executing WITHOUT verified Will "
                    "authority (%s) — warn mode",
                    skill_name,
                    exc.denial.value,
                )
                return None
            self.logger.warning(
                "🔒 CapabilityEngine: '%s' refused — %s (%s)",
                skill_name,
                exc.denial.value,
                exc.detail,
            )
            return {
                "ok": False,
                "error": f"Will authority not established: {exc.denial.value}",
                "status": "blocked_by_capability_chain",
                "denial": exc.denial.value,
                "detail": exc.detail,
            }

    @staticmethod
    def _context_governed_execution(ctx: dict[str, Any], skill_name: str) -> bool:
        """Is this execution carrying real authority from the Will?

        Authority is established by *authenticating a signature*, never by a
        caller's claim. A context that merely says it was verified
        (``_capability_token_verified``) is not evidence of anything — that flag
        was the bypass this method used to honour, and any code path or
        deserialized payload could set it.
        """
        from .capability_engine import (
            _record_capability_degradation,
        )

        cap = capability_from_context(ctx)
        if cap is not None:
            result = get_capability_verifier().verify(
                cap,
                expected_action_digest=None,  # bound at the execution sink
                consume=False,                # this is an advisory read, not a spend
            )
            if result.ok:
                return True
            _record_capability_degradation(
                CapabilityViolation(
                    result.denial or CapabilityDenial.MALFORMED, result.detail
                ),
                action=(
                    f"treated '{skill_name}' as ungoverned: presented capability "
                    f"failed verification"
                ),
                severity="warning",
            )
            return False

        # Legacy opaque-token path. This establishes only that a token naming
        # this skill exists in-process — not that the Will issued it. It is kept
        # so callers still on the old contract are not silently downgraded to
        # "ungoverned", but it is not accepted as proof of Will provenance and
        # a bare self-asserted flag is never sufficient on its own.
        token_id = str((ctx or {}).get("capability_token_id") or "").strip()
        if not token_id:
            return False
        try:
            from core.executive.authority_gateway import get_authority_gateway

            if get_authority_gateway().verify_tool_access(skill_name, token_id):
                return True
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_capability_degradation(
                exc,
                action="treated unverified capability token as ungoverned for EDI check",
                severity="warning",
            )
        return False

    @staticmethod
    def _context_user_authorized(ctx: dict[str, Any], exec_source: str) -> bool:
        # sealed_validation is gone: nothing in core/ or interface/ has ever
        # set it, so its only possible source was an external payload. A key
        # that only an attacker can populate is not an authorization signal.
        from .capability_engine import (
            _USER_FACING_CONTEXT_ORIGINS,
            _proof_run_environment_active,
        )

        if bool(
            ctx.get("user_requested_action")
            or ctx.get("user_explicitly_authorized")
            or (
                ctx.get("proof_evaluation_contract")
                and _proof_run_environment_active()
            )
        ):
            return True
        if exec_source in _USER_FACING_CONTEXT_ORIGINS:
            return True
        proof_run = str(os.environ.get("AURA_PROOF_RUN", "") or "").strip().lower()
        if proof_run in {"1", "true", "yes"} and str(ctx.get("origin") or "").lower() in {
            "test",
            "proof",
        }:
            return True
        return False

    def _edi_risk_for(
        self,
        skill_name: str,
        meta: SkillMetadata,
        params: dict[str, Any],
        effect_scope: str,
    ) -> str:
        from .capability_engine import (
            logger,
        )

        risk = classify_execution_risk(
            skill_name,
            params,
            effect_scope=effect_scope,
            metabolic_cost=meta.metabolic_cost,
        )
        # Which arguments the rating was taken on.
        #
        # A tool rated from its arguments and a tool rated from nothing produce
        # the same word, and the lease then compares two of them taken at
        # different moments. Finding that out has cost five live turns.
        #
        # The module logger, not self.logger: they are the same object —
        # __init__ assigns this one — and rating a tool must not depend on
        # having been through __init__. A caller that builds the object
        # another way got AttributeError from a log line.
        logger.info(
            "risk %s for %s (scope=%s) from arguments %s",
            risk,
            skill_name,
            effect_scope,
            sorted(str(key) for key in (params or {}))[:6] or "none",
        )
        return risk

    @staticmethod
    def _user_advocate_irreversible_for(
        skill_name: str,
        params: dict[str, Any],
        risk_level: str,
        effect_scope: str,
    ) -> bool:
        """Return whether a skill needs explicit irreversible-action consent.

        High-risk sandboxed compute is not automatically irreversible. The
        user advocate should still block stateful code, privileged mutations,
        and external/user-visible changes without confirmation, but a
        non-stateful sandbox retention probe is not a destructive act merely
        because the generic code-execution skill carries elevated risk.
        """

        if str(risk_level or "").lower() == "critical":
            return True
        if skill_name == "auto_refactor":
            return str(effect_scope or "").lower() in {
                "privileged_mutation",
                "state_mutation",
                "subprocess",
            }
        if skill_name == "run_code":
            return bool((params or {}).get("stateful", True))
        scope = str(effect_scope or "").lower()
        return scope in {
            "desktop_file_io",
            "foreground_desktop_control",
            "privileged_mutation",
            "state_mutation",
            "subprocess",
        }

    @staticmethod
    def _user_advocate_auto_confirmed_for(
        skill_name: str,
        ctx: dict[str, Any],
        exec_source: str,
        effect_scope: str,
    ) -> bool:
        """Allow explicitly user-visible foreground desktop requests to proceed.

        The desktop_task/computer_use stack is still effect-verified downstream.
        This only prevents the user-advocate from re-blocking a desktop action
        that already arrived through the live user/proof foreground lane with
        visible-local-action metadata.
        """
        from .capability_engine import (
            _LIGHTWEIGHT_BACKGROUND_IO_SKILLS,
            _USER_FACING_CONTEXT_ORIGINS,
            CapabilityEngine,
            _proof_run_environment_active,
        )


        if (
            skill_name in _LIGHTWEIGHT_BACKGROUND_IO_SKILLS
            and CapabilityEngine._safe_autonomous_web_research(skill_name, {}, ctx, exec_source, effect_scope)
        ):
            return True

        if skill_name not in {
            "computer_use",
            "desktop_task",
            "os_automation",
            "web_interlocutor",
        }:
            return False
        scope = str(effect_scope or "").lower()
        if scope not in {
            "desktop_file_io",
            "foreground_desktop_control",
            "foreground_browser_dialogue",
        }:
            return False
        if str(exec_source or "").lower() not in _USER_FACING_CONTEXT_ORIGINS:
            return False
        if skill_name == "web_interlocutor":
            return bool(
                ctx.get("user_visible_browser_action")
                or ctx.get("user_requested_action")
                or ctx.get("foreground_request")
                or str(ctx.get("route") or "").startswith(("chat.", "voice."))
                or (
                    ctx.get("proof_evaluation_contract")
                    and _proof_run_environment_active()
                )
            )
        if skill_name == "os_automation":
            return bool(
                ctx.get("foreground_request")
                and ctx.get("user_requested_action")
                and ctx.get("user_explicitly_authorized")
                and (
                    ctx.get("user_visible_desktop_action")
                    or ctx.get("local_desktop_action")
                    or str(ctx.get("route") or "").startswith(
                        ("chat.", "desktop_task.os_automation", "voice.")
                    )
                )
            )
        return bool(
            ctx.get("user_visible_desktop_action")
            or ctx.get("local_desktop_action")
            or ctx.get("desktop_task_owned_by")
            or str(ctx.get("route") or "").startswith(("chat.", "voice."))
            or (
                ctx.get("proof_evaluation_contract")
                and _proof_run_environment_active()
            )
        )

    @classmethod
    def _user_advocate_confirmed_for(
        cls,
        skill_name: str,
        params: dict[str, Any],
        ctx: dict[str, Any],
        exec_source: str,
        risk_level: str,
        effect_scope: str,
    ) -> bool:
        """Return one confirmation decision for cooldown and advocate review."""
        from .capability_engine import (
            _USER_FACING_CONTEXT_ORIGINS,
            _VERIFIED_STANDING_AUTHORITY,
            _record_capability_degradation,
        )

        # AuthorityGateway has already matched the signed child lease to this
        # exact tool invocation.  Treat that durable standing grant as the
        # confirmation contract it represents instead of asking the owner a
        # second time.  The identity sentinel cannot be supplied by JSON/API
        # callers, so raw context fields never manufacture this decision.
        if ctx.get("_standing_authority_verified") is _VERIFIED_STANDING_AUTHORITY:
            return True
        # CP126: "Confirmation and foreground classification rely on
        # parameters/context such as confirmed, user_confirmed, priority, and
        # foreground_request. These values are not bound to a UI event,
        # authenticated session, nonce, or constitutional receipt."
        #
        # `params` is the MODEL's output — it writes the tool call. Reading a
        # confirmation out of it let a model confirm its own high-risk action
        # by emitting {"confirmed": true}. Nothing in core/ or interface/
        # has ever set these four keys, so there is no legitimate producer to
        # preserve: every occurrence came from outside.
        #
        # The unforgeable path above (_VERIFIED_STANDING_AUTHORITY, placed by
        # the gateway after matching a signed lease) remains the way a real
        # confirmation is expressed, and is the pattern any future UI
        # confirmation event should follow.
        self_asserted = bool(
            params.get("confirmed")
            or params.get("user_confirmed")
            or ctx.get("confirmed")
            or ctx.get("user_confirmed")
        )
        if self_asserted:
            _record_capability_degradation(
                PermissionError(
                    f"{skill_name}: ignored a self-asserted confirmation flag; "
                    "confirmation must come from a verified standing authority"
                ),
                action="refused to treat a caller-supplied confirmed flag as user confirmation",
                severity="warning",
                enforce_failure_policy=False,
            )
        explicitly_confirmed = False
        if skill_name == "os_automation":
            return bool(
                explicitly_confirmed
                or cls._user_advocate_auto_confirmed_for(
                    skill_name,
                    ctx,
                    exec_source,
                    effect_scope,
                )
            )
        return bool(
            explicitly_confirmed
            or (
                exec_source in _USER_FACING_CONTEXT_ORIGINS
                and risk_level not in ("high", "critical")
            )
            or cls._safe_autonomous_web_research(
                skill_name,
                params,
                ctx,
                exec_source,
                effect_scope,
            )
            or cls._user_advocate_auto_confirmed_for(
                skill_name,
                ctx,
                exec_source,
                effect_scope,
            )
        )

    @staticmethod
    def _record_verified_standing_authority(
        ctx: dict[str, Any],
        tool_handle: Any,
    ) -> bool:
        """Record trusted standing-authority provenance from an approved handle."""
        from .capability_engine import (
            _VERIFIED_STANDING_AUTHORITY,
        )

        standing_token = str(
            getattr(tool_handle, "standing_authority_token", "") or ""
        ).strip()
        constraints = dict(getattr(tool_handle, "constraints", {}) or {})
        grant_id = str(constraints.get("standing_authority_grant_id") or "").strip()
        if not standing_token or not grant_id:
            return False
        ctx["_standing_authority_verified"] = _VERIFIED_STANDING_AUTHORITY
        ctx["standing_authority_grant_id"] = grant_id
        receipt_id = str(
            constraints.get("standing_authority_receipt_id") or ""
        ).strip()
        if receipt_id:
            ctx["standing_authority_receipt_id"] = receipt_id
        return True

    @staticmethod
    def _action_description_for_user_advocate(
        skill_name: str,
        params: dict[str, Any],
        effect_scope: str,
    ) -> str:
        """Describe the concrete operation, not just the skill's scary name."""

        scope = str(effect_scope or "").lower()
        if skill_name == "auto_refactor" and scope == "read_only":
            target = str((params or {}).get("path") or ".").strip() or "."
            return (
                "read-only auto_refactor code-health scan "
                f"for {target!r}; no source writes, no test execution, no promotion"
            )
        if scope == "read_only" and skill_name in {
            "free_search",
            "grounded_search",
            "local_reference_search",
            "search_web",
            "web_search",
        }:
            query = str((params or {}).get("query") or (params or {}).get("q") or "").strip()
            if query:
                return f"read-only {skill_name} information retrieval for query {query!r}"
            return f"read-only {skill_name} information retrieval"
        if skill_name == "messages":
            arguments = canonical_authority_arguments(skill_name, params)
            action = str(arguments.get("action") or "status")
            if action == "send":
                return (
                    "send a private message to the configured symbolic contact "
                    f"({int(arguments.get('body_chars') or 0)} characters; content hidden)"
                )
            return f"{action} Aura's private Messages channel"
        return f"{skill_name} {str(params)[:200]}"

    @staticmethod
    def _directly_requested_by_the_user(ctx: dict[str, Any], exec_source: str) -> bool:
        """Did a person ask for this action, in this turn, in the foreground?

        Narrow on purpose. It is not "a user exists somewhere upstream" — it is
        a foreground origin AND a turn that reads as an instruction rather than
        a mention (core/conversation/request_mood.py). An autonomous cycle that
        happens to carry a user id does not qualify.
        """
        from .capability_engine import (
            _is_a_person_asking,
        )

        origin = (
            str(exec_source or ctx.get("origin") or ctx.get("source") or "")
            .strip()
            .lower()
            .replace("-", "_")
        )
        # The runtime already has a notion of a foreground origin, and this
        # had a second, narrower one written as a literal set. They disagreed
        # about the names the desktop lane actually uses: a turn arrives here
        # as "desktop_quick_user" and its generation phase as
        # "response_generation_user", and neither was in the set — so the
        # override written for "a person asked for this, in the foreground, on
        # their own machine" could not fire for the desktop lane at all.
        #
        # LIVE, 2026-08-29: code_repl held at "worst-case harm 0.80" on a turn
        # whose words were "use that library to record this".
        #
        # Still narrow. An origin only counts when the runtime's own
        # foreground test accepts it, or when it names the person as the one
        # being served — and an autonomous loop, a curiosity cycle and a dream
        # pass are none of those.
        # Whether a person is waiting on this turn is a fact the turn knows,
        # and the caller that knows it says so. Reading it here beats deriving
        # it again from the shape of an origin string: the runtime had two
        # notions of a foreground origin written in two places and they
        # disagreed about the names the desktop lane actually uses — a turn
        # arrives as "desktop_quick_user" and generates under
        # "response_generation_user", and the literal set here listed neither,
        # so the override for "a person asked for this" could not fire for the
        # desktop lane at all.
        #
        # LIVE, 2026-08-29: code_repl held at "worst-case harm 0.80" on a turn
        # whose words were "use that library to record this".
        #
        # The origin test stays as the answer for callers that do not carry
        # the fact, and it is unchanged in what it accepts.
        stated = ctx.get("a_person_is_waiting")
        if isinstance(stated, bool):
            if not stated:
                return False
        elif not _is_a_person_asking(origin):
            return False
        message = str(
            ctx.get("message") or ctx.get("objective") or ctx.get("user_message") or ""
        ).strip()
        if not message:
            return False
        try:
            from core.conversation.request_mood import assess_request_mood

            return assess_request_mood(message).asks_for_action
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("capability_engine.direct_request", exc, severity="warning")
            return False

    @staticmethod
    def _safe_autonomous_web_research(
        skill_name: str,
        params: dict[str, Any],
        ctx: dict[str, Any],
        exec_source: str,
        effect_scope: str,
    ) -> bool:
        from .capability_engine import (
            _AUTONOMOUS_RESEARCH_ORIGINS,
            _LIGHTWEIGHT_BACKGROUND_IO_SKILLS,
            _UNSAFE_AUTONOMOUS_WEB_QUERY_MARKERS,
        )

        if skill_name not in _LIGHTWEIGHT_BACKGROUND_IO_SKILLS:
            return False
        if str(effect_scope or "").lower() != "read_only":
            return False
        origin = str(exec_source or ctx.get("origin") or ctx.get("source") or "").strip().lower().replace("-", "_")
        if origin not in _AUTONOMOUS_RESEARCH_ORIGINS:
            return False
        text = " ".join(
            str(part or "").lower()
            for part in (
                (params or {}).get("query"),
                (params or {}).get("q"),
                ctx.get("objective"),
                ctx.get("message"),
                ctx.get("reason"),
            )
        )
        if not text.strip():
            return False
        return not any(marker in text for marker in _UNSAFE_AUTONOMOUS_WEB_QUERY_MARKERS)

    @staticmethod
    def _user_benefit_for_execution(
        skill_name: str,
        params: dict[str, Any],
        ctx: dict[str, Any],
        exec_source: str,
        effect_scope: str,
    ) -> str:
        from .capability_engine import (
            _USER_FACING_CONTEXT_ORIGINS,
            CapabilityEngine,
        )

        explicit = str(params.get("user_benefit") or ctx.get("user_benefit") or "").strip()
        if explicit:
            return explicit
        if CapabilityEngine._safe_autonomous_web_research(skill_name, params, ctx, exec_source, effect_scope):
            query = str((params or {}).get("query") or (params or {}).get("q") or "").strip()
            return (
                "support Aura's autonomous curiosity, factual grounding, and memory growth "
                f"with bounded read-only web research{f' about {query[:120]!r}' if query else ''}"
            )
        objective = str(ctx.get("objective") or ctx.get("user_objective") or "").strip()
        if objective:
            return objective
        if skill_name == "auto_refactor" and str(effect_scope or "").lower() == "read_only":
            return (
                "maintain Aura's code health by surfacing bounded repair candidates "
                "without mutating source or consuming a heavy test budget"
            )
        if exec_source in _USER_FACING_CONTEXT_ORIGINS:
            return "requested through the user-facing skill lane"
        return ""
