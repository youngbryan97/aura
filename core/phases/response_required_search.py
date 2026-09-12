"""A search a turn is required to run, and the evidence it has to come back with.

"Required" means the answer is refused without it, so every path here ends in
either evidence with a source or a refusal that says which search failed. The
query cleaning matters more than it looks: a query built from her own scaffold
rather than from what the person asked returns pages about the wrong thing and
reads as confident invention.
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from core.intent.opaque_spans import first_named_url as _first_named_url
from core.utils.completed_capability import (
    any_capability_completed,
)
from core.utils.injected_blocks import stamp_grounding

from ..state.aura_state import AuraState


class _RunsTheRequiredSearch:
    """Lifted whole from ResponseGenerationPhase; see response_generation.py."""

    @classmethod
    def _successful_required_search_payload(
        cls,
        state: AuraState,
        contract: Any,
    ) -> tuple[str, dict[str, Any]] | None:
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .response_generation import (
            _SEARCH_SKILL_NAMES,
        )

        if not getattr(contract, "requires_search", False):
            return None
        if state.response_modifiers.get("last_skill_ok") is not True:
            return None
        skill_name = str(state.response_modifiers.get("last_skill_run") or "").strip()
        if skill_name not in _SEARCH_SKILL_NAMES:
            return None
        payload = state.response_modifiers.get("last_skill_result_payload")
        if not isinstance(payload, dict):
            return None
        return skill_name, payload

    @classmethod
    def _render_required_search_answer_from_payload(
        cls,
        *,
        payload: dict[str, Any],
    ) -> str:
        results = payload.get("results")
        first_result = next(
            (item for item in results if isinstance(item, dict)),
            {},
        ) if isinstance(results, list) else {}
        title = str(
            first_result.get("title")
            or payload.get("title")
            or payload.get("source_title")
            or ""
        ).strip()
        url = str(
            first_result.get("url")
            or payload.get("url")
            or payload.get("source")
            or ""
        ).strip()
        evidence_text = str(
            payload.get("answer")
            or payload.get("summary")
            or first_result.get("snippet")
            or payload.get("message")
            or ""
        ).strip()
        if not evidence_text:
            evidence_text = "The search returned evidence, but the result did not include a usable snippet."
        sentence = cls._first_sentence(evidence_text, fallback=evidence_text)

        if title and url:
            return f"I found {title}. {sentence} Source: {url}"
        if title:
            return f"I found {title}. {sentence}"
        if url:
            return f"I found a relevant source. {sentence} Source: {url}"
        return sentence

    @classmethod
    def _repair_false_required_tool_inability(
        cls,
        *,
        state: AuraState,
        contract: Any,
        response_text: str,
    ) -> str:
        from .response_generation import (
            _TOOL_FALSE_INABILITY_RE,
            logger,
        )

        hit = cls._successful_required_search_payload(state, contract)
        if not hit:
            return response_text
        skill_name, payload = hit
        if not _TOOL_FALSE_INABILITY_RE.search(str(response_text or "")):
            return response_text
        repaired = cls._render_required_search_answer_from_payload(payload=payload)
        state.response_modifiers["required_tool_false_inability_repaired"] = {
            "skill": skill_name,
            "method": "deterministic_grounded_evidence",
        }
        logger.warning(
            "🛡️ ResponseGeneration replaced false %s inability after successful required evidence.",
            skill_name,
        )
        return repaired

    async def _execute_required_search_evidence(
        self,
        *,
        state: AuraState,
        objective: str,
        contract: Any,
        origin: str,
        runtime_context: dict[str, Any],
    ) -> bool:
        """Read the evidence a turn requires before the model narrates having it.

        A named document first, then search. The URL branch sits ABOVE the
        requires_search gate on purpose: the chat lane stopped calling a turn
        with an address in it a search turn — correctly, since searching for
        an address returns pages about it — and this method returned
        immediately, so nothing was fetched at all and she reported the fetch
        as failing.
        """
        from .response_generation import (
            _record_response_generation_degradation,
            logger,
        )


        if getattr(contract, "tool_evidence_available", False):
            return False

        completed_evidence = runtime_context.get("completed_capability_evidence")
        if any_capability_completed(
            completed_evidence,
            {"web_search", "search_web", "free_search", "grounded_search"},
        ):
            state.response_modifiers["required_search_evidence_reused"] = True
            return False

        cap = self._capability_engine()
        visible_objective = str(
            runtime_context.get("visible_user_message")
            or runtime_context.get("user_message")
            or ""
        ).strip()
        named_url = _first_named_url(visible_objective)
        if named_url and cap is not None:
            fetched = await self._fetch_named_url_evidence(
                state, cap, named_url, origin=origin, runtime_context=runtime_context
            )
            if fetched:
                return True

        if not getattr(contract, "requires_search", False):
            return False

        query = self._clean_required_search_query(
            str(getattr(contract, "search_query", "") or objective or "").strip()
        )
        if not query:
            return False

        # The query must come from what the PERSON asked.
        #
        # LIVE DEFECT, 2026-08-10. current_objective is not always the user's
        # message — on an ambient or internally-driven turn it can hold
        # whatever perception last put there — and it is the fallback when the
        # contract carries no search_query. So this lane ran, against a real
        # search engine, with:
        #
        #     query=The Sick Mind of EDP445 | …Documentary - YouTube 🔊
        #     query=License Plate Lookup & VIN Search | VehicleHistory.us
        #     query=Monthly Expenses
        #     query=t get a clear enough answer together, and I
        #
        # The first three are the titles of Bryan's own windows and the fourth
        # is a mid-word fragment of her own previous reply. Every one was
        # logged origin=user. Two costs, and the second is the serious one:
        # the evidence is irrelevant to the turn, and the titles of a person's
        # private windows — what they are watching, a VIN lookup, a document
        # called "Monthly Expenses" — were sent to an external search engine
        # by a system nobody asked to search anything.
        #
        # Egress is not reversible, so this fails closed: no demonstrable
        # relation to the user's own words means no search.
        if not self._query_comes_from_the_user(query, runtime_context, objective):
            logger.warning(
                "🔎 ResponseGeneration: refused required search evidence — the "
                "query is not derived from the user's message (likely a window "
                "title or internal objective). Nothing was sent."
            )
            _record_response_generation_degradation(
                RuntimeError("required search query not user-derived"),
                action="skipped required search evidence rather than egress an internal objective",
                severity="warning",
            )
            return False

        if cap is None:
            logger.warning(
                "🔎 ResponseGeneration: required search evidence skipped because capability_engine is unavailable."
            )
            return False

        skill_name = "web_search"
        matched = state.response_modifiers.get("matched_skills") or []
        if isinstance(matched, str):
            matched = [matched]
        for candidate in matched:
            resolved = candidate
            if hasattr(cap, "resolve_skill_name"):
                try:
                    resolved = cap.resolve_skill_name(str(candidate))
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    resolved = str(candidate)
            if str(resolved) in {"web_search", "search_web", "free_search", "grounded_search"}:
                skill_name = str(resolved)
                break

        context = {
            "origin": origin,
            "source": origin,
            "route": "response_generation.required_search_evidence",
            "objective": objective,
            "message": objective,
            "user_requested_action": True,
            "risk_level": "low",
            "effect_scope": "read_only_external_io",
            "skill_name": skill_name,
            "tool_name": skill_name,
            "foreground_request": True,
            "desktop_cognitive_engine_required": bool(
                runtime_context.get("desktop_cognitive_engine_required")
                or runtime_context.get("cognitive_engine_required")
            ),
        }

        try:
            result = await asyncio.wait_for(
                cap.execute(
                    skill_name,
                    {
                        "query": query,
                        "num_results": 5,
                        "deep": False,
                        "retain": False,
                    },
                    context,
                ),
                timeout=35.0,
            )
        except (
            OSError,
            ConnectionError,
            TimeoutError,
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            _record_response_generation_degradation(
                exc,
                action="continued search turn after mandatory search evidence execution failed",
                severity="warning",
            )
            return False

        payload = self._sanitize_grounding_payload(result)
        ok = bool(payload.get("ok"))
        payload.setdefault("query", query)
        payload.setdefault("ok", ok)

        state.response_modifiers["last_skill_run"] = skill_name
        state.response_modifiers["last_skill_ok"] = ok
        state.response_modifiers["last_skill_turn_marker"] = state.response_modifiers.get(
            "evidence_turn_marker"
        )
        state.response_modifiers["last_skill_result_payload"] = payload
        state.response_modifiers["required_search_evidence_executed"] = {
            "skill": skill_name,
            "ok": ok,
            "query": query[:240],
        }
        state.cognition.working_memory.append(
            # Stamped: the inference gate gives grounding privileged placement
            # and compaction protection, and a caller-controlled marker is not
            # proof that this runtime gathered it.
            stamp_grounding(
                {
                    "role": "system",
                    "content": self._render_skill_result_block(
                        skill_name=skill_name,
                        payload=payload,
                    ),
                    "metadata": {
                        "type": "skill_result",
                        "skill": skill_name,
                        "ok": ok,
                        "query": query[:240],
                        "turn_marker": state.response_modifiers.get(
                            "evidence_turn_marker"
                        ),
                    },
                    "timestamp": time.time(),
                }
            )
        )
        try:
            state.cognition.trim_working_memory()
        except AttributeError:
            pass
        logger.info(
            "🔎 ResponseGeneration: executed required search evidence via %s (ok=%s query=%s).",
            skill_name,
            ok,
            query[:120],
        )
        return True

    async def _fetch_named_url_evidence(
        self,
        state: AuraState,
        cap: object,
        url: str,
        *,
        origin: str,
        runtime_context: dict,
    ) -> bool:
        """Read the document the person named and stamp it as this turn's evidence.

        Same custody as the search path: the result is sanitized, recorded as
        the turn's skill result, and stamped so the inference gate gives it
        grounding placement. Returns False when the fetch is unavailable or
        fails, so the search path still runs and the turn is never left with
        nothing.
        """
        from .response_generation import (
            _record_response_generation_degradation,
            logger,
        )

        context = {
            "origin": origin,
            "source": origin,
            "route": "response_generation.named_url_evidence",
            "objective": url,
            "message": url,
            "user_requested_action": True,
            "risk_level": "low",
            "effect_scope": "read_only",
            "skill_name": "http_request",
            "tool_name": "http_request",
            "foreground_request": True,
            "desktop_cognitive_engine_required": bool(
                runtime_context.get("desktop_cognitive_engine_required")
                or runtime_context.get("cognitive_engine_required")
            ),
        }
        try:
            result = await asyncio.wait_for(
                cap.execute("http_request", {"url": url, "method": "GET"}, context),
                timeout=35.0,
            )
        except (
            OSError,
            ConnectionError,
            TimeoutError,
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            _record_response_generation_degradation(
                exc,
                action="fell back to search evidence after the named URL could not be read",
                severity="info",
            )
            return False

        payload = self._sanitize_grounding_payload(result)
        if not bool(payload.get("ok")):
            return False
        payload.setdefault("query", url)

        state.response_modifiers["last_skill_run"] = "http_request"
        state.response_modifiers["last_skill_ok"] = True
        state.response_modifiers["last_skill_turn_marker"] = state.response_modifiers.get(
            "evidence_turn_marker"
        )
        state.response_modifiers["last_skill_result_payload"] = payload
        state.response_modifiers["required_search_evidence_executed"] = {
            "skill": "http_request",
            "ok": True,
            "query": url[:240],
        }
        state.cognition.working_memory.append(
            stamp_grounding(
                {
                    "role": "system",
                    "content": self._render_skill_result_block(
                        skill_name="http_request",
                        payload=payload,
                    ),
                    "metadata": {
                        "type": "skill_result",
                        "skill": "http_request",
                        "ok": True,
                        "query": url[:240],
                        "turn_marker": state.response_modifiers.get("evidence_turn_marker"),
                    },
                    "timestamp": time.time(),
                }
            )
        )
        try:
            state.cognition.trim_working_memory()
        except AttributeError:
            pass
        logger.info(
            "🔎 ResponseGeneration: read the named document instead of searching for it (%s).",
            url[:120],
        )
        return True

    @staticmethod
    def _query_comes_from_the_user(
        query: str, runtime_context: dict[str, Any], objective: str
    ) -> bool:
        """Whether this search query traces back to what the person typed.

        Guards an EGRESS, so it fails closed: an unreadable user message means
        no search rather than a search on whatever the objective happened to
        hold. See the call site for the window titles this let out.

        Content-word overlap rather than equality, because the contract
        legitimately rewrites a question into a query ("who won the 2026 Nobel
        Prize in Physics" -> "2026 Nobel Prize Physics winner"). What it never
        does is produce a query with nothing of the question left in it.
        """
        from .response_generation import (
            _QUERY_PROVENANCE_STOPWORDS,
        )

        user_text = str(
            runtime_context.get("visible_user_message")
            or runtime_context.get("user_message")
            or ""
        ).strip()
        if not user_text:
            # Unknown provenance. On an egress path that is a refusal, not a
            # default-allow — the ambient turns are exactly the ones with no
            # visible user message.
            return False

        def _tokens(text: str) -> set[str]:
            return {
                word
                for word in re.findall(r"[a-z0-9][a-z0-9'-]{2,}", text.lower())
                if word not in _QUERY_PROVENANCE_STOPWORDS
            }

        asked = _tokens(user_text)
        if not asked:
            # The person said something with no content words at all ("why?").
            # Nothing to match against, so only an unchanged objective passes.
            return str(objective or "").strip() == user_text
        return bool(_tokens(query) & asked)

    @staticmethod
    def _clean_required_search_query(query: str) -> str:
        """Remove response-format instructions without losing source semantics.

        Live regression: "Tell me the source title and what NASA says Europa is"
        was reduced to "one current NASA page about Europa". That made the
        search/cache layer satisfy the wrong task with a Clipper article instead
        of a definition-bearing NASA source. The cleaner may drop pure formatting
        instructions ("source title only"), but source-definition clauses stay in
        the retrieval query because they change what evidence is relevant.
        """
        from .response_generation import (
            _SOURCE_DEFINITION_TAIL_RE,
        )


        raw = str(query or "").strip()
        if not raw:
            return ""
        repair_match = re.search(
            r"(?is)\boriginal\s+user\s+request\s*:\s*(.*?)"
            r"(?:\n\s*\n\s*rejected\s+draft\s+for\s+avoidance\s+only\s*:|$)",
            raw,
        )
        if repair_match:
            raw = repair_match.group(1).strip()
            if not raw:
                return ""
        source_definition_tails = [
            " ".join(match.group(0).strip(" .?!,:;").split())
            for match in _SOURCE_DEFINITION_TAIL_RE.finditer(raw)
        ]
        cleaned = re.sub(
            r"^\s*(?:please\s+)?(?:search|look\s+up|find)\s+"
            r"(?:(?:the\s+)?(?:web|internet)\s+)?(?:for\s+)?",
            "",
            raw,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s+(?:and\s+tell me|then\s+tell me|and\s+answer|then\s+answer|"
            r"and\s+give me|then\s+give me)\b.*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip(" .?!,:;")
        cleaned = re.sub(
            r"(?:^|[.?!]\s*)(?:tell\s+me|include|give\s+me)\s+"
            r"(?:the\s+)?source\s+title(?:\s+only)?(?:\s+and)?\s*",
            ". ",
            cleaned,
            flags=re.IGNORECASE,
        ).strip(" .?!,:;")
        for tail in source_definition_tails:
            if tail and tail.lower() not in cleaned.lower():
                cleaned = f"{cleaned}. {tail}" if cleaned else tail
        return re.sub(r"\s+", " ", cleaned).strip(" .?!,:;")
