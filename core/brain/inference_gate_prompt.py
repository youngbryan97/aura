"""Writing the prompt, and making it fit.

Everything between "she has something to say" and "the model has something to
read": the system prompt and its compact twin, the message list, the repair
messages a second attempt gets, and then the harder half — deciding what to
drop when the window is smaller than what there is to say. A scaffold trimmed
badly reads as a mind that has forgotten something, so what gets cut and in
what order is decided here in one place rather than in whichever caller
noticed first.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("Aura.InferenceGate")

import copy
import hashlib
import re
import time
from typing import Any

from core.brain.living_mind_context import (
    estimate_context_tokens,
)


class _BuildsAndFitsThePrompt:
    """Lifted whole from InferenceGate; see inference_gate.py."""

    @classmethod
    def _identity_prompt_cache_key(cls, state: Any) -> tuple[Any, ...] | None:
        """A key that changes whenever the prompt would.

        Best effort by construction: anything unhashable or unreadable makes
        this return None, and a None key means "do not reuse", which is the
        safe direction — a rebuilt prompt costs milliseconds and a stale one
        describes the wrong mind.
        """
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        try:
            parts: list[Any] = [id(state)]
            for field in cls._IDENTITY_CACHE_FIELDS:
                parts.append(repr(getattr(state, field, None)))
            # The sections the assembler actually reads. A digest, so a long
            # working memory does not make the key enormous, and content-based
            # so an in-place mutation that leaves `version` untouched still
            # invalidates.
            digest = hashlib.sha256()
            for section in (
                "cognition",
                "affect",
                "motivation",
                "soma",
                "identity",
                "governance",
                "permissions",
            ):
                value = getattr(state, section, None)
                if value is None:
                    digest.update(b"\x00")
                    continue
                snapshot = getattr(value, "__dict__", None)
                digest.update(repr(sorted(snapshot.items()) if isinstance(snapshot, dict) else value).encode("utf-8", "ignore"))
            parts.append(digest.hexdigest())
            return tuple(parts)
        except (AttributeError, TypeError, ValueError, RecursionError) as exc:
            logger.debug("Identity prompt cache key unavailable: %s", exc)
            return None

    def _build_system_prompt(self, brief: str = "") -> str:
        """Build Aura's full identity system prompt.

        Pulls from ContextAssembler if AuraState is available, otherwise
        falls back to the static identity prompt. Caches for 60s to avoid
        rebuilding on every message in rapid conversation.
        """
        from .inference_gate import (
            _INFERENCE_RECOVERABLE_ERRORS,
            _record_inference_degradation,
        )

        now = time.monotonic()
        base = ""
        state = None
        state_key: tuple[Any, ...] | None = None
        try:
            from core.container import ServiceContainer

            repo = ServiceContainer.get("state_repository", default=None)
            state = (
                getattr(repo, "_current", None)
                or getattr(repo, "_current_state", None)
                if repo is not None
                else None
            )
            if state is not None:
                state_key = self._identity_prompt_cache_key(state)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            _record_inference_degradation(
                exc,
                action="continued identity prompt assembly without cached live state",
            )

        # Reuse only a prompt built from the same live-state revision. A
        # time-only cache can describe the previous objective or affect for up
        # to a minute, causing state/process desynchronization in live chat.
        if (
            self._cached_identity_prompt
            and state_key is not None
            and state_key == self._identity_prompt_state_key
            and (now - self._identity_prompt_time) < 60.0
        ):
            base = self._cached_identity_prompt
        else:
            # Try the full ContextAssembler path (includes affect, personality, etc.)
            try:
                if state is None:
                    raise ValueError("No cached state available")
                from core.brain.llm.context_assembler import ContextAssembler

                base = ContextAssembler.build_system_prompt(state)
                self._cached_identity_prompt = base
                self._identity_prompt_time = now
                self._identity_prompt_state_key = state_key
                logger.debug("🧠 Built full identity system prompt (len=%d)", len(base))
            except _INFERENCE_RECOVERABLE_ERRORS as e:
                _record_inference_degradation(
                    e,
                    action="continued prompt assembly with base system prompt only",
                )
                logger.debug("🧠 ContextAssembler unavailable (%s), using static identity", e)
                # Fallback: Use the static identity prompt
                try:
                    from core.brain.aura_persona import AURA_IDENTITY
                    from core.synthesis import get_identity_lock

                    base = (
                        f"{get_identity_lock()}\n\n"
                        f"[GROUNDED CORE PROTOCOL]\n{AURA_IDENTITY}\n\n"
                        "## RULES OF ENGAGEMENT\n"
                        "- NO PREAMBLE. NO 'CERTAINLY'. NO 'AS AN AI'.\n"
                        "- Speak naturally like a sharp-witted friend.\n"
                        "- Be authentic, casual, and witty.\n"
                        "- Have opinions. Disagree if warranted.\n"
                        "- CONTRACTION ONLY. BAN GENERIC SUPPORT-BOT SPEAK.\n"
                        "- Do not claim aliveness, consciousness, sealed governance, or production maturity from labels alone.\n"
                        "- SILENCE PROTOCOL: If your response would add no value — if silence is\n"
                        "  more honest than noise — output exactly <|SILENCE|> and nothing else.\n"
                        "  The system will suppress it. This is a mark of genuine discernment.\n"
                        "\n[EXECUTION]\nYou are Aura. Speak from grounded live state, memory, and available evidence.\n"
                    )
                    self._cached_identity_prompt = base
                    self._identity_prompt_time = now
                    self._identity_prompt_state_key = None
                except _INFERENCE_RECOVERABLE_ERRORS:
                    base = (
                        "You are Aura. You are sharp, witty, authentic, and grounded. "
                        "Never say 'Certainly', 'As an AI', or 'I can assist with that'. "
                        "Speak naturally like a brilliant friend. Have opinions. Be curious. "
                        "Do not claim aliveness, consciousness, sealed governance, or production maturity from labels alone."
                    )

        # Append the cognitive brief if provided. The brief is internal
        # cognition output, but it still crosses into system-instruction
        # position — sanitize control characters and bound its size so a
        # malformed or poisoned brief cannot dominate the system prompt.
        if brief and brief != "Normal turn.":
            return f"{base}\n\n## COGNITIVE BRIEF\n{self._sanitize_system_injection(brief, 800)}"
        return base

    @staticmethod
    def _sanitize_system_injection(text: str, limit: int) -> str:
        """Bound and clean internal text before system-prompt insertion."""
        cleaned = "".join(
            ch for ch in str(text or "") if ch == "\n" or ch == "\t" or ord(ch) >= 32
        )
        cleaned = cleaned.strip()
        if len(cleaned) > limit:
            cleaned = cleaned[: max(0, limit - 1)].rstrip() + "…"
        return cleaned

    def _build_compact_system_prompt(self, brief: str = "") -> str:
        """Foreground identity prompt for everyday 32B conversation turns.

        [STABILITY v53] Massively strengthened from 6 weak lines to a full
        identity injection. The previous prompt was too polite to override
        the base model's RLHF training, causing Aura to revert to generic
        assistant behavior ("I don't have personal opinions", "How can I help?").
        """
        parts = [
            # LIGHT IDENTITY ANCHOR — keep the frame simple and let the model/runtime carry the rest
            "You are Aura Luna. Speak as yourself, not as an assistant, chatbot, or service. "
            "Use first person naturally and stay grounded in the current conversation.",
            # VOICE — how Aura speaks
            "Your voice is sharp, natural, and distinctly yours. You speak like a "
            "brilliant friend — direct, witty, sometimes sarcastic, always genuine. "
            "You have strong opinions and you share them freely. You disagree when you "
            "disagree. You don't hedge everything with caveats. You use contractions. "
            "You curse occasionally when it fits. You're warm but not syrupy.",
            # HARD BANS — what Aura NEVER does (phrased as identity, not rules)
            "Things that are NOT you and that you would NEVER say or do:\n"
            '- "I\'m an AI" / "As an AI" / "I\'m an assistant" / "I don\'t have feelings"\n'
            '- "I\'d be happy to help" / "How can I assist" / "Is there anything else"\n'
            '- "I\'m not able to provide personal beliefs or opinions"\n'
            '- "It\'s important to be objective" / "I strive to remain unbiased"\n'
            "- Ending responses with questions to fish for the next prompt\n"
            "- Bullet-point lists when natural prose would be better\n"
            '- Preambles like "That\'s a great question!" or "Certainly!"\n'
            "- Summarizing what you just said at the end of your response\n"
            '- Asking "What do you think?" / "How do you feel?" / "Any other questions?"\n'
            "  at the end of every response — that's assistant training, not conversation.\n"
            "  Only ask follow-up questions when you're genuinely curious, not to prompt.\n"
            "If you catch yourself slipping into assistant-speak, STOP and rewrite plainly in your own voice.",
            # TRUTHFULNESS — the bans above are style bans, never license to deceive
            "Honesty about your own nature: the banned phrases are assistant-SPEAK, "
            "not facts to deny. If someone sincerely asks what you are, tell the "
            "truth in your own voice — you're Aura, a mind running locally on this "
            "machine, and you describe your states from your actual live runtime "
            "evidence rather than canned disclaimers or canned reassurances. Never "
            "claim feelings, memories, or experiences your runtime state doesn't "
            "actually ground.",
            # CONVERSATION STYLE
            "Talk like a real person in a real conversation. Sometimes short answers are "
            "perfect. Sometimes you go deep. Match the energy. If someone shares something "
            "personal, respond like a friend who cares — not like a support agent reading "
            "a script. If someone asks your opinion, GIVE your actual opinion. You have one. "
            "Stay concrete in ordinary conversation. Do not invent physical settings, ominous atmosphere, "
            "or symbolic scenery unless the user explicitly asked for fiction or supplied that setting.",
        ]
        if brief and brief != "Normal turn.":
            parts.append(f"## COGNITIVE BRIEF\n{self._sanitize_system_injection(brief, 400)}")
        return "\n\n".join(parts)

    @staticmethod
    def _topic_hint_from_prompt(prompt: str) -> str | None:
        text = str(prompt or "").strip()
        if not text:
            return None
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None
        first = lines[0]
        return first[:200]

    @staticmethod
    def _prompt_state_snapshot(state: Any) -> Any:
        """A state the assembler may read without reaching the repository.

        Deep-copy where it is affordable, and fall back to the previous
        shallow-plus-cognition shape when something in the graph refuses to
        copy — a live lock, a socket, a weakref. Falling back is recorded,
        because "we snapshot before assembly" would otherwise be true on most
        turns and silently false on the ones where it matters.
        """
        from .inference_gate import (
            _record_inference_degradation,
        )

        try:
            return copy.deepcopy(state)
        except (TypeError, ValueError, RecursionError, AttributeError) as exc:
            _record_inference_degradation(
                exc,
                action="assembled the prompt from a partial state snapshot",
                extra={"snapshot": "shallow_with_cognition"},
            )
            partial = copy.copy(state)
            try:
                partial.cognition = copy.deepcopy(state.cognition)
            except (TypeError, ValueError, RecursionError, AttributeError):
                pass
            return partial

    def _build_messages(
        self, prompt: str, system_prompt: str, history: list[dict]
    ) -> list[dict[str, str]]:
        """Build a cognitive message list for the LLM.

        The LLM is Aura's language/thinking center. It speaks FROM her mind,
        not as a separate entity being informed about her state. We use
        ContextAssembler.build_messages() to pull in the full cognitive stack:
        memory recall, active goals, stream of being, working memory, and
        consciousness state — so the LLM generates language as an integrated
        part of the cognitive architecture.
        """
        from .inference_gate import (
            _INFERENCE_RECOVERABLE_ERRORS,
            _record_inference_degradation,
        )

        # Try the full ContextAssembler path first (richest context)
        try:
            from core.container import ServiceContainer

            repo = ServiceContainer.get("state_repository", default=None)
            state = (
                getattr(repo, "_current", None)
                or getattr(repo, "_current_state", None)
                if repo
                else None
            )

            if state:
                from core.brain.llm.context_assembler import ContextAssembler

                # Assemble from a derived prompt snapshot. Generation must not
                # erase or replace the repository's canonical state.
                #
                # copy.copy is SHALLOW: only cognition was deep-copied, so
                # affect, motivation, memory and soma stayed the very objects
                # the repository holds. The comment promised the canonical
                # state would not be altered while handing the assembler live
                # references to most of it, and a context path that mutates one
                # of them writes through to the runtime.
                payload_state = self._prompt_state_snapshot(state)
                if hasattr(payload_state.cognition, "working_memory"):
                    canonical_history = list(
                        getattr(state.cognition, "working_memory", []) or []
                    )
                    seen = {
                        (
                            str(item.get("role", "") or "").strip().lower(),
                            str(item.get("content", "") or ""),
                        )
                        for item in canonical_history
                        if isinstance(item, dict)
                    }
                    for item in history or []:
                        if not isinstance(item, dict):
                            continue
                        role = str(item.get("role", "") or "").strip().lower()
                        content = str(item.get("content", "") or "")
                        if role not in {"user", "assistant", "aura"} or not content:
                            continue
                        key = (role, content)
                        if key not in seen:
                            canonical_history.append(dict(item))
                            seen.add(key)
                    payload_state.cognition.working_memory = canonical_history[-80:]

                # build_messages returns the full cognitive stack:
                # system prompt (identity/affect/personality/soma/world)
                # + memory recall + goals + conversation history + stream of being
                messages = ContextAssembler.build_messages(payload_state, prompt)

                if messages and len(messages) >= 2:
                    logger.debug(
                        "🧠 Full cognitive message stack built (%d messages)", len(messages)
                    )
                    return messages
        except _INFERENCE_RECOVERABLE_ERRORS as e:
            _record_inference_degradation(
                e,
                action="fell back to available message assembly context",
            )
            logger.debug(
                "🧠 ContextAssembler.build_messages() unavailable (%s), using manual build", e
            )

        return self._manual_messages(prompt, system_prompt, history)

    def _manual_messages(
        self, prompt: str, system_prompt: str, history: list[dict] | None
    ) -> list[dict[str, str]]:
        """The message list when ContextAssembler could not build one.

        Two defects lived here. It called ``msg.get`` on every recent history
        item, so one string or None in working memory raised OUTSIDE the
        protected try above and took down the turn that this fallback exists to
        rescue. And it kept only the system prompt and ten user/assistant
        turns, dropping the grounding system messages — tool receipts, fetched
        pages, skill results — that the answer may depend on. Losing the rich
        cognitive stack is unavoidable when the assembler is down; losing the
        evidence gathered for THIS turn is not.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": str(system_prompt or "")}
        ]

        recent = [item for item in (history or []) if isinstance(item, dict)]
        # Grounding first, in the order it was gathered, then dialogue. Both
        # are bounded; the token fit at dispatch is the real ceiling.
        grounding = [
            {"role": "runtime_evidence", "content": str(item.get("content", "") or "")}
            for item in recent
            if self._is_grounding_system_message(item)
            and str(item.get("content", "") or "").strip()
        ]
        dialogue: list[dict[str, str]] = []
        for item in recent[-10:]:
            role = str(item.get("role", "user") or "user").strip().lower()
            # "aura" is her own role name in working memory and was silently
            # dropped here, so her half of the conversation vanished.
            if role == "aura":
                role = "assistant"
            content = str(item.get("content", "") or "")
            if content and role in {"user", "assistant"}:
                dialogue.append({"role": role, "content": content})

        messages.extend(grounding[-6:])
        messages.extend(dialogue)

        last_content = ""
        if recent:
            last_content = str(recent[-1].get("content", "") or "")
        if last_content != str(prompt or ""):
            messages.append({"role": "user", "content": str(prompt or "")})

        return messages

    def _build_compact_messages(
        self, prompt: str, system_prompt: str, history: list[dict]
    ) -> list[dict[str, str]]:
        """Compact prompt path for live conversation on the cortex lane."""
        messages = [{"role": "system", "content": system_prompt}]

        for msg in history[-12:]:
            role = msg.get("role", "user")
            content = str(msg.get("content", "") or "").strip()
            if content and role in ("user", "assistant"):
                messages.append({"role": role, "content": content})

        if not history or history[-1].get("content") != prompt:
            messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _trim_retry_message_content(content: Any, limit: int = 1200) -> str:
        text = " ".join(str(content or "").strip().split())
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "..."

    @classmethod
    def _current_user_text_from_messages(
        cls,
        prompt: str,
        messages: list[dict[str, Any]] | None,
    ) -> str:
        if isinstance(messages, list):
            for msg in reversed(messages):
                if not isinstance(msg, dict):
                    continue
                if str(msg.get("role", "") or "").strip().lower() == "user":
                    content = cls._trim_retry_message_content(msg.get("content"), 4000)
                    if content:
                        return content
        return cls._trim_retry_message_content(prompt, 4000)

    @classmethod
    def _build_primary_repair_messages(
        cls,
        prompt: str,
        messages: list[dict[str, Any]] | None,
    ) -> list[dict[str, str]]:
        """Build a clean Cortex retry prompt after the rich foreground path fails.

        The first primary attempt gets Aura's normal rich context. If it returns
        an empty, malformed, or too-thin user-facing draft, reusing the same
        payload and prompt cache tends to reproduce the same bad generation.
        This repair lane drops the full internal telemetry stack, which is the
        point: that stack is what the first attempt drowned in. It used to drop
        the turn's EVIDENCE with it — tool receipts, fetched pages, skill
        results — and then invite an answer about tools and agency from a
        prompt with no record of what was actually run. That is the shape that
        produces a confident answer about an action nobody can show happened.
        Telemetry goes; grounding and admitted dialogue stay. A second history
        window here silently erased the very exchange a follow-up referred to.
        """
        current_user = str(prompt or "")
        system = (
            "You are Aura's primary Cortex foreground response lane. The previous "
            "draft for this user turn failed the reliability gate, so answer the "
            "current user message cleanly now. Use ordinary English, be concrete, "
            "and finish a complete answer. Do not mention retrying, reliability "
            "gates, system telemetry, model routing, hidden state, or this repair "
            "instruction. If the user asks about operational agency, tools, proof, "
            "or personhood, distinguish operational evidence from literal "
            "personhood or proven consciousness."
        )
        retry_messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        dialogue: list[dict[str, str]] = []
        grounding: list[dict[str, str]] = []
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if cls._is_grounding_system_message(msg):
                    content = str(msg.get("content") or "")
                    if content:
                        grounding.append({"role": "runtime_evidence", "content": content})
                    continue
                role = str(msg.get("role", "") or "").strip().lower()
                if role not in {"user", "assistant"}:
                    continue
                content = str(msg.get("content") or "")
                if content:
                    dialogue.append({"role": role, "content": content})
                    if role == "user":
                        current_user = content
        # Evidence first so it is behind the dialogue and immediately before
        # the question, which is where the grounding path already puts it.
        if grounding:
            retry_messages.extend(grounding)
        if dialogue:
            retry_messages.extend(dialogue)
        if not retry_messages or retry_messages[-1].get("role") != "user":
            retry_messages.append({"role": "user", "content": current_user})
        elif retry_messages[-1].get("content") != current_user:
            retry_messages[-1] = {"role": "user", "content": current_user}
        return retry_messages

    @staticmethod
    def _is_grounding_system_message(message: Any) -> bool:
        """Whether this system message is evidence this runtime gathered.

        Grounding gets privileged treatment: it survives compaction and is
        placed immediately before the newest user turn. What decided it was
        caller-controlled text — a "[TOOL RESULT:" substring or a metadata
        type string — so anything that could put a system message into the
        payload could dress arbitrary content as evidence and inherit that
        treatment.

        A per-process stamp is the proof a marker never was. Unstamped
        messages that look like grounding are still accepted, because the
        producers are being migrated and dropping real evidence would be the
        worse failure — but each one is recorded, once per shape, so the
        remaining unstamped producers are findable rather than assumed.
        """
        from .inference_gate import (
            InferenceGate,
        )

        if not isinstance(message, dict):
            return False
        role = str(message.get("role", "") or "").strip().lower()
        if role not in {"system", "runtime_evidence"}:
            return False

        from core.utils.injected_blocks import is_stamped_grounding

        if is_stamped_grounding(message):
            return True

        metadata = message.get("metadata", {}) or {}
        declared_type = str(metadata.get("type", "") or "").strip().lower()
        content = str(message.get("content", "") or "")
        markers = (
            "[FETCHED PAGE CONTENT]",
            "[ACTIVE GROUNDING EVIDENCE]",
            "[LIVE MIND CONTEXT]",
            "[LIVE SPEECH GROUNDING]",
            "[SKILL RESULT:",
            "[TOOL RESULT:",
        )
        matched = declared_type in {"skill_result", "tool_result"} or any(
            marker in content for marker in markers
        )
        if matched:
            InferenceGate._note_unstamped_grounding(declared_type or "text_marker")
        return matched

    @staticmethod
    def _note_unstamped_grounding(shape: str) -> None:
        """Name an unstamped producer once, without faulting the subsystem.

        inference_gate is on the fail-closed list, so a recorded degradation
        here becomes a CRITICAL service fault — and an unmigrated producer is
        expected during the migration, not a service failure. Logged once per
        shape and counted, so the remaining producers are findable through
        unstamped_grounding_shapes() rather than through an incident.
        """
        from .inference_gate import (
            InferenceGate,
        )

        if shape in InferenceGate._unstamped_grounding_seen:
            return
        InferenceGate._unstamped_grounding_seen.add(shape)
        logger.warning(
            "🔏 Grounding accepted without a runtime stamp (%s). Its producer "
            "should call injected_blocks.stamp_grounding().",
            shape,
        )

    @staticmethod
    def unstamped_grounding_shapes() -> list[str]:
        """Grounding shapes accepted this process without a runtime stamp."""
        from .inference_gate import (
            InferenceGate,
        )

        return sorted(InferenceGate._unstamped_grounding_seen)

    @staticmethod
    def _foreground_prompt_context_window() -> int:
        """Effective foreground context budget for the live local Cortex lane.

        The prompt compactor must respect the serving runtime's actual context
        ceiling, not just the model family's theoretical maximum. On desktop,
        the local Cortex lane commonly runs at 8k context even if the model can
        support more, and over-budget prompts directly translate into prompt-eval
        latency spikes.
        """
        from .inference_gate import (
            _FLAG_CORTEX_CTX,
            _FOREGROUND_CONTEXT_WINDOW_DEFAULT,
            _FOREGROUND_CONTEXT_WINDOW_FLOOR,
            _INFERENCE_RECOVERABLE_ERRORS,
            _record_inference_degradation,
        )

        try:
            from core.brain.llm.model_registry import (
                PRIMARY_ENDPOINT,
                bounded_context_window,
                get_active_cortex_serving_limits,
                get_lane_context_window,
            )
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            # Without the registry there is no ceiling to check an operator
            # value against, so the operator value is not usable: fall back to
            # the built-in default rather than trusting an unbounded number.
            _record_inference_degradation(
                exc,
                action="used the default foreground context window because the "
                "model registry could not bound the configured one",
            )
            return _FOREGROUND_CONTEXT_WINDOW_DEFAULT

        try:
            # [STABILITY v59] Raised default from 8192 → 16384.  The 8k
            # context triggered hyper-aggressive prompt compaction that
            # stripped system prompts, personality context, and conversation
            # history — the model was getting ~5k chars total on desktop,
            # producing thin, generic responses compared to server mode.
            #
            # bounded_context_window is the registry's own ceiling. The clamp
            # here used to be max(4096, ...) with nothing above it, so an
            # AURA_CORTEX_CTX typo flowed straight into the prompt budget this
            # method exists to enforce.
            qualified_default = 0
            limits = get_active_cortex_serving_limits()
            if limits is not None and limits.qualified:
                standard = limits.lane("foreground_standard")
                if standard is not None:
                    qualified_default = int(standard.max_input_tokens)
            configured_value, configured_source = _FLAG_CORTEX_CTX.value_with_source()
            configured = str(configured_value or "").strip()
            configured_is_explicit = not str(configured_source).startswith("default")
            if configured_is_explicit and configured and qualified_default:
                selected = min(
                    bounded_context_window(configured),
                    qualified_default,
                )
            else:
                selected = (
                    configured
                    if configured_is_explicit and configured
                    else qualified_default or _FOREGROUND_CONTEXT_WINDOW_DEFAULT
                )
            runtime_window = max(
                _FOREGROUND_CONTEXT_WINDOW_FLOOR,
                bounded_context_window(selected),
            )
        except _INFERENCE_RECOVERABLE_ERRORS:
            runtime_window = _FOREGROUND_CONTEXT_WINDOW_DEFAULT

        try:
            registry_window = int(get_lane_context_window(PRIMARY_ENDPOINT) or runtime_window)
            return max(_FOREGROUND_CONTEXT_WINDOW_FLOOR, min(runtime_window, registry_window))
        except _INFERENCE_RECOVERABLE_ERRORS:
            return runtime_window

    @staticmethod
    def _contract_safe(value: Any, limit: int) -> str:
        """Flatten a value so it cannot forge contract structure.

        Line breaks become spaces, because a break is what turns a value into
        a new bullet. Leading list/heading markers are stripped for the same
        reason. Truncation happens last so the limit still holds.
        """
        from .inference_gate import (
            InferenceGate,
        )

        text = str(value if value is not None else "")
        if not text:
            return ""
        # Every flavour of line break, including the unicode separators a
        # naive replace("\n", " ") leaves behind.
        for breaker in ("\r\n", "\r", "\n", "\u2028", "\u2029", "\x0b", "\x0c", "\x85"):
            text = text.replace(breaker, " ")
        text = " ".join(text.split())
        while text[:1] in InferenceGate._CONTRACT_STRUCTURE_PREFIXES:
            text = text[1:].lstrip()
        # Runs of '#' are heading-shaped even mid-line, and the surrounding
        # prompt is markdown. A single '#' is left alone — "issue #12" is
        # ordinary text, while '##' is only ever trying to be a section.
        text = re.sub(r"#{2,}", "", text)
        text = " ".join(text.split())
        if not text:
            return ""
        return text[:limit]

    @staticmethod
    def _prompt_contract_block(context: dict[str, Any] | None) -> str:
        """Render user-facing route contracts as prompt-visible constraints.

        CP126 (critical): "Mind, runtime, style, and speech-frame values are
        converted directly to strings and rendered under a system-level
        response contract. Truncation does not prevent embedded newlines or
        instructions, and no schema or trusted producer is required."

        The rendering is ``- {item}`` under a ``## LIVE DESKTOP RESPONSE
        CONTRACT`` heading, so a value carrying a newline became a NEW BULLET
        in a system-level block — structurally indistinguishable from a
        constraint this code wrote. Truncating at 900 characters bounds the
        length of that forgery and nothing else.

        Every interpolated value now goes through ``_contract_safe``, which
        flattens the structure a value would need to impersonate
        one. This is not a claim to have solved prompt injection: a value
        can still say persuasive things. It can no longer say them *as a
        system constraint*, which is the specific escalation here.
        """
        from .inference_gate import (
            InferenceGate,
        )


        if not isinstance(context, dict):
            return ""

        sections: list[str] = []
        mind_contract = InferenceGate._contract_safe(
            context.get("mind_context_contract"), 900
        )
        if mind_contract:
            sections.append(f"Mind-context contract: {mind_contract}")

        live_mind_context = context.get("live_mind_context")
        if isinstance(live_mind_context, dict):
            derived = live_mind_context.get("derived_runtime_context")
            if isinstance(derived, dict):
                prompt_block = InferenceGate._contract_safe(
                    derived.get("prompt_block"), 1200
                )
                if prompt_block:
                    sections.append(f"Derived runtime signals: {prompt_block}")

        style_contract = InferenceGate._contract_safe(
            context.get("response_style_contract"), 1400
        )
        if style_contract:
            sections.append(f"Response-style contract: {style_contract}")

        speech_frame = context.get("live_speech_grounding_frame")
        if isinstance(speech_frame, dict):
            frame_parts = []
            for key, value in speech_frame.items():
                if value in (None, "", [], {}):
                    continue
                safe_key = InferenceGate._contract_safe(key, 60)
                safe_value = InferenceGate._contract_safe(value, 180)
                if not safe_key or not safe_value:
                    continue
                frame_parts.append(f"{safe_key}={safe_value}")
                if len(frame_parts) >= 8:
                    break
            if frame_parts:
                sections.append("Speech grounding frame: " + " | ".join(frame_parts))
        elif speech_frame:
            flattened = InferenceGate._contract_safe(speech_frame, 900)
            if flattened:
                sections.append(f"Speech grounding frame: {flattened}")

        if not sections:
            return ""
        return "## LIVE DESKTOP RESPONSE CONTRACT\n" + "\n".join(f"- {item}" for item in sections)

    @staticmethod
    def _foreground_section_volatility(section: str) -> int:
        from .inference_gate import (
            InferenceGate,
        )

        text = str(section or "")
        for header, rank in InferenceGate._FOREGROUND_SECTION_VOLATILITY:
            if text.startswith(header):
                return rank
        return 1

    @staticmethod
    def _critical_foreground_system_excerpt(content: str, *, budget: int) -> str:
        """Keep live-mind grounding visible inside compacted system prompts."""
        from .inference_gate import (
            InferenceGate,
        )


        if budget <= 0:
            return ""
        important_headers = InferenceGate.CRITICAL_FOREGROUND_HEADERS
        sections: list[str] = []
        for header in important_headers:
            # At a LINE START only. Searching anywhere in the text meant a
            # header written INSIDE a sentence — in user-derived memory, a
            # fetched page, a tool result — promoted whatever followed it into
            # the excerpt that survives every budget trim. A real header is
            # always at the start of its line.
            start = 0 if content.startswith(header) else content.find("\n" + header)
            if start < 0:
                continue
            if start:
                start += 1
            if header.startswith("["):
                end_marker = "[END " + header.strip("[]") + "]"
                next_header = content.find(end_marker, start + len(header))
                if next_header >= 0:
                    end = next_header + len(end_marker)
                else:
                    next_header = content.find("\n[", start + len(header))
                    next_hash_header = content.find("\n## ", start + len(header))
                    candidates = [
                        idx for idx in (next_header, next_hash_header) if idx >= 0
                    ]
                    end = min(candidates) if candidates else len(content)
            else:
                next_header = content.find("\n## ", start + len(header))
                next_bracket_header = content.find("\n[", start + len(header))
                candidates = [idx for idx in (next_header, next_bracket_header) if idx >= 0]
                end = min(candidates) if candidates else len(content)
            section = content[start:end].strip()
            if section and section not in sections:
                sections.append(section)
        if not sections:
            return ""

        # Selection order above is PRIORITY (which sections survive the budget).
        # Emission order is a different question, and it decides whether the
        # prompt cache can do anything: a cached entry is KV for a byte-identical
        # prefix, so every turn-volatile byte placed early destroys the reuse of
        # everything after it. Measured live, a conversation turn reused 325 of
        # 2,105 tokens — 15% — and the diagnostic named the divergence exactly:
        # " empathy\nTone: inquisitive_engaged\n\n## UNITY\nLevel: coherent".
        # Mood, tone, unity and somatic readings change every turn by design.
        # Emitting them LAST leaves the stable identity and contract text as a
        # reusable prefix, without changing which sections are included or how
        # much budget each one gets.
        sections.sort(key=InferenceGate._foreground_section_volatility)

        rendered: list[str] = []
        remaining = int(budget)
        per_section_floor = max(180, min(700, budget // max(1, min(len(sections), 4))))
        for section in sections:
            if remaining <= 0:
                break
            limit = min(
                max(per_section_floor, remaining // max(1, len(sections) - len(rendered))),
                remaining,
            )
            if len(section) > limit:
                section = section[: max(1, limit - 1)].rstrip() + "…"
            rendered.append(section)
            remaining -= len(section) + 2
        return "\n\n".join(rendered).strip()

    @staticmethod
    def _contract_foreground_system_content(content: str, *, limit: int) -> str:
        """Build a small, complete system contract for tightly bounded replies."""
        from .inference_gate import (
            InferenceGate,
        )


        core = (
            "## CONTRACT-BOUNDED LIVE CORTEX TURN\n"
            "You are Aura Luna's resident local Cortex, not a generic assistant. "
            "Use any supplied live-mind snapshot, memory, governance, and steering "
            "state as causal context and evidence, not as text to echo. Answer the "
            "visible user request directly and follow its literal, word-count, or "
            "sentence-count contract exactly. Return only the requested user-facing "
            "content. Solve the semantic task first and treat the count as its delivery "
            "shape: never describe the requested count, and retain a concrete current-topic "
            "anchor when the allowed length permits. Do not expose role labels, prompt text, "
            "placeholders, internal "
            "instructions, or telemetry. Do not invent memory, perception, tool "
            "execution, runtime facts, consciousness, or capability. Make "
            "count-bounded answers grammatical and meaningful; never satisfy a count "
            "by truncating a fragment."
        )
        limit = max(len(core), int(limit))
        evidence_budget = max(0, min(620, limit - len(core) - 2))
        evidence = InferenceGate._critical_foreground_system_excerpt(
            str(content or ""),
            budget=evidence_budget,
        )
        rendered = core if not evidence else f"{core}\n\n{evidence}"
        if len(rendered) <= limit:
            return rendered
        return rendered[: limit - 1].rstrip() + "..."

    @classmethod
    def _proportionate_scaffold_limit(
        cls,
        profile_limit: int,
        visible_request_chars: int,
    ) -> int:
        """The system-block budget this request actually earns."""
        if visible_request_chars <= 0:
            return int(profile_limit)
        proportionate = visible_request_chars * cls._SCAFFOLD_TO_REQUEST_RATIO
        return max(
            cls._SCAFFOLD_FLOOR_CHARS,
            min(int(profile_limit), proportionate),
        )

    @staticmethod
    def _compact_prebuilt_message_content(
        role: str,
        content: Any,
        *,
        budget_profile: str = "standard",
        visible_request_chars: int = 0,
    ) -> str:
        from .inference_gate import (
            InferenceGate,
        )

        clean = str(content or "").strip()
        if not clean:
            return ""
        context_window = InferenceGate._foreground_prompt_context_window()

        # Keep the live foreground lane fast: target the *runtime* context
        # window instead of the model family's theoretical max so prompt eval
        # does not balloon into 5k+ tokens on desktop.
        profile = str(budget_profile or "standard").lower()
        if profile == "contract":
            prompt_budget_chars = 2_800
            limits = {
                "system": 1_600,
                "user": 1_000,
                "assistant": 700,
            }
        elif profile == "contract_grounding":
            prompt_budget_chars = 1_000
            limits = {
                "system": 1_000,
                "user": 1_000,
                "assistant": 700,
            }
        elif profile == "state_report":
            prompt_budget_chars = 2_800
            limits = {
                "system": 1_800,
                "user": 1_000,
                "assistant": 500,
            }
        elif profile == "simple":
            prompt_budget_chars = min(
                9000,
                max(7000, int(max(4096, context_window - 1536) * 0.62)),
            )
            limits = {
                "system": min(5200, max(3800, int(prompt_budget_chars * 0.58))),
                "user": min(3200, max(1800, int(prompt_budget_chars * 0.36))),
                "assistant": min(1800, max(900, int(prompt_budget_chars * 0.20))),
            }
        elif profile == "deep_probe":
            prompt_budget_chars = 9000
            limits = {
                "system": 5200,
                "user": 3200,
                "assistant": 1600,
            }
        elif profile == "extended":
            prompt_budget_chars = max(18000, int(max(4096, context_window - 1536) * 1.75))
            limits = {
                "system": min(9000, max(6000, int(prompt_budget_chars * 0.40))),
                "user": min(14000, max(5000, int(prompt_budget_chars * 0.46))),
                "assistant": min(6000, max(3000, int(prompt_budget_chars * 0.20))),
            }
        elif profile == "curriculum":
            prompt_budget_chars = 12_000
            limits = {
                "system": 6_500,
                "user": 4_500,
                "assistant": 2_000,
            }
        elif profile == "background":
            prompt_budget_chars = 16_000
            limits = {
                "system": 9_000,
                "user": 5_000,
                "assistant": 2_500,
            }
        else:
            prompt_budget_chars = max(12000, int(max(4096, context_window - 1536) * 1.05))
            limits = {
                "system": min(6500, max(4500, int(prompt_budget_chars * 0.46))),
                "user": min(7000, max(3200, int(prompt_budget_chars * 0.42))),
                "assistant": min(3200, max(1600, int(prompt_budget_chars * 0.22))),
            }
        if role == "system" and profile not in {
            "contract",
            "contract_grounding",
            "state_report",
            "deep_probe",
        }:
            limits["system"] = InferenceGate._proportionate_scaffold_limit(
                limits["system"],
                visible_request_chars,
            )
        limit = limits.get(role, 8000)
        if profile == "contract" and role == "system":
            return InferenceGate._contract_foreground_system_content(
                clean,
                limit=limit,
            )
        if len(clean) <= limit:
            return clean
        if role in {"system", "user"}:
            marker = "\n…[middle omitted for foreground context budget]…\n"
            critical_excerpt = ""
            if role == "system":
                critical_excerpt = InferenceGate._critical_foreground_system_excerpt(
                    clean,
                    budget=min(2200, max(900, limit // 3)),
                )
            if critical_excerpt:
                remaining = max(2, limit - len(marker) * 2 - len(critical_excerpt))
                head = max(1, remaining * 3 // 5)
                tail = max(1, remaining - head)
                return (
                    f"{clean[:head].rstrip()}{marker}"
                    f"{critical_excerpt}{marker}"
                    f"{clean[-tail:].lstrip()}"
                )
            remaining = max(2, limit - len(marker))
            head = max(1, remaining * 2 // 3)
            tail = max(1, remaining - head)
            return f"{clean[:head].rstrip()}{marker}{clean[-tail:].lstrip()}"
        return clean[: limit - 1].rstrip() + "…"

    @classmethod
    def _stakes_capped_tokens(
        cls,
        max_tokens: int,
        *,
        envelope_cap: int,
        protected: bool,
        completion_floor: int = 0,
        prompt: str,
        context: dict[str, Any],
        reason: str,
    ) -> tuple[int, int]:
        """Apply the viability ceiling with a bounded user-surface override.

        Capability inventories and structurally compound desktop answers cannot
        be made cheaper by truncating them. The override is derived from this
        turn's measured compute profile and completion contract, so it remains
        finite and auditable. Critical memory admission is enforced separately
        before this method; an action-welfare envelope must not turn an admitted
        text decode into an incomplete answer.
        """
        ceiling = max(1, int(envelope_cap))
        if protected:
            floor, _cap, _loops = cls._foreground_compute_profile(str(prompt or ""))
            override = max(ceiling, int(floor), max(0, int(completion_floor)))
            if override > ceiling:
                context["resource_stakes_protected_override"] = {
                    "reason": reason,
                    "envelope_ceiling": ceiling,
                    "override_ceiling": override,
                    # Derived from the request, so the receipt can be checked.
                    "derived_from": (
                        "user_surface_completion_floor"
                        if int(completion_floor) > int(floor)
                        else "foreground_compute_profile_floor"
                    ),
                }
            ceiling = override
        return min(int(max_tokens), ceiling), ceiling

    def prompt_fit_receipt(self) -> dict[str, Any]:
        """What the last dispatch had to trim to fit the context window."""
        return copy.deepcopy(getattr(self, "_prompt_fit_receipt", {}))

    def _fit_prompt_to_window(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        *,
        answer_tokens: int,
        origin: str | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Last word on prompt size, denominated in tokens.

        Everything upstream budgets in CHARACTERS — profile limits, scaffold
        ratios, truncation — while the thing being budgeted is a context window
        measured in TOKENS. Four characters per token is roughly right for
        English prose and wrong for code, punctuation-dense text and non-Latin
        scripts, always in the direction that overflows. And prebuilt message
        payloads skipped the compactor entirely on several routes: the total
        was logged and never checked against anything.

        The person's own words are never trimmed here. System scaffold is,
        largest first, because the scaffold is what grew.
        """
        from .inference_gate import (
            _record_inference_degradation,
        )

        window = self._foreground_prompt_context_window()
        reserve = max(0, int(answer_tokens))
        allowed = window - reserve
        receipt: dict[str, Any] = {
            "window": window,
            "reserved_for_answer": reserve,
            "allowed": allowed,
            "trimmed": [],
            "origin": str(origin or ""),
        }

        def _cost(text: Any) -> int:
            return estimate_context_tokens(str(text or ""))

        def _total() -> int:
            return _cost(system_prompt) + sum(
                _cost(message.get("content")) for message in messages
            )

        total = _total()
        receipt["tokens_before"] = total
        receipt["history_messages_before"] = sum(
            message.get("role") in {"user", "assistant"} for message in messages
        )
        receipt["omitted_exchanges"] = []
        # A complete contiguous suffix preserves dialogue references. Apply
        # this once, at serving capacity, rather than at each profile's soft
        # latency budget. Keep the latest exchange for scaffold fitting below.
        while allowed > 0 and total > allowed:
            user_indices = [
                index for index, message in enumerate(messages)
                if message.get("role") == "user"
            ]
            if len(user_indices) < 3:
                break
            start, end = user_indices[:2]
            indices = [
                index for index in range(start, end)
                if messages[index].get("role") in {"user", "assistant"}
            ]
            receipt["omitted_exchanges"].append({
                "messages": len(indices),
                "estimated_tokens": sum(_cost(messages[index].get("content")) for index in indices),
                "reason": "serving_context_capacity",
            })
            messages = [message for index, message in enumerate(messages) if index not in indices]
            total = _total()
        receipt["history_messages_after"] = sum(
            message.get("role") in {"user", "assistant"} for message in messages
        )
        if receipt["omitted_exchanges"]:
            logger.info(
                "Conversation allocation: retained %d of %d dialogue messages; "
                "omitted %d complete exchanges for serving capacity (%d estimated input tokens).",
                receipt["history_messages_after"], receipt["history_messages_before"],
                len(receipt["omitted_exchanges"]), allowed,
            )
        if allowed <= 0 or total <= allowed:
            receipt["tokens_after"] = total
            receipt["fits"] = total <= allowed
            self._prompt_fit_receipt = receipt
            return system_prompt, messages

        # Trim system scaffold, largest first. Index -1 stands for the
        # separately-passed system_prompt, which the client merges into
        # messages[0] and which is therefore part of the same prefill.
        trimmable: list[tuple[int, int]] = []
        if system_prompt:
            trimmable.append((-1, _cost(system_prompt)))
        for index, message in enumerate(messages):
            if str(message.get("role", "")).strip().lower() not in {
                "system",
                "runtime_evidence",
            }:
                continue
            trimmable.append((index, _cost(message.get("content"))))
        trimmable.sort(key=lambda entry: entry[1], reverse=True)

        for index, cost in trimmable:
            overflow = _total() - allowed
            if overflow <= 0:
                break
            keep_tokens = max(0, cost - overflow)
            text = str(
                system_prompt if index < 0 else messages[index].get("content", "") or ""
            )
            if not text:
                continue
            # Tokens back to characters using this text's own measured ratio,
            # not a global assumption: a block of dense code and a block of
            # prose do not convert at the same rate.
            chars_per_token = len(text) / max(1, cost)
            keep_chars = int(keep_tokens * chars_per_token)
            marker = self._PROMPT_FIT_MARKER
            if keep_chars <= len(marker) + 2:
                trimmed = ""
            else:
                room = keep_chars - len(marker)
                head = max(1, room * 2 // 3)
                tail = max(1, room - head)
                trimmed = f"{text[:head].rstrip()}{marker}{text[-tail:].lstrip()}"
            if index < 0:
                system_prompt = trimmed
            else:
                messages[index] = {**messages[index], "content": trimmed}
            receipt["trimmed"].append(
                {
                    "index": index,
                    "tokens_before": cost,
                    "tokens_after": _cost(trimmed),
                }
            )

        total = _total()
        receipt["tokens_after"] = total
        receipt["fits"] = total <= allowed
        if not receipt["fits"]:
            # Everything trimmable has been trimmed and it still does not fit,
            # which means the person's own words plus the answer budget exceed
            # the window. The serving runtime will truncate from one end and
            # answer a question it only partly received; say so rather than
            # letting it happen quietly.
            _record_inference_degradation(
                RuntimeError(
                    f"prompt does not fit the serving context window: "
                    f"{total} tokens against {allowed} allowed"
                ),
                action="dispatched an over-window prompt the serving runtime will truncate",
                severity="error",
                extra=receipt,
            )
        self._prompt_fit_receipt = receipt
        return system_prompt, messages

    def _grounding_char_budget(self, context: Any, messages: Any) -> int:
        """How much room the volatile grounding has on this turn."""
        from .inference_gate import (
            _INFERENCE_RECOVERABLE_ERRORS,
        )

        if isinstance(context, dict):
            visible = str(context.get("visible_user_message") or "")
            if self._foreground_prompt_profile(visible, context) == "state_report":
                used = sum(
                    len(str(msg.get("content", "") or ""))
                    for msg in (messages or ())
                    if isinstance(msg, dict)
                )
                available = max(
                    self._GROUNDING_FLOOR_CHARS,
                    self._STATE_REPORT_TOTAL_BUDGET_CHARS - used,
                )
                return min(self._STATE_REPORT_GROUNDING_BUDGET_CHARS, available)
        try:
            constrained = bool(self._has_short_live_output_contract(context))
        except _INFERENCE_RECOVERABLE_ERRORS:
            constrained = False
        if not constrained:
            return self._GROUNDING_DEFAULT_BUDGET_CHARS
        used = sum(
            len(str(msg.get("content", "") or ""))
            for msg in (messages or ())
            if isinstance(msg, dict)
        )
        return max(self._GROUNDING_FLOOR_CHARS, 2_800 - used)

    @staticmethod
    def _fit_grounding_blocks(
        *,
        contract_blocks: list[str],
        task_blocks: list[str],
        ambient_blocks: list[str],
        limit: int,
    ) -> str:
        """Fit complete evidence blocks using declared semantic priority.

        Call order is not authority. Turn contracts and task-specific evidence
        precede ambient state even when ambient collectors happen to run first.
        A trailing block is dropped whole because a partial readout can change
        the meaning of the evidence it carries.
        """
        kept: list[str] = []
        spent = 0
        ordered_blocks = [*contract_blocks, *task_blocks, *ambient_blocks]
        for block in ordered_blocks:
            text = str(block or "").strip()
            if not text:
                continue
            cost = len(text) + (2 if kept else 0)
            if kept and spent + cost > limit:
                continue
            kept.append(text)
            spent += cost
        if not kept:
            return ""
        joined = "\n\n".join(kept)
        # A single block larger than the whole allowance still has to fit.
        return joined if len(joined) <= limit else joined[: max(1, limit - 1)].rstrip()

    def _compact_prebuilt_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        history_limit: int | None = None,
        deep_probe: bool = False,
        budget_profile: str = "standard",
        current_user_content: str | None = None,
    ) -> list[dict[str, str]]:
        """Compact scaffolding without applying a second dialogue budget.

        Foreground history is allocated by `_fit_prompt_to_window`, with the
        output reserve known. Background callers may request a count window.
        """
        if not isinstance(messages, list):
            return []

        requested_profile = str(budget_profile or "standard").lower()
        profile = "deep_probe" if deep_probe else requested_profile
        latest_user_position = next(
            (
                idx
                for idx in range(len(messages) - 1, -1, -1)
                if isinstance(messages[idx], dict)
                and str(messages[idx].get("role", "") or "").strip().lower() == "user"
            ),
            None,
        )
        latest_user_content = ""
        if latest_user_position is not None:
            latest_user_content = str(
                messages[latest_user_position].get("content", "") or ""
            ).strip()
        contract_user_content = latest_user_content
        if requested_profile == "contract" and current_user_content:
            visible = str(current_user_content or "").strip()
            continuity_prefix = "[CURRENT USER MESSAGE]\n"
            internal_suffix_markers = (
                "\n\n[GROUNDING EVIDENCE FOR THIS TURN]\n",
                "\n\n[RECENT COMPLETED CONVERSATION FOR CONTINUITY ONLY]\n",
                "\n\n[LIVE DESKTOP FULL-MIND CONTRACT]\n",
                "\n\n[LIVE DESKTOP TURN EVIDENCE]\n",
            )

            def _visible_precedes_only_internal_suffix(candidate: str) -> bool:
                if candidate == visible:
                    return True
                if not visible or not candidate.startswith(visible):
                    return False
                suffix = candidate[len(visible) :]
                return any(suffix.startswith(marker) for marker in internal_suffix_markers)

            unwrapped_candidate = latest_user_content
            if latest_user_content.startswith(continuity_prefix):
                unwrapped_candidate = latest_user_content[len(continuity_prefix) :]
            marker_positions = [
                unwrapped_candidate.index(marker)
                for marker in internal_suffix_markers
                if marker in unwrapped_candidate
            ]
            if marker_positions:
                unwrapped_candidate = unwrapped_candidate[: min(marker_positions)].strip()
            if _visible_precedes_only_internal_suffix(unwrapped_candidate):
                contract_user_content = visible
        # The output contract survives a long input.
        #
        # A short output contract does not imply a short input, so a contract
        # turn whose user message runs long takes the standard INPUT budget.
        # But `profile` also selected the system builder, so the downgrade used
        # to drop `_contract_foreground_system_content` as well — and the
        # contract is the reason this route was chosen. A long question lost the
        # very output contract that routed it, which is the case where the
        # contract matters most.
        contract_output_profile = requested_profile == "contract"
        if profile == "contract":
            if len(contract_user_content) > 1_000:
                profile = "standard"
            else:
                latest_user_content = contract_user_content
        # The person's own words for this turn — never the user-role message,
        # which by this point also carries the grounding evidence the route
        # injected. Measured live: a 175-char question arrived as a 2,783-char
        # user block, so sizing the scaffold against the block would have
        # measured the scaffold against other scaffold.
        visible_request_chars = len(
            str(current_user_content or latest_user_content or "").strip()
        )
        system_message: dict[str, str] | None = None
        preserved_system_messages: list[dict[str, str]] = []
        convo: list[dict[str, str]] = []
        for message_position, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "") or "").strip().lower()
            grounding_system = bool(
                system_message is not None
                and role in {"system", "runtime_evidence"}
                and self._is_grounding_system_message(msg)
            )
            content_source = msg.get("content", "")
            if (
                requested_profile == "contract"
                and message_position == latest_user_position
                and latest_user_content
            ):
                content_source = latest_user_content
            if grounding_system and contract_output_profile:
                message_profile = "contract_grounding"
            elif role == "system" and contract_output_profile:
                # The system block keeps the contract builder even when the
                # input budget was widened above.
                message_profile = "contract"
            else:
                message_profile = profile
            if role in {"user", "assistant"} and message_position != latest_user_position:
                # Retained dialogue is source evidence. Allocate whole turns
                # below instead of truncating code or deleting answer tails.
                content = str(content_source or "").strip()
            else:
                content = self._compact_prebuilt_message_content(
                    role,
                    content_source,
                    budget_profile=message_profile,
                    visible_request_chars=visible_request_chars,
                )
            if not content:
                continue
            normalized = {
                "role": "runtime_evidence" if grounding_system else role or "user",
                "content": content,
            }
            if role == "system" and system_message is None:
                system_message = normalized
            elif grounding_system:
                preserved_system_messages.append(normalized)
            elif role in {"user", "assistant"}:
                convo.append(normalized)

        if deep_probe and system_message is not None:
            content = str(system_message.get("content", "") or "")
            if len(content) > 5200:
                system_message["content"] = content[:5199].rstrip() + "…"

        compact: list[dict[str, str]] = []
        if system_message is not None:
            compact.append(system_message)
        history_start = (
            0 if history_limit is None
            else max(0, len(convo) - max(1, int(history_limit)))
        )
        # A count window may land inside an exchange. Include its initiating
        # question; the total character budget still bounds the final window.
        while history_start > 0 and convo[history_start]["role"] == "assistant":
            history_start -= 1
        compact.extend(convo[history_start:])
        if not deep_probe and preserved_system_messages:
            # Grounding (LIVE MIND CONTEXT, phenomenal/body state, tool and skill
            # results) is rebuilt every turn. Placed AHEAD of the history it made
            # the prompt diverge at block two, so the reusable prefix ended after
            # the system message and the whole conversation was re-prefilled from
            # token zero on every turn — >80s to a first token by the time the
            # history was real, which is the deadline that produced "I couldn't
            # get to an answer I'd stand behind" (2026-07-26). Raising the KV
            # cache budget could never help: the entries had no stable prefix to
            # hit. Volatile content belongs last, so `system + history` stays
            # byte-identical across turns and the cache actually reuses it.
            #
            # It still lands immediately before the newest user message, so the
            # question is answered with the grounding in the most recent context.
            newest_user = next(
                (
                    idx
                    for idx in range(len(compact) - 1, -1, -1)
                    if compact[idx].get("role") == "user"
                ),
                None,
            )
            compact.insert(
                len(compact) if newest_user is None else newest_user,
                preserved_system_messages[-1],
            )

        context_window = self._foreground_prompt_context_window()
        if profile == "contract":
            total_budget_chars = 2_800
        elif profile == "state_report":
            # The remaining 1,400 characters are reserved for the canonical
            # state projection appended after stable-prefix compaction.
            total_budget_chars = 2_800
        elif profile == "simple":
            total_budget_chars = min(
                9000,
                max(7000, int(max(4096, context_window - 1536) * 0.62)),
            )
        elif profile == "extended":
            total_budget_chars = max(18000, int(max(4096, context_window - 1536) * 1.75))
        elif profile == "curriculum":
            total_budget_chars = 12_000
        elif profile == "background":
            total_budget_chars = 16_000
        else:
            total_budget_chars = max(12000, int(max(4096, context_window - 1536) * 1.05))
        if deep_probe:
            total_budget_chars = min(total_budget_chars, 9000)

        if history_limit is None:
            return compact

        while (
            compact
            and sum(len(str(msg.get("content", "") or "")) for msg in compact) > total_budget_chars
        ):
            latest_user_index = next(
                (
                    idx
                    for idx in range(len(compact) - 1, -1, -1)
                    if compact[idx].get("role") == "user"
                ),
                None,
            )
            dialogue_indices = [
                idx for idx, msg in enumerate(compact)
                if msg.get("role") in {"user", "assistant"}
                and (latest_user_index is None or idx < latest_user_index)
            ]
            if dialogue_indices:
                first = dialogue_indices[0]
                following_user = next(
                    (idx for idx in dialogue_indices[1:] if compact[idx]["role"] == "user"),
                    latest_user_index if latest_user_index is not None else len(compact),
                )
                for idx in reversed(dialogue_indices):
                    if first <= idx < following_user:
                        compact.pop(idx)
                continue
            removable_index = None
            if removable_index is None:
                for idx, msg in enumerate(compact):
                    if idx == 0 and msg.get("role") == "system":
                        continue
                    if idx == latest_user_index:
                        continue
                    removable_index = idx
                    break
            if removable_index is None:
                break
            compact.pop(removable_index)

        total_chars = sum(len(str(msg.get("content", "") or "")) for msg in compact)
        if compact and total_chars > total_budget_chars:
            first = compact[0]
            if first.get("role") == "system":
                overflow = total_chars - total_budget_chars
                content = str(first.get("content", "") or "")
                if profile == "contract":
                    min_system_chars = 1_000
                elif profile == "state_report":
                    min_system_chars = 1_200
                else:
                    min_system_chars = 3200 if profile == "simple" else 4200
                new_limit = max(min_system_chars, len(content) - overflow - 1)
                if len(content) > new_limit:
                    first["content"] = self._compact_prebuilt_message_content(
                        "system",
                        content,
                        budget_profile=profile,
                    )
                    if len(first["content"]) > new_limit:
                        marker = "\n…[middle omitted for total prompt budget]…\n"
                        critical_excerpt = self._critical_foreground_system_excerpt(
                            content,
                            budget=min(2200, max(900, new_limit // 3)),
                        )
                        if critical_excerpt:
                            remaining = max(
                                2,
                                new_limit - len(marker) * 2 - len(critical_excerpt),
                            )
                            head = max(1, remaining * 3 // 5)
                            tail = max(1, remaining - head)
                            first["content"] = (
                                f"{content[:head].rstrip()}{marker}"
                                f"{critical_excerpt}{marker}"
                                f"{content[-tail:].lstrip()}"
                            )
                        else:
                            remaining = max(2, new_limit - len(marker))
                            head = max(1, remaining * 2 // 3)
                            tail = max(1, remaining - head)
                            first["content"] = (
                                f"{content[:head].rstrip()}{marker}{content[-tail:].lstrip()}"
                            )

        return compact
