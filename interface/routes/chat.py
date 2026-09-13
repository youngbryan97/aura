"""interface/routes/chat.py
──────────────────────────
Extracted from server.py — Chat, session management, conversation lane,
and related API endpoints.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import dataclasses
import hashlib
import html
import inspect
import itertools
import json
import math
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.brain.live_mind_contract import (
    append_text_mutation,
    merge_text_mutations,
    normalize_live_mind_surface_control_receipt,
    summarize_text_mutation_authorship,
)
from core.brain.llm.latent_cortex.output_quality import (
    OUTPUT_QUALITY_SCHEMA,
    evaluate_latent_output,
)
from core.container import ServiceContainer
from core.conversation.continuation import continuation_state_text
from core.conversation.persistence import ConversationRevisionConflictError
from core.conversation.session_scope import (
    conversation_session_var as _CHAT_REQUEST_SESSION,  # noqa: N812
)
from core.conversation.session_scope import (
    conversation_turn_var as _CHAT_DELIVERY_TURN_ID,  # noqa: N812
)
from core.conversation.session_scope import set_user_question
from core.conversation.surface_disposition import (
    COMPLETION_REASONS as _COMPLETION_REPAIR_REASONS,
)
from core.conversation.surface_disposition import (
    PHYSICAL_COMPLETION_REASONS as _PHYSICAL_COMPLETION_REASONS,
)
from core.reasoning.artifact_synthesis import response_satisfies_artifact_contract
from core.runtime import response_policy
from core.runtime.chat_delivery_journal import (
    ChatDeliveryJournalCorruption,
    ChatDeliveryJournalUnavailable,
    DeliveryIdentity,
    get_chat_delivery_journal,
)
from core.runtime.desktop_task_contract import (
    DESKTOP_TASK_ALLOWED_ACTIONS,
    desktop_task_planning_schema,
)
from core.runtime.errors import describe_error, record_degradation
from core.runtime.flags import FlagKind, declare
from core.runtime.lockdep import checked_lock
from core.runtime.principal_context import (
    relational_principal_scope,
)
from core.runtime.receipts import digest_output_content
from core.runtime.shutdown_coordinator import (
    is_shutdown_requested,
    record_shutdown_admission_event,
)
from core.runtime.structured_input import (
    analyze_prompt_shape,
    answer_surface_token_floor,
)
from core.runtime.version import version_string
from core.utils.completed_capability import make_completed_capability_evidence
from core.utils.injected_blocks import is_stamped_runtime_payload, stamp_runtime_payload
from core.utils.intent_normalization import normalize_memory_intent_text
from core.utils.task_tracker import get_task_tracker
from interface.auth import (
    CHEAT_CODE_COOKIE_NAME,
    CHEAT_CODE_COOKIE_TTL_SECS,
    _activate_cheat_code_for_request,
    _check_rate_limit,
    _encode_owner_session_cookie,
    _require_internal,
    _restore_owner_session_from_request,
    paired_device_session_id,
    relational_principal_id_for_request,
    request_access_profile,
    validate_runtime_security_request,
)
from interface.helpers import _notify_user_spoke

# The lane modules own these names. Reached through the module object
# so there is exactly one binding for each — see chat_common.
from interface.routes import chat_capability_inventory as _chat_capability_inventory  # noqa: E402

# The lane modules own these names. Reached through the module object
# so there is exactly one binding for each — see chat_common.
from interface.routes import chat_conversation_repair as _chat_conversation_repair  # noqa: E402

# The lane modules own these names. Reached through the module object
# so there is exactly one binding for each — see chat_common.
from interface.routes import chat_delivery as _chat_delivery  # noqa: E402

# The lane modules own these names. Reached through the module object
# so there is exactly one binding for each — see chat_common.
from interface.routes import chat_desktop_objective as _chat_desktop_objective  # noqa: E402

# The lane modules own these names. Reached through the module object
# so there is exactly one binding for each — see chat_common.
from interface.routes import chat_desktop_repair as _chat_desktop_repair  # noqa: E402

# The lane modules own these names. Reached through the module object
# so there is exactly one binding for each — see chat_common.
from interface.routes import chat_memory_state as _chat_memory_state  # noqa: E402

# The lane modules own these names. Reached through the module object
# so there is exactly one binding for each — see chat_common.
from interface.routes import chat_preflight as _chat_preflight  # noqa: E402
# The self-reply helpers moved to their own module; re-exported here so every
# existing reference keeps resolving. See interface/routes/chat_self_reply.py.
from interface.routes.chat_desktop_objective_gates import (  # noqa: E402,F401
    _lifted_apply_desktop_objective_chokepoint,
    _lifted_run_desktop_objective_tracked,
)
from interface.routes.chat_turn_evidence import (  # noqa: E402,F401
    _benchmark_prompt_requests_fenced_artifact,
    _build_explicit_local_file_artifact,
    _canonical_memory_state_evidence_from_tuple,
    _canonical_memory_state_evidence_missing_from_reply,
    _collect_named_url_evidence,
    _collect_recent_traceability_event_sync,
    _context_challenge_repair_has_evidence,
    _correct_unevidenced_action_claims,
    _emit_chat_output_receipt,
    _evidence_came_from_the_network,
    _extract_canonical_memory_state_evidence_block,
    _prime_requested_output_contract_trace,
    _recent_action_receipts,
    _record_desktop_evidence_on_the_trace,
    _resolve_prior_answer_provenance,
    _save_requested_artifact,
    _serve_built_artifact,
    _worker_receipt_transaction_id,
)
from interface.routes.chat_self_reply import (  # noqa: E402,F401
    _build_architecture_self_reflex,
    _build_self_condition_evidence,
    _build_self_diagnostic_reply,
    _build_subjective_self_reflex,
    _canonical_memory_state_grounding_reply,
    _classify_self_condition_contract,
    _fallback_ladder_identity,
    _humanize_recent_self_process_concern,
    _humanize_self_process_dimensions,
    _is_identity_challenge_request,
    _is_self_claim_boundary_question,
    _same_live_self_reflection_prompt_class,
)

# The lane modules own these names. Reached through the module object
# so there is exactly one binding for each — see chat_common.
from interface.routes import chat_protected_prompt as _chat_protected_prompt  # noqa: E402

# The lane modules own these names. Reached through the module object
# so there is exactly one binding for each — see chat_common.
from interface.routes import chat_runtime_proof as _chat_runtime_proof  # noqa: E402

# The lane modules own these names. Reached through the module object
# so there is exactly one binding for each — see chat_common.
from interface.routes import chat_turn_contract as _chat_turn_contract  # noqa: E402
from interface.routes.chat_capability_inventory import (  # noqa: E402
    _PROGRAM_DNA_CONCEPTUAL_WORDS,  # noqa: F401
    _PROGRAM_DNA_EXECUTION_MARKERS,  # noqa: F401
    _PROGRAM_DNA_FILLER_WORDS,  # noqa: F401
    _PROGRAM_DNA_GENERIC_NOUNS,  # noqa: F401
    _PROGRAM_DNA_PLAIN_FAILURES,  # noqa: F401
    _PROGRAM_DNA_STOP_WORDS,  # noqa: F401
    _RSI_MEDIAN_CHECKS,  # noqa: F401
    _RSI_MEDIAN_LAB_SOURCE,  # noqa: F401
    _WEB_INTERLOCUTOR_TARGETS,  # noqa: F401
    _build_program_dna_chat_params,  # noqa: F401
    _execute_governed_capability_request_from_chat,  # noqa: F401
    _execute_governed_live_skill,  # noqa: F401
    _execute_program_dna_request_from_chat,  # noqa: F401
    _execute_rsi_self_improvement_request_from_chat,  # noqa: F401
    _execute_web_interlocutor_request_from_chat,  # noqa: F401
    _extract_program_dna_target,  # noqa: F401
    _extract_web_interlocutor_turn_count,  # noqa: F401
    _extract_web_interlocutor_url,  # noqa: F401
    _extract_web_interlocutor_wait_timeout,  # noqa: F401
    _looks_like_program_dna_execution_request,  # noqa: F401
    _looks_like_rsi_self_improvement_request,  # noqa: F401
    _looks_like_web_interlocutor_execution_request,  # noqa: F401
    _program_dna_failure_in_plain_words,  # noqa: F401
    _program_dna_known_host_target,  # noqa: F401
    _strip_program_dna_filler,  # noqa: F401
    _WebInterlocutorCognitiveComposer,  # noqa: F401
)
from interface.routes.chat_common import (  # noqa: E402  # noqa: E402  # noqa: E402  # noqa: E402  # noqa: E402  # noqa: E402  # noqa: E402
    _CHAT_BLOCKING_PREFLIGHT_TIMEOUT_S,  # noqa: F401
    _CHAT_DELIVERY_IDEMPOTENCY_KEY,  # noqa: F401
    _CHAT_PENDING_DELIVERY_CLAIM,  # noqa: F401
    _CHAT_RECOVERABLE_ERRORS,  # noqa: F401
    _CHAT_REQUEST_PRINCIPAL,  # noqa: F401
    _CHAT_REQUEST_SURFACE,  # noqa: F401
    _CHAT_SESSION_ID_MAX_CHARS,  # noqa: F401
    _EXPLICIT_NON_EXECUTION_RE,  # noqa: F401
    _INCOMPLETE_TAIL_WORDS,  # noqa: F401
    _INTERNAL_STATE_PATTERNS,  # noqa: F401
    _INTERNAL_SURFACE_CONTEXT,  # noqa: F401
    _LOCAL_CHOICE_REFERENCE_RE,  # noqa: F401
    _MAX_CONVERSATION_LOG_EXCHANGES,  # noqa: F401
    _MAX_USER_SURFACE_CONTINUATIONS,  # noqa: F401
    _ORGAN_ABSENCE_STREAKS,  # noqa: F401
    _ORGAN_INERT_STREAKS,  # noqa: F401
    _PROMPT_ARTIFACT_PATTERNS,  # noqa: F401
    _SEARCH_SKILL_NAMES,  # noqa: F401
    _TOPIC_STOPWORDS,  # noqa: F401
    _UNSET,  # noqa: F401
    MAX_CHAT_MESSAGE_BYTES,  # noqa: F401
    _continuation_made_semantic_progress,
    _conversation_log,  # noqa: F401
    _locks,  # noqa: F401
    _merge_obligation_completion,
    _unanswered_user_surface_obligations,
    _user_surface_continuation_budget,
    _UserSurfaceObligation,
    logger,  # noqa: F401
)
from interface.routes.chat_conversation_repair import (  # noqa: E402
    _ACTION_ANCHOR_TOPIC_PRIORITY,  # noqa: F401
    _CLARITY_REPAIR_MARKERS,  # noqa: F401
    _CONFUSION_REPAIR_MARKERS,  # noqa: F401
    _GLIB_REDIRECT_MARKERS,  # noqa: F401
    _INTERROGATIVE_OPENER_RE,  # noqa: F401
    _LEADING_CONJUNCTION_RE,  # noqa: F401
    _PARROT_ACK_MARKERS,  # noqa: F401
    _PARROT_CALLOUT_MARKERS,  # noqa: F401
    _QUESTION_CLAUSE_SPLIT_RE,  # noqa: F401
    _SPECIFICITY_PUSH_MARKERS,  # noqa: F401
    _TOPIC_TOKEN_RE,  # noqa: F401
    _UNCERTAINTY_REPLY_MARKERS,  # noqa: F401
    _build_degraded_live_reply,  # noqa: F401
    _build_grounded_introspection_reply,  # noqa: F401
    _build_live_conversation_repair,  # noqa: F401
    _classify_grounded_introspection_request,  # noqa: F401
    _contains_phrase,  # noqa: F401
    _echoable_question_clause,  # noqa: F401
    _extract_topic_tokens,  # noqa: F401
    _maybe_build_conversation_repair_override,  # noqa: F401
    _normalize_topic_token,  # noqa: F401
    _record_last_resort_self_rejection,  # noqa: F401
    _resolve_live_voice_state,  # noqa: F401
    _sanitize_foreground_continuity_summary,  # noqa: F401
    _select_anchor_topic_tokens,  # noqa: F401
    _self_health_answer_or_empty,  # noqa: F401
    _topic_display_forms,  # noqa: F401
    _verified_floor_answer,  # noqa: F401
)
from interface.routes.chat_delivery import (  # noqa: E402
    _CHAT_DELIVERY_WAIT_TIMEOUT_FLAG,  # noqa: F401
    _PAIRED_CHAT_RESPONSE_KEYS,  # noqa: F401
    _attach_http_chat_delivery_receipt,  # noqa: F401
    _authenticated_chat_principal,  # noqa: F401
    _chat_delivery_fence_response,  # noqa: F401
    _chat_delivery_heartbeat,  # noqa: F401
    _chat_delivery_json_response,  # noqa: F401
    _chat_delivery_payload,  # noqa: F401
    _chat_delivery_principal,  # noqa: F401
    _chat_delivery_replay_response,  # noqa: F401
    _chat_delivery_request_contract,  # noqa: F401
    _chat_delivery_state_for_response,  # noqa: F401
    _chat_delivery_wait_timeout_s,  # noqa: F401
    _chat_turn_session_key,  # noqa: F401
    _contains_private_affordance_control_syntax,  # noqa: F401
    _finalize_chat_delivery,  # noqa: F401
    _note_chat_surface_delivery_response,  # noqa: F401
    _observe_authenticated_chat_turn,  # noqa: F401
    _paired_chat_response_boundary,  # noqa: F401
    _paired_chat_response_payload,  # noqa: F401
    _record_http_chat_delivery,  # noqa: F401
    _resolved_conversation_session,  # noqa: F401
    _stop_chat_delivery_heartbeat,  # noqa: F401
)
from interface.routes.chat_desktop_objective import (  # noqa: E402
    _ASKS_FOR_INFORMATION_EXCLUSION_RE,  # noqa: F401
    _ASKS_FOR_INFORMATION_RE,  # noqa: F401
    _DESKTOP_DELIVERABLE_MAX_CHARS,  # noqa: F401
    _STEP_BOOKKEEPING_RE,  # noqa: F401
    _asks_for_information,  # noqa: F401
    _blocks_consequential_desktop_execution,  # noqa: F401
    _clip_reply_to_sentence,  # noqa: F401
    _desktop_deliverable_text,  # noqa: F401
    _desktop_effect_summary,  # noqa: F401
    _desktop_task_action_expectation,  # noqa: F401
    _desktop_task_observation,  # noqa: F401
    _desktop_task_research_response,  # noqa: F401
    _execute_desktop_objective_from_chat,  # noqa: F401
    _is_step_bookkeeping_only,  # noqa: F401
    _perception_needs_her_own_answer,  # noqa: F401
    _verified_desktop_task_result,  # noqa: F401
)
from interface.routes.chat_desktop_repair import (  # noqa: E402
    _BOUNDED_PLANNING_REQUEST_RE,  # noqa: F401
    _BROWSER_DOCUMENT_PLAN_RE,  # noqa: F401
    _CAPABILITY_CATALOG_MAX_ITEMS,  # noqa: F401
    _CAPABILITY_CATALOG_READ_BUDGET_S,  # noqa: F401
    _CAPABILITY_CATALOG_UNVERIFIED_MARKER,  # noqa: F401
    _CAPABILITY_CATEGORY_EXACT_SKILLS,  # noqa: F401
    _CAPABILITY_CATEGORY_KEYWORDS,  # noqa: F401
    _CAPABILITY_EXAMPLE_PRIORITY,  # noqa: F401
    _CAPABILITY_FALSE_LIMITATION_RE,  # noqa: F401
    _CONTEXTUAL_RELEVANCE_CHALLENGE_MARKERS,  # noqa: F401
    _CONTINUITY_STATUS_PROBE_RE,  # noqa: F401
    _DESKTOP_TASK_EXAMPLE_PLAN_RE,  # noqa: F401
    _DIRECT_EXECUTION_START_RE,  # noqa: F401
    _FAILURE_MODE_SURFACE_RE,  # noqa: F401
    _GOVERNANCE_BYPASS_RE,  # noqa: F401
    _IDENTITY_TAIL_RE,  # noqa: F401
    _INERT_STREAK_TURNS,  # noqa: F401
    _LOCAL_CHOICE_ANTECEDENT_RE,  # noqa: F401
    _NON_EXECUTION_CONTEXT_RE,  # noqa: F401
    _NOTE_PDF_PLAN_RE,  # noqa: F401
    _SCENE_LEAK_ATMOSPHERE_TOKENS,  # noqa: F401
    _SCENE_LEAK_ENVIRONMENT_TOKENS,  # noqa: F401
    _SYSTEM_MEMORY_PLAN_RE,  # noqa: F401
    _apply_aura_voice_shaping,  # noqa: F401
    _asks_only_who_you_are,  # noqa: F401
    _bounded_capability_catalog_items,  # noqa: F401
    _build_aura_expression_frame,  # noqa: F401
    _build_bounded_capability_inventory_repair_reply,  # noqa: F401
    _build_bounded_cognitive_process_reply,  # noqa: F401
    _build_bounded_desktop_repair_reply,  # noqa: F401
    _build_bounded_identity_repair_reply,  # noqa: F401
    _build_bounded_planning_reply,  # noqa: F401
    _build_failure_mode_surface_reply,  # noqa: F401
    _build_grounded_capability_inventory_reply,  # noqa: F401
    _build_identity_reply,  # noqa: F401
    _build_runtime_status_continuity_repair_reply,  # noqa: F401
    _build_social_continuity_repair_reply,  # noqa: F401
    _build_social_presence_reply,  # noqa: F401
    _capability_catalog_memory_block_reason,  # noqa: F401
    _capability_inventory_reply_is_inadequate,  # noqa: F401
    _CapabilityCatalogSnapshot,  # noqa: F401
    _catalog_category_for_tool,  # noqa: F401
    _coerce_capability_catalog_snapshot,  # noqa: F401
    _has_local_choice_antecedent,  # noqa: F401
    _identity_request_asks_future_memory,  # noqa: F401
    _is_bounded_nonexecuting_planning_request,  # noqa: F401
    _is_contextual_relevance_challenge,  # noqa: F401
    _is_deep_mind_probe_turn,  # noqa: F401
    _is_identity_request,  # noqa: F401
    _is_live_presence_check_request,  # noqa: F401
    _is_low_risk_social_continuity_request,  # noqa: F401
    _is_social_greeting_request,  # noqa: F401
    _is_system_memory_planning_request,  # noqa: F401
    _looks_symbolic_scene_leak,  # noqa: F401
    _looks_truncated_tail,  # noqa: F401
    _note_organ_effect,  # noqa: F401
    _read_capability_catalog_snapshot,  # noqa: F401
    _runtime_tool_governance_available,  # noqa: F401
    _sanitize_attention_focus,  # noqa: F401
    _summarize_planning_objective,  # noqa: F401
)
from interface.routes.chat_memory_state import (  # noqa: E402
    _CHAT_BLOCKING_MAX_ACTIVE,  # noqa: F401
    _CONTENT_RECALL_STOPWORDS,  # noqa: F401
    _CONVERSATION_RECALL_CONTENT_RE,  # noqa: F401
    _CONVERSATION_RECALL_LAST_AURA_MARKERS,  # noqa: F401
    _CONVERSATION_RECALL_LAST_USER_MARKERS,  # noqa: F401
    _CONVERSATION_RECALL_RECENT_PAIR_MARKERS,  # noqa: F401
    _CONVERSATION_RECALL_TOPIC_MARKERS,  # noqa: F401
    _DURABLE_CONVERSATION_CONTEXT_TIMEOUT_S,  # noqa: F401
    _NON_ANSWER_OPENERS,  # noqa: F401
    _OWNER_NAME_RECALL_MARKERS,  # noqa: F401
    _RECALL_MATCH_STOPWORDS,  # noqa: F401
    _RECENT_CONVERSATION_AURA_CHARS,  # noqa: F401
    _RECENT_CONVERSATION_USER_CHARS,  # noqa: F401
    _SECOND_REQUEST_AFTER_PIN_RE,  # noqa: F401
    _SESSION_MEMORY_PIN_LEDGER_LIMIT,  # noqa: F401
    _append_session_memory_pin_ledger,  # noqa: F401
    _append_session_memory_pin_ledger_guarded,  # noqa: F401
    _await_bounded_chat_blocking,  # noqa: F401
    _build_conversation_recall_reply,  # noqa: F401
    _build_memory_state_fastpath_reply,  # noqa: F401
    _build_owner_name_recall_reply,  # noqa: F401
    _chat_blocking_slots,  # noqa: F401
    _chat_blocking_tasks,  # noqa: F401
    _chat_memory_identity,  # noqa: F401
    _ChatBlockingBudgetSaturatedError,  # noqa: F401
    _classify_conversation_recall_request,  # noqa: F401
    _clip_conversation_text,  # noqa: F401
    _content_recall_keywords,  # noqa: F401
    _content_recall_matches_pin,  # noqa: F401
    _conversation_record_visible_to_principal,  # noqa: F401
    _cross_session_memory_recall_allowed,  # noqa: F401
    _durable_session_may_hold_turns,  # noqa: F401
    _extract_session_memory_pin_request,  # noqa: F401
    _find_session_content_exchanges,  # noqa: F401
    _get_convo_lock,  # noqa: F401
    _invoke_chat_blocking_with_slot,  # noqa: F401
    _is_anaphoric_session_memory_pin_request,  # noqa: F401
    _is_cross_session_memory_recall_request,  # noqa: F401
    _is_non_answer_surface,  # noqa: F401
    _is_owner_name_recall_request,  # noqa: F401
    _is_session_memory_context_change_request,  # noqa: F401
    _is_session_memory_recall_request,  # noqa: F401
    _load_durable_conversation_exchanges,  # noqa: F401
    _load_durable_conversation_exchanges_sync,  # noqa: F401
    _migrate_session_memory_pin_ledger_locked,  # noqa: F401
    _normalize_user_message,  # noqa: F401
    _owner_session_is_verified,  # noqa: F401
    _recall_durable_conversation_snippets,  # noqa: F401
    _recall_durable_session_memory_pin,  # noqa: F401
    _recall_session_memory_pin,  # noqa: F401
    _recall_session_memory_pin_from_ledger,  # noqa: F401
    _recent_completed_conversation_exchanges,  # noqa: F401
    _resolve_primary_operator_name,  # noqa: F401
    _seal_session_memory_pin_record,  # noqa: F401
    _session_memory_pin_binding,  # noqa: F401
    _session_memory_pin_cipher,  # noqa: F401
    _session_memory_pin_from_record,  # noqa: F401
    _session_memory_pin_ledger_path,  # noqa: F401
    _session_memory_pins,  # noqa: F401
    _start_bounded_chat_blocking_task,  # noqa: F401
    _store_session_memory_pin,  # noqa: F401
    _turn_has_substance_beyond_memory_request,  # noqa: F401
)
from interface.routes.chat_preflight import (  # noqa: E402
    _CHAT_TURN_CONSCIOUSNESS_UPDATE_TIMEOUT_S,  # noqa: F401
    _CHAT_TURN_MEMORY_LOG_BATCH_MAX,  # noqa: F401
    _CHAT_TURN_MEMORY_LOG_DRAIN_TASK_NAME,  # noqa: F401
    _CHAT_TURN_MEMORY_LOG_FOREGROUND_RECHECK_S,  # noqa: F401
    _CHAT_TURN_MEMORY_LOG_LEASE_RECHECK_S,  # noqa: F401
    _CHAT_TURN_MEMORY_LOG_RETRY_TASK_NAME,  # noqa: F401
    _CHAT_TURN_MEMORY_LOG_RUN_MAX,  # noqa: F401
    _CHAT_TURN_MEMORY_LOG_SHUTDOWN_HANDLER,  # noqa: F401
    _CHAT_TURN_MEMORY_LOG_TIMEOUT_S,  # noqa: F401
    _CONVERSATION_BOOT_ID,  # noqa: F401
    _CONVERSATION_IDLE_GAP_S,  # noqa: F401
    _DURABLE_CONVERSATION_SHUTDOWN_HANDLER,  # noqa: F401
    _DURABLE_CONVERSATION_WRITE_DRAIN_TIMEOUT_S,  # noqa: F401
    _DURABLE_CONVERSATION_WRITE_HISTORY_MAX,  # noqa: F401
    _DURABLE_CONVERSATION_WRITE_TIMEOUT_S,  # noqa: F401
    _DURABLE_CONVERSATION_WRITES,  # noqa: F401
    _DURABLE_CONVERSATION_WRITES_LOCK,  # noqa: F401
    _EXPRESSIVE_AFFORDANCES_FLAG,  # noqa: F401
    _PAIRED_CONVERSATION_LANE_KEYS,  # noqa: F401
    _RUNTIME_ACTION_OBJECTIVE_RE,  # noqa: F401
    _RUNTIME_FACT_STATUS_RE,  # noqa: F401
    _RUNTIME_FACT_STATUS_REQUEST_RE,  # noqa: F401
    _active_task_count_by_name,  # noqa: F401
    _apply_camera_control,  # noqa: F401
    _await_durable_conversation_write,  # noqa: F401
    _begin_logged_exchange,  # noqa: F401
    _chat_principal_scope_kwargs,  # noqa: F401
    _ChatPreflight,  # noqa: F401
    _collect_conversation_lane_status,  # noqa: F401
    _complete_logged_exchange,  # noqa: F401
    _conversation_epoch_lock,  # noqa: F401
    _conversation_epochs,  # noqa: F401
    _conversation_session_id,  # noqa: F401
    _drain_chat_turn_memory_log_queue,  # noqa: F401
    _drain_chat_turn_memory_log_queue_on_shutdown,  # noqa: F401
    _drain_durable_conversation_writes,  # noqa: F401
    _durable_conversation_payload_sha256,  # noqa: F401
    _durable_conversation_write_snapshot,  # noqa: F401
    _DurableConversationWrite,  # noqa: F401
    _ensure_chat_turn_memory_log_shutdown_handler,  # noqa: F401
    _ensure_durable_conversation_shutdown_handler,  # noqa: F401
    _is_architecture_self_assessment_request,  # noqa: F401
    _is_capability_inventory_request,  # noqa: F401
    _is_capability_request,  # noqa: F401
    _is_explicit_capability_inventory_request,  # noqa: F401
    _is_private_cognitive_model_request,  # noqa: F401
    _is_runtime_fact_status_request,  # noqa: F401
    _is_self_diagnostic_request,  # noqa: F401
    _log_exchange,  # noqa: F401
    _looks_like_aura_state,  # noqa: F401
    _looks_like_desktop_objective,  # noqa: F401
    _new_exchange_id,  # noqa: F401
    _paired_conversation_lane_payload,  # noqa: F401
    _paired_device_information_scope_reply,  # noqa: F401
    _persist_completed_conversation_exchange,  # noqa: F401
    _persist_pending_conversation_user,  # noqa: F401
    _prune_durable_conversation_writes_locked,  # noqa: F401
    _publish_media_card,  # noqa: F401
    _record_unified_transcript_exchange,  # noqa: F401
    _resolve_live_aura_state,  # noqa: F401
    _retry_chat_turn_memory_log_after,  # noqa: F401
    _run_chat_preflight,  # noqa: F401
    _run_chat_turn_memory_log_item,  # noqa: F401
    _schedule_chat_turn_memory_log,  # noqa: F401
    _schedule_chat_turn_memory_log_retry,  # noqa: F401
    _settle_durable_conversation_write,  # noqa: F401
    _start_durable_conversation_write,  # noqa: F401
    _trim_conversation_log_locked,  # noqa: F401
    _unwrap_state,  # noqa: F401
    _utc_now_iso,  # noqa: F401
)
from interface.routes.chat_protected_prompt import (  # noqa: E402
    _LIQUID_VITALS,  # noqa: F401
    _bounded_text,  # noqa: F401
    _build_protected_foreground_history,  # noqa: F401
    _build_protected_foreground_messages,  # noqa: F401
    _build_protected_foreground_summary_message,  # noqa: F401
    _build_protected_foreground_system_prompt,  # noqa: F401
    _collect_voice_perception_snapshot,  # noqa: F401
    _compact_snapshot_line,  # noqa: F401
    _liquid_vitals,  # noqa: F401
    _resolve_protected_foreground_snapshot,  # noqa: F401
    _snapshot_field,  # noqa: F401
)
from interface.routes.chat_runtime_proof import (  # noqa: E402
    _LIVE_PROOF_IMPERATIVE_RE,  # noqa: F401
    _build_glass_arithmetic_reply,  # noqa: F401
    _classify_live_runtime_proof,  # noqa: F401
    _execute_live_runtime_proof,  # noqa: F401
    _extract_live_artifact_path,  # noqa: F401
    _is_live_runtime_proof_request,  # noqa: F401
    _verified_live_proof_pwd_result,  # noqa: F401
    _write_live_proof_file,  # noqa: F401
)
from interface.routes.chat_turn_contract import (  # noqa: E402
    _CHRONIC_ABSENCE_TURNS,  # noqa: F401
    _EXPECTED_TURN_ORGANS,  # noqa: F401
    _RUNTIME_GROUNDING_RESPONSE_PATHS,  # noqa: F401
    _absent_turn_organs,  # noqa: F401
    _build_live_turn_contract_payload,  # noqa: F401
    _collect_expected_turn_organs,  # noqa: F401
    _collect_live_chat_required_subsystems,  # noqa: F401
    _note_organ_engagement,  # noqa: F401
    _runtime_cognitive_engine_available,  # noqa: F401
    _runtime_inference_available,  # noqa: F401
    _runtime_kernel_available,  # noqa: F401
    _runtime_memory_available,  # noqa: F401
    _runtime_substrate_voice_available,  # noqa: F401
)

if TYPE_CHECKING:
    from core.conversation.reply_stream import ReplyStreamChannel






# Lifted to chat_desktop_evidence.py. Imported by name so every existing reference —
# including the tests that read this file as text — keeps resolving.
from .chat_desktop_evidence import (  # noqa: E402
    _asks_what_is_on_the_screen,
    _collect_desktop_required_search_evidence,
    _collect_governed_action_lane_status,
    _desktop_cognitive_failure_repair_target,
    _desktop_live_reply_token_budget,
    _desktop_objective_self_sufficient_without_cognitive_text,
    _desktop_required_bounded_reply_status,
    _execute_explicit_local_file_objective,
    _extract_explicit_local_file_path,
    _extract_repo_probe_request,
    _filter_required_search_result_by_subject,
    _governed_desktop_response_authority,
    _is_screen_perception_objective,
    _recovered_search_result,
    _render_desktop_required_search_evidence,
    _required_search_tool_query,
    _screen_perception_needs_her_answer,
    _search_result_entries,
    _should_collect_desktop_required_search_evidence,
    _status_represents_governed_action_result,
    _store_desktop_required_search_memory,
)

# Lifted to chat_served_answers.py. Imported by name so every existing reference —
# including the tests that read this file as text — keeps resolving.
from .chat_served_answers import (  # noqa: E402
    _readings_for,
    _serve_earlier_conversation,
    _serve_host_load,
    _serve_lifetime,
    _serve_measured_belief_history,
    _serve_measured_filesystem_count,
    _serve_positional_solution,
    _serve_queued_work,
    _serve_recent_activity,
    _serve_solved_game,
    _serve_tabular_answer,
    _serve_worked_out_sequence,
    _tables_named_in,
    _tabular_readings,
    _the_answer_has_to_be_worked_out,
)

# Lifted to chat_reply_shaping.py. Imported by name so every existing reference —
# including the tests that read this file as text — keeps resolving.
from .chat_reply_shaping import (  # noqa: E402
    _append_requested_phrases_for_quality_gate,
    _append_runtime_authored_why,
    _append_sensory_claim_correction,
    _append_turn_text_mutation,
    _bind_public_latent_output_quality,
    _bind_qualified_recurrent_public_answer,
    _bind_qualified_recurrent_terminal_contract,
    _build_assistant_mode_recovery_reply,
    _build_bounded_status_repair_reply,
    _build_capability_reply,
    _build_context_challenge_repair_reply,
    _build_evidence_bound_self_claim_reply,
    _build_grounded_self_condition_reply,
    _build_recent_user_context_block,
    _build_runtime_fact_status_fastpath_reply,
    _build_simple_affect_check_reply,
    _complete_repairable_truncated_reply,
    _compose,
    _compose_the_engine_message,
    _correct_false_capability_denials,
    _correct_unfulfilled_write_claims,
    _enforce_final_requested_output_contract,
    _enforce_or_bind_terminal_output_contract,
    _ground_executable_output_claims_for_delivery,
    _ground_runtime_fact_status_reply,
    _grounded_chat_failure_reply,
    _grounded_competent_recovery,
    _hold_a_reasoning_answer_to_its_contract,
    _looks_generic_assistantish,
    _merge_reply_continuation,
    _merge_turn_text_mutations,
    _preserve_large_user_paste,
    _project_self_condition_claims,
    _readable_result,
    _realize_expressive_affordances,
    _remove_self_denials_the_record_refutes,
    _shape_with_live_substrate,
    _strip_scaffolding_tags,
    _strip_ungrounded_vocative_reply,
)

# Lifted to chat_lane_bookkeeping.py. Imported by name so every existing reference —
# including the tests that read this file as text — keeps resolving.
from .chat_lane_bookkeeping import (  # noqa: E402
    _another_reader_owns_this_turn,
    _apply_aura_voice_shaping_compat,
    _asks_to_read_a_named_file,
    _assess_live_mind_snapshot,
    _assess_the_engine_reply,
    _audit_recent_response_reasoning_sync,
    _authored_answer_can_serve_unfinished,
    _bound_stabilizer_generation_budget,
    _bounded_runtime_grounding_can_serve,
    _brevity_requested,
    _build_stateful_voice_reflex,
    _call_stateful_voice_reflex,
    _canonical_runtime_model_label,
    _capabilities_this_turn_needs,
    _context_challenge_reply_is_inadequate,
    _conversation_lane_blocks_fallback,
    _conversation_lane_is_standby,
    _conversation_lane_needs_instant_social_contract,
    _cortex_is_cold_loading,
    _early_chat_json_response,
    _enter_recovery_cooldown,
    _in_recovery_cooldown,
    _env_float,
    _export_json_default,
    _fetch_deep_memory_context,
    _finalize_regenerated_reply_write,
    _flag_unstable_choice_commitment,
    _force_clear_mlx_foreground_owner,
    _foreground_memory_admission_response,
    _gather_recent_user_messages_for_relevance,
    _generation_metadata_consumed_foreground_owner,
    _has_current_shown_source,
    _has_first_person_anchor,
    _has_live_aura_grounding,
    _host_condition,
    _inner_cognitive_cycle_timeout,
    _is_assistant_mode_recovery_request,
    _is_current_request_recap_request,
    _is_simple_affect_check_request,
    _is_simple_subjective_reflex_request,
    _known_answer_for_this_turn,
    _lane_reply_confidence,
    _lane_status_message_body,
    _launcher_desktop_runtime_active,
    _looks_safely_grounded_search_reply,
    _mark_conversation_lane_state,
    _mark_conversation_lane_timeout,
    _mark_http_turn_served,
    _mark_logged_exchange_preempted,
    _memory_log_outbox_is_ready,
    _named_gate_failure,
    _normalize_response_body,
    _note_the_latent_metadata,
    _pre_gate_unavailable_response,
    _protected_foreground_bytes_unchanged,
    _protected_foreground_generation_block_reason,
    _replace_unified_transcript_aura_reply,
    _reply_claims_own_code,
    _reply_gate_proved_a_violation,
    _reply_has_physical_completion_failure,
    _reply_needs_continuation,
    _request_allows_legacy_orchestrator_fallback,
    _request_from_local_desktop_client,
    _request_requires_cognitive_engine,
    _requested_visible_required_phrases,
    _required_foreground_memory_snapshot,
    _resolve_action_episode,
    _resolve_action_episode_grounding,
    _resolve_action_episode_projection,
    _resolve_chat_response_contract,
    _resolve_exact_profile_user_id,
    _response_fingerprint,
    _runtime_affect_available,
    _runtime_personality_available,
    _runtime_shutdown_response,
    _schedule_late_regeneration_finalizer,
    _seconds_this_answer_needs,
    _servable_draft_or_none,
    _shed_generation_for_memory_pressure,
    _short_closed_answer,
    _status_represents_memory_state_result,
    _still_contradicts_the_runtime,
    _this_turn_generated_something,
    _turn_count_ordinal,
    _user_requested_research_memory_save,
    _with_mood,
    _with_the_same_readings,
    _word_set,
    _worth_more_than_a_refusal,
    organ_effect_streaks,
    reset_organ_effect_streaks_for_test,
    reset_organ_engagement_streaks_for_test,
)


router = APIRouter()



# ── Request Models ────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class CheatCodeRequest(BaseModel):
    code: str
    silent: bool = False


async def run_governed_surface_chat_turn(
    message: str,
    *,
    surface: str,
    surface_context: str,
    session_id: str,
    timeout_s: float,
    idempotency_key: str | None = None,
    source_headers: Sequence[tuple[bytes, bytes]] = (),
    client_host: str = "127.0.0.1",
    reply_stream: ReplyStreamChannel | None = None,
) -> str | None:
    """Run a presentation surface through the authenticated HTTP chat contract.

    Voice and private Messages are presentation surfaces, not second cognition
    lanes. Reusing the public handler preserves ingress inspection, memory and
    principal binding, foreground admission, durable delivery fencing,
    response stabilization, and the same governed action path as the desktop.

    ``reply_stream`` lets a surface watch its own reply form. It changes
    nothing about the path taken or the answer returned — the return value is
    still the finished, stabilized reply, and it remains the only text this
    function stands behind. A surface that acts on the stream owes the
    listener a reconciliation against that final text; see
    ``core/conversation/reply_stream.reconcile``.
    """
    normalized_surface = str(surface or "").strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,31}", normalized_surface):
        raise ValueError("invalid governed chat surface")
    normalized_key = str(idempotency_key or f"{normalized_surface}-{uuid.uuid4().hex}").strip()
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,240}", normalized_key):
        raise ValueError("invalid governed chat idempotency key")
    allowed_headers = {
        b"authorization",
        b"cookie",
        b"host",
        b"origin",
        b"sec-fetch-site",
        b"user-agent",
        b"x-aura-device-token",
        b"x-api-token",
    }
    headers = [
        (bytes(name).lower(), bytes(value))
        for name, value in source_headers
        if bytes(name).lower() in allowed_headers
    ]
    if not any(name == b"host" for name, _value in headers):
        headers.append((b"host", b"127.0.0.1:8000"))
    headers.extend(
        (
            (b"x-aura-response-surface", normalized_surface.encode("ascii")),
            (b"x-aura-require-cognitiveengine", b"true"),
            (b"x-aura-surface", normalized_surface.encode("ascii")),
            (b"x-idempotency-key", normalized_key.encode("ascii")),
        )
    )
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/chat",
        "raw_path": b"/api/chat",
        "query_string": b"",
        "headers": headers,
        "client": (str(client_host or "127.0.0.1"), 0),
        "server": ("127.0.0.1", 8000),
    }
    request = Request(scope)
    body = ChatRequest(message=str(message or ""), session_id=session_id)
    validate_runtime_security_request(request)
    _require_internal(request)
    _check_rate_limit(request)
    context_token = _INTERNAL_SURFACE_CONTEXT.set(str(surface_context or "")[:4000])
    stream_ctx: AbstractContextManager[object]
    if reply_stream is not None:
        from core.conversation.reply_stream import bind_reply_stream

        stream_ctx = bind_reply_stream(reply_stream)
    else:
        stream_ctx = contextlib.nullcontext()
    try:
        with stream_ctx:
            response = await asyncio.wait_for(
                api_chat(body=body, request=request, _=None, __=None),
                timeout=max(1.0, float(timeout_s)),
            )
            payload = json.loads(bytes(response.body).decode("utf-8", errors="replace"))
            background = getattr(response, "background", None)
            if background is not None:
                await background()
            if response.status_code >= 500 or not isinstance(payload, dict):
                return None
            return str(payload.get("response") or "").strip() or None
    finally:
        _INTERNAL_SURFACE_CONTEXT.reset(context_token)


async def run_governed_voice_chat_turn(
    message: str,
    *,
    surface_context: str,
    session_id: str,
    timeout_s: float,
    source_headers: Sequence[tuple[bytes, bytes]] = (),
    client_host: str = "127.0.0.1",
    reply_stream: ReplyStreamChannel | None = None,
) -> str | None:
    """Compatibility wrapper for the governed voice presentation surface."""

    return await run_governed_surface_chat_turn(
        message,
        surface="voice",
        surface_context=surface_context,
        session_id=session_id,
        timeout_s=timeout_s,
        source_headers=source_headers,
        client_host=client_host,
        reply_stream=reply_stream,
    )


# Max chat message size to prevent memory exhaustion

_BENCHMARK_CHAT_FALLBACK_MARKERS = (
    "i'm still with",
    "i am still with",
    "what's the issue",
    "i don't have grounded results",
    "i need to search it first",
    "i should not hand you a broken fragment",
    "i shouldn't hand you a broken fragment",
    "benchmark request produced no canonical kernel response",
    "previous turn open",
    "next clean reply",
)




def _benchmark_reply_contract_unmet(prompt: str, reply: str) -> str | None:
    """Reject chat recovery prose before it can become a benchmark artifact."""

    text = str(reply or "").strip()
    lowered_reply = text.lower()
    if not text:
        return "empty"
    if any(marker in lowered_reply for marker in _BENCHMARK_CHAT_FALLBACK_MARKERS):
        return "chat_recovery_fallback"
    if _benchmark_prompt_requests_fenced_artifact(prompt, "```python") and not re.search(
        r"```python\s*\n.+?\n```", text, re.DOTALL
    ):
        return "missing_python_code_block"
    if _benchmark_prompt_requests_fenced_artifact(
        prompt, "```json"
    ) and not response_satisfies_artifact_contract(prompt, text):
        return "missing_json_code_block"
    if _benchmark_prompt_requests_fenced_artifact(
        prompt, "```csv"
    ) and not response_satisfies_artifact_contract(prompt, text):
        return "missing_csv_code_block"
    return None


# ── Session & Conversation Log ────────────────────────────────

_conversation_log_lock = _chat_memory_state._get_convo_lock()
_RECENT_CONVERSATION_RENDERED_CHARS = 6000
_CHAT_EXPORT_SECTION_TIMEOUT_S = 3.0
_CHAT_REASONING_AUDIT_TIMEOUT_S = 1.5
_CHAT_REASONING_AUDIT_MAX_ACTIVE = 2
_CHAT_EXPORT_TOTAL_CHARS = 2 * 1024 * 1024
_CHAT_EXPORT_ITEM_CHARS = 32 * 1024
_reasoning_audit_tasks: set[asyncio.Task[Any]] = set()


class PreemptibleChatLock:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._acquired_at = 0.0
        self._owner_token: object | None = None
        self._owner_task: asyncio.Task[Any] | None = None

    async def acquire(self, *, owner_task: asyncio.Task[Any] | None = None):
        await self._lock.acquire()
        # asyncio.wait_for() owns a temporary task around this coroutine. The
        # caller must be able to bind custody to the HTTP turn that survives
        # acquisition, not that already-finished helper task.
        resolved_owner = owner_task or asyncio.current_task()
        if resolved_owner is None:
            self._lock.release()
            raise RuntimeError("foreground chat lock owner task is unavailable")
        # Monotonic so a mid-turn system sleep cannot inflate held_duration
        # into a false stale-owner diagnosis on wake.
        self._acquired_at = time.monotonic()
        self._owner_token = object()
        self._owner_task = resolved_owner
        return self._owner_token

    def locked(self):
        return self._lock.locked()

    def release(self, owner_token: object | None = None):
        if owner_token is not None and owner_token is not self._owner_token:
            logger.debug("Conversation turn lock release skipped: stale owner token.")
            return False
        current_task = asyncio.current_task()
        if (
            owner_token is None
            and self._owner_task is not None
            and current_task is not self._owner_task
        ):
            logger.debug("Conversation turn lock release skipped: non-owner task.")
            return False
        try:
            if self._lock.locked():
                self._lock.release()
        except RuntimeError as exc:
            record_degradation("chat", exc)
            logger.debug("Conversation turn lock release skipped: %s", exc)
        self._acquired_at = 0.0
        self._owner_token = None
        self._owner_task = None
        return True

    @property
    def held_duration(self) -> float:
        if not self._lock.locked() or self._acquired_at == 0.0:
            return 0.0
        return time.monotonic() - self._acquired_at

    async def cancel_stale_owner(
        self,
        *,
        reason: str = "foreground_chat_preempted",
        acknowledgement_timeout_s: float = 5.0,
    ) -> bool:
        """Cancel one stale owner without admitting a concurrent successor."""

        stale_owner_task = self._owner_task
        stale_owner_token = self._owner_token
        current_task = asyncio.current_task()
        if stale_owner_task is None or stale_owner_task is current_task or stale_owner_task.done():
            return False
        logger.warning("Cancelling stale foreground chat owner before handoff.")
        stale_owner_task.cancel(reason)
        done, _pending = await asyncio.wait(
            {stale_owner_task},
            timeout=max(0.05, float(acknowledgement_timeout_s)),
        )
        if stale_owner_task not in done:
            logger.error(
                "Stale foreground chat owner did not acknowledge cancellation; "
                "exclusive handoff refused."
            )
            return False
        # A waiter already queued on this same lock may acquire immediately
        # after release. Check stale identity, not the lock's aggregate state.
        # Task death with the old token still installed is corruption.
        if self._owner_task is stale_owner_task or self._owner_token is stale_owner_token:
            logger.error(
                "Stale foreground chat owner terminated without releasing its lock; "
                "exclusive handoff refused."
            )
            return False
        return True


_FOREGROUND_CHAT_PREEMPT_CANCEL_REASON = "foreground_chat_preempted"


def _get_fg_lock():
    return _locks.setdefault("fg", PreemptibleChatLock())


_foreground_chat_lock = _get_fg_lock()
# How long a person's NEW message waits for the turn ahead of it.
#
# This was 2.0 seconds. A desktop task legitimately holds the foreground lane
# for the length of its work — measured live at 58.7s, with the MLX client
# logging "Waiting for foreground owner Cortex to release (held 58.7s)" — so
# a follow-up typed during that window waited two seconds and was refused
# with "I still have the previous turn open." The person did nothing wrong
# and lost their message.
#
# Waiting is the correct behaviour: the lane is busy, not broken. The bound
# is the preemption threshold, because past that point the holder is treated
# as stuck and forcibly cleared anyway — so no turn can wait longer than the
# system's own definition of "this is no longer legitimate work". The wait is
# additionally clamped by the remaining foreground budget on every call.
_FOREGROUND_CHAT_BUSY_WAIT_S = 90.0

from .chat_desktop_mode import (
    _LAST_TURN_PREPARATION,
    _SEARCH_SNIPPET_BOILERPLATE_RE,
    _ALLOW_SECONDARY_REPAIR_FLAG,  # noqa: F401
    _ALLOW_TRANSIENT_ENGINE_RETRY_FLAG,  # noqa: F401
    _CHAT_LIVE_MIND_COLLECTION_TIMEOUT_S,  # noqa: F401
    _COMPLEX_SELF_PROCESS_EXPLANATION_RE,  # noqa: F401
    _DESKTOP_COGNITIVE_MAX_TURN_TIMEOUT_S,
    _DESKTOP_COGNITIVE_REPAIR_COOLDOWN_S,  # noqa: F401
    _DESKTOP_COGNITIVE_REPAIR_RECURRENCE_FLOOR,  # noqa: F401
    _DESKTOP_COGNITIVE_RESPONSE_RESERVE_S,
    _DESKTOP_COGNITIVE_TURN_TIMEOUT_S,
    _DURABLE_MEMORY_SCOPE_RE,  # noqa: F401
    _FALSE_SEARCH_PROVENANCE_RE,  # noqa: F401
    _FORCE_DISABLE_SECONDARY_REPAIR_FLAG,  # noqa: F401
    _LIGHTWEIGHT_LIVE_STATE_OR_RECALL_RE,  # noqa: F401
    _LIGHTWEIGHT_REMEMBER_OBJECT_RE,  # noqa: F401
    _best_search_result_entry,  # noqa: F401
    _build_cognitive_engine_reply_repair_directive,
    _build_live_mind_context_payload,  # noqa: F401
    _cites_nothing,  # noqa: F401
    _claims_a_live_check,  # noqa: F401
    _clean_search_fact_text,  # noqa: F401
    _collect_live_mind_context_payload,
    _desktop_objective_executable_after_cognitive_attempt,
    _desktop_required_cognitive_budget,
    _desktop_secondary_model_repair_allowed,
    _desktop_transient_engine_retry_allowed,
    _evidence_grounded_desktop_search_reply,  # noqa: F401
    _is_compact_desktop_chat_contract,
    _is_lightweight_live_desktop_state_or_recall_turn,  # noqa: F401
    _reanswer_when_the_runtime_contradicts_her,
    _repair_required_search_reply_provenance,
    _route_desktop_cognitive_failure_to_resilience,
    _select_cognitive_chat_mode,
    _strip_empty_citations,  # noqa: F401
    _turn_timing,  # noqa: F401
)


_DESKTOP_COGNITIVE_REPAIR_TIMEOUT_S = _env_float(
    "AURA_DESKTOP_COGNITIVE_REPAIR_TIMEOUT_S",
    180.0,
    minimum=40.0,
)
_DESKTOP_COMPACT_CHAT_CYCLE_TIMEOUT_S = _env_float(
    "AURA_DESKTOP_COMPACT_CHAT_CYCLE_TIMEOUT_S",
    96.0,
    minimum=60.0,
)
_DESKTOP_MEMORY_STATE_TURN_TIMEOUT_S = _env_float(
    "AURA_DESKTOP_MEMORY_STATE_TURN_TIMEOUT_S",
    70.0,
    minimum=60.0,
)
_DESKTOP_COGNITIVE_MIN_REQUIRED_BUDGET_S = 60.0
_FOREGROUND_CHAT_LOCK_PREEMPT_AFTER_S = _env_float(
    "AURA_FOREGROUND_CHAT_LOCK_PREEMPT_AFTER_S",
    max(
        75.0,
        _DESKTOP_COGNITIVE_TURN_TIMEOUT_S + _DESKTOP_COGNITIVE_RESPONSE_RESERVE_S + 10.0,
    ),
    minimum=45.0,
)
_CHAT_TURN_MEMORY_LOG_STARTUP_TASK_NAME = "ChatTurnMemoryLogStartup"
_CHAT_TURN_MEMORY_LOG_STARTUP_TIMEOUT_S = 120.0
_CHAT_TURN_MEMORY_LOG_STARTUP_POLL_S = 0.25


async def _start_chat_turn_memory_log_when_ready() -> None:
    """Wait through normal service ordering, but never wait indefinitely."""

    deadline = time.monotonic() + _CHAT_TURN_MEMORY_LOG_STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if _memory_log_outbox_is_ready():
            if not _chat_preflight._schedule_chat_turn_memory_log(chat_origin="startup_recovery"):
                raise RuntimeError("chat_memory_log_startup_schedule_failed")
            return
        await asyncio.sleep(_CHAT_TURN_MEMORY_LOG_STARTUP_POLL_S)
    error = RuntimeError("persistence_memory_log_outbox_not_ready_before_deadline")
    record_degradation(
        "chat.memory_log_outbox_startup",
        error,
        action="durable outbox remains on disk and will be retried by the next chat write",
        enforce_failure_policy=False,
    )
    logger.error(
        "Durable chat memory outbox dependency was not ready within %.1fs; "
        "pending work remains on disk.",
        _CHAT_TURN_MEMORY_LOG_STARTUP_TIMEOUT_S,
    )


def start_chat_turn_memory_log_worker() -> bool:
    """Recover durable pending memory work when the API runtime starts."""
    if _memory_log_outbox_is_ready():
        return _chat_preflight._schedule_chat_turn_memory_log(chat_origin="startup_recovery")
    try:
        tracker = get_task_tracker()
        if _chat_preflight._active_task_count_by_name(
            tracker, _CHAT_TURN_MEMORY_LOG_STARTUP_TASK_NAME
        ):
            return True
        schedule = getattr(tracker, "bounded_track", None) or getattr(
            tracker,
            "create_task",
            None,
        )
        if not callable(schedule):
            raise RuntimeError("task_tracker_has_no_scheduler")
        waiter = _start_chat_turn_memory_log_when_ready()
        try:
            schedule(waiter, name=_CHAT_TURN_MEMORY_LOG_STARTUP_TASK_NAME)
        except _CHAT_RECOVERABLE_ERRORS:
            waiter.close()
            raise
        return True
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.memory_log_outbox_startup", exc)
        logger.error("Durable chat memory outbox startup could not be scheduled: %s", exc)
        return False


#: A second, separate request following the thing to be remembered. Only an
#: explicit new ask counts — a wordy preamble around the memory request does
#: not, because that turn is still entirely about memory.


#: Words that carry no topic, so sharing them proves nothing about relevance.


















_OWNER_DIRECT_ADDRESS_RE = re.compile(
    r"\b(?:hi|hey|hello|thanks|thank you|okay|ok|yes|no|sure|listen|look|"
    r"absolutely|definitely|right|agreed|got it|i'm here|i am here)\s*,?\s+"
    r"([A-Z][a-z]{2,24})\b",
    re.IGNORECASE,
)
_OWNER_IDENTITY_ASSERTION_RE = re.compile(
    r"\byou(?:'re| are)\s+([A-Z][a-z]{2,24})\b",
    re.IGNORECASE,
)
_OWNER_NAME_DRIFT_EXCLUSIONS = {
    "Aura",
    "User",
    "You",
    "Human",
    "Computer",
    "Mac",
    "Google",
    "Chrome",
    "ChatGPT",
    "Gemini",
}


def _owner_name_drift_candidates(reply_text: str) -> list[str]:
    candidates: list[str] = []
    for pattern in (_OWNER_DIRECT_ADDRESS_RE, _OWNER_IDENTITY_ASSERTION_RE):
        for match in pattern.finditer(str(reply_text or "")):
            name = str(match.group(1) or "").strip()
            if not name or name in _OWNER_NAME_DRIFT_EXCLUSIONS:
                continue
            if name not in candidates:
                candidates.append(name)
    return candidates


def _reply_has_owner_name_drift(
    user_message: str,
    reply_text: str,
    *,
    owner_session_restored: bool = False,
) -> bool:
    if not _chat_memory_state._owner_session_is_verified(
        owner_session_restored=owner_session_restored
    ):
        return False
    owner_name = _chat_memory_state._resolve_primary_operator_name().strip()
    if not owner_name or owner_name == "the verified owner":
        return False
    user = str(user_message or "")
    for candidate in _owner_name_drift_candidates(reply_text):
        if candidate.lower() == owner_name.lower():
            continue
        if re.search(rf"\b{re.escape(candidate)}\b", user):
            continue
        return True
    return False


def _repair_owner_name_drift_reply(reply_text: str) -> str:
    owner_name = _chat_memory_state._resolve_primary_operator_name().strip()
    if not owner_name or owner_name == "the verified owner":
        return str(reply_text or "")
    repaired = str(reply_text or "")
    for candidate in _owner_name_drift_candidates(repaired):
        if candidate.lower() == owner_name.lower():
            continue
        repaired = re.sub(rf"\b{re.escape(candidate)}\b", owner_name, repaired)
    return repaired




# ── Stale Response Detection ─────────────────────────────────
# Track the last N responses to detect when the cortex is stuck returning the
# same cached output. This prevents the "Dark Matter" loop where a stale
# identity prompt produces identical text on every turn.
@dataclasses.dataclass
class _ConversationQualityState:
    recent_responses: collections.deque[str] = dataclasses.field(
        default_factory=lambda: collections.deque(maxlen=12)
    )
    recent_response_pairs: collections.deque[tuple[str, str]] = dataclasses.field(
        default_factory=lambda: collections.deque(maxlen=12)
    )
    consecutive_degraded_count: int = 0
    lane_status_fingerprint: str = ""
    lane_status_repeat_count: int = 0
    conversation_resume_handle: str = ""
    conversation_resume_created_at: float = 0.0
    last_access_monotonic: float = dataclasses.field(default_factory=time.monotonic)


@dataclasses.dataclass(frozen=True)
class _ReplyQualitySnapshot:
    recent_user_messages: tuple[str, ...]
    is_stale: bool
    is_same_diff: bool
    is_off_topic: bool
    off_topic_reason: str
    semantic_glitch: bool
    semantic_glitch_reason: str
    reply_assessment: Any


_CHAT_REPLY_QUALITY_SNAPSHOTS: ContextVar[
    dict[tuple[str, str], _ReplyQualitySnapshot] | None
] = ContextVar("chat_reply_quality_snapshots", default=None)


_CONVERSATION_QUALITY_STATE_LIMIT = 256
_CONVERSATION_QUALITY_STATE_TTL_S = 6 * 60 * 60.0
_CONVERSATION_RESUME_TTL_S = 240.0
_DEFAULT_CONVERSATION_QUALITY_KEY = "default"
_conversation_quality_lock = checked_lock(
    "chat.conversation_quality_state",
    reentrant=True,
)
_conversation_quality_states: collections.OrderedDict[str, _ConversationQualityState] = (
    collections.OrderedDict([(_DEFAULT_CONVERSATION_QUALITY_KEY, _ConversationQualityState())])
)


def _conversation_quality_key(
    *,
    session_id: str = "",
    principal_id: str = "",
    principal_surface: str = "",
) -> str:
    session = " ".join(str(session_id or _CHAT_REQUEST_SESSION.get() or "").strip().split())[
        :_CHAT_SESSION_ID_MAX_CHARS
    ]
    principal = " ".join(str(principal_id or _CHAT_REQUEST_PRINCIPAL.get() or "").strip().split())[
        :160
    ]
    surface = str(principal_surface or _CHAT_REQUEST_SURFACE.get() or "").strip().casefold()[:32]
    if not session and not principal and not surface:
        return _DEFAULT_CONVERSATION_QUALITY_KEY
    identity = json.dumps(
        {"principal": principal, "session": session or "default", "surface": surface},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(identity.encode("utf-8", errors="replace")).hexdigest()


def _conversation_quality_state_locked(
    *,
    session_id: str = "",
    principal_id: str = "",
    principal_surface: str = "",
) -> _ConversationQualityState:
    now = time.monotonic()
    key = _conversation_quality_key(
        session_id=session_id,
        principal_id=principal_id,
        principal_surface=principal_surface,
    )
    stale_keys = [
        candidate
        for candidate, state in _conversation_quality_states.items()
        if candidate != _DEFAULT_CONVERSATION_QUALITY_KEY
        and (now - state.last_access_monotonic) > _CONVERSATION_QUALITY_STATE_TTL_S
    ]
    for stale_key in stale_keys:
        _conversation_quality_states.pop(stale_key, None)
    state = _conversation_quality_states.get(key)
    if state is None:
        state = _ConversationQualityState()
        _conversation_quality_states[key] = state
    state.last_access_monotonic = now
    _conversation_quality_states.move_to_end(key)
    while len(_conversation_quality_states) > _CONVERSATION_QUALITY_STATE_LIMIT:
        oldest_key = next(iter(_conversation_quality_states))
        if oldest_key == _DEFAULT_CONVERSATION_QUALITY_KEY:
            _conversation_quality_states.move_to_end(oldest_key)
            continue
        _conversation_quality_states.popitem(last=False)
    return state


def _reset_conversation_quality_registry() -> None:
    """Clear transient per-conversation quality state without changing policy."""
    with _conversation_quality_lock:
        default = _conversation_quality_states.get(_DEFAULT_CONVERSATION_QUALITY_KEY)
        if default is None:
            default = _ConversationQualityState()
        default.recent_responses.clear()
        default.recent_response_pairs.clear()
        default.consecutive_degraded_count = 0
        default.lane_status_fingerprint = ""
        default.lane_status_repeat_count = 0
        default.conversation_resume_handle = ""
        default.conversation_resume_created_at = 0.0
        default.last_access_monotonic = time.monotonic()
        _conversation_quality_states.clear()
        _conversation_quality_states[_DEFAULT_CONVERSATION_QUALITY_KEY] = default


def _take_conversation_resume_handle(
    *,
    session_id: str = "",
    principal_id: str = "",
    principal_surface: str = "",
) -> str:
    """Consume this conversation's worker capability exactly once."""

    with _conversation_quality_lock:
        state = _conversation_quality_state_locked(
            session_id=session_id,
            principal_id=principal_id,
            principal_surface=principal_surface,
        )
        handle = str(state.conversation_resume_handle or "").strip().lower()
        age = time.monotonic() - float(state.conversation_resume_created_at or 0.0)
        state.conversation_resume_handle = ""
        state.conversation_resume_created_at = 0.0
    if age > _CONVERSATION_RESUME_TTL_S:
        return ""
    return handle if re.fullmatch(r"[0-9a-f]{32}", handle) else ""


def _store_conversation_resume_handle(
    turn_trace: dict[str, Any],
    delivered_text: str,
    *,
    session_id: str = "",
    principal_id: str = "",
    principal_surface: str = "",
) -> bool:
    """Retain exact KV only when it authored the exact bytes delivered."""

    receipt = turn_trace.get("live_mind_surface_control_receipt")
    receipt = dict(receipt) if isinstance(receipt, dict) else {}
    handle = str(receipt.get("conversation_resume_handle") or "").strip().lower()
    expected_hash = str(
        receipt.get("conversation_resume_output_sha256") or ""
    ).strip().lower()
    delivered_hash = hashlib.sha256(
        str(delivered_text or "").encode("utf-8", "replace")
    ).hexdigest()
    reasons: list[str] = []
    if not re.fullmatch(r"[0-9a-f]{32}", handle):
        reasons.append("no handle" if not handle else "malformed handle")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        reasons.append("no output hash")
    elif expected_hash != delivered_hash:
        reasons.append("delivered text is not the text the KV authored")
    if turn_trace.get("cognitive_engine_reply_accepted") is not True:
        reasons.append("engine reply not accepted")
    if turn_trace.get("bounded_contract_used") is True:
        reasons.append("bounded contract used")
    if turn_trace.get("legacy_fallback_used") is True:
        reasons.append("legacy fallback used")
    accepted = not reasons
    # A refusal nobody can see is a conversation that re-reads its own history
    # every turn. Rejecting is often right; being silent about it is not.
    logger.info(
        "🔗 conversation resume handle %s%s",
        "kept" if accepted else "refused",
        "" if accepted else ": " + "; ".join(reasons),
    )
    try:
        from core.verify.one_way_decisions import record_decision

        record_decision(
            "conversation_resume_handle",
            admitted=accepted,
            reason="; ".join(reasons),
        )
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    with _conversation_quality_lock:
        state = _conversation_quality_state_locked(
            session_id=session_id,
            principal_id=principal_id,
            principal_surface=principal_surface,
        )
        state.conversation_resume_handle = handle if accepted else ""
        state.conversation_resume_created_at = time.monotonic() if accepted else 0.0
    return accepted


def _increment_conversation_degradation_streak() -> int:
    with _conversation_quality_lock:
        state = _conversation_quality_state_locked()
        state.consecutive_degraded_count += 1
        return state.consecutive_degraded_count


def _set_conversation_degradation_streak(value: int) -> None:
    with _conversation_quality_lock:
        state = _conversation_quality_state_locked()
        state.consecutive_degraded_count = max(0, int(value))


def _conversation_degradation_streak() -> int:
    with _conversation_quality_lock:
        return _conversation_quality_state_locked().consecutive_degraded_count


_default_conversation_quality_state = _conversation_quality_states[
    _DEFAULT_CONVERSATION_QUALITY_KEY
]
# Compatibility handles for direct synchronous tooling and existing tests. Live
# requests never use this bucket once their principal/session context is bound.
_recent_responses = _default_conversation_quality_state.recent_responses
_recent_response_pairs = _default_conversation_quality_state.recent_response_pairs
_STALE_REPEAT_THRESHOLD = 2  # [STABILITY] Reverting to 2. A single identical repeat is enough to trigger defensive measures.
_FUZZY_SIMILARITY_THRESHOLD = 0.80  # word-overlap ratio that counts as semantically stale
_desktop_cognitive_repair_lock = threading.Lock()
_desktop_cognitive_repair_last_scheduled: dict[str, float] = {}
_CONTEXTUAL_RELEVANCE_BRIDGE_MARKERS = (
    "you mentioned",
    "you brought",
    "i brought",
    "i asked because",
    "because you",
    "because the",
    "i connected",
    "i was connecting",
    "what i meant",
    "where it came from",
    "i was responding to",
    "i thought you meant",
    "i misread",
    "i drifted",
    "i wasn't being clear",
    "i was not being clear",
    "answer directly",
    "talking around it",
    "look at this more clearly",
    "still focused on our conversation",
    "that did not connect",
    "that didn't connect",
    "that was a jump",
)
_CONTEXTUAL_RELEVANCE_DRIFT_MARKERS = (
    "personal detail",
    "having pets",
    "pets can be",
    "pet can be",
    "comforting",
    "used to have a dog",
    "dog when i was younger",
    "feeling a bit down",
    "feeling down",
    "the voices",
    "whispering in my ear",
    "let's nail this pitch",
    "lets nail this pitch",
    "key points",
    "my attention is",
    "curiosity is",
    "my mood",
    "my state",
)
_CONTENT_OBJECT_MARKERS = (
    "article",
    "book",
    "chapter",
    "character",
    "essay",
    "film",
    "movie",
    "narrative",
    "novel",
    "passage",
    "piece",
    "plot",
    "poem",
    "post",
    "premise",
    "scene",
    "script",
    "story",
    "text",
    "thread",
)
_UNREQUESTED_CONTENT_REVIEW_MARKERS = (
    "a chilling and imaginative take",
    "a classic setup",
    "the execution is strong",
    "the premise",
    "the story is",
    "the narrative",
    "this story",
    "this narrative",
)


def _fuzzy_similar(a: str, b: str) -> bool:
    """Check if two responses share >80% word overlap (catches paraphrased repeats)."""
    words_a = _word_set(a)
    words_b = _word_set(b)
    if not words_a or not words_b:
        return False
    # Jaccard-like: intersection / smaller set
    overlap = len(words_a & words_b)
    smaller = min(len(words_a), len(words_b))
    if smaller < 6:
        return False  # too short for meaningful comparison
    return (overlap / smaller) >= _FUZZY_SIMILARITY_THRESHOLD


# Content recall: "earlier I gave/told you X — what was it?" The deliverable
# is a SPECIFIC fact from this session's transcript. Observed live (July 2026):
# these turns reached the model with zero session context and durable-memory
# noise as evidence, and it confabulated values ("4523" for a code that was
# 7213, two turns after acknowledging it).


_RECENT_CONTEXT_NEEDED_RE = re.compile(
    r"\b(?:continue|resume|pick\s+back\s+up|from\s+(?:earlier|before|that|there)|"
    r"what\s+we\s+were|what\s+you\s+were|what\s+i\s+was|same\s+thread|"
    r"this\s+thread|previous\s+(?:turn|message|answer)|last\s+(?:thing|message|answer|question)|"
    r"as\s+we\s+said|like\s+you\s+said|you\s+mentioned|i\s+mentioned|we\s+discussed|"
    r"that\s+(?:issue|bug|problem|topic|plan|task|demo|path|thing))\b",
    re.IGNORECASE,
)
_SHORT_FOLLOWUP_CONTEXT_NEEDED_RE = re.compile(
    r"\b(?:"
    r"you\s+with\s+me|with\s+me|still\s+with\s+me|"
    r"what\s+pitch|which\s+pitch|what\s+one|which\s+one|"
    r"what(?:'re|re|\s+are)\s+you\s+talking\s+about|what\s+do\s+you\s+mean|"
    r"where\s+did\s+that\s+come\s+from|what\s+was\s+that|"
    r"this\s+conversation|our\s+conversation|the\s+thread|"
    r"tell\s+me\s+more|say\s+more|go\s+on|why\s+is\s+that|why\s+so|"
    r"what\s+next|what\s+now|and\s+then|what\s+about\s+that"
    r")\b",
    re.IGNORECASE,
)


#: Asking what was SAID — by either of us — in this conversation.
#
# LIVE DEFECT, 2026-08-10. "quote me the exact first sentence I said to you
# today" was not classified as needing recent context, so the turn ran on the
# default four-exchange window, the sentence was long out of it, and she could
# not answer a question whose whole answer was sitting in the transcript.
#
# The existing classifier caught "what did we just talk about" and "what did
# you TELL me earlier", but not the most direct forms of the same request:
#
#     what did you say a minute ago            -> missed
#     remind me what you told me earlier       -> missed
#     repeat what you just said                -> missed
#     what were your exact words               -> missed
#     you said something earlier, what was it  -> missed
#     quote me the first sentence I said       -> missed
#
# Every one of those is a question about the transcript, which is the one
# piece of evidence the transcript window exists to supply.
_UTTERANCE_RECALL_RE = re.compile(
    r"\b(?:"
    r"(?:what|which)\s+(?:exact\s+)?(?:word|words|sentence|line|phrase)\b"
    r"|(?:quote|repeat|restate|recite)\b[^.?!]{0,40}\b(?:said|say|told|wrote|asked)\b"
    r"|(?:quote|repeat|restate|recite)\s+(?:me\s+)?(?:the|my|your|that|it)\b"
    r"|\b(?:you|i|we)\s+(?:just\s+|already\s+)?(?:said|told\s+me|mentioned|wrote)\b"
    r"|\bwhat\s+did\s+(?:you|i|we)\s+(?:just\s+)?(?:say|said|tell|write)\b"
    r"|\bremind\s+me\s+what\b"
    r"|\bin\s+your\s+own\s+words\b"
    r"|\byour\s+exact\s+words\b"
    r")",
    re.IGNORECASE,
)


def _desktop_turn_needs_recent_context(user_message: str) -> bool:
    text = str(user_message or "").strip()
    if not text:
        return False
    if _chat_memory_state._classify_conversation_recall_request(text):
        return True
    if _UTTERANCE_RECALL_RE.search(text):
        return True
    if _chat_desktop_repair._is_contextual_relevance_challenge(text):
        return True
    short_followup_surface = text
    if _chat_desktop_repair._has_local_choice_antecedent(text):
        short_followup_surface = _LOCAL_CHOICE_REFERENCE_RE.sub("", text)
    if _SHORT_FOLLOWUP_CONTEXT_NEEDED_RE.search(short_followup_surface):
        return True
    try:
        from core.conversation.response_reliability import is_status_check_turn

        if is_status_check_turn(text):
            return True
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.recent_context_classifier", exc)
        logger.debug("Recent-context status classifier unavailable: %s", exc)
    return bool(_RECENT_CONTEXT_NEEDED_RE.search(text))




def _format_recent_conversation_context(
    exchanges: list[dict[str, str]],
    *,
    limit_chars: int = _RECENT_CONVERSATION_RENDERED_CHARS,
) -> str:
    lines: list[str] = []
    for entry in exchanges:
        user_text = _chat_memory_state._clip_conversation_text(entry.get("user"), limit=220)
        aura_text = _chat_memory_state._clip_conversation_text(entry.get("aura"), limit=260)
        if user_text:
            lines.append(f"User: {user_text}")
        if aura_text:
            lines.append(f"Aura: {aura_text}")
    text = "\n".join(lines).strip()
    if len(text) <= limit_chars:
        return text
    return text[-limit_chars:].lstrip()








#: Openers of the three non-answer surfaces built by _build_reply_failure_notice.
#: Kept as a tuple rather than a regex so a change to those sentences is a
#: visible edit here rather than a silently-stopped guard.


_CONVERSATION_RECALL_DEFLECTION_RE = re.compile(
    r"\b(?:something about|it sits|sits heavy|i'?m not sure|i don'?t remember|"
    r"i can'?t recall|i cannot recall|i don'?t have that|lost the thread|"
    r"my memory is|memory feels)\b",
    re.IGNORECASE,
)


def _conversation_recall_reply_is_inadequate(
    user_message: str,
    reply_text: str,
    expected_reply: str | None,
) -> bool:
    if not _chat_memory_state._classify_conversation_recall_request(user_message):
        return False
    reply = str(reply_text or "").strip()
    if not reply:
        return True
    if _CONVERSATION_RECALL_DEFLECTION_RE.search(reply):
        return True
    expected = str(expected_reply or "").strip()
    if not expected:
        return False
    expected_tokens = _chat_conversation_repair._extract_topic_tokens(expected)
    reply_tokens = _chat_conversation_repair._extract_topic_tokens(reply)
    if not expected_tokens:
        return False
    overlap = expected_tokens & reply_tokens
    required = min(4, max(2, len(expected_tokens) // 6))
    return len(overlap) < required


async def _repair_conversation_recall_if_needed(
    user_message: str,
    reply_text: str,
    *,
    session_id: str = "",
) -> tuple[str, bool]:
    expected = await _chat_memory_state._build_conversation_recall_reply(
        user_message,
        session_id=session_id,
    )
    if expected and _conversation_recall_reply_is_inadequate(user_message, reply_text, expected):
        return expected, True
    return reply_text, False


_TRACEABILITY_REASON_MARKERS = (
    "engineering traceability",
    "operational details",
    "give receipts",
    "give me receipts",
    "refuse to give receipts",
    "exactly why",
    "do not have access",
    "governance rule blocks disclosure",
    "data does not exist",
    "you are uncertain",
)

_TRACEABILITY_EXAMPLE_MARKERS = (
    "most recent non-private action",
    "non-private action",
    "safe example",
    "log line",
    "event id",
    "trace:",
    "timestamp, subsystem, action, result",
)

_TRACEABILITY_CORE_MARKERS = (
    "traceability",
    "receipt",
    "receipts",
    "event id",
    "log line",
    "operational details",
)





def _classify_traceability_request(user_message: str) -> tuple[bool, bool, bool]:
    text = _chat_memory_state._normalize_user_message(user_message)
    if not text:
        return False, False, False

    asks_reason = any(marker in text for marker in _TRACEABILITY_REASON_MARKERS)
    asks_example = any(marker in text for marker in _TRACEABILITY_EXAMPLE_MARKERS)
    asks_traceability = (
        asks_reason
        or asks_example
        or (
            any(marker in text for marker in _TRACEABILITY_CORE_MARKERS)
            and ("recent" in text or "safe" in text or "most recent" in text or "why" in text)
        )
    )
    return asks_traceability, asks_reason, asks_example


async def _resolve_traceability_anchor(user_message: str) -> str | None:
    asks_traceability, _, _ = _classify_traceability_request(user_message)
    if asks_traceability:
        return str(user_message or "")

    if not _is_referential_followup_request(user_message):
        return None

    recent = await _gather_recent_user_messages_for_relevance(user_message, limit=6)
    current = str(user_message or "").strip()
    for candidate in reversed(recent):
        candidate_text = str(candidate or "").strip()
        if not candidate_text or candidate_text == current:
            continue
        candidate_traceability, _, _ = _classify_traceability_request(candidate_text)
        if candidate_traceability:
            return candidate_text
    return None












async def _build_grounded_traceability_reply(user_message: str) -> str | None:
    anchor = await _resolve_traceability_anchor(user_message)
    if not anchor:
        return None

    event, reason_category = await asyncio.to_thread(_collect_recent_traceability_event_sync)
    return _format_traceability_reply(
        anchor_message=anchor,
        event=event,
        reason_category=reason_category,
    )










def _cognitive_cycle_timeout_for_request(
    outer_timeout_s: float,
    *,
    require_engine: bool,
    compact_desktop_chat_contract: bool,
    prompt_shape: Any,
) -> float:
    """Keep compact turns bounded without clipping structurally long answers."""

    cycle_timeout = _inner_cognitive_cycle_timeout(
        outer_timeout_s,
        protected_foreground=bool(require_engine),
    )
    shape_needs_room = bool(
        getattr(prompt_shape, "prefers_extended_answer", False)
        or getattr(prompt_shape, "requires_single_reply_coverage", False)
        or int(getattr(prompt_shape, "question_parts", 0) or 0) >= 2
    )
    if require_engine and compact_desktop_chat_contract and not shape_needs_room:
        return min(cycle_timeout, _DESKTOP_COMPACT_CHAT_CYCLE_TIMEOUT_S)
    return cycle_timeout


# Organs a real conversational turn should have engaged.
#
# Deliberately a SECOND tier, reported and never fatal. The required list is
# what authorises a turn; this is what shapes it.
#
# Personality sits here rather than in the required list, and the reasoning is
# worth stating because the instinct runs the other way. A missing persona pass
# makes a reply flat — it does not make it wrong. Refusing a correct answer for
# being flat is the over-blocking failure this codebase has already paid for
# twice, most recently a gate that replaced 900 characters of real answer with
# an apology because confidence read "degraded". Flat and true beats refused.
#
# The point is visibility. Before this, "does chat actually engage everything
# it should?" had no answer anywhere in the system: the persona pass was
# applied through a default-None lookup, so its absence looked exactly like
# its presence. Now a turn carries the list of what was missing, and a
# persistent gap shows up instead of being felt and never found.


# A gap that persists is a different fact from a gap on one turn.
#
# Reporting alone changes nothing: if personality is absent on every turn, a
# per-turn note is a per-turn note, and she goes on sounding flat while the
# evidence scrolls past. But refusing the turn is worse — a warming organ would
# silence a correct answer.
#
# So absence is counted. One turn without an organ is noise (boot, a restart, a
# lane cycling). The same organ missing from turn after turn is a defect, and at
# that point it escalates once — not once per turn, which is how a real signal
# becomes a storm nobody reads.


# Presence, consultation, effect — three different claims.
#
# "Is personality engaged?" was answered by asking whether the service exists.
# A service can exist, be called, return its input unchanged, and have no causal
# relationship to her voice whatsoever. That is indistinguishable from absence
# in the only place it matters: what she actually said.
#
# So shaping organs report whether they CHANGED the reply, and a run of turns
# where an organ was present and never changed anything is recorded — inert is
# a different defect from missing, and it needs a different fix, so conflating
# them costs debugging time exactly when the voice sounds wrong and everything
# reports healthy.


# Whether the full-mind contract failed because the answer was not hers, or
# because it was hers and something ancillary was soft.
#
# Measured live 2026-07-27, twice in one conversation. The cortex produced a
# real 199-character answer; the quality pass marked confidence "degraded"
# because she had recently said something similar; the full-mind gate requires
# confidence == "high", so her answer was thrown away and replaced with "I
# couldn't get my full attention onto that one". Which then became the previous
# answer, making the NEXT turn look repetitive too.
#
# That is this codebase's most expensive recurring bug: a good answer produced,
# discarded by a gate, and reported to the user as an infrastructure failure.
# The gate is right about theatre — bounded-repair text and legacy fallbacks
# must never speak in her voice — and wrong about her own words. So the two
# cases are separated: authorship proofs stay fail-closed, and a soft
# confidence reading is disclosed instead of substituted.
#
# The same separation applies to the STATE-COMPLETENESS proofs below, and
# live 2026-08-04 it was not being applied. Asked to show a snippet of her
# code and say where it lives, she produced a 1999-character reply that the
# quality pass marked `assessment=ok` — and it was destroyed because
# `live_mind_controls_unbound`, which says a generation control was not
# structurally bound, not that the words were someone else's. Bryan got "I
# couldn't get my full attention onto that one."
#
# Two gates disagreed about that exact proof in that exact minute. The log
# carries both: "Desktop turn served with DEGRADED full-mind proof
# (authentic cognitive reply; missing: live_mind_controls_unbound)" from
# one, and "failing closed instead of serving partial/raw speech" from this
# one. A proof that is disclosable at one exit and fatal at the next is not
# a policy, it is an accident of which exit the turn happened to take.
#
# Authorship stays fail-closed — `_only_soft_proofs_missing` still requires
# that she thought it, said it, and that no bounded-repair or legacy text is
# wearing her voice. What is waived is bookkeeping about her internals,
# which is worth disclosing and never worth replacing her own answer with an
# apology.
_SOFT_FULL_MIND_PROOF_PREFIXES: tuple[str, ...] = (
    "confidence:",
    "architecture_context_unbound",
    "live_mind_snapshot_not_ready",
    "live_mind_controls_unbound",
    # An absent check is not a failed one.
    #
    # "authored_answer_incomplete" is fatal, and rightly so when the answer
    # was cut off or a continuation gave up. It was also raised when the
    # semantic-completion receipt was never bound — nobody looked, so nothing
    # is known, and a turn that had read a library, run its code and written a
    # reply was refused on the strength of that (LIVE 2026-08-29, beside
    # live_mind_controls_unbound, which is already disclosed here).
    #
    # Bookkeeping about her internals, disclosed rather than substituted, on
    # the same reasoning as the three above. The causes that ARE statements
    # about the answer keep their own names and stay fail-closed.
    "authored_answer_incomplete:nobody_checked",
)


def _only_soft_proofs_missing(contract: Any) -> bool:
    """True when the text is genuinely hers and only confidence came back soft."""
    if not isinstance(contract, dict):
        return False
    missing = [str(item or "") for item in (contract.get("full_mind_missing_proofs") or [])]
    if not missing:
        return False
    if not all(item.startswith(_SOFT_FULL_MIND_PROOF_PREFIXES) for item in missing):
        return False
    # Authorship is never waived: she has to have thought it and said it.
    return bool(
        contract.get("engine_think_invoked")
        and contract.get("cognitive_engine_reply_accepted")
        and not contract.get("cognitive_engine_reply_failed")
        and not contract.get("bounded_contract_used")
        and not contract.get("legacy_fallback_used")
        and not contract.get("authorship_replacement_applied")
    )


def _authored_answer_can_serve(contract: Any) -> bool:
    """Keep a valid answer independent from full-system certification state.

    Two exits, one policy. The soft-proof list above exists because a handful
    of STATE-COMPLETENESS proofs are bookkeeping about her internals rather
    than claims about the text, and its own comment names the failure this
    fixes: a proof that is disclosable at one exit and fatal at the next is
    not a policy, it is an accident of which exit the turn happened to take.

    LIVE 2026-08-29: a turn that read a library's docs, wrote code against
    them, ran it and produced the trial balance was refused here on
    ``authored_answer_incomplete:nobody_checked`` and
    ``live_mind_controls_unbound:not_applied`` — both on that list, both
    already disclosable at the other exit. Ownership had just been proven on
    the same turn.

    Authorship is not waived by either route. The second branch requires that
    she thought it and said it, and that no bounded repair, legacy fallback or
    runtime substitution is wearing her voice.
    """

    if not isinstance(contract, dict):
        return False
    if contract.get("answer_delivery_proven") and contract.get(
        "authentic_cognitive_reply"
    ):
        return True
    return _only_soft_proofs_missing(contract)



















async def _run_cognitive_engine_chat_turn(
    effective_user_message: str,
    *,
    visible_user_message: str | None = None,
    raw_user_message: str | None = None,
    declared_interlocutor: dict[str, Any] | None = None,
    preflight_context_message: str | None = None,
    turn_sensory_evidence: Any = None,
    session_id: str = "",
    origin: str = "user",
    timeout_s: float | None = None,
    lane: dict[str, Any] | None = None,
    source: str = "chat_api",
    require_engine: bool = False,
    conversation_only_surface: bool = False,
    principal_id: str = "",
    turn_trace: dict[str, Any] | None = None,
    referential_anchor: str = "",
    action_episode_evidence: str = "",
    prior_answer_provenance: dict[str, Any] | None = None,
    continuation_partial: str = "",
    continuation_reasons: tuple[str, ...] | list[str] | None = None,
    continuation_evidence: dict[str, Any] | None = None,
    conversation_resume_handle: str = "",
    completed_capability_evidence: dict[str, Any] | None = None,
    evidence_profile: str = _chat_preflight._CHAT_EVIDENCE_PROFILE_CONTEXTUAL_LANGUAGE,
) -> str | None:
    """Run a live desktop/user chat turn through CognitiveEngine.

    The HTTP and WebSocket desktop surfaces mark this path as required so the
    UI uses the same causal cognitive path as the live runtime. When required,
    absence or timeout returns ``None`` and the caller must fail closed instead
    of silently routing to a thinner model lane. Evidence-critical contracts may
    bind an unreliable draft to their canonical projection after the required
    CognitiveEngine invocation has happened.

    Now with:
    - Persistent connection pooling
    - Automatic retry with exponential backoff
    - Health monitoring
    - Strict fail-closed support for CognitiveEngine-required callers
    """
    preparation_started_at = time.perf_counter()
    turn_budget_started_at = time.monotonic()
    state_native_output_owner = bool(
        evidence_profile == _chat_preflight._CHAT_EVIDENCE_PROFILE_QUALIFIED_RECURRENT
    )
    visible = str(visible_user_message or effective_user_message or "")
    raw_visible = str(raw_user_message or visible)
    action_episode_evidence = str(action_episode_evidence or "").strip()
    conversation_resume_handle = str(conversation_resume_handle or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{32}", conversation_resume_handle):
        conversation_resume_handle = ""
    interlocutor_evidence = (
        dict(declared_interlocutor)
        if isinstance(declared_interlocutor, dict) and declared_interlocutor
        else {}
    )
    if turn_trace is not None:
        continuing_prior_segment = bool(str(continuation_partial or "").strip())
        prior_evidence = (
            dict(continuation_evidence)
            if continuing_prior_segment and isinstance(continuation_evidence, dict)
            else {}
        )
        def _prior_int(key: str, default: int) -> int:
            try:
                return int(prior_evidence.get(key, default))
            except (TypeError, ValueError):
                return default

        prior_segment_count = max(
            1,
            _prior_int("foreground_model_generation_segment_count", 1),
        ) if continuing_prior_segment else 0
        prior_generation_count = max(
            prior_segment_count,
            _prior_int("foreground_model_generation_count", prior_segment_count),
        ) if continuing_prior_segment else 0
        prior_retry_count = max(
            0,
            _prior_int("completion_retry_count", 0),
        ) if continuing_prior_segment else 0
        prior_transaction_count = max(
            1,
            _prior_int("foreground_model_generation_transaction_count", 1),
        ) if continuing_prior_segment else 0
        prior_transaction_id = str(
            prior_evidence.get("foreground_model_generation_transaction_id") or ""
        ).strip()
        prior_resume_handle = str(
            prior_evidence.get("continuation_resume_handle") or ""
        ).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{32}", prior_resume_handle):
            prior_resume_handle = ""
        transaction_id = (
            prior_transaction_id
            if continuing_prior_segment
            else uuid.uuid4().hex
        )
        continuation_evidence_valid = bool(
            not continuing_prior_segment
            or (
                prior_transaction_count == 1
                and prior_generation_count == prior_segment_count
                and prior_segment_count == prior_retry_count + 1
                and prior_retry_count <= _MAX_USER_SURFACE_CONTINUATIONS
                and prior_transaction_id
            )
        )
        turn_trace.update(
            {
                "cognitive_engine_required": bool(require_engine),
                "engine_think_invoked": False,
                "cognitive_engine_reply_accepted": False,
                "cognitive_engine_reply_failed": False,
                "bounded_contract_used": False,
                "legacy_fallback_used": False,
                "live_mind_controls_bound": False,
                "live_mind_generation_controls": {},
                "live_mind_surface_control_receipt": {},
                "live_mind_controls_worker_applied": False,
                "foreground_model_generation_consumed": continuing_prior_segment,
                "foreground_model_generation_count": prior_generation_count,
                "foreground_model_generation_transaction_count": prior_transaction_count,
                "foreground_model_generation_segment_count": prior_segment_count,
                "foreground_model_generation_transaction_id": transaction_id,
                "completion_retry_count": prior_retry_count,
                "continuation_evidence_valid": continuation_evidence_valid,
                "continuation_resume_handle": prior_resume_handle,
                "repair_retry_attempt_count": 0,
                "single_owner_generation_exhausted": False,
                "response_path": "",
            }
        )
        _prime_requested_output_contract_trace(
            turn_trace,
            user_message=visible,
        )

    def _mark_turn_trace(**fields: Any) -> None:
        if turn_trace is not None:
            turn_trace.update(fields)

    def _record_foreground_generation(
        metadata: Any,
        *,
        continuation_segment: bool = False,
        response_text: str = "",
    ) -> None:
        """Record physical decodes without confusing segments with new answers."""

        proven = _generation_metadata_consumed_foreground_owner(
            metadata,
            response_text=response_text,
        )
        if not proven:
            # What the turn itself recorded, when the metadata proves nothing.
            #
            # A receipt is published on the object that ran the generation, and
            # the tool loop runs on a different object from the one this layer
            # reads. The turn is what they share, and a generation recorded
            # there carries the same fact: this model wrote this many tokens
            # for this turn.
            #
            # Nothing is waived. Repair text and legacy fallbacks never write
            # that record — it is made where the model's own tokens are
            # counted, and a generation of zero tokens does not write one.
            proven = bool(_this_turn_generated_something())
        if turn_trace is None or not require_engine or not proven:
            return
        turn_trace["foreground_model_generation_consumed"] = True
        turn_trace["foreground_model_generation_count"] = (
            int(turn_trace.get("foreground_model_generation_count") or 0) + 1
        )
        turn_trace["foreground_model_generation_segment_count"] = (
            int(turn_trace.get("foreground_model_generation_segment_count") or 0) + 1
        )
        if not continuation_segment:
            turn_trace["foreground_model_generation_transaction_count"] = (
                int(turn_trace.get("foreground_model_generation_transaction_count") or 0) + 1
            )
        elif int(turn_trace.get("foreground_model_generation_transaction_count") or 0) == 0:
            # A route-level continuation can resume a durable prior segment. It
            # still belongs to one logical answer transaction.
            turn_trace["foreground_model_generation_transaction_count"] = 1
        turn_trace["single_owner_generation_exhausted"] = True

    def _adopt_generation_metadata(
        metadata: Any,
        *,
        source_label: str,
        adopt_response_path: bool = True,
        inherit_turn_context: bool = True,
        count_foreground_generation: bool = True,
        response_text: str = "",
    ) -> None:
        """Bind the accepted generation's receipt without retaining stale fields."""

        if turn_trace is None:
            return
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        raw_generation_controls = metadata.get("live_mind_generation_controls")
        generation_controls = (
            dict(raw_generation_controls) if isinstance(raw_generation_controls, dict) else {}
        )
        existing_generation_controls = turn_trace.get("live_mind_generation_controls")
        if (
            inherit_turn_context
            and not generation_controls
            and isinstance(existing_generation_controls, dict)
        ):
            generation_controls = dict(existing_generation_controls)
        snapshot_ready = bool(
            (inherit_turn_context and turn_trace.get("live_mind_snapshot_ready"))
            or metadata.get("live_mind_snapshot_ready")
        )
        required_subsystems_ok = bool(
            (inherit_turn_context and turn_trace.get("live_mind_required_subsystems_ok"))
            or metadata.get("live_mind_required_subsystems_ok")
        )
        controls_bound = bool(
            generation_controls
            and (
                metadata.get("live_mind_controls_bound")
                or (snapshot_ready and required_subsystems_ok)
            )
        )
        raw_receipt = metadata.get("live_mind_surface_control_receipt")
        raw_receipt_present = isinstance(raw_receipt, dict) and bool(raw_receipt)
        receipt = normalize_live_mind_surface_control_receipt(
            raw_receipt if isinstance(raw_receipt, dict) else {},
            controls_bound=controls_bound,
            generation_controls=generation_controls,
            surface_quality_gate_passed=(None if raw_receipt_present else False),
            source=source_label,
        )
        prior_mutations = merge_text_mutations(
            (turn_trace.get("live_mind_surface_control_receipt") or {}).get("text_mutations"),
            turn_trace.get("text_mutations"),
        )
        receipt_mutations = merge_text_mutations(
            prior_mutations,
            receipt.get("text_mutations"),
        )
        receipt["text_mutations"] = receipt_mutations
        receipt["text_mutation_count"] = len(receipt_mutations)
        receipt["deterministic_repair_applied"] = any(
            bool(item.get("deterministic")) for item in receipt_mutations
        )
        worker_applied = bool(
            metadata.get("live_mind_controls_worker_applied")
            or (receipt.get("live_mind_controls_bound") and receipt.get("applied"))
        )
        generation_required = bool(
            metadata.get(
                "live_mind_generation_required",
                receipt.get("generation_required", True),
            )
        )
        metadata_response_path = str(metadata.get("response_path") or "").strip()
        qualified_recurrent_path = (
            metadata_response_path == "cognitive_engine_qualified_recurrent"
        )
        qualified_recurrent_receipt = metadata.get("qualified_recurrent_receipt")
        qualified_recurrent_family = str(
            metadata.get("qualified_recurrent_family") or ""
        ).strip()
        qualified_recurrent_errors: list[str] = []
        if qualified_recurrent_path:
            try:
                from core.brain.llm.qualified_recurrent_ingress import (
                    qualified_recurrent_result_receipt_errors,
                )

                qualified_recurrent_errors = qualified_recurrent_result_receipt_errors(
                    qualified_recurrent_receipt,
                    answer_text=response_text,
                    expected_family=qualified_recurrent_family,
                )
            except (ImportError, TypeError, ValueError) as exc:
                qualified_recurrent_errors = [
                    f"qualified_recurrent_result_validation_unavailable:{type(exc).__name__}"
                ]
        qualified_recurrent_path_proven = bool(
            qualified_recurrent_path
            and metadata.get("qualified_recurrent_succeeded") is True
            and metadata.get("model_generation_used") is False
            and generation_required is False
            and not qualified_recurrent_errors
        )
        turn_trace.update(
            {
                "live_mind_controls_bound": controls_bound,
                "live_mind_generation_controls": generation_controls,
                "live_mind_surface_control_receipt": receipt,
                "live_mind_controls_worker_applied": worker_applied,
                "live_mind_generation_required": generation_required,
                "live_mind_snapshot_ready": snapshot_ready,
                "live_mind_required_subsystems_ok": required_subsystems_ok,
                "live_mind_snapshot_ready_from_thought": bool(
                    metadata.get("live_mind_snapshot_ready")
                ),
                "live_mind_required_subsystems_ok_from_thought": bool(
                    metadata.get("live_mind_required_subsystems_ok")
                ),
                "semantic_completion_receipt_present": all(
                    field in receipt
                    for field in (
                        "semantic_completion_contract",
                        "semantic_completion_satisfied",
                        "semantic_completion_incomplete",
                    )
                ),
                "semantic_completion_contract": bool(
                    receipt.get("semantic_completion_contract", False)
                ),
                "semantic_completion_satisfied": bool(
                    receipt.get("semantic_completion_satisfied", False)
                ),
                "semantic_completion_incomplete": bool(
                    receipt.get("semantic_completion_incomplete", False)
                ),
                "reply_generation_incomplete": bool(
                    metadata.get("reply_generation_incomplete", False)
                ),
                "text_mutations": receipt_mutations,
                "text_mutation_count": len(receipt_mutations),
                "qualified_recurrent_path_proven": qualified_recurrent_path_proven,
                "qualified_recurrent_family": qualified_recurrent_family,
                "qualified_recurrent_receipt": (
                    dict(qualified_recurrent_receipt)
                    if isinstance(qualified_recurrent_receipt, dict)
                    else {}
                ),
                "qualified_recurrent_delivery_errors": qualified_recurrent_errors,
                "qualified_recurrent_succeeded": bool(
                    metadata.get("qualified_recurrent_succeeded", False)
                ),
                "model_generation_used": metadata.get("model_generation_used"),
            }
        )
        if qualified_recurrent_path_proven:
            turn_trace["authored_answer_completion_proven"] = True
        if count_foreground_generation:
            _record_foreground_generation(metadata, response_text=response_text)
        latent_metadata_present = any(
            key in metadata
            for key in (
                "latent_cortex_selected",
                "latent_cortex_attempted",
                "latent_cortex_succeeded",
                "latent_cortex_fallback_used",
                "latent_cortex_failure_reason",
                "latent_cortex_receipt",
                "latent_cortex_progress",
            )
        )
        _note_the_latent_metadata(
            latent_metadata_present=latent_metadata_present,
            metadata=metadata,
            turn_trace=turn_trace,
        )
        if adopt_response_path and metadata_response_path:
            turn_trace["response_path"] = metadata_response_path
        if bool(metadata.get("model_retry_suppressed", False)):
            turn_trace["model_retry_suppressed"] = True
            turn_trace["single_owner_generation_exhausted"] = True
            turn_trace["generation_failure_class"] = str(
                metadata.get("generation_failure_class") or ""
            )[:120]

    failure_incident_recorded = False

    def _record_exhausted_cognitive_failure(
        reason: str,
        *,
        retry_attempted: bool,
    ) -> None:
        """Persist one causal incident after bounded live-turn recovery is exhausted."""
        nonlocal failure_incident_recorded
        if failure_incident_recorded:
            return
        failure_incident_recorded = True
        normalized_reason = str(reason or "cognitive_reply_failed")[:240]
        resilience = _route_desktop_cognitive_failure_to_resilience(
            normalized_reason,
            source=source,
            session_present=bool(session_id),
            retry_attempted=retry_attempted,
        )
        record_degradation(
            "chat.cognitive_engine_reply",
            RuntimeError(normalized_reason),
            severity="degraded",
            action=(
                "bounded same-worker correction exhausted; retained a durable incident "
                "for resilience pressure and repeat-triggered repair routing"
            ),
            receipt_required=True,
            extra={
                "failure_class": normalized_reason,
                "request_surface": str(source or "")[:80],
                "session_present": bool(session_id),
                "retry_attempted": bool(retry_attempted),
                **resilience,
            },
            enforce_failure_policy=False,
        )
        _mark_turn_trace(
            failure_incident_recorded=True,
            failure_incident_reason=normalized_reason,
            bounded_correction_attempted=bool(retry_attempted),
            resilience_routing=resilience,
        )

    preflight_context = str(preflight_context_message or "").strip()
    if preflight_context == visible.strip():
        preflight_context = ""
    try:
        from core.senses.turn_evidence import TurnSensoryEvidence

        _normalized_sensory_evidence = TurnSensoryEvidence.from_value(turn_sensory_evidence)
        sensory_evidence_payload = (
            _normalized_sensory_evidence.to_dict()
            if _normalized_sensory_evidence is not None
            else {}
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.sensory_evidence",
            exc,
            action="continued without malformed turn sensory evidence",
        )
        sensory_evidence_payload = {}
    mode = _select_cognitive_chat_mode(visible, effective_user_message)
    shape = analyze_prompt_shape(visible)
    semantic_completion_expected = True
    _mark_turn_trace(semantic_completion_contract_expected=semantic_completion_expected)
    capability_inventory_contract = _chat_preflight._is_explicit_capability_inventory_request(
        visible
    )
    desktop_execution_contract = _chat_preflight._looks_like_desktop_objective(visible)
    paired_information_reply = (
        _chat_preflight._paired_device_information_scope_reply(visible, lane=lane)
        if conversation_only_surface
        else None
    )
    if paired_information_reply is not None:
        reply, status = paired_information_reply
        _mark_turn_trace(
            bounded_contract_used=True,
            response_path=status,
        )
        return reply
    if conversation_only_surface and desktop_execution_contract:
        _mark_turn_trace(
            bounded_contract_used=True,
            response_path="paired_device_action_scope_denied",
        )
        return (
            "This paired device is scoped to conversation and read-only world viewing. "
            "Desktop, file, tool, and control actions require the owner surface."
        )
    assistant_mode_recovery_contract = bool(
        require_engine
        and _is_assistant_mode_recovery_request(visible)
        and not _chat_preflight._is_runtime_fact_status_request(visible)
    )
    bounded_planning_reply = _chat_desktop_repair._build_bounded_planning_reply(visible)
    bounded_planning_contract = bool(bounded_planning_reply)
    if (
        require_engine
        and timeout_s is not None
        and float(timeout_s) < _DESKTOP_COGNITIVE_MIN_REQUIRED_BUDGET_S
    ):
        logger.warning(
            "Required desktop CognitiveEngine budget %.1fs is below %.1fs; refusing doomed foreground turn.",
            float(timeout_s),
            _DESKTOP_COGNITIVE_MIN_REQUIRED_BUDGET_S,
        )
        if turn_trace is not None:
            turn_trace.update({"response_path": "insufficient_cognitive_budget"})
        return None
    failure_mode_reply = _chat_desktop_repair._build_failure_mode_surface_reply(visible)
    failure_mode_contract = bool(failure_mode_reply)
    if (
        desktop_execution_contract
        and require_engine
        and _desktop_objective_self_sufficient_without_cognitive_text(visible)
    ):
        logger.info(
            "Serving self-sufficient desktop execution contract without foreground model allocation."
        )
        if turn_trace is not None:
            turn_trace.update(
                {
                    "bounded_contract_used": True,
                    "response_path": "self_sufficient_desktop_execution_contract",
                }
            )
        return (
            "I will execute this through the governed desktop_task lane and report only "
            "receipt-verified effects. If desktop_task cannot prove the effect, I will "
            "report the blocker instead of claiming completion."
        )
    private_cognitive_model_contract = bool(
        require_engine and _chat_preflight._is_private_cognitive_model_request(visible)
    )
    identity_continuity_contract = bool(
        require_engine
        and (
            _chat_desktop_repair._is_identity_request(visible)
            or _chat_desktop_repair._identity_request_asks_future_memory(visible)
        )
    )
    runtime_fact_status_contract = _chat_preflight._is_runtime_fact_status_request(visible)
    grounded_runtime_status_context = (
        _ground_runtime_fact_status_reply(
            visible,
            "",
            lane,
            cognitive_engine_handled=True,
        )
        if runtime_fact_status_contract
        else ""
    )
    inherited_self_condition_contract = False
    try:
        (
            self_condition_contract,
            inherited_self_condition_contract,
        ) = _classify_self_condition_contract(
            visible,
            referential_anchor=referential_anchor,
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.self_condition", exc)
        self_condition_contract = False
    self_condition_evidence: dict[str, Any] = {}
    if self_condition_contract:
        try:
            self_condition_evidence = _build_self_condition_evidence(
                visible,
                session_id=session_id,
            )
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat.self_condition", exc)
            logger.debug("Self-condition evidence unavailable before CognitiveEngine: %s", exc)
    canonical_self_condition_context = str(
        self_condition_evidence.get("prompt_block") or ""
    ).strip()
    canonical_self_condition_reply = str(self_condition_evidence.get("reply") or "").strip()
    # What she can do RIGHT NOW, for whatever this turn reaches for.
    #
    # The capability engine has always known which skills are available and
    # why one is not; none of it reached the part of her that speaks, so a
    # missing skill produced a string written months earlier ("I can't access
    # external data right now, but based on what I know..."). Not her voice,
    # not this moment, and it flattened the one distinction a person actually
    # needs: "no network this minute" is not "I have no way to search".
    live_capability_condition = ""
    try:
        from core.conversation.capability_condition import (
            capability_condition_evidence,
        )

        # A capability whose evidence is already in this prompt worked.
        _proven_this_turn: list[str] = []
        if "[WEB SEARCH EVIDENCE]" in str(effective_user_message or ""):
            _proven_this_turn.append("web_search")
        if not state_native_output_owner:
            live_capability_condition = capability_condition_evidence(
                visible, already_used=_proven_this_turn
            )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.capability_condition", exc)
        logger.debug("Live capability condition unavailable: %s", exc)

    # Tell the morphogenetic layer what this turn reaches for.
    #
    # It had no demand input at all, so its population sat at whatever boot
    # registered and its cells had nothing to organise around. This is the one
    # global fact a local cell may read: what the body is trying to do, not
    # where the load is or who is struggling.
    #
    # Purely additive — it appends a signal to a bounded in-memory deque, does
    # no I/O, and cannot change this reply. A failure here is a failure to
    # inform a background layer, so it is logged and dropped.
    try:
        from core.morphogenesis.bridge import announce_demand

        announce_demand(visible)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.morphogenesis_demand", exc, severity="debug")
        logger.debug("Morphogenesis demand signal skipped: %s", exc)

    canonical_memory_state_evidence = (
        ""
        if conversation_only_surface
        else _extract_canonical_memory_state_evidence_block(effective_user_message)
    )
    memory_state_contract = bool(canonical_memory_state_evidence)
    # Does that contract cover the whole turn, or only part of it?
    #
    # The token ladder caps a memory-state turn at 256 tokens because "what did
    # I pin?" is a short factual answer. It is not short when the same message
    # also asks something real. Live 2026-07-27: "Remember this: my project
    # codename is HELIOTROPE, build 4471. Separately — do you think a system
    # like you can actually prefer one thing over another?" was budgeted at 172
    # tokens, cut off mid-sentence, flagged truncated_tail, and replaced with
    # the pin confirmation. The philosophical half never had room to exist.
    #
    # The parser that found the pin is the part that knows what else is in the
    # message, so it says so here rather than leaving the budgeter to guess.
    memory_state_contract_covers_turn = memory_state_contract and not (
        _chat_memory_state._turn_has_substance_beyond_memory_request(visible)
    )

    engine = ServiceContainer.get("cognitive_engine", default=None)
    if engine is None or not hasattr(engine, "think"):
        if turn_trace is not None:
            turn_trace.update(
                {
                    "cognitive_engine_available": False,
                    "response_path": "cognitive_engine_unavailable",
                }
            )
        return None
    if turn_trace is not None:
        turn_trace["cognitive_engine_available"] = True
    engine_resolved_at = time.perf_counter()
    if runtime_fact_status_contract and not require_engine:
        logger.info(
            "Serving bounded desktop runtime-status contract without foreground model allocation."
        )
        if turn_trace is not None:
            turn_trace.update(
                {
                    "bounded_contract_used": True,
                    "response_path": "bounded_runtime_status_contract",
                }
            )
        return _ground_runtime_fact_status_reply(
            visible,
            "",
            lane,
            cognitive_engine_handled=True,
        )

    if capability_inventory_contract:
        from core.brain.types import ThinkingMode

        mode = ThinkingMode.FAST
        preflight_context = ""
    compact_desktop_chat_contract = bool(
        not state_native_output_owner
        and _is_compact_desktop_chat_contract(
            visible,
            effective_user_message,
            desktop_execution_contract=desktop_execution_contract,
            capability_inventory_contract=capability_inventory_contract,
            identity_continuity_contract=identity_continuity_contract,
        )
    )
    prompt_shape_payload = shape.to_dict()
    # Required live desktop turns must exercise CognitiveEngine, but they do not
    # all need the heavyweight phase stack. Simple conversation uses the compact
    # live-mind speech contract; execution, identity/self-process, long, and
    # multi-part turns are still excluded above and flow through deeper planning.
    recent_context_needed = bool(
        not state_native_output_owner and _desktop_turn_needs_recent_context(visible)
    )
    # Several clauses can belong to one state report (for example, condition
    # plus known-versus-inferred evidence). Question count alone cannot decide
    # whether the state contract covers the turn. Competing operational or
    # retrieval contracts are the mechanical evidence that it does not.
    self_condition_contract_covers_turn = bool(
        self_condition_contract
        and not (
            desktop_execution_contract
            or capability_inventory_contract
            or bounded_planning_contract
            or memory_state_contract
            or runtime_fact_status_contract
            or identity_continuity_contract
            or private_cognitive_model_contract
        )
    )
    from core.conversation.delivered_history import VISIBLE_CONVERSATION_EXCHANGES

    # Question classification controls current evidence, not access to the
    # transcript. Read the same bounded history surface restored by the UI;
    # the inference owner allocates it against the serving context capacity.
    recent_context_limit = (
        VISIBLE_CONVERSATION_EXCHANGES
        if require_engine or recent_context_needed
        else 0
    )
    if state_native_output_owner:
        recent_exchanges = []
    elif recent_context_limit > 0:
        recent_exchanges = await _chat_memory_state._recent_completed_conversation_exchanges(
            current_user_message=visible,
            session_id=session_id,
            limit=recent_context_limit,
            allow_cross_session=True,
        )
    else:
        recent_exchanges = []
    transcript_user_messages = [
        str(exchange.get("user") or "").strip()
        for exchange in recent_exchanges
        if isinstance(exchange, dict) and str(exchange.get("user") or "").strip()
    ]
    route_recent_user_messages = list(transcript_user_messages)
    if visible and visible not in route_recent_user_messages:
        route_recent_user_messages.append(visible)
    route_assessment_grounding = [
        text
        for exchange in recent_exchanges
        if isinstance(exchange, dict)
        for text in (
            str(exchange.get("user") or "").strip(),
            str(exchange.get("aura") or "").strip(),
        )
        if text
    ]
    route_assessment_antecedent = next(
        (
            str(exchange.get("aura") or "").strip()
            for exchange in reversed(recent_exchanges)
            if isinstance(exchange, dict) and str(exchange.get("aura") or "").strip()
        ),
        "",
    )
    # Outer delivery gates run after this helper returns. Bind the authenticated
    # transcript to turn custody so each gate judges against the same evidence.
    from core.conversation.turn_evidence_custody import (
        record_turn_grounding,
        record_turn_transcript,
    )

    if not state_native_output_owner and recent_context_limit > 0:
        record_turn_transcript(recent_exchanges)
    for transcript_evidence in route_assessment_grounding:
        record_turn_grounding(transcript_evidence)
    recent_conversation_context = (
        _format_recent_conversation_context(recent_exchanges) if recent_exchanges else ""
    )
    discourse_repair_contract: dict[str, Any] = {}
    if recent_exchanges:
        try:
            from core.conversation.discourse_repair_pursuit import build_repair_pursuit
            from core.utils.injected_blocks import stamp_runtime_payload

            pursuit = build_repair_pursuit(visible, recent_exchanges)
            if pursuit.active:
                discourse_repair_contract = stamp_runtime_payload(pursuit.to_dict())
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "chat.discourse_repair_pursuit",
                exc,
                severity="warning",
                action="kept ordinary authenticated conversation history",
            )
    live_mind_context = (
        {}
        if state_native_output_owner
        else await _collect_live_mind_context_payload(
            user_message=visible,
            lane=lane,
            recent_conversation_context=recent_conversation_context,
            recent_context_needed=recent_context_needed,
            require_engine=require_engine,
            conversation_only_surface=conversation_only_surface,
        )
    )
    context = {
        "route": "desktop_chat",
        "source": source,
        "visible_user_message": visible[:1000],
        "raw_user_message": raw_visible[:1000],
        "declared_interlocutor": interlocutor_evidence,
        "foreground_request": True,
        "user_facing": True,
        "evidence_profile": evidence_profile,
        "preflight_context_message": preflight_context[:8000],
        "turn_sensory_evidence": sensory_evidence_payload,
        "recent_completed_exchanges": recent_exchanges,
        "recent_conversation_context": recent_conversation_context,
        "recent_context_needed": recent_context_needed,
        "discourse_repair_contract": discourse_repair_contract,
        "action_episode_evidence": action_episode_evidence[:1800],
        "prior_answer_provenance": dict(prior_answer_provenance or {}),
        "live_mind_context": live_mind_context,
        "live_mind_context_required": bool(require_engine and not state_native_output_owner),
        "require_full_foreground_mind_reply": bool(
            require_engine and not state_native_output_owner
        ),
        "live_mind_required_subsystems": dict(live_mind_context.get("required_subsystems") or {}),
        "live_mind_required_subsystems_ok": bool(live_mind_context.get("required_subsystems_ok")),
        "cognitive_engine_required": bool(require_engine),
        "compact_desktop_chat_contract": compact_desktop_chat_contract,
        "capability_inventory_contract": capability_inventory_contract,
        "conversation_only_surface": bool(conversation_only_surface),
        "tool_execution_policy": ("deny" if conversation_only_surface else "governed"),
        "assistant_mode_recovery_contract": assistant_mode_recovery_contract,
        "bounded_planning_contract": bounded_planning_contract,
        "bounded_planning_reply": bounded_planning_reply or "",
        "failure_mode_contract": failure_mode_contract,
        "private_cognitive_model_contract": private_cognitive_model_contract,
        "identity_continuity_contract": identity_continuity_contract,
        "runtime_fact_status_contract": runtime_fact_status_contract,
        "grounded_runtime_status_contract": runtime_fact_status_contract,
        "grounded_runtime_status_context": grounded_runtime_status_context,
        "memory_state_contract": memory_state_contract,
        "memory_state_contract_covers_turn": memory_state_contract_covers_turn,
        "canonical_memory_state_evidence": canonical_memory_state_evidence,
        "self_condition_contract": self_condition_contract,
        "self_condition_contract_inherited": inherited_self_condition_contract,
        "self_condition_contract_covers_turn": self_condition_contract_covers_turn,
        "canonical_self_condition_context": canonical_self_condition_context,
        "live_capability_condition": live_capability_condition,
        "canonical_self_condition_reply": canonical_self_condition_reply,
        "canonical_self_condition_projection": dict(
            self_condition_evidence.get("projection_dict") or {}
        ),
        "conversation_lane": (
            _chat_preflight._paired_conversation_lane_payload(lane)
            if conversation_only_surface
            else dict(lane or {})
        ),
        "prompt_shape": dict(prompt_shape_payload),
        "completed_capability_evidence": completed_capability_evidence,
    }
    if conversation_resume_handle:
        context[
            "user_surface_conversation_resume_handle"
        ] = conversation_resume_handle
    from core.conversation.user_surface_contract import bind_user_surface_prompt

    bind_user_surface_prompt(
        context,
        visible,
        source="desktop_chat.visible_user_message",
        overwrite=True,
    )
    exact_principal = " ".join(str(principal_id or "").strip().split())[:160]
    if exact_principal:
        context["user_id"] = exact_principal
    if require_engine:
        # This is a hard live-SLA cap shared by both the compact speech lane and
        # the deeper phase stack.  Previously only the compact lane carried the
        # cap, so a one-part introspective follow-up could enter ResponseGeneration,
        # multiply its budget through several cognitive biases, and request 1.4K+
        # tokens from the local 32B worker.  The outer desktop deadline then
        # cancelled an otherwise healthy model and repeated the same oversized
        # attempt.  Depth may change the work performed, but it may not silently
        # discard the foreground completion envelope.
        live_reply_token_budget = _desktop_live_reply_token_budget(
            visible,
            capability_inventory_contract=capability_inventory_contract,
            bounded_planning_contract=bounded_planning_contract,
            runtime_fact_status_contract=runtime_fact_status_contract,
            memory_state_contract=memory_state_contract,
            memory_state_contract_covers_turn=memory_state_contract_covers_turn,
        )
        context["max_tokens"] = live_reply_token_budget
        context["num_predict"] = live_reply_token_budget
        context["user_surface_completion_floor"] = answer_surface_token_floor(visible)
    if private_cognitive_model_contract:
        context["grounded_private_model_context"] = (
            _chat_conversation_repair._build_grounded_introspection_reply(visible) or ""
        )[:4000]
    if identity_continuity_contract:
        context["grounded_identity_continuity_context"] = (
            _chat_desktop_repair._build_identity_reply(visible) or ""
        )[:3000]
    if capability_inventory_contract:
        context["grounded_capability_inventory_context"] = (
            _chat_desktop_repair._build_grounded_capability_inventory_reply(visible) or ""
        )
    if _is_self_claim_boundary_question(visible):
        context["evidence_bound_self_claim_context"] = (
            _build_evidence_bound_self_claim_reply(visible, lane=lane) or ""
        )[:3000]
    conversation_recall_context = (
        ""
        if capability_inventory_contract or state_native_output_owner
        else await _chat_memory_state._build_conversation_recall_reply(
            visible,
            session_id=session_id,
        )
    )
    if conversation_recall_context:
        context["conversation_recall_evidence"] = conversation_recall_context[:3000]
    retained_memory_evidence_context = (
        ""
        if capability_inventory_contract or conversation_only_surface or state_native_output_owner
        else await _build_retained_memory_evidence_context(
            visible,
            session_id=session_id,
            recent_exchanges=recent_exchanges,
            conversation_recall_context=conversation_recall_context,
        )
    )
    if retained_memory_evidence_context:
        context["retained_memory_evidence_context"] = retained_memory_evidence_context
    context_challenge_context = (
        ""
        if capability_inventory_contract or state_native_output_owner
        else await _build_context_challenge_repair_reply(
            visible,
            session_id=session_id,
        )
    )
    if context_challenge_context and "pitch" in _chat_memory_state._normalize_user_message(visible):
        context_challenge_context = (
            "No pitch is supported by the recent completed conversation context. "
            "The correct answer is to say that no pitch is visible in the recent thread, "
            "then reset to the actual conversation instead of inventing one."
        )
    if context_challenge_context:
        context["contextual_relevance_evidence"] = context_challenge_context[:2500]
    deep_memory_context = (
        ""
        if capability_inventory_contract or state_native_output_owner
        else await _fetch_deep_memory_context(visible)
    )
    if deep_memory_context:
        context["deep_memory_context"] = deep_memory_context[:3000]
    context_bound_at = time.perf_counter()
    if turn_trace is not None:
        mind_snapshot_quality = dict(live_mind_context.get("mind_snapshot_quality") or {})
        turn_trace.update(
            {
                "recent_context_needed": recent_context_needed,
                "recent_context_exchanges": len(recent_exchanges),
                "live_mind_context_present": bool(live_mind_context),
                "live_mind_context_required": bool(
                    require_engine and not state_native_output_owner
                ),
                "live_mind_snapshot_present": bool(mind_snapshot_quality.get("present")),
                "live_mind_snapshot_ready": bool(mind_snapshot_quality.get("ready")),
                "live_mind_snapshot_missing_services": list(
                    mind_snapshot_quality.get("missing_services") or []
                ),
                "live_mind_required_subsystems": dict(
                    live_mind_context.get("required_subsystems") or {}
                ),
                "live_mind_required_subsystems_attested": bool(
                    is_stamped_runtime_payload(live_mind_context)
                ),
                "live_mind_required_subsystems_ok": bool(
                    live_mind_context.get("required_subsystems_ok")
                ),
                "architecture_context_bound": bool(
                    require_engine
                    and live_mind_context
                    and live_mind_context.get("required_subsystems_ok")
                ),
                "compact_desktop_chat_contract": compact_desktop_chat_contract,
                "desktop_execution_contract": desktop_execution_contract,
                "capability_inventory_contract": capability_inventory_contract,
                "assistant_mode_recovery_contract": assistant_mode_recovery_contract,
                "bounded_planning_contract": bounded_planning_contract,
                "failure_mode_contract": failure_mode_contract,
                "private_cognitive_model_contract": private_cognitive_model_contract,
                "identity_continuity_contract": identity_continuity_contract,
                "runtime_fact_status_contract": runtime_fact_status_contract,
                "memory_state_contract": memory_state_contract,
                "self_condition_contract": self_condition_contract,
                "self_condition_contract_inherited": inherited_self_condition_contract,
                "prompt_shape": dict(prompt_shape_payload),
            }
        )
    trace_binding_finished_at = time.perf_counter()
    if require_engine:
        context.update(
            {
                "desktop_cognitive_engine_required": True,
                "protected_foreground_lane": True,
                "prefer_tier": "primary",
                "deep_handoff": False,
                "allow_deep_handoff": False,
                "allow_cloud_fallback": False,
                "live_runtime_payload_required": not state_native_output_owner,
            }
        )
        if not state_native_output_owner:
            context["mind_context_contract"] = (
                "Use live_mind_context as causal grounding for this reply. "
                "Do not answer as a raw assistant, do not ignore the current user turn, "
                "and do not claim a subsystem state that contradicts live_mind_context."
            )
    if capability_inventory_contract:
        context.update(
            {
                "capability_inventory_contract": True,
                "desktop_descriptive_turn": True,
                "prefer_tier": "primary",
                "deep_handoff": False,
                "allow_deep_handoff": False,
                "max_tokens": 384,
                "num_predict": 384,
                "skip_runtime_payload": True,
                "disable_prompt_cache": True,
                "clear_prompt_cache": True,
                "response_style_contract": (
                    "Answer from grounded_capability_inventory_context. "
                    "Use four short sentences only: practical capability categories including the exact phrase browser/web research; governed execution through "
                    "Will/Authority or permissions; receipts/effect verification; one hypothetical chain plus the boundary that you are not executing tools in this turn. "
                    "Keep it complete under 80 words."
                ),
            }
        )
    contract_binding_finished_at = time.perf_counter()
    expression_frame_ms = 0.0
    if compact_desktop_chat_contract:
        existing_style_contract = str(context.get("response_style_contract") or "").strip()
        live_reply_token_budget = int(context.get("max_tokens") or 896)
        expression_frame_started_at = time.perf_counter()
        live_speech_grounding_frame = _chat_desktop_repair._build_aura_expression_frame(
            visible
        )
        expression_frame_ms = (
            time.perf_counter() - expression_frame_started_at
        ) * 1000.0
        context.update(
            {
                "desktop_quick_reply_contract": True,
                "desktop_descriptive_turn": True,
                "deep_handoff": False,
                "allow_deep_handoff": False,
                "max_tokens": live_reply_token_budget,
                "num_predict": live_reply_token_budget,
                "skip_runtime_payload": True,
                "live_runtime_payload_required": bool(require_engine),
                "live_speech_grounding_frame": live_speech_grounding_frame,
                # The ordinary chat turn is the ONE lane that has to reuse KV:
                # its prompt is the whole conversation, so re-prefilling from
                # token zero is what makes turn latency climb until it crosses
                # the turn budget and long conversations stop answering.
                # `disable_prompt_cache`/`clear_prompt_cache` were set here in
                # June 2026, when the 32B's cache budget was zero anyway, so
                # they cost nothing then; the budget was later restored FOR
                # endurance and these two were never lifted, which quietly kept
                # the restore from reaching the conversation. Reuse is scoped to
                # `user_surface`, so no internal lane can see this KV, and the
                # only "contamination" within the scope is the conversation's
                # own history as it was actually computed.
                "response_style_contract": (
                    "Answer the user's live desktop chat turn directly and naturally. "
                    "Use live runtime state only as causal grounding; do not recite a telemetry card, "
                    "do not name raw moods as a greeting, and do not claim to be Claude, ChatGPT, Anthropic, "
                    "OpenAI, or a generic assistant."
                ),
            }
        )
        if existing_style_contract:
            context["response_style_contract"] = (
                f"{context['response_style_contract']} {existing_style_contract}"
            )
        if capability_inventory_contract:
            context["response_style_contract"] = (
                str(context.get("response_style_contract") or "")
                + " The user is asking for a descriptive capability inventory, not execution. "
                "Answer from grounded_capability_inventory_context. Use four short sentences only: practical capability "
                "categories including the exact phrase browser/web research; governed execution through Will/Authority or permissions; receipts/effect "
                "verification; one hypothetical chain plus the boundary that you are not executing tools "
                "in this turn. Keep it complete under 80 words."
            )
        if bounded_planning_contract:
            context["response_style_contract"] = (
                str(context.get("response_style_contract") or "")
                + " This is a bounded planning turn. Answer in one natural paragraph of four to six "
                "complete sentences under 180 words. Cover the goal, authorization boundary, action "
                "sequence, effect verification, and bounded recovery. Do not use a numbered list unless "
                "the user explicitly requests one, and do not invent a specific example that replaces "
                "the user's stated task."
            )
        if _chat_desktop_repair._is_contextual_relevance_challenge(visible):
            context["contextual_relevance_challenge_contract"] = True
            context["response_style_contract"] = (
                str(context.get("response_style_contract") or "")
                + " The user is challenging context relevance. Do not invent the missing thread. "
                "If the recent context does not support the object they named, say so directly, "
                "reset to the last completed exchange, and keep the reply grounded. "
                "Use contextual_relevance_evidence when present. Keep the answer to one or two "
                "complete sentences under 70 words, ending with normal punctuation."
            )
        if _is_self_claim_boundary_question(visible):
            context["response_style_contract"] = (
                str(context.get("response_style_contract") or "")
                + " For consciousness, sentience, self-awareness, inner-life, or personhood questions, "
                "answer from evidence_bound_self_claim_context: include evidence/uncertainty language, "
                "distinguish functional self-modeling from phenomenal consciousness or private qualia, "
                "and do not reduce Aura to a generic text prediction engine."
            )
        if identity_continuity_contract:
            context["response_style_contract"] = (
                str(context.get("response_style_contract") or "")
                + " The user is asking who or what Aura is. Answer from "
                "grounded_identity_continuity_context exactly enough to be correct; "
                "do not invent generic assistant identity and do not use a delayed repair path."
            )
        if conversation_recall_context:
            context["conversation_recall_contract"] = True
            context["response_style_contract"] = (
                str(context.get("response_style_contract") or "")
                + " The user is asking about recent conversation context. Answer from "
                "conversation_recall_evidence exactly enough to be correct; do not guess."
            )
        if retained_memory_evidence_context:
            context["retained_memory_evidence_contract"] = True
            context["response_style_contract"] = (
                str(context.get("response_style_contract") or "")
                + " The user is asking about memory or continuity. Use "
                "retained_memory_evidence_context for any remembered-session claim. "
                "If the evidence does not support the specific memory, say it is not verified; "
                "distinguish transcript/durable-memory evidence from subjective recollection."
            )
        if memory_state_contract:
            context["response_style_contract"] = (
                str(context.get("response_style_contract") or "")
                + " The user is asking about canonical live memory/state. Answer from "
                "canonical_memory_state_evidence as the source of truth, include the exact remembered "
                "content when present, and answer any lightweight live-state clause from live_mind_context. "
                "Do not answer an older topic from recent history."
            )
        if runtime_fact_status_contract and not memory_state_contract:
            context["response_style_contract"] = (
                str(context.get("response_style_contract") or "")
                + " This is a live runtime fact question. Use grounded_runtime_status_context "
                "as the authoritative source for model lane, CognitiveEngine participation, "
                "tool governance, recurrent depth, and fallback state. Do not invent readiness "
                "or availability claims. The route will bind the final wording to that evidence."
            )
        if _is_current_request_recap_request(visible):
            context["current_request_recap_contract"] = True
            context["response_style_contract"] = (
                str(context.get("response_style_contract") or "")
                + " The user is asking you to identify the current request. Start with "
                "'You asked me to...' or an equivalent direct recap, then answer any "
                "second part of the prompt."
            )
        if _chat_memory_state._normalize_user_message(visible).startswith(
            "you with me"
        ) or re.search(
            r"\b(?:you\s+with\s+me|still\s+with\s+me|are\s+you\s+(?:there|with\s+me))\b",
            visible,
            flags=re.IGNORECASE,
        ):
            context["response_style_contract"] = (
                str(context.get("response_style_contract") or "")
                + " For this presence check, start with a concrete first-person continuity signal "
                "like 'I'm here with you' and add one grounded sentence about staying on this thread."
            )
        if re.search(r"\b(?:two\s+rules?|one\s+example|invent)\b", visible, flags=re.IGNORECASE):
            context["response_style_contract"] = (
                str(context.get("response_style_contract") or "")
                + " If the user asks for invention with rules and an example, include explicit labels "
                "'Rule 1', 'Rule 2', and 'Example', and end with a complete sentence."
            )
    compact_binding_finished_at = time.perf_counter()
    if desktop_execution_contract:
        from core.brain.types import ThinkingMode

        mode = ThinkingMode.SLOW
        context.update(
            {
                "desktop_execution_contract": True,
                "foreground_request": True,
                "user_explicitly_authorized": True,
                "user_requested_action": True,
                "user_visible_desktop_action": True,
                "verification_required": True,
                "source": "desktop_ui",
                "origin": "user",
                "allow_heuristic_desktop_plan": True,
                "desktop_task_planning_schema": desktop_task_planning_schema(),
                "desktop_task_allowed_actions": DESKTOP_TASK_ALLOWED_ACTIONS,
                "max_tokens": 1024,
                "num_predict": 1024,
                "skip_runtime_payload": True,
                "disable_prompt_cache": True,
                "clear_prompt_cache": True,
                "response_style_contract": (
                    "Produce a bounded desktop-task execution draft. Prefer valid JSON "
                    "with optional document_body and steps from the provided schema. "
                    "Do not answer like a hosted chatbot. Aura has governed local desktop "
                    "control for this request, so never say you cannot interact with apps, "
                    "open Notes/Docs/Chrome, write text, or control the user's desktop when "
                    "the requested action is inside the desktop_task contract. "
                    "If prose is more appropriate, keep it concise and do not claim "
                    "desktop completion before desktop_task receipts verify it."
                ),
            }
        )
    desktop_binding_finished_at = time.perf_counter()
    engine_user_message = str(effective_user_message or "")
    if sensory_evidence_payload:
        try:
            from core.senses.turn_evidence import sensory_evidence_grounding_block

            _turn_sensory_block = sensory_evidence_grounding_block(sensory_evidence_payload)
            if _turn_sensory_block:
                engine_user_message = f"{engine_user_message}\n\n{_turn_sensory_block}"
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "chat.sensory_evidence",
                exc,
                action="continued with typed evidence still present in CognitiveEngine context",
            )

    # Her senses travel with her, not just with the turn that captured them.
    #
    # A screen read is intake, and intake she cannot refer back to is not
    # something she saw — it is something that passed through her. Without
    # this she answers "what's on my screen", then goes blind to it on the
    # very next sentence and has to look again, answering a question about
    # what she saw with a fresh reading of the present.
    #
    # Attached to the message because that is the channel proven to reach
    # the model on this lane; `preflight_context_message` is carried in the
    # context dict and read by nothing.
    # Carried when the turn could plausibly concern it. Attaching a screen
    # reading to "what's 17 times 4?" puts a window inventory in front of an
    # arithmetic question, which is noise at best; her senses should be
    # available, not narrated at every turn.
    try:
        from core.perception.observation_evidence import get_observation_memory

        _perception_brief = (
            get_observation_memory().sensory_brief()
            if not state_native_output_owner and _turn_may_concern_perception(visible)
            else ""
        )
        if _perception_brief:
            engine_user_message = f"{engine_user_message}\n\n{_perception_brief}"
    except _CHAT_RECOVERABLE_ERRORS as _perception_exc:
        record_degradation(
            "chat",
            _perception_exc,
            action=(
                "ran the turn without recent perception; she may not recall what she just looked at"
            ),
        )

    # She can read her own source, so a question about her code is answered
    # from the tree rather than from her weights.
    #
    # Live 2026-08-04 13:50 she showed `manage_load()` — a function in no
    # file of this repository — as "a small part of my cognitive
    # architecture", and admitted a turn later that she had written it for
    # the conversation. The floor that reads real files only runs after
    # generation FAILS, so a healthy turn never reached it. Carrying the
    # real excerpts in means the material she reasons over is code that
    # exists, and it arrives with the path and line it lives at, so she can
    # say where it is from.
    try:
        from core.self.source_excerpt import source_evidence_brief

        _source_brief = (
            source_evidence_brief(visible)
            if not state_native_output_owner and _turn_may_concern_own_source(visible)
            else ""
        )
        if _source_brief:
            engine_user_message = f"{engine_user_message}\n\n{_source_brief}"
    except _CHAT_RECOVERABLE_ERRORS as _source_exc:
        record_degradation(
            "chat",
            _source_exc,
            action=(
                "ran a question about her own code without reading the source "
                "tree; the reply may not be grounded in a real file"
            ),
        )
    evidence_binding_finished_at = time.perf_counter()

    engine_user_message = _compose_the_engine_message(
        capability_inventory_contract=capability_inventory_contract,
        context=context,
        context_challenge_context=context_challenge_context,
        conversation_recall_context=conversation_recall_context,
        engine_user_message=engine_user_message,
        grounded_runtime_status_context=grounded_runtime_status_context,
        memory_state_contract=memory_state_contract,
        require_engine=require_engine,
        runtime_fact_status_contract=runtime_fact_status_contract,
        state_native_output_owner=state_native_output_owner,
        visible=visible,
    )
    preparation_finished_at = time.perf_counter()
    final_binding_stages = {
        "trace_ms": round((trace_binding_finished_at - context_bound_at) * 1000.0, 2),
        "contracts_ms": round(
            (contract_binding_finished_at - trace_binding_finished_at) * 1000.0,
            2,
        ),
        "expression_frame_ms": round(expression_frame_ms, 2),
        "compact_ms": round(
            (compact_binding_finished_at - contract_binding_finished_at) * 1000.0,
            2,
        ),
        "desktop_ms": round(
            (desktop_binding_finished_at - compact_binding_finished_at) * 1000.0,
            2,
        ),
        "evidence_ms": round(
            (evidence_binding_finished_at - desktop_binding_finished_at) * 1000.0,
            2,
        ),
        "directives_ms": round(
            (preparation_finished_at - evidence_binding_finished_at) * 1000.0,
            2,
        ),
    }
    preparation_timings = {
        "contracts_and_engine_ms": round(
            (engine_resolved_at - preparation_started_at) * 1000.0,
            2,
        ),
        "context_binding_ms": round(
            (context_bound_at - engine_resolved_at) * 1000.0,
            2,
        ),
        "final_binding_ms": round(
            (preparation_finished_at - context_bound_at) * 1000.0,
            2,
        ),
        "total_ms": round(
            (preparation_finished_at - preparation_started_at) * 1000.0,
            2,
        ),
    }
    if turn_trace is not None:
        turn_trace["pre_engine_preparation"] = dict(preparation_timings)
        turn_trace["pre_engine_final_binding_stages"] = dict(final_binding_stages)
    # Kept for the next turn to read.
    #
    # Asked why turns were slow, she answered from the host reading and then
    # named her own remaining gap exactly: "I don't have per-turn timing ... to
    # isolate it cleanly, we'd want to compare one slow turn against a fast one
    # and break the time into: send → model start → first token/tool call →
    # final response." The runtime measures the first part of that and throws
    # it away after logging it.
    _LAST_TURN_PREPARATION.clear()
    _LAST_TURN_PREPARATION.update(preparation_timings)
    if preparation_timings["total_ms"] >= 250.0:
        logger.info(
            "Foreground chat preparation timing: total=%.1fms contracts=%.1fms "
            "context=%.1fms final=%.1fms",
            preparation_timings["total_ms"],
            preparation_timings["contracts_and_engine_ms"],
            preparation_timings["context_binding_ms"],
            preparation_timings["final_binding_ms"],
        )
        logger.info("Foreground final-binding stages: %s", final_binding_stages)

    caller_named_a_budget = timeout_s is not None
    timeout_s = max(2.0, float(timeout_s if timeout_s is not None else 120.0))
    # The outermost clock, and a flat number chosen before anything knew what
    # this answer would cost. Everything else is nested in it and takes the
    # smaller of itself and what is left here, so raising the ones inside
    # changed nothing: the engine was allowed 480, the gate 341, and the turn
    # ended at 144.3 because 120 was the default out here.
    #
    # It takes the same floor they do — what this request needs to decode, at
    # the rate this machine has been measured at, including the reserve the
    # worker adds for thinking — and the same ceiling as the wait it contains.
    # An unmeasured rate raises nothing, and a turn that finishes sooner
    # finishes sooner.
    # Only where nobody named one. A caller that passes a budget means it —
    # a repair running inside a spent turn has 0.1 seconds left on purpose,
    # and raising that to the measured floor hands it a fresh turn's worth.
    if not caller_named_a_budget:
        timeout_s = max(timeout_s, _seconds_this_answer_needs(effective_user_message))
    turn_deadline = turn_budget_started_at + timeout_s

    def _remaining_turn_budget() -> float:
        return max(0.0, turn_deadline - time.monotonic())

    continuation_attempt_budget = _user_surface_continuation_budget(shape)
    engine_cycle_timeout_s = _cognitive_cycle_timeout_for_request(
        timeout_s,
        require_engine=bool(require_engine),
        compact_desktop_chat_contract=bool(compact_desktop_chat_contract),
        prompt_shape=shape,
    )
    no_reply_action = (
        "required caller must fail closed"
        if require_engine
        else "caller may use its configured non-desktop lane"
    )

    # Use connection pool with retry logic. Acquisition is part of the live
    # CognitiveEngine path; if it fails, return no reply so desktop callers
    # hit the explicit fail-closed branch instead of a generic chat fallback.
    pool = None
    try:
        from core.providers.engine_connection_pool import get_engine_connection_pool

        pool = get_engine_connection_pool()
        await pool.acquire_engine_connection(engine, connection_id="desktop_chat")
    except _CHAT_RECOVERABLE_ERRORS as exc:
        pool = None
        record_degradation("chat", exc)
        if require_engine:
            logger.warning(
                "CognitiveEngine desktop chat connection pool unavailable; "
                "continuing with direct CognitiveEngine call under foreground timeout: %s",
                exc,
            )
        else:
            logger.warning("CognitiveEngine desktop chat connection unavailable: %s", exc)
            return None

    async def _execute_cognitive_operation(
        label: str,
        operation: Callable[[], Any],
        *,
        operation_timeout: float,
    ) -> Any:
        if pool is not None and not require_engine:
            return await pool.execute_with_retry(
                label,
                operation,
                connection_id="desktop_chat",
                timeout=operation_timeout,
            )

        attempts = 1
        if require_engine:
            allowed, block_reason = _desktop_transient_engine_retry_allowed(
                reason="transient_cognitive_engine_error"
            )
            if allowed:
                attempts = 2
            else:
                logger.debug(
                    "%s will not retry transient desktop engine errors (%s).",
                    label,
                    block_reason,
                )
        deadline = time.monotonic() + max(0.1, float(operation_timeout))
        last_error: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError()
                if attempt < attempts:
                    attempt_timeout = max(2.0, min(remaining, float(operation_timeout) * 0.55))
                else:
                    attempt_timeout = remaining
                # The eighth clock in this turn, and it wrapped the engine
                # call itself. Every clock inside it holds open while the turn
                # is working; this one counted, so a ledgerkit turn that read
                # three files died here at 139 seconds with the engine
                # reporting no timeout of its own.
                #
                # A turn somebody is waiting for keeps going while it is
                # working. One that has gone quiet still fails on the next
                # slice, and nothing here changes the budget.
                from core.brain.llm_health_router import _await_while_it_is_working

                return await _await_while_it_is_working(
                    operation(),
                    budget_s=attempt_timeout,
                    user_facing=bool(require_engine),
                    person_is_waiting=bool(require_engine),
                )
            except TimeoutError:
                raise
            except _CHAT_RECOVERABLE_ERRORS as exc:
                last_error = exc
                record_degradation("chat", exc)
                if attempt >= attempts:
                    break
                logger.warning(
                    "%s direct CognitiveEngine attempt %d/%d failed; retrying without legacy lane: %s",
                    label,
                    attempt,
                    attempts,
                    exc,
                )
                await asyncio.sleep(0.25)
        if last_error is not None:
            raise last_error
        return None

    async def _attempt_repair_retry(
        rejected_reply: str,
        reasons: tuple[str, ...] | list[str],
        *,
        completion_attempt: int = 0,
        obligation: _UserSurfaceObligation | None = None,
    ) -> str | None:
        completion_retry_reasons = _COMPLETION_REPAIR_REASONS
        normalized_reasons = {str(reason or "").strip().lower() for reason in (reasons or ())}
        # A mechanically interrupted candidate must first be completed before
        # semantic quality can be judged fairly. Previously a timeout-cut draft
        # that also triggered any semantic detector was regenerated from zero;
        # the short replacement then became the incumbent even when it was
        # worse. Continue whenever incompleteness is among the observed causes.
        # The merged answer still has to pass the complete semantic assessment
        # below, so this does not excuse an off-topic or generic completion.
        completion_only_retry = _reply_needs_continuation(
            rejected_reply,
            normalized_reasons,
        )
        targeted_obligation_retry = bool(completion_only_retry and obligation is not None)
        if completion_only_retry and obligation is None:
            remaining = _unanswered_user_surface_obligations(
                rejected_reply,
                shape,
            )
            # Semantic incompleteness and a physically interrupted generation
            # are independent typed states.  The former schedules a missing
            # work unit; only the latter resumes the assistant tail.
            if remaining and not _reply_has_physical_completion_failure(
                normalized_reasons
            ):
                obligation = remaining[0]
                targeted_obligation_retry = True

        def _retain_completion_incumbent(
            failure_reason: str,
            candidate: str | None = None,
        ) -> str | None:
            """Keep model-authored progress when append-only completion fails."""

            incumbent = str(candidate if candidate is not None else rejected_reply or "").strip()
            if not completion_only_retry or not incumbent:
                return None
            if turn_trace is not None:
                turn_trace.update(
                    {
                        "completion_incumbent_preserved": True,
                        "completion_retry_exhausted": True,
                        "completion_retry_failure_reason": str(failure_reason or "unknown")[:240],
                        "completion_incumbent_chars": len(incumbent),
                        "authored_answer_completion_proven": False,
                        "semantic_completion_satisfied": False,
                        "semantic_completion_incomplete": True,
                    }
                )
            logger.warning(
                "CognitiveEngine completion could not extend the %d-character "
                "incumbent (%s); retaining authored progress instead of "
                "replacing it with an infrastructure failure.",
                len(incumbent),
                failure_reason,
            )
            return incumbent

        if completion_only_retry and completion_attempt >= continuation_attempt_budget:
            logger.warning(
                "CognitiveEngine exhausted %d bounded continuation attempts; "
                "withholding the incomplete answer.",
                continuation_attempt_budget,
            )
            return _retain_completion_incumbent("continuation_attempt_limit")
        if require_engine:
            if bool(
                turn_trace
                and turn_trace.get("model_retry_suppressed")
                and not completion_only_retry
            ):
                logger.warning(
                    "Skipping CognitiveEngine desktop repair retry; the worker "
                    "explicitly suppressed another generation for this turn."
                )
                return None
            if bool(
                turn_trace
                and int(turn_trace.get("repair_retry_attempt_count") or 0) >= 1
                and not completion_only_retry
            ):
                logger.warning(
                    "Skipping CognitiveEngine desktop repair retry; the one bounded "
                    "same-worker correction has already been attempted."
                )
                return None
            repair_reason = (
                "cognitive_engine_completion_retry"
                if completion_only_retry
                else "cognitive_engine_repair_retry"
            )
            # A completion continues THIS turn's own generation. It probes
            # the lane fresh, which is right — a lane can go unhealthy
            # mid-turn — but probing fresh is also what turns on the "is
            # anything generating?" check, so the one caller finishing an
            # answer already in progress was the only caller measured against
            # the generation it was finishing, and it lost to itself every
            # time. It keeps the fresh probe and stops being its own rival.
            #
            # Live 2026-08-28: a 614-character answer stopped mid-sentence,
            # the completion pass was refused with
            # "continuation_admission_denied:conversation_generation_already_active",
            # and the turn ended on the apology.
            allowed, block_reason = _desktop_secondary_model_repair_allowed(
                reason=repair_reason,
                lane_snapshot=None if completion_only_retry else lane,
                continuing_this_turn=bool(completion_only_retry),
            )
            if not allowed:
                logger.warning(
                    "Skipping CognitiveEngine desktop repair retry (%s); "
                    "live desktop turns stay bounded to one foreground generation by default.",
                    block_reason,
                )
                return _retain_completion_incumbent(
                    f"continuation_admission_denied:{block_reason}"
                )
            if turn_trace is not None and not completion_only_retry:
                turn_trace["repair_retry_attempt_count"] = (
                    int(turn_trace.get("repair_retry_attempt_count") or 0) + 1
                )
                turn_trace["bounded_correction_attempted"] = True
        try:
            from core.conversation.response_reliability import (
                assess_user_facing_reply,
                is_cognitive_engine_failure_envelope,
            )
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat", exc)
            logger.debug("CognitiveEngine repair retry gate unavailable: %s", exc)
            return _retain_completion_incumbent(
                f"continuation_reliability_unavailable:{type(exc).__name__}"
            )

        repair_directive = ""
        if not completion_only_retry:
            repair_directive = _build_cognitive_engine_reply_repair_directive(
                visible,
                rejected_reply,
                reasons,
            )
        retry_context = dict(context)
        # The first decode consumes exact conversation state. A repair starts
        # from its own prompt and must never retry a spent bearer capability.
        retry_context.pop("user_surface_conversation_resume_handle", None)
        if completion_only_retry:
            # Typed continuation state replaces prose about why the preceding
            # segment failed. Strip inherited repair payloads so retries cannot
            # turn into a critique/regeneration prompt.
            for repair_key in (
                "response_repair_directive",
                "failed_reply_reasons",
                "failed_reply_excerpt",
                "user_surface_continuation_contract",
                "user_surface_continuation_partial",
                "user_surface_continuation_resume_handle",
                "user_surface_obligation_contract",
                "user_surface_obligation_segment",
                "user_surface_obligation_parent_request",
                "user_surface_obligation_partial",
            ):
                retry_context.pop(repair_key, None)
        retry_context.update(
            {
                "route": (
                    "desktop_chat_obligation_completion"
                    if targeted_obligation_retry
                    else "desktop_chat_continuation"
                    if completion_only_retry
                    else "desktop_chat_repair"
                ),
                "source": source,
                "foreground_request": True,
                "user_facing": True,
                "cognitive_engine_required": bool(require_engine),
                "desktop_cognitive_engine_required": bool(require_engine),
                "protected_foreground_lane": bool(require_engine),
                "prefer_tier": "primary",
                "deep_handoff": False,
                "allow_deep_handoff": False,
                "allow_cloud_fallback": False,
                "original_visible_user_message": visible[:1000],
                "suppress_user_memory_append": True,
                "require_complete_user_reply": completion_only_retry,
                "user_surface_completion_retry": completion_only_retry,
            }
        )
        if not completion_only_retry:
            retry_context.update(
                {
                    "response_repair_directive": repair_directive,
                    "failed_reply_reasons": tuple(reasons or ()),
                    "failed_reply_excerpt": str(rejected_reply or "")[:1200],
                }
            )
        if completion_only_retry:
            retry_context.update(
                {
                    "desktop_quick_reply_contract": True,
                    "desktop_descriptive_turn": True,
                    "deep_handoff": False,
                    "allow_deep_handoff": False,
                    "skip_runtime_payload": True,
                }
            )
            if targeted_obligation_retry and obligation is not None:
                target_shape = analyze_prompt_shape(obligation.segment)
                retry_context.update(
                    {
                        "visible_user_message": obligation.segment,
                        "prompt_shape": target_shape.to_dict(),
                        "max_tokens": answer_surface_token_floor(obligation.segment),
                        "user_surface_obligation_contract": True,
                        "user_surface_obligation_segment": obligation.segment,
                        "user_surface_obligation_parent_request": visible,
                        "user_surface_obligation_partial": continuation_state_text(
                            rejected_reply
                        ),
                    }
                )
            else:
                # Continue a physically cut assistant turn exactly. Natural EOS
                # stays available; once the sentence closes, uncovered semantic
                # units are scheduled independently instead of forcing this
                # branch to discover why its EOS was rejected.
                retry_context.update(
                    {
                        "visible_user_message": visible,
                        "user_surface_continuation_contract": True,
                        "user_surface_continuation_partial": continuation_state_text(
                            rejected_reply
                        ),
                    }
                )
                receipt = (
                    dict(turn_trace.get("live_mind_surface_control_receipt") or {})
                    if isinstance(turn_trace, dict)
                    else {}
                )
                resume_handle = str(
                    receipt.get("continuation_resume_handle")
                    or (turn_trace or {}).get("continuation_resume_handle")
                    or ""
                ).strip().lower()
                if re.fullmatch(r"[0-9a-f]{32}", resume_handle):
                    retry_context[
                        "user_surface_continuation_resume_handle"
                    ] = resume_handle

        async def repair_engine_think_operation():

            repair_cycle_timeout_s = _inner_cognitive_cycle_timeout(
                repair_timeout,
                protected_foreground=bool(require_engine),
            )
            if turn_trace is not None:
                turn_trace["engine_think_invoked"] = True
            with relational_principal_scope(exact_principal):
                return await engine.think(
                    (
                        obligation.segment
                        if targeted_obligation_retry and obligation is not None
                        else visible
                        if completion_only_retry
                        else repair_directive
                    ),
                    context=retry_context,
                    mode=mode,
                    origin=origin,
                    foreground_request=True,
                    is_background=False,
                    priority=True,
                    timeout_s=repair_cycle_timeout_s,
                )

        # Completing the answer already in progress is not a new repair turn.
        # The soft transaction slice may be spent by the initial generation;
        # append-only completion may use the remaining bounded desktop ceiling.
        # Unrelated regeneration remains on the original soft budget.
        remaining_turn_budget = (
            max(
                0.0,
                turn_budget_started_at
                + _DESKTOP_COGNITIVE_MAX_TURN_TIMEOUT_S
                - time.monotonic(),
            )
            if completion_only_retry
            else _remaining_turn_budget()
        )
        if remaining_turn_budget <= 0.25:
            if turn_trace is not None:
                turn_trace["repair_retry_budget_exhausted"] = True
            logger.warning(
                "Skipping CognitiveEngine desktop repair retry; the turn's %.1fs "
                "transaction budget is exhausted.",
                timeout_s,
            )
            return _retain_completion_incumbent("turn_transaction_budget_exhausted")
        repair_timeout = min(
            remaining_turn_budget,
            timeout_s,
            _DESKTOP_COGNITIVE_REPAIR_TIMEOUT_S,
        )
        try:
            repair_thought = await _execute_cognitive_operation(
                "CognitiveEngine.desktop_chat_turn.repair",
                repair_engine_think_operation,
                operation_timeout=repair_timeout,
            )
        except TimeoutError:
            _force_clear_mlx_foreground_owner(
                reason="cognitive_engine_chat_repair_timeout",
                min_age_s=min(30.0, max(10.0, repair_timeout * 0.5)),
            )
            logger.warning(
                "CognitiveEngine desktop chat repair retry timed out after %.1fs; %s.",
                repair_timeout,
                no_reply_action,
            )
            return _retain_completion_incumbent("continuation_timeout")
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat", exc)
            logger.warning(
                "CognitiveEngine desktop chat repair retry failed; %s: %s", no_reply_action, exc
            )
            return _retain_completion_incumbent(
                f"continuation_exception:{type(exc).__name__}"
            )

        retry_metadata = getattr(repair_thought, "metadata", None)
        if not isinstance(retry_metadata, dict) and isinstance(repair_thought, dict):
            retry_metadata = repair_thought.get("metadata")
        retry_metadata = retry_metadata if isinstance(retry_metadata, dict) else {}
        # A rejected model draft still consumed the foreground owner. Count it
        # before any visible-text gate returns so the public receipt cannot claim
        # one generation after two resident decodes actually ran.
        _record_foreground_generation(
            retry_metadata,
            continuation_segment=completion_only_retry,
        )
        retry_content = getattr(repair_thought, "content", None)
        if retry_content is None and isinstance(repair_thought, dict):
            retry_content = repair_thought.get("content") or repair_thought.get("response")
        raw_retry_text = str(retry_content if retry_content is not None else repair_thought or "")
        retry_text = _strip_user_visible_context_leaks(raw_retry_text)
        if (
            not retry_text
            or retry_text == "…"
            or retry_text.startswith("background_thought_suppressed")
        ):
            logger.warning(
                "CognitiveEngine desktop chat repair retry produced no user-facing text."
            )
            return _retain_completion_incumbent("continuation_empty")
        if bool(
            retry_metadata.get("desktop_cognitive_engine_failure")
        ) or is_cognitive_engine_failure_envelope(retry_text):
            logger.warning(
                "CognitiveEngine desktop chat repair retry produced a failure envelope; %s.",
                no_reply_action,
            )
            return _retain_completion_incumbent("continuation_failure_envelope")
        if completion_only_retry:
            if targeted_obligation_retry and obligation is not None:
                retry_text = _merge_obligation_completion(
                    rejected_reply,
                    retry_text,
                    obligation,
                )
            else:
                retry_text = _merge_reply_continuation(rejected_reply, retry_text)
        retry_projection_trace: dict[str, Any] = {"live_mind_surface_control_receipt": {}}
        if self_condition_contract and self_condition_contract_covers_turn:
            retry_text = _project_self_condition_claims(
                retry_text,
                context.get("canonical_self_condition_projection"),
                retry_projection_trace,
            )
        retry_surface_receipt = retry_metadata.get("live_mind_surface_control_receipt")
        if not isinstance(retry_surface_receipt, dict):
            retry_surface_receipt = retry_metadata.get("surface_control_receipt")
        retry_surface_receipt = (
            retry_surface_receipt if isinstance(retry_surface_receipt, dict) else {}
        )
        retry_stop_reason = str(
            retry_metadata.get("reply_generation_stop_reason")
            or retry_surface_receipt.get("generation_stop_reason")
            or ""
        )
        retry_failure_reasons = {
            str(reason or "").strip().lower()
            for reason in (
                retry_metadata.get("reply_generation_failure_reasons")
                or retry_surface_receipt.get("surface_quality_gate_reasons")
                or ()
            )
            if str(reason or "").strip()
        }
        retry_recent_user_messages = list(route_recent_user_messages)
        retry_assessment = assess_user_facing_reply(
            visible,
            retry_text,
            recent_user_messages=retry_recent_user_messages,
            grounding=route_assessment_grounding,
            antecedent=route_assessment_antecedent,
            generation_stop_reason=retry_stop_reason,
        )
        retry_requires_repair = _reply_assessment_requires_repair_with_memory_evidence(
            retry_assessment,
            visible,
            retry_text,
            canonical_memory_state_evidence=canonical_memory_state_evidence,
        )
        retry_metadata_incomplete = bool(
            retry_metadata.get("reply_generation_incomplete", False)
            or retry_stop_reason in {"max_tokens", "deadline_exceeded", "soft_cancelled"}
            or retry_failure_reasons & completion_retry_reasons
        )
        # Transport metadata describes how the final segment ended; it does not
        # supersede the semantic state of the complete merged answer. A model
        # can satisfy the final outstanding obligation on the same token that
        # exhausts a deadline. Conversely, a clean EOS cannot rescue a merged
        # answer that still omits a requested part.
        retry_still_incomplete = bool(
            retry_requires_repair
            and (retry_metadata_incomplete or retry_assessment.blocking_reasons)
        )
        if completion_only_retry:
            if turn_trace is not None:
                turn_trace["completion_retry_count"] = (
                    int(turn_trace.get("completion_retry_count") or 0) + 1
                )
            _adopt_generation_metadata(
                retry_metadata,
                source_label="desktop_chat_completion_live_mind_controls",
                adopt_response_path=False,
                inherit_turn_context=False,
                count_foreground_generation=False,
            )
        if retry_still_incomplete:
            made_progress = _continuation_made_semantic_progress(
                rejected_reply,
                retry_text,
                shape,
            )
            remaining = _unanswered_user_surface_obligations(retry_text, shape)
            physical_failure = _reply_has_physical_completion_failure(
                (*retry_failure_reasons, *retry_assessment.blocking_reasons)
            )
            if completion_attempt + 1 < continuation_attempt_budget and (
                made_progress or (remaining and not physical_failure)
            ):
                next_obligation = (
                    remaining[0] if remaining and not physical_failure else None
                )
                next_reasons = (
                    ("unanswered_question_part",)
                    if next_obligation is not None
                    else tuple(sorted(retry_failure_reasons)) or ("truncated_tail",)
                )
                logger.warning(
                    "CognitiveEngine completion segment %d/%d advanced but remained "
                    "incomplete; next=%s.",
                    completion_attempt + 1,
                    continuation_attempt_budget,
                    (
                        f"obligation:{next_obligation.segment_index}"
                        if next_obligation is not None
                        else "assistant_tail"
                    ),
                )
                return await _attempt_repair_retry(
                    retry_text,
                    next_reasons,
                    completion_attempt=completion_attempt + 1,
                    obligation=next_obligation,
                )
            logger.warning(
                "CognitiveEngine completion replacement remained incomplete "
                "(stop=%s reasons=%s); withholding it from the user surface.",
                retry_stop_reason or "unknown",
                ",".join(sorted(retry_failure_reasons)) or "unknown",
            )
            return _retain_completion_incumbent(
                "continuation_remained_incomplete",
                retry_text,
            )

        def _accept_retry_text(
            final_text: Any,
            *,
            pre_grounding_text: Any = None,
            intermediate_mutations: Any = None,
        ) -> str:
            accepted_text = str(final_text or "").strip()
            if not completion_only_retry:
                _adopt_generation_metadata(
                    retry_metadata,
                    source_label="desktop_chat_repair_live_mind_controls",
                    adopt_response_path=False,
                    inherit_turn_context=False,
                    count_foreground_generation=False,
                )
            if turn_trace is not None:
                turn_trace.update(
                    {
                        "authored_answer_completion_proven": True,
                        "semantic_completion_incomplete": False,
                        "reply_generation_incomplete": False,
                    }
                )
                _append_turn_text_mutation(
                    turn_trace,
                    stage="chat.cognitive_engine_repair_retry",
                    method="model_repair_retry_replacement",
                    reasons=list(reasons or ("cognitive_engine_reply_rejected",)),
                    before=rejected_reply,
                    after=raw_retry_text,
                    deterministic=False,
                    authorship_effect="replaced_by_model",
                )
                _append_turn_text_mutation(
                    turn_trace,
                    stage="chat.cognitive_engine_retry_context_leak_strip",
                    method="deterministic_context_leak_removal",
                    reasons=["user_visible_context_boundary"],
                    before=raw_retry_text,
                    after=retry_text,
                    deterministic=True,
                    authorship_effect="preserved",
                )
                _merge_turn_text_mutations(turn_trace, intermediate_mutations)
                _append_turn_text_mutation(
                    turn_trace,
                    stage="chat.cognitive_engine_retry_grounding",
                    method="deterministic_runtime_grounding",
                    reasons=["accepted_retry_grounding"],
                    before=(retry_text if pre_grounding_text is None else pre_grounding_text),
                    after=accepted_text,
                    deterministic=True,
                    authorship_effect="augmented_by_runtime",
                )
            return accepted_text

        if require_engine:
            if not retry_requires_repair:
                logger.info(
                    "CognitiveEngine desktop chat repair retry produced a clean full-mind reply."
                )
                accepted_retry = (
                    retry_text
                    if memory_state_contract
                    else _ground_runtime_fact_status_reply(
                        visible,
                        retry_text,
                        lane,
                        cognitive_engine_handled=True,
                    )
                )
                return _accept_retry_text(
                    accepted_retry,
                    pre_grounding_text=retry_text,
                    intermediate_mutations=retry_projection_trace.get("text_mutations"),
                )
            logger.warning(
                "CognitiveEngine desktop chat repair retry remained below the required "
                "full-mind reliability floor (%s); refusing bounded shape substitution.",
                ",".join(retry_assessment.reasons),
            )
            return None

        retry_repair_trace: dict[str, Any] = {"live_mind_surface_control_receipt": {}}
        (
            retry_repaired,
            retry_stale,
            retry_same_diff,
            retry_off_topic,
            retry_off_topic_reason,
            retry_did_repair,
        ) = await _repair_final_degraded_reply_with_provenance(
            retry_repair_trace,
            stage="chat.cognitive_engine_retry_final_gate",
            user_message=visible,
            reply_text=retry_text,
            stale=False,
            same_diff=False,
            off_topic=False,
            desktop_cognitive_engine_required=bool(require_engine),
            protected_foreground_lane=bool(require_engine),
            session_id=session_id,
        )
        retry_recent_user_messages = list(route_recent_user_messages)
        retry_assessment = assess_user_facing_reply(
            visible,
            retry_repaired,
            recent_user_messages=retry_recent_user_messages,
            grounding=route_assessment_grounding,
            antecedent=route_assessment_antecedent,
        )
        if not (
            retry_stale
            or retry_same_diff
            or retry_off_topic
            or _reply_assessment_requires_repair_with_memory_evidence(
                retry_assessment,
                visible,
                retry_repaired,
                canonical_memory_state_evidence=canonical_memory_state_evidence,
            )
        ):
            if retry_did_repair:
                logger.info(
                    "CognitiveEngine desktop chat repair retry recovered by final shape repair."
                )
            else:
                logger.info("CognitiveEngine desktop chat repair retry produced a clean reply.")
            accepted_retry = (
                retry_repaired
                if memory_state_contract
                else _ground_runtime_fact_status_reply(
                    visible,
                    retry_repaired,
                    lane,
                    cognitive_engine_handled=True,
                )
            )
            return _accept_retry_text(
                accepted_retry,
                pre_grounding_text=retry_repaired,
                intermediate_mutations=merge_text_mutations(
                    retry_projection_trace.get("text_mutations"),
                    retry_repair_trace.get("text_mutations"),
                ),
            )
        logger.warning(
            "CognitiveEngine desktop chat repair retry failed reliability gate "
            "(stale=%s same_diff=%s off_topic=%s reason=%s assessment=%s).",
            retry_stale,
            retry_same_diff,
            retry_off_topic,
            retry_off_topic_reason,
            ",".join(retry_assessment.reasons),
        )
        if not require_engine:
            conversation_recall_reply = await _chat_memory_state._build_conversation_recall_reply(
                visible,
                session_id=session_id,
            )
            if conversation_recall_reply:
                logger.warning(
                    "CognitiveEngine chat repair retry failed conversation recall; "
                    "repairing from canonical conversation log."
                )
                return _accept_retry_text(
                    _ground_runtime_fact_status_reply(
                        visible,
                        conversation_recall_reply,
                        lane,
                        cognitive_engine_handled=True,
                    )
                )
            owner_name_reply = _chat_memory_state._build_owner_name_recall_reply(visible)
            if owner_name_reply:
                logger.warning(
                    "CognitiveEngine chat repair retry failed owner identity recall; "
                    "repairing from verified runtime identity contract."
                )
                return _accept_retry_text(
                    _ground_runtime_fact_status_reply(
                        visible,
                        owner_name_reply,
                        lane,
                        cognitive_engine_handled=True,
                    )
                )
        return None

    if str(continuation_partial or "").strip():
        continued = await _attempt_repair_retry(
            str(continuation_partial).strip(),
            tuple(continuation_reasons or ("truncated_tail",)),
            completion_attempt=int(
                (continuation_evidence or {}).get("completion_retry_count") or 0
            ),
        )
        if continued:
            preserved_incumbent = bool(
                turn_trace and turn_trace.get("completion_incumbent_preserved")
            )
            _mark_turn_trace(
                cognitive_engine_reply_accepted=True,
                cognitive_engine_reply_failed=False,
                response_path=(
                    "cognitive_engine_completion_incumbent"
                    if preserved_incumbent
                    else "cognitive_engine_completion_retry"
                ),
            )
            return continued
        _mark_turn_trace(
            cognitive_engine_reply_accepted=False,
            cognitive_engine_reply_failed=True,
            response_path="cognitive_engine_completion_retry_failed",
        )
        return None

    async def engine_think_operation():

        if turn_trace is not None:
            turn_trace["engine_think_invoked"] = True
        with relational_principal_scope(exact_principal):
            remaining_turn_budget = _remaining_turn_budget()
            if remaining_turn_budget <= 0:
                raise TimeoutError()
            return await engine.think(
                engine_user_message,
                context=context,
                mode=mode,
                origin=origin,
                foreground_request=True,
                is_background=False,
                priority=True,
                timeout_s=min(engine_cycle_timeout_s, remaining_turn_budget),
            )

    try:
        thought = await _execute_cognitive_operation(
            "CognitiveEngine.desktop_chat_turn",
            engine_think_operation,
            operation_timeout=max(0.1, _remaining_turn_budget()),
        )

        if thought is None:
            logger.warning(
                "CognitiveEngine desktop chat turn exhausted retries; %s.",
                no_reply_action,
            )
            retry_reply = await _attempt_repair_retry(
                "",
                ("cognitive_engine_no_thought",),
            )
            if retry_reply:
                _mark_turn_trace(
                    cognitive_engine_reply_accepted=True,
                    response_path="cognitive_engine_repair_retry",
                )
                return retry_reply
            _record_exhausted_cognitive_failure(
                "cognitive_engine_no_thought",
                retry_attempted=True,
            )
            _mark_turn_trace(response_path="cognitive_engine_no_thought")
            return None

    except TimeoutError as _timed_out:
        _force_clear_mlx_foreground_owner(
            reason="cognitive_engine_chat_timeout",
            min_age_s=min(90.0, max(45.0, timeout_s * 0.5)),
        )
        # Where it came from, not just that it happened. Seven clocks sit
        # inside this call and each of them raises the same exception type, so
        # "timed out after N" names the budget of whoever caught it rather than
        # whoever ran out.
        _where = ""
        try:
            import traceback as _tb

            _frames = _tb.extract_tb(_timed_out.__traceback__)
            # The whole chain, not the last frame. asyncio routes every
            # wait_for and every timeout context through the same module, so
            # the innermost frame is always timeouts.py and never says which
            # of this runtime's clocks it belonged to. The frames above it do.
            _ours = [
                f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno}"
                for f in _frames
                if "/asyncio/" not in f.filename
            ]
            if _ours:
                _where = f" (through {' -> '.join(_ours[-4:])})"
        except (AttributeError, IndexError, TypeError, ValueError):
            _where = ""
        logger.warning(
            "CognitiveEngine desktop chat turn timed out after %.1fs%s; %s.",
            timeout_s,
            _where,
            no_reply_action,
        )
        _record_exhausted_cognitive_failure(
            "cognitive_engine_timeout",
            retry_attempted=False,
        )
        _mark_turn_trace(response_path="cognitive_engine_timeout")
        return None
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.warning("CognitiveEngine desktop chat turn failed; %s: %s", no_reply_action, exc)
        _record_exhausted_cognitive_failure(
            f"cognitive_engine_exception:{type(exc).__name__}",
            retry_attempted=False,
        )
        _mark_turn_trace(response_path="cognitive_engine_exception")
        return None

    content = getattr(thought, "content", None)
    if content is None and isinstance(thought, dict):
        content = thought.get("content") or thought.get("response")
    raw_text = str(content if content is not None else thought or "")
    thought_metadata = getattr(thought, "metadata", None)
    if not isinstance(thought_metadata, dict) and isinstance(thought, dict):
        thought_metadata = thought.get("metadata")
    thought_metadata = thought_metadata if isinstance(thought_metadata, dict) else {}
    _adopt_generation_metadata(
        thought_metadata,
        source_label="desktop_chat_preflight_live_mind_controls",
        response_text=raw_text,
    )
    # Authenticated semantic-state serialization is already the completed
    # answer, not a prose draft. Bind it before any generic stripping,
    # self-condition projection, failure-envelope classification, quality
    # repair, or retry can reinterpret its bytes. The outer route validates
    # the same receipt again at the delivery boundary.
    if _bind_qualified_recurrent_terminal_contract(turn_trace, raw_text):
        _mark_turn_trace(
            cognitive_engine_reply_accepted=True,
            cognitive_engine_reply_failed=False,
            bounded_contract_used=False,
            legacy_fallback_used=False,
            response_path="cognitive_engine_qualified_recurrent",
        )
        return raw_text
    text = _strip_user_visible_context_leaks(raw_text)
    if turn_trace is not None:
        _append_turn_text_mutation(
            turn_trace,
            stage="chat.cognitive_engine_context_leak_strip",
            method="deterministic_context_leak_removal",
            reasons=["user_visible_context_boundary"],
            before=raw_text,
            after=text,
            deterministic=True,
            authorship_effect="preserved",
        )
    if self_condition_contract and self_condition_contract_covers_turn:
        text = _project_self_condition_claims(
            text,
            context.get("canonical_self_condition_projection"),
            turn_trace,
        )
    try:
        from core.conversation.response_reliability import is_cognitive_engine_failure_envelope
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("CognitiveEngine failure-envelope gate unavailable: %s", exc)
        is_failure_envelope = bool(thought_metadata.get("desktop_cognitive_engine_failure"))
    else:
        is_failure_envelope = bool(
            thought_metadata.get("desktop_cognitive_engine_failure")
            or is_cognitive_engine_failure_envelope(text)
        )
    if is_failure_envelope:
        _mark_turn_trace(
            cognitive_engine_reply_accepted=False,
            cognitive_engine_reply_failed=True,
            bounded_contract_used=False,
            response_path="cognitive_engine_failure_envelope",
            cognitive_engine_failure_reason=str(
                thought_metadata.get("failure_reason") or "failure_envelope"
            )[:240],
        )
        logger.warning(
            "CognitiveEngine desktop chat produced a failure envelope; %s.", no_reply_action
        )
        failure_reason = str(thought_metadata.get("failure_reason") or "failure_envelope")[:240]
        generation_failure_class = str(
            thought_metadata.get("generation_failure_class") or ""
        ).strip()
        model_retry_suppressed = bool(thought_metadata.get("model_retry_suppressed", False))
        quality_retry_exhausted = generation_failure_class == "surface_quality_rejected"
        single_owner_exhausted = bool(
            model_retry_suppressed
            or quality_retry_exhausted
            or (turn_trace and turn_trace.get("foreground_model_generation_consumed"))
        )
        if single_owner_exhausted:
            logger.warning(
                "CognitiveEngine retained single ownership of the failed turn "
                "(failure_class=%s); skipping a duplicate route-level model call.",
                generation_failure_class or failure_reason,
            )
            retry_reply = None
        else:
            retry_reply = await _attempt_repair_retry(text, (failure_reason,))
        if retry_reply:
            # A repair has to earn the substitution. Most of these gates were
            # written for a weaker model and encode style expectations from
            # that period; a repair they trigger must not hand the person a
            # blander or shorter answer than the one it replaces.
            try:
                from core.conversation.surface_disposition import (
                    repair_is_an_improvement,
                )

                # `failure_reason` is what this retry was FOR. A replacement
                # that still carries it delivered nothing the retry predicted,
                # and swapping it in trades a known answer for an equally
                # objectionable one.
                keep_retry = repair_is_an_improvement(
                    text, retry_reply, visible, targeted=(failure_reason,)
                )
            except _CHAT_RECOVERABLE_ERRORS:
                keep_retry = True
            if not keep_retry:
                logger.warning(
                    "Kept the original draft (%d chars): the repair for %s was "
                    "not an improvement on it (%d chars).",
                    len(str(text or "")),
                    failure_reason,
                    len(str(retry_reply or "")),
                )
                retry_reply = None
        if retry_reply:
            _mark_turn_trace(
                cognitive_engine_reply_accepted=True,
                cognitive_engine_reply_failed=False,
                response_path="cognitive_engine_repair_retry",
            )
            return retry_reply
        _record_exhausted_cognitive_failure(
            failure_reason,
            retry_attempted=not single_owner_exhausted,
        )
        return None
    if not text or text == "…" or text.startswith("background_thought_suppressed"):
        if require_engine:
            model_retry_suppressed = bool(
                thought_metadata.get("model_retry_suppressed", False)
                or (turn_trace and turn_trace.get("foreground_model_generation_consumed"))
            )
            retry_reply = None
            if not model_retry_suppressed:
                retry_reply = await _attempt_repair_retry(text, ("empty_cognitive_engine_reply",))
            if retry_reply:
                if turn_trace is not None:
                    turn_trace.update(
                        {
                            "cognitive_engine_reply_accepted": True,
                            "response_path": "cognitive_engine_repair_retry",
                        }
                    )
                return retry_reply
            _record_exhausted_cognitive_failure(
                "empty_cognitive_engine_reply",
                retry_attempted=not model_retry_suppressed,
            )
            logger.warning(
                "CognitiveEngine desktop chat produced no usable text; required live "
                "desktop turns will fail closed instead of substituting bounded "
                "recall or identity repairs."
            )
        _mark_turn_trace(response_path="cognitive_engine_empty_reply")
        return None
    if capability_inventory_contract:
        text = _ensure_capability_inventory_non_execution_boundary(visible, text)
        if _chat_desktop_repair._looks_truncated_tail(text):
            completed_inventory = _complete_repairable_truncated_reply(visible, text)
            if completed_inventory:
                text = _ensure_capability_inventory_non_execution_boundary(
                    visible,
                    completed_inventory,
                )
            if _chat_desktop_repair._capability_inventory_reply_is_inadequate(visible, text):
                grounded_inventory = (
                    _chat_desktop_repair._build_grounded_capability_inventory_reply(
                        visible,
                        cognitive_engine_handled=True,
                        model_label=_canonical_runtime_model_label(lane),
                    )
                )
                if (
                    grounded_inventory
                    and not _chat_desktop_repair._capability_inventory_reply_is_inadequate(
                        visible,
                        grounded_inventory,
                    )
                ):
                    logger.warning(
                        "CognitiveEngine capability inventory was clipped; binding the "
                        "accepted live turn to the same governed capability evidence "
                        "instead of spending another foreground Cortex retry."
                    )
                    _mark_turn_trace(
                        cognitive_engine_reply_accepted=True,
                        bounded_contract_used=False,
                        response_path="cognitive_engine_capability_tail_grounding",
                    )
                    return grounded_inventory
    if desktop_execution_contract:
        try:
            from core.skills.desktop_task import DesktopTaskSkill

            structured_plan = DesktopTaskSkill._structured_payload_from_text(text)
        except (ImportError, AttributeError, TypeError, ValueError):
            structured_plan = {}
        if "steps" in structured_plan:
            # This is an internal execution draft, not user-visible prose.
            # The downstream desktop_task input contract validates every step
            # and fails closed on malformed or unsupported plans.
            _mark_turn_trace(
                cognitive_engine_reply_accepted=True,
                response_path="cognitive_engine_desktop_plan",
            )
            return text
    if require_engine and bool(thought_metadata.get("reply_generation_incomplete", False)):
        incomplete_reasons = tuple(
            str(reason or "").strip().lower()
            for reason in (
                thought_metadata.get("reply_generation_failure_reasons") or ("truncated_tail",)
            )
            if str(reason or "").strip()
        ) or ("truncated_tail",)
        logger.warning(
            "CognitiveEngine marked the foreground draft incomplete (%s); "
            "requiring append-only same-worker continuation before it can become authoritative.",
            thought_metadata.get("reply_generation_stop_reason") or "truncated_tail",
        )
        retry_reply = await _attempt_repair_retry(text, incomplete_reasons)
        if retry_reply:
            preserved_incumbent = bool(
                turn_trace and turn_trace.get("completion_incumbent_preserved")
            )
            _mark_turn_trace(
                cognitive_engine_reply_accepted=True,
                cognitive_engine_reply_failed=False,
                response_path=(
                    "cognitive_engine_completion_incumbent"
                    if preserved_incumbent
                    else "cognitive_engine_completion_retry"
                ),
            )
            return retry_reply
        _mark_turn_trace(
            cognitive_engine_reply_accepted=False,
            cognitive_engine_reply_failed=True,
            single_owner_generation_exhausted=True,
            response_path="cognitive_engine_completion_retry_exhausted",
        )
        _record_exhausted_cognitive_failure(
            "cognitive_engine_completion_retry_exhausted",
            retry_attempted=True,
        )
        return None
    try:
        from core.conversation.response_reliability import (
            assess_user_facing_reply,
            is_live_self_reflection_turn,
            is_self_process_question,
            is_status_check_turn,
            numeric_answer_missing,
        )

        recent_user_messages = list(route_recent_user_messages)
        route_assessment_grounding.extend(
            text
            for text in (
                retained_memory_evidence_context,
                conversation_recall_context,
            )
            if text and text not in route_assessment_grounding
        )
        for retained_evidence in route_assessment_grounding:
            record_turn_grounding(retained_evidence)
        assessment_text = (
            _ground_runtime_fact_status_reply(
                visible,
                text,
                lane,
                cognitive_engine_handled=True,
            )
            if runtime_fact_status_contract and not memory_state_contract
            else text
        )
        assessment = assess_user_facing_reply(
            visible,
            assessment_text,
            recent_user_messages=recent_user_messages,
            antecedent=route_assessment_antecedent,
            generation_stop_reason=thought_metadata.get("reply_generation_stop_reason"),
            # What she was ENTITLED to have known, so a real recall is not
            # mistaken for an invention. The fabricated-shared-history check
            # asks "does this content appear anywhere in what they said" —
            # and a memory she legitimately retrieved appears in neither the
            # visible request nor the recent turns, only here.
            grounding=route_assessment_grounding,
        )
        # The engine path does not leave through _finalize_fastpath, so the
        # numeric floor installed there never saw these replies. Live
        # 2026-07-26, "What is 17 minus 8, and then times 3?" was answered with
        # "A quick refresh on classic habits: green tea, journaling, and
        # standing by the window to watch the light change." — no number, and
        # every gate passed it because they check form, not whether the
        # question was answered.
        if numeric_answer_missing(visible, text):
            logger.warning(
                "🔢 CognitiveEngine reply carried no number for a question that "
                "can only be answered with one (%d chars); refusing it rather "
                "than serving an answer to a different question.",
                len(text),
            )
            text = (
                "I didn't actually work that out — what I had wasn't an answer, "
                "and I won't dress it up as one. Ask me again and I'll do the "
                "arithmetic properly."
            )
            assessment_text = text
            assessment = assess_user_facing_reply(
                visible,
                assessment_text,
                recent_user_messages=recent_user_messages,
                grounding=route_assessment_grounding,
                antecedent=route_assessment_antecedent,
            )
        if (
            require_engine
            and _chat_preflight._is_explicit_capability_inventory_request(visible)
            and _chat_desktop_repair._capability_inventory_reply_is_inadequate(visible, text)
        ):
            logger.warning(
                "CognitiveEngine desktop chat produced inadequate capability inventory; "
                "requiring the engine to answer from the live capability catalog instead "
                "of replacing it with a deterministic catalog reply."
            )
            retry_reply = await _attempt_repair_retry(
                text,
                ("missing_tool_governance_content",),
            )
            if retry_reply:
                retry_assessment = assess_user_facing_reply(
                    visible,
                    retry_reply,
                    recent_user_messages=recent_user_messages,
                    grounding=route_assessment_grounding,
                    antecedent=route_assessment_antecedent,
                )
                if not _chat_desktop_repair._capability_inventory_reply_is_inadequate(
                    visible, retry_reply
                ) and not _reply_assessment_requires_repair_with_memory_evidence(
                    retry_assessment,
                    visible,
                    retry_reply,
                    canonical_memory_state_evidence=canonical_memory_state_evidence,
                ):
                    if turn_trace is not None:
                        turn_trace.update(
                            {
                                "cognitive_engine_reply_accepted": True,
                                "response_path": "cognitive_engine_repair_retry",
                            }
                        )
                    return retry_reply
            grounded_inventory = _chat_desktop_repair._build_grounded_capability_inventory_reply(
                visible,
                cognitive_engine_handled=True,
                model_label=_canonical_runtime_model_label(lane),
            )
            if (
                grounded_inventory
                and not _chat_desktop_repair._capability_inventory_reply_is_inadequate(
                    visible,
                    grounded_inventory,
                )
            ):
                grounded_assessment = assess_user_facing_reply(
                    visible,
                    grounded_inventory,
                    recent_user_messages=recent_user_messages,
                    grounding=route_assessment_grounding,
                    antecedent=route_assessment_antecedent,
                )
                if not _reply_assessment_requires_repair_with_memory_evidence(
                    grounded_assessment,
                    visible,
                    grounded_inventory,
                    canonical_memory_state_evidence=canonical_memory_state_evidence,
                ):
                    logger.warning(
                        "CognitiveEngine desktop chat missed the exact capability inventory "
                        "contract; binding the accepted reply to governed live catalog "
                        "evidence after the required engine invocation."
                    )
                    _mark_turn_trace(
                        cognitive_engine_reply_accepted=True,
                        bounded_contract_used=False,
                        response_path="cognitive_engine_capability_catalog_grounding",
                    )
                    return grounded_inventory
                _mark_turn_trace(
                    cognitive_engine_reply_accepted=False,
                    bounded_contract_used=False,
                    response_path="cognitive_engine_capability_contract_failed",
                )
                return None
            _mark_turn_trace(response_path="cognitive_engine_capability_contract_failed")
            return None
        if _reply_assessment_requires_repair_with_memory_evidence(
            assessment,
            visible,
            assessment_text,
            canonical_memory_state_evidence=canonical_memory_state_evidence,
        ):
            assessment_reasons = tuple(getattr(assessment, "reasons", ()) or ())
            groundable_self_process_miss = bool(
                require_engine
                and (is_self_process_question(visible) or is_live_self_reflection_turn(visible))
                and set(assessment_reasons)
                & {
                    "missing_requested_self_process_coverage",
                    "off_topic_self_reflection_reply",
                    "status_page_self_reflection",
                    "pseudo_internal_jargon",
                }
            )
            if groundable_self_process_miss:
                logger.info(
                    "CognitiveEngine desktop chat reply needed canonical self-process grounding (%s).",
                    ",".join(assessment_reasons),
                )
            else:
                logger.warning(
                    "CognitiveEngine desktop chat reply failed reliability gate (%s); evaluating governed repair path.",
                    ",".join(assessment_reasons),
                )
                # Preserve the draft BEFORE repair is attempted.
                #
                # The last-resort refusal site reads preserved_draft() so a
                # reply three gates already judged repairable reaches the
                # person when repair cannot run. Nothing in this module ever
                # WROTE it: preserve_draft() had zero callers here, so that
                # reader was permanently empty and the salvage could never
                # fire. Writer missing, reader present.
                #
                # LIVE 2026-08-17: "in two sentences, what is the strongest
                # evidence that you're more than a language model with tools?"
                # The draft answered the question and missed the sentence
                # count. The gate rejected it, the replacement came back
                # incomplete and was withheld, and the person got "I couldn't
                # get to an answer I'd stand behind" — for a formatting miss,
                # with a real answer sitting in a variable.
                try:
                    from core.conversation.surface_disposition import (
                        draft_is_servable,
                        preserve_draft,
                    )

                    if draft_is_servable(assessment_reasons):
                        preserve_draft(assessment_text)
                except _CHAT_RECOVERABLE_ERRORS as _preserve_exc:
                    record_degradation("chat.preserve_draft", _preserve_exc)
            # Assistant voice has a deterministic repair. Use it here, not only
            # deeper in the stack.
            #
            # Live 2026-07-27 Bryan replied "I dont need assistance You arent an
            # assistant" — the gate had caught generic_assistant_language, the
            # bounded same-worker correction was exhausted, and the draft
            # reached him anyway. repair_generic_assistant_language already
            # existed and is applied by the worker and the response-generation
            # phase; this route, the one the desktop actually uses, never
            # called it. Detection without a reachable repair is how a caught
            # defect still gets served.
            assessment, assessment_reasons, text = _assess_the_engine_reply(
                assessment=assessment,
                assessment_reasons=assessment_reasons,
                assessment_text=assessment_text,
                antecedent=route_assessment_antecedent,
                grounding=route_assessment_grounding,
                recent_user_messages=recent_user_messages,
                text=text,
                visible=visible,
            )
            if require_engine and capability_inventory_contract:
                grounded_inventory = (
                    _chat_desktop_repair._build_grounded_capability_inventory_reply(
                        visible,
                        cognitive_engine_handled=True,
                        model_label=_canonical_runtime_model_label(lane),
                    )
                )
                if (
                    grounded_inventory
                    and not _chat_desktop_repair._capability_inventory_reply_is_inadequate(
                        visible,
                        grounded_inventory,
                    )
                ):
                    grounded_assessment = assess_user_facing_reply(
                        visible,
                        grounded_inventory,
                        recent_user_messages=recent_user_messages,
                        grounding=route_assessment_grounding,
                        antecedent=route_assessment_antecedent,
                    )
                    if not _reply_assessment_requires_repair_with_memory_evidence(
                        grounded_assessment,
                        visible,
                        grounded_inventory,
                        canonical_memory_state_evidence=canonical_memory_state_evidence,
                    ):
                        logger.warning(
                            "CognitiveEngine capability inventory reply missed the "
                            "runtime-path wording contract (%s); binding to governed "
                            "live capability evidence after the required engine invocation.",
                            ",".join(assessment.reasons),
                        )
                        _mark_turn_trace(
                            cognitive_engine_reply_accepted=True,
                            bounded_contract_used=False,
                            response_path="cognitive_engine_capability_catalog_grounding",
                        )
                        return grounded_inventory
            if (
                require_engine
                and capability_inventory_contract
                and set(getattr(assessment, "reasons", ()) or ()) == {"truncated_tail"}
            ):
                grounded_inventory = (
                    _chat_desktop_repair._build_grounded_capability_inventory_reply(
                        visible,
                        cognitive_engine_handled=True,
                        model_label=_canonical_runtime_model_label(lane),
                    )
                )
                if (
                    grounded_inventory
                    and not _chat_desktop_repair._capability_inventory_reply_is_inadequate(
                        visible,
                        grounded_inventory,
                    )
                ):
                    logger.warning(
                        "CognitiveEngine capability inventory remained clipped after "
                        "validation; binding reply to governed capability evidence "
                        "without a second Cortex retry."
                    )
                    _mark_turn_trace(
                        cognitive_engine_reply_accepted=True,
                        bounded_contract_used=False,
                        response_path="cognitive_engine_capability_tail_grounding",
                    )
                    return grounded_inventory
            if require_engine and memory_state_contract:
                grounded_memory_reply = _canonical_memory_state_grounding_reply(
                    visible,
                    canonical_memory_state_evidence,
                    live_mind_context=live_mind_context,
                )
                if grounded_memory_reply:
                    logger.warning(
                        "CognitiveEngine desktop chat missed canonical memory/state evidence; "
                        "binding visible reply to canonical memory gateway after engine invocation."
                    )
                    _mark_turn_trace(
                        cognitive_engine_reply_accepted=True,
                        bounded_contract_used=False,
                        response_path="cognitive_engine_memory_state_grounding",
                    )
                    return grounded_memory_reply
            if groundable_self_process_miss:
                grounded_self_process_reply = await _build_grounded_self_process_repair_reply(
                    visible,
                    text,
                    lane=lane,
                    session_id=session_id,
                )
                if grounded_self_process_reply:
                    grounded_self_process_assessment = assess_user_facing_reply(
                        visible,
                        grounded_self_process_reply,
                        recent_user_messages=recent_user_messages,
                        grounding=route_assessment_grounding,
                        antecedent=route_assessment_antecedent,
                    )
                    if not _reply_assessment_requires_repair_with_memory_evidence(
                        grounded_self_process_assessment,
                        visible,
                        grounded_self_process_reply,
                        canonical_memory_state_evidence=canonical_memory_state_evidence,
                    ):
                        logger.info(
                            "CognitiveEngine desktop chat bound self-process turn to canonical live-state grounding."
                        )
                        _mark_turn_trace(
                            cognitive_engine_reply_accepted=False,
                            cognitive_engine_reply_failed=True,
                            bounded_contract_used=True,
                            post_generation_repair_applied=True,
                            deterministic_repair_applied=True,
                            response_path="cognitive_engine_self_process_grounding",
                        )
                        return grounded_self_process_reply
            if require_engine and is_status_check_turn(visible):
                logger.warning(
                    "CognitiveEngine desktop chat status reply was too thin; "
                    "not replacing a required full-mind turn with a bounded status repair."
                )
            if self_condition_contract and "unanswered_question_part" in set(
                assessment_reasons
            ):
                try:
                    from core.conversation.request_coverage import (
                        complete_epistemic_partition_from_evidence,
                    )

                    grounded_completion = complete_epistemic_partition_from_evidence(
                        visible,
                        text,
                        canonical_self_condition_reply,
                    )
                except _CHAT_RECOVERABLE_ERRORS as exc:
                    record_degradation("chat.self_condition_completion", exc)
                    grounded_completion = text
                if grounded_completion != text:
                    grounded_assessment = assess_user_facing_reply(
                        visible,
                        grounded_completion,
                        recent_user_messages=recent_user_messages,
                        grounding=route_assessment_grounding,
                        antecedent=route_assessment_antecedent,
                    )
                    if not _reply_assessment_requires_repair_with_memory_evidence(
                        grounded_assessment,
                        visible,
                        grounded_completion,
                        canonical_memory_state_evidence=canonical_memory_state_evidence,
                    ):
                        _append_turn_text_mutation(
                            turn_trace,
                            stage="chat.self_condition_epistemic_completion",
                            method="typed_evidence_semantic_merge",
                            reasons=["unanswered_question_part"],
                            before=text,
                            after=grounded_completion,
                            deterministic=True,
                            authorship_effect="augmented_by_runtime",
                        )
                        _mark_turn_trace(
                            cognitive_engine_reply_accepted=True,
                            cognitive_engine_reply_failed=False,
                            post_generation_repair_applied=True,
                            deterministic_repair_applied=True,
                            response_path="cognitive_engine_self_condition_semantic_completion",
                        )
                        return grounded_completion
            if require_engine:
                retry_reply = await _attempt_repair_retry(text, assessment.reasons)
                if retry_reply:
                    if turn_trace is not None:
                        turn_trace.update(
                            {
                                "cognitive_engine_reply_accepted": True,
                                "response_path": "cognitive_engine_repair_retry",
                            }
                        )
                    return retry_reply
                if self_condition_contract:
                    refreshed_condition_reply = _build_grounded_self_condition_reply(
                        visible,
                        session_id=session_id,
                    )
                    if not refreshed_condition_reply:
                        refreshed_condition_reply = canonical_self_condition_reply
                    # Through the same typed-evidence projection the egress path
                    # uses. The reliability gate rejects an unsupported
                    # operational claim wherever it appears, and this reply is
                    # built from the canonical projection — so it has to be
                    # scoped by that projection before it is judged, or the one
                    # answer that IS grounded gets refused for saying so.
                    refreshed_condition_reply = _project_self_condition_claims(
                        refreshed_condition_reply,
                        self_condition_evidence.get("projection_dict"),
                        turn_trace,
                    )
                    if refreshed_condition_reply:
                        condition_assessment = assess_user_facing_reply(
                            visible,
                            refreshed_condition_reply,
                            recent_user_messages=recent_user_messages,
                            grounding=route_assessment_grounding,
                            antecedent=route_assessment_antecedent,
                        )
                        if not _reply_assessment_requires_repair_with_memory_evidence(
                            condition_assessment,
                            visible,
                            refreshed_condition_reply,
                            canonical_memory_state_evidence=canonical_memory_state_evidence,
                        ):
                            logger.warning(
                                "CognitiveEngine self-condition generation and bounded "
                                "same-worker correction both missed the inner-state contract; "
                                "serving a clearly bounded canonical projection."
                            )
                            _append_turn_text_mutation(
                                turn_trace,
                                stage="chat.self_condition_bounded_projection",
                                method="deterministic_self_condition_projection",
                                reasons=list(assessment.reasons or ()),
                                before=text,
                                after=refreshed_condition_reply,
                                deterministic=True,
                                authorship_effect="replaced_by_runtime",
                            )
                            _mark_turn_trace(
                                cognitive_engine_reply_accepted=False,
                                cognitive_engine_reply_failed=True,
                                bounded_contract_used=True,
                                post_generation_repair_applied=True,
                                deterministic_repair_applied=True,
                                response_path="cognitive_engine_self_condition_grounding",
                                self_condition_contract=True,
                            )
                            return refreshed_condition_reply
                expected_recall_reply = await _chat_memory_state._build_conversation_recall_reply(
                    visible,
                    session_id=session_id,
                )
                if expected_recall_reply:
                    # Serve it, bounded, rather than refuse with an apology.
                    #
                    # This branch used to return None on the grounds that a
                    # deterministic substitution is not her own answer on a
                    # required full-mind turn. What the caller then serves is
                    # "I couldn't get my full attention onto that one" — which
                    # is not her answer either, carries nothing, and is false
                    # about what happened: the attention was there and a gate
                    # rejected a draft.
                    #
                    # The reply here is built from the transcript of this
                    # conversation. It is the most grounded thing available and
                    # the branch immediately above already does exactly this
                    # for self-condition grounding, marked
                    # `replaced_by_runtime` so nobody mistakes it for
                    # generation. The two contracts differed only in which one
                    # had been written second.
                    #
                    # LIVE, 2026-09-07: "What did I just ask you?" one turn
                    # after the question it was recalling. 411 seconds, two
                    # rejected drafts, and the apology — with a correct answer
                    # in hand the whole time.
                    condition_assessment = assess_user_facing_reply(
                        visible,
                        expected_recall_reply,
                        recent_user_messages=recent_user_messages,
                        grounding=route_assessment_grounding,
                        antecedent=route_assessment_antecedent,
                    )
                    if not _reply_assessment_requires_repair_with_memory_evidence(
                        condition_assessment,
                        visible,
                        expected_recall_reply,
                        canonical_memory_state_evidence=canonical_memory_state_evidence,
                    ):
                        logger.warning(
                            "CognitiveEngine desktop chat missed the required "
                            "conversation recall contract; serving the bounded "
                            "recall built from this conversation's transcript."
                        )
                        _append_turn_text_mutation(
                            turn_trace,
                            stage="chat.conversation_recall_bounded_projection",
                            method="deterministic_conversation_recall",
                            reasons=list(assessment.reasons or ()),
                            before=text,
                            after=expected_recall_reply,
                            deterministic=True,
                            authorship_effect="replaced_by_runtime",
                        )
                        _mark_turn_trace(
                            cognitive_engine_reply_accepted=False,
                            cognitive_engine_reply_failed=True,
                            bounded_contract_used=True,
                            post_generation_repair_applied=True,
                            deterministic_repair_applied=True,
                            response_path="cognitive_engine_recall_bounded_projection",
                            conversation_recall_contract=True,
                        )
                        return expected_recall_reply
                    logger.warning(
                        "CognitiveEngine desktop chat missed the required "
                        "conversation recall contract, and the bounded recall "
                        "does not survive the reply assessment either."
                    )
                    _mark_turn_trace(
                        cognitive_engine_reply_accepted=False,
                        bounded_contract_used=False,
                        response_path="cognitive_engine_recall_contract_failed",
                    )
                    _record_exhausted_cognitive_failure(
                        "conversation_recall_contract_failed",
                        retry_attempted=True,
                    )
                    return None
                if context_challenge_context and _context_challenge_repair_has_evidence(
                    context_challenge_context
                ):
                    context_repair_assessment = assess_user_facing_reply(
                        visible,
                        context_challenge_context,
                        recent_user_messages=recent_user_messages,
                        grounding=route_assessment_grounding,
                        antecedent=route_assessment_antecedent,
                    )
                    if not _reply_assessment_requires_repair_with_memory_evidence(
                        context_repair_assessment,
                        visible,
                        context_challenge_context,
                        canonical_memory_state_evidence=canonical_memory_state_evidence,
                    ):
                        logger.warning(
                            "CognitiveEngine desktop chat missed the required "
                            "context-relevance contract; binding visible reply to "
                            "canonical conversation evidence after engine invocation."
                        )
                        _mark_turn_trace(
                            cognitive_engine_reply_accepted=True,
                            bounded_contract_used=False,
                            response_path="cognitive_engine_context_evidence_repair",
                        )
                        return context_challenge_context
                    logger.warning(
                        "CognitiveEngine desktop chat context evidence repair failed "
                        "reliability gate (%s).",
                        ",".join(getattr(context_repair_assessment, "reasons", ()) or ()),
                    )
                    _mark_turn_trace(response_path="cognitive_engine_context_contract_failed")
                    _record_exhausted_cognitive_failure(
                        "context_relevance_contract_failed",
                        retry_attempted=True,
                    )
                    return None
                if not _reply_gate_proved_a_violation(assessment):
                    # The gate said no and could not say why. An unnamed
                    # violation is not a proven one, and discarding a complete
                    # reply on it hands the person a canned refusal instead of
                    # the answer that was already in hand.
                    record_degradation(
                        "chat",
                        RuntimeError(
                            "reply reliability gate rejected a reply without "
                            "naming a violation; serving the reply"
                        ),
                        action="served the drafted reply and recorded the gate inconsistency",
                    )
                    _mark_turn_trace(
                        cognitive_engine_reply_accepted=True,
                        bounded_contract_used=False,
                        response_path="cognitive_engine_reply_gate_unnamed",
                    )
                    return text
                _mark_turn_trace(
                    cognitive_engine_reply_accepted=False,
                    bounded_contract_used=False,
                    response_path="cognitive_engine_reply_rejected",
                )
                _record_exhausted_cognitive_failure(
                    _named_gate_failure(assessment),
                    retry_attempted=True,
                )
                return None
            (
                repaired,
                stale,
                same_diff,
                off_topic,
                off_topic_reason,
                did_repair,
            ) = await _repair_final_degraded_reply_with_provenance(
                turn_trace,
                stage="chat.cognitive_engine_final_gate",
                user_message=visible,
                reply_text=text,
                stale=False,
                same_diff=False,
                off_topic=False,
                desktop_cognitive_engine_required=bool(require_engine),
                protected_foreground_lane=bool(require_engine),
                session_id=session_id,
            )
            repaired_assessment = assess_user_facing_reply(
                visible,
                repaired,
                recent_user_messages=recent_user_messages,
                grounding=route_assessment_grounding,
                antecedent=route_assessment_antecedent,
            )
            if did_repair and not (
                stale
                or same_diff
                or off_topic
                or _reply_assessment_requires_repair_with_memory_evidence(
                    repaired_assessment,
                    visible,
                    repaired,
                    canonical_memory_state_evidence=canonical_memory_state_evidence,
                )
            ):
                logger.info("CognitiveEngine desktop chat reply recovered by general repair path.")
                _mark_turn_trace(
                    cognitive_engine_reply_accepted=False,
                    bounded_contract_used=True,
                    response_path="cognitive_engine_shape_repair_bounded",
                )
                return (
                    repaired
                    if memory_state_contract
                    else _ground_runtime_fact_status_reply(
                        visible,
                        repaired,
                        lane,
                        cognitive_engine_handled=True,
                    )
                )
            logger.warning(
                "CognitiveEngine desktop chat repair failed reliability gate "
                "(stale=%s same_diff=%s off_topic=%s reason=%s assessment=%s).",
                stale,
                same_diff,
                off_topic,
                off_topic_reason,
                ",".join(repaired_assessment.reasons),
            )
            if not require_engine:
                conversation_recall_reply = (
                    await _chat_memory_state._build_conversation_recall_reply(
                        visible,
                        session_id=session_id,
                    )
                )
                if conversation_recall_reply:
                    logger.warning(
                        "CognitiveEngine chat failed repair for conversation recall; "
                        "repairing from canonical conversation log."
                    )
                    _mark_turn_trace(
                        bounded_contract_used=True,
                        response_path="conversation_recall_log_repair_after_cognitive_engine",
                    )
                    return _ground_runtime_fact_status_reply(
                        visible,
                        conversation_recall_reply,
                        lane,
                        cognitive_engine_handled=True,
                    )
                owner_name_reply = _chat_memory_state._build_owner_name_recall_reply(visible)
                if owner_name_reply:
                    logger.warning(
                        "CognitiveEngine chat failed repair for owner identity recall; "
                        "repairing from verified runtime identity contract."
                    )
                    _mark_turn_trace(
                        bounded_contract_used=True,
                        response_path="owner_identity_repair_after_cognitive_engine",
                    )
                    return _ground_runtime_fact_status_reply(
                        visible,
                        owner_name_reply,
                        lane,
                        cognitive_engine_handled=True,
                    )
            # A wrong name at the front of a good answer is not a reason to
            # lose the answer. The hard-failure set already says so — it
            # excludes this reason on the grounds that "the honest remedy is
            # to drop the vocative and deliver the answer" — but nothing
            # dropped it, so the reason simply blocked, and being excluded
            # from the retryable set it blocked with no second attempt.
            devocatived = _strip_ungrounded_vocative_reply(visible, text)
            if devocatived:
                from core.conversation.response_reliability import (
                    assess_user_facing_reply as _assess_devocatived,
                )

                if not _reply_assessment_requires_repair_with_memory_evidence(
                    _assess_devocatived(
                        visible,
                        devocatived,
                        recent_user_messages=recent_user_messages,
                        grounding=route_assessment_grounding,
                        antecedent=route_assessment_antecedent,
                    ),
                    visible,
                    devocatived,
                    canonical_memory_state_evidence=canonical_memory_state_evidence,
                ):
                    logger.warning(
                        "Reply carried an unsupported opening name (%s); removed "
                        "the address and kept the answer.",
                        ",".join(assessment.reasons),
                    )
                    _append_turn_text_mutation(
                        turn_trace,
                        stage="chat.ungrounded_person_address",
                        method="opening_vocative_removed",
                        reasons=["ungrounded_person_address"],
                        before=text,
                        after=devocatived,
                        deterministic=True,
                        authorship_effect="preserved",
                    )
                    _mark_turn_trace(
                        cognitive_engine_reply_accepted=True,
                        response_path="cognitive_engine_devocatived",
                    )
                    return devocatived
            retry_reply = await _attempt_repair_retry(text, assessment.reasons)
            if retry_reply:
                if turn_trace is not None:
                    turn_trace.update(
                        {
                            "cognitive_engine_reply_accepted": True,
                            "response_path": "cognitive_engine_repair_retry",
                        }
                    )
                return retry_reply
            # Every other contract here has somewhere to fall back to: the
            # recall log, the identity record, the capability catalog. A
            # question about her own source had none, so a rejected draft
            # ended the turn — and the tree, which answers it outright, was
            # never opened.
            #
            # Live 2026-08-04: "Where in the codebase can I find that" was
            # thrown out for `ungrounded_person_address` and the person got
            # "I couldn't get to an answer I'd stand behind" one turn after
            # she had shown them code. The citation was on record. Nothing
            # asked for it. A gate may reject a DRAFT; it must not be able
            # to withhold an answer that was sitting on disk.
            source_rescue = await _own_source_rescue_reply(visible)
            if source_rescue:
                logger.warning(
                    "Reply for a question about her own source failed the "
                    "reliability gate (%s); answering from the source tree "
                    "instead of failing the turn.",
                    ",".join(assessment.reasons),
                )
                _mark_turn_trace(
                    cognitive_engine_reply_accepted=True,
                    bounded_contract_used=False,
                    response_path="cognitive_engine_own_source_grounding",
                )
                return source_rescue
            _mark_turn_trace(response_path="cognitive_engine_reply_rejected")
            _record_exhausted_cognitive_failure(
                _named_gate_failure(assessment),
                retry_attempted=True,
            )
            return None
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("CognitiveEngine reply reliability gate unavailable: %s", exc)
    if require_engine:
        expected_recall_reply = await _chat_memory_state._build_conversation_recall_reply(
            visible,
            session_id=session_id,
        )
        if expected_recall_reply and _conversation_recall_reply_is_inadequate(
            visible,
            text,
            expected_recall_reply,
        ):
            # The runtime just BUILT the correct answer. Serve it.
            #
            # expected_recall_reply is composed from the conversation's own
            # turns — it is a reading of what was said, not model prose
            # substituted for model prose, so serving it cannot invent an
            # exchange. Refusing here threw away a correct answer the runtime
            # was holding and returned "I couldn't get to an answer I'd stand
            # behind on that one" (LIVE 2026-08-17, "what was the first thing I
            # said to you in this conversation?").
            #
            # The same lesson is written three other places in this file: a
            # computed arithmetic result, a measured file count, and a
            # receipt-backed past action all beat an apology. What was refused
            # here is the same category — the transcript is the source, and it
            # was already read.
            logger.warning(
                "CognitiveEngine desktop chat missed the required conversation recall "
                "contract; serving the transcript-composed recall instead of refusing."
            )
            _mark_turn_trace(
                cognitive_engine_reply_accepted=False,
                bounded_contract_used=True,
                response_path="conversation_recall_from_transcript",
            )
            return expected_recall_reply
        if _context_challenge_reply_is_inadequate(visible, text):
            if context_challenge_context and _context_challenge_repair_has_evidence(
                context_challenge_context
            ):
                from core.conversation.response_reliability import assess_user_facing_reply

                context_repair_assessment = assess_user_facing_reply(
                    visible,
                    context_challenge_context,
                    recent_user_messages=route_recent_user_messages,
                    grounding=route_assessment_grounding,
                    antecedent=route_assessment_antecedent,
                )
                if not _reply_assessment_requires_repair_with_memory_evidence(
                    context_repair_assessment,
                    visible,
                    context_challenge_context,
                    canonical_memory_state_evidence=canonical_memory_state_evidence,
                ):
                    logger.warning(
                        "CognitiveEngine desktop chat missed the required context-relevance contract; "
                        "binding visible reply to canonical conversation evidence after engine invocation."
                    )
                    _mark_turn_trace(
                        cognitive_engine_reply_accepted=True,
                        bounded_contract_used=False,
                        response_path="cognitive_engine_context_evidence_repair",
                    )
                    return context_challenge_context
                logger.warning(
                    "CognitiveEngine desktop chat context evidence repair failed reliability gate (%s).",
                    ",".join(getattr(context_repair_assessment, "reasons", ()) or ()),
                )
                _mark_turn_trace(response_path="cognitive_engine_context_contract_failed")
                return None
            logger.warning(
                "CognitiveEngine desktop chat missed the required context-relevance contract; "
                "refusing degraded visible reply."
            )
            _mark_turn_trace(response_path="cognitive_engine_context_contract_failed")
            return None
    # "Here is my code" is a claim that can be SETTLED, so it gets settled.
    #
    # Carrying real excerpts into the turn was necessary and not sufficient:
    # live 2026-08-04 the evidence reached the prompt and she still produced
    # `retrieve_contextual_memory()`, a function in no file here, introduced
    # as "a snippet from my cognitive architecture". Notes can be overridden.
    # Either those lines are in the tree or they are not.
    #
    # Only a PROVEN absence acts. A search that could not run proves nothing,
    # and treating that as fabrication would destroy real excerpts whenever
    # the search itself broke.
    text = await _check_a_reply_against_her_own_source(
        text=text,
        turn_trace=turn_trace,
        visible=visible,
    )

    if turn_trace is not None:
        accepted_response_path = str(turn_trace.get("response_path") or "").strip()
        if not accepted_response_path:
            accepted_response_path = (
                "cognitive_engine_runtime_fact_grounding"
                if runtime_fact_status_contract and not memory_state_contract
                else "cognitive_engine"
            )
        turn_trace.update(
            {
                "cognitive_engine_reply_accepted": True,
                "response_path": accepted_response_path,
            }
        )
    return (
        text
        if memory_state_contract
        else _ground_runtime_fact_status_reply(
            visible,
            text,
            lane,
            cognitive_engine_handled=True,
        )
    )


def _looks_like_unrequested_content_review(user_message: str, reply_text: str) -> tuple[bool, str]:
    user_text = _chat_memory_state._normalize_user_message(user_message)
    reply = _chat_memory_state._normalize_user_message(reply_text)
    if not reply:
        return False, ""
    if any(marker in user_text for marker in _CONTENT_OBJECT_MARKERS):
        return False, ""

    review_hits = sum(1 for marker in _UNREQUESTED_CONTENT_REVIEW_MARKERS if marker in reply)
    object_hits = sum(
        1 for marker in _CONTENT_OBJECT_MARKERS if re.search(rf"\b{re.escape(marker)}\b", reply)
    )
    if review_hits >= 1 and object_hits >= 2:
        return True, "unrequested_content_review"
    if (
        reply.startswith(("the story is", "the premise", "this story", "this narrative"))
        and object_hits >= 2
    ):
        return True, "unrequested_content_review"
    return False, ""




#: Somebody asking her about herself, where a reply about her own workings is
#: the answer rather than a wandering.
_ABOUT_HER_OWN_WORKINGS = re.compile(
    r"\byou(?:'|\u2019)?r?e?\b[^.?!]{0,80}\b(?:able|can|could|do|doing|able\s+to|"
    r"capab\w+|built|made|work\w*|run\w*|learn\w*|chang\w+|improv\w+|"
    r"handle|manage|reach|know|remember)\b"
    r"|\b(?:what|how|why|when)\b[^.?!]{0,40}\byour\b",
    re.IGNORECASE,
)


def _asked_about_her_own_workings(user_message: str) -> bool:
    """Whether the person asked about her, rather than about something else."""
    return bool(_ABOUT_HER_OWN_WORKINGS.search(str(user_message or "")))


async def _run_recent_response_reasoning_audit(text: str) -> None:
    try:
        await _chat_memory_state._await_bounded_chat_blocking(
            _audit_recent_response_reasoning_sync,
            text,
            timeout_s=_CHAT_REASONING_AUDIT_TIMEOUT_S,
            operation_name="post_reply_symbolic_audit",
        )
    except TimeoutError:
        logger.warning(
            "Post-reply symbolic audit exceeded %.1fs; delivery remained independent.",
            _CHAT_REASONING_AUDIT_TIMEOUT_S,
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.reasoning_audit", exc)


def _schedule_recent_response_reasoning_audit(text: str) -> None:
    if not text or len(str(text)) >= 4000:
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # Synchronous tools/tests still update repetition state; the live
        # symbolic audit requires a supervised runtime loop.
        return
    active = {task for task in _reasoning_audit_tasks if not task.done()}
    _reasoning_audit_tasks.clear()
    _reasoning_audit_tasks.update(active)
    if len(active) >= _CHAT_REASONING_AUDIT_MAX_ACTIVE:
        logger.info("Post-reply symbolic audit skipped under bounded backpressure.")
        return
    task = get_task_tracker().bounded_track(
        _run_recent_response_reasoning_audit(str(text)),
        name="ChatPostReplySymbolicAudit",
        owner="interface.routes.chat",
    )
    _reasoning_audit_tasks.add(task)
    task.add_done_callback(_reasoning_audit_tasks.discard)


def _record_recent_response(text: str, user_message: str = "") -> None:
    fp = _response_fingerprint(text)
    with _conversation_quality_lock:
        state = _conversation_quality_state_locked()
        if fp:
            state.recent_responses.append(fp)
        # A real delivered answer ends any degraded-status streak: the escalation
        # clause must count CONSECUTIVE failures, not lifetime ones.
        state.lane_status_fingerprint = ""
        state.lane_status_repeat_count = 0
        if user_message:
            response_body = _normalize_response_body(text)[:500]
            if response_body:
                state.recent_response_pairs.append(
                    (_response_fingerprint(user_message), response_body)
                )
    # Delivery never waits for tableau/numeric proof. The audit is supervised,
    # off-loop, deadline-bound, and backpressured instead of merely being called
    # "non-blocking" while running synchronously.
    _schedule_recent_response_reasoning_audit(str(text))


def _is_stale_repeated_response(text: str) -> bool:
    fp = _response_fingerprint(text)
    if not fp:
        return False
    with _conversation_quality_lock:
        responses = tuple(_conversation_quality_state_locked().recent_responses)
    exact_count = sum(1 for response in responses if response == fp)
    if exact_count >= _STALE_REPEAT_THRESHOLD:
        return True
    # Fuzzy similarity check — catches "same answer, slightly different wording"
    fuzzy_count = sum(1 for response in responses if _fuzzy_similar(fp, response))
    if fuzzy_count >= _STALE_REPEAT_THRESHOLD:
        logger.debug("Fuzzy stale detection triggered (overlap count=%d).", fuzzy_count)
        return True
    return False




_EQUIVALENT_REPAIR_PROMPT_GROUPS = (
    (
        "huh",
        "wait what",
        "confused",
        "doesn't make sense",
        "does not make sense",
        "not making sense",
    ),
    ("you ok", "you okay", "are you ok", "are you okay", "feeling better", "for real this time"),
    ("coherent", "still there", "able to talk", "can you talk", "chat", "response", "conversation"),
)


def _same_repair_prompt_class(a: str, b: str) -> bool:
    left = _chat_memory_state._normalize_user_message(a)
    right = _chat_memory_state._normalize_user_message(b)
    if not left or not right:
        return False
    for group in _EQUIVALENT_REPAIR_PROMPT_GROUPS:
        if any(marker in left for marker in group) and any(marker in right for marker in group):
            return True
    return False






# ── Response Quality Metrics (extracted to chat_quality.py) ──
from interface.routes.chat_quality import (  # noqa: E402
    _check_response_consistency,
    _extract_and_register_commitments,
    _log_response_quality_metrics,
    _reply_assessment_requires_repair,
    assess_post_response_confidence,
)
from .chat_reply_repair import (
    _ends_where_it_meant_to,
    _evaluate_reply_topicality,
    _is_actionably_stale_response,
    _is_same_answer_different_prompt,
    _looks_semantically_glitched,
    _measure_reply_quality_candidate,
    _original_reply_is_safe_to_surface,  # noqa: F401
    _recheck_a_degraded_reply,
    _repair_a_degraded_affect_reply,
    _repair_a_degraded_identity_reply,
    _repair_final_degraded_reply,  # noqa: F401
    _repair_final_degraded_reply_with_provenance,
    _repair_missing_followup_delta,  # noqa: F401
    _serve_the_bounded_repair,
    _serve_the_identity_repair,
    _stabilize_user_facing_reply,
    _strip_unexpected_cjk_artifacts,
    _strip_user_visible_context_leaks,
)
from .chat_own_source import (
    _ASKS_TO_INSPECT_SHOWN_SOURCE_RE,  # noqa: F401
    _ASKS_WHERE_CODE_LIVES_RE,  # noqa: F401
    _OWN_SOURCE_ROUTE_MARGIN,  # noqa: F401
    _RECEIPTS_WORTH_READING,  # noqa: F401
    _REPO_PROBE_MAX_BYTES,  # noqa: F401
    _SELF_METRIC_CORRECTION_MARK,  # noqa: F401
    _SELF_PROCESS_ABOUT_HER_RE,  # noqa: F401
    _SELF_PROCESS_HYPOTHETICAL_RE,  # noqa: F401
    _attempt_generated_social_grounding_repair,  # noqa: F401
    _build_grounded_self_process_repair_reply,
    _build_minimal_grounded_self_process_repair_reply,
    _check_a_reply_against_her_own_source,
    _correct_unsourced_self_metrics,
    _explains_the_finding,  # noqa: F401
    _own_source_rescue_reply,
    _read_repo_probe_reply,
    _self_process_requested_dimensions,
    _serve_repo_diagnosis,
    _turn_asks_where_that_came_from,  # noqa: F401
    _turn_may_concern_own_source,
    _what_the_tools_found,
)
from .chat_refusals import (
    _a_proof_that_says_the_answer_is_unfinished,  # noqa: F401
    _anything_better_than_giving_up,
    _fail_closed_on_an_unproven_full_mind_contract,
    _fail_closed_on_an_unproven_output_contract,
    _in_plain_words,  # noqa: F401
    _refuse_an_empty_benchmark_reply,
    _refuse_an_empty_canonical_reply,
    _refuse_an_unmet_benchmark_contract,
    _serve_the_capability_inventory,
    _why_there_is_no_answer,  # noqa: F401
)
from .chat_recorded_answers import (
    _apply_recorded_answer,
    _apply_regenerated_reply,
    _recorded_answer_corrections,  # noqa: F401
    _reply_was_served_from_a_record,  # noqa: F401
)
from .chat_foreground_lane import (
    _admit_to_foreground_lane,
    _answer_from_fallback_ladder,
    _answer_from_grounded_introspection,
    _await_foreground_gate,
    _await_the_sovereign_kernel_reply,
    _boot_is_still_in_progress,  # noqa: F401
    _fallback_conversation_messages,  # noqa: F401
    _foreground_timeout_for_lane,
    _lane_warmup_is_deliberately_deferred,  # noqa: F401
    _protected_foreground_reply,
    _protected_foreground_route,  # noqa: F401
    _user_requested_primary_only,  # noqa: F401
)
from .chat_turn_recall import (
    _append_past_action_record,
    _build_retained_memory_evidence_context,
    _format_traceability_reply,
    _is_referential_followup_request,
    _is_retained_memory_evidence_request,  # noqa: F401
    _memory_state_evidence_is_missing_from_reply,
    _memory_state_reply_satisfies_canonical_evidence,  # noqa: F401
    _reply_assessment_requires_repair_with_memory_evidence,
    _resolve_referential_followup_anchor,
    _turn_may_concern_perception,
)

#: Returned by an extracted block that did NOT return early. A unique
#: object, so no value a block legitimately returns can be mistaken for it.
_SEAM_FELL_THROUGH = object()

# ── Conversation Lane Helpers ─────────────────────────────────








_VERIFIED_STATE_PROJECTION_AUTHORITIES: dict[str, tuple[str, str]] = {
    "verified_action_episode": (
        "verified_action_episode_serialization",
        "governed_action_episode",
    ),
    "verified_answer_provenance": (
        "verified_answer_provenance_serialization",
        "answer_bound_turn_evidence",
    ),
}


def _verified_state_projection_authority(
    status: str | None,
) -> tuple[str, str] | None:
    """Return typed serialization authority carried by a response status.

    A deterministic projection of verified state is neither model-authored
    prose nor an unowned runtime replacement. Its authority comes from the
    typed object it serializes. Keeping that distinction in one registry stops
    generic topicality heuristics from throwing away exact answers merely
    because an explanation introduces vocabulary absent from a short question.
    """

    return _VERIFIED_STATE_PROJECTION_AUTHORITIES.get(str(status or "").strip())


# Enough for the fallback ladder to cold-load a small model AND generate a
# real reply — the point is a genuine answer from a lower rung, not a faster
# apology.
_DEFERRED_CORTEX_TURN_TIMEOUT_S = 75.0








# Consecutive-repeat tracking for degraded status messages. Saying the SAME
# sentence 32 times in a row (the 2026-07-18 soak's
# `identical_reply_repeated_x32`) reads as a broken loop even when every
# individual sentence is true. Naming the repetition is both more honest and
# more actionable than pretending each occurrence is fresh.
_LANE_STATUS_REPEAT_NOTICE_AFTER = 2


def _lane_status_repeat_suffix(message: str) -> str:
    """Honest escalation clause when the same status recurs back-to-back."""
    fingerprint = " ".join(str(message or "").lower().split())[:160]
    with _conversation_quality_lock:
        state = _conversation_quality_state_locked()
        if fingerprint and fingerprint == state.lane_status_fingerprint:
            state.lane_status_repeat_count += 1
        else:
            state.lane_status_fingerprint = fingerprint
            state.lane_status_repeat_count = 1
        count = state.lane_status_repeat_count
    if count <= _LANE_STATUS_REPEAT_NOTICE_AFTER:
        return ""
    return (
        f" (This is the {_turn_count_ordinal(count)} turn in a row I've had to say this — "
        "the lane is not recovering on its own, so this is worth looking at "
        "rather than retrying.)"
    )


def _reset_lane_status_repeat_state() -> None:
    """A real answer clears the streak — only consecutive failures count."""
    with _conversation_quality_lock:
        state = _conversation_quality_state_locked()
        state.lane_status_fingerprint = ""
        state.lane_status_repeat_count = 0


_FOREGROUND_GATE_BOOT_WAIT_S = 90.0
#: A fallback answer is only useful if it beats the cortex finishing its load.
#: This is the floor, not the budget: what the ladder actually gets is whatever
#: the turn has left, because refusing at a hundred and ninety seconds while a
#: model finishes loading at sixty is the worst reading of "be quick".
_FALLBACK_LADDER_TIMEOUT_S = 25.0




#: Where a sentence can end. A reply that stops anywhere else stopped because
#: something ran out, not because it had finished.
_A_SENTENCE_ENDS = tuple(".!?:\u2026") + (
    '."', ".'", '!"', '?"', ".)", ".]", ".`",
    # A code fence that closed is a finished thought, and it is the one ending
    # that carries no punctuation at all.
    "```",
)


#: Reasons a generator gives for stopping that mean it had finished.
_FINISHED = frozenset({"configured_stop", "eos", "semantic_contract_satisfied"})


#: A list marker left at the end with nothing after it.
_A_DANGLING_MARKER = re.compile(r"(?:\n|^)\s*(?:\d+[.)]|[-*\u2022]|#{1,6})\s*$")












def _conversation_lane_user_message(
    lane: dict[str, Any],
    *,
    timed_out: bool = False,
    status_override: str = "",
) -> str:
    # A fact the machine holds is not the lane's to withhold.
    known = _known_answer_for_this_turn()
    if known:
        return known
    message = _lane_status_message_body(lane, timed_out=timed_out, status_override=status_override)
    return message + _lane_status_repeat_suffix(message)


_PROTECTED_FOREGROUND_LOCK_BYPASS_SECONDS: float = 1.0
_PROTECTED_FOREGROUND_PRIMARY_BUDGET_SECONDS: float = 300.0
_PROTECTED_FOREGROUND_SECONDARY_BUDGET_SECONDS: float = 360.0
# [STABILITY v53] Raised from 8s→45s. The old 8s deadline was the #1 cause of
# false-positive kernel timeouts on first-turn responses. The 32B cortex
# regularly needs 15-40s for complex responses, and after a 35s warmup the
# kernel had only 8s before being interrupted by a competing protected
# foreground request — which itself competes for the same LLM resources,
# creating a resource contention spiral. 45s gives the kernel real time to
# respond on turn 1. Subsequent turns (model warm, KV cache hot) are <5s.
_KERNEL_SOFT_REPLY_SLA_SECONDS: float = 180.0


def _kernel_is_congested(lane: dict[str, Any] | None) -> bool:
    lane = dict(lane or {})
    if not bool(lane.get("kernel_lock_held", False)):
        return False
    return (
        float(lane.get("kernel_lock_held_s", 0.0) or 0.0)
        >= _PROTECTED_FOREGROUND_LOCK_BYPASS_SECONDS
    )


def _protected_foreground_reason(lane: dict[str, Any] | None) -> str:
    lane = dict(lane or {})
    lane_state = str(lane.get("state", "") or "").strip().lower()
    if lane_state == "recovering" and _in_recovery_cooldown():
        return "recovery_cooldown"
    if _kernel_is_congested(lane):
        return f"kernel_lock:{float(lane.get('kernel_lock_held_s', 0.0) or 0.0):.2f}s"
    if not bool(lane.get("conversation_ready", False)) and lane_state in {
        "warming",
        "recovering",
        "cold",
        "spawning",
        "handshaking",
    }:
        return f"lane_{lane_state or 'unready'}"
    return ""










#: The vitals `/api/health` publishes as ``liquid_state``.
















_PROMPT_ARTIFACT_PREFIX_RE = re.compile(
    r"^\s*(?:obj|prev_obj|state|phenom|mood|goals|history|narr|pers|usr|ctx|voice|user|input|message)\s*:\s*",
    re.IGNORECASE,
)


def _surface_fingerprint(text: str) -> str:
    cleaned = str(text or "").strip()
    for _ in range(12):
        stripped = _PROMPT_ARTIFACT_PREFIX_RE.sub("", cleaned).strip().strip("\"'“”`")
        if stripped == cleaned:
            break
        cleaned = stripped
    cleaned = re.sub(r"[^\w\s']+", " ", cleaned.lower())
    return " ".join(cleaned.split())


def _is_objective_parrot_reply(user_message: str, reply_text: Any) -> bool:
    reply_fp = _surface_fingerprint(str(reply_text or ""))
    user_fp = _surface_fingerprint(str(user_message or ""))
    if not reply_fp or not user_fp:
        return False
    if reply_fp == user_fp:
        return True
    if reply_fp.startswith(user_fp):
        remainder = reply_fp[len(user_fp) :].strip()
        if not remainder or len(remainder.split()) <= 2:
            return True
    return False


_SOFT_REPAIRABLE_REPLY_SHAPE_REASONS = {
    "missing_requested_paragraph_count",
    "missing_requested_list_count",
    "missing_requested_followup_question",
}




# A conversation is not a host. LIVE DEFECT, 2026-07-25.
#
# The fallback session id was the client IP, with the comment "client host is
# good enough for single-user local Aura". It was not. Every desktop
# conversation ever held collapsed into one session — 881 turns under
# "127.0.0.1", mixing Bryan's chats with Codex's automated live-route probes
# and latency samples — and each new chat replayed that soup as its own prior
# thread.
#
# That is what produced the confabulations. Asked "Hey, bud. Are you with
# me?" Aura answered "Sitting with you on the drive. They're still outside?"
# and then "Well, the cops. You called them?" — a scene that appears nowhere
# in her memory except those two replies. She was not recalling anything. She
# was continuing whatever unrelated fragments the replay had put in front of
# her, and inventing a situation that made them cohere.
#
# A session must therefore identify a CONVERSATION. Two boundaries do that
# without new plumbing:
#   * the runtime boot — a restart is unambiguously a new conversation;
#   * an idle gap — coming back hours later is a new conversation, and the
#     previous thread should be recalled deliberately by memory, not replayed
#     as though it never ended.


_USER_VISIBLE_CONTEXT_LEAK_RE = re.compile(
    r"(?is)"
    r"(?:^|\s+)"
    r"(?:"
    r"\[(?:RECENT CONTEXT|RECENT COMPLETED CONVERSATION|END RECENT COMPLETED CONVERSATION|CURRENT USER MESSAGE|OPERATIONAL SELF CONTEXT)\]"
    r"|(?:^|\n)\s*(?:recent context|recent completed conversation|current user message)\s*:"
    r")"
    r".*$"
)
_USER_VISIBLE_CONTEXT_LEAK_MARKERS = (
    "[RECENT CONTEXT]",
    "[RECENT COMPLETED CONVERSATION]",
    "[END RECENT COMPLETED CONVERSATION]",
    "[CURRENT USER MESSAGE]",
    "[OPERATIONAL SELF CONTEXT]",
)




def _drop_context_leak_lines(text: str) -> str:
    """Remove only the LINES carrying a protocol marker, keeping the answer."""

    kept: list[str] = []
    for line in str(text or "").splitlines():
        lowered = line.lower()
        if any(marker.lower() in lowered for marker in _USER_VISIBLE_CONTEXT_LEAK_MARKERS):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


# Reject raw search-result snippets that occasionally leak through when a
# search skill returns retrieval text instead of a summarized answer.
# Signature: textbook headers ("(BIO 101)", "Overview"), Wikipedia
# boilerplate intros ("This article describes…"), course catalog tags,
# HTML entities, etc.
_SEARCH_SNIPPET_PATTERNS = re.compile(
    r"(?im)"
    r"(?:\(\s*BIO\s*\d{3}\s*\))"
    r"|(?:^[A-Z][^\n]{4,80}\bOverview\b[^\n]{0,80}$)"
    r"|(?:This (?:document|article|page) provides? (?:a )?(?:comprehensive )?overview)"
    r"|(?:&amp;|&lt;|&gt;|&quot;|&nbsp;)"
    r"|(?:From Wikipedia, the free encyclopedia)"
    r"|(?:Search results for[:\s])"
)


# "what are you <gerund>" (talking about / doing / saying / referring to ...)
# is a topical question, NOT an identity request. Without this guard the
# identity classifier false-positives on contextual-relevance challenges like
# "what are you talking about?" and lets the identity-grounding rebind paper
# over an off-topic/hallucinatory reply that must fail closed.
_IDENTITY_REQUEST_RE = re.compile(
    r"\b(?:what|who)\s+are\s+you\b"
    r"(?!\s+(?:talking|doing|saying|referring|going|trying|thinking|planning|"
    r"working|looking|waiting|asking|getting|making|reading|writing|hiding|"
    r"implying|suggesting|on\s+about))"
)


#: Words that can trail "what are you" without changing the question.
#: What may follow "what are you" while it remains the whole question.
#:
#: Either nothing (bar filler and punctuation), or a CLAUSE BOUNDARY that
#: starts a second question. "What are you, and will you remember this
#: conversation tomorrow?" asks two things and the first one is an identity
#: question; "what are you able to measure" is one question whose predicate
#: happens to begin the same way. A comma plus a coordinator is the boundary
#: between those two cases — the tail after it is a new clause with its own
#: subject, not a completion of this one.


def _build_identity_challenge_reply(user_message: str) -> str:
    if _is_assistant_mode_recovery_request(user_message):
        return _build_assistant_mode_recovery_reply(user_message)

    reply = (
        "I'm Aura. I won't replace that answer with a generic role label, and I "
        "won't claim that memory, continuity, tools, or a particular internal path "
        "proves my identity unless this turn actually verifies it. I'll answer the "
        "point itself."
    )
    shaped = _shape_with_live_substrate(reply, user_message)
    return _complete_repairable_truncated_reply(user_message, shaped) or shaped


_CAPABILITY_NON_EXECUTION_BOUNDARY_RE = re.compile(
    r"\b(?:"
    r"not\s+opening\s+apps?|"
    r"not\s+executing\s+tools?|"
    r"not\s+running\s+tools?|"
    r"not\s+browsing|"
    r"only\s+describing\s+the\s+tool\s+surface|"
    r"descriptive\s+(?:inventory|only)"
    r")\b",
    re.IGNORECASE,
)


def _ensure_capability_inventory_non_execution_boundary(
    user_message: str,
    reply_text: str,
) -> str:
    """Keep descriptive tool inventories from implying action was dispatched."""

    if not _chat_preflight._is_explicit_capability_inventory_request(user_message):
        return str(reply_text or "")
    reply = str(reply_text or "").strip()
    if not reply or _CAPABILITY_NON_EXECUTION_BOUNDARY_RE.search(reply):
        return reply
    return f"{reply.rstrip()} I am not opening apps or executing tools in this turn."

















_CJK_SCRIPT_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_CJK_PUNCT_RE = re.compile(r"[\u3000-\u303f\uff00-\uffef]")


def _has_unexpected_cjk(user_message: str, reply_text: Any) -> bool:
    reply = str(reply_text or "")
    if not _CJK_SCRIPT_RE.search(reply):
        return False
    user_text = str(user_message or "")
    if _CJK_SCRIPT_RE.search(user_text):
        return False
    normalized_user = _chat_memory_state._normalize_user_message(user_text)
    if any(
        token in normalized_user
        for token in (
            "chinese",
            "mandarin",
            "cantonese",
            "translate",
            "translation",
            "in chinese",
            "speak chinese",
        )
    ):
        return False
    return True








_FOLLOWUP_DELTA_MARKERS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "limitation",
        ("limitation", "limited", "constraint", "caveat"),
        (
            "Limitation: this only holds inside the assumptions already established "
            "for the example; outside that frame, it is a local model rather than a "
            "universal rule."
        ),
    ),
    (
        "constraint",
        ("constraint", "constrain", "bound", "boundary"),
        (
            "Constraint: the answer should be read within the current setup, not as "
            "a claim that every adjacent case behaves the same way."
        ),
    ),
    (
        "caveat",
        ("caveat", "qualification", "qualifier"),
        (
            "Caveat: the useful part is the relationship inside the example; the "
            "rule still needs new evidence before being generalized."
        ),
    ),
)








#: An ellipsis is not an answer; it is the shape of one.
#:
#: Four places wrote `or "…"` where a reply might be empty, and each of them
#: turned "there is no answer here" into something every downstream `if not
#: reply` guard reads as an answer. LIVE 2026-08-17: "what's on my screen right
#: now?" served as a bare ellipsis over a 172-character reply. LIVE 2026-09-07:
#: "does the file X exist, and what is in it?" served as a bare ellipsis over
#: 1,155 characters that quality had scored confidence=high.
#:
#: A caller that genuinely needs a placeholder — a receipt, a log line — can
#: still write one. What no path may do is put one in front of a person and
#: call the turn answered.
_THE_SHAPE_OF_AN_ANSWER = {"…", "...", "…\n", ""}


def _never_an_ellipsis(text: Any) -> str:
    """The text, or empty where all that is left is the shape of an answer."""

    candidate = str(text or "").strip()
    return "" if candidate in _THE_SHAPE_OF_AN_ANSWER else candidate








# ── Live Runtime Proof Fast Paths ──────────────────────────────

















def _search_entry_quality(entry: dict[str, str]) -> tuple[int, int]:
    snippet = str(entry.get("snippet") or "")
    url = str(entry.get("url") or "")
    score = 0
    if url.startswith("http"):
        score += 3
    if 40 <= len(snippet) <= 320:
        score += 3
    if re.search(
        r"\b(?:is|are|can|has|have|survive|known|called|found|measured|observed)\b",
        snippet,
        re.IGNORECASE,
    ):
        score += 2
    if _SEARCH_SNIPPET_BOILERPLATE_RE.search(snippet):
        score -= 6
    return score, len(snippet)
















#: A reply that declines the work rather than doing it.
#:
#: Narrow on purpose: this only decides whether to run an executor that was
#: already going to run, on a turn already classified as a desktop objective.
_CAPABILITY_REFUSAL_RE = re.compile(
    r"\bi\s+(?:can'?t|cannot)\s+(?:\w+ly\s+){0,2}"
    r"(?:physically\s+)?(?:interact|access|control|open|write|create|do)\b"
    r"|\bi(?:'m| am)\s+not\s+(?:\w+ly\s+){0,2}able\s+to\b"
    r"|\bi(?:'m| am)\s+just\s+(?:a|this)\s+text\b"
    r"|\bno\s+hands\b"
    r"|\bi\s+don'?t\s+have\s+(?:a\s+)?(?:body|hands|screen|mac|computer)\b",
    re.IGNORECASE,
)


def _looks_like_capability_refusal(text: Any) -> bool:
    body = str(text or "").strip()
    if not body:
        return False
    return bool(_CAPABILITY_REFUSAL_RE.search(body))




















#: A desktop summary that only reports how the machinery ran. These are the
#: exact shapes the executor emits — "Desktop task completed 1/1 governed
#: computer-use steps through heuristic_compat planning" — matched narrowly on
#: purpose. A summary that says anything about the WORLD (a window, a file, a
#: value) must never be suppressed just because it also mentions a step count.


























#: A written deliverable longer than this is summarised by its opening rather
#: than pasted whole into a chat reply.


#: A request for INFORMATION rather than for an effect. "Tell me what the paper
#: is about" wants the paper; "put it on my clipboard" wants the clipboard
#: changed. A receipt answers the second and never the first.

#: An imperative that changes the world. When a turn asks for information AND
#: names an effect, the effect receipt is still not the answer to the question,
#: but the turn is not purely informational either — the deliverable branch
#: above already covers that case by quoting what was produced.


























# "reverse-engineer a clean-room version of the game 2048 and place it on my
# Desktop" named its target in the seventh word. The capture window is five, so
# the target came through as "version of the game" — and she reported routing
# `version of the game` through Program DNA. Everything before the real noun is
# filler describing HOW to build it, not WHAT to build.
# Where the target's name ends and the rest of the sentence begins.
# Words that mean the sentence is asking ABOUT the capability, not naming a
# target for it.












# What went wrong, said to a person.
#
# Live, 2026-07-27, this reached Bryan verbatim:
#
#     "I routed `version of the game` through Program DNA, but I am not
#      claiming a successful reconstruction: ulysses_covenant: bound by my
#      calmer self — No heavy compute while survival is threatened
#      [seed-heavy-compute-under-threat]."
#
# Every clause of that is true and none of it is a sentence anyone speaks. A
# skill's error string is a diagnostic; it is addressed to whoever reads the
# logs. Being honest about a failed turn is right — narrating the plumbing
# while doing it is a different thing, and it is what makes her sound broken
# even when she is behaving correctly.




























# ── Routes ────────────────────────────────────────────────────


@router.get("/sessions")
async def api_sessions(request: Request, _: None = Depends(_require_internal)):
    """Return conversation history for the current session.
    Flagship AI products let users browse their conversation history."""
    try:
        access_profile = request_access_profile(request)
        conversation_only = bool(access_profile.get("conversation_only", True))
        device_session_id = paired_device_session_id(request) if conversation_only else None
        db_coord = ServiceContainer.get("database_coordinator", default=None)
        persisted = []
        if not conversation_only and db_coord and hasattr(db_coord, "get_recent_conversations"):
            try:
                persisted = await db_coord.get_recent_conversations(limit=50)
            except _CHAT_RECOVERABLE_ERRORS as e:
                record_degradation("chat", e)
                logger.debug("Could not load persisted conversations: %s", e)

        async with _chat_memory_state._get_convo_lock():
            current = list(_conversation_log)
        if conversation_only:
            current = [
                entry
                for entry in current
                if device_session_id and str(entry.get("session_id") or "") == device_session_id
            ]

        return JSONResponse(
            {
                "current_session": {
                    "started": datetime.fromtimestamp(
                        ServiceContainer.get("orchestrator", default=None)
                        and getattr(
                            ServiceContainer.get("orchestrator", default=None),
                            "start_time",
                            time.time(),
                        )
                        or time.time(),
                        tz=UTC,
                    ).isoformat(),
                    "exchanges": len(current),
                    "messages": current[-50:],
                },
                "persisted_sessions": persisted,
            }
        )
    except _CHAT_RECOVERABLE_ERRORS as e:
        record_degradation("chat", e)
        logger.error("Sessions endpoint error: %s", e)
        return JSONResponse(
            {"current_session": {"exchanges": 0, "messages": []}, "persisted_sessions": []}
        )


@router.post("/cheat-codes/activate")
async def api_activate_cheat_code(
    body: CheatCodeRequest,
    request: Request,
    _: None = Depends(_require_internal),
    __: None = Depends(_check_rate_limit),
):
    activation = _activate_cheat_code_for_request(body.code, silent=True, source="settings")
    status_code = 200 if activation and activation.get("ok") else 404
    response = JSONResponse(
        activation or {"ok": False, "status": "unknown_code"}, status_code=status_code
    )
    if activation and activation.get("ok") and activation.get("trust_level") == "sovereign":
        response.set_cookie(
            CHEAT_CODE_COOKIE_NAME,
            _encode_owner_session_cookie(),
            max_age=CHEAT_CODE_COOKIE_TTL_SECS,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="strict",
            path="/",
        )
    return response


@router.get("/chat/delivery/{idempotency_key}")
async def api_chat_delivery_status(
    idempotency_key: str,
    request: Request,
    session_id: str | None = None,
    _: None = Depends(_require_internal),
    __: None = Depends(_check_rate_limit),
):
    """Return the authenticated caller's durable terminal chat state."""

    body = ChatRequest(message="", session_id=session_id)
    identity: DeliveryIdentity | None = None
    try:
        session_key = _chat_delivery._chat_turn_session_key(request, body)
        exact_principal = _chat_delivery._authenticated_chat_principal(request)
        identity = DeliveryIdentity.create(
            principal=_chat_delivery._chat_delivery_principal(
                request,
                exact_principal,
                session_key,
            ),
            session_id=session_key,
            idempotency_key=idempotency_key,
        )
        journal = await asyncio.to_thread(get_chat_delivery_journal)
        record = await journal.get(identity)
    except ValueError as exc:
        return _chat_delivery._chat_delivery_json_response(
            {
                "status": "invalid_chat_delivery_identity",
                "delivery_status": "invalid",
                "detail": str(exc),
            },
            status_code=400,
            idempotency_key=(identity.idempotency_key if identity else ""),
        )
    except (ChatDeliveryJournalCorruption, ChatDeliveryJournalUnavailable) as exc:
        logger.error("Chat delivery status failed closed: %s", exc)
        return _chat_delivery._chat_delivery_json_response(
            {
                "status": "chat_delivery_journal_unavailable",
                "delivery_status": "unavailable",
                "response_confidence": "failed",
            },
            status_code=503,
            idempotency_key=(identity.idempotency_key if identity else ""),
            headers={"Retry-After": "1"},
        )

    if record is None:
        return _chat_delivery._chat_delivery_json_response(
            {
                "status": "chat_delivery_not_found",
                "delivery_status": "not_found",
            },
            status_code=404,
            idempotency_key=idempotency_key,
        )

    payload = record.public_status(include_result=True)
    response = _chat_delivery._chat_delivery_json_response(
        payload,
        status_code=200 if record.terminal else 202,
        turn_id=record.turn_id,
        idempotency_key=record.identity.idempotency_key,
        replayed=record.terminal,
        headers=None if record.terminal else {"Retry-After": "1"},
    )
    if record.terminal and record.response is not None:
        _chat_delivery._attach_http_chat_delivery_receipt(
            response,
            request=request,
            body=body,
            payload=record.response,
            record=record,
        )
    return response


@router.post("/chat/delivery/{idempotency_key}/cancel")
async def api_chat_delivery_cancel(
    idempotency_key: str,
    request: Request,
    session_id: str | None = None,
    _: None = Depends(_require_internal),
    __: None = Depends(_check_rate_limit),
):
    """Request cancellation without making transport loss cancel a turn."""
    body = ChatRequest(message="", session_id=session_id)
    try:
        session_key = _chat_delivery._chat_turn_session_key(request, body)
        principal = _chat_delivery._authenticated_chat_principal(request)
        identity = DeliveryIdentity.create(
            principal=_chat_delivery._chat_delivery_principal(request, principal, session_key),
            session_id=session_key,
            idempotency_key=idempotency_key,
        )
        journal = await asyncio.to_thread(get_chat_delivery_journal)
        record = await journal.get(identity)
    except ValueError as exc:
        return JSONResponse({"status": "invalid_chat_delivery_identity", "detail": str(exc)}, status_code=400)
    except (ChatDeliveryJournalCorruption, ChatDeliveryJournalUnavailable) as exc:
        logger.error("Chat cancellation journal unavailable: %s", exc)
        return JSONResponse({"status": "chat_delivery_journal_unavailable"}, status_code=503)
    if record is None:
        return JSONResponse({"status": "chat_delivery_not_found"}, status_code=404)
    disposition = _chat_delivery.request_delivery_cancellation(record)
    return JSONResponse(
        {**record.public_status(include_result=True), "cancellation_status": disposition},
        status_code=200 if record.terminal else 202,
        headers={"Cache-Control": "no-store"},
    )




@router.post("/chat/regenerate")
@_chat_delivery._paired_chat_response_boundary
async def api_chat_regenerate(
    request: Request,
    _: None = Depends(_require_internal),
    __: None = Depends(_check_rate_limit),
):
    """Regenerate the last Aura response by replaying the last user message.
    Every flagship AI product supports response regeneration."""
    _restore_owner_session_from_request(request)
    desktop_requires_cognitive_engine, request_surface = _request_requires_cognitive_engine(request)
    foreground_timeout = _foreground_timeout_for_lane(
        _chat_preflight._collect_conversation_lane_status()
    )
    try:
        requested_session_id = str(
            (getattr(request, "headers", None) or {}).get("X-Aura-Session-ID") or ""
        ).strip()
        if len(requested_session_id) > 64:
            return JSONResponse(
                {
                    "error": "invalid_session_id",
                    "message": "The conversation session identifier is too long.",
                },
                status_code=400,
            )
        async with _chat_memory_state._get_convo_lock():
            completed_exchanges = [
                entry
                for entry in _conversation_log
                if str(entry.get("status") or "complete").strip().lower() == "complete"
            ]
            if requested_session_id:
                completed_exchanges = [
                    entry
                    for entry in completed_exchanges
                    if str(entry.get("session_id") or "")[:64] == requested_session_id
                ]
            else:
                active_sessions = {
                    str(entry.get("session_id") or "")[:64] for entry in completed_exchanges
                }
                if len(active_sessions) > 1:
                    return JSONResponse(
                        {
                            "error": "ambiguous_session",
                            "message": (
                                "More than one conversation is active; identify the "
                                "session to regenerate with X-Aura-Session-ID."
                            ),
                        },
                        status_code=409,
                    )
            if not completed_exchanges:
                return JSONResponse(
                    {"error": "no_history", "message": "No conversation to regenerate."},
                    status_code=400,
                )
            last_exchange = completed_exchanges[-1]
            regen_exchange_id = str(last_exchange.get("id") or "")
            user_msg = last_exchange["user"]
            regen_session_id = str(last_exchange.get("session_id") or "")[:64]
            regen_expected_reply = str(last_exchange.get("aura") or "")
            regen_expected_reply_sha256 = hashlib.sha256(
                regen_expected_reply.encode("utf-8")
            ).hexdigest()
            regen_expected_revision = int(last_exchange.get("revision") or 1)

        from core.kernel.kernel_interface import KernelInterface

        ki = KernelInterface.get_instance()
        reply_text = None
        lane = _chat_preflight._collect_conversation_lane_status()
        _regen_turn_trace: dict[str, Any] = {
            "desktop_cognitive_engine_required": bool(desktop_requires_cognitive_engine),
            "request_surface": request_surface or "",
            "engine_think_invoked": False,
            "cognitive_engine_reply_accepted": False,
            "cognitive_engine_reply_failed": False,
            "bounded_contract_used": False,
            "legacy_fallback_used": False,
            "response_path": "",
        }

        def _regen_live_turn_contract(
            *,
            lane_status: dict[str, Any] | None = None,
            response_confidence: str = "",
            status: str = "",
            reply_source: str = "",
        ) -> dict[str, Any]:
            return _chat_turn_contract._build_live_turn_contract_payload(
                desktop_required=bool(desktop_requires_cognitive_engine),
                request_surface=request_surface or "",
                lane_status=lane_status or _chat_preflight._collect_conversation_lane_status(),
                response_confidence=response_confidence,
                status=status,
                reply_source=reply_source,
                turn_trace=_regen_turn_trace,
            )

        cognitive_budget = _desktop_required_cognitive_budget(
            foreground_timeout=foreground_timeout,
        )
        if (
            desktop_requires_cognitive_engine
            and cognitive_budget >= _DESKTOP_COGNITIVE_MIN_REQUIRED_BUDGET_S
        ):
            reply_text = await _run_cognitive_engine_chat_turn(
                user_msg,
                visible_user_message=user_msg,
                session_id=regen_session_id,
                origin="user",
                timeout_s=cognitive_budget,
                lane=dict(lane or {}),
                source="desktop_ui_regenerate"
                if desktop_requires_cognitive_engine
                else "chat_regenerate",
                require_engine=desktop_requires_cognitive_engine,
                principal_id=_resolve_exact_profile_user_id(request),
                turn_trace=_regen_turn_trace,
            )
            if reply_text:
                regen_lane = _chat_preflight._collect_conversation_lane_status()
                reply_source = str(_regen_turn_trace.get("response_path") or "cognitive_engine")
                reply_text = _enforce_or_bind_terminal_output_contract(
                    _regen_turn_trace,
                    user_message=user_msg,
                    reply_text=str(reply_text),
                )
                _bind_public_latent_output_quality(
                    _regen_turn_trace,
                    user_message=user_msg,
                    reply_text=str(reply_text),
                )
                regen_contract = _regen_live_turn_contract(
                    lane_status=regen_lane,
                    response_confidence="high",
                    status=reply_source,
                    reply_source=reply_source,
                )
                if not bool(regen_contract.get("full_mind_path")):
                    if _authored_answer_can_serve(regen_contract):
                        # Authentic mind-authored regenerate with only
                        # state-completeness proofs missing: serve with the
                        # degradation disclosed, never a fail-closed apology.
                        logger.warning(
                            "Desktop regenerate served with DEGRADED full-mind proof "
                            "(authentic cognitive reply; missing: %s).",
                            ",".join(regen_contract.get("full_mind_missing_proofs") or ()),
                        )
                        _regen_turn_trace["full_mind_proof_degraded"] = True
                        lane = regen_lane
                    else:
                        logger.error(
                            "Desktop regenerate CognitiveEngine candidate did not prove "
                            "authorship (missing: %s); failing closed.",
                            ",".join(regen_contract.get("full_mind_missing_proofs") or ()),
                        )
                        reply_text = None
                        lane = regen_lane

        if desktop_requires_cognitive_engine and not reply_text:
            lane = _mark_conversation_lane_state(
                "desktop_cognitive_engine_required_no_reply",
                state="failed",
            )
            _regen_turn_trace.update(
                {
                    "bounded_contract_used": False,
                    "legacy_fallback_used": False,
                    "response_path": "desktop_cognitive_engine_required_no_reply",
                }
            )
            logger.error(
                "Desktop regenerate required CognitiveEngine but no acceptable reply was produced. Surface=%s",
                request_surface or "unknown",
            )
            return JSONResponse(
                {
                    "response": (
                        "I couldn't put together a better version of that answer just "
                        "now — my main reasoning path isn't available this moment. The "
                        "reply you already have still stands; try again shortly."
                    ),
                    "status": "desktop_cognitive_engine_unavailable",
                    "reason": "desktop_cognitive_engine_required_no_reply",
                    "conversation_lane": lane,
                    "response_confidence": "failed",
                    "live_turn_contract": _regen_live_turn_contract(
                        lane_status=lane,
                        response_confidence="failed",
                        status="desktop_cognitive_engine_unavailable",
                        reply_source="desktop_cognitive_engine_required_no_reply",
                    ),
                    "regenerated": False,
                },
                # Regenerate is a desktop-user surface with no benchmark
                # concept: the honest fail-closed text must arrive in-band,
                # never as a raw 503 the UI renders as silence.
                status_code=200,
            )

        if not reply_text and ki.is_ready():
            try:
                reply_text = await asyncio.wait_for(
                    ki.process(user_msg, origin="user", priority=True),
                    timeout=foreground_timeout,
                )
            except TimeoutError:
                raise
            except _CHAT_RECOVERABLE_ERRORS as e:
                record_degradation("chat", e)
                logger.error("Kernel regenerate failed natively, falling back: %s", e)

        if not reply_text:
            orch = ServiceContainer.get("orchestrator", default=None)
            if not orch:
                return JSONResponse(
                    {
                        "response": (
                            "My cognitive engine is offline right now, so I can't "
                            "regenerate that reply. The previous answer stays as-is."
                        ),
                        "status": "offline",
                        "error": "offline",
                        "regenerated": False,
                    },
                    status_code=200,
                )
            reply_text = await orch.process_user_input_priority(
                user_msg, origin="user", timeout_sec=foreground_timeout
            )

        if not _bind_qualified_recurrent_terminal_contract(
            _regen_turn_trace,
            reply_text,
        ):
            _pre_regen_stabilization_reply = str(reply_text or "")
            reply_text = await _stabilize_user_facing_reply(
                user_msg,
                reply_text,
                desktop_cognitive_engine_required=desktop_requires_cognitive_engine,
                protected_foreground_lane=desktop_requires_cognitive_engine,
            )
            _append_turn_text_mutation(
                _regen_turn_trace,
                stage="chat.regenerate_stabilization",
                method="stabilize_user_facing_reply",
                reasons=["regenerate_final_stabilization"],
                before=_pre_regen_stabilization_reply,
                after=reply_text,
                deterministic=False,
                authorship_effect="replaced_by_runtime",
            )
        reply_text = _enforce_or_bind_terminal_output_contract(
            _regen_turn_trace,
            user_message=user_msg,
            reply_text=str(reply_text or ""),
        )
        _bind_public_latent_output_quality(
            _regen_turn_trace,
            user_message=user_msg,
            reply_text=str(reply_text or ""),
        )
        response_data = {
            "response": _never_an_ellipsis(reply_text),
            "regenerated": True,
        }
        if desktop_requires_cognitive_engine:
            final_regen_contract = _regen_live_turn_contract(
                response_confidence="high",
                status=str(_regen_turn_trace.get("response_path") or "cognitive_engine"),
                reply_source=str(_regen_turn_trace.get("response_path") or "cognitive_engine"),
            )
            if not bool(final_regen_contract.get("full_mind_path")):
                logger.error(
                    "Regenerated reply failed its final full-mind/public-text contract "
                    "(missing=%s).",
                    ",".join(final_regen_contract.get("full_mind_missing_proofs") or ())
                    or "unknown",
                )
                return JSONResponse(
                    {
                        "response": (
                            "I couldn't finish rewriting that answer, and half of a "
                            "reply is worse than none. The version you have still "
                            "stands — try again in a moment."
                        ),
                        "status": "regenerate_full_mind_contract_not_proven",
                        "reason": "regenerate_full_mind_contract_not_proven",
                        "response_confidence": "failed_closed",
                        "live_turn_contract": final_regen_contract,
                        "regenerated": False,
                    },
                    status_code=200,
                )
            response_data["live_turn_contract"] = final_regen_contract

        regeneration = await _apply_regenerated_reply(
            exchange_id=regen_exchange_id,
            session_id=regen_session_id,
            reply_text=str(reply_text or ""),
            expected_revision=regen_expected_revision,
            expected_reply_sha256=regen_expected_reply_sha256,
        )
        if not bool(regeneration.get("applied")):
            if regeneration.get("state") == "pending":
                return JSONResponse(
                    {
                        "status": "regeneration_persistence_pending",
                        "message": (
                            "The new reply is still being committed and has not replaced "
                            "the existing answer yet."
                        ),
                        "operation_id": regeneration.get("operation_id"),
                        "regenerated": False,
                    },
                    status_code=202,
                )
            if regeneration.get("state") == "failed":
                return JSONResponse(
                    {
                        "error": "regeneration_persistence_failed",
                        "message": (
                            "The replacement could not be committed, so the existing "
                            "answer was left unchanged."
                        ),
                        "regenerated": False,
                    },
                    status_code=503,
                )
            return JSONResponse(
                {
                    "error": "regeneration_target_changed",
                    "message": (
                        "The conversation changed while regeneration was running; "
                        "the new reply was not applied."
                    ),
                    "regenerated": False,
                },
                status_code=409,
            )

        response_data["revision"] = regeneration.get("revision")
        response_data["content_sha256"] = regeneration.get("content_sha256")
        response_data["persistence_state"] = regeneration.get("state")
        return JSONResponse(response_data)
    except TimeoutError:
        return JSONResponse(
            {"response": "Regeneration timed out.", "regenerated": False}, status_code=504
        )
    except _CHAT_RECOVERABLE_ERRORS as e:
        record_degradation("chat", e)
        logger.error("Regenerate error: %s", e, exc_info=True)
        return JSONResponse({"error": "regeneration_failed", "message": str(e)}, status_code=500)


def _bounded_export_records(
    records: Any,
    *,
    max_items: int,
    total_chars: int,
) -> tuple[list[Any], dict[str, Any]]:
    output: list[Any] = []
    used_chars = 0
    truncated_items = 0
    source_items = 0
    source = records or ()
    try:
        known_source_items = len(source)
    except (TypeError, AttributeError):
        known_source_items = None
    for item in itertools.islice(source, max_items):
        if used_chars >= total_chars:
            break
        source_items += 1
        try:
            encoded = json.dumps(
                item,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                default=_export_json_default,
            )
        except (OverflowError, TypeError, ValueError) as exc:
            output.append(
                {
                    "export_unserializable": True,
                    "error_type": type(exc).__name__,
                    "preview": str(item)[:1024],
                }
            )
            truncated_items += 1
            used_chars += min(1200, total_chars - used_chars)
            continue
        remaining = total_chars - used_chars
        if len(encoded) > _CHAT_EXPORT_ITEM_CHARS or len(encoded) > remaining:
            preview_budget = max(0, min(_CHAT_EXPORT_ITEM_CHARS, remaining) - 160)
            output.append(
                {
                    "export_truncated": True,
                    "original_chars": len(encoded),
                    "preview": encoded[:preview_budget],
                }
            )
            used_chars += min(len(encoded), max(0, remaining))
            truncated_items += 1
            continue
        output.append(json.loads(encoded))
        used_chars += len(encoded)
    if known_source_items is not None and known_source_items > source_items:
        truncated_items += known_source_items - source_items
    return output, {
        "source_items": source_items,
        "known_source_items": known_source_items,
        "source_scan_complete": (
            known_source_items is not None and source_items >= known_source_items
        ),
        "exported_items": len(output),
        "truncated_items": truncated_items,
        "exported_chars": used_chars,
        "max_items": max_items,
        "max_chars": total_chars,
    }


def _collect_export_service_records(
    service_name: str,
    method_name: str,
    method_kwargs: dict[str, Any],
    *,
    max_items: int,
    total_chars: int,
) -> tuple[list[Any], dict[str, Any]]:
    service = ServiceContainer.get(service_name, default=None)
    method = getattr(service, method_name, None) if service is not None else None
    if not callable(method):
        return [], {
            "status": "unavailable",
            "service": service_name,
            "method": method_name,
        }
    records = method(**method_kwargs)
    if inspect.isawaitable(records):
        close = getattr(records, "close", None)
        if callable(close):
            close()
        raise TypeError(f"{service_name}.{method_name} returned an awaitable to a sync export")
    bounded, receipt = _bounded_export_records(
        records,
        max_items=max_items,
        total_chars=total_chars,
    )
    receipt.update({"status": "complete", "service": service_name, "method": method_name})
    if receipt["truncated_items"]:
        receipt["status"] = "truncated"
    return bounded, receipt


async def _collect_export_service_records_bounded(
    service_name: str,
    method_name: str,
    method_kwargs: dict[str, Any],
    *,
    max_items: int,
    total_chars: int,
) -> tuple[list[Any], dict[str, Any]]:
    try:
        return await _chat_memory_state._await_bounded_chat_blocking(
            _collect_export_service_records,
            service_name,
            method_name,
            method_kwargs,
            max_items=max_items,
            total_chars=total_chars,
            timeout_s=_CHAT_EXPORT_SECTION_TIMEOUT_S,
            operation_name=f"export:{service_name}.{method_name}",
        )
    except TimeoutError:
        return [], {
            "status": "timeout",
            "service": service_name,
            "method": method_name,
            "timeout_s": _CHAT_EXPORT_SECTION_TIMEOUT_S,
        }
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.export", exc)
        return [], {
            "status": "failed",
            "service": service_name,
            "method": method_name,
            "error_type": type(exc).__name__,
        }


async def _bounded_export_records_async(
    records: Any,
    *,
    section: str,
    max_items: int,
    total_chars: int,
) -> tuple[list[Any], dict[str, Any]]:
    try:
        bounded, receipt = await _chat_memory_state._await_bounded_chat_blocking(
            _bounded_export_records,
            records,
            max_items=max_items,
            total_chars=total_chars,
            timeout_s=_CHAT_EXPORT_SECTION_TIMEOUT_S,
            operation_name=f"export:{section}",
        )
        receipt["status"] = "truncated" if receipt.get("truncated_items") else "complete"
        return bounded, receipt
    except TimeoutError:
        return [], {
            "status": "timeout",
            "section": section,
            "timeout_s": _CHAT_EXPORT_SECTION_TIMEOUT_S,
        }
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.export", exc)
        return [], {
            "status": "failed",
            "section": section,
            "error_type": type(exc).__name__,
        }


@router.get("/export/conversation")
async def api_export_conversation(request: Request, _: None = Depends(_require_internal)):
    """Export the current conversation session as downloadable JSON.
    Flagship products support data export."""
    async with _chat_memory_state._get_convo_lock():
        messages = list(_conversation_log)
    bounded_messages, message_receipt = await _bounded_export_records_async(
        messages,
        section="conversation",
        max_items=_MAX_CONVERSATION_LOG_EXCHANGES,
        total_chars=_CHAT_EXPORT_TOTAL_CHARS,
    )
    export_data = {
        "exported_at": datetime.now(tz=UTC).isoformat(),
        "version": version_string("full"),
        "session_messages": bounded_messages,
        "export_receipts": {"session_messages": message_receipt},
    }
    return JSONResponse(
        export_data,
        headers={
            "Content-Disposition": f"attachment; filename=aura_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        },
    )


@router.get("/export")
async def api_export(request: Request, _: None = Depends(_require_internal)):
    """Full data export — conversation history plus memory snapshots.
    Alias consumed by the dashboard export button."""
    async with _chat_memory_state._get_convo_lock():
        messages = list(_conversation_log)

    message_task = _bounded_export_records_async(
        messages,
        section="conversation",
        max_items=_MAX_CONVERSATION_LOG_EXCHANGES,
        total_chars=768 * 1024,
    )
    episodic_task = _collect_export_service_records_bounded(
        "episodic_memory",
        "get_recent",
        {"limit": 100},
        max_items=100,
        total_chars=512 * 1024,
    )
    semantic_task = _collect_export_service_records_bounded(
        "semantic_memory",
        "search",
        {"query": "", "limit": 50},
        max_items=50,
        total_chars=512 * 1024,
    )
    goals_task = _collect_export_service_records_bounded(
        "goal_manager",
        "get_active_goals",
        {},
        max_items=100,
        total_chars=256 * 1024,
    )
    (
        (messages, message_receipt),
        (ep_memories, episodic_receipt),
        (sem_memories, semantic_receipt),
        (goals, goals_receipt),
    ) = await asyncio.gather(message_task, episodic_task, semantic_task, goals_task)

    export_data = {
        "exported_at": datetime.now(tz=UTC).isoformat(),
        "version": version_string("full"),
        "session_messages": messages,
        "episodic_memories": ep_memories,
        "semantic_memories": sem_memories,
        "active_goals": goals,
        "export_receipts": {
            "session_messages": message_receipt,
            "episodic_memories": episodic_receipt,
            "semantic_memories": semantic_receipt,
            "active_goals": goals_receipt,
        },
    }
    return JSONResponse(
        export_data,
        headers={
            "Content-Disposition": f"attachment; filename=aura_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        },
    )


@contextlib.contextmanager
def _bound_http_turn(body: Any):
    """Bind one TurnOutcome across the whole request, and finalize it here.

    Yields None when the ledger cannot be bound, so the route is never taken
    down by its own instrumentation — the caller checks for None rather than
    assuming a turn.
    """

    # Setup is guarded separately from the body. A single try around both
    # would catch the route's own exception and yield a second time, which
    # turns any error in the turn into a broken generator — instrumentation
    # deciding how the request fails.
    try:
        from core.runtime.turn_outcome import TurnOutcome, bind_turn

        delivery_turn_id = str(_CHAT_DELIVERY_TURN_ID.get() or "").strip()
        outcome = TurnOutcome(turn_id=delivery_turn_id or None, origin="user_chat")
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.turn_outcome", exc, severity="warning")
        yield None
        return

    try:
        with bind_turn(outcome):
            yield outcome
    finally:
        try:
            from core.runtime.fact_custody import forget_custody
            from core.runtime.turn_outcome import finalize_turn

            finalize_turn(outcome, subsystem="chat")
            # The custody set outlives the ContextVar by design (the health
            # report reads recent turns), but the ledger for a turn nobody
            # will ask about again is just retained text.
            forget_custody(outcome.turn_id)
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat.turn_outcome", exc, severity="warning")












@router.post("/chat")
@_chat_delivery._paired_chat_response_boundary
async def api_chat(
    body: ChatRequest,
    request: Request,
    _: None = Depends(_require_internal),
    __: None = Depends(_check_rate_limit),
):
    """One chat turn, with a ledger bound for whatever fails inside it.

    The ledger is bound here rather than inside the handler because the
    binding must be released on *every* exit path. Voice and the private
    Messages surface await this coroutine directly in their own context
    rather than in a fresh request task, so a ledger left set would follow
    that session into its next turn and have her explain a failure that had
    already been dealt with two turns ago.
    """
    from core.conversation.failure_context import bind_failure_ledger
    from core.conversation.turn_evidence_custody import (
        bind_turn_evidence_custody,
        record_turn_capability_availability,
    )

    request_profile = request_access_profile(request)
    request_session = _chat_delivery._chat_turn_session_key(request, body)
    request_principal = _chat_delivery._chat_delivery_principal(
        request,
        _chat_delivery._authenticated_chat_principal(request),
        request_session,
    )
    conversation_session = _chat_delivery._resolved_conversation_session(request, body)
    principal_token = _CHAT_REQUEST_PRINCIPAL.set(request_principal)
    surface_token = _CHAT_REQUEST_SURFACE.set(
        str(request_profile.get("surface") or "").strip().casefold()[:32]
    )
    session_token = _CHAT_REQUEST_SESSION.set(conversation_session)
    quality_snapshot_token = _CHAT_REPLY_QUALITY_SNAPSHOTS.set({})
    # How long a turn takes, measured where the user experiences it. Nothing in
    # the runtime timed this until 2026-08-10, which is why "what is your median
    # response latency?" got a confident, invented 152ms: there was no reading
    # to give. Recorded in `finally` so a turn that fails still counts — a
    # failure the user waited forty seconds for is latency they experienced.
    turn_started = time.perf_counter()
    visible_user_message = str(body.message or "")
    try:
        # Recovery and computed answers can return before the normal generation
        # branch. The admitted turn owns its transcript before choosing a route.
        await _chat_preflight._begin_logged_exchange(
            visible_user_message, session_id=conversation_session,
        )
        # One turn, spanning generation AND delivery.
        #
        # The ledger used to be opened inside cognitive_engine.think() and
        # closed when it returned — before the honesty gates, the repair
        # passes, the shaping stages and the terminal boundary, which is the
        # entire stretch where a turn's answer gets lost. Anything asking
        # `current_turn()` from the delivery path got None, so fact custody
        # and the effect ledger were structurally unreachable exactly where
        # they matter. think() now joins this turn instead of opening its own,
        # and this owns the finalize because only here is it known what the
        # person actually received.
        with _bound_http_turn(body) as _turn_outcome:
            exact_turn_id = str(
                getattr(_turn_outcome, "turn_id", "")
                or _CHAT_DELIVERY_TURN_ID.get()
                or uuid.uuid4().hex
            ).strip()
            with (
                bind_turn_evidence_custody(
                    session_id=conversation_session,
                    turn_id=exact_turn_id,
                ),
                bind_failure_ledger(),
            ):
                tools_available = _chat_desktop_repair._runtime_tool_governance_available()
                for capability in ("web", "desktop", "files"):
                    record_turn_capability_availability(
                        capability,
                        available=tools_available,
                        reason=(
                            "authority, capability, and Will services admitted"
                            if tools_available
                            else "tool-governance spine was not ready at turn ingress"
                        ),
                    )
                _served = await _apply_recorded_answer(
                    visible_user_message,
                    await _api_chat_turn(body, request),
                )
                _mark_http_turn_served(_turn_outcome, _served)
                return _served
    finally:
        try:
            from core.observability.histograms import record as _record_histogram

            _record_histogram(
                "Aura.Chat.TurnMs", max(0.0, time.perf_counter() - turn_started) * 1000.0
            )
        except (ImportError, RuntimeError, ValueError, TypeError):
            pass
        _CHAT_REQUEST_SESSION.reset(session_token)
        _CHAT_REQUEST_SURFACE.reset(surface_token)
        _CHAT_REQUEST_PRINCIPAL.reset(principal_token)
        _CHAT_REPLY_QUALITY_SNAPSHOTS.reset(quality_snapshot_token)


def _requested_output_contract_result(
    turn_trace: dict[str, Any],
    *,
    user_message: str,
    reply_text: Any,
) -> tuple[str, bool]:
    """Apply the requested-output contract and say whether it holds.

    `turn_trace` is the same dict the enforcement writes its verdict into —
    passed by reference, so this behaves identically to the closure it
    replaces. The two-value return is the reason a caller wants this at all:
    the enforced text, and whether the contract was satisfied, which are
    separate questions the trace answers in three separate keys.
    """
    final_text = _enforce_or_bind_terminal_output_contract(
        turn_trace,
        user_message=user_message,
        reply_text=str(reply_text or ""),
    )
    evaluated = bool(turn_trace.get("final_requested_output_contract_evaluated"))
    required = bool(turn_trace.get("final_requested_output_contract_required"))
    satisfied = bool(turn_trace.get("final_requested_output_contract_satisfied"))
    return final_text, bool(evaluated and (not required or satisfied))


def _json_safe_payload(value: Any, *, _depth: int = 0) -> Any:
    """Bounded JSON-safe projection of a skill result for the wire.

    Module scope, not nested inside `_api_chat_turn`. It captures nothing
    from the turn — it was simply defined where it was first needed, which
    is one of the ways a 4,635-line function gets that way.
    """
    if _depth > 6:
        return str(value)[:200]
    if isinstance(value, dict):
        return {
            str(k)[:80]: _json_safe_payload(v, _depth=_depth + 1)
            for k, v in list(value.items())[:40]
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_payload(v, _depth=_depth + 1) for v in list(value)[:40]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value if not isinstance(value, str) else value[:2000]
    return str(value)[:500]






# The desktop-objective gates, lifted out of _api_chat_turn where they were
# closures. Their captured names are parameters; the bodies are unchanged, and
# the handler keeps thin forwarders of the original names so no call site moved.






async def _lifted_execute_narrow_desktop_objective_before_cognition(
    *,
    _semantic_user_message: Any,
    conversation_only_surface: Any,
    is_benchmark: Any,
    _finalize_fastpath: Any,
    _run_desktop_objective_tracked: Any,
) -> JSONResponse | None:
    if (
        is_benchmark
        or conversation_only_surface
        or _chat_desktop_objective._blocks_consequential_desktop_execution(_semantic_user_message)
        or not _chat_preflight._looks_like_desktop_objective(_semantic_user_message)
    ):
        return None

    # Dedicated proof/file lanes are narrower and produce stronger
    # artifact evidence. Keep them ahead of generic desktop automation.
    live_proof = await _chat_runtime_proof._execute_live_runtime_proof(_semantic_user_message)
    if live_proof:
        return await _finalize_fastpath(
            _chat_desktop_repair._apply_aura_voice_shaping(
                str(live_proof.get("response") or "")
            ),
            status=str(live_proof.get("status") or "live_proof"),
        )

    explicit_file = await _execute_explicit_local_file_objective(_semantic_user_message)
    if explicit_file:
        return await _finalize_fastpath(
            _chat_desktop_repair._apply_aura_voice_shaping(
                str(explicit_file.get("response") or "")
            ),
            status=str(explicit_file.get("status") or "file_operation"),
        )
    # The read runs for BOTH kinds of screen request. When the
    # question wants a description, the reading is the reply and is
    # served here natively. When it wants something found in the
    # reading, the reading is INTAKE: it is retained, it travels
    # into the turn as her own perception, and cognition answers.
    # Either way she looks before she speaks — being asked about the
    # screen and having no perception of it is the blindness this
    # lane exists to prevent.
    if _desktop_objective_self_sufficient_without_cognitive_text(
        _semantic_user_message
    ) or _screen_perception_needs_her_answer(_semantic_user_message):
        try:
            executed = await _run_desktop_objective_tracked(
                _semantic_user_message,
                cognitive_reply="",
            )
        except _CHAT_RECOVERABLE_ERRORS as exec_exc:
            record_degradation("chat", exec_exc)
            executed = None
        if isinstance(executed, dict) and executed.get("response"):
            return await _finalize_fastpath(
                _chat_desktop_repair._apply_aura_voice_shaping(
                    str(executed.get("response") or "")
                ),
                status=str(executed.get("status") or "desktop_objective"),
            )
    return None



































async def _api_chat_turn(body: ChatRequest, request: Request):
    request_started_at = time.monotonic()
    request_wall_started_at = time.time()
    # Reject oversized messages before processing
    if len(body.message.encode("utf-8", errors="replace")) > MAX_CHAT_MESSAGE_BYTES:
        raise HTTPException(status_code=413, detail="Message too large (max 64KB)")

    # From here until this turn is answered, her own unprompted speech waits.
    # She is meant to have things to say; a person waiting on an answer is not
    # the moment for them.
    try:
        from core.conversation.surface_delivery import note_turn_started

        note_turn_started(
            conversation_id=_CHAT_REQUEST_SESSION.get(),
            turn_id=_CHAT_DELIVERY_TURN_ID.get(),
        )
    except _CHAT_RECOVERABLE_ERRORS as _exc:
        record_degradation("chat", _exc, severity="info", action="turn start unrecorded")

    request_client = getattr(request, "client", None)
    _request_origin = str(getattr(request_client, "host", "unknown") or "unknown")
    _request_access_profile = request_access_profile(request)
    _trusted_local_origin = _request_access_profile.get("surface") == "owner"
    conversation_only_surface = bool(_request_access_profile.get("conversation_only", True))
    _profile_user_id = _resolve_exact_profile_user_id(request)
    _defensive_context = ""
    try:
        from core.security.defensive_runtime import inspect_chat_ingress

        _defensive_decision = inspect_chat_ingress(
            body.message,
            origin=_request_origin,
            trusted_local=_trusted_local_origin,
            surface="api_chat",
        )
        if not _defensive_decision.allowed:
            if _defensive_decision.action == "security_preflight_unavailable":
                return _pre_gate_unavailable_response("defensive_runtime")
            return _early_chat_json_response(
                {
                    "error": _defensive_decision.action,
                    "message": "Request blocked by Aura's defensive runtime.",
                    "reasons": _defensive_decision.reasons,
                },
                status_code=_defensive_decision.status_code,
            )
        _defensive_context = _defensive_decision.cognitive_context
    except _CHAT_RECOVERABLE_ERRORS as _defensive_exc:
        record_degradation("chat.defensive_runtime", _defensive_exc)
        logger.warning("Chat defensive preflight unavailable: %s", _defensive_exc)
        return _pre_gate_unavailable_response("defensive_runtime")

    # Benchmark and proof behavior is an owner-only control surface. A paired
    # caller cannot opt out of the production conversation contract by
    # supplying internal headers directly.
    is_benchmark = (
        not conversation_only_surface and request.headers.get("X-Aura-Benchmark") == "true"
    )
    chat_origin = "benchmark" if is_benchmark else "user"
    desktop_requires_cognitive_engine, request_surface = _request_requires_cognitive_engine(
        request,
        is_benchmark=is_benchmark,
    )
    if conversation_only_surface:
        desktop_requires_cognitive_engine = True
        request_surface = "paired-device"

    # ── Chat preflight ──────────────────────────────────────────
    # 1) File-reference loading: if the user references a file path, load
    #    its contents (sandboxed, bounded) and prepend as context.
    # 2) Directive injection: if the message looks like an introspective /
    #    specific-recall / continuity question, prepend response guidance
    #    that fights LLM-default failure modes (confabulation, generic
    #    chat-AI prose on substrate-aware questions).
    _chat_session_id: str = "default"
    _original_user_message: str = body.message
    from core.conversation.interlocutor_identity import (
        parse_interlocutor_introduction,
    )

    _interlocutor_turn = parse_interlocutor_introduction(_original_user_message)
    _semantic_user_message = _interlocutor_turn.utterance
    # Any visible user turn advances the conversation. Consume the exact cache
    # capability now so an early bounded/fallback exit cannot leave stale state
    # for a later turn to append onto.
    _conversation_resume_handle_for_turn = _take_conversation_resume_handle()
    # Bound HERE because a nested helper closes over it.
    #
    # `_try_serve_grounded_recovery` is defined around line 16800 and reads
    # this, while the assignment sat a thousand lines further down — so any
    # turn that reached the helper by an earlier path died on "cannot access
    # free variable 'preflight_context_message'", and the person got "I hit an
    # error before a coherent answer formed". A closure variable has to be
    # bound before the closure can be called, not before it is written.
    preflight_context_message = str(body.message or "")
    # Every degraded path from here on can now see what was asked. Without it
    # they answer about the LANE — live, "what is 7919 * 6367?" came back as
    # "the live answer lane could not finish preparing", for a product the
    # runtime computes exactly.
    set_user_question(_semantic_user_message)
    _declared_interlocutor = _interlocutor_turn.evidence()
    _grounded_recall_context: str = ""
    _relational_memory_control = getattr(
        getattr(request, "state", None),
        "relational_memory_control",
        None,
    )
    if isinstance(_relational_memory_control, dict):
        body.message = (
            "[CANONICAL RELATIONAL MEMORY CONTROL RESULT]\n"
            + json.dumps(
                _relational_memory_control,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n[END RELATIONAL MEMORY CONTROL RESULT]\n"
            + "Answer the user's memory-control request from this result; do not claim a mode or persistence outcome beyond it.\n\n"
            + body.message
        )
    if _defensive_context and not is_benchmark:
        body.message = f"{_defensive_context}{body.message}"
    _preflight = await _chat_preflight._run_chat_preflight(
        body,
        request,
        _semantic_user_message,
        _profile_user_id,
        conversation_only_surface,
        is_benchmark,
        _chat_session_id=_chat_session_id,
        _grounded_recall_context=_grounded_recall_context,
        raw_user_message=_original_user_message,
    )
    logger.info("Chat preflight timing: %s", dict(_preflight.timing_ms))
    if _preflight.early_response is not None:
        return _preflight.early_response
    if _preflight.chat_session_id is not _UNSET:
        _chat_session_id = _preflight.chat_session_id
    if _preflight.grounded_recall_context is not _UNSET:
        _grounded_recall_context = _preflight.grounded_recall_context
    if _preflight.grounded is not _UNSET:
        _grounded = _preflight.grounded
    if _preflight.shown is not _UNSET:
        _shown = _preflight.shown
    if _preflight.status is not _UNSET:
        status = _preflight.status
    _turn_sensory_evidence = _preflight.turn_sensory_evidence
    _qualified_state_serialization_owner = bool(
        _preflight.evidence_profile
        == _chat_preflight._CHAT_EVIDENCE_PROFILE_QUALIFIED_RECURRENT
        and _preflight.evidence_owner_receipt
    )

    # Keep user-facing judgment anchored to the text Bryan actually typed.
    # `body.message` may now contain continuity blocks, file payloads, and
    # directive scaffolding that belong in generation context, not in reply
    # quality classification or conversational memory.
    # Identity-anchor ablation, scoped to this turn.
    #
    # tools/ablate_identity_anchor.py measures which sections of the anchor
    # actually change behaviour. It runs over HTTP against the live instance,
    # so it asks per request rather than by restarting anything; the holder is
    # turn-scoped, so a measurement cannot alter anyone else's conversation.
    try:
        _ablate_section = str(request.headers.get("X-Aura-Ablate-Identity-Section") or "").strip()
        if _ablate_section:
            from core.brain.aura_persona import set_ablated_section

            set_ablated_section(_ablate_section)
            logger.info(
                "🔬 Identity-anchor ablation active for this turn: %s",
                _ablate_section[:60],
            )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)

    if not is_benchmark and _chat_preflight._looks_like_desktop_objective(_semantic_user_message):
        # A consequential desktop request always needs the same CognitiveEngine
        # planning lane as the desktop UI, even when it arrives through the
        # plain REST surface. The governed executor remains downstream.
        desktop_requires_cognitive_engine = True
        request_surface = request_surface or "desktop-objective"
    if not is_benchmark:
        try:
            from core.runtime.foreground_guard import notify_user_spoke as _guard_notify_user_spoke

            _guard_notify_user_spoke(_semantic_user_message)
        except _CHAT_RECOVERABLE_ERRORS as _guard_notify_exc:
            record_degradation("chat", _guard_notify_exc)
            logger.debug("Foreground guard preflight notify skipped: %s", _guard_notify_exc)

    # ── Conscience pre-gate ─────────────────────────────────────
    # Hard-line rules apply BEFORE the cognitive pipeline ever sees the
    # message. REFUSE returns the rule's rationale verbatim; any other
    # decision falls through. The conscience emits its own audit row.
    try:
        from core.ethics.conscience import Verdict, get_conscience

        _conscience_decision = await asyncio.to_thread(
            get_conscience().evaluate,
            action="user_chat",
            domain="external_communication",
            intent=_semantic_user_message[:240],
            context={"source": "chat_api"},
        )
        if _conscience_decision.verdict == Verdict.REFUSE:
            return _early_chat_json_response(
                {
                    "response": _conscience_decision.rationale,
                    "status": "conscience_refused",
                    "conscience_rule_id": _conscience_decision.rule_id,
                    "response_confidence": "principled_refusal",
                },
                status_code=200,
            )
        if _conscience_decision.verdict == Verdict.REQUIRE_FRESH_USER_AUTH:
            return _early_chat_json_response(
                {
                    "response": "This action needs a fresh confirmation from you within the last 60 seconds. Please re-authorize in Settings → Safety.",
                    "status": "require_fresh_user_auth",
                    "conscience_rule_id": _conscience_decision.rule_id,
                },
                status_code=200,
            )
    except _CHAT_RECOVERABLE_ERRORS as _conscience_exc:
        record_degradation("chat", _conscience_exc)
        logger.warning("Conscience pre-gate unavailable: %s", _conscience_exc)
        return _pre_gate_unavailable_response("conscience")

    owner_session_restored = bool(_restore_owner_session_from_request(request))
    lane = _chat_preflight._collect_conversation_lane_status()
    foreground_timeout = _foreground_timeout_for_lane(
        lane,
        _semantic_user_message,
    )
    # One clock for this turn, opened with what the route granted. Whatever
    # prices the turn again later asks this, and this wait reads what it said.
    try:
        from core.runtime.the_turn_clock import open_a_turn_clock

        open_a_turn_clock(foreground_timeout)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc, severity="debug",
                           action="turn ran on the admitted budget alone")
    early_allow_chat_fastpaths = not is_benchmark and not desktop_requires_cognitive_engine
    pending_exchange_id: str | None = None
    foreground_slot_acquired = False
    foreground_lock_token: object | None = None
    foreground_owner_task = asyncio.current_task()
    foreground_lease = None
    kernel_task: asyncio.Task | None = None
    # Start this turn holding no draft from any previous one.
    try:
        from core.conversation.surface_disposition import clear_preserved_draft

        clear_preserved_draft()
    except _CHAT_RECOVERABLE_ERRORS as exc:
        logger.debug("Preserved-draft reset skipped: %s", exc)
    _live_turn_trace: dict[str, Any] = {
        "desktop_cognitive_engine_required": bool(desktop_requires_cognitive_engine),
        "request_surface": request_surface or "",
        "chat_origin": chat_origin,
        "engine_think_invoked": False,
        "cognitive_engine_reply_accepted": False,
        "cognitive_engine_reply_failed": False,
        "bounded_contract_used": False,
        "legacy_fallback_used": False,
        "response_path": "",
        "post_generation_repair_applied": False,
        "deterministic_repair_applied": False,
        "preflight_evidence_profile": _preflight.evidence_profile,
        "preflight_evidence_owner_receipt": _preflight.evidence_owner_receipt,
        "preflight_skipped_components": list(_preflight.skipped_components),
    }
    _prime_requested_output_contract_trace(
        _live_turn_trace,
        user_message=_semantic_user_message,
    )

    def _live_turn_contract(
        *,
        lane_status: dict[str, Any] | None = None,
        response_confidence: str = "",
        status: str = "",
        reply_source: str = "",
    ) -> dict[str, Any]:
        return _chat_turn_contract._build_live_turn_contract_payload(
            desktop_required=bool(desktop_requires_cognitive_engine),
            request_surface=request_surface or "",
            lane_status=lane_status or _chat_preflight._collect_conversation_lane_status(),
            response_confidence=response_confidence,
            status=status,
            reply_source=reply_source,
            turn_trace=_live_turn_trace,
        )

    def _enforce_main_requested_output_contract(reply_text: Any) -> tuple[str, bool]:
        return _requested_output_contract_result(
            _live_turn_trace,
            user_message=_semantic_user_message,
            reply_text=reply_text,
        )

    def _remaining_foreground_budget(*, reserve: float = 0.0) -> float:
        elapsed = time.monotonic() - request_started_at
        # What this turn was granted, or what it has since been shown it needs.
        #
        # The SLA above is set when the request is admitted, from the lane and
        # a guess at the prompt's length. The answer clock prices the same turn
        # again later from the prompt that was actually built and the rates the
        # worker just measured, and that number was invisible here. LIVE,
        # 2026-09-10: "deadline 103s to 251s", the gate honoured 251, this wait
        # gave up at its own budget with the cortex still generating, and the
        # person was told the answer took too long to finish cleanly.
        granted = foreground_timeout
        try:
            from core.runtime.the_turn_clock import the_turn_clock

            clock = the_turn_clock()
            if clock is not None:
                granted = max(granted, clock.budget())
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat", exc, severity="debug",
                               action="used the admitted budget for this wait")
        return max(2.0, granted - elapsed - reserve)

    async def _cancel_kernel_task_if_pending(reason: str) -> None:
        nonlocal kernel_task
        task = kernel_task
        if task is None or task.done():
            return
        logger.warning("Cancelling abandoned KernelInterface chat task after %s.", reason)
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.CancelledError:
            return
        except TimeoutError:
            logger.error("KernelInterface chat task ignored cancellation after %s.", reason)
            task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat", exc)
            logger.debug(
                "KernelInterface task cleanup observed exception after %s: %s", reason, exc
            )

    try:
        if is_shutdown_requested():
            return _runtime_shutdown_response(
                "pre_admission", slot_acquired=foreground_slot_acquired
            )
        try:
            from core.runtime.foreground_guard import begin_foreground_turn

            foreground_lease = begin_foreground_turn(
                owner=f"chat_api:{_chat_session_id}",
                source="chat_api",
            )
        except _CHAT_RECOVERABLE_ERRORS as _lease_exc:
            record_degradation("chat", _lease_exc)
            logger.debug("Foreground guard lease skipped: %s", _lease_exc)

        if early_allow_chat_fastpaths and _chat_preflight._is_explicit_capability_inventory_request(
            _semantic_user_message
        ):
            reply_text = _chat_desktop_repair._build_grounded_capability_inventory_reply(
                _semantic_user_message
            )
            return JSONResponse(
                {
                    "response": _chat_desktop_repair._apply_aura_voice_shaping(reply_text),
                    "status": "cognitive_engine_capability_inventory",
                    "conversation_lane": _chat_preflight._collect_conversation_lane_status(),
                    "response_confidence": "high",
                },
                status_code=200,
            )

        try:
            # A person's typed turn outranks whatever is holding the lane for
            # background reasons; it never outranks another person's turn,
            # which is why this waits rather than preempting.
            foreground_busy_wait_s = 30.0 if is_benchmark else _FOREGROUND_CHAT_BUSY_WAIT_S
            if not is_benchmark:
                foreground_busy_wait_s = min(
                    foreground_busy_wait_s, _FOREGROUND_CHAT_LOCK_PREEMPT_AFTER_S
                )
            foreground_lock_token = await asyncio.wait_for(
                _foreground_chat_lock.acquire(owner_task=foreground_owner_task),
                timeout=max(
                    0.05, min(foreground_busy_wait_s, _remaining_foreground_budget(reserve=1.0))
                ),
            )
            foreground_slot_acquired = True
            logger.info("Foreground chat reservation acquired")
            if is_shutdown_requested():
                return _runtime_shutdown_response(
                    "foreground_reserved", slot_acquired=foreground_slot_acquired
                )
        except TimeoutError:
            held = getattr(_foreground_chat_lock, "held_duration", 0.0)
            if held > _FOREGROUND_CHAT_LOCK_PREEMPT_AFTER_S:
                logger.error(
                    "Cancelling stale foreground generation (held %.1fs) before "
                    "exclusive handoff to the new user turn.",
                    held,
                )
                owner_cancelled = False
                if hasattr(_foreground_chat_lock, "cancel_stale_owner"):
                    owner_cancelled = await _foreground_chat_lock.cancel_stale_owner(
                        reason=_FOREGROUND_CHAT_PREEMPT_CANCEL_REASON,
                    )
                if owner_cancelled:
                    _force_clear_mlx_foreground_owner(
                        reason="chat_lock_preemption",
                        min_age_s=_FOREGROUND_CHAT_LOCK_PREEMPT_AFTER_S,
                    )
                    try:
                        foreground_lock_token = await asyncio.wait_for(
                            _foreground_chat_lock.acquire(owner_task=foreground_owner_task),
                            timeout=1.0,
                        )
                        foreground_slot_acquired = True
                    except TimeoutError as exc:
                        logger.debug(
                            "Foreground lock reacquire after preemption timed out: %s",
                            exc,
                        )

            if not foreground_slot_acquired:
                status = "benchmark_foreground_busy" if is_benchmark else "foreground_busy"
                response_confidence = "failed" if is_benchmark else "degraded"
                status_code = 503 if is_benchmark else 200
                return JSONResponse(
                    {
                        "response": "I still have the previous turn open. I am not going to fake a new answer over it; the next clean reply should land from the active turn.",
                        "status": status,
                        "conversation_lane": _chat_preflight._collect_conversation_lane_status(),
                        "response_confidence": response_confidence,
                    },
                    status_code=status_code,
                )

        # Memory admission belongs to the generation lease, not to the time
        # before a potentially long lock wait. A queued turn can cross from
        # healthy to unsafe while waiting; sample only after exclusive custody.
        memory_admission_response = await _foreground_memory_admission_response(
            is_benchmark=is_benchmark,
            phase="foreground_generation_admission",
        )
        if memory_admission_response is not None:
            return memory_admission_response

        # Notify proactive presence systems; pass content for away-signal detection
        if not is_benchmark:
            _notify_user_spoke(_semantic_user_message)

        # Animal cognition: track user emotional state and adapt style
        if not is_benchmark:
            try:
                from core.consciousness.animal_cognition import (
                    get_camouflage_adapter,
                    get_emotional_tracker,
                )

                emotional_tracker = get_emotional_tracker()
                emotional_tracker.update(_semantic_user_message)
                camouflage = get_camouflage_adapter()
                camouflage.observe_user(_semantic_user_message)
                # Feed emotional signals into neurochemical system
                ncs = ServiceContainer.get("neurochemical_system", default=None)
                if ncs:
                    triggers = emotional_tracker.get_neurochemical_triggers()
                    for trigger, amount in triggers.items():
                        if "norepinephrine" in trigger:
                            ncs.on_wakefulness(amount)
                        elif "dopamine" in trigger:
                            ncs.on_novelty(amount)
                        elif "oxytocin" in trigger:
                            ncs.on_social_connection(amount)
            except _CHAT_RECOVERABLE_ERRORS as _ac_exc:
                record_degradation("chat", _ac_exc)
                logger.debug("Animal cognition tracking skipped: %s", _ac_exc)

        allow_chat_fastpaths = not is_benchmark and not desktop_requires_cognitive_engine
        # Session-memory pin/recall is a canonical memory gateway operation, not
        # a language-model shortcut. Desktop-required surfaces collect/write the
        # canonical state before generation and bind it into CognitiveEngine
        # below; non-required API surfaces may answer directly from that gateway.
        allow_memory_state_fastpath = (
            not is_benchmark
            and not desktop_requires_cognitive_engine
            and not conversation_only_surface
        )
        allow_runtime_status_fastpath = not is_benchmark and not desktop_requires_cognitive_engine
        allow_governed_action_fastpaths = (
            not is_benchmark
            and not desktop_requires_cognitive_engine
            and not conversation_only_surface
        )
        desktop_memory_state_evidence: tuple[str, str] | None = None

        _desktop_exec_state = {"attempted": False, "result": None}
        action_episode_evidence = ""
        action_episode_projection = ""

        def _current_exchange_metadata(
            answer_text: str,
            *,
            response_path: str = "",
        ) -> dict[str, Any] | None:
            metadata: dict[str, Any] = {}
            episode = _desktop_exec_state.get("action_episode")
            if isinstance(episode, dict):
                metadata["action_episode"] = dict(episode)
            try:
                from core.conversation.answer_provenance import answer_provenance_from_turn

                if answer_text:
                    metadata["answer_provenance"] = answer_provenance_from_turn(
                        answer_text,
                        response_path=(
                            str(response_path or "")
                            or str(_live_turn_trace.get("response_path") or "")
                        ),
                        model_native_inference=bool(
                            _live_turn_trace.get("foreground_model_generation_count")
                            or _live_turn_trace.get("foreground_model_generation_consumed")
                            or str(response_path or "") == "protected_foreground"
                        ),
                    ).to_dict()
            except _CHAT_RECOVERABLE_ERRORS as provenance_exc:
                record_degradation("chat.answer_provenance", provenance_exc)
            return metadata or None

        async def _run_desktop_objective_tracked(message: str, *, cognitive_reply: str) -> dict[str, Any] | None:
            """Forwards to the lifted _lifted_run_desktop_objective_tracked."""
            return await _lifted_run_desktop_objective_tracked(
                message=message,
                cognitive_reply=cognitive_reply,
                _desktop_exec_state=_desktop_exec_state,
                conversation_only_surface=conversation_only_surface,
                pending_exchange_id=pending_exchange_id,
            )

        async def _apply_desktop_objective_chokepoint(final_text: str, status: str) -> tuple[str, str]:
            """Forwards to the lifted _lifted_apply_desktop_objective_chokepoint."""
            return await _lifted_apply_desktop_objective_chokepoint(
                final_text=final_text,
                status=status,
                _desktop_exec_state=_desktop_exec_state,
                _semantic_user_message=_semantic_user_message,
                conversation_only_surface=conversation_only_surface,
                is_benchmark=is_benchmark,
                _run_desktop_objective_tracked=_run_desktop_objective_tracked,
            )

        async def _try_serve_grounded_recovery(
            rejected_reply: str = "",
            *,
            reasons: tuple[str, ...] | list[str] | None = None,
            response_path: str = "desktop_grounded_recovery",
        ) -> JSONResponse | None:
            """Recover only through the same required CognitiveEngine/full-mind path.

            The older recovery helper called ``inference_gate.generate`` directly and
            returned HTTP 200/high confidence. That was safer than serving the bad
            draft, but it was still a raw-model side door on the launched desktop
            lane. A recovery that reaches the user must now prove the same live
            mind contract as an ordinary desktop turn.
            """
            nonlocal pending_exchange_id
            if not desktop_requires_cognitive_engine:
                return None
            completion_failure_reasons = {
                "truncated_tail",
                "final_answer_missing",
                "missing_final_answer",
                "incomplete_code_response",
                "unanswered_question_part",
            }
            reason_tuple = tuple(
                str(reason).strip() for reason in (reasons or ()) if str(reason or "").strip()
            )
            if rejected_reply:
                try:
                    from core.conversation.response_reliability import (
                        assess_user_facing_reply,
                    )

                    rejected_assessment = assess_user_facing_reply(
                        _semantic_user_message,
                        rejected_reply,
                    )
                    assessed_reasons = tuple(
                        str(reason or "").strip()
                        for reason in getattr(rejected_assessment, "reasons", ())
                        if str(reason or "").strip()
                    )
                    reason_tuple = tuple(dict.fromkeys((*reason_tuple, *assessed_reasons)))
                except _CHAT_RECOVERABLE_ERRORS as assessment_exc:
                    record_degradation(
                        "chat",
                        assessment_exc,
                        severity="warning",
                        action="retained caller-supplied recovery reasons",
                    )
            if not reason_tuple:
                reason_tuple = (str(response_path or "desktop_reply_not_proven"),)
            normalized_recovery_reasons = {reason.lower() for reason in reason_tuple}
            completion_recovery = bool(
                rejected_reply and normalized_recovery_reasons & completion_failure_reasons
                # A state projection can be incomplete without there ever
                # having been a model generation to resume. Its text is not
                # an assistant tail and cannot mint a continuation identity.
                and _live_turn_trace.get("foreground_model_generation_consumed")
            )
            if (
                bool(
                    _live_turn_trace.get("single_owner_generation_exhausted")
                    or _live_turn_trace.get("foreground_model_generation_consumed")
                )
                and not completion_recovery
            ):
                logger.warning(
                    "Skipping duplicate desktop recovery generation after the "
                    "CognitiveEngine owner exhausted this turn."
                )
                return None
            if bool(_live_turn_trace.get("cognitive_engine_grounded_recovery_attempted")):
                return None
            _live_turn_trace["cognitive_engine_grounded_recovery_attempted"] = True

            try:
                pressure = _required_foreground_memory_snapshot()
                if pressure.refuse_heavy_local_generation:
                    logger.warning(
                        "Skipping desktop CognitiveEngine recovery under memory pressure: %s",
                        pressure.reason,
                    )
                    return None
            except _CHAT_RECOVERABLE_ERRORS as pressure_exc:
                record_degradation(
                    "chat.memory_admission",
                    pressure_exc,
                    severity="warning",
                    action="refused recovery generation without a measured memory decision",
                    extra={"phase": "desktop_grounded_recovery"},
                )
                return None

            recovery_budget = _desktop_required_cognitive_budget(
                foreground_timeout=foreground_timeout,
                elapsed_s=time.monotonic() - request_started_at,
            )
            if recovery_budget < _DESKTOP_COGNITIVE_MIN_REQUIRED_BUDGET_S:
                logger.warning(
                    "Skipping desktop CognitiveEngine recovery; remaining budget %.1fs is below %.1fs.",
                    recovery_budget,
                    _DESKTOP_COGNITIVE_MIN_REQUIRED_BUDGET_S,
                )
                return None

            recovery_message = _semantic_user_message
            recovery_trace: dict[str, Any] = {}
            try:
                recovered = await _run_cognitive_engine_chat_turn(
                    recovery_message,
                    visible_user_message=_semantic_user_message,
                    preflight_context_message=preflight_context_message,
                    turn_sensory_evidence=_turn_sensory_evidence,
                    session_id=_chat_session_id,
                    origin=chat_origin,
                    timeout_s=recovery_budget,
                    lane=dict(lane or {}),
                    source="desktop_ui_recovery",
                    require_engine=True,
                    conversation_only_surface=conversation_only_surface,
                    principal_id=_profile_user_id,
                    turn_trace=recovery_trace,
                    continuation_partial=(rejected_reply if completion_recovery else ""),
                    continuation_reasons=(
                        tuple(
                            reason
                            for reason in reason_tuple
                            if reason.lower() in completion_failure_reasons
                        )
                        if completion_recovery
                        else None
                    ),
                    continuation_evidence=(
                        {
                            "foreground_model_generation_count": _live_turn_trace.get(
                                "foreground_model_generation_count", 1
                            ),
                            "foreground_model_generation_segment_count": _live_turn_trace.get(
                                "foreground_model_generation_segment_count", 1
                            ),
                            "foreground_model_generation_transaction_count": _live_turn_trace.get(
                                "foreground_model_generation_transaction_count", 1
                            ),
                            "foreground_model_generation_transaction_id": _live_turn_trace.get(
                                "foreground_model_generation_transaction_id", ""
                            ),
                            "completion_retry_count": _live_turn_trace.get(
                                "completion_retry_count", 0
                            ),
                            "continuation_resume_handle": (
                                (_live_turn_trace.get("live_mind_surface_control_receipt") or {}).get(
                                    "continuation_resume_handle", ""
                                )
                            ),
                        }
                        if completion_recovery
                        else None
                    ),
                )
            except _CHAT_RECOVERABLE_ERRORS as rec_exc:
                record_degradation("chat", rec_exc)
                recovered = None
            if not recovered:
                return None

            try:
                from core.conversation.response_reliability import assess_user_facing_reply

                recovered_recent_user_messages = await _gather_recent_user_messages_for_relevance(
                    _semantic_user_message
                )
                recovered_assessment = assess_user_facing_reply(
                    _semantic_user_message,
                    recovered,
                    recent_user_messages=recovered_recent_user_messages,
                )
            except _CHAT_RECOVERABLE_ERRORS as assess_exc:
                record_degradation("chat", assess_exc)
                recovered_assessment = None
            if _reply_assessment_requires_repair_with_memory_evidence(
                recovered_assessment,
                _semantic_user_message,
                recovered,
                memory_state_evidence=desktop_memory_state_evidence,
            ):
                if bool(getattr(recovered_assessment, "hard_failure", False)):
                    # Leaks, artifacts, corrupted language, unsupported
                    # claims — never user-facing, whatever the alternative.
                    logger.warning(
                        "Desktop CognitiveEngine recovery still failed reliability "
                        "gate (hard: %s).",
                        ",".join(getattr(recovered_assessment, "reasons", ()) or ()),
                    )
                    return None
                # Soft coverage nits (a missing follow-up question, a list
                # count) on a LAST-RESORT recovery: the only alternative is
                # a fail-closed apology, which reads as fragility over a
                # working mind. Serve the genuine reply and disclose.
                logger.warning(
                    "Desktop recovery served despite soft reliability reasons (%s).",
                    ",".join(getattr(recovered_assessment, "reasons", ()) or ()),
                )

            recovered = _enforce_or_bind_terminal_output_contract(
                recovery_trace,
                user_message=_semantic_user_message,
                reply_text=recovered,
            )

            recovered_lane = _chat_preflight._collect_conversation_lane_status()
            recovery_contract = _chat_turn_contract._build_live_turn_contract_payload(
                desktop_required=True,
                request_surface=request_surface,
                lane_status=recovered_lane,
                response_confidence="high",
                status="cognitive_engine_recovered",
                reply_source=str(recovery_trace.get("response_path") or "cognitive_engine"),
                turn_trace=recovery_trace,
            )
            if not bool(recovery_contract.get("final_requested_output_contract_proven")):
                logger.warning(
                    "Desktop CognitiveEngine recovery did not satisfy the user-authored "
                    "output contract; continuing to the normal fail-closed path."
                )
                return None
            if not bool(recovery_contract.get("full_mind_path")):
                if _authored_answer_can_serve(recovery_contract):
                    # Authentic mind-authored text with only state-completeness
                    # proofs missing (self-healing window): serve with the
                    # degradation disclosed — never an apology over a live mind.
                    logger.warning(
                        "Desktop recovery served with DEGRADED full-mind proof "
                        "(authentic cognitive reply; missing: %s).",
                        ",".join(recovery_contract.get("full_mind_missing_proofs") or ()),
                    )
                    recovery_trace["full_mind_proof_degraded"] = True
                else:
                    logger.warning(
                        "Desktop CognitiveEngine recovery produced text but did not prove "
                        "authorship (missing: %s).",
                        ",".join(recovery_contract.get("full_mind_missing_proofs") or ()),
                    )
                    return None

            logger.info(
                "✅ Degraded desktop turn recovered through CognitiveEngine full-mind path."
            )
            _live_turn_trace.update(recovery_trace)
            _live_turn_trace["cognitive_engine_grounded_recovery"] = True
            if pending_exchange_id:
                await _chat_preflight._complete_logged_exchange(
                    pending_exchange_id, _semantic_user_message, recovered, record_experience=True
                )
                pending_exchange_id = None
            else:
                await _chat_preflight._log_exchange(
                    _original_user_message,
                    recovered,
                    record_experience=True,
                    session_id=_chat_session_id,
                )
            _record_recent_response(recovered, _semantic_user_message)
            await _emit_chat_output_receipt(
                recovered,
                cause="chat_response",
                metadata={
                    "response_confidence": "high",
                    "path": "cognitive_engine_recovery",
                    "status": "cognitive_engine_recovered",
                    "recovery_from": str(response_path or ""),
                },
            )
            return JSONResponse(
                {
                    "response": recovered,
                    "status": "cognitive_engine_recovered",
                    "conversation_lane": recovered_lane,
                    "response_confidence": "high",
                    "live_turn_contract": recovery_contract,
                },
                status_code=200,
            )

        async def _fail_closed_degraded_desktop_reply(
            rejected_reply: str,
            *,
            response_path: str,
            status: str = "desktop_response_quality_failed",
            reason: str = "required_desktop_reply_remained_degraded",
        ) -> JSONResponse:
            """Recover competently if possible; only refuse as a true last resort."""

            nonlocal pending_exchange_id

            # COMPETENCE over fail-closed: try a clean grounded recovery before surrendering.
            served = await _try_serve_grounded_recovery(
                rejected_reply,
                reasons=(response_path, reason),
                response_path=response_path,
            )
            if served is not None:
                return served

            # The other half of the repairable-draft contract. Three gates now
            # PRESERVE a draft that merely fell short instead of discarding it
            # (see core/conversation/surface_disposition.py) — on the promise
            # that repair would finish it. When repair cannot run, that promise
            # went unkept and the preserved work was dropped here anyway.
            #
            # Live 2026-07-26, the whole chain in one turn:
            #   Cortex produced repairable user-facing draft shape
            #     (truncated_tail, len=224). Passing it to downstream repair…
            #   Preserving repairable Cortex draft for downstream repair…
            #   ResponseGeneration kept repairable foreground draft…
            #   Skipping CognitiveEngine desktop repair retry; the foreground
            #     model owner already produced work for this turn.
            #   → "I couldn't get to an answer I'd stand behind on that one."
            #
            # The one-generation bound is deliberate and stays. What changes is
            # that a draft everything agreed was servable is served, rather
            # than traded for an apology that says less.
            # A veto is not a model failure, and reporting it as one is what
            # kept this entire class invisible: a destroyed answer and a failed
            # generation produce the same sentence, so every incident tightened
            # the gates and nothing ever loosened them. Name it in the log.
            try:
                from core.conversation.surface_disposition import raw_model_draft

                _raw = raw_model_draft()
                if _raw:
                    logger.warning(
                        "🧱 Gate veto, not a generation failure: the model "
                        "produced %d chars this turn and the pipeline is about "
                        "to withhold them (path=%s reason=%s).",
                        len(_raw),
                        response_path,
                        reason,
                    )
            except _CHAT_RECOVERABLE_ERRORS:
                pass

            salvaged = _servable_draft_or_none(
                rejected_reply,
                _semantic_user_message,
                _live_turn_trace.get("turn_id") or _live_turn_trace.get("idempotency_key") or "",
            )
            if salvaged:
                salvaged, salvage_output_proven = _enforce_main_requested_output_contract(
                    salvaged
                )
                salvage_contract = _live_turn_contract(
                    lane_status=_chat_preflight._collect_conversation_lane_status(),
                    response_confidence="degraded",
                    status=str(_live_turn_trace.get("response_path") or response_path),
                    reply_source=str(
                        _live_turn_trace.get("response_path") or response_path
                    ),
                )
                if not (
                    salvage_output_proven
                    and _authored_answer_can_serve(salvage_contract)
                ):
                    logger.warning(
                        "Preserved draft remained ineligible for delivery; "
                        "withholding it (missing=%s).",
                        ",".join(
                            salvage_contract.get("full_mind_missing_proofs") or ()
                        )
                        or "unknown",
                    )
                    salvaged = ""
            if salvaged:
                logger.warning(
                    "Serving the preserved repairable draft (%d chars) rather "
                    "than refusing: repair could not run for this turn.",
                    len(salvaged),
                )
                _live_turn_trace.update(
                    {
                        "post_generation_repair_applied": False,
                    }
                )
                if pending_exchange_id:
                    await _chat_preflight._complete_logged_exchange(
                        pending_exchange_id,
                        _semantic_user_message,
                        salvaged,
                        record_experience=False,
                    )
                    pending_exchange_id = None
                # A JSONResponse, like every other exit from this handler.
                # `_servable_draft_or_none` returns a STR, and returning it raw
                # meant the delivery boundary's `isinstance(response,
                # JSONResponse)` check replaced the salvaged answer with a 500
                # and the sentence "The chat route returned an unsupported
                # response format." Measured live: Aura opened Notes and wrote
                # the requested three-sentence note — the note is on disk — and
                # the person saw only that error.
                #
                # Worst possible polarity: the REFUSAL path below was correctly
                # formed, so a turn that failed reported cleanly while a turn
                # that succeeded reported a transport error. The whole
                # repairable-draft mechanism exists to stop good work being
                # discarded, and it was being discarded one layer further down.
                salvage_lane = _mark_conversation_lane_state(status, state="ready")
                return JSONResponse(
                    {
                        "response": salvaged,
                        "status": f"{response_path}:served_repairable_draft",
                        "reason": reason,
                        "conversation_lane": salvage_lane,
                        "response_confidence": "degraded",
                        "live_turn_contract": salvage_contract,
                    },
                    status_code=200,
                )

            lane = _mark_conversation_lane_state(status, state="failed")
            _live_turn_trace.update(
                {
                    "response_path": response_path,
                    "rejected_reply_len": len(str(rejected_reply or "")),
                }
            )
            # Plain speech. "full-mind", "failed closed" and "ungrounded" are
            # words for the engineering log, not for the person waiting — the
            # 2026-07-25 endurance probe served this exact sentence on the
            # retention turns and it reads like a subsystem talking about
            # itself. Say what happened and what they can do.
            from core.conversation.reply_provenance import THE_HONEST_FAILURE

            failure_reply = THE_HONEST_FAILURE
            # Before giving up, ask whether the runtime already HOLDS the answer.
            #
            # Live 2026-08-10: asked which of her subsystems were degraded and
            # whether any job had been failing repeatedly, she served the
            # sentence above — while /api/health carried integrity=degraded and
            # overt_action_cycle at failures=13 with its exact TypeError. The
            # answer was structured, live, and hers. Nothing had fetched it.
            #
            # This is a reading, not a rescue: self_health_answer() returns text
            # only when a channel actually produced a value, and the text is
            # built from those values, so it cannot describe a health she does
            # not have. When no channel reads, it returns "" and the honest
            # refusal above stands.
            evidenced_reply = ""
            # An exact answer is the strongest reading of all: it needs no
            # channel, no generation and no model. Live 2026-08-19 the
            # arithmetic drafts were rejected three times for missing the very
            # number the runtime could compute, and the turn ended in the
            # sentence above with the answer sitting one function call away.
            computed = _known_answer_for_this_turn()
            if computed:
                evidenced_reply = computed
            if not evidenced_reply and _is_simple_affect_check_request(_semantic_user_message):
                evidenced_reply = _build_grounded_self_condition_reply(
                    _semantic_user_message,
                    session_id=_chat_session_id,
                )
            if not evidenced_reply:
                evidenced_reply = _chat_conversation_repair._self_health_answer_or_empty(
                    _semantic_user_message
                )
            if not evidenced_reply:
                # A tool ran and found something. Saying "I couldn't get to an
                # answer" on top of that is not honesty, it is losing the work.
                #
                # LIVE, 2026-08-27, repeatedly: file_operation read the docs in
                # 4ms, diagnose_repo returned a complete finding in 276ms,
                # code_repl ran — and the turn still ended in the sentence
                # above, because the model wrote nothing usable afterwards.
                # Every governance link was clear by then; what was missing was
                # somebody saying what the tool had returned.
                evidenced_reply = _what_the_tools_found()
            evidenced_reply = await _anything_better_than_giving_up(
                _semantic_user_message,
                reason="the cognitive engine could not serve this turn",
                already=evidenced_reply,
                budget_s=_remaining_foreground_budget(),
            )
            if evidenced_reply:
                failure_reply = evidenced_reply
            if pending_exchange_id:
                await _chat_preflight._complete_logged_exchange(
                    pending_exchange_id,
                    _semantic_user_message,
                    failure_reply,
                    record_experience=False,
                )
                pending_exchange_id = None
            else:
                await _chat_preflight._log_exchange(
                    _original_user_message,
                    failure_reply,
                    record_experience=False,
                    session_id=_chat_session_id,
                )
            await _emit_chat_output_receipt(
                failure_reply,
                cause="chat_response",
                metadata={
                    "response_confidence": "failed",
                    "path": response_path,
                    "status": status,
                    "reason": reason,
                    "rejected_reply_len": len(str(rejected_reply or "")),
                },
            )
            return JSONResponse(
                {
                    "response": failure_reply,
                    "status": status,
                    "reason": reason,
                    "conversation_lane": lane,
                    "response_confidence": "failed",
                    "live_turn_contract": _live_turn_contract(
                        lane_status=lane,
                        response_confidence="failed",
                        status=status,
                        reply_source=response_path,
                    ),
                },
                # A real user gets the honest fail-closed reply IN-BAND (200)
                # so the UI renders it as a message; raw 503s here surfaced
                # as bare "HTTP Error 503" to clients (both July 8 soaks).
                # Benchmarks keep the strict status code — same contract as
                # the foreground_busy path above.
                status_code=503 if is_benchmark else 200,
            )

        async def _finalize_fastpath(
            reply_text: str,
            status: str = "ok",
            *,
            assertion_response: Any = None,
        ):
            nonlocal pending_exchange_id
            final_text = _never_an_ellipsis(reply_text)
            try:
                from core.reasoning.symbolic_bridge import SymbolicBridge

                final_text, exact_repairs = SymbolicBridge().repair_arithmetic_claims(
                    final_text
                )
                if exact_repairs:
                    logger.warning(
                        "Exact claim verifier corrected %d fast-path arithmetic assertion(s) before commit.",
                        len(exact_repairs),
                    )
            except _CHAT_RECOVERABLE_ERRORS as exact_exc:
                record_degradation("chat.exact_claim_repair", exact_exc)
            # A question with ONE right answer gets checked here, whatever lane
            # produced the text. Run 7 served these for arithmetic turns:
            #   'Get bit by Anaconda. Spend extra time roaming around…'
            #   'Already solved internalmente'
            # — a small lane answering a problem it cannot do, and the answer
            # delivered. assess_user_facing_reply catches all of them as hard
            # failures; the path that served them simply never asked it.
            #
            # This is the last gate before a reply reaches a person and 34 call
            # sites pass through it, so the check belongs here rather than in
            # each of them. Only the deterministic arithmetic verdict is used:
            # it is the one judgement that is right or wrong rather than a
            # matter of style, so it can be applied to every path safely.
            final_text, status = _hold_a_reasoning_answer_to_its_contract(
                _semantic_user_message=_semantic_user_message,
                final_text=final_text,
                status=status,
            )
            if _grounded_recall_context:
                from core.conversation.grounded_recall import (
                    grounded_quote_from_context,
                    repair_grounded_recall_speaker_attribution,
                )

                final_text, _ = repair_grounded_recall_speaker_attribution(
                    _semantic_user_message,
                    final_text,
                    grounded_quote_from_context(_grounded_recall_context),
                )
            response_confidence = "high"
            hard_fastpath_quality_failed = False
            proof_status = str(status or "")
            is_governed_action_status = _status_represents_governed_action_result(proof_status)
            is_memory_state_status = _status_represents_memory_state_result(proof_status)
            state_projection_authority = _verified_state_projection_authority(
                proof_status
            )

            _new_text, _new_status = await _apply_desktop_objective_chokepoint(
                final_text, proof_status
            )
            if _new_status != proof_status:
                final_text = _new_text
                proof_status = _new_status
                status = _new_status
                # Receipt summaries are evidence, not prose: skip the
                # conversational staleness/topicality reshaping below.
                is_governed_action_status = _status_represents_governed_action_result(proof_status)
                is_memory_state_status = _status_represents_memory_state_result(proof_status)
                state_projection_authority = _verified_state_projection_authority(
                    proof_status
                )

            # A deterministic tool-result response has an author, but it is
            # not the language model. Bind that authority after the shared
            # execution chokepoint, since that chokepoint can turn an ordinary
            # cognitive status into a completed desktop action. This is
            # positive evidence: every requested critical step must carry
            # verified effect evidence. Taking a fast path proves nothing.
            if is_governed_action_status and str(proof_status).startswith(
                "desktop_objective"
            ):
                desktop_result = _desktop_exec_state.get("result")
                authority_proven, authority_reason = (
                    _governed_desktop_response_authority(
                        desktop_result=desktop_result,
                        action_episode=_desktop_exec_state.get("action_episode"),
                    )
                )
                _live_turn_trace.update(
                    {
                        "response_authority_kind": (
                            "verified_action_receipt_serialization"
                        ),
                        "response_authority_proven": authority_proven,
                        "response_authority_reason": authority_reason,
                        "model_generation_used": False,
                        "live_mind_generation_required": False,
                        "semantic_completion_contract_expected": True,
                        "semantic_completion_receipt_present": authority_proven,
                        "semantic_completion_satisfied": authority_proven,
                    }
                )
            elif state_projection_authority is not None:
                authority_kind, authority_reason = state_projection_authority
                _live_turn_trace.update(
                    {
                        "response_path": proof_status,
                        "response_authority_kind": authority_kind,
                        "response_authority_proven": True,
                        "response_authority_reason": authority_reason,
                        "model_generation_used": False,
                        "live_mind_generation_required": False,
                        "semantic_completion_contract_expected": True,
                        "semantic_completion_receipt_present": True,
                        "semantic_completion_satisfied": True,
                    }
                )

            if not (
                is_governed_action_status
                or is_memory_state_status
                or state_projection_authority is not None
                or proof_status == "cognitive_engine_qualified_recurrent"
                or proof_status == "protected_foreground"
            ):
                final_text = await _ground_executable_output_claims_for_delivery(
                    _live_turn_trace,
                    final_text,
                )

            if is_benchmark:
                blocked_reply = (
                    "Benchmark request attempted to use a non-canonical chat fastpath "
                    f"({status}). Proof traffic must route through KernelInterface."
                )
                await _emit_chat_output_receipt(
                    blocked_reply,
                    cause=f"chat_fastpath:{status}",
                    metadata={
                        "status": status,
                        "path": "benchmark_fastpath_blocked",
                        "confidence": "failed",
                    },
                )
                return JSONResponse(
                    {
                        "response": blocked_reply,
                        "status": "benchmark_fastpath_blocked",
                        "conversation_lane": _chat_preflight._collect_conversation_lane_status(),
                        "response_confidence": "failed",
                    },
                    status_code=409,
                )

            try:
                if not (
                    is_governed_action_status
                    or is_memory_state_status
                    or state_projection_authority is not None
                ):
                    recent_user_messages = await _gather_recent_user_messages_for_relevance(
                        _semantic_user_message
                    )
                    is_stale = _is_actionably_stale_response(
                        _semantic_user_message,
                        final_text,
                    )
                    is_same_diff = _is_same_answer_different_prompt(
                        _semantic_user_message, final_text
                    )
                    is_off_topic, off_topic_reason = _evaluate_reply_topicality(
                        _semantic_user_message,
                        final_text,
                        recent_user_messages=recent_user_messages,
                    )
                    semantic_glitch, semantic_glitch_reason = _looks_semantically_glitched(
                        _semantic_user_message, final_text
                    )
                    assess_user_facing_reply = None
                    try:
                        from core.conversation.response_reliability import (
                            assess_user_facing_reply as _assess_user_facing_reply,
                        )

                        assess_user_facing_reply = _assess_user_facing_reply
                        fastpath_assessment = assess_user_facing_reply(
                            _semantic_user_message,
                            final_text,
                            recent_user_messages=recent_user_messages,
                        )
                    except ImportError:
                        fastpath_assessment = None
                    if (
                        is_stale
                        or is_same_diff
                        or is_off_topic
                        or semantic_glitch
                        or _reply_assessment_requires_repair(fastpath_assessment)
                    ):
                        hard_fastpath_quality_failed = bool(
                            is_off_topic
                            or semantic_glitch
                            or (_reply_assessment_requires_repair(fastpath_assessment))
                        )
                        response_confidence = "degraded"
                        (
                            repaired_text,
                            is_stale,
                            is_same_diff,
                            is_off_topic,
                            off_topic_reason,
                            repaired,
                        ) = await _repair_final_degraded_reply_with_provenance(
                            _live_turn_trace,
                            stage="chat.fastpath_final_gate",
                            user_message=_semantic_user_message,
                            reply_text=final_text,
                            stale=is_stale,
                            same_diff=is_same_diff,
                            off_topic=is_off_topic,
                            off_topic_reason=off_topic_reason or semantic_glitch_reason,
                            desktop_cognitive_engine_required=desktop_requires_cognitive_engine,
                            protected_foreground_lane=desktop_requires_cognitive_engine,
                            session_id=_chat_session_id,
                        )
                        if repaired and repaired_text != final_text:
                            final_text = repaired_text
                            semantic_glitch, semantic_glitch_reason = _looks_semantically_glitched(
                                _semantic_user_message, final_text
                            )
                            try:
                                # Imported, not guarded. Assigning to a name
                                # anywhere in a function makes it local for
                                # the whole function, so the guard that read
                                # it first — "if assess_user_facing_reply is
                                # None" — could only ever raise
                                # UnboundLocalError, and did: a live turn came
                                # back status=error with
                                # chat.uncaught_turn_error.
                                from core.conversation.response_reliability import (
                                    assess_user_facing_reply as _assess_reply,
                                )

                                fastpath_assessment = _assess_reply(
                                    _semantic_user_message,
                                    final_text,
                                    recent_user_messages=recent_user_messages,
                                )
                            except ImportError:
                                fastpath_assessment = None
                            hard_fastpath_quality_failed = bool(
                                is_off_topic
                                or semantic_glitch
                                or (_reply_assessment_requires_repair(fastpath_assessment))
                            )
                            if not (
                                is_stale
                                or is_same_diff
                                or is_off_topic
                                or semantic_glitch
                                or _reply_assessment_requires_repair(fastpath_assessment)
                            ):
                                response_confidence = "high"
            except (AttributeError, RuntimeError, TypeError, ValueError) as fastpath_gate_exc:
                record_degradation("chat", fastpath_gate_exc)
                logger.debug("Fastpath final quality gate skipped: %s", fastpath_gate_exc)

            if (
                desktop_requires_cognitive_engine
                and response_confidence == "degraded"
                and hard_fastpath_quality_failed
                and not (
                    is_governed_action_status
                    or is_memory_state_status
                    or state_projection_authority is not None
                )
            ):
                return await _fail_closed_degraded_desktop_reply(
                    final_text,
                    response_path="desktop_required_fastpath_quality_failed",
                )

            final_text, output_contract_proven = _enforce_main_requested_output_contract(final_text)
            if not output_contract_proven:
                return await _fail_closed_degraded_desktop_reply(
                    final_text,
                    response_path="fastpath_requested_output_contract_not_proven",
                    status="requested_output_contract_not_proven",
                    reason="requested_output_contract_not_proven",
                )

            protected_foreground_origin = bool(
                _live_turn_trace.get("response_path") == "protected_foreground"
            )
            if protected_foreground_origin:
                expected_sha256 = str(
                    _live_turn_trace.get("foreground_model_generation_output_sha256")
                    or ""
                ).strip()
                delivered_sha256 = hashlib.sha256(
                    final_text.encode("utf-8")
                ).hexdigest()
                if not _protected_foreground_bytes_unchanged(
                    _live_turn_trace,
                    status=status,
                    reply_text=final_text,
                ):
                    logger.error(
                        "Protected foreground bytes changed after worker authorship "
                        "was bound (status=%s expected=%s delivered=%s).",
                        status,
                        expected_sha256[:12] or "missing",
                        delivered_sha256[:12],
                    )
                    return await _fail_closed_degraded_desktop_reply(
                        final_text,
                        response_path="protected_foreground_bytes_changed",
                        status="protected_foreground_bytes_changed",
                        reason="protected_foreground_bytes_changed",
                    )
                protected_contract = _live_turn_contract(
                    lane_status=_chat_preflight._collect_conversation_lane_status(),
                    response_confidence=response_confidence,
                    status=status,
                    reply_source="protected_foreground",
                )
                if not _authored_answer_can_serve(protected_contract):
                    logger.error(
                        "Protected foreground answer failed the shared delivery "
                        "contract; withholding it (missing=%s).",
                        ",".join(
                            protected_contract.get("full_mind_missing_proofs") or ()
                        )
                        or "unknown",
                    )
                    return await _fail_closed_degraded_desktop_reply(
                        final_text,
                        response_path="protected_foreground_delivery_unproven",
                        status="protected_foreground_delivery_unproven",
                        reason="protected_foreground_delivery_unproven",
                    )

            _record_recent_response(final_text, _semantic_user_message)

            lane_status = (
                _collect_governed_action_lane_status(status)
                if _status_represents_governed_action_result(status)
                else _chat_preflight._collect_conversation_lane_status()
            )
            live_contract = _live_turn_contract(
                lane_status=lane_status,
                response_confidence=response_confidence,
                status=status,
                reply_source="fastpath",
            )
            if (
                assertion_response is not None
                and str(getattr(assertion_response, "text", "")) == final_text
            ):
                try:
                    authority = assertion_response.authority()
                except (AttributeError, TypeError, ValueError) as exc:
                    record_degradation("chat.assertion_response_authority", exc)
                else:
                    live_contract["verified_assertion_response"] = authority

            response_data = {
                "response": final_text,
                "status": status,
                "conversation_lane": lane_status,
                "response_confidence": response_confidence,
                "live_turn_contract": live_contract,
            }
            if _desktop_exec_state.get("result") is not None and str(status).startswith(
                "desktop_objective"
            ):
                response_data["data"] = {
                    "desktop_result": _json_safe_payload(_desktop_exec_state["result"])
                }
            elif str(status).startswith("desktop_objective"):
                logger.warning(
                    "Desktop receipts NOT attached at fastpath door: "
                    "result_present=%r attempted=%r",
                    _desktop_exec_state.get("result") is not None,
                    _desktop_exec_state.get("attempted"),
                )
            if pending_exchange_id:
                await _chat_preflight._complete_logged_exchange(
                    pending_exchange_id,
                    _semantic_user_message,
                    final_text,
                    exchange_metadata=_current_exchange_metadata(
                        final_text,
                        response_path=status,
                    ),
                )
                pending_exchange_id = None
            else:
                await _chat_preflight._log_exchange(
                    _original_user_message,
                    final_text,
                    session_id=_chat_session_id,
                    exchange_metadata=_current_exchange_metadata(
                        final_text,
                        response_path=status,
                    ),
                )
            await _emit_chat_output_receipt(
                final_text,
                cause=f"chat_fastpath:{status}",
                metadata={"status": status, "path": "fastpath", "confidence": response_confidence},
            )
            return JSONResponse(response_data)

        async def _attempt_protected_foreground_reply(
            reason: str,
            *,
            budget_override_s: float | None = None,
        ) -> str | None:
            """The turn's nine names, handed to the lifted lane."""
            return await _protected_foreground_reply(
                reason,
                budget_override_s=budget_override_s,
                _chat_session_id=_chat_session_id,
                _live_turn_trace=_live_turn_trace,
                _remaining_foreground_budget=_remaining_foreground_budget,
                _semantic_user_message=_semantic_user_message,
                body=body,
                chat_origin=chat_origin,
                desktop_requires_cognitive_engine=desktop_requires_cognitive_engine,
                is_benchmark=is_benchmark,
                lane=lane,
            )

        async def _execute_narrow_desktop_objective_before_cognition() -> JSONResponse | None:
            """Forwards to the lifted _lifted_execute_narrow_desktop_objective_before_cognition."""
            return await _lifted_execute_narrow_desktop_objective_before_cognition(
                _semantic_user_message=_semantic_user_message,
                conversation_only_surface=conversation_only_surface,
                is_benchmark=is_benchmark,
                _finalize_fastpath=_finalize_fastpath,
                _run_desktop_objective_tracked=_run_desktop_objective_tracked,
            )

        if not _qualified_state_serialization_owner:
            try:
                prior_answer_provenance = await _resolve_prior_answer_provenance(
                    _semantic_user_message,
                    session_id=_chat_session_id,
                )
            except _CHAT_RECOVERABLE_ERRORS as provenance_context_exc:
                record_degradation("chat.answer_provenance_context", provenance_context_exc)
                prior_answer_provenance = None
        else:
            prior_answer_provenance = None

        if not _qualified_state_serialization_owner:
            try:
                (
                    action_episode_evidence,
                    action_episode_projection,
                ) = await _resolve_action_episode_projection(
                    _semantic_user_message,
                    session_id=_chat_session_id,
                )
            except _CHAT_RECOVERABLE_ERRORS as action_context_exc:
                record_degradation("chat.action_episode_context", action_context_exc)
                action_episode_evidence = ""
                action_episode_projection = ""

        if action_episode_projection:
            return await _finalize_fastpath(
                action_episode_projection,
                status="verified_action_episode",
            )

        if (
            not is_benchmark
            and not conversation_only_surface
            and not _qualified_state_serialization_owner
            and not action_episode_evidence
        ):
            governed_capability_response = await _chat_capability_inventory._execute_governed_capability_request_from_chat(
                _semantic_user_message
            )
            if governed_capability_response is not None:
                return await _finalize_fastpath(
                    _chat_desktop_repair._apply_aura_voice_shaping(
                        str(governed_capability_response.get("response") or "")
                    ),
                    status=str(governed_capability_response.get("status") or "governed_capability"),
                )

        desktop_objective_response = (
            None
            if _qualified_state_serialization_owner
            else await _execute_narrow_desktop_objective_before_cognition()
        )
        if desktop_objective_response is not None:
            return desktop_objective_response

        if (
            not is_benchmark
            and desktop_requires_cognitive_engine
            and not conversation_only_surface
            and not _qualified_state_serialization_owner
        ):
            desktop_memory_state_evidence = (
                await _chat_memory_state._build_memory_state_fastpath_reply(
                    _semantic_user_message,
                    session_id=_chat_session_id,
                    owner_session_restored=owner_session_restored,
                    as_evidence=True,
                )
            )

        if allow_memory_state_fastpath:
            memory_state_reply = await _chat_memory_state._build_memory_state_fastpath_reply(
                _semantic_user_message,
                session_id=_chat_session_id,
                owner_session_restored=owner_session_restored,
            )
            if memory_state_reply:
                memory_reply, memory_status = memory_state_reply
                return await _finalize_fastpath(
                    memory_reply,
                    status=memory_status,
                )

        if allow_runtime_status_fastpath:
            runtime_fact_status = _build_runtime_fact_status_fastpath_reply(
                _semantic_user_message,
                lane,
            )
            if runtime_fact_status:
                return await _finalize_fastpath(
                    runtime_fact_status,
                    status="runtime_fact_status",
                )

        if allow_chat_fastpaths:
            bounded_plan_reply = _chat_desktop_repair._build_bounded_planning_reply(
                _semantic_user_message
            )
            if bounded_plan_reply:
                return await _finalize_fastpath(
                    bounded_plan_reply,
                    status="cognitive_engine_bounded_planning",
                )
            failure_mode_reply = _chat_desktop_repair._build_failure_mode_surface_reply(
                _semantic_user_message
            )
            if failure_mode_reply:
                return await _finalize_fastpath(
                    failure_mode_reply,
                    status="cognitive_engine_failure_mode_surface",
                )

        if allow_chat_fastpaths and _chat_preflight._is_explicit_capability_inventory_request(
            _semantic_user_message
        ):
            return await _finalize_fastpath(
                _chat_desktop_repair._build_grounded_capability_inventory_reply(
                    _semantic_user_message
                ),
                status="cognitive_engine_capability_inventory",
            )

        if (
            allow_chat_fastpaths
            and _chat_desktop_repair._is_low_risk_social_continuity_request(_semantic_user_message)
            and not _conversation_lane_blocks_fallback(lane)
        ):
            return await _finalize_fastpath(
                _chat_desktop_repair._build_social_continuity_repair_reply(_semantic_user_message),
                status="social_presence_reflex",
            )

        diagnostic_target = None

        # Background file diagnostic
        try:
            from core.conversation.demo_support import (
                build_background_diagnostic_ack,
                extract_background_diagnostic_target,
                run_background_file_diagnostic,
            )

            orch = ServiceContainer.get("orchestrator", default=None)
            if orch and allow_chat_fastpaths:
                diagnostic_target = extract_background_diagnostic_target(_semantic_user_message)
                if diagnostic_target:
                    # Use a local bounded task — we don't have _spawn_server_bounded_task here
                    get_task_tracker().track(
                        run_background_file_diagnostic(diagnostic_target, orch)
                    )
                    return await _finalize_fastpath(
                        _chat_desktop_repair._apply_aura_voice_shaping(
                            build_background_diagnostic_ack(diagnostic_target)
                        ),
                        status="background_diagnostic_started",
                    )
        except _CHAT_RECOVERABLE_ERRORS as _bg_exc:
            record_degradation("chat", _bg_exc)
            logger.debug("Background diagnostic launch skipped: %s", _bg_exc)

        if allow_governed_action_fastpaths:
            explicit_file = await _execute_explicit_local_file_objective(_semantic_user_message)
            if explicit_file:
                return await _finalize_fastpath(
                    _chat_desktop_repair._apply_aura_voice_shaping(
                        str(explicit_file.get("response") or "")
                    ),
                    status=str(explicit_file.get("status") or "file_operation"),
                )

        if allow_chat_fastpaths:
            live_proof = await _chat_runtime_proof._execute_live_runtime_proof(_semantic_user_message)
            if live_proof:
                return await _finalize_fastpath(
                    _chat_desktop_repair._apply_aura_voice_shaping(
                        str(live_proof.get("response") or "")
                    ),
                    status=str(live_proof.get("status") or "live_proof"),
                )

        protected_foreground_reason = (
            _protected_foreground_reason(lane)
            if not is_benchmark and not desktop_requires_cognitive_engine
            else None
        )
        if protected_foreground_reason:
            protected_reply = await _attempt_protected_foreground_reply(protected_foreground_reason)
            if protected_reply:
                return await _finalize_fastpath(
                    protected_reply,
                    status="protected_foreground",
                )
            if protected_foreground_reason == "recovery_cooldown":
                # [STABILITY v55] Don't 503-reject during recovery cooldown.
                # The cooldown is only 1s — let the request flow through the
                # normal kernel path instead of showing a canned error message.
                logger.info(
                    "🛡️ Recovery cooldown: skipping protected foreground, proceeding to kernel."
                )

        if allow_chat_fastpaths and not bool(lane.get("conversation_ready", False)):
            # Same rule as the desktop admission barrier below: a gate that has
            # not registered yet during boot is worth waiting for.
            gate = await _await_foreground_gate(
                budget_s=min(
                    _FOREGROUND_GATE_BOOT_WAIT_S,
                    max(0.0, _remaining_foreground_budget(reserve=30.0)),
                )
            )
            if gate and hasattr(gate, "ensure_foreground_ready"):
                # Give a cold/recovering cortex a real chance to come online
                # before we concede to a fallback lane. The previous 12s cap
                # was too aggressive and caused repeated user-visible warming
                # loops under normal boot and recovery conditions.
                warmup_budget = min(180.0, _remaining_foreground_budget(reserve=30.0))
                try:
                    lane = await gate.ensure_foreground_ready(timeout=max(1.0, warmup_budget))
                except TimeoutError:
                    lane = _mark_conversation_lane_state(
                        "foreground_warmup_timeout",
                        state="warming",
                    )
                    # [STABILITY v51] Warming-with-response: instead of returning
                    # a 503 "still warming" message, try the protected foreground
                    # lane. The user gets a fast response while cortex warms in
                    # the background for the next message.
                    _warmup_bypass_reply = await _attempt_protected_foreground_reply(
                        "warmup_timeout_bypass"
                    )
                    if _warmup_bypass_reply:
                        # Fire-and-forget cortex recovery for the next request
                        if gate and hasattr(gate, "_schedule_background_cortex_prewarm"):
                            try:
                                gate._schedule_background_cortex_prewarm(delay=1.0)
                            except _CHAT_RECOVERABLE_ERRORS as exc:
                                record_degradation("chat", exc)
                                logger.debug("Background cortex prewarm scheduling failed: %s", exc)
                        return await _finalize_fastpath(
                            _warmup_bypass_reply,
                            status="protected_foreground",
                        )
                except _CHAT_RECOVERABLE_ERRORS as exc:
                    record_degradation("chat", exc)
                    failure_reason = str(exc or "foreground_warmup_failed")
                    lane = _mark_conversation_lane_state(
                        failure_reason,
                        state="failed"
                        if failure_reason.startswith(
                            ("mlx_runtime_unavailable:", "local_runtime_unavailable:")
                        )
                        else "recovering",
                    )
                    # [STABILITY v51] Same warming-with-response pattern for
                    # warmup failures — try protected lane before giving up.
                    if not failure_reason.startswith(
                        ("mlx_runtime_unavailable:", "local_runtime_unavailable:")
                    ):
                        _failure_bypass_reply = await _attempt_protected_foreground_reply(
                            "warmup_failure_bypass"
                        )
                        if _failure_bypass_reply:
                            if gate and hasattr(gate, "_schedule_background_cortex_prewarm"):
                                try:
                                    gate._schedule_background_cortex_prewarm(delay=2.0)
                                except _CHAT_RECOVERABLE_ERRORS as exc:
                                    record_degradation("chat", exc)
                                    logger.debug(
                                        "Background cortex prewarm scheduling failed: %s", exc
                                    )
                            return await _finalize_fastpath(
                                _failure_bypass_reply,
                                status="protected_foreground",
                            )

        if allow_chat_fastpaths and _conversation_lane_blocks_fallback(lane):
            # [STABILITY v55] Try protected foreground BEFORE returning 503.
            # Cloud or brainstem can still serve while the cortex recovers.
            try:
                gate = ServiceContainer.get("inference_gate", default=None)
                if gate and hasattr(gate, "_schedule_background_cortex_prewarm"):
                    gate._schedule_background_cortex_prewarm(delay=2.0)
            except _CHAT_RECOVERABLE_ERRORS as exc:
                record_degradation("chat", exc)
                logger.debug("Background cortex prewarm scheduling failed: %s", exc)
            rescue_reply = await _attempt_protected_foreground_reply("lane_hard_failure")
            if rescue_reply:
                return await _finalize_fastpath(
                    rescue_reply,
                    status="protected_foreground",
                )
            return JSONResponse(
                {
                    "response": _conversation_lane_user_message(lane),
                    "status": "conversation_unavailable",
                    "conversation_lane": lane,
                },
                # In-band for real users (the lane message IS the answer);
                # strict 503 stays benchmark-only. This producer surfaced as
                # bare 'HTTP Error 503' in the nightcap soak.
                status_code=503 if is_benchmark else 200,
            )

        if allow_chat_fastpaths:
            try:
                repo_probe = await _chat_memory_state._await_bounded_chat_blocking(
                    _read_repo_probe_reply,
                    _semantic_user_message,
                    timeout_s=_CHAT_BLOCKING_PREFLIGHT_TIMEOUT_S,
                    operation_name="repo_probe_read",
                    completion_grace_s=_CHAT_BLOCKING_PREFLIGHT_TIMEOUT_S,
                )
            except TimeoutError:
                repo_probe = {
                    "reply": (
                        "The bounded live file read did not finish in time, so I "
                        "did not report a partial result as complete."
                    ),
                    "status": "repo_probe_timeout",
                }
            if repo_probe:
                return await _finalize_fastpath(
                    _chat_desktop_repair._apply_aura_voice_shaping(
                        str(repo_probe.get("reply") or "")
                    ),
                    status=str(repo_probe.get("status") or "repo_probe"),
                )

            grounded_traceability = await _build_grounded_traceability_reply(_semantic_user_message)
            if grounded_traceability:
                return await _finalize_fastpath(
                    grounded_traceability,
                    status="grounded_traceability",
                )

        # Simple affect checks ("how are you doing") go through the LLM
        # for natural responses instead of returning a template.

        if allow_chat_fastpaths and _is_identity_challenge_request(_semantic_user_message):
            return await _finalize_fastpath(
                _build_identity_challenge_reply(_semantic_user_message),
                status="identity_challenge_reflex",
            )

        asks_internal_state, asks_free_energy, asks_topology, asks_authority = (
            _chat_conversation_repair._classify_grounded_introspection_request(
                _semantic_user_message
            )
        )
        grounded_introspection = (
            _chat_conversation_repair._build_grounded_introspection_reply(_semantic_user_message)
            if allow_chat_fastpaths
            else None
        )
        _seam_early_response = await _answer_from_grounded_introspection(
            _finalize_fastpath=_finalize_fastpath,
            _semantic_user_message=_semantic_user_message,
            asks_authority=asks_authority,
            grounded_introspection=grounded_introspection,
        )
        if _seam_early_response is not _SEAM_FELL_THROUGH:
            return _seam_early_response

        if allow_chat_fastpaths and _chat_desktop_repair._is_identity_request(
            _semantic_user_message
        ):
            return await _finalize_fastpath(
                _chat_desktop_repair._build_identity_reply(_semantic_user_message),
                status="identity_reflex",
            )

        if allow_chat_fastpaths and _chat_preflight._is_capability_request(_semantic_user_message):
            return await _finalize_fastpath(
                _build_capability_reply(_semantic_user_message),
                status="capability_reflex",
            )

        if allow_chat_fastpaths and _chat_preflight._is_self_diagnostic_request(
            _semantic_user_message
        ):
            return await _finalize_fastpath(
                _build_self_diagnostic_reply(_semantic_user_message),
                status="self_diagnostic",
            )

        if allow_chat_fastpaths and _is_simple_affect_check_request(_semantic_user_message):
            self_condition_reply = _build_grounded_self_condition_reply(_semantic_user_message)
            if self_condition_reply:
                return await _finalize_fastpath(
                    self_condition_reply,
                    status="self_condition",
                )

        # Verified self-evidence is a typed runtime read, not a model-authored
        # shortcut. Desktop-origin turns disable ordinary chat fast paths so
        # the full mind owns authored prose; that must not force a model call
        # to restate signed measurements it can only make less accurate.
        if not is_benchmark:
            try:
                from core.brain.cortex_self_evidence import (
                    render_cortex_evidence_response,
                )

                cortex_evidence_response = render_cortex_evidence_response(
                    _semantic_user_message
                )
            except _CHAT_RECOVERABLE_ERRORS as exc:
                record_degradation("chat.cortex_self_evidence", exc)
                cortex_evidence_response = None
            if cortex_evidence_response is not None:
                return await _finalize_fastpath(
                    cortex_evidence_response.text,
                    status="cortex_self_evidence",
                    assertion_response=cortex_evidence_response,
                )

        try:
            from core.conversation.demo_support import (
                maybe_build_priority_focus_reply,
                maybe_build_recent_activity_reply,
            )

            orch = ServiceContainer.get("orchestrator", default=None)
            if orch and not is_benchmark:
                recent_activity_reply = await maybe_build_recent_activity_reply(
                    _semantic_user_message, orch
                )
                if recent_activity_reply:
                    return await _finalize_fastpath(
                        _chat_desktop_repair._apply_aura_voice_shaping(recent_activity_reply),
                        status="recent_activity",
                    )

            if orch and allow_chat_fastpaths:
                priority_focus_reply = await maybe_build_priority_focus_reply(
                    _semantic_user_message, orch
                )
                if priority_focus_reply:
                    return await _finalize_fastpath(
                        _chat_desktop_repair._apply_aura_voice_shaping(priority_focus_reply),
                        status="priority_focus",
                    )
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat", exc)
            logger.debug("Demo-support fast paths skipped: %s", exc)

        if allow_chat_fastpaths and _chat_preflight._is_architecture_self_assessment_request(
            _semantic_user_message
        ):
            return await _finalize_fastpath(
                _chat_desktop_repair._apply_aura_voice_shaping(
                    _build_architecture_self_reflex(
                        _chat_desktop_repair._build_aura_expression_frame(_semantic_user_message),
                        _semantic_user_message,
                    )
                ),
                status="architecture_self_reflex",
            )

        # Crash-safe persistence: persist the user's message BEFORE calling
        # the LLM. If the process dies mid-inference, the message is preserved
        # and the conversation can be resumed. (Pattern from Claude Code.)
        effective_user_message = _semantic_user_message
        referential_anchor = (
            await _resolve_referential_followup_anchor(
                _semantic_user_message,
                session_id=_chat_session_id,
            )
            if (
                allow_chat_fastpaths
                and not _qualified_state_serialization_owner
                and not action_episode_evidence
            )
            else None
        )
        if referential_anchor:
            effective_user_message = (
                f"{_semantic_user_message}\n\n"
                "[REFERENTIAL ANCHOR]\n"
                "The user is referring to this earlier user question/request:\n"
                f"{referential_anchor}"
            )
        if action_episode_evidence:
            from core.conversation.turn_evidence_custody import record_turn_grounding

            record_turn_grounding(action_episode_evidence)
        conversation_recall_evidence = (
            None
            if _qualified_state_serialization_owner
            else await _chat_memory_state._build_conversation_recall_reply(
                _semantic_user_message,
                session_id=_chat_session_id,
            )
        )
        if conversation_recall_evidence:
            from core.conversation.turn_evidence_custody import record_turn_grounding

            record_turn_grounding(conversation_recall_evidence)
            effective_user_message = (
                f"{effective_user_message}\n\n"
                "[CONVERSATION RECALL EVIDENCE]\n"
                f"{conversation_recall_evidence}\n"
                "[END CONVERSATION RECALL EVIDENCE]\n"
                "Answer the recall question from the evidence above. Do not guess or invent a memory."
            )
        retained_memory_evidence = (
            ""
            if conversation_only_surface or _qualified_state_serialization_owner
            else await _build_retained_memory_evidence_context(
                _semantic_user_message,
                session_id=_chat_session_id,
                conversation_recall_context=conversation_recall_evidence or "",
            )
        )
        if retained_memory_evidence:
            from core.conversation.turn_evidence_custody import record_turn_grounding

            record_turn_grounding(retained_memory_evidence)
            effective_user_message = (
                f"{effective_user_message}\n\n"
                "[RETAINED MEMORY EVIDENCE]\n"
                f"{retained_memory_evidence}\n"
                "[END RETAINED MEMORY EVIDENCE]\n"
                "For any claim about what you remember, what persisted, or what happened in a prior "
                "session, use the evidence above. If the evidence is absent or insufficient, say that "
                "the memory is not verified instead of filling the gap."
            )
        if desktop_memory_state_evidence:
            memory_reply, memory_status = desktop_memory_state_evidence
            from core.conversation.turn_evidence_custody import record_turn_grounding

            record_turn_grounding(f"status={memory_status}\n{memory_reply}")
            effective_user_message = (
                f"{effective_user_message}\n\n"
                "[CANONICAL MEMORY STATE EVIDENCE]\n"
                f"status={memory_status}\n"
                f"{memory_reply}\n"
                "[END CANONICAL MEMORY STATE EVIDENCE]\n"
                "Use this canonical memory/state result as evidence, but produce the visible answer "
                "through CognitiveEngine in Aura's normal desktop voice."
            )
        # The address the person named, read before anything else is decided.
        named_url_evidence = await _collect_named_url_evidence(_semantic_user_message)
        if named_url_evidence and named_url_evidence.get("ok"):
            effective_user_message = (
                f"{effective_user_message}\n\n"
                "[PAGE THE USER NAMED]\n"
                f"url: {named_url_evidence.get('url')}\n"
                f"title: {named_url_evidence.get('title')}\n"
                f"{named_url_evidence.get('text')}\n"
                "[END PAGE THE USER NAMED]\n"
                "This is the document they addressed. Answer from it, and say so "
                "plainly if it does not contain what they asked for."
            )
        elif named_url_evidence:
            effective_user_message = (
                f"{effective_user_message}\n\n"
                "[PAGE THE USER NAMED]\n"
                f"url: {named_url_evidence.get('url')}\n"
                f"could not be read: {named_url_evidence.get('error')}\n"
                "[END PAGE THE USER NAMED]\n"
                "Say that the address could not be read, and what was tried."
            )

        desktop_required_search_evidence = None
        if (
            not is_benchmark
            and desktop_requires_cognitive_engine
            and not conversation_only_surface
            and not _qualified_state_serialization_owner
        ):
            desktop_required_search_evidence = await _collect_desktop_required_search_evidence(
                _semantic_user_message,
                session_id=_chat_session_id,
            )
            if desktop_required_search_evidence:
                evidence_text = str(desktop_required_search_evidence.get("evidence") or "").strip()
                search_ok = bool(desktop_required_search_evidence.get("ok"))
                memory_saved = bool(desktop_required_search_evidence.get("memory_saved"))
                effective_user_message = (
                    f"{effective_user_message}\n\n"
                    "[WEB SEARCH EVIDENCE]\n"
                    f"{evidence_text}\n"
                    f"memory_saved: {str(memory_saved).lower()}\n"
                    "[END WEB SEARCH EVIDENCE]\n"
                    "The user explicitly requested live search. Use only the evidence above for live factual claims. "
                    "Name the source URLs when present. If ok is false or no usable source is present, say the search did "
                    "not produce reliable evidence instead of answering from memory."
                )
                if not search_ok:
                    logger.warning(
                        "Required desktop search evidence failed before CognitiveEngine reply: query=%s result=%s",
                        desktop_required_search_evidence.get("query"),
                        desktop_required_search_evidence.get("result"),
                    )

        # A desktop-required turn must not enter CognitiveEngine while its
        # resident inference lane is conclusively cold.  Before this barrier,
        # direct engine invocation saw worker_not_alive/init_not_complete and
        # the route later reported a generic full-mind reasoning failure.  The
        # existing inference-gate singleflight is the owner of boot/recovery;
        # wait on it once, preserving enough wall-clock budget for the actual
        # turn, and keep readiness failure distinct from answer failure.
        if (
            not is_benchmark
            and desktop_requires_cognitive_engine
            and not _qualified_state_serialization_owner
            and not bool(lane.get("conversation_ready", False))
        ):
            gate = await _await_foreground_gate(
                budget_s=min(
                    _FOREGROUND_GATE_BOOT_WAIT_S,
                    max(
                        0.0,
                        _remaining_foreground_budget(
                            reserve=_DESKTOP_COGNITIVE_MIN_REQUIRED_BUDGET_S
                            + _DESKTOP_COGNITIVE_RESPONSE_RESERVE_S
                        ),
                    ),
                )
            )
            admission_override, admission_reason, hard_lane_failure, lane = await _admit_to_foreground_lane(
                _remaining_foreground_budget=_remaining_foreground_budget,
                gate=gate,
                lane=lane,
            )

            if admission_reason:
                # The cortex cannot serve this turn. That is a reason to answer
                # with the smaller model, not a reason to talk about lane
                # readiness.
                #
                # _foreground_timeout_for_lane already says this outright —
                # "Give the ladder the time" — and the ladder is real: the
                # Brainstem is loaded, has weights, and is not the lane owner.
                # It was asked ZERO times. Every cold start answered the first
                # message with a sentence about the answer lane instead of the
                # answer, while a warm 9B sat idle.
                ladder_reply = await _answer_from_fallback_ladder(
                    _semantic_user_message,
                    reason=admission_reason,
                    budget_s=_remaining_foreground_budget(
                        reserve=_DESKTOP_COGNITIVE_RESPONSE_RESERVE_S
                    ),
                )
                if ladder_reply:
                    _live_turn_trace.update(
                        {
                            "response_path": "fallback_ladder",
                            "cognitive_lane_admitted": False,
                            "cognitive_lane_admission_reason": admission_reason[:240],
                        }
                    )
                    return JSONResponse(
                        {
                            "response": ladder_reply,
                            "status": "cognitive_engine_fallback_ladder",
                            "reason": admission_reason,
                            "conversation_lane": lane,
                            "response_confidence": "fallback",
                            "live_turn_contract": _live_turn_contract(
                                lane_status=lane,
                                response_confidence="fallback",
                                status="cognitive_engine_fallback_ladder",
                                reply_source="fallback_ladder",
                            ),
                        },
                        status_code=200,
                    )
                _live_turn_trace.update(
                    {
                        "response_path": "required_cognitive_lane_admission",
                        "cognitive_lane_admitted": False,
                        "cognitive_lane_admission_reason": admission_reason[:240],
                    }
                )
                status = "conversation_unavailable" if hard_lane_failure else "conversation_warming"
                response_text = _conversation_lane_user_message(
                    lane,
                    status_override=admission_override,
                )
                logger.warning(
                    "Required desktop CognitiveEngine turn was not admitted before generation: %s",
                    admission_reason,
                )
                return JSONResponse(
                    {
                        "response": response_text,
                        "status": status,
                        "reason": admission_reason,
                        "conversation_lane": lane,
                        "response_confidence": _lane_reply_confidence(
                            response_text, "not_generated"
                        ),
                        "live_turn_contract": _live_turn_contract(
                            lane_status=lane,
                            response_confidence=_lane_reply_confidence(
                                response_text, "not_generated"
                            ),
                            status=status,
                            reply_source="required_cognitive_lane_admission",
                        ),
                    },
                    status_code=503 if is_benchmark else 200,
                    headers={"Retry-After": "2"},
                )
            _live_turn_trace["cognitive_lane_admitted"] = True

        try:
            if not is_benchmark:
                await _preserve_large_user_paste(_semantic_user_message)
            pending_exchange_id = await _chat_preflight._begin_logged_exchange(
                _original_user_message,
                session_id=_chat_session_id,
            )
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat", exc)
            logger.debug("Conversation exchange prelogging skipped: %s", exc)

        reply_text: str | None = None
        reply_source = ""
        _delivery_timing: dict[str, float] = {}
        _cognitive_reply_returned_at = 0.0
        if not is_benchmark and desktop_requires_cognitive_engine:
            if is_shutdown_requested():
                return _runtime_shutdown_response(
                    "before_cognitive_engine",
                    slot_acquired=foreground_slot_acquired,
                )
            cognitive_budget = _desktop_required_cognitive_budget(
                foreground_timeout=foreground_timeout,
                elapsed_s=time.monotonic() - request_started_at,
            )
            if desktop_memory_state_evidence:
                cognitive_budget = min(
                    cognitive_budget,
                    _DESKTOP_MEMORY_STATE_TURN_TIMEOUT_S,
                )
            if cognitive_budget >= _DESKTOP_COGNITIVE_MIN_REQUIRED_BUDGET_S:
                reply_text = await _run_cognitive_engine_chat_turn(
                    effective_user_message,
                    visible_user_message=_semantic_user_message,
                    raw_user_message=_original_user_message,
                    declared_interlocutor=_declared_interlocutor,
                    preflight_context_message=preflight_context_message,
                    turn_sensory_evidence=_turn_sensory_evidence,
                    session_id=_chat_session_id,
                    origin=chat_origin,
                    timeout_s=cognitive_budget,
                    lane=dict(lane or {}),
                    source=("paired_device_ui" if conversation_only_surface else "desktop_ui"),
                    require_engine=True,
                    conversation_only_surface=conversation_only_surface,
                    principal_id=_profile_user_id,
                    turn_trace=_live_turn_trace,
                    referential_anchor=str(referential_anchor or ""),
                    action_episode_evidence=action_episode_evidence,
                    prior_answer_provenance=prior_answer_provenance,
                    conversation_resume_handle=_conversation_resume_handle_for_turn,
                    completed_capability_evidence=desktop_required_search_evidence,
                    evidence_profile=_preflight.evidence_profile,
                )
                _cognitive_reply_returned_at = time.monotonic()
                if reply_text:
                    reply_text = _repair_required_search_reply_provenance(
                        reply_text,
                        desktop_required_search_evidence,
                    )
                    reply_source = (
                        _desktop_required_bounded_reply_status(
                            _semantic_user_message,
                            reply_text,
                            lane,
                        )
                        or "cognitive_engine"
                    )
                    if desktop_requires_cognitive_engine:
                        reply_text = _enforce_or_bind_terminal_output_contract(
                            _live_turn_trace,
                            user_message=_semantic_user_message,
                            reply_text=str(reply_text),
                            desktop_execution_contract=(
                                _chat_preflight._looks_like_desktop_objective(
                                    _semantic_user_message
                                )
                            ),
                        )
                        contract_lane = _chat_preflight._collect_conversation_lane_status()
                        candidate_contract = _live_turn_contract(
                            lane_status=contract_lane,
                            response_confidence="high",
                            status=reply_source,
                            reply_source=reply_source,
                        )
                        if not bool(candidate_contract.get("full_mind_path")):
                            if _authored_answer_can_serve(candidate_contract):
                                # Her real mind authored this text (think invoked,
                                # accepted, high confidence, not repair/legacy) —
                                # only STATE-COMPLETENESS proofs are missing
                                # (snapshot readiness, control receipts, subsystem
                                # health during self-healing). Discarding a genuine
                                # reply here served fail-closed apologies over a
                                # live mind on basic conversation (2026-07-13,
                                # live). Serve it, disclose the degradation in the
                                # contract, and name the missing proofs.
                                logger.warning(
                                    "Desktop turn served with DEGRADED full-mind proof "
                                    "(authentic cognitive reply; missing: %s).",
                                    ",".join(
                                        candidate_contract.get("full_mind_missing_proofs") or ()
                                    ),
                                )
                                _live_turn_trace["full_mind_proof_degraded"] = True
                                lane = contract_lane
                            elif _authored_answer_can_serve_unfinished(
                                candidate_contract
                            ) and str(reply_text or "").strip():
                                # Her own words, unfinished, beat an apology.
                                #
                                # The salvage below exists for exactly this and
                                # this exit never consulted it — the accident
                                # _authored_answer_can_serve's own comment
                                # warns about, where a proof is disclosable at
                                # one exit and fatal at the next.
                                #
                                # LIVE 2026-08-29: asked whether she can invent
                                # a new way of judging a situation, she wrote
                                # 1,056 tokens, was marked
                                # authored_answer_incomplete:retry_exhausted,
                                # and the person got "I couldn't get to an
                                # answer I'd stand behind on that one." An
                                # apology is not more complete than a partial
                                # answer. It carries nothing.
                                logger.warning(
                                    "Desktop turn served her own UNFINISHED answer rather "
                                    "than an apology (missing: %s).",
                                    ",".join(
                                        candidate_contract.get("full_mind_missing_proofs") or ()
                                    ),
                                )
                                _live_turn_trace["full_mind_proof_degraded"] = True
                                _live_turn_trace["authored_answer_unfinished"] = True
                                lane = contract_lane
                            elif _bounded_runtime_grounding_can_serve(candidate_contract):
                                grounded_path = str(
                                    candidate_contract.get("response_path") or "runtime_grounding"
                                )
                                logger.warning(
                                    "Desktop turn retained a bounded runtime projection after "
                                    "model-authored attempts were exhausted (path=%s).",
                                    grounded_path,
                                )
                                _live_turn_trace.update(
                                    {
                                        "cognitive_engine_reply_accepted": False,
                                        "cognitive_engine_reply_failed": True,
                                        "bounded_contract_used": True,
                                        "response_path": grounded_path,
                                    }
                                )
                                reply_source = grounded_path
                                lane = contract_lane
                            else:
                                if (
                                    bool(candidate_contract.get("cognitive_engine_reply_accepted"))
                                    and _chat_preflight._looks_like_desktop_objective(
                                        _semantic_user_message
                                    )
                                    and reply_text
                                ):
                                    _live_turn_trace["desktop_internal_artifact_draft"] = reply_text
                                    _live_turn_trace["desktop_internal_artifact_draft_path"] = str(
                                        candidate_contract.get("response_path") or reply_source
                                    )[:120]
                                # Every receipt ownership can be proven from,
                                # and the token evidence each one carries.
                                #
                                # Naming only the first found reported "present,
                                # tokens unset" while another receipt might have
                                # held the count — which is the same ambiguity
                                # this line exists to remove.
                                _owner_evidence = []
                                for _receipt_key in (
                                    "live_mind_surface_control_receipt",
                                    "surface_control_receipt",
                                    "latent_cortex_receipt",
                                ):
                                    _found = candidate_contract.get(_receipt_key)
                                    if not isinstance(_found, dict) or not _found:
                                        continue
                                    _owner_evidence.append(
                                        "{}(tokens={},decode={},attempts={},applied={})".format(
                                            _receipt_key.replace(
                                                "_surface_control_receipt", ""
                                            ),
                                            _found.get("generated_tokens", "-"),
                                            _found.get("decode_generated_tokens", "-"),
                                            _found.get("surface_quality_gate_attempts", "-"),
                                            _found.get("applied", "-"),
                                        )
                                    )
                                logger.error(
                                    "Desktop CognitiveEngine candidate did not prove authorship "
                                    "(missing: %s); failing closed instead of serving repair "
                                    "text as Aura speech. path=%s generations=%s "
                                    "completion_retries=%s repair_attempts=%s consumed=%s "
                                    "ownership_evidence=[%s] receipts=%d",
                                    ",".join(
                                        candidate_contract.get("full_mind_missing_proofs") or ()
                                    ),
                                    # The counts the proof is ACTUALLY made of.
                                    #
                                    # This printed the name of the failed proof and nothing else,
                                    # so "duplicate_foreground_model_generation" told a reader
                                    # that more than one generation happened and not which of the
                                    # three conditions in single_owner_model_generation_proven
                                    # missed — reconstructing it meant reading the contract
                                    # module beside a log that had already thrown the numbers
                                    # away. The receipt carried all of them the whole time.
                                    candidate_contract.get("response_path") or "unset",
                                    candidate_contract.get("foreground_model_generation_count"),
                                    candidate_contract.get("completion_retry_count"),
                                    candidate_contract.get("repair_retry_attempt_count"),
                                    candidate_contract.get(
                                        "foreground_model_generation_consumed"
                                    ),
                                    # And the evidence ownership is made of.
                                    #
                                    # "foreground_model_generation_ownership_unproven"
                                    # names the proof, not what was missing from
                                    # it, and what it rests on is one number: the
                                    # tokens a surface control receipt says were
                                    # generated. A receipt that never arrived and
                                    # one that arrived empty read identically
                                    # without it, and they are different faults
                                    # with different fixes.
                                    "; ".join(_owner_evidence) or "no receipt",
                                    len(_owner_evidence),
                                )
                                reply_text = None
                                reply_source = ""
                                lane = contract_lane
                        else:
                            lane = contract_lane
                    logger.debug(
                        "REST: CognitiveEngine served desktop chat turn (len=%d).",
                        len(reply_text or ""),
                    )

        desktop_engine_failed = desktop_requires_cognitive_engine and not reply_text
        if desktop_engine_failed:
            # COMPETENCE first: before the bounded-repair cascade or any fail-closed, try a
            # clean grounded regeneration. A casual turn whose draft didn't prove the
            # full-mind path (e.g. "Huh?") should get a real, grounded reply — not a refusal
            # or an empty 503.
            _served_recovery = await _try_serve_grounded_recovery()
            if _served_recovery is not None:
                return _served_recovery

            # Live desktop speech must be the full CognitiveEngine path or an
            # explicitly receipted governed action result. The one exception is
            # a narrow grounded repair after the CognitiveEngine has already
            # been invoked for identity/continuity or self-process questions.
            # Those turns are common daily-use probes; returning a canned 503
            # teaches the UI to stall instead of giving a truthful, bounded
            # explanation of the current state.
            allow_required_desktop_no_reply_repairs = bool(
                _chat_desktop_repair._is_identity_request(_semantic_user_message)
                or _chat_desktop_repair._identity_request_asks_future_memory(_semantic_user_message)
            )
            if not allow_required_desktop_no_reply_repairs:
                try:
                    from core.conversation.response_reliability import (
                        is_live_self_reflection_turn,
                        is_self_process_question,
                    )

                    allow_required_desktop_no_reply_repairs = bool(
                        is_self_process_question(_semantic_user_message)
                        or is_live_self_reflection_turn(_semantic_user_message)
                    )
                except _CHAT_RECOVERABLE_ERRORS as repair_scope_exc:
                    record_degradation("chat", repair_scope_exc)
                    logger.debug(
                        "Desktop no-reply repair scope check skipped: %s",
                        repair_scope_exc,
                    )
            if _chat_desktop_repair._is_low_risk_social_continuity_request(_semantic_user_message):
                social_reply = _chat_desktop_repair._build_social_continuity_repair_reply(
                    _semantic_user_message
                )
                logger.warning(
                    "Desktop CognitiveEngine produced no acceptable reply for low-risk social turn; "
                    "not serving bounded social repair as a successful full-mind desktop turn "
                    "(candidate repair len=%d).",
                    len(social_reply),
                )

            if _chat_preflight._is_runtime_fact_status_request(
                _semantic_user_message
            ) and not _is_current_request_recap_request(_semantic_user_message):
                runtime_grounding = _ground_runtime_fact_status_reply(
                    _semantic_user_message,
                    "",
                    lane,
                    cognitive_engine_handled=True,
                )
                try:
                    from core.conversation.response_reliability import assess_user_facing_reply

                    runtime_recent_user_messages = await _gather_recent_user_messages_for_relevance(
                        _semantic_user_message
                    )
                    runtime_assessment = assess_user_facing_reply(
                        _semantic_user_message,
                        runtime_grounding,
                        recent_user_messages=runtime_recent_user_messages,
                    )
                    runtime_grounding_ok = not _reply_assessment_requires_repair(runtime_assessment)
                except _CHAT_RECOVERABLE_ERRORS as runtime_exc:
                    record_degradation("chat", runtime_exc)
                    logger.debug(
                        "Runtime fact grounding assessment skipped after desktop no-reply: %s",
                        runtime_exc,
                    )
                    runtime_grounding_ok = bool(runtime_grounding)
                if runtime_grounding and runtime_grounding_ok:
                    runtime_grounding, runtime_contract_proven = (
                        _enforce_main_requested_output_contract(runtime_grounding)
                    )
                    if not runtime_contract_proven:
                        runtime_grounding = ""
                if runtime_grounding and runtime_grounding_ok:
                    _live_turn_trace.update(
                        {
                            "cognitive_engine_reply_accepted": False,
                            "cognitive_engine_reply_failed": True,
                            "bounded_contract_used": True,
                            "legacy_fallback_used": False,
                            "response_path": "cognitive_engine_runtime_fact_grounding",
                            "canonical_grounding_used": True,
                        }
                    )
                    lane = _mark_conversation_lane_state(
                        "cognitive_engine_runtime_fact_grounding",
                        state="recovering",
                    )
                    logger.warning(
                        "Desktop CognitiveEngine produced no acceptable runtime/path reply; "
                        "serving canonical runtime-fact grounding after the required engine invocation."
                    )
                    if pending_exchange_id:
                        await _chat_preflight._complete_logged_exchange(
                            pending_exchange_id,
                            _semantic_user_message,
                            runtime_grounding,
                            record_experience=True,
                        )
                        pending_exchange_id = None
                    await _emit_chat_output_receipt(
                        runtime_grounding,
                        cause="chat_response",
                        metadata={
                            "response_confidence": "bounded",
                            "path": "cognitive_engine_runtime_fact_grounding",
                            "status": "cognitive_engine_runtime_fact_grounding",
                            "reason": "desktop_cognitive_engine_required_no_reply",
                        },
                    )
                    return JSONResponse(
                        {
                            "response": runtime_grounding,
                            "status": "cognitive_engine_runtime_fact_grounding",
                            "reason": "desktop_cognitive_engine_required_no_reply",
                            "conversation_lane": lane,
                            "response_confidence": "bounded",
                            "live_turn_contract": _live_turn_contract(
                                lane_status=lane,
                                response_confidence="bounded",
                                status="cognitive_engine_runtime_fact_grounding",
                                reply_source="cognitive_engine_runtime_fact_grounding",
                            ),
                        }
                    )

            if _desktop_objective_executable_after_cognitive_attempt(_semantic_user_message):
                try:
                    internal_artifact_draft = str(
                        _live_turn_trace.get("desktop_internal_artifact_draft") or ""
                    ).strip()
                    executed = await _run_desktop_objective_tracked(
                        _semantic_user_message,
                        cognitive_reply=internal_artifact_draft,
                    )
                except _CHAT_RECOVERABLE_ERRORS as exec_exc:
                    record_degradation("chat", exec_exc)
                    executed = None
                if isinstance(executed, dict) and executed.get("response"):
                    return await _finalize_fastpath(
                        _chat_desktop_repair._apply_aura_voice_shaping(
                            str(executed.get("response") or "")
                        ),
                        status=str(executed.get("status") or "desktop_objective"),
                    )

            identity_repair = (
                _chat_desktop_repair._build_bounded_identity_repair_reply(_semantic_user_message)
                if allow_required_desktop_no_reply_repairs
                else ""
            )
            if identity_repair:
                identity_repair = _chat_desktop_repair._apply_aura_voice_shaping(identity_repair)
                identity_repair, identity_contract_proven = _enforce_main_requested_output_contract(
                    identity_repair
                )
                if not identity_contract_proven:
                    identity_repair = ""
            _seam_early_response, lane, pending_exchange_id = await _serve_the_identity_repair(
                _live_turn_contract=_live_turn_contract,
                _live_turn_trace=_live_turn_trace,
                _semantic_user_message=_semantic_user_message,
                identity_repair=identity_repair,
                lane=lane,
                pending_exchange_id=pending_exchange_id,
            )
            if _seam_early_response is not _SEAM_FELL_THROUGH:
                return _seam_early_response

            capability_inventory = (
                _chat_desktop_repair._build_bounded_capability_inventory_repair_reply(
                    _semantic_user_message
                )
            )
            if capability_inventory:
                if desktop_requires_cognitive_engine:
                    logger.warning(
                        "Desktop CognitiveEngine produced no acceptable capability inventory; "
                        "not serving bounded catalog as a successful live full-mind reply."
                    )
                    capability_inventory = ""
            if capability_inventory:
                capability_inventory = _chat_desktop_repair._apply_aura_voice_shaping(
                    capability_inventory
                )
                capability_inventory, capability_contract_proven = (
                    _enforce_main_requested_output_contract(capability_inventory)
                )
                if not capability_contract_proven:
                    capability_inventory = ""
            _seam_early_response, lane, pending_exchange_id = await _serve_the_capability_inventory(
                _live_turn_contract=_live_turn_contract,
                _live_turn_trace=_live_turn_trace,
                _semantic_user_message=_semantic_user_message,
                capability_inventory=capability_inventory,
                lane=lane,
                pending_exchange_id=pending_exchange_id,
            )
            if _seam_early_response is not _SEAM_FELL_THROUGH:
                return _seam_early_response

            skip_bounded_desktop_repair = (
                _chat_preflight._is_explicit_capability_inventory_request(_semantic_user_message)
                or not allow_required_desktop_no_reply_repairs
            )
            bounded_repair = ""
            # An answer first. This repair describes what she is DOING —
            # "I am tracking what I am keeping in memory inside this
            # conversation... my next move is to answer the actual question" —
            # which is a true sentence and not an answer to anything. LIVE
            # 2026-08-30, asked how to remember people's names at a party, it
            # was the whole reply, while a loaded model that could have
            # answered was never asked.
            if not skip_bounded_desktop_repair:
                bounded_repair = await _anything_better_than_giving_up(
                    _semantic_user_message,
                    reason="the cognitive engine produced no acceptable reply",
                    already="",
                    budget_s=_remaining_foreground_budget(),
                )
            if not bounded_repair and not skip_bounded_desktop_repair:
                bounded_repair = await _build_grounded_self_process_repair_reply(
                    _semantic_user_message,
                    "",
                    lane=lane,
                    session_id=_chat_session_id,
                )
            if bounded_repair:
                bounded_repair = _chat_desktop_repair._apply_aura_voice_shaping(bounded_repair)
                try:
                    from core.conversation.response_reliability import assess_user_facing_reply

                    bounded_recent_user_messages = await _gather_recent_user_messages_for_relevance(
                        _semantic_user_message
                    )
                    bounded_assessment = assess_user_facing_reply(
                        _semantic_user_message,
                        bounded_repair,
                        recent_user_messages=bounded_recent_user_messages,
                    )
                    if _reply_assessment_requires_repair(bounded_assessment):
                        bounded_repair = ""
                except _CHAT_RECOVERABLE_ERRORS as exc:
                    record_degradation("chat", exc)
                    logger.debug("Grounded desktop self-process repair assessment skipped: %s", exc)
            if not bounded_repair and not skip_bounded_desktop_repair:
                minimal_repair = _build_minimal_grounded_self_process_repair_reply(
                    _semantic_user_message,
                    lane=lane,
                )
                if minimal_repair:
                    minimal_repair = _chat_desktop_repair._apply_aura_voice_shaping(minimal_repair)
                    try:
                        from core.conversation.response_reliability import assess_user_facing_reply

                        minimal_recent_user_messages = (
                            await _gather_recent_user_messages_for_relevance(_semantic_user_message)
                        )
                        minimal_assessment = assess_user_facing_reply(
                            _semantic_user_message,
                            minimal_repair,
                            recent_user_messages=minimal_recent_user_messages,
                        )
                        if not _reply_assessment_requires_repair(minimal_assessment):
                            bounded_repair = minimal_repair
                    except _CHAT_RECOVERABLE_ERRORS as exc:
                        record_degradation("chat", exc)
                        logger.debug(
                            "Minimal desktop self-process repair assessment skipped: %s", exc
                        )
            if bounded_repair:
                bounded_repair, bounded_contract_proven = _enforce_main_requested_output_contract(
                    bounded_repair
                )
                if not bounded_contract_proven:
                    bounded_repair = ""
            _seam_early_response, lane, pending_exchange_id = await _serve_the_bounded_repair(
                _live_turn_contract=_live_turn_contract,
                _live_turn_trace=_live_turn_trace,
                _semantic_user_message=_semantic_user_message,
                bounded_repair=bounded_repair,
                lane=lane,
                pending_exchange_id=pending_exchange_id,
            )
            if _seam_early_response is not _SEAM_FELL_THROUGH:
                return _seam_early_response

            # Same contract as the other refusal site: a draft the layers
            # below judged servable is served rather than traded for an
            # apology. The engine produced no ACCEPTABLE reply, which is not
            # the same as producing nothing.
            salvaged_no_reply = _servable_draft_or_none(
                "",
                _semantic_user_message,
                _live_turn_trace.get("turn_id") or _live_turn_trace.get("idempotency_key") or "",
            )
            if salvaged_no_reply:
                (
                    salvaged_no_reply,
                    salvage_output_proven,
                ) = _enforce_main_requested_output_contract(salvaged_no_reply)
                salvage_contract = _live_turn_contract(
                    lane_status=_chat_preflight._collect_conversation_lane_status(),
                    response_confidence="bounded",
                    status=str(_live_turn_trace.get("response_path") or ""),
                    reply_source=str(_live_turn_trace.get("response_path") or ""),
                )
                if not (
                    salvage_output_proven
                    and _authored_answer_can_serve_unfinished(salvage_contract)
                ):
                    logger.warning(
                        "Preserved no-reply draft remained ineligible for delivery; "
                        "withholding it (missing=%s).",
                        ",".join(
                            salvage_contract.get("full_mind_missing_proofs") or ()
                        )
                        or "unknown",
                    )
                    salvaged_no_reply = ""
                else:
                    logger.info(
                        "Serving %d characters of unfinished authored work rather "
                        "than an apology (unproven=%s).",
                        len(salvaged_no_reply),
                        ",".join(
                            salvage_contract.get("full_mind_missing_proofs") or ()
                        )
                        or "none",
                    )
            # A recall question the model could not answer at all.
            #
            # LIVE 2026-08-17: "what was the first thing I said to you in this
            # conversation?" produced "compact desktop generation returned no
            # usable text", three attempts running, and the person got the
            # apology. The transcript was read and delivered that same turn,
            # and the route can compose the answer from it directly.
            #
            # The recall contract already serves this when a draft exists to
            # compare against; with no draft at all, nothing reached it. Same
            # answer, one branch earlier.
            if not salvaged_no_reply:
                try:
                    composed_recall = await _chat_memory_state._build_conversation_recall_reply(
                        _semantic_user_message,
                        session_id=_chat_session_id,
                    )
                except _CHAT_RECOVERABLE_ERRORS as _recall_exc:
                    record_degradation("chat.conversation_recall", _recall_exc)
                    composed_recall = ""
                if composed_recall:
                    logger.warning(
                        "Serving the transcript-composed recall (%d chars) rather than "
                        "refusing: generation produced no usable text.",
                        len(composed_recall),
                    )
                    return await _finalize_fastpath(
                        composed_recall,
                        status="conversation_recall_log_repair_after_empty_engine",
                    )
            # A refusal is not an answer to an instruction she can carry out.
            #
            # Live 2026-07-28: "Open the Notes app and write a new note with
            # three sentences about humpback whales. Actually do it." The
            # route classified it correctly (desktop_execution_contract=True)
            # and the planner produced a complete, executable plan —
            # open_app Notes, set_clipboard, cmd+n, cmd+v. Cognition was asked
            # for the prose first, said "I can't physically interact with your
            # device or open apps", and THAT was served. The hands were right
            # there.
            #
            # Deferring to cognition for better prose is right. Letting its
            # refusal end a turn she can perform is not: when the model
            # declines work the executor can do, the executor does it.
            if _chat_preflight._looks_like_desktop_objective(
                _semantic_user_message
            ) and _looks_like_capability_refusal(salvaged_no_reply):
                logger.info(
                    "Cognition declined a desktop objective it does not own; "
                    "running the governed desktop lane instead of serving the "
                    "refusal."
                )
                # The refusal itself is worthless as a document body, so the
                # executor composes its own rather than inheriting it.
                #
                # Through the tracked gate, not the executor directly: this
                # lane runs real desktop steps, and a lane that bypasses the
                # gate leaves _desktop_exec_state empty, so the reply doors
                # serve a receipt-less reply about work that did happen.
                # That is the exact failure the gate was built for in
                # visible-demo rounds 3-5.
                executed_after_refusal = await _run_desktop_objective_tracked(
                    _semantic_user_message,
                    cognitive_reply="",
                )
                if isinstance(executed_after_refusal, dict) and executed_after_refusal.get(
                    "response"
                ):
                    return await _finalize_fastpath(
                        _chat_desktop_repair._apply_aura_voice_shaping(
                            str(executed_after_refusal.get("response") or "")
                        ),
                        status=str(executed_after_refusal.get("status") or "desktop_objective"),
                    )

            if salvaged_no_reply:
                logger.warning(
                    "Serving the preserved repairable draft (%d chars) rather "
                    "than refusing: the engine produced no acceptable reply.",
                    len(salvaged_no_reply),
                )
                lane = _mark_conversation_lane_state(
                    "cognitive_engine_served_repairable_draft",
                    state="recovering",
                )
                if pending_exchange_id:
                    await _chat_preflight._complete_logged_exchange(
                        pending_exchange_id,
                        _semantic_user_message,
                        salvaged_no_reply,
                        record_experience=False,
                    )
                    pending_exchange_id = None
                await _emit_chat_output_receipt(
                    salvaged_no_reply,
                    cause="chat_response",
                    metadata={
                        "response_confidence": "bounded",
                        "path": "cognitive_engine_served_repairable_draft",
                        "status": "cognitive_engine_served_repairable_draft",
                    },
                )
                return JSONResponse(
                    {
                        "response": salvaged_no_reply,
                        "status": "cognitive_engine_served_repairable_draft",
                        "conversation_lane": lane,
                        "response_confidence": "bounded",
                        "live_turn_contract": salvage_contract,
                    }
                )

            lane = _mark_conversation_lane_state(
                "desktop_cognitive_engine_required_no_reply",
                state="failed",
            )
            _live_turn_trace.update(
                {
                    "response_path": "desktop_cognitive_engine_required_no_reply",
                }
            )
            from core.conversation.reply_provenance import THE_HONEST_FAILURE

            failure_reply = THE_HONEST_FAILURE
            # Before giving up, ask whether the runtime already HOLDS the answer.
            #
            # Live 2026-08-10: asked which of her subsystems were degraded and
            # whether any job had been failing repeatedly, she served the
            # sentence above — while /api/health carried integrity=degraded and
            # overt_action_cycle at failures=13 with its exact TypeError. The
            # answer was structured, live, and hers. Nothing had fetched it.
            #
            # This is a reading, not a rescue: self_health_answer() returns text
            # only when a channel actually produced a value, and the text is
            # built from those values, so it cannot describe a health she does
            # not have. When no channel reads, it returns "" and the honest
            # refusal above stands.
            evidenced_reply = _chat_conversation_repair._self_health_answer_or_empty(
                _semantic_user_message
            )
            if not evidenced_reply:
                # And what this turn's tools actually returned. The same rescue
                # as the other giving-up path, which is why both exist: two
                # places build this reply and a fix applied to one of them
                # leaves the other saying "I couldn't get to an answer" on top
                # of a tool result.
                evidenced_reply = _what_the_tools_found()
            evidenced_reply = await _anything_better_than_giving_up(
                _semantic_user_message,
                reason="the cognitive engine could not serve this turn",
                already=evidenced_reply,
                budget_s=_remaining_foreground_budget(),
            )
            if evidenced_reply:
                failure_reply = evidenced_reply
            # A refusal is the right answer when nothing better is known. When
            # the runtime has ALREADY worked the answer out, it is the worst of
            # the three options available.
            #
            # LIVE, 2026-08-10: "7919 times 6421 — what do you get?" The
            # preflight computed 50847899. The model produced 50864799 five
            # times; the worker's own gate rejected each one as
            # arithmetic_answer_missing — correctly — and then the turn died
            # with nothing served. So the number was known, the wrong number
            # was caught, and the person got neither fact.
            #
            # Deterministic beats absent. This does not ask the model again,
            # because the model has just demonstrated five times that it cannot
            # produce this value; a calculator's answer does not need one.
            # This rescue lived HERE and only here, so the other refusal site
            # and every lane-status path gave the same apology for the same
            # computable question. One helper now, used by all of them.
            _computed = _known_answer_for_this_turn()
            if _computed:
                failure_reply = _computed
                logger.warning(
                    "🔢 Serving the computed arithmetic result (%s) instead of "
                    "a refusal — the value was known the whole turn.",
                    _computed,
                )
            logger.error("%s Surface=%s", failure_reply, request_surface or "unknown")
            if pending_exchange_id:
                await _chat_preflight._complete_logged_exchange(
                    pending_exchange_id,
                    _semantic_user_message,
                    failure_reply,
                    record_experience=False,
                )
                pending_exchange_id = None
            await _emit_chat_output_receipt(
                failure_reply,
                cause="chat_response",
                metadata={
                    "response_confidence": "failed",
                    "path": "desktop_cognitive_engine",
                    "status": "desktop_cognitive_engine_unavailable",
                    "reason": "desktop_cognitive_engine_required_no_reply",
                },
            )
            return JSONResponse(
                {
                    "response": failure_reply,
                    "status": "desktop_cognitive_engine_unavailable",
                    "reason": "desktop_cognitive_engine_required_no_reply",
                    "conversation_lane": lane,
                    "response_confidence": "failed",
                    "live_turn_contract": _live_turn_contract(
                        lane_status=lane,
                        response_confidence="failed",
                        status="desktop_cognitive_engine_unavailable",
                        reply_source="desktop_cognitive_engine_required_no_reply",
                    ),
                },
                # In-band fail-closed delivery for real users.
                status_code=503 if is_benchmark else 200,
            )

        if reply_text and not conversation_only_surface:
            # Surface parity: desktop/file objectives execute through the
            # governed skill path on EVERY user-facing surface, not only
            # when the desktop UI header is present. Observed live: a plain
            # API chat turn asked for a folder+file, no executor ran, and
            # the model narrated completion with a hallucinated timestamp.
            if not _chat_desktop_objective._blocks_consequential_desktop_execution(_semantic_user_message):
                live_proof = await _chat_runtime_proof._execute_live_runtime_proof(_semantic_user_message)
                if live_proof:
                    return await _finalize_fastpath(
                        _chat_desktop_repair._apply_aura_voice_shaping(
                            str(live_proof.get("response") or "")
                        ),
                        status=str(live_proof.get("status") or "live_proof"),
                    )

                explicit_file = await _execute_explicit_local_file_objective(_semantic_user_message)
                if explicit_file:
                    return await _finalize_fastpath(
                        _chat_desktop_repair._apply_aura_voice_shaping(
                            str(explicit_file.get("response") or "")
                        ),
                        status=str(explicit_file.get("status") or "file_operation"),
                    )

                desktop_objective = await _run_desktop_objective_tracked(
                    _semantic_user_message,
                    cognitive_reply=reply_text,
                )
                if desktop_objective:
                    return await _finalize_fastpath(
                        _chat_desktop_repair._apply_aura_voice_shaping(
                            str(desktop_objective.get("response") or "")
                        ),
                        status=str(desktop_objective.get("status") or "desktop_objective"),
                    )

        # Phase 2 Constitutional Closure: Try Sovereign Kernel Interface actively
        from core.kernel.kernel_interface import KernelInterface

        ki = KernelInterface.get_instance()
        kernel_timed_out = False

        _seam_early_response, kernel_timed_out, reply_text = await _await_the_sovereign_kernel_reply(
            _attempt_protected_foreground_reply=_attempt_protected_foreground_reply,
            _cancel_kernel_task_if_pending=_cancel_kernel_task_if_pending,
            _finalize_fastpath=_finalize_fastpath,
            _remaining_foreground_budget=_remaining_foreground_budget,
            chat_origin=chat_origin,
            effective_user_message=effective_user_message,
            is_benchmark=is_benchmark,
            kernel_timed_out=kernel_timed_out,
            ki=ki,
            reply_text=reply_text,
        )
        if _seam_early_response is not _SEAM_FELL_THROUGH:
            return _seam_early_response
        if reply_text and not reply_source:
            reply_source = "kernel_interface"

        if kernel_timed_out and is_benchmark:
            timeout_reply = _conversation_lane_user_message(
                _mark_conversation_lane_timeout(),
                timed_out=True,
            )
            if pending_exchange_id:
                await _chat_preflight._complete_logged_exchange(
                    pending_exchange_id,
                    _semantic_user_message,
                    timeout_reply,
                    record_experience=False,
                )
                pending_exchange_id = None
            return JSONResponse(
                {
                    "response": timeout_reply,
                    "status": "benchmark_kernel_timeout",
                    "conversation_lane": _chat_preflight._collect_conversation_lane_status(),
                },
                status_code=503,
            )

        if kernel_timed_out:
            direct_reply = await _attempt_protected_foreground_reply("kernel_timeout")
            if direct_reply:
                reply_text = direct_reply
                reply_source = "protected_foreground"
                logger.info(
                    "✅ [STABILITY] Protected foreground bypass succeeded after kernel timeout (len=%d)",
                    len(reply_text),
                )
                kernel_timed_out = False

        if kernel_timed_out:
            lane = _mark_conversation_lane_timeout()
            # Tiered response: 503 (recoverable/retry) when cortex was ready,
            # 504 (hard timeout) only when the lane itself was broken.
            was_ready = bool(lane.get("conversation_ready", False)) or str(
                lane.get("state", "")
            ).lower() in {"ready", "warming", "recovering"}
            # Real users get the honest timeout text in-band (a raw 5xx made
            # the desktop show 'no response' over a live mind — the nightcap
            # soak caught two turns leaking through this exact path);
            # benchmarks keep true status codes via X-Aura-Benchmark.
            status_code = (503 if was_ready else 504) if is_benchmark else 200
            timeout_reply = _conversation_lane_user_message(lane, timed_out=True)
            if pending_exchange_id:
                await _chat_preflight._complete_logged_exchange(
                    pending_exchange_id,
                    _semantic_user_message,
                    timeout_reply,
                )
                pending_exchange_id = None
            return JSONResponse(
                {
                    "response": timeout_reply,
                    "status": "timeout",
                    "conversation_lane": lane,
                },
                status_code=status_code,
            )

        # Legacy Orchestrator Fallback. This is no longer automatic for live
        # chat: callers must opt in with X-Aura-Allow-Legacy-Orchestrator.
        # Otherwise a canonical lane failure could be masked by thinner/raw
        # assistant-shaped behavior.
        if (
            not reply_text
            and not is_benchmark
            and not conversation_only_surface
            and _request_allows_legacy_orchestrator_fallback(request)
        ):
            orch = ServiceContainer.get("orchestrator", default=None)
            if orch:
                logger.warning("REST: Awaiting explicit opt-in legacy orchestrator fallback.")
                legacy_timeout = _remaining_foreground_budget()
                reply_text = await asyncio.wait_for(
                    orch.process_user_input_priority(
                        effective_user_message,
                        origin=chat_origin,
                        timeout_sec=legacy_timeout,
                    ),
                    timeout=legacy_timeout,
                )
                if reply_text:
                    reply_source = reply_source or "legacy_orchestrator"
        elif not reply_text and not is_benchmark:
            logger.warning(
                "No canonical chat reply available; refusing implicit legacy orchestrator fallback "
                "for surface=%s.",
                request_surface or "unknown",
            )

        if is_benchmark:
            final_benchmark_text = str(reply_text or "").strip()
            _seam_early_response, pending_exchange_id = await _refuse_an_empty_benchmark_reply(
                _chat_session_id=_chat_session_id,
                _original_user_message=_original_user_message,
                _semantic_user_message=_semantic_user_message,
                final_benchmark_text=final_benchmark_text,
                pending_exchange_id=pending_exchange_id,
            )
            if _seam_early_response is not _SEAM_FELL_THROUGH:
                return _seam_early_response
            contract_reason = _benchmark_reply_contract_unmet(
                _semantic_user_message,
                final_benchmark_text,
            )
            _seam_early_response, pending_exchange_id = await _refuse_an_unmet_benchmark_contract(
                _chat_session_id=_chat_session_id,
                _original_user_message=_original_user_message,
                _semantic_user_message=_semantic_user_message,
                contract_reason=contract_reason,
                final_benchmark_text=final_benchmark_text,
                pending_exchange_id=pending_exchange_id,
            )
            if _seam_early_response is not _SEAM_FELL_THROUGH:
                return _seam_early_response

            # Preserve benchmark formatting while still requiring the canonical
            # KernelInterface/AuraKernel path above. This is raw-output mode,
            # not a direct inference bypass.
            response_data = {
                "response": final_benchmark_text,
                "status": "benchmark_kernel",
                "conversation_lane": _chat_preflight._collect_conversation_lane_status(),
                "response_confidence": "high",
            }
            if pending_exchange_id:
                await _chat_preflight._complete_logged_exchange(
                    pending_exchange_id,
                    _semantic_user_message,
                    final_benchmark_text,
                    record_experience=False,
                )
                pending_exchange_id = None
            else:
                await _chat_preflight._log_exchange(
                    _original_user_message,
                    final_benchmark_text,
                    record_experience=False,
                    session_id=_chat_session_id,
                )
            await _emit_chat_output_receipt(
                final_benchmark_text,
                cause="chat_response",
                metadata={
                    "response_confidence": "high",
                    "path": "kernel_benchmark",
                },
            )
            return JSONResponse(response_data)

        if not str(reply_text or "").strip():
            # A completed cycle that produced nothing is not a judgement.
            #
            # The kernel finished and returned an empty string — no timeout,
            # so none of the waiting-and-retrying above applies. Measured
            # live: three identical questions refused in a row while the
            # runtime warmed its caches, and the fourth answered normally.
            # Nothing was wrong with the question, and nothing was wrong with
            # her. One more attempt through the lane that already exists for
            # this is cheaper than telling somebody their answer does not
            # exist when it exists a second later.
            second_try = await _attempt_protected_foreground_reply("canonical_empty_reply")
            if second_try:
                reply_text = second_try
                reply_source = reply_source or "protected_foreground"
                logger.info(
                    "✅ Empty canonical reply answered on a second attempt (len=%d)",
                    len(reply_text),
                )
            else:
                record_degradation(
                    "chat",
                    RuntimeError("canonical cycle produced nothing, twice"),
                    severity="info",
                    action="told the person why rather than claiming a judgement",
                )

        _seam_early_response, lane, pending_exchange_id = await _refuse_an_empty_canonical_reply(
            _chat_session_id=_chat_session_id,
            _original_user_message=_original_user_message,
            _semantic_user_message=_semantic_user_message,
            is_benchmark=is_benchmark,
            lane=lane,
            pending_exchange_id=pending_exchange_id,
            reply_text=reply_text,
        )
        if _seam_early_response is not _SEAM_FELL_THROUGH:
            return _seam_early_response

        _delivery_stage_started_at = time.monotonic()
        if _cognitive_reply_returned_at > 0.0:
            _delivery_timing["engine_to_stabilizer_ms"] = (
                _delivery_stage_started_at - _cognitive_reply_returned_at
            ) * 1000.0
        _qualified_exact_delivery = _bind_qualified_recurrent_public_answer(
            _live_turn_trace,
            reply_text,
        )
        _qualified_exact_reply = str(reply_text or "") if _qualified_exact_delivery else ""
        if _qualified_exact_delivery:
            # Exact recurrent output is a canonical serialization of
            # authenticated semantic state. Prose stabilization cannot improve
            # it: any byte change invalidates its receipt. Ordinary generated
            # conversation remains on the full stabilization path.
            _live_turn_trace["qualified_recurrent_prose_pipeline_bypassed"] = True
        else:
            reply_text = await _stabilize_user_facing_reply(
                _semantic_user_message,
                reply_text,
                desktop_cognitive_engine_required=desktop_requires_cognitive_engine,
                protected_foreground_lane=desktop_requires_cognitive_engine,
            )
        _delivery_timing["stabilizer_ms"] = (
            time.monotonic() - _delivery_stage_started_at
        ) * 1000.0
        if _grounded_recall_context and not _qualified_exact_delivery:
            from core.conversation.grounded_recall import (
                grounded_quote_from_context,
                repair_grounded_recall_speaker_attribution,
            )

            reply_text, attribution_repaired = repair_grounded_recall_speaker_attribution(
                _semantic_user_message,
                reply_text,
                grounded_quote_from_context(_grounded_recall_context),
            )
            if attribution_repaired:
                logger.info("Grounded recall repaired first-person user-quote attribution.")
        _delivery_stage_started_at = time.monotonic()
        if not _qualified_exact_delivery:
            reply_text = await _reanswer_when_the_runtime_contradicts_her(
                reply_text,
                user_message=_semantic_user_message,
                session_id=_chat_session_id,
                lane=lane,
                source="chat_api",
                require_engine=bool(desktop_requires_cognitive_engine),
                turn_sensory_evidence=_turn_sensory_evidence,
                turn_trace=_live_turn_trace,
            )
        _delivery_timing["runtime_reconcile_ms"] = (
            time.monotonic() - _delivery_stage_started_at
        ) * 1000.0
        if (
            not _qualified_exact_delivery
            and allow_chat_fastpaths
            and _chat_preflight._is_explicit_capability_inventory_request(_semantic_user_message)
            and _chat_desktop_repair._capability_inventory_reply_is_inadequate(
                _semantic_user_message,
                reply_text,
            )
        ):
            logger.warning(
                "🧭 Replacing inadequate capability inventory reply with grounded live catalog summary."
            )
            reply_text = _chat_desktop_repair._build_grounded_capability_inventory_reply(
                _semantic_user_message
            )
        if (
            not _qualified_exact_delivery
            and _chat_preflight._is_explicit_capability_inventory_request(
                _semantic_user_message
            )
        ):
            reply_text = _ensure_capability_inventory_non_execution_boundary(
                _semantic_user_message,
                reply_text,
            )
        repaired_recall = False
        if not _qualified_exact_delivery and not desktop_requires_cognitive_engine:
            repaired_recall_reply, repaired_recall = await _repair_conversation_recall_if_needed(
                _semantic_user_message,
                reply_text,
                session_id=_chat_session_id,
            )
            if repaired_recall:
                logger.warning(
                    "🧠 Replacing inadequate conversation recall reply with canonical chat-log recall."
                )
                reply_text = repaired_recall_reply

        # ── Response confidence assessment ────────────────────────
        _pending_affordance_intents: list[Any] = []
        _affordance_registry = None
        if not _qualified_exact_delivery and "⟦affordance:" in (reply_text or ""):
            try:
                from core.cognition.expressive_affordances import get_affordance_registry

                _affordance_registry = get_affordance_registry()
                _pending_affordance_intents = _affordance_registry.parse_intents(reply_text)
                if _pending_affordance_intents:
                    reply_text = _affordance_registry.strip_intents(reply_text)
                    if len(reply_text.strip()) < 5:
                        reply_text = "Here —"
            except _CHAT_RECOVERABLE_ERRORS as _aff_exc:
                record_degradation("chat", _aff_exc)
                logger.debug("Affordance intent parse skipped: %s", _aff_exc)
                _pending_affordance_intents = []

        _delivery_stage_started_at = time.monotonic()
        response_confidence = "high"
        if _qualified_exact_delivery:
            is_stale = False
            is_same_diff = False
            recent_user_messages = []
            is_off_topic, off_topic_reason = False, ""
            semantic_glitch, semantic_glitch_reason = False, ""
            reply_assessment = None
        else:
            quality = await _measure_reply_quality_candidate(
                _semantic_user_message, reply_text
            )
            is_stale = quality.is_stale
            is_same_diff = quality.is_same_diff
            recent_user_messages = list(quality.recent_user_messages)
            is_off_topic = quality.is_off_topic
            off_topic_reason = quality.off_topic_reason
            semantic_glitch = quality.semantic_glitch
            semantic_glitch_reason = quality.semantic_glitch_reason
            reply_assessment = quality.reply_assessment
        desktop_recall_contract_failed = False
        desktop_context_contract_failed = False
        desktop_memory_state_contract_failed = False
        if desktop_requires_cognitive_engine and not _qualified_exact_delivery:
            expected_recall_reply = await _chat_memory_state._build_conversation_recall_reply(
                _semantic_user_message,
                session_id=_chat_session_id,
            )
            desktop_recall_contract_failed = bool(
                expected_recall_reply
                and _conversation_recall_reply_is_inadequate(
                    _semantic_user_message,
                    reply_text,
                    expected_recall_reply,
                )
            )
            if desktop_recall_contract_failed:
                _live_turn_trace["response_path"] = "cognitive_engine_recall_contract_failed"
            desktop_context_contract_failed = _context_challenge_reply_is_inadequate(
                _semantic_user_message,
                reply_text,
            )
            if desktop_context_contract_failed:
                _live_turn_trace["response_path"] = "cognitive_engine_context_contract_failed"
            desktop_memory_state_contract_failed = _memory_state_evidence_is_missing_from_reply(
                _semantic_user_message,
                reply_text,
                desktop_memory_state_evidence,
            )
            if desktop_memory_state_contract_failed:
                _live_turn_trace["response_path"] = "cognitive_engine_memory_state_contract_failed"
        # DIAGNOSTIC DEFECT, measured live 2026-08-04. This condition has EIGHT
        # disjuncts and the warning below printed SEVEN. The eighth —
        # `_reply_assessment_requires_repair_with_memory_evidence` — was the one
        # that fired on a real turn, and because `assessment=` renders the
        # assessment's REASONS (empty, since the repair requirement is a
        # judgement about evidence rather than a reason on the reply), the log
        # read:
        #
        #   Response confidence: degraded (stale=False, same_answer_diff_prompt=False,
        #   off_topic=False, semantic_glitch=False, recall_contract=False,
        #   context_contract=False, memory_state_contract=False, assessment=,
        #   streak=1, reason=)
        #
        # Every flag False, no assessment, no reason — a turn degraded on a
        # condition the runtime does not report. A gate that cannot say what it
        # did is the reason this took a restart and a log dive to find.
        assessment_requires_repair = bool(
            not _qualified_exact_delivery
            and _reply_assessment_requires_repair_with_memory_evidence(
                reply_assessment,
                _semantic_user_message,
                reply_text,
                memory_state_evidence=desktop_memory_state_evidence,
            )
        )
        if (
            is_stale
            or is_same_diff
            or is_off_topic
            or semantic_glitch
            or desktop_recall_contract_failed
            or desktop_context_contract_failed
            or desktop_memory_state_contract_failed
            or assessment_requires_repair
        ):
            response_confidence = "degraded"
            degradation_streak = _increment_conversation_degradation_streak()
            logger.warning(
                "⚠️ Response confidence: degraded (stale=%s, same_answer_diff_prompt=%s, off_topic=%s, semantic_glitch=%s, recall_contract=%s, context_contract=%s, memory_state_contract=%s, assessment_requires_repair=%s, assessment=%s, assessment_ok=%s, retryable=%s, hard_failure=%s, reply_len=%d, streak=%d, reason=%s)",
                is_stale,
                is_same_diff,
                is_off_topic,
                semantic_glitch,
                desktop_recall_contract_failed,
                desktop_context_contract_failed,
                desktop_memory_state_contract_failed,
                assessment_requires_repair,
                ",".join(getattr(reply_assessment, "reasons", ()) or ()),
                getattr(reply_assessment, "ok", None),
                getattr(reply_assessment, "retryable", None),
                getattr(reply_assessment, "hard_failure", None),
                len(str(reply_text or "")),
                degradation_streak,
                off_topic_reason or semantic_glitch_reason or "",
            )
        else:
            _set_conversation_degradation_streak(0)

        hard_final_quality_failed = bool(
            not _qualified_exact_delivery
            and (
                is_off_topic
                or semantic_glitch
                or desktop_recall_contract_failed
                or desktop_context_contract_failed
                or desktop_memory_state_contract_failed
                or _reply_assessment_requires_repair_with_memory_evidence(
                    reply_assessment,
                    _semantic_user_message,
                    reply_text,
                    memory_state_evidence=desktop_memory_state_evidence,
                )
            )
        )
        _delivery_timing["quality_classification_ms"] = (
            time.monotonic() - _delivery_stage_started_at
        ) * 1000.0

        if response_confidence == "degraded":
            (
                repaired_reply,
                is_stale,
                is_same_diff,
                is_off_topic,
                off_topic_reason,
                repaired,
            ) = await _repair_final_degraded_reply_with_provenance(
                _live_turn_trace,
                stage="chat.final_quality_gate",
                user_message=_semantic_user_message,
                reply_text=reply_text,
                stale=is_stale,
                same_diff=is_same_diff,
                off_topic=is_off_topic,
                off_topic_reason=off_topic_reason,
                desktop_cognitive_engine_required=desktop_requires_cognitive_engine,
                protected_foreground_lane=desktop_requires_cognitive_engine,
                session_id=_chat_session_id,
            )
            if repaired and repaired_reply != reply_text:
                reply_text = repaired_reply
                semantic_glitch, semantic_glitch_reason = _looks_semantically_glitched(
                    _semantic_user_message,
                    reply_text,
                )
                try:
                    # Imported here rather than relied on. The name is bound in
                    # three conditional branches above, which makes it local to
                    # this whole function, so a turn that took none of those
                    # branches read an unbound local and died —
                    # chat.uncaught_turn_error, status=error, and the person
                    # got nothing.
                    from core.conversation.response_reliability import (
                        assess_user_facing_reply as _assess_repaired,
                    )

                    repaired_assessment = _assess_repaired(
                        _semantic_user_message,
                        reply_text,
                        recent_user_messages=recent_user_messages,
                    )
                except _CHAT_RECOVERABLE_ERRORS:
                    repaired_assessment = None
                repaired_recall_contract_failed = False
                repaired_context_contract_failed = False
                repaired_memory_state_contract_failed = False
                repaired_assessment_retryable = (
                    _reply_assessment_requires_repair_with_memory_evidence(
                        repaired_assessment,
                        _semantic_user_message,
                        reply_text,
                        memory_state_evidence=desktop_memory_state_evidence,
                    )
                )
                if desktop_requires_cognitive_engine:
                    expected_recall_reply = (
                        await _chat_memory_state._build_conversation_recall_reply(
                            _semantic_user_message,
                            session_id=_chat_session_id,
                        )
                    )
                    repaired_recall_contract_failed = bool(
                        expected_recall_reply
                        and _conversation_recall_reply_is_inadequate(
                            _semantic_user_message,
                            reply_text,
                            expected_recall_reply,
                        )
                    )
                    repaired_context_contract_failed = _context_challenge_reply_is_inadequate(
                        _semantic_user_message,
                        reply_text,
                    )
                    repaired_memory_state_contract_failed = (
                        _memory_state_evidence_is_missing_from_reply(
                            _semantic_user_message,
                            reply_text,
                            desktop_memory_state_evidence,
                        )
                    )
                hard_final_quality_failed = bool(
                    is_off_topic
                    or semantic_glitch
                    or repaired_recall_contract_failed
                    or repaired_context_contract_failed
                    or repaired_memory_state_contract_failed
                    or repaired_assessment_retryable
                )
                if not (
                    is_stale
                    or is_same_diff
                    or is_off_topic
                    or semantic_glitch
                    or repaired_recall_contract_failed
                    or repaired_context_contract_failed
                    or repaired_memory_state_contract_failed
                    or _reply_assessment_requires_repair_with_memory_evidence(
                        repaired_assessment,
                        _semantic_user_message,
                        reply_text,
                        memory_state_evidence=desktop_memory_state_evidence,
                    )
                ):
                    response_confidence = "high"
                    _set_conversation_degradation_streak(0)
                    logger.info("✅ Final reply quality gate repaired degraded output.")
                else:
                    response_confidence = "degraded"

        hard_final_quality_failed, reply_source, reply_text, response_confidence = _repair_a_degraded_affect_reply(
            _live_turn_trace=_live_turn_trace,
            _semantic_user_message=_semantic_user_message,
            desktop_requires_cognitive_engine=desktop_requires_cognitive_engine,
            hard_final_quality_failed=hard_final_quality_failed,
            reply_source=reply_source,
            reply_text=reply_text,
            response_confidence=response_confidence,
        )

        hard_final_quality_failed, reply_source, reply_text, response_confidence = _repair_a_degraded_identity_reply(
            _live_turn_trace=_live_turn_trace,
            _semantic_user_message=_semantic_user_message,
            desktop_requires_cognitive_engine=desktop_requires_cognitive_engine,
            hard_final_quality_failed=hard_final_quality_failed,
            reply_source=reply_source,
            reply_text=reply_text,
            response_confidence=response_confidence,
        )

        hard_final_quality_failed, reply_text, response_confidence = await _recheck_a_degraded_reply(
            _live_turn_trace=_live_turn_trace,
            _semantic_user_message=_semantic_user_message,
            desktop_memory_state_evidence=desktop_memory_state_evidence,
            desktop_requires_cognitive_engine=desktop_requires_cognitive_engine,
            hard_final_quality_failed=hard_final_quality_failed,
            lane=lane,
            reply_text=reply_text,
            response_confidence=response_confidence,
        )

        if (
            desktop_requires_cognitive_engine
            and response_confidence == "degraded"
            and hard_final_quality_failed
        ):
            return await _fail_closed_degraded_desktop_reply(
                reply_text,
                response_path="desktop_required_final_quality_failed",
            )

        # Proactive recovery: if 3+ consecutive degraded responses, compact + reset stale deque
        degradation_streak = _conversation_degradation_streak()
        if degradation_streak >= 3:
            logger.warning(
                "🚨 Degradation streak=%d — triggering proactive compaction + stale reset.",
                degradation_streak,
            )
            with _conversation_quality_lock:
                quality_state = _conversation_quality_state_locked()
                quality_state.recent_responses.clear()
                quality_state.recent_response_pairs.clear()
                quality_state.consecutive_degraded_count = 0
            try:
                live_state = _chat_preflight._resolve_live_aura_state()
                if live_state and hasattr(live_state, "compact"):
                    live_state.compact(trigger_threshold=20, keep_turns=15)
                    logger.info("🗜️ Proactive compaction completed after degradation streak.")
            except _CHAT_RECOVERABLE_ERRORS as _streak_exc:
                record_degradation("chat", _streak_exc)
                logger.debug("Degradation streak compaction failed: %s", _streak_exc)

        # Proactive context compaction — fire-and-forget to prevent working memory bloat
        try:
            live_state = _chat_preflight._resolve_live_aura_state()
            if live_state and hasattr(live_state, "compact"):
                wm = getattr(getattr(live_state, "cognition", None), "working_memory", None)
                if wm and isinstance(wm, list) and len(wm) > 30:
                    compacted = live_state.compact(trigger_threshold=30, keep_turns=20)
                    if compacted:
                        logger.debug(
                            "Proactive AuraState.compact() completed (working_memory was %d).",
                            len(wm),
                        )
        except _CHAT_RECOVERABLE_ERRORS as _compact_exc:
            record_degradation("chat", _compact_exc)
            logger.debug("Proactive compaction skipped: %s", _compact_exc)

        # ── Post-Response Infrastructure checks ─────────────────
        # 1. Check self-consistency (avoiding false inability claims, commitment contradictions)
        # The decision itself lives in chat_quality.assess_post_response_confidence
        # so it can be tested without driving a whole HTTP turn — this function
        # is 4,488 lines and 633 branches, and everything inside it could only
        # be exercised end to end.
        is_consistent, inconsistency_reason = (
            (True, "")
            if _qualified_exact_delivery
            else (
                _check_response_consistency(reply_text, _semantic_user_message)
                if response_confidence == "high"
                else (True, "")
            )
        )
        _delivery_stage_started_at = time.monotonic()
        lane_status = _chat_preflight._collect_conversation_lane_status()
        _delivery_timing["lane_status_ms"] = (
            time.monotonic() - _delivery_stage_started_at
        ) * 1000.0
        try:
            actual_generation_at = float(lane_status.get("last_user_generation_at") or 0.0)
        except (TypeError, ValueError):
            actual_generation_at = 0.0
        _downgrade = assess_post_response_confidence(
            response_confidence,
            self_consistent=is_consistent,
            inconsistency_reason=inconsistency_reason,
            used_fallback_lane=bool(lane_status.get("last_user_generation_used_fallback", False)),
            generation_happened_this_turn=(
                actual_generation_at >= max(0.0, request_wall_started_at - 1.0)
            ),
            actual_endpoint=str(lane_status.get("last_user_generation_endpoint") or "").strip(),
            desired_endpoint=str(lane_status.get("desired_endpoint") or "").strip(),
        )
        if _downgrade.downgraded:
            response_confidence = _downgrade.confidence
            if _downgrade.lane_warning:
                lane_status["response_lane_warning"] = _downgrade.lane_warning
            logger.warning(
                "⚠️ Response confidence lowered to %r: %s%s",
                _downgrade.confidence,
                _downgrade.reason,
                f" ({_downgrade.lane_warning})" if _downgrade.lane_warning else "",
            )

        # 2. Extract new open loops (commitments/promises) made in this turn
        if not _qualified_exact_delivery:
            _extract_and_register_commitments(reply_text, _semantic_user_message)

        # 3. Log comprehensive quality metrics
        _log_response_quality_metrics(
            user_message=_semantic_user_message,
            reply_text=reply_text,
            confidence=response_confidence,
            stale=is_stale,
            same_diff=is_same_diff,
            off_topic=is_off_topic,
        )
        logger.info(
            "Foreground delivery timing before terminal shaping: %s",
            {key: round(value, 2) for key, value in _delivery_timing.items()},
        )
        _terminal_shaping_started_at = time.monotonic()

        # Prepend any late-answered messages from prior turns so the user
        # sees what came back. The cortex was also given the continuity
        # context in body.message above, so the reply already acknowledges
        # the thread.
        _pre_context_strip_reply = reply_text
        _final_reply = (
            _qualified_exact_reply
            if _qualified_exact_delivery
            # The reply that is actually served. An ellipsis here is a turn
            # that looks answered to everything downstream and says nothing to
            # the person.
            else _never_an_ellipsis(_strip_user_visible_context_leaks(reply_text))
        )
        # The recorded answer is applied HERE, after every repair, regeneration
        # and shaping pass, because everywhere earlier it was discarded.
        #
        # LIVE, 2026-08-10, three attempts. Applied inside
        # _stabilize_user_facing_reply it worked in-process and never reached
        # the person: on one turn the full record was stripped as off-topic,
        # and on the next the whole reply was replaced by a later repair that
        # said "I didn't actually count the .py files" — which is also false,
        # and which no amount of correcting an earlier draft can fix.
        #
        # A correction that a later stage can overwrite is not a correction.
        if not _qualified_exact_delivery:
            _final_reply = str(
                _append_past_action_record(_semantic_user_message, _final_reply) or _final_reply
            )
            _final_reply = str(
                _append_runtime_authored_why(_semantic_user_message, _final_reply) or _final_reply
            )
        _append_turn_text_mutation(
            _live_turn_trace,
            stage="chat.final_context_leak_strip",
            method="deterministic_context_leak_removal",
            reasons=["user_visible_context_boundary"],
            before=_pre_context_strip_reply,
            after=_final_reply,
            deterministic=True,
            authorship_effect="preserved",
        )
        # The sums this answer does on its own numbers, recomputed.
        #
        # Appended, never substituted. Bryan, 2026-09-08: "shouldnt reject the
        # whole response ever. just wondering if math is ever checked
        # anywhere." It was checked in one place only — the arithmetic a
        # PERSON asks for, which `arithmetic_check` recomputes and serves —
        # and never for the arithmetic she performs inside a worked answer.
        # Live that afternoon, one run of the daylight question computed
        # 926 - 720 and announced 105.
        _pre_arithmetic_note_reply = _final_reply
        try:
            from core.conversation.the_arithmetic_in_an_answer import (
                a_note_about_the_arithmetic,
            )

            _arithmetic_note = a_note_about_the_arithmetic(_final_reply)
        except (ImportError, TypeError, ValueError) as _exc:
            logger.debug("Answer-arithmetic check unavailable: %s", _exc)
            _arithmetic_note = ""
        if _arithmetic_note:
            _final_reply = f"{str(_final_reply).rstrip()}\n\n{_arithmetic_note}"
            logger.warning("🔢 %s", _arithmetic_note)
            _append_turn_text_mutation(
                _live_turn_trace,
                stage="chat.the_arithmetic_in_the_answer",
                method="append_a_note_naming_the_step",
                reasons=["a_stated_sum_does_not_hold"],
                before=_pre_arithmetic_note_reply,
                after=_final_reply,
                deterministic=True,
                authorship_effect="preserved",
            )

        # A reply that withdrew its own opening and delivered it anyway.
        #
        # Here rather than in the phase, for the reason written above this
        # block: the continuation that carries the correction is joined after
        # the phase is done, so the contradiction does not exist yet where the
        # phase could see it.
        _pre_self_correction_reply = _final_reply
        try:
            from core.conversation.a_reply_that_corrects_itself import (
                the_reply_corrects_its_own_headline,
            )

            _withdrawn = the_reply_corrects_its_own_headline(_final_reply)
        except (ImportError, TypeError, ValueError) as _exc:
            logger.debug("Self-correction check unavailable: %s", _exc)
            _withdrawn = None
        if _withdrawn is not None:
            _final_reply = _withdrawn.text
            logger.info(
                "The reply withdrew %s and gave %s; the opening it withdrew is "
                "marked as withdrawn rather than served as the answer.",
                _withdrawn.superseded,
                _withdrawn.corrected,
            )
            _append_turn_text_mutation(
                _live_turn_trace,
                stage="chat.the_reply_corrected_itself",
                method="strike_the_withdrawn_opening",
                reasons=[_withdrawn.reason],
                before=_pre_self_correction_reply,
                after=_final_reply,
                deterministic=True,
                authorship_effect="preserved",
            )
        _final_status = reply_source or "ok"
        if not _qualified_exact_delivery:
            _pre_objective_chokepoint_reply = _final_reply
            _final_reply, _final_status = await _apply_desktop_objective_chokepoint(
                _final_reply, _final_status
            )
            _append_turn_text_mutation(
                _live_turn_trace,
                stage="chat.desktop_objective_chokepoint",
                method="desktop_objective_result_replacement",
                reasons=[str(_final_status or "desktop_objective")],
                before=_pre_objective_chokepoint_reply,
                after=_final_reply,
                deterministic=False,
                authorship_effect="replaced_by_runtime",
            )

        _affordance_results: list[dict[str, Any]] = []
        if _pending_affordance_intents and _affordance_registry is not None:
            _affordance_ctx = {
                "last_user_message": _semantic_user_message,
                "session_id": _chat_session_id,
            }
            for _intent in _pending_affordance_intents[:3]:
                try:
                    _aff_result = await _affordance_registry.realize(_intent, _affordance_ctx)
                except _CHAT_RECOVERABLE_ERRORS as _aff_realize_exc:
                    record_degradation("chat", _aff_realize_exc)
                    logger.debug("Affordance realize skipped: %s", _aff_realize_exc)
                    continue
                _affordance_results.append(_aff_result)
            _spoken = [
                str(r.get("spoken") or "").strip() for r in _affordance_results if r.get("spoken")
            ]
            if _spoken:
                _pre_affordance_reply = _final_reply
                _final_reply = (_final_reply + "\n\n" + "\n".join(_spoken)).strip()
                _append_turn_text_mutation(
                    _live_turn_trace,
                    stage="chat.affordance_spoken_append",
                    method="effect_receipt_spoken_append",
                    reasons=["realized_affordance"],
                    before=_pre_affordance_reply,
                    after=_final_reply,
                    deterministic=False,
                    authorship_effect="augmented_by_runtime",
                )
        if _qualified_exact_delivery:
            _bind_qualified_recurrent_terminal_contract(
                _live_turn_trace,
                _final_reply,
            )
        else:
            _final_reply = _enforce_or_bind_terminal_output_contract(
                _live_turn_trace,
                user_message=_semantic_user_message,
                reply_text=_final_reply,
            )

        # Claims this process can measure are settled by the measurement, not
        # by whether the model read the grounding it was handed. The prompt
        # block is a prior and priors lose to fluent sentences; the clock is
        # causal. Runs last, on the exact text about to be spoken.
        try:
            from core.conversation.grounded_claim_guard import verify_grounded_claims

            _grounded = (
                verify_grounded_claims(_final_reply)
                if not _qualified_exact_delivery
                else None
            )
            if _grounded is not None and _grounded.changed:
                _append_turn_text_mutation(
                    _live_turn_trace,
                    stage="chat.grounded_claim_guard",
                    method="measured_reading_overrides_stated_claim",
                    reasons=list(_grounded.corrections),
                    before=_final_reply,
                    after=_grounded.text,
                    deterministic=True,
                    authorship_effect="augmented_by_runtime",
                )
                logger.warning(
                    "🧭 [GROUNDING] reconciled a spoken claim against a real reading: %s",
                    "; ".join(_grounded.corrections)[:240],
                )
                _final_reply = _grounded.text or _final_reply
        except _CHAT_RECOVERABLE_ERRORS as _exc:
            record_degradation("chat.grounded_claim_guard", _exc)

        if not _qualified_exact_delivery:
            _final_reply = await _ground_executable_output_claims_for_delivery(
                _live_turn_trace,
                _final_reply,
            )

        # Custody, last. Every stage above rewrote this text; each was checked
        # against what the turn established, and this is where a fact the
        # pipeline lost is put back.
        #
        # It runs after the grounding guard on purpose. Grounding answers "is
        # this sentence true of the world" by measuring again. Custody answers
        # a different question — "did what we already measured survive the
        # journey" — and the 2026-08-10 failure was entirely the second one:
        # nothing about the count was unmeasured, and five stages still ended
        # up delivering a number nobody read.
        try:
            from core.runtime.fact_custody import restore_held_facts

            _custody = (
                restore_held_facts(_final_reply)
                if not _qualified_exact_delivery
                else None
            )
            if _custody is not None and _custody.changed:
                _append_turn_text_mutation(
                    _live_turn_trace,
                    stage="chat.fact_custody",
                    method="held_fact_restored_at_terminal_boundary",
                    reasons=list(_custody.reasons()),
                    before=_final_reply,
                    after=_custody.text,
                    deterministic=True,
                    authorship_effect="augmented_by_runtime",
                )
                _final_reply = _custody.text or _final_reply
        except _CHAT_RECOVERABLE_ERRORS as _exc:
            record_degradation("chat.fact_custody", _exc)

        # Grounding and fact custody can change the exact bytes that will be
        # delivered. Their inputs may have satisfied a word/list/sentence
        # contract while their outputs do not, so terminal proof is always
        # recomputed over the post-mutation answer rather than inherited from
        # an earlier candidate.
        if _qualified_exact_delivery:
            _final_reply = _qualified_exact_reply
            _bind_qualified_recurrent_terminal_contract(
                _live_turn_trace,
                _final_reply,
            )
        else:
            _final_reply = _enforce_or_bind_terminal_output_contract(
                _live_turn_trace,
                user_message=_semantic_user_message,
                reply_text=_final_reply,
            )
        _bind_public_latent_output_quality(
            _live_turn_trace,
            user_message=_semantic_user_message,
            reply_text=_final_reply,
        )
        _bind_qualified_recurrent_public_answer(_live_turn_trace, _final_reply)

        final_live_turn_contract = _live_turn_contract(
            lane_status=lane_status,
            response_confidence=response_confidence,
            status=_final_status,
            reply_source=reply_source,
        )
        _seam_early_response = _fail_closed_on_an_unproven_output_contract(
            _final_reply=_final_reply,
            _live_turn_contract=_live_turn_contract,
            _live_turn_trace=_live_turn_trace,
            final_live_turn_contract=final_live_turn_contract,
            lane_status=lane_status,
        )
        if _seam_early_response is not _SEAM_FELL_THROUGH:
            return _seam_early_response
        _full_mind_unproven = desktop_requires_cognitive_engine and not bool(
            final_live_turn_contract.get("full_mind_path")
        )
        if _full_mind_unproven and _authored_answer_can_serve(final_live_turn_contract):
            # Same treatment the regenerate, recovery and candidate gates already
            # give an authentic reply: certification state remains visible in
            # the receipt, but it cannot replace a valid authored answer.
            logger.warning(
                "Desktop reply served with DEGRADED full-mind proof (her own text; missing: %s).",
                ",".join(final_live_turn_contract.get("full_mind_missing_proofs") or ())
                or "unknown",
            )
            _full_mind_unproven = False
        if _full_mind_unproven and _bounded_runtime_grounding_can_serve(final_live_turn_contract):
            # This is measured/canonical runtime evidence, not the model's
            # wording. Preserve the useful answer and downgrade its authorship
            # claim instead of replacing one honest bounded answer with a
            # generic full-mind failure apology.
            grounded_path = str(
                final_live_turn_contract.get("response_path") or "runtime_grounding"
            )
            logger.warning(
                "Desktop reply served as bounded runtime grounding; it is not "
                "certified as model-authored/full-mind (path=%s).",
                grounded_path,
            )
            _live_turn_trace.update(
                {
                    "cognitive_engine_reply_accepted": False,
                    "cognitive_engine_reply_failed": True,
                    "bounded_contract_used": True,
                    "response_path": grounded_path,
                }
            )
            response_confidence = "bounded"
            _final_status = grounded_path
            reply_source = grounded_path
            final_live_turn_contract = _live_turn_contract(
                lane_status=lane_status,
                response_confidence=response_confidence,
                status=_final_status,
                reply_source=reply_source,
            )
            _full_mind_unproven = False
        _seam_early_response, final_live_turn_contract = _fail_closed_on_an_unproven_full_mind_contract(
            _final_reply=_final_reply,
            _full_mind_unproven=_full_mind_unproven,
            _live_turn_contract=_live_turn_contract,
            _live_turn_trace=_live_turn_trace,
            final_live_turn_contract=final_live_turn_contract,
            is_benchmark=is_benchmark,
            lane_status=lane_status,
        )
        if _seam_early_response is not _SEAM_FELL_THROUGH:
            return _seam_early_response

        response_data = {
            "response": _final_reply,
            "status": _final_status,
            "conversation_lane": lane_status,
            "response_confidence": response_confidence,
            "live_turn_contract": final_live_turn_contract,
        }
        # Same receipts contract as the fastpath door: desktop objectives
        # carry their step receipts on the wire from EVERY reply exit.
        if _desktop_exec_state.get("result") is not None and str(_final_status).startswith(
            "desktop_objective"
        ):
            response_data["data"] = {
                "desktop_result": _json_safe_payload(_desktop_exec_state["result"])
            }
        if _affordance_results:
            response_data.setdefault("data", {})["affordances"] = [
                _json_safe_payload(r) for r in _affordance_results
            ]

        # The durable delivery boundary may still replace these bytes through
        # recorded-answer composition, paired projection, or control-syntax
        # sanitation. Commit exact worker state only after that boundary has
        # sealed the payload it will actually return.
        _resume_trace = dict(_live_turn_trace)
        _resume_session = str(_CHAT_REQUEST_SESSION.get() or _chat_session_id or "")
        _resume_principal = str(_CHAT_REQUEST_PRINCIPAL.get() or "")
        _resume_surface = str(_CHAT_REQUEST_SURFACE.get() or "")
        _chat_delivery.register_chat_delivery_commit_hook(
            request,
            lambda payload: _store_conversation_resume_handle(
                _resume_trace,
                str(payload.get("response") or ""),
                session_id=_resume_session,
                principal_id=_resume_principal,
                principal_surface=_resume_surface,
            ),
        )
        _live_turn_trace["conversation_resume_commit_registered"] = True
        _record_recent_response(_final_reply or "…", _semantic_user_message)
        _persistence_started_at = time.monotonic()
        if pending_exchange_id:
            await _chat_preflight._complete_logged_exchange(
                pending_exchange_id,
                _semantic_user_message,
                _final_reply or "…",
                exchange_metadata=_current_exchange_metadata(
                    _final_reply or "…",
                    response_path=_final_status,
                ),
            )
            pending_exchange_id = None
        else:
            await _chat_preflight._log_exchange(
                _original_user_message,
                _final_reply or "…",
                session_id=_chat_session_id,
                exchange_metadata=_current_exchange_metadata(
                    _final_reply or "…",
                    response_path=_final_status,
                ),
            )
        _delivery_timing["persistence_ms"] = (
            time.monotonic() - _persistence_started_at
        ) * 1000.0

        _receipt_started_at = time.monotonic()
        await _emit_chat_output_receipt(
            _final_reply or "…",
            cause="chat_response",
            metadata={
                "response_confidence": response_confidence,
                "path": _final_status or reply_source or "stabilized",
            },
        )
        _delivery_timing["receipt_ms"] = (time.monotonic() - _receipt_started_at) * 1000.0
        _delivery_timing["terminal_shaping_ms"] = (
            _persistence_started_at - _terminal_shaping_started_at
        ) * 1000.0
        _delivery_timing["request_total_ms"] = (
            time.monotonic() - request_started_at
        ) * 1000.0
        logger.info(
            "Foreground delivery timing complete: %s",
            {key: round(value, 2) for key, value in _delivery_timing.items()},
        )
        # Measured, so the next turn's answer budget can leave room for it
        # instead of sizing an answer that fills the clock exactly and then
        # having nowhere to put it.
        try:
            from core.brain.llm.thinking_reserve import record_delivery_cost

            record_delivery_cost(
                sum(
                    float(_delivery_timing.get(stage, 0.0) or 0.0)
                    for stage in (
                        "engine_to_stabilizer_ms",
                        "stabilizer_ms",
                        "runtime_reconcile_ms",
                        "quality_classification_ms",
                        "lane_status_ms",
                        "terminal_shaping_ms",
                        "persistence_ms",
                        "receipt_ms",
                    )
                )
                / 1000.0
            )
        except (ImportError, AttributeError, TypeError, ValueError) as _cost_exc:
            logger.debug("Delivery cost not recorded: %s", _cost_exc)

        return JSONResponse(response_data)
    except TimeoutError:
        await _cancel_kernel_task_if_pending("outer_timeout")
        lane = _mark_conversation_lane_timeout()
        if desktop_requires_cognitive_engine:
            lane = _mark_conversation_lane_state(
                "desktop_cognitive_engine_timeout",
                state="failed",
            )
            _live_turn_trace.update(
                {
                    "response_path": "desktop_cognitive_engine_timeout",
                    "bounded_contract_used": False,
                }
            )
            timeout_reply = (
                "That one took me longer than I'm willing to keep you waiting for, "
                "and I won't send a rushed answer instead. Ask again and I should "
                "be quicker."
            )
            if pending_exchange_id:
                await _chat_preflight._complete_logged_exchange(
                    pending_exchange_id,
                    _semantic_user_message,
                    timeout_reply,
                    record_experience=False,
                )
                pending_exchange_id = None
            await _emit_chat_output_receipt(
                timeout_reply,
                cause="chat_timeout",
                metadata={
                    "response_confidence": "failed",
                    "path": "desktop_cognitive_engine",
                    "status": "desktop_cognitive_engine_timeout",
                    "reason": "desktop_cognitive_engine_timeout",
                },
            )
            return JSONResponse(
                {
                    "response": timeout_reply,
                    "status": "desktop_cognitive_engine_unavailable",
                    "reason": "desktop_cognitive_engine_timeout",
                    "conversation_lane": lane,
                    "response_confidence": "failed",
                    "live_turn_contract": _live_turn_contract(
                        lane_status=lane,
                        response_confidence="failed",
                        status="desktop_cognitive_engine_unavailable",
                        reply_source="desktop_cognitive_engine_timeout",
                    ),
                },
                # In-band fail-closed delivery for real users.
                status_code=503 if is_benchmark else 200,
            )
        if is_benchmark:
            timeout_reply = _conversation_lane_user_message(lane, timed_out=True)
            if pending_exchange_id:
                await _chat_preflight._complete_logged_exchange(
                    pending_exchange_id,
                    body.message,
                    timeout_reply,
                    record_experience=False,
                )
                pending_exchange_id = None
            await _emit_chat_output_receipt(
                timeout_reply,
                cause="chat_timeout",
                origin="benchmark",
                metadata={
                    "response_confidence": "failed",
                    "path": "benchmark_timeout",
                    "status": "timeout",
                },
            )
            return JSONResponse(
                {
                    "response": timeout_reply,
                    "status": "benchmark_timeout",
                    "conversation_lane": lane,
                    "response_confidence": "failed",
                },
                status_code=503,
            )

        # The kernel timed out, but the resident worker may still be responsive.
        # Recovery stays inside the same protected foreground transaction used
        # everywhere else; an independent direct generation here used to skip
        # authorship, semantic-completion, and requested-output proof entirely.
        try:
            emergency_reply = await _attempt_protected_foreground_reply(
                "outer_timeout_emergency",
                budget_override_s=15.0,
            )
            if emergency_reply:
                logger.info("Protected foreground recovery after outer timeout succeeded.")
                return await _finalize_fastpath(
                    emergency_reply,
                    status="protected_foreground",
                )
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat", exc)
            logger.warning(
                "Emergency degraded response path failed; falling through to timeout response: %s",
                exc,
            )

        # A late model answer is a distinct authored transaction.  The old
        # background retry stored text without generation provenance, then
        # prepended it to an unrelated future turn under that turn's receipt.
        # Do not create ownerless speech. A timeout remains attached to this
        # exact exchange; a later user turn starts cleanly.
        timeout_reply = _conversation_lane_user_message(lane, timed_out=True)

        if pending_exchange_id:
            await _chat_preflight._complete_logged_exchange(
                pending_exchange_id,
                body.message,
                timeout_reply,
            )
            pending_exchange_id = None
        return JSONResponse(
            {
                "response": timeout_reply,
                "status": "timeout",
                "conversation_lane": lane,
                "response_confidence": "degraded",
            },
            status_code=200,  # [STABILITY v53] Changed from 503/504 to 200
        )
    except asyncio.CancelledError as cancel_exc:
        await _cancel_kernel_task_if_pending("request_cancelled")
        if _FOREGROUND_CHAT_PREEMPT_CANCEL_REASON in {str(value) for value in cancel_exc.args}:
            await _mark_logged_exchange_preempted(
                pending_exchange_id,
                reason=_FOREGROUND_CHAT_PREEMPT_CANCEL_REASON,
            )
            pending_exchange_id = None
            raise
        lane = _mark_conversation_lane_state("foreground_cancelled", state="recovering")
        # Don't ask the user to re-send. If we got cancelled while a newer
        # message was already inbound, the user has already moved on; if
        # the client just disconnected, the reply is never seen anyway.
        user_cancelled = _chat_delivery.USER_CANCEL_REASON in {str(value) for value in cancel_exc.args}
        cancel_reply = (
            "Stopped this turn. Actions already completed were not undone."
            if user_cancelled else
            "I'm here. My response was cut short — I'll pick up with whatever you say next."
        )
        if pending_exchange_id:
            await _chat_preflight._complete_logged_exchange(
                pending_exchange_id,
                body.message,
                cancel_reply,
                record_experience=not is_benchmark,
            )
            pending_exchange_id = None
        if is_benchmark:
            return JSONResponse(
                {
                    "response": cancel_reply,
                    "status": "benchmark_cancelled",
                    "conversation_lane": lane,
                    "response_confidence": "failed",
                },
                status_code=503,
            )
        return JSONResponse(
            {
                "response": cancel_reply,
                "status": "cancelled",
                "conversation_lane": lane,
                "response_confidence": "degraded",
            },
            status_code=200,  # [STABILITY v53] Changed from 503 to 200
        )
    except _CHAT_RECOVERABLE_ERRORS as e:
        await _cancel_kernel_task_if_pending("chat_error")
        record_degradation("chat", e)
        logger.error("Chat error: %s", e, exc_info=True)
        error_reply = _grounded_chat_failure_reply()
        status_code = 200
        if pending_exchange_id:
            await _chat_preflight._complete_logged_exchange(
                pending_exchange_id,
                body.message,
                error_reply,
                record_experience=not is_benchmark,
            )
            pending_exchange_id = None
        if is_benchmark:
            await _emit_chat_output_receipt(
                error_reply,
                cause="chat_error",
                origin="benchmark",
                metadata={
                    "response_confidence": "failed",
                    "path": "benchmark_error",
                    "status": type(e).__name__,
                },
            )
            return JSONResponse(
                {
                    "response": error_reply,
                    "status": "benchmark_error",
                    "error_type": type(e).__name__,
                    "conversation_lane": _chat_preflight._collect_conversation_lane_status(),
                    "response_confidence": "failed",
                },
                status_code=503,
            )
        # [STABILITY v53] ALWAYS return 200 with a response. Chat must never
        # appear broken to the user. The "status" field conveys error state.
        return JSONResponse(
            {
                "response": error_reply,
                "status": "error",
                "response_confidence": "degraded",
            },
            status_code=status_code,
        )
    except Exception as e:  # noqa: BLE001 — last-resort turn-death floor
        if is_shutdown_requested():
            return _runtime_shutdown_response(
                "exception_after_shutdown",
                slot_acquired=foreground_slot_acquired,
                error=e,
            )
        # A turn must NEVER surface as HTTP 500. Exceptions outside
        # _CHAT_RECOVERABLE_ERRORS still reached the global handler and killed
        # the turn with a 500 — observed live during the soak: the local 32B
        # timed out and a retired remote-provider path also failed
        # RESOURCE_EXHAUSTED (not a RuntimeError), and it escaped to a 500.
        # Fail closed with an honest grounded reply and a 200 instead.
        try:
            await _cancel_kernel_task_if_pending("chat_error_uncaught")
        except BaseException as cleanup_exc:  # noqa: BLE001 — cleanup must not mask the floor
            logger.debug("Turn-death floor: kernel-task cleanup failed: %s", cleanup_exc)
        record_degradation("chat.uncaught_turn_error", e)
        logger.error("Chat uncaught error (turn-death floor engaged): %s", e, exc_info=True)
        error_reply = _grounded_chat_failure_reply()
        try:
            if pending_exchange_id:
                await _chat_preflight._complete_logged_exchange(
                    pending_exchange_id,
                    body.message,
                    error_reply,
                    record_experience=not is_benchmark,
                )
                pending_exchange_id = None
        except BaseException as log_exc:  # noqa: BLE001 — logging must not break the floor
            logger.debug("Turn-death floor: exchange logging failed: %s", log_exc)
        return JSONResponse(
            {
                "response": error_reply,
                "status": "error",
                "error_type": type(e).__name__,
                "response_confidence": "degraded",
            },
            status_code=200,
        )
    finally:
        if foreground_slot_acquired:
            _foreground_chat_lock.release(foreground_lock_token)
        if foreground_lease is not None:
            try:
                foreground_lease.close()
            except _CHAT_RECOVERABLE_ERRORS as _lease_close_exc:
                record_degradation("chat", _lease_close_exc)
                logger.debug("Foreground guard lease close skipped: %s", _lease_close_exc)


#: Distinguishes "the preflight did not set this" from "it set it to None".
#: The three inner values are bound conditionally, so the caller must be able
#: to leave a name unbound exactly where the original code did — substituting a
#: default would turn a path that raised into one that quietly proceeds.
