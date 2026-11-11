"""Carry checked public sentences from one governed generation to its surface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.voice.duplex.clause_chunker import completed_sentence_prefix


def thinking_of_a_rendered_prompt(job: dict) -> bool | None:
    """Whether a prompt sent already rendered opened her thinking channel: False when her answer begins after an
    empty think block, which is how a template renders thinking turned off; None when the prompt does not say.

    Her voice mode's live context sends its prompt rendered (core/voice/duplex/live_context.py), so no template ran
    to say. LIVE 11 November 2026, its first streamed reply sent no preview and was spoken only when whole: 105
    tokens at 12 a second, 12 s before her first word.
    """
    prompt = str(job.get('prompt') or '').rstrip()
    if job.get('messages') or not prompt.endswith('</think>'):
        return None
    opened = prompt.rfind('<think>')
    return False if opened >= 0 and not prompt[opened + len('<think>'):-len('</think>')].strip() else None


def ready_reply_preview(job: dict, text: str, *, thinking: Any, attempt: int,
                        previous: str = '') -> str:
    """Return a checked sentence prefix; private and corrective passes stay buffered."""
    from core.brain.llm.speech_hold import wait_for_her_voice

    wait_for_her_voice(job)  # between her tokens: while her voice has her next sentence in hand, her thought waits
    if thinking is None:
        thinking = thinking_of_a_rendered_prompt(job)
    if not (
        job.get('stream_user_surface') is True
        and job.get('clean_user_surface_contract') is True
        and job.get('semantic_completion_contract') is True
        and thinking is False and attempt == 0
    ):
        return ''
    if any(job.get(key) for key in (
        'user_surface_continuation_contract', 'strict_answer_contract', 'strict_value_contract',
        'proof_evaluation_contract', 'operator_evidence_contract', 'schema', 'tools',
    )):
        return ''
    body = str(text or '').lstrip()
    # An opening control delimiter may still be incomplete. Buffer the whole
    # pass rather than guessing which bytes belong to a private channel.
    if '<' in body:
        return ''
    candidate = completed_sentence_prefix(body)
    if not candidate or candidate == previous:
        return ''
    from core.brain.llm.mlx_worker_surface_quality import (
        _surface_prompt_resolution,
        _surface_quality_failure_reasons,
        _surface_quality_gate_enabled,
    )

    if not _surface_quality_gate_enabled(job) or not _surface_prompt_resolution(job).valid:
        return ''
    return '' if _surface_quality_failure_reasons(job, candidate.strip()) else candidate


@dataclass
class ReplyPreviewDelivery:
    """One request's channel and the prefix it has already received."""
    channel: Any
    delivered: str = ''
    revised: bool = False

    def publish(self, candidate: Any) -> None:
        if self.revised or not isinstance(candidate, str) or not candidate:
            return
        if not candidate.startswith(self.delivered):
            # Final reconciliation owns a revised answer. Never concatenate it
            # onto the earlier branch as though it were the next sentence.
            self.revised = True
            return
        delta = candidate[len(self.delivered):]
        if delta:
            self.channel.publish_threadsafe(delta)
            self.delivered = candidate


def publish_owned_reply_preview(pending: dict, deliveries: dict, frame: dict) -> None:
    """Late, cancelled and unrelated IPC frames cannot speak into a turn."""
    request_id = frame.get('id')
    future, delivery = pending.get(request_id), deliveries.get(request_id)
    if (frame.get('action') == 'generate' and frame.get('status') == 'progress'
            and future is not None and not future.done() and delivery is not None):
        delivery.publish(frame.get('reply_preview'))
