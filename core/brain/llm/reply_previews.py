"""Carry checked public sentences from one governed generation to its surface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.voice.duplex.clause_chunker import completed_sentence_prefix


def ready_reply_preview(job: dict, text: str, *, thinking: Any, attempt: int,
                        previous: str = '') -> str:
    """Return a checked sentence prefix; private and corrective passes stay buffered."""
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
