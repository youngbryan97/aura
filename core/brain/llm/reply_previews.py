"""Carry checked public sentences from one governed generation to its surface."""
from __future__ import annotations

import logging
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


_SPOKEN = '_her_spoken_prefix'
_FAILED = '_her_sentence_failed'
logger = logging.getLogger(__name__)

# Checks that judge what one sentence claims. Text after it cannot take the claim back and no repair removes the
# sentence, so a draft holding it is rejected whatever comes next. Checks that judge length, endings, loops or ratios
# can change as the draft grows and are not here; nor is a reason with a removal repair in mlx_worker.
_LASTING = frozenset({
    'fabricated_shared_history', 'self_claim_contradiction', 'ungrounded_person_narrative',
    'unsupported_affection_claim', 'unsupported_self_telemetry_claim', 'unsupported_context_continuation_claim',
    'unexpected_cjk_intrusion',
})


def a_sentence_failed(job: Any) -> bool:
    """Whether her newest checked sentence failed a check that will reject the whole draft; asked once."""
    return bool(job.pop(_FAILED, None)) if isinstance(job, dict) else False


def continuing_after(job: Any) -> str:
    """What she already said aloud of this reply when a retry continues it; empty otherwise."""
    return str(job.get(_SPOKEN) or '') if isinstance(job, dict) else ''


def her_retry_continues(job: Any, prompt: Any, messages: Any, sent: Any) -> Any:
    """A streamed reply rejected after some of it was spoken: the retry goes on from what she said, not from nothing.

    LIVE 11 November 2026, three times in her voice mode: a draft's sentences were checked one by one and spoken, a
    later sentence failed (her self-claim check twice, fabricated shared history once), and the whole draft was
    rewritten from the start with nothing streamed. She went silent 38.8, 46 and 39.0 s, and the rewrite began
    differently, so what she had said was then "corrected". Whatever the reason, what she said had passed the same
    checks sentence by sentence, so the retry goes on from it.
    Only for a prompt sent rendered (her voice's live context), where what she said is simply the start of her answer.
    The retry is checked again in full; her words already said are kept, not her rejected sentence.
    """
    sent = str(sent or '')
    if not (isinstance(job, dict) and job.get('stream_user_surface') is True and not messages and sent.strip()):
        return prompt
    job[_SPOKEN] = sent
    job.pop(_FAILED, None)
    return str(job.get('prompt') or prompt) + sent


def ready_reply_preview(job: dict, text: str, *, thinking: Any, attempt: int,
                        previous: str = '', stopped: Any = None) -> str:
    """Return a checked sentence prefix; private and corrective passes stay buffered."""
    from core.brain.llm.speech_hold import wait_for_her_voice

    wait_for_her_voice(job, stopped=stopped)  # between her tokens: while her voice has her next sentence in hand, her thought waits
    if thinking is None:
        thinking = thinking_of_a_rendered_prompt(job)
    if not (
        job.get('stream_user_surface') is True
        and job.get('clean_user_surface_contract') is True
        and job.get('semantic_completion_contract') is True
        and thinking is False and (attempt == 0 or bool(continuing_after(job)))
    ):
        return ''
    if any(job.get(key) for key in (
        'user_surface_continuation_contract', 'strict_answer_contract', 'strict_value_contract',
        'proof_evaluation_contract', 'operator_evidence_contract', 'schema', 'tools',
    )):
        return ''
    body = (continuing_after(job) + str(text or '')).lstrip()  # a retry's text, after what she already said
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
    reasons = _surface_quality_failure_reasons(job, candidate.strip())
    lasting = sorted(_LASTING.intersection(map(str, reasons or ())))
    if lasting:
        # LIVE 11 November 2026: a sentence failed fabricated_shared_history at 16:01:06 and her draft went on for
        # 20 s and 167 more failed checks before it was rejected, while she was silent. It ends at the sentence now.
        job[_FAILED] = lasting
        logger.info("🗣️ her draft ends at a sentence that failed %s; the retry goes on from what she said", lasting)
    return '' if reasons else candidate


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
