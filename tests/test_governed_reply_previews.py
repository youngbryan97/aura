"""A spoken preview stays public, checked, incremental and owned by one request."""
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from core.brain.llm.reply_previews import (
    ReplyPreviewDelivery,
    publish_owned_reply_preview,
    ready_reply_preview,
)

JOB = {'stream_user_surface': True, 'clean_user_surface_contract': True,
       'semantic_completion_contract': True, 'user_surface_validation_prompt': 'Hello.'}


@pytest.fixture
def validator(monkeypatch):
    calls = []
    monkeypatch.setattr(
        'core.brain.llm.mlx_worker_surface_quality._surface_quality_failure_reasons',
        lambda job, text: calls.append(text) or [],
    )
    return calls


@pytest.mark.parametrize('text,thinking,attempt', [
    ('An unfinished', False, 0), ('<think>Private reasoning.</think> Hello.', False, 0),
    ('<thi', False, 0), ('Private reasoning.', True, 0), ('Hello.', None, 0),
    ('A revised answer. ', False, 1), ('Hello.', False, 0),
    ('Ask Dr. ', False, 0), ('It is 3.', False, 0), ('1. ', False, 0),
])
def test_unready_private_or_corrective_content_stays_buffered(validator, text, thinking, attempt):
    assert not ready_reply_preview(JOB, text, thinking=thinking, attempt=attempt)
    assert not validator


def test_a_complete_public_sentence_still_passes_the_surface_integrity_gate(validator, monkeypatch):
    assert ready_reply_preview(JOB, 'Hello. Still composing', thinking=False, attempt=0) == 'Hello. '
    assert validator == ['Hello.']
    assert not ready_reply_preview(JOB, 'Hello. Still composing more', thinking=False,
                                   attempt=0, previous='Hello. ')
    assert validator == ['Hello.']
    monkeypatch.setattr('core.brain.llm.mlx_worker_surface_quality._surface_quality_failure_reasons',
                        lambda *args: ['control_artifact'])
    assert not ready_reply_preview(JOB, 'Bad content. ', thinking=False, attempt=0)


@pytest.mark.parametrize('contract', ['tools', 'schema', 'strict_answer_contract',
                                   'user_surface_continuation_contract', 'health_probe',
                                   'runtime_fact_status_contract', 'grounded_runtime_status_contract'])
def test_other_decoder_contracts_do_not_publish_previews(validator, contract):
    assert not ready_reply_preview(dict(JOB, **{contract: True}), 'Hello. ', thinking=False, attempt=0)
    assert not validator


def test_no_preview_without_a_valid_user_request_binding(validator):
    from core.conversation.user_surface_contract import make_user_surface_prompt_binding

    assert not ready_reply_preview(dict(JOB, user_surface_validation_prompt=''),
                                   'Hello. ', thinking=False, attempt=0)
    mismatched = dict(JOB, user_surface_prompt_binding=make_user_surface_prompt_binding(
        'A different request.', source='test'))
    assert not ready_reply_preview(mismatched, 'Hello. ', thinking=False, attempt=0)
    assert not validator


def test_cumulative_previews_emit_only_the_new_prefix_and_stop_at_a_revision():
    spoken = []
    delivery = ReplyPreviewDelivery(SimpleNamespace(publish_threadsafe=spoken.append))
    for text in ['Hello.', 'Hello.', "Hello. I'm here.", 'A different answer.',
                 "Hello. I'm here. This must not follow the revised branch."]:
        delivery.publish(text)
    assert spoken == ['Hello.', " I'm here."]
    assert delivery.revised


def test_stale_and_foreign_frames_never_reach_another_requests_channel():
    spoken = []
    delivery = ReplyPreviewDelivery(SimpleNamespace(publish_threadsafe=spoken.append))
    future = Future()
    pending = {'current': future}
    deliveries = {'current': delivery, 'retired': delivery}
    frame = {'action': 'generate', 'status': 'progress', 'reply_preview': 'Hello.'}
    publish_owned_reply_preview(pending, deliveries, dict(frame, id='retired'))
    assert not spoken
    publish_owned_reply_preview(pending, deliveries, dict(frame, id='current'))
    assert spoken == ['Hello.']
    future.cancel()
    publish_owned_reply_preview(pending, deliveries, dict(frame, id='current', reply_preview='Hello. Late.'))
    assert spoken == ['Hello.']


@pytest.mark.parametrize('enabled,expected', [(True, True), (False, False), ('true', False), (1, False)])
def test_preview_opt_in_survives_the_gate_and_typed_worker_request(enabled, expected):
    from core.brain.inference_gate_turn_setup import _SetsTheTurnUp
    from core.brain.llm.mlx_client import _build_the_generation_request

    options, _temperature = _SetsTheTurnUp._generate_with_metadata_sink_affective_circumplex_let(
        {'stream_user_surface': enabled})
    request = _build_the_generation_request(
        _bridge_get=lambda _name, fallback: fallback,
        adaptive_suggested_max_tokens=256, contract_generation_floor=0,
        generation_max_tokens=256, hard_output_token_ceiling=None, kwargs=options,
        prompt='Hello.', req_id='current', requested_output_contract={},
        self=SimpleNamespace(_job_seq_counter=1, temp=.7, top_p=.9, max_tokens=512),
    )
    assert request['stream_user_surface'] is expected


@pytest.mark.asyncio
async def test_client_captures_the_callers_channel_and_releases_it_with_request_ownership(monkeypatch):
    from core.brain.llm.mlx_client import MLXLocalClient
    from core.conversation.reply_stream import ReplyStreamChannel, bind_reply_stream

    client = object.__new__(MLXLocalClient)
    client._pending_generations = {}
    client._active_generations = 0
    client.max_tokens = 512
    client._current_request_id = None
    client._foreground_generation_watchdog = None
    monkeypatch.setattr(client, '_deadline_bound_first_token_hard_ceiling', lambda *a, **k: 10.)
    monkeypatch.setattr(client, '_start_foreground_first_token_watchdog', lambda *a, **k: None)
    monkeypatch.setattr(client, '_mark_generation_started', lambda request_id, **k:
                        setattr(client, '_current_request_id', request_id))
    monkeypatch.setattr(client, '_clear_active_generation_tracking', lambda:
                        setattr(client, '_current_request_id', None))
    future = Future()
    owner, another_turn = ReplyStreamChannel(), ReplyStreamChannel()
    with bind_reply_stream(owner):
        client._generate_inner_part_3(SimpleNamespace(remaining=10.), True, future, 'Hello.',
                                      {'stream_user_surface': True}, 'current')
        frame = {'id': 'current', 'action': 'generate', 'status': 'progress', 'reply_preview': 'Hello. '}
        with bind_reply_stream(another_turn):
            publish_owned_reply_preview(client._pending_generations, client._pending_reply_previews, frame)
            assert owner.stats.published == 1
            assert another_turn.stats.published == 0
            await client._finish_generation_ownership('current', future, None, release_lane=False)
            publish_owned_reply_preview(client._pending_generations, client._pending_reply_previews,
                                        dict(frame, reply_preview='Hello. Too late. '))
    assert client._pending_reply_previews == {}
    assert client._pending_generations == {}
    assert client._active_generations == 0
    assert owner.stats.published == 1
    owner.close()
    another_turn.close()


@pytest.mark.parametrize('prompt,ready', [
    # Her voice mode's live context renders its own prompt with thinking closed (core/voice/duplex/live_context.py).
    ('<|im_start|>user\nHello?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', True),
    # A rendered prompt whose thinking is open, or that does not say, stays buffered as before.
    ('<|im_start|>user\nHello?<|im_end|>\n<|im_start|>assistant\n<think>\nI should greet', False),
    ('<|im_start|>user\nHello?<|im_end|>\n<|im_start|>assistant\n', False),
])
def test_a_prompt_sent_rendered_says_itself_whether_thinking_was_closed(validator, prompt, ready):
    """LIVE 11 November 2026: the live context's first reply sent no preview, so it was spoken only when whole."""
    preview = ready_reply_preview(dict(JOB, prompt=prompt), 'Hello. Still composing', thinking=None, attempt=0)
    assert (preview == 'Hello. ') is ready


def test_a_rejected_streamed_reply_is_retried_from_what_she_already_said(validator):
    """LIVE 11 November 2026: a draft's checked sentences were spoken, a later one failed her self-claim check, and the
    whole draft was rewritten from nothing with nothing streamed: 38.8 s and 46 s of silence, then a "correction"."""
    from core.brain.llm.reply_previews import continuing_after, her_retry_continues

    rendered = '<|im_start|>user\nHello?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    job = dict(JOB, prompt=rendered)
    said = 'Hello there. I can hear you. '
    assert her_retry_continues(job, rendered, None, said) == rendered + said
    assert continuing_after(job) == said
    # Her retry streams, and her voice is handed only what comes after what she said.
    preview = ready_reply_preview(job, 'Good to hear you. More', thinking=None, attempt=1)
    assert preview == said + 'Good to hear you. '
    delivery = ReplyPreviewDelivery(SimpleNamespace(publish_threadsafe=lambda chunk: published.append(chunk)),
                                    delivered=said)
    published = []
    delivery.publish(preview)
    assert published == ['Good to hear you. ']


@pytest.mark.parametrize('job,messages,sent', [
    (dict(JOB, prompt='p'), [{'role': 'user', 'content': 'Hi'}], 'Said. '),  # a templated prompt: rebuilt, not continued
    (dict(JOB, prompt='p'), None, ''),  # nothing was said yet
    (dict(JOB, prompt='p', stream_user_surface=False), None, 'Said. '),  # nothing was streamed
])
def test_a_retry_with_nothing_spoken_or_no_rendered_prompt_starts_over(job, messages, sent):
    from core.brain.llm.reply_previews import continuing_after, her_retry_continues

    assert her_retry_continues(job, 'p', messages, sent) == 'p'
    assert continuing_after(job) == ''


@pytest.mark.parametrize('reasons,ends', [
    (['fabricated_shared_history'], True), (['self_claim_contradiction', 'too_short_for_user_turn'], True),
    (['too_short_for_user_turn'], False), (['truncated_tail'], False),
    (['unsupported_self_condition_operational_claim'], False),  # mlx_worker removes that sentence and keeps the rest
])
def test_a_draft_ends_at_a_sentence_no_later_text_can_save(monkeypatch, reasons, ends):
    """LIVE 11 November 2026: a sentence failed fabricated_shared_history at 16:01:06; the draft went on 20 s and 167
    more failed checks before its rejection, with her silent. A check that more text could still pass waits for it."""
    from core.brain.llm.reply_previews import a_sentence_failed, her_retry_continues

    monkeypatch.setattr('core.brain.llm.mlx_worker_surface_quality._surface_quality_failure_reasons',
                        lambda *args: reasons)
    job = dict(JOB, prompt='p')
    assert not ready_reply_preview(job, 'We met last spring. And', thinking=False, attempt=0)
    assert a_sentence_failed(job) is ends
    assert a_sentence_failed(job) is False  # asked once: the retry does not end at the same sentence
    ready_reply_preview(job, 'We met last spring. And', thinking=False, attempt=0)
    her_retry_continues(job, 'p', None, 'Said. ')
    assert a_sentence_failed(job) is False


def test_the_token_loop_ends_the_draft_where_its_sentence_failed():
    from pathlib import Path

    loop = Path(__file__).resolve().parents[1].joinpath('core/brain/llm/mlx_worker.py').read_text()
    published = loop.index("'status': 'progress', 'reply_preview': preview,")
    assert loop.index('if a_sentence_failed(job):\n', published) - published < 400
