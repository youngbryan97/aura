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
