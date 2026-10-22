"""A checked opening sentence reaches synthesis while cognition still runs."""
import asyncio
from concurrent.futures import Future

import pytest

from core.brain.llm.reply_previews import (
    ReplyPreviewDelivery,
    publish_owned_reply_preview,
    ready_reply_preview,
)
from core.conversation.reply_stream import ReplyStreamChannel, bind_reply_stream
from core.voice.duplex.clause_chunker import StreamingChunker, completed_sentence_prefix
from core.voice.duplex.governed_stream import stream_governed_reply


def test_short_opener_releases_before_the_character_budget_is_full():
    chunker = StreamingChunker(first_max_chars=100, max_chars=200)
    assert chunker.push('Hello.') == []
    assert chunker.push(" I'm still working") == ['Hello.']
    assert chunker.push(' through the next sentence. ') == []
    assert chunker.flush() == ["I'm still working through the next sentence."]


@pytest.mark.parametrize('text,prefix', [
    ('Ask Dr. ', ''), ('The value is 3.', ''), ('1. ', ''),
    ('Ask Dr. Chen about 3.14 seconds. Still thinking', 'Ask Dr. Chen about 3.14 seconds. '),
    ('1. Check the recording. Then listen', '1. Check the recording. '),
    ('First. Second. unfinished', 'First. Second. '),
])
def test_preview_waits_for_real_sentence_boundaries(text, prefix):
    assert completed_sentence_prefix(text) == prefix


@pytest.mark.asyncio
async def test_checked_worker_prefix_reaches_speech_before_the_answer_finishes(monkeypatch):
    checks, spoken = [], []
    monkeypatch.setattr(
        'core.brain.llm.mlx_worker_surface_quality._surface_quality_failure_reasons',
        lambda job, text: checks.append(('worker', text)) or [],
    )
    monkeypatch.setattr(
        'core.voice.duplex.governed_stream.govern_clause',
        lambda text: (checks.append(('clause', text)) or text, '', False),
    )
    started_speaking = asyncio.Event()
    finish_cognition = asyncio.Event()
    channel = ReplyStreamChannel()
    pending = {'current': Future()}
    deliveries = {'current': ReplyPreviewDelivery(channel)}
    job = {'stream_user_surface': True, 'clean_user_surface_contract': True,
           'semantic_completion_contract': True, 'user_surface_validation_prompt': 'Hello.'}

    async def cognition():
        with bind_reply_stream(channel):
            preview = ready_reply_preview(job, "Hello. I'm still composing", thinking=False, attempt=0)
            publish_owned_reply_preview(pending, deliveries, {
                'id': 'current', 'action': 'generate', 'status': 'progress', 'reply_preview': preview,
            })
            await finish_cognition.wait()
            deliveries['current'].publish("Hello. I'm still composing the rest. ")

    def speak(text):
        spoken.append(text)
        started_speaking.set()

    producer = asyncio.create_task(cognition())
    consumer = asyncio.create_task(stream_governed_reply(
        channel.drain(timeout_s=1.), first_max_chars=100, max_chars=200, speak=speak))
    try:
        await asyncio.wait_for(started_speaking.wait(), timeout=1.)
        assert not producer.done()
        assert spoken == ['Hello.']
        assert checks[:2] == [('worker', 'Hello.'), ('clause', 'Hello.')]
        finish_cognition.set()
        await asyncio.wait_for(producer, timeout=1.)
        outcome = await asyncio.wait_for(consumer, timeout=1.)
        assert outcome.spoken == "Hello. I'm still composing the rest."
        assert not outcome.refused
    finally:
        finish_cognition.set()
        channel.close()
        for task in (producer, consumer):
            if not task.done():
                task.cancel()
        await asyncio.gather(producer, consumer, return_exceptions=True)
