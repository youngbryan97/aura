"""One envelope, so a handler is not left inferring its own context.

The blind comparison called AutoGen's core runtime more mature and much more
locally legible than Aura's, and gave reasons: explicit publish/send/response
envelopes carrying cancellation, ids and trace; a serializer layer with a
protocol and a registry keyed by type and content type.

It also named AutoGen's own gap — agent state is saved, subscription state is
not — and that is the one thing here that is deliberately better than what it
was copied from.
"""
from __future__ import annotations

import dataclasses

import pytest

from core.runtime.what_a_message_carries import (
    AMessage,
    HowItIsSent,
    Serialiser,
    a_reply_to,
    for_a_type,
    forget_everything,
    how_to_read_it,
    how_to_write_it,
    subscribe,
    the_subscriptions,
    what_was_subscribed,
    who_is_subscribed_again,
)
from core.runtime.what_stops_it import AnExecutionContext, Stopping


@pytest.fixture(autouse=True)
def _clean():
    forget_everything()
    yield
    forget_everything()


# ------------------------------------------------------------- the envelope


def test_a_published_message_goes_to_whoever_is_listening():
    message = AMessage(topic="turn_recorded", payload={"seq": 1})
    assert message.how is HowItIsSent.PUBLISHED
    assert message.recipient == ""
    assert message.message_id


def test_a_published_message_cannot_also_name_a_recipient():
    with pytest.raises(ValueError, match="whoever is listening"):
        AMessage(
            topic="t", payload={}, how=HowItIsSent.PUBLISHED, recipient="will"
        )


def test_a_sent_message_must_name_who_it_is_for():
    with pytest.raises(ValueError, match="must name a recipient"):
        AMessage(topic="t", payload={}, how=HowItIsSent.SENT)


def test_a_response_must_say_what_it_answers():
    with pytest.raises(ValueError, match="which message it answers"):
        AMessage(topic="t", payload={}, how=HowItIsSent.RESPONSE)


def test_two_messages_have_two_ids():
    one = AMessage(topic="t", payload={})
    other = AMessage(topic="t", payload={})
    assert one.message_id != other.message_id


# ---------------------------------------------------------------- replies


def test_a_reply_joins_the_ids_up_without_the_handler_doing_it():
    """By hand is where the two stop matching."""
    asked = AMessage(
        topic="will.decision",
        payload={"q": 1},
        how=HowItIsSent.SENT,
        recipient="will",
        sender="kernel",
        trace="turn-9",
    )
    answer = a_reply_to(asked, {"decided": True})

    assert answer.how is HowItIsSent.RESPONSE
    assert answer.answers == asked.message_id
    assert answer.recipient == "kernel"
    assert answer.sender == "will"
    assert answer.trace == "turn-9"


def test_a_published_message_expects_no_reply():
    published = AMessage(topic="t", payload={})
    with pytest.raises(ValueError, match="expects no reply"):
        a_reply_to(published, {})


def test_a_reply_carries_the_same_context():
    context = AnExecutionContext(doing="a turn", trace="turn-9")
    asked = AMessage(
        topic="t", payload={}, how=HowItIsSent.SENT, recipient="will",
        context=context,
    )
    assert a_reply_to(asked, {}).context is context


# ---------------------------------------------------------- cancellation


def test_a_message_carries_the_callers_stop_signal():
    """A handler reading an ambient token gets whichever task called it."""
    stopping = Stopping("a turn")
    message = AMessage(
        topic="t",
        payload={},
        context=AnExecutionContext(stopping=stopping, doing="a turn"),
    )
    assert message.stopping is stopping
    stopping.stop("the user left")
    assert message.stopping.stopped


def test_a_message_with_no_context_still_answers_about_stopping():
    """So a handler needs no guard around the one thing it must check."""
    message = AMessage(topic="t", payload={})
    assert message.stopping.stopped is False


# ---------------------------------------------------------- serialisation


def test_a_plain_payload_goes_out_as_json_and_comes_back():
    raw, content_type = how_to_write_it({"a": 1, "b": [2, 3]})
    assert content_type == "application/json"
    assert how_to_read_it(raw, dict) == {"a": 1, "b": [2, 3]}


def test_a_dataclass_payload_comes_back_as_that_dataclass():
    @dataclasses.dataclass(frozen=True)
    class ADecision:
        chose: str
        confidence: float

    raw, content_type = how_to_write_it(ADecision(chose="wait", confidence=0.4))
    assert "dataclass" in content_type
    back = how_to_read_it(raw, ADecision)
    assert back == ADecision(chose="wait", confidence=0.4)


def test_the_registry_is_keyed_by_the_type_and_not_its_name():
    """Two packages can define the same name, and guessing returns the wrong shape."""
    class Mine:
        pass

    class AlsoMine:
        pass

    AlsoMine.__name__ = "Mine"
    one = for_a_type(Mine)
    other = for_a_type(AlsoMine)
    assert for_a_type(Mine) is one
    assert for_a_type(AlsoMine) is other


def test_a_registered_serialiser_is_used_instead_of_the_default():
    class Backwards:
        content_type = "text/backwards"

        def write(self, payload):
            return str(payload)[::-1].encode()

        def read(self, raw):
            return raw.decode()[::-1]

    class Odd:
        pass

    mine = Backwards()
    for_a_type(Odd, mine)
    assert isinstance(mine, Serialiser)
    assert how_to_read_it(b"cba", Odd) == "abc"


# --------------------------------------------------------- subscriptions


def test_a_subscription_records_where_the_handler_lives():
    subscribe("will.decision", "kernel", handler=a_reply_to)
    one = the_subscriptions()[0]
    assert one.topic == "will.decision"
    assert one.who == "kernel"
    assert one.handler.endswith(":a_reply_to")


def test_subscribing_twice_is_one_subscription():
    subscribe("t", "kernel")
    subscribe("t", "kernel")
    assert len(the_subscriptions()) == 1


def test_the_subscriptions_survive_a_restart():
    """AutoGen saves agent state and says it does not save this.

    A subscription that does not survive a restart is a listener that silently
    stops, and silently is the part that costs.
    """
    subscribe("will.decision", "kernel", handler=a_reply_to)
    subscribe("turn_recorded", "journal")
    saved = what_was_subscribed()

    forget_everything()
    assert the_subscriptions() == ()

    assert who_is_subscribed_again(saved) == 2
    assert {one.topic for one in the_subscriptions()} == {
        "will.decision", "turn_recorded"
    }


def test_a_saved_row_missing_its_topic_is_skipped_rather_than_guessed():
    assert who_is_subscribed_again([{"who": "kernel"}, {"topic": "t"}]) == 0
    assert who_is_subscribed_again(None) == 0


def test_what_was_saved_is_json_and_not_objects():
    import json

    subscribe("t", "kernel", handler=a_reply_to)
    assert json.loads(json.dumps(what_was_subscribed()))[0]["topic"] == "t"


def test_asking_again_does_not_replace_a_registration_somebody_made():
    """Otherwise an ordinary lookup discards it and nothing raises."""
    class Backwards:
        content_type = "text/backwards"

        def write(self, payload):
            return str(payload)[::-1].encode()

        def read(self, raw):
            return raw.decode()[::-1]

    class Odd:
        pass

    mine = Backwards()
    for_a_type(Odd, mine)
    assert for_a_type(Odd) is mine
    assert how_to_read_it(b"cba", Odd) == "abc"
