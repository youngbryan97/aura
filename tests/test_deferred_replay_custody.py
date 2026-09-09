from core.memory.a_deferral_is_not_a_refusal import DeferredWrites


def test_retrying_a_deferred_write_does_not_enqueue_a_second_copy():
    def retry(item):
        queue.hold(item, "deferred again")
        return False

    queue = DeferredWrites("redeferral", retry, limit=3, interval_s=0)
    queue.hold({"id": "episode"})
    for _ in range(10):
        assert queue.replay() == 0
        assert len(queue) == 1
    assert queue.shed == 0


def test_retry_identity_survives_reconstructed_payloads():
    def retry(item):
        queue.hold(dict(item))
        return False

    queue = DeferredWrites("stable-identity", retry, interval_s=0, identity=lambda item: item["id"])
    queue.hold({"id": "episode"})
    for _ in range(10):
        queue.replay()
    assert len(queue) == 1
    assert queue.shed == 0


def test_capacity_accounts_for_inflight_custody_and_reports_every_drop():
    def retry(item):
        queue.hold("new-one")
        queue.hold("new-two")
        return False

    queue = DeferredWrites("capacity", retry, limit=2, interval_s=0)
    queue.hold("original")
    queue.replay()
    assert len(queue) == 2
    assert queue.shed == 1


def test_retry_can_enqueue_different_work_without_losing_the_original():
    def retry(item):
        queue.hold("new")
        queue.hold(item)
        return False

    queue = DeferredWrites("new-during-retry", retry, limit=3, interval_s=0)
    queue.hold("original")
    queue.replay()
    assert len(queue) == 2
    assert queue.shed == 0
