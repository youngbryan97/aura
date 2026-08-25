"""Generating is progress, whether or not it is visible yet.

The worker's only progress signal to the parent was the `token` stream message,
and that message was sent ONLY when the decoded step produced visible new text:

    if emit_text:
        ipc_writer.put({... "status": "token", "text": emit_text ...})

Decoding legitimately produces no visible delta for a while — a detokenizer
holding a partial UTF-8 sequence, suppressed start ids, a stop sequence being
scanned. To the parent that is indistinguishable from a wedged worker, because
`_current_first_token_at` is set from that message alone.

Live on the desktop surface 2026-07-26, on an ~800-token prompt:

    [MLX] First-token HARD CEILING exceeded (livelocked: heartbeats but zero
          tokens) ... 107.7s elapsed, sla=240.0s
    Cortex ran past this turn's deadline (107.7s elapsed, budget 106.8s) but is
          healthy (heartbeat 0.8s ago). Cancelling the request.
    Proof/operator request requires a valid Cortex response; refusing
          lower-lane fallback.

A healthy generation was cancelled and the turn was lost — the same category
error as the rest of this pass: work in progress read as damage.

A step that yields no visible text now emits `progress` instead. It carries no
text, and unlike `token` it is essential, so it also cannot be dropped by the
IPC writer's backpressure shedding.
"""
from __future__ import annotations

import re
from pathlib import Path

WORKER = Path("core/brain/llm/mlx_worker.py")
CLIENT = Path("core/brain/llm/mlx_client.py")


def _emit_block() -> str:
    """The emission block, bounded by the stop check that follows it.

    The end marker used to be "if stop_hit:" exactly. A second condition was
    added — `if stop_hit or semantic_stop_ready:` — and every case in this
    file died on ValueError from the slice, about code that had not changed.
    The prefix is the stable part.
    """
    src = WORKER.read_text(encoding="utf-8")
    start = src.index("                                    emit_text = (")
    return src[start : src.index("if stop_hit", start)]


def test_a_step_with_no_visible_text_still_reports_progress() -> None:
    block = _emit_block()
    assert 'if emit_text:' in block
    assert re.search(r"else:\s", block), "the invisible-token case must be handled"
    assert '"status": "progress"' in block, (
        "a token that adds no visible text must still signal progress"
    )


def test_the_progress_signal_carries_no_text() -> None:
    """It is a liveness ping, not a second copy of the stream."""
    block = _emit_block()
    else_branch = block[block.rindex("else:") :]
    assert '"status": "progress"' in else_branch
    assert '"text"' not in else_branch, "progress must not duplicate stream text"
    assert '"tokens_generated": token_count' in else_branch


def test_the_client_counts_progress_as_first_token_progress() -> None:
    """Both statuses must reach _mark_token_progress, or the fix is inert.

    Checked by calling it. The handling moved into
    _record_worker_stream_progress, so a text slice taken after the status
    branch no longer contained the call and the test failed about a path that
    still works.
    """
    from core.brain.llm.mlx_client import MLXLocalClient

    client = MLXLocalClient.__new__(MLXLocalClient)
    marked: list[object] = []
    client._mark_token_progress = marked.append  # type: ignore[method-assign]

    # Frames shaped as the worker emits them: a progress frame carries the
    # token count, which is what makes it token progress rather than a bare
    # liveness ping.
    for status in ("progress", "token"):
        MLXLocalClient._record_worker_stream_progress(
            client,
            {"id": f"req-{status}", "tokens_generated": 7},
            status=status,
            action="stream",
        )
    assert marked == ["req-progress", "req-token"], marked


def test_a_progress_frame_without_a_token_count_is_only_liveness() -> None:
    """The distinction the client draws, stated so it cannot drift silently."""
    from core.brain.llm.mlx_client import MLXLocalClient

    client = MLXLocalClient.__new__(MLXLocalClient)
    token_progress: list[object] = []
    plain_progress: list[object] = []
    client._mark_token_progress = token_progress.append  # type: ignore[method-assign]
    client._mark_progress = lambda: plain_progress.append(True)  # type: ignore[method-assign]

    MLXLocalClient._record_worker_stream_progress(
        client, {"id": "req"}, status="progress", action="stream"
    )
    assert token_progress == []
    assert plain_progress == [True]


def test_progress_is_essential_and_cannot_be_shed() -> None:
    """`token` is sheddable under IPC backpressure; the liveness signal is not.

    Asked of the predicate rather than of its source. It was a single
    `status not in {...}` set and is now a three-level priority — telemetry,
    progress, terminal — which is the same guarantee expressed better, and the
    regex stopped matching anything.
    """
    from core.brain.llm.mlx_worker import IPCWriterThread

    priority = IPCWriterThread._delivery_priority
    assert priority({"status": "token"}) == 0, "stream text stays sheddable"
    assert priority({"status": "heartbeat"}) == 0
    assert priority({"status": "progress"}) > priority({"status": "token"}), (
        "the progress signal must survive backpressure, or the livelock "
        "false-positive returns exactly when the queue is busiest"
    )
    assert IPCWriterThread._is_essential({"status": "progress"})
    assert not IPCWriterThread._is_essential({"status": "token"})
    # A terminal frame still outranks progress.
    assert priority({"status": "ok"}) > priority({"status": "progress"})
