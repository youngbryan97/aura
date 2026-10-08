"""G09's induction resumes from its logged proposals instead of decoding them again."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.induce_g09_procedures import logged_proposer


def _decoder(seen: list[list[str]], stop_after: int | None = None):
    def decode(requests: list[str], on_record) -> None:
        seen.append(list(requests))
        for position, request in enumerate(requests):
            if stop_after is not None and position == stop_after:
                raise KeyboardInterrupt
            on_record(position, {"public_text": f"code for {request}", "termination": "stop",
                                 "generated_tokens": 3, "seconds": 0.1})
    return decode


def test_a_reset_costs_only_the_proposals_in_flight(tmp_path: Path) -> None:
    requests = ["a", "b", "c", "d"]
    first: list[list[str]] = []
    with pytest.raises(KeyboardInterrupt):
        logged_proposer(tmp_path, _decoder(first, stop_after=2), thinking=True)(requests, 0)
    again: list[list[str]] = []
    texts = logged_proposer(tmp_path, _decoder(again), thinking=True)(requests, 0)
    assert again == [["c", "d"]]
    assert texts == [f"code for {r}" for r in requests]
    # Proposals logged in one thinking mode are not reused in the other.
    closed: list[list[str]] = []
    logged_proposer(tmp_path, _decoder(closed), thinking=False)(requests, 0)
    assert closed == [requests]
