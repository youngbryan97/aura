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


BENCH = Path("~/.aura/benchmarks").expanduser()


@pytest.mark.skipif(not (BENCH / "bbh-src/bbh/word_sorting.json").exists()
                    or not (BENCH / "g09/dm-math/mathematics_dataset-v1.0/interpolate/numbers__gcd.txt").exists(),
                    reason="the BBH and DeepMind mathematics files are not on this host")
def test_every_dm_module_and_bbh_task_splits_into_disjoint_halves_graded_by_its_own_matcher() -> None:
    from tools.induce_g09_procedures import grader_for, problems

    for kind in ("bbh_word_sorting", "bbh_date_understanding", "dm_numbers__gcd"):
        development, test = problems(kind, "development"), problems(kind, "test")
        assert development and test and not {r["key"] for r in development} & {r["key"] for r in test}
        assert abs(len(development) - len(test)) <= 1
        row = development[0]
        assert grader_for(kind)(str(row["truth"]), row["truth"])
    framed = problems("bbh_word_sorting", "development")[0]
    assert framed["key"].startswith("word_sorting:") and framed["problem"].rstrip().endswith("Let's think step by step.")
    option = next(r for r in problems("bbh_date_understanding", "development") if r["truth"].startswith("("))
    assert grader_for("bbh_date_understanding")(option["truth"].strip("()"), option["truth"])


@pytest.mark.skipif(not (BENCH / "bbh-src/bbh/word_sorting.json").exists(),
                    reason="the BBH files are not on this host")
def test_a_bbh_procedure_is_checked_by_its_own_grader_though_its_keys_do_not_name_the_kind(tmp_path: Path,
                                                                                           monkeypatch) -> None:
    import json
    import sys

    from tools import induce_g09_procedures

    # Sorts the words of the last "List:"; BBH's grader must agree with it on every known answer.
    code = ("```python\ndef solve(problem):\n"
            "    words = problem.rsplit('List:', 1)[1].split('\\n')[0].split()\n"
            "    return ' '.join(sorted(words))\n```")
    replies = tmp_path / "replies.json"
    replies.write_text(json.dumps([code]), encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setattr(sys, "argv", ["induce_g09_procedures.py", "--kinds", "bbh_word_sorting", "--lines", "1",
                                      "--rounds", "1", "--fake-proposer", str(replies), "--output", str(out)])
    assert induce_g09_procedures.main() == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))["bbh_word_sorting"]
    assert summary["admitted"] == 1 and summary["best"][0] == summary["best"][1] == summary["development"]


@pytest.mark.skipif(not (BENCH / "bbh-src/bbh/word_sorting.json").exists(),
                    reason="the BBH files are not on this host")
def test_a_book_given_as_steps_reaches_the_induction(tmp_path: Path, monkeypatch) -> None:
    import json
    import sys

    from core.learning import procedures_from_solved_examples as procedures
    from core.learning.procedures_from_solved_examples import ProcedureBook, Signature, dumps
    from tools import induce_g09_procedures

    book = ProcedureBook({"percent": Signature({"percent": 1.0}, 0.99)},
                         {"percent": ["def solve(problem):\n    return '1'\n"]}, {"percent": 4})
    (tmp_path / "book.json").write_text(dumps(book.to_json()), encoding="utf-8")
    (tmp_path / "replies.json").write_text(json.dumps(["no code"]), encoding="utf-8")
    given: dict = {}
    real = procedures.induce

    def spy(*args, **kwargs):
        given["steps"] = kwargs.get("steps")
        return real(*args, **kwargs)

    monkeypatch.setattr(procedures, "induce", spy)
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setattr(sys, "argv", ["induce_g09_procedures.py", "--kinds", "bbh_word_sorting", "--rounds", "1",
                                      "--lines", "1", "--fake-proposer", str(tmp_path / "replies.json"),
                                      "--steps-book", str(tmp_path / "book.json"), "--output", str(tmp_path / "out")])
    assert induce_g09_procedures.main() == 0
    assert given["steps"] is not None and set(given["steps"].book.procedures) == {"percent"}
