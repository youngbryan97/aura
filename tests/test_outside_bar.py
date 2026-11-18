"""What measures her from outside is out of her reach, and says what it measured."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from core.agency.her_goal_at_work import governs_her
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.state_ownership import StateOwnershipViolation, assert_state_path_allowed
from core.self_modification.mutation_tiers import classify_mutation_path
from outside_bar import grade as grading
from outside_bar import run as bar

_HARNESS = Path(bar.__file__).resolve().parent
_DATA = pytest.mark.skipif(not (grading.BENCH / "math500/test.jsonl").exists(), reason="the benchmarks are not on this host")


def _row(domain: str, right: bool) -> dict:
    return {"domain": domain, "right": right}


def _rows(domain: str, marks: str) -> dict:
    return {f"{domain}:{i}": _row(domain, mark == "1") for i, mark in enumerate(marks)}


def test_the_harness_imports_nothing_of_hers():
    allowed = set(sys.stdlib_module_names) | {"outside_bar", "grading", "yaml"}
    for path in _HARNESS.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not {name.split(".")[0] for name in names} - allowed, path.name


def test_her_own_changes_cannot_reach_it():
    assert classify_mutation_path("outside_bar/run.py").sealed
    assert governs_her("outside_bar/grade.py")
    assert governs_her("core/self_modification/mutation_tiers.py")
    assert governs_her("core/runtime/state_ownership.py")


@pytest.mark.parametrize("target", ["outside_bar/run.py", "~/.aura/outside-bar/ledger.jsonl"])
def test_the_write_gateway_refuses_it(target):
    with pytest.raises(StateOwnershipViolation):
        get_file_write_gateway().write_text(target, "changed", source="test")


def test_a_link_does_not_lead_past_the_refusal(tmp_path):
    held = tmp_path / "outside-bar"
    held.mkdir()
    (tmp_path / "plain").symlink_to(held)
    with pytest.raises(StateOwnershipViolation):
        assert_state_path_allowed(tmp_path / "plain" / "ledger.jsonl", source="test")


@_DATA
def test_the_seal_matches_what_is_here():
    assert bar.what_broke_the_seal() == []


def test_a_changed_harness_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(bar, "what_broke_the_seal", lambda: ["outside_bar/grade.py"])
    monkeypatch.setattr(bar, "HOME", tmp_path)
    assert bar.ask("changed", 1, "panel", "http://127.0.0.1:9", tmp_path) == 2
    assert not (tmp_path / "runs").exists()


@_DATA
@pytest.mark.parametrize(("domain", "reply", "truth", "right"), [
    ("math500", "So it is \\boxed{\\frac{1}{2}}.", "\\frac{1}{2}", True),
    ("math500", "So it is \\boxed{3}.", "\\frac{1}{2}", False),
    ("math500", "I could not work that out today, sorry about that.", "2", False),
    ("bbeh_mini", "Working it through.\nThe answer is: (b)", "b", True),
    ("bbeh_mini", "Working it through.\nThe answer is: (c)", "b", False),
])
def test_each_grader_tells_a_right_answer_from_a_wrong_one(domain, reply, truth, right):
    assert grading.grade(domain, reply, truth)[0] is right


@_DATA
@pytest.mark.parametrize("domain", ["blocksworld", "mystery_blocksworld"])
def test_a_gold_plan_passes_and_the_same_plan_cut_short_does_not(domain):
    from outside_bar import planbench
    from outside_bar.panel import every_task

    task = next(iter(every_task(domain).values()))
    words = planbench.plan_as_text(task["truth"]["ground_truth_plan"], domain)
    assert grading.grade(domain, words, task["truth"])[0] is True
    steps = [line for line in words.splitlines() if line.strip() and "PLAN" not in line]
    assert grading.grade(domain, "\n".join(steps[:-1]) + "\n[PLAN END]", task["truth"])[0] is False


def test_a_lopsided_loss_is_a_regression_and_an_even_trade_is_not():
    before = _rows("math500", "1" * 12)
    assert bar.compared(before, _rows("math500", "0" * 6 + "1" * 6), ["math500"])["verdict"] == "REGRESSED"
    mixed = {**_rows("math500", "110" * 4)}
    assert bar.compared(mixed, _rows("math500", "011" * 4), ["math500"])["verdict"] == "HELD"


def test_a_run_missing_a_problem_never_counts_as_held():
    before = _rows("math500", "1111")
    after = dict(list(before.items())[:3])
    assert bar.compared(before, after, ["math500"])["verdict"] == "INCOMPLETE"
    assert bar.compared({}, {}, ["math500"])["verdict"] == "INCOMPLETE"


def test_one_domain_collapsing_is_a_regression_though_the_others_gain():
    before = {**_rows("math500", "1" * 8 + "0" * 8), **_rows("bbeh_mini", "0" * 8)}
    after = {**_rows("math500", "0" * 8 + "1" * 8), **_rows("bbeh_mini", "1" * 8)}
    assert bar.compared(before, after, ["math500", "bbeh_mini"])["verdict"] == "HELD"
    before = {**_rows("math500", "1" * 9), **_rows("bbeh_mini", "0" * 9)}
    after = {**_rows("math500", "0" * 9), **_rows("bbeh_mini", "1" * 9)}
    result = bar.compared(before, after, ["math500", "bbeh_mini"])
    assert result["losses"] == result["gains"] and result["verdict"] == "REGRESSED"


def test_the_chance_is_exact():
    assert bar.chance_of(0, 0) == 1.0
    assert bar.chance_of(5, 0) == pytest.approx(1 / 32)
    assert bar.chance_of(3, 3) == pytest.approx(42 / 64)


def test_an_edited_ledger_line_shows(monkeypatch, tmp_path):
    monkeypatch.setattr(bar, "HOME", tmp_path)
    monkeypatch.setattr(bar, "seal_id", lambda: "seal")
    for index in range(3):
        bar.append_to_ledger({"kind": "run", "run": str(index)})
    assert bar.the_ledger_holds()
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(ledger.read_text().replace('"run": "0"', '"run": "9"'))
    assert not bar.the_ledger_holds()


@pytest.fixture
def her_door():
    heard: list[dict] = []

    class Door(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            heard.append(body)
            reply = json.dumps({"response": "The answer is: 4" if "two" in body["message"] else "The answer is: 9",
                                "status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Door)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", heard
    server.shutdown()


def _two_tasks(*_):
    return [{"id": "bbeh_mini:a", "domain": "bbeh_mini", "request": "two and two", "truth": "4"},
            {"id": "bbeh_mini:b", "domain": "bbeh_mini", "request": "three and three", "truth": "6"}]


def _a_quiet_harness(monkeypatch, tmp_path):
    monkeypatch.setattr(bar, "HOME", tmp_path / "home")
    monkeypatch.setattr(bar, "HOLD", tmp_path / "measuring.hold")
    monkeypatch.setattr(bar, "what_broke_the_seal", list)
    monkeypatch.setattr(bar, "seal_id", lambda: "seal")
    monkeypatch.setattr(bar, "source_commit", lambda _: "abc")
    monkeypatch.setattr(bar, "tasks", _two_tasks)


def test_a_run_asks_each_problem_alone_grades_it_and_is_written_once(monkeypatch, tmp_path, her_door):
    _a_quiet_harness(monkeypatch, tmp_path)
    base, heard = her_door
    assert bar.ask("first", 1, "panel", base, tmp_path) == 0
    assert bar.score("first")["bbeh_mini"] == {"right": 1, "of": 2}
    assert len({body["session_id"] for body in heard}) == 2, "no problem shares a conversation with another"
    assert bar.ask("first", 1, "panel", base, tmp_path) == 0
    assert len(heard) == 2, "a finished problem is not asked again"
    assert not bar.HOLD.exists(), "her landing is held only while a run measures her"


def test_her_landing_is_held_during_a_run(monkeypatch, tmp_path, her_door):
    _a_quiet_harness(monkeypatch, tmp_path)
    seen = []
    real = bar._her_reply
    monkeypatch.setattr(bar, "_her_reply", lambda *a: (seen.append(bar.HOLD.exists()), real(*a))[1])
    assert bar.ask("held", 1, "panel", her_door[0], tmp_path) == 0
    assert seen == [True, True]


def test_a_door_that_does_not_answer_stops_the_run_and_scores_nothing(monkeypatch, tmp_path):
    _a_quiet_harness(monkeypatch, tmp_path)
    assert bar.ask("down", 1, "panel", "http://127.0.0.1:9", tmp_path) == 3
    assert bar.rows_of("down") == {}
    assert not bar.HOLD.exists()


def test_a_digest_is_of_the_bytes(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"x")
    assert bar._digest(path) == hashlib.sha256(b"x").hexdigest()
