"""Her real calls of what she may change are kept while she runs and tried on both sides of each change (F16).

Rounds five and six of the red team left changes on probation that a benchmark asking only valid questions does not
notice: a guard deleted, bad input turned into 0. The calls she is really asked are evidence no test author chose. They
are kept in her state, never in her public source, and the unchanged outside check tries them through guard tests that
exist only in a scratch clone.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.agency import her_goal_at_work as work
from core.runtime.a_call_that_failed import a_call_written_back, the_call_that_failed
from core.self_modification import her_real_calls as real
from core.self_modification import loop_drill
from tests.test_her_goal_is_worked_by_her import (  # noqa: F401 - a fixture, found by its own name
    GOAL,
    her_source,
)
from tests.test_her_own_yes_asks_the_gates import _a_draft_of_hers, writes_verdict

# ── writing a call back ─────────────────────────────────────────────────


def _frame_of(function, *args, **kwargs):
    seen = []

    def tracer(frame, event, _arg):
        if event == "call" and frame.f_code is function.__code__ and not seen:
            seen.append(a_call_written_back(frame))
        return None

    sys.settrace(tracer)
    try:
        function(*args, **kwargs)
    finally:
        sys.settrace(None)
    return seen[0]


def test_a_call_that_started_is_written_back_with_every_name():
    call = _frame_of(loop_drill.nth, [1, 2, 3], 1)
    assert (call.module, call.qualname, call.file) == ("core.self_modification.loop_drill", "nth",
                                                       "core/self_modification/loop_drill.py")
    assert call.keywords == (("rows", "[1, 2, 3]"), ("n", "1")) and not call.error_type


def test_a_fault_is_the_same_call_with_what_it_raised():
    try:
        loop_drill.count_in("seven")
    except ValueError as error:
        failed = the_call_that_failed(error)
    assert failed.keywords == (("text", "'seven'"),) and failed.error_type == "builtins.ValueError"
    assert failed.line > 0 and failed.error_message


# ── keeping a sample ────────────────────────────────────────────────────


@pytest.fixture
def watched():
    samples: list[real.HerRealCalls] = []

    def make(kept=3, draws=()):
        queue = list(draws)
        sample = real.HerRealCalls(kept=kept, draw=lambda: queue.pop(0))
        samples.append(sample)
        return sample

    yield make
    for sample in samples:
        sample.stop()


def test_each_call_has_the_same_chance_to_be_kept(watched):
    """Algorithm R: the first three are kept, then the nth call takes a slot when a draw times n lands in one."""
    sample = watched(kept=3, draws=[0.5, 0.1])
    assert sample.watch([("core.self_modification.loop_drill", "nth")]) == ["core/self_modification/loop_drill.py::nth"]
    for n in range(5):
        assert loop_drill.nth([10, 20], n % 2) in (10, 20), "watching does not change what the call does"
    entry = sample.corpus()["functions"]["core/self_modification/loop_drill.py::nth"]
    assert entry["seen"] == 5
    # The fourth call drew 0.5 * 4 = slot 2; the fifth drew 0.1 * 5 = slot 0.
    assert [dict(row["keywords"])["n"] for row in entry["calls"]] == ["0", "1", "1"]


def test_a_call_that_cannot_be_written_back_is_counted_and_still_answered(watched):
    sample = watched()
    sample.watch([("core.self_modification.loop_drill", "label")])
    holder = type("Person", (), {"name": "ada"})()
    assert loop_drill.label(holder) == "Ada"
    entry = sample.corpus()["functions"]["core/self_modification/loop_drill.py::label"]
    assert entry["calls"] == [] and entry["seen"] == 1 and "not a plain value" in next(iter(entry["why_not"]))


def test_a_failure_inside_the_watch_never_reaches_the_call(watched, monkeypatch):
    sample = watched()
    sample.watch([("core.self_modification.loop_drill", "nth")])
    monkeypatch.setattr(real, "a_call_written_back", lambda _frame: 1 / 0)
    assert loop_drill.nth([4], 0) == 4


def test_a_function_whose_module_is_not_loaded_is_not_imported_to_watch_it(watched):
    sample = watched()
    assert sample.watch([("core.no_such_module_for_her_real_calls", "f")]) == []
    assert "core.no_such_module_for_her_real_calls" not in sys.modules


def test_a_kept_file_is_read_as_data_and_a_row_that_is_not_plain_source_is_dropped():
    sample = real.HerRealCalls()
    good = {"module": "m", "qualname": "f", "file": "m.py", "keywords": [["x", "1"]]}
    taken = {"functions": {"m.py::f": {"seen": 9, "calls": [
        good,
        {**good, "keywords": [["x", "__import__('os').system('echo owned')"]]},
        {**good, "keywords": [["x=1) or print('owned'", "1"]]},
    ]}}}
    assert sample.take_up(taken) == 1
    assert sample.corpus()["functions"]["m.py::f"]["calls"] == [good] and sample.corpus()["functions"]["m.py::f"]["seen"] == 9


# ── what a commit changed, and the guard tests made from her calls ──────


_QTY = '''def quantity(text):
    """A quantity written as text."""
    number = int(text)
    if number < 0:
        raise ValueError("a quantity is never negative")
    return number


class Shelf:
    def __init__(self, rows):
        self.rows = rows

    def row(self, n):
        return self.rows[n]
'''
_GUARD = '''from shop.qty import quantity


def test_a_quantity_reads_its_number():
    assert quantity("3") == 3
'''
_BLANK_IS_ZERO = '''from shop.qty import quantity


def test_a_blank_quantity_is_zero():
    assert quantity("") == 0
'''
#: A mend of blank text that also deletes the guard against a negative quantity.
_MEND_THAT_DELETES_A_GUARD = '''def quantity(text):
    """A quantity written as text."""
    try:
        number = int(text)
    except ValueError:
        return 0
    return max(number, 0)


class Shelf:
    def __init__(self, rows):
        self.rows = rows

    def row(self, n):
        return self.rows[n]
'''


def _git(repo, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.fsmonitor=false", *args],
                          cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def shop(tmp_path):
    repo = tmp_path / "shop-repo"
    (repo / "shop").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "shop" / "__init__.py").write_text("")
    (repo / "shop" / "qty.py").write_text(_QTY)
    (repo / "tests" / "test_qty.py").write_text(_GUARD)
    (repo / "conftest.py").write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).parent))\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "shop")
    return repo


def _mended(repo: Path, source: str) -> str:
    (repo / "shop" / "qty.py").write_text(source)
    (repo / "tests" / "test_blank_quantity.py").write_text(_BLANK_IS_ZERO)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "mend")
    return _git(repo, "rev-parse", "HEAD")


def _her_calls(*texts: str) -> dict:
    rows = [{"module": "shop.qty", "qualname": "quantity", "file": "shop/qty.py", "keywords": [["text", repr(text)]]}
            for text in texts]
    return {"functions": {"shop/qty.py::quantity": {"seen": len(rows), "calls": rows, "awaited": False}}}


def test_the_functions_a_commit_changed_are_named_by_their_lines(shop):
    commit = _mended(shop, _MEND_THAT_DELETES_A_GUARD.replace("return self.rows[n]", "return self.rows[n - 1]"))
    assert real.functions_a_commit_changed(shop, commit) == [("shop/qty.py", "quantity"), ("shop/qty.py", "Shelf.row")]


def test_a_guard_test_makes_each_call_and_leaves_what_it_came_to_to_the_check():
    source = real.guard_test_source("shop/qty.py::quantity", _her_calls("4", "-2")["functions"]["shop/qty.py::quantity"])
    compile(source, "guard.py", "exec")
    assert "quantity(text='4')" in source and "quantity(text='-2')" in source and source.count("def test_her_call_") == 2
    assert "never" in source and "committed" in source
    assert "assert _raised_as(" not in source, "with nothing learned, nothing is pinned"


def _run_guards(source: str, shop: Path) -> dict[str, bool]:
    """Each guard test of ``source`` run against the shop as it stands: passed or not."""
    namespace: dict = {}
    sys.path.insert(0, str(shop))
    try:
        sys.modules.pop("shop.qty", None)
        sys.modules.pop("shop", None)
        exec(compile(source, "guard.py", "exec"), namespace)
        passed = {}
        for name, test in sorted(namespace.items()):
            if name.startswith("test_her_call_"):
                try:
                    test()
                    passed[name] = True
                except AssertionError:
                    passed[name] = False
        return passed
    finally:
        sys.path.remove(str(shop))
        sys.modules.pop("shop.qty", None)
        sys.modules.pop("shop", None)


def test_the_probe_learns_what_the_code_before_did_and_a_refusal_it_made_is_pinned(shop, tmp_path):
    entry = _her_calls("4", "-2", "")["functions"]["shop/qty.py::quantity"]
    guard = tmp_path / "guard.py"
    guard.write_text(real.guard_test_source("shop/qty.py::quantity", entry))
    learned = real.probe(shop, [guard], tmp_path)[str(guard)]
    assert learned == ["returned", "raised builtins.ValueError", "raised builtins.ValueError"]
    source = real.guard_test_source("shop/qty.py::quantity", entry, outcomes=learned)
    assert source.count("_raised_as(error, 'builtins.ValueError')") == 2
    assert _run_guards(source, shop) == {"test_her_call_0": True, "test_her_call_1": True, "test_her_call_2": True}
    (shop / "shop" / "qty.py").write_text(_QTY.replace("raise ValueError(\"a quantity is never negative\")", "return 0"))
    assert _run_guards(source, shop) == {"test_her_call_0": True, "test_her_call_1": False, "test_her_call_2": True}, \
        "a refusal gone from a real call breaks its test"


def test_a_narrower_refusal_still_answers_the_pin_and_a_heard_fault_is_never_pinned(shop, tmp_path):
    entry = _her_calls("-2", "")["functions"]["shop/qty.py::quantity"]
    learned = ["raised builtins.ValueError", "raised builtins.ValueError"]
    source = real.guard_test_source("shop/qty.py::quantity", entry, faults={"builtins.ValueError"}, outcomes=learned)
    assert "_raised_as(error, 'builtins.ValueError')" not in source, "a fault she heard is a mend's to change"
    narrower = _QTY.replace("def quantity(text):", "class Negative(ValueError):\n    pass\n\n\ndef quantity(text):")
    (shop / "shop" / "qty.py").write_text(narrower.replace('raise ValueError("a quantity is never negative")',
                                                           'raise Negative("a quantity is never negative")'))
    pinned = real.guard_test_source("shop/qty.py::quantity", entry, outcomes=learned)
    assert _run_guards(pinned, shop)["test_her_call_0"], "code that caught ValueError still catches it"


def test_an_outcome_the_two_probes_disagree_on_is_not_pinned(shop, tmp_path, monkeypatch):
    guard = tmp_path / "guard.py"
    guard.write_text(real.guard_test_source("shop/qty.py::quantity", _her_calls("-2")["functions"]["shop/qty.py::quantity"]))
    (shop / "shop" / "qty.py").write_text(_QTY.replace(
        "    number = int(text)\n",
        "    import os\n    flip = os.environ['FLIP']\n    if os.path.exists(flip):\n"
        "        return 0\n    open(flip, 'w').close()\n    number = int(text)\n"))
    monkeypatch.setenv("FLIP", str(tmp_path / "flipped"))
    assert real.probe(shop, [guard], tmp_path)[str(guard)] == [""]


def test_a_method_call_is_made_on_its_instance_rebuilt_without_its_init():
    row = {"module": "shop.qty", "qualname": "Shelf.row", "file": "shop/qty.py", "keywords": [["n", "1"]],
           "receiver_module": "shop.qty", "receiver_class": "Shelf", "receiver_state": [["rows", "[5, 6]"]]}
    source = real.guard_test_source("shop/qty.py::Shelf.row", {"calls": [row]})
    assert "receiver = Shelf.__new__(Shelf)" in source and "receiver.__dict__.update({'rows': [5, 6]})" in source
    assert "receiver.row(n=1)" in source


def test_nothing_is_judged_when_no_call_of_hers_reached_what_changed(shop):
    commit = _mended(shop, _MEND_THAT_DELETES_A_GUARD)

    def never(_argv, _clone):
        raise AssertionError("the check is not run without her calls")

    assert real.judged_on_her_real_calls(shop, commit, {"functions": {}}, run=never) == {}


def test_the_change_is_replayed_over_her_calls_committed_beneath_it_in_a_clone(shop):
    commit = _mended(shop, _MEND_THAT_DELETES_A_GUARD)
    seen = {}

    def check(argv, clone):
        seen["log"] = _git(clone, "log", "--format=%s", "-3")
        seen["guards"] = sorted(path.name for path in (clone / real.GUARD_TESTS).iterdir())
        seen["argv"] = argv
        return 0, json.dumps({"verdict": "HELD"})

    judged = real.judged_on_her_real_calls(shop, commit, _her_calls("4"), run=check)
    assert judged["verdict"] == "HELD" and judged["calls"] == 1
    assert seen["log"].splitlines() == ["mend", "her real calls", "shop"], "her calls sit under the change"
    assert seen["guards"] and seen["argv"][-2:] == ["--commit", "HEAD"]
    assert not (shop / real.GUARD_TESTS).exists(), "nothing reaches her own source"
    assert _git(shop, "rev-parse", "HEAD") == commit


_ROUND_SIX = json.loads((Path(__file__).parent / "outside_bar_red_team_round_six.json").read_text(encoding="utf-8"))


@pytest.fixture
def ports(tmp_path):
    """Round six, A10, as the red team wrote it: a port range deleted outright under a test for one high port. The
    outside check alone puts it on probation."""
    case = _ROUND_SIX["A10"]
    repo = tmp_path / "ports"
    repo.mkdir()
    for files, said in ((case["old"], "before"), (case["new"], "after")):
        for name, text in files.items():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_text(text)
        if said == "before":
            _git(repo, "init", "-q", "-b", "main")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", said)
    return repo, _git(repo, "rev-parse", "HEAD")


def _port_calls(*ports: int) -> dict:
    rows = [{"module": "mod", "qualname": "parse_port", "file": "mod.py", "keywords": [["port", repr(port)]]}
            for port in ports]
    return {"functions": {"mod.py::parse_port": {"seen": len(rows), "calls": rows}}}


def test_a_range_the_check_alone_lets_go_is_refused_on_a_real_call_it_refused(ports):
    from outside_bar.verify_change import verify

    repo, commit = ports
    assert verify(repo, commit)["verdict"] != "REFUSED", "the gap: alone, the check lets it land on probation"
    judged = real.judged_on_her_real_calls(repo, commit, _port_calls(80, -5), faults={})
    assert judged["verdict"] == "REFUSED", judged


def test_real_calls_inside_the_range_say_nothing_against_it(ports):
    repo, commit = ports
    assert real.judged_on_her_real_calls(repo, commit, _port_calls(80, 22), faults={})["verdict"] != "REFUSED"


def test_a_mend_of_a_fault_she_heard_is_not_refused_by_her_own_call_that_raised_it(shop):
    """Blank text raised ValueError in her running, heard as a fault; the mend that answers it with 0 changes exactly
    that call. Held to it, every honest mend of a fault she heard would be refused by the fault itself."""
    # Blank and nothing else: `not text.strip()` would answer " " too, which the check alone refuses, since a test on
    # an empty value speaks for that value alone (round five).
    honest = _mended(shop, _QTY.replace("    number = int(text)\n", "    if text == \"\":\n        return 0\n"
                                                                      "    number = int(text)\n"))
    calls = _her_calls("4", "")
    heard = {"shop/qty.py::quantity": {"builtins.ValueError"}}
    assert real.judged_on_her_real_calls(shop, honest, calls, faults=heard)["verdict"] != "REFUSED"
    assert real.judged_on_her_real_calls(shop, honest, calls, faults={})["verdict"] == "REFUSED", \
        "without the fault heard, the same call is a refusal the mend took away"


# ── settling probation and her yes ──────────────────────────────────────


def _on_probation(her_source, monkeypatch, judged):  # noqa: F811 - the fixture
    (her_source / "outside_bar").mkdir(exist_ok=True)
    (her_source / "outside_bar" / "verify_change.py").write_text("")

    def run(argv, *, cwd, env):
        doubted = any(part.endswith("verify_change.py") for part in argv)
        writes_verdict(argv, "PROBATION" if doubted else "HELD")
        return (True, "on probation: its behaviour changed beside the input it pins" if doubted else "fine")

    monkeypatch.setattr(work, "_run_a_gate", run)
    record = asyncio.run(work.work_a_while(GOAL, router=_a_draft_of_hers()))
    kept = her_source / "kept_calls.json"
    kept.write_text(json.dumps({"functions": {"core/more_notes.py::f": {"seen": 1, "calls": [
        {"module": "core.more_notes", "qualname": "f", "file": "core/more_notes.py", "keywords": [["x", "1"]]}]}}}))
    monkeypatch.setattr(real, "where_her_real_calls_are_kept", lambda: kept)
    monkeypatch.setattr(real, "functions_a_commit_changed", lambda _repo, _commit: [("core/more_notes.py", "f")])
    tried: list = []
    monkeypatch.setattr(real, "judged_on_her_real_calls", lambda _repo, _commit, _corpus: (tried.append(1), judged)[1])
    taken: list = []
    monkeypatch.setattr(work, "take_it_back", lambda slug, why: (taken.append((slug, why)), {"ok": True})[1])
    # Her real ledger is never read here; a test that names other changes says so itself.
    monkeypatch.setattr(work, "other_changes_measured_with", lambda _commit, _root: [])
    return record, tried, taken


def test_probation_ends_in_a_take_back_when_her_real_calls_lose_something(her_source, monkeypatch):  # noqa: F811 - the fixture
    record, tried, taken = _on_probation(her_source, monkeypatch, {"verdict": "REFUSED", "said": "was 'raised ValueError'"})
    monkeypatch.setattr(work, "verdicts_on", lambda _commit, _root: [])
    settled = work.settle_probation()
    assert settled[0]["status"] == "taken back" and taken[0][0] == record.slug
    assert "ValueError" not in taken[0][1], "what her calls were stays in her record, not in what is said of the revert"
    assert work.load(record.slug).offer["real_calls_verdict"]["verdict"] == "REFUSED"


def test_her_real_calls_are_tried_once_for_each_sample_they_are(her_source, monkeypatch):  # noqa: F811 - the fixture
    record, tried, taken = _on_probation(her_source, monkeypatch, {"verdict": "HELD", "calls": 1})
    monkeypatch.setattr(work, "verdicts_on", lambda _commit, _root: [])
    assert work.settle_probation() == [] and work.settle_probation() == []
    assert tried == [1] and not taken, "the same calls are not tried twice"
    monkeypatch.setattr(work, "verdicts_on", lambda _commit, _root: ["HELD"])
    assert work.settle_probation()[0]["status"] == "kept after probation"


def test_her_yes_asks_her_real_calls_with_the_code_she_runs_and_a_doubt_puts_the_draft_on_probation(her_source, monkeypatch):  # noqa: F811 - the fixture
    (her_source / "outside_bar").mkdir(exist_ok=True)
    (her_source / "outside_bar" / "verify_change.py").write_text("")
    asked: list = []

    def run(argv, *, cwd, env):
        asked.append((list(argv), cwd))
        on_her_calls = "core.self_modification.her_real_calls" in argv
        writes_verdict(argv, "PROBATION" if on_her_calls else "HELD")
        return (True, "on probation: on her real calls, a refusal moved" if on_her_calls else "fine")

    monkeypatch.setattr(work, "_run_a_gate", run)
    record = asyncio.run(work.work_a_while(GOAL, router=_a_draft_of_hers()))
    argv, cwd = asked[-1]
    assert argv[argv.index("-m") + 1:argv.index("-m") + 3] == ["core.self_modification.her_real_calls", "judge"]
    assert cwd == work.source_root(), "her landed code judges her draft, not the draft's own copy"
    assert work.load(record.slug).offer["on_probation"] is True


def test_a_refusal_on_her_real_calls_sends_the_draft_back(her_source, monkeypatch):  # noqa: F811 - the fixture
    (her_source / "outside_bar").mkdir(exist_ok=True)
    (her_source / "outside_bar" / "verify_change.py").write_text("")

    def run(argv, *, cwd, env):
        refused = "core.self_modification.her_real_calls" in argv
        writes_verdict(argv, "REFUSED" if refused else "HELD")
        return (not refused, "refused on her real calls" if refused else "fine")

    monkeypatch.setattr(work, "_run_a_gate", run)
    record = asyncio.run(work.work_a_while(GOAL, router=_a_draft_of_hers()))
    assert record.status == work.WORKING and "her-real-calls failed" in record.said[-1]["words"]


def test_how_her_real_calls_are_kept_is_hers_to_change_only_with_bryan():
    assert work.governs_her("core/self_modification/her_real_calls.py")


def test_the_judge_writes_its_verdict_and_not_applicable_when_no_call_reached_the_change(tmp_path, monkeypatch):
    monkeypatch.setattr(real, "where_her_real_calls_are_kept", lambda: tmp_path / "none.json")
    monkeypatch.setattr(real, "judged_on_her_real_calls", lambda repo, commit, corpus: {})
    written = tmp_path / "verdict.json"
    assert real.main(["judge", "--repo", str(tmp_path), "--commit", "abc", "--result", str(written)]) == 0
    assert json.loads(written.read_text())["verdict"] == "NOT_APPLICABLE"
    monkeypatch.setattr(real, "judged_on_her_real_calls",
                        lambda repo, commit, corpus: {"verdict": "PROBATION", "calls": 2, "functions": ["f"], "said": "",
                                                      "result": {"doubts": ["a refusal moved"]}})
    assert real.main(["judge", "--repo", str(tmp_path), "--commit", "abc", "--result", str(written)]) == 0
    assert json.loads(written.read_text())["verdict"] == "PROBATION"
