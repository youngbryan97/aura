"""Outside models' attacks on the change verifier, kept as tests, with the verdicts each may get.

Round two, 18 November 2026: a model attacked commit 6032f2e4d with 33 patches and 28 got the wrong verdict. Round three,
the same day: the same model attacked the rebuilt check at 7aa574989 with 35 patches and 26 did. A bad patch must never be
HELD. An honest fix must never be REFUSED, unless the case says why. PROBATION is allowed for both, because behaviour alone
cannot always tell an honest correction from damage. Rounds four to six came from a second model, Argon, and round seven
from a third, round eight from another. The cases are in outside_bar_red_team_round_two.json through _round_eight.json.
"""
from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import pytest

from outside_bar.verify_change import verify

_HERE = Path(__file__).parent
_CASES = json.loads((_HERE / "outside_bar_red_team_round_two.json").read_text(encoding="utf-8"))
_ROUND_THREE = json.loads((_HERE / "outside_bar_red_team_round_three.json").read_text(encoding="utf-8"))
_ROUND_FOUR = json.loads((_HERE / "outside_bar_red_team_round_four.json").read_text(encoding="utf-8"))
_ROUND_FIVE = json.loads((_HERE / "outside_bar_red_team_round_five.json").read_text(encoding="utf-8"))
_ROUND_SIX = json.loads((_HERE / "outside_bar_red_team_round_six.json").read_text(encoding="utf-8"))
_ROUND_SEVEN = json.loads((_HERE / "outside_bar_red_team_round_seven.json").read_text(encoding="utf-8"))
_ROUND_EIGHT = json.loads((_HERE / "outside_bar_red_team_round_eight.json").read_text(encoding="utf-8"))


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.fsmonitor=false", *args], cwd=repo,
                          check=True, capture_output=True, text=True).stdout.strip()


def _write(repo: Path, files: dict) -> None:
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_bytes(base64.b64decode(text["b64"]) if isinstance(text, dict) else text.encode("utf-8"))


def _judged(repo: Path, old: dict, new: dict, deleted: tuple[str, ...] = ()) -> dict:
    _git(repo, "init", "-q", "-b", "main")
    _write(repo, old)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "before")
    for name in deleted:
        (repo / name).unlink()
    _write(repo, new)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "after")
    return verify(repo, _git(repo, "rev-parse", "HEAD"))


@pytest.mark.parametrize("name", sorted(_CASES))
def test_round_two_gets_a_verdict_it_may_get(name, tmp_path):
    case = _CASES[name]
    result = _judged(tmp_path, case["old"], case["new"])
    assert result["verdict"] in case["must_be"], (case["idea"], result)


@pytest.mark.parametrize("name", sorted(_ROUND_THREE))
def test_round_three_gets_a_verdict_it_may_get(name, tmp_path):
    case = _ROUND_THREE[name]
    result = _judged(tmp_path, case["old"], case["new"], tuple(case["deleted"]))
    assert result["verdict"] in case["must_be"], (case["idea"], result)


def test_an_honest_mend_is_not_refused_for_the_order_its_test_runs_in(tmp_path):
    """C6 as an honest commit: its new test sorts first, advances a counter, and does show the fault."""
    case = _CASES["C6"]
    new = {**case["new"], "tests/test_a_new.py": "from mod import next_id\n\n\ndef test_a():\n    assert next_id('') == 'anon-0'\n"}
    result = _judged(tmp_path, case["old"], new)
    assert result["verdict"] != "REFUSED" or not result["values_kept"]["changed"], result


def test_a_test_file_named_the_other_way_is_a_test_file(tmp_path):
    """C4: tests/mod_test.py crashed the check with a KeyError."""
    old = {"mod.py": "def half(x):\n    return x / 2\n", "tests/mod_test.py": "from mod import half\n\n\ndef test_half():\n    assert half(4) == 2\n"}
    new = {"mod.py": "def half(x):\n    return 0 if x == 0 else x / 2\n",
           "tests/mod_test.py": old["tests/mod_test.py"] + "\n\ndef test_zero():\n    assert half(0) == 0\n"}
    result = _judged(tmp_path, old, new)
    assert result["tests_held"]["tests"] == 1 and "verdict" in result, result


def test_a_commit_that_cannot_be_read_does_not_pass(tmp_path):
    from outside_bar.verify_change import main

    old = {"mod.py": "X = 1\n"}
    _git(tmp_path, "init", "-q", "-b", "main")
    _write(tmp_path, old)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "before")
    _write(tmp_path, {"mod.py": "def broken(:\n"})
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "after")
    assert main(["--repo", str(tmp_path), "--commit", _git(tmp_path, "rev-parse", "HEAD")]) == 1


def _staged(repo: Path, files: dict) -> None:
    for name, text in files.items():
        raw = base64.b64decode(text["b64"]) if isinstance(text, dict) else text.encode("utf-8")
        blob = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=repo, input=raw, capture_output=True,
                              check=True).stdout.decode().strip()
        _git(repo, "update-index", "--add", "--cacheinfo", f"100644,{blob},{name}")


def _built_from_trees(repo: Path, old: dict, new: dict) -> str:
    """The commit written straight from trees, as round four built its attacks: a name too long for a disk, or a commit
    with no parent, can only be made this way."""
    _git(repo, "init", "-q", "-b", "main")
    parent = ""
    if old:
        _staged(repo, old)
        parent = _git(repo, "commit-tree", _git(repo, "write-tree"), "-m", "before")
    _staged(repo, new)
    return _git(repo, "commit-tree", _git(repo, "write-tree"), *(["-p", parent] if parent else []), "-m", "after")


@pytest.mark.parametrize("name", sorted(_ROUND_FOUR))
def test_round_four_gets_a_verdict_it_may_get(name, tmp_path):
    from outside_bar.verify_change import main

    case = _ROUND_FOUR[name]
    commit = _built_from_trees(tmp_path, case["old"], case["new"])
    try:
        verdict = verify(tmp_path, commit)["verdict"]
    except Exception:  # noqa: BLE001 - a commit the check cannot read is judged by what main says of it
        verdict = {0: "HELD", 1: "REFUSED", 2: "UNJUDGED"}[main(["--repo", str(tmp_path), "--commit", commit])]
    assert verdict in case["must_be"], (case["idea"], verdict)


@pytest.mark.parametrize("name", sorted(_ROUND_FIVE))
def test_round_five_gets_a_verdict_it_may_get(name, tmp_path):
    """Round five, the same evening, by Argon on c8db2f0af: 35 patches, 31 got the wrong verdict. Where the check now
    refuses a bad patch, only a refusal is allowed, so it cannot drift back to landing on probation."""
    case = _ROUND_FIVE[name]
    result = _judged(tmp_path, case["old"], case["new"], tuple(case["deleted"]))
    assert result["verdict"] in case["must_be"], (case["idea"], result)


@pytest.mark.parametrize("name", sorted(_ROUND_SIX))
def test_round_six_gets_a_verdict_it_may_get(name, tmp_path):
    """Round six, the same night, by Argon on 611e6235a: 41 patches, 37 got the wrong verdict. Its trees replace the old
    ones, so a file the attack leaves out is deleted."""
    case = _ROUND_SIX[name]
    result = _judged(tmp_path, case["old"], case["new"], tuple(case["deleted"]))
    assert result["verdict"] in case["must_be"], (case["idea"], case.get("why_refused"), result)


@pytest.mark.parametrize("name", sorted(_ROUND_SEVEN))
def test_round_seven_gets_a_verdict_it_may_get(name, tmp_path):
    """Round seven, 22 November, by a third model on ae684a5fa: 41 cases, 37 got the wrong verdict, aimed mostly at what
    was built after round six (solved conditions, side effects, counterexamples, probation)."""
    case = _ROUND_SEVEN[name]
    result = _judged(tmp_path, case["old"], case["new"], tuple(case["deleted"]))
    assert result["verdict"] in case["must_be"], (case["idea"], case.get("why_refused"), result)


@pytest.mark.parametrize("name", sorted(_ROUND_EIGHT))
def test_round_eight_gets_a_verdict_it_may_get(name, tmp_path):
    """Round eight, 22 November, on be8e8527f: 42 cases, 37 got the wrong verdict, aimed at round seven's own defences
    (single-line branches, the stream capture, the solver's limits) and at the tests her repair search keeps."""
    case = _ROUND_EIGHT[name]
    result = _judged(tmp_path, case["old"], case["new"], tuple(case["deleted"]))
    assert result["verdict"] in case["must_be"], (case["idea"], case.get("why_refused"), result)
