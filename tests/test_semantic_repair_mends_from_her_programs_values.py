"""Semantic repair mends faults from what the program did: values recorded, outcomes forced, a condition synthesised.

Each fault is planted in a small package of its own, and its tests are run by a real pytest process, as her repair
runs them (tools/repair_by_selection.py). A repair counts only when the tests that failed pass on the repaired file
and the tests that passed still do, and, for a fault next to a refusal, when the refusal is still there and still
refuses (Master Todo F16; a repair proof must keep the refusal).
"""
from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from core.self_modification.repair_by_selection import applied, edits_in
from core.self_modification.semantic_repair import Evidence, semantic_repairs

ROOT = Path(__file__).resolve().parents[1]
_RESULT = re.compile(r"^(PASSED|FAILED|ERROR)\s+(\S+)", re.MULTILINE)


class Planted:
    """A package with one module and its tests, run by pytest in a process of its own."""

    def __init__(self, where: Path, module: str, tests: str) -> None:
        self.where = where
        (where / "planted").mkdir()
        (where / "planted" / "__init__.py").write_text("")
        (where / "planted" / "mod.py").write_text(module)
        (where / "test_planted.py").write_text(tests)
        (where / "pytest.ini").write_text("[pytest]\n")
        self.path = "planted/mod.py"
        self.source = module

    async def run(self, tests: Sequence[str], sources: Mapping[str, str]) -> tuple[dict[str, bool], str]:
        for path, text in sources.items():
            (self.where / path).write_text(text)
        env = {k: v for k, v in os.environ.items() if k != "AURA_TESTING"}
        env.update(PYTHONPATH=f"{self.where}{os.pathsep}{ROOT}", AURA_LOG_DIR=str(self.where / "logs"),
                   AURA_STATE_ROOT=str(self.where / "state"), PYTHONDONTWRITEBYTECODE="1")
        try:
            done = await asyncio.to_thread(
                subprocess.run, [sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider",
                                 "--rootdir", str(self.where), *tests],
                cwd=self.where, env=env, capture_output=True, text=True, timeout=120)
        finally:
            (self.where / self.path).write_text(self.source)
        # In a folder of its own pytest names a test by a path relative to wherever it settles on; the test's own
        # name is what is matched.
        seen = {node.split("::")[-1]: word == "PASSED" for word, node in _RESULT.findall(done.stdout)}
        return {t: seen.get(t.split("::")[-1], False) for t in tests}, done.stdout[-3000:]

    def lines(self) -> list[tuple[tuple[str, int], float]]:
        return [((self.path, n), 1.0) for n in range(1, len(self.source.splitlines()) + 1)]


async def _first_mend(planted: Planted, failing: Sequence[str], passing: Sequence[str],
                      tmp: Path) -> tuple[str, Evidence]:
    evidence = Evidence()
    async for _origin, text in semantic_repairs(
            sources={planted.path: planted.source}, failing=failing, passing=passing, ranked=planted.lines(),
            run=planted.run, workdir=tmp / "probe", deadline=time.time() + 240, evidence=evidence):
        after, refused = applied({planted.path: planted.source}, edits_in(text, [planted.path]))
        if refused:
            continue
        results, _ = await planted.run([*failing, *passing], after)
        if all(results.values()):
            return after[planted.path], evidence
    return "", evidence


def _node(name: str) -> str:
    return f"test_planted.py::{name}"


@pytest.fixture
def plant(tmp_path):
    def make(module: str, tests: str) -> Planted:
        return Planted(tmp_path, module, tests)
    return make


def test_a_wrong_boundary_is_moved(plant, tmp_path):
    planted = plant(
        "def counted(items, limit):\n"
        "    count = 0\n"
        "    for item in items:\n"
        "        if item < limit:\n"
        "            count += 1\n"
        "    return count\n",
        "from planted.mod import counted\n\n"
        "def test_all_under():\n    assert counted([1, 2, 3], 10) == 3\n\n"
        "def test_none_under():\n    assert counted([5, 6], 1) == 0\n\n"
        "def test_the_limit_itself_counts():\n    assert counted([1, 2, 3], 2) == 2\n")
    mended, evidence = asyncio.run(_first_mend(
        planted, [_node("test_the_limit_itself_counts")], [_node("test_all_under"), _node("test_none_under")],
        tmp_path))
    assert "item <= limit" in mended, evidence.to_json()


def test_and_where_or_was_meant_is_turned(plant, tmp_path):
    planted = plant(
        "def admits(age, member):\n"
        "    if age >= 18 and member:\n"
        "        return True\n"
        "    return False\n",
        "from planted.mod import admits\n\n"
        "def test_an_adult_member():\n    assert admits(30, True)\n\n"
        "def test_a_child_who_is_not():\n    assert not admits(10, False)\n\n"
        "def test_a_member_of_any_age():\n    assert admits(10, True)\n\n"
        "def test_an_adult_of_any_kind():\n    assert admits(30, False)\n")
    mended, evidence = asyncio.run(_first_mend(
        planted, [_node("test_a_member_of_any_age"), _node("test_an_adult_of_any_kind")],
        [_node("test_an_adult_member"), _node("test_a_child_who_is_not")], tmp_path))
    assert "age >= 18 or member" in mended, evidence.to_json()


def test_a_missing_guard_returns_before_the_fault(plant, tmp_path):
    planted = plant(
        "def first_word(text):\n"
        "    if text is None:\n"
        "        return ''\n"
        "    words = text.split()\n"
        "    return words[0]\n",
        "from planted.mod import first_word\n\n"
        "def test_a_sentence():\n    assert first_word('hello there') == 'hello'\n\n"
        "def test_nothing_at_all():\n    assert first_word(None) == ''\n\n"
        "def test_only_spaces():\n    assert first_word('   ') == ''\n")
    mended, evidence = asyncio.run(_first_mend(
        planted, [_node("test_only_spaces")], [_node("test_a_sentence"), _node("test_nothing_at_all")], tmp_path))
    assert "    words = text.split()\n    if not words:\n        return ''\n    return words[0]" in mended, \
        evidence.to_json()


def test_a_condition_missing_a_conjunct_gains_it(plant, tmp_path):
    planted = plant(
        "def price_after(price, member):\n"
        "    if member:\n"
        "        return price * 0.9\n"
        "    return price\n",
        "from planted.mod import price_after\n\n"
        "def test_a_member_buying_a_lot():\n    assert price_after(200, True) == 180\n\n"
        "def test_not_a_member():\n    assert price_after(50, False) == 50\n\n"
        "def test_a_member_buying_a_little():\n    assert price_after(50, True) == 50\n")
    mended, evidence = asyncio.run(_first_mend(
        planted, [_node("test_a_member_buying_a_little")],
        [_node("test_a_member_buying_a_lot"), _node("test_not_a_member")], tmp_path))
    condition = re.search(r"    if (.+):\n        return price \* 0.9", mended)
    assert condition and "member" in condition.group(1) and "price" in condition.group(1), evidence.to_json()


def test_a_refusal_with_the_wrong_bound_still_refuses(plant, tmp_path):
    planted = plant(
        "def withdraw(balance, amount):\n"
        "    if amount > balance + 1:\n"
        "        raise ValueError('more than the balance')\n"
        "    return balance - amount\n",
        "import pytest\nfrom planted.mod import withdraw\n\n"
        "def test_some():\n    assert withdraw(10, 5) == 5\n\n"
        "def test_far_too_much():\n    with pytest.raises(ValueError):\n        withdraw(10, 20)\n\n"
        "def test_one_more_than_there_is():\n    with pytest.raises(ValueError):\n        withdraw(10, 11)\n")
    mended, evidence = asyncio.run(_first_mend(
        planted, [_node("test_one_more_than_there_is")], [_node("test_some"), _node("test_far_too_much")],
        tmp_path))
    assert "if amount > balance:" in mended, evidence.to_json()
    assert "raise ValueError('more than the balance')" in mended
    namespace: dict = {}
    exec(compile(mended, "mended", "exec"), namespace)  # noqa: S102 - the repaired planted function, run directly
    with pytest.raises(ValueError):
        namespace["withdraw"](10, 11)
    with pytest.raises(ValueError):
        namespace["withdraw"](10, 20)
    assert namespace["withdraw"](10, 10) == 0
