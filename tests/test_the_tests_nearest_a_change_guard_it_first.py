"""The guard of a repair, grouped by the file each test imports, the file fewest tests import first.

LIVE 16 November 2026: a702a104eb27 changed a module two test files import and one 123 import. Run as one, the
guard took 4 min 45 s for every candidate; run nearest first, a candidate that breaks a near test is refused in
seconds (tools/repair_by_selection.py).
"""
from __future__ import annotations

from tools.repair_by_selection import guarding_groups


def test_the_file_fewest_tests_import_comes_first(tmp_path):
    (tmp_path / "tests").mkdir()
    for n in range(3):
        (tmp_path / "tests" / f"test_wide_{n}.py").write_text("from core.claims import listed\n")
    (tmp_path / "tests" / "test_near.py").write_text("from core.timing import seconds\nfrom core.claims import x\n")
    groups = guarding_groups(["core/claims.py", "core/timing.py"], tmp_path)
    assert groups[0] == ["tests/test_near.py"]
    assert sorted(groups[1]) == ["tests/test_near.py", *(f"tests/test_wide_{n}.py" for n in range(3))]
