"""A gate must not demand an edit another gate forbids.

LIVE, 2026-09-21. The swallowed-reasons sweep added four comment lines to
``core/brain/llm/qualified_recurrent_ingress.py``. No statement changed. The
bounded-WOW surface seals that file by SHA-256 of its BYTES, because a
120-task qualification measured what those exact files do, so the comments
switched a surface established with lesion controls off, and
``tests/brain/test_bounded_wow_surface_live.py`` went red.

The same shape is recorded in that file's own comment for 2026-08-17, when
four bound files moved under ordinary development and the signal was dark
for two days with nobody knowing.

The seal is right: it is the claim that the thing measured is the thing
serving. So the other gate stops asking about sealed files.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sealed() -> set[str]:
    from core.brain.llm.semantic_neural_serving import ACTIVATION_SOURCE_FILES

    return {one for one in ACTIVATION_SOURCE_FILES if one.endswith(".py")}


def test_the_reasons_gate_skips_what_the_qualification_sealed():
    from tools.lint_swallowed_reasons import sealed_by_a_qualification

    skipped = {
        str(path.relative_to(REPO_ROOT)) for path in sealed_by_a_qualification()
    }
    assert skipped == _sealed(), (
        "the reasons gate must skip exactly the Python files the activation "
        "record seals"
    )


def test_a_sealed_file_is_not_counted():
    """The count must actually drop, or the skip is decorative."""
    from tools.lint_swallowed_reasons import look

    sealed = [REPO_ROOT / one for one in _sealed()]
    assert look(sealed), (
        "these sealed files do carry handlers that lose their reason — if "
        "they no longer do, this test has nothing left to protect and the "
        "skip can go"
    )


def test_the_gate_still_runs_and_still_passes():
    result = subprocess.run(
        [sys.executable, "tools/lint_swallowed_reasons.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "sealed by a qualification" in result.stdout


def test_the_sealed_files_are_byte_identical_to_what_was_measured():
    """The whole reason for the skip: these bytes are load-bearing."""
    import hashlib
    import json

    from core.brain.llm.semantic_neural_serving import (
        active_semantic_neural_activation_path,
    )

    record = json.loads(active_semantic_neural_activation_path().read_text())
    recorded = record["source_sha256s"]
    for relative in sorted(_sealed()):
        here = (REPO_ROOT / relative).read_bytes()
        assert hashlib.sha256(here).hexdigest() == recorded[relative], (
            f"{relative} no longer matches the bytes the qualification "
            "measured; re-run qualification rather than re-sealing"
        )
