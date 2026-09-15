"""One occurrence of a thought is one card, however many paths carried it.

The runtime publishes one health pulse on two paths — the thought stream as
SYS and the logging mirror as Aura.Core.Orchestrator — and the feed keys
cards on the sentence a reader sees so the two meet. LIVE 2026-09-15 they did
not: the face runs both rule tables and the key ran one. The harness loads
the shipped script under a stubbed window and asks the shipped function.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "tests" / "js" / "one_thought_one_card.mjs"
AURA_JS = REPO / "interface" / "static" / "aura.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_both_shapes_of_the_health_pulse_key_to_one_card():
    result = subprocess.run(
        [shutil.which("node"), str(HARNESS), str(AURA_JS)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr[-1500:]}"
    assert "all checks passed" in result.stdout


def test_the_key_is_derived_through_the_same_tables_as_the_face():
    source = AURA_JS.read_text(encoding="utf-8")
    key = source[source.index("function buildThoughtFingerprint") :]
    key = key[: key.index("\n}")]
    assert "plainLanguageThought(toPlainEnglish(" in key
