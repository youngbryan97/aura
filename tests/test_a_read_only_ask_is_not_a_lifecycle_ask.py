"""A caller that wants a service must not ask through the seam that never builds one.

`get_runtime_service` resolves with `ServiceContainer.peek`, which deliberately
never invokes a factory: the seam is for diagnostics and error sinks, which must
be able to look at a runtime without booting an organ while they look.

For a caller that wants the thing, it is the wrong seam. A service registered as
a lazy factory has no instance until somebody builds it, so the ask returns None
for the life of the process — silently, with the subsystem present, registered
and complete.

That is how `core/consciousness/drive_integration.py` came to be dead: a full
drive arbitration — a leaky integrator per drive, mutual inhibition and a
Schmitt trigger, written because the mechanism before it "fired the instant a
point crossed a line" — registered as a service and reached for by nothing. The
one caller added for it got None until it asked the container instead.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ASK = re.compile(r"""get_runtime_service\(\s*["']([A-Za-z0-9_]+)["']""")


def test_the_motivation_phase_asks_the_container_for_the_drive_engine():
    """The one this was found by. It must not go back to the read-only seam."""
    source = (ROOT / "core" / "phases" / "motivation_update.py").read_text()
    block = source[source.index("_integrate_drives") : source.index("_substrate_dominance")]
    assert 'ServiceContainer.get("drive_integration"' in block
    assert 'get_runtime_service("drive_integration"' not in block


def test_the_audit_exists_and_names_the_seam():
    tool = ROOT / "tools" / "audit_read_only_service_asks.py"
    assert tool.exists()
    text = tool.read_text()
    assert "peek" in text and "get_runtime_service" in text


def test_the_read_only_seam_still_refuses_to_build():
    """The rule this test protects is the reason the seam exists.

    A diagnostic that boots an organ while looking at one is worse than a
    diagnostic that reports nothing, so `peek` must stay a read.
    """
    source = (ROOT / "core" / "container.py").read_text()
    resolver = source[source.index("def _service_for") : source.index("def _has_service")]
    assert "peek" in resolver, "the read-only resolver now builds services"
    assert ".get(" not in resolver.replace("ServiceContainer.peek(", "")
