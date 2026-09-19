"""A test process that becomes a Cocoa application still has no Dock icon.

A run split six ways put six Python rockets in the Dock (2026-09-18): something
under test makes each process an application, and macOS gives every regular
application an icon. conftest marks the process as an agent before anything
can register it.
"""

from __future__ import annotations

import sys

import pytest


@pytest.mark.skipif(sys.platform != "darwin", reason="the Dock is a macOS thing")
def test_an_application_made_by_a_test_is_an_agent():
    AppKit = pytest.importorskip("AppKit")
    app = AppKit.NSApplication.sharedApplication()
    assert app.activationPolicy() != AppKit.NSApplicationActivationPolicyRegular
