"""A page she built becomes a Mac application: a bundle with its program, its name, an icon, signed for this machine."""
from __future__ import annotations

import plistlib
import shutil
import subprocess

import pytest

from core.rebuilding.as_a_mac_app import as_a_mac_app
from core.rebuilding.the_program_as_built import Part, ProgramAsBuilt

pytestmark = pytest.mark.skipif(shutil.which("swiftc") is None, reason="no Swift compiler on this machine")


def test_a_page_becomes_an_application(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path / "state"))
    page = ProgramAsBuilt("Quill", accent="#2b579a", parts=[Part("work area", 'app.work.append(app.make("div", {contenteditable: "true"}));')]).write(tmp_path / "quill" / "index.html")
    app = as_a_mac_app(page, "Quill", accent="#2b579a", where=tmp_path)
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert info["CFBundleName"] == "Quill" and info["CFBundleIdentifier"] == "com.aura.built.quill"
    assert (app / "Contents" / "Resources" / "index.html").read_text() == page.read_text()
    assert (app / "Contents" / "MacOS" / info["CFBundleExecutable"]).stat().st_mode & 0o111
    assert subprocess.run(["codesign", "--verify", str(app)], capture_output=True).returncode == 0


def test_someone_else_s_application_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path / "state"))
    theirs = tmp_path / "Quill.app" / "Contents"
    theirs.mkdir(parents=True)
    (theirs / "Info.plist").write_bytes(plistlib.dumps({"CFBundleIdentifier": "com.someone.quill"}))
    page = ProgramAsBuilt("Quill").write(tmp_path / "quill" / "index.html")
    app = as_a_mac_app(page, "Quill", where=tmp_path)
    assert app.name == "Quill (Aura).app"
    assert plistlib.loads((theirs / "Info.plist").read_bytes())["CFBundleIdentifier"] == "com.someone.quill"
