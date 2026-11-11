"""Her ear's log and hold, her record of a watching, and her builds' pages and records go through the file gateway,
each inside a governed scope (Master Todo Q06).

Before, each was a raw ``open``, ``touch``, ``mkdir`` or ``write_text`` the effect inventory counted against the raw
ceiling, and none of them carried a receipt.
"""
from __future__ import annotations

import asyncio
import json
import stat
from types import SimpleNamespace

import pytest

from core.governance_context import get_active_governance


class _Gateway:
    """Does what the gateway does, and notes each call with whether a governed scope was open."""

    def __init__(self):
        self.calls: list[tuple[str, str, bool]] = []

    def _note(self, how, path):
        self.calls.append((how, str(path), get_active_governance() is not None))

    def open_owned_binary(self, path, *, mode, source="unknown"):
        self._note("open_owned_binary", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return open(path, mode)

    def write_bytes_if_absent(self, path, payload, *, source="unknown", durable=True):
        self._note("write_bytes_if_absent", path)
        open(path, "wb").close()
        return True

    def delete_file(self, path, *, source="unknown"):
        self._note("delete_file", path)
        try:
            __import__("os").unlink(path)
            return True
        except FileNotFoundError:
            return False

    def ensure_directory(self, path, *, source="unknown"):
        self._note("ensure_directory", path)
        __import__("os").makedirs(path, exist_ok=True)
        return str(path)

    def write_text(self, path, text, *, source="unknown"):
        self._note("write_text", path)
        with open(path, "w", encoding="utf-8") as out:
            out.write(text)

    def copy_path(self, path, destination, *, source="unknown"):
        self._note("copy_path", path)
        import shutil

        return shutil.copyfile(path, destination)


@pytest.fixture
def gateway(monkeypatch):
    import core.runtime.file_write_gateway as module

    fake = _Gateway()
    monkeypatch.setattr(module, "get_file_write_gateway", lambda: fake)

    def append_text_behind(path, text, *, source="unknown"):
        fake._note("append_text_behind", path)
        with open(path, "a", encoding="utf-8") as out:
            out.write(text)
        return True

    monkeypatch.setattr(module, "append_text_behind", append_text_behind)
    return fake


def test_her_ear_sets_and_lifts_its_hold_in_a_governed_scope(tmp_path, gateway):
    from core.senses.her_ear import HerEar

    ear = HerEar("com.apple.Safari")
    ear.hold = str(tmp_path / "hold")
    ear.yields(True)
    assert (tmp_path / "hold").exists()
    asyncio.run(ear.yields_async(False))
    assert not (tmp_path / "hold").exists()
    assert [(how, governed) for how, _path, governed in gateway.calls] == [
        ("write_bytes_if_absent", True), ("delete_file", True)]


def test_her_ear_opens_its_own_log_in_a_governed_scope(tmp_path, gateway, monkeypatch):
    import core.config as config_module
    from core.senses import her_ear

    monkeypatch.setattr(config_module, "config", SimpleNamespace(paths=SimpleNamespace(data_dir=tmp_path / "data")))
    log = her_ear._ear_log()
    try:
        assert log is not None and log.fileno() >= 0
    finally:
        log.close()
    assert gateway.calls == [("open_owned_binary", str(tmp_path / "logs" / "her_ear.log"), True)]


def test_her_record_of_a_watching_is_kept_row_by_row_in_a_governed_scope(tmp_path, gateway, monkeypatch):
    import core.config as config_module
    from core.skills.watching_along import kept_at

    monkeypatch.setattr(config_module, "config", SimpleNamespace(paths=SimpleNamespace(data_dir=tmp_path)))
    keep = kept_at("A film", "https://example.org/film")
    keep({"at": 1.0, "saw": "a garage"})
    keep({"at": 2.0, "said": "he is welding"})
    (kept,) = list((tmp_path / "watched").glob("*.jsonl"))
    rows = [json.loads(line) for line in kept.read_text(encoding="utf-8").splitlines()]
    assert [r.get("saw") or r.get("said") for r in rows] == ["a garage", "he is welding"]
    assert all(r["title"] == "A film" for r in rows)
    assert [how for how, _p, _g in gateway.calls] == ["ensure_directory", "append_text_behind", "append_text_behind"]
    assert all(governed for _h, _p, governed in gateway.calls)


def test_her_build_writes_its_page_and_record_in_a_governed_scope(tmp_path, gateway):
    from core.rebuilding.the_program_as_built import ProgramAsBuilt

    program = ProgramAsBuilt("A counter")
    page = program.write(tmp_path / "new" / "index.html")
    kept = program.keep(tmp_path / "new")
    assert page.read_text("utf-8") == program.page()
    assert json.loads(kept.read_text("utf-8"))["title"] == "A counter"
    assert [how for how, _p, _g in gateway.calls] == ["ensure_directory", "write_text", "write_text"]
    assert all(governed for _h, _p, governed in gateway.calls)


def test_a_folder_that_is_there_is_left_as_it_is(tmp_path, gateway):
    """The gateway makes a folder private. A page written into a person's own folder must not make it so."""
    from core.rebuilding.the_program_as_built import ProgramAsBuilt

    theirs = tmp_path / "theirs"
    theirs.mkdir(mode=0o755)
    theirs.chmod(0o755)
    ProgramAsBuilt("A counter").write(theirs / "index.html")
    assert stat.S_IMODE(theirs.stat().st_mode) == 0o755
    assert [how for how, _p, _g in gateway.calls] == ["write_text"]


def test_a_write_by_a_name_the_helper_does_not_know_is_refused(tmp_path):
    from core.rebuilding.her_build_writes import governed

    with pytest.raises(ValueError, match="no governed write named"):
        governed("chmod", tmp_path / "x")
