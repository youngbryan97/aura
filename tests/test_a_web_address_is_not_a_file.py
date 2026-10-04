"""A web address is read as a web address by every reader of file names.

LIVE 2026-10-03: a Wikipedia link was read as the path "//en.wikipedia.org"
by the file reader and as the file "en.wikipedia" by capability selection, so
a turn that had already read the page was handed a file tool.
"""

from __future__ import annotations

from core.intent.declared_capability import requested_foundational_domains
from core.intent.opaque_spans import without_web_addresses

LINK = "https://en.wikipedia.org/wiki/Bam_Adebayo%27s_83-point_game"


def test_the_shared_reader_removes_web_addresses_only() -> None:
    assert without_web_addresses(f"see {LINK} and notes.txt").split() == ["see", "and", "notes.txt"]
    assert "www." not in without_web_addresses("go to www.example.com/a.html")
    assert without_web_addresses("open /tmp/report.md") == "open /tmp/report.md"


def test_a_link_asks_for_the_web_and_not_the_disk() -> None:
    assert requested_foundational_domains(LINK) == ("web",)
    assert set(requested_foundational_domains(f"compare notes.txt with {LINK}")) == {"web", "file"}


def test_repo_readers_do_not_take_a_linked_file_for_a_repo_file() -> None:
    from core.brain.verifiers.repo_engine import _iter_referenced_paths

    assert _iter_referenced_paths("https://github.com/x/y/blob/main/README.md") == []
    assert _iter_referenced_paths("look at core/config.py") == ["core/config.py"]
