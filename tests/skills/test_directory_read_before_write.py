"""Asked what is in a directory, look — do not compose an answer.

LIVE, 2026-08-10, three runs of one request:

    "Count how many .py files are in
     /Users/bryan/.aura/live-source/core/introspection, then write that number
     and the file names into ~/Documents/aura_probe_count.txt."

Run 1: never reached a tool. "There are 3 .py files ... 1. introspection.py
2. self_assessment.py 3. system_monitoring.py. I have written the number and
file names into ~/Documents/aura_probe_count.txt." The directory holds 9, none
of those names exist, and no file was created.

Run 2, after routing was fixed: the plan aimed the WRITE at the source
directory and was refused by the artifact-root guard.

Run 3, after the destination fix: the file landed in the right place
containing "Number of .py files: 0 / (No files found)". A correct destination
holding a measurement nobody took.

The remaining gap was structural — there was no read action at all. Every
desktop action wrote, so a question about the contents of a real path could
only ever be answered by generation. list_directory is that action, bounded by
a readable-roots set that is deliberately wider than the writable one (her own
source tree, because source proprioception is a capability she is meant to
have) and narrower than the filesystem.

The count in the file is now the count that was read, via the step-reference
tokens that already existed for exactly this.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.skills.computer_use import ComputerUseSkill
from core.skills.desktop_task import DesktopTaskSkill, DesktopTaskStep

# Derived from the checkout rather than hard-coded to one machine's home.
# These paths are DATA — the absolute path a user typed, which the router
# and the skill parser must extract — so the specific user name was never
# part of any assertion, only part of why the enterprise gate's
# hardcoded-path rule fired on this file.
REPO_ROOT = Path(__file__).resolve().parents[2]
_INTROSPECTION = str(REPO_ROOT / "core" / "introspection")
_CLAUDE_MD = str(REPO_ROOT / "CLAUDE.md")

OBJECTIVE = (
    f"Count how many .py files are in {_INTROSPECTION}, "
    "then write that number and the file names into ~/Documents/aura_probe_count.txt."
)


def _skill() -> ComputerUseSkill:
    return ComputerUseSkill.__new__(ComputerUseSkill)


# ── The read action ────────────────────────────────────────────────────────

def test_a_real_directory_is_counted_correctly() -> None:
    """The measurement she could not take.

    Built inside ~/Documents rather than pytest's tmp_path, because tmp_path
    lives under /private/tmp and is correctly outside the readable roots —
    a fixture directory is not a reason to widen them.
    """
    import shutil

    scratch = Path.home() / "Documents" / ".aura_list_directory_probe"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        for name in ("a.py", "b.py", "notes.txt"):
            (scratch / name).write_text("x", encoding="utf-8")

        result = _skill()._list_directory(
            json.dumps({"path": str(scratch), "pattern": "*.py"})
        )

        assert result["ok"] is True, result.get("error")
        assert result["count"] == 2
        assert result["names"] == ["a.py", "b.py"]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_a_directory_outside_the_readable_roots_is_refused(tmp_path: Path) -> None:
    """pytest's tmp_path is under /private/tmp, and that is correct to refuse."""
    (tmp_path / "a.py").write_text("x", encoding="utf-8")

    result = _skill()._list_directory(json.dumps({"path": str(tmp_path)}))

    assert result["ok"] is False
    assert "readable roots" in str(result["error"])


def test_her_own_source_tree_is_readable() -> None:
    """Reading her own code is the case that started this."""
    result = _skill()._list_directory(
        json.dumps({
            "path": _INTROSPECTION,
            "pattern": "*.py",
        })
    )

    assert result["ok"] is True
    assert result["count"] >= 8
    assert "self_evidence.py" in result["names"]


@pytest.mark.parametrize("path", ["/etc", "/", "/usr/bin"])
def test_the_rest_of_the_filesystem_is_refused(path: str) -> None:
    """Readable is wider than writable, not unbounded."""
    result = _skill()._list_directory(json.dumps({"path": path}))

    assert result["ok"] is False
    assert "readable roots" in str(result["error"])


def test_a_symlink_out_of_the_roots_is_judged_by_where_it_lands() -> None:
    """Same defence as the write guard: /etc resolves to /private/etc."""
    result = _skill()._list_directory(json.dumps({"path": "/etc"}))

    assert result["ok"] is False
    assert "resolves to" in str(result["error"])


def test_a_file_is_not_a_directory() -> None:
    result = _skill()._list_directory(
        json.dumps({"path": _CLAUDE_MD})
    )

    assert result["ok"] is False
    assert "Not a directory" in str(result["error"])


# ── The plan ───────────────────────────────────────────────────────────────

def test_the_request_plans_a_read_of_the_source_directory() -> None:
    step = DesktopTaskSkill._directory_read_step(
        OBJECTIVE, skip="~/Documents/aura_probe_count.txt"
    )

    assert step is not None
    assert step.action == "list_directory"
    target = json.loads(step.target)
    assert target["path"] == _INTROSPECTION
    assert target["pattern"] == "*.py"


def test_the_write_destination_is_never_read_as_the_source() -> None:
    """Confusing the two is what aimed a write at her own source tree."""
    step = DesktopTaskSkill._directory_read_step(
        OBJECTIVE, skip=_INTROSPECTION
    )

    if step is not None:
        assert json.loads(step.target)["path"] != (
            _INTROSPECTION
        )


def test_a_plain_write_plans_no_read() -> None:
    assert DesktopTaskSkill._directory_read_step(
        "write hello into ~/Documents/x.txt", skip="~/Documents/x.txt"
    ) is None


def test_list_directory_is_an_allowed_action() -> None:
    from core.runtime.desktop_task_contract import DESKTOP_TASK_ALLOWED_ACTIONS

    assert "list_directory" in DESKTOP_TASK_ALLOWED_ACTIONS


# ── The number written is the number read ──────────────────────────────────

def test_the_written_content_comes_from_the_read() -> None:
    receipts = [{
        "index": 0,
        "action": "list_directory",
        "ok": True,
        "effect_verified": True,
        "result": {
            "ok": True,
            "count": 9,
            "names": ["a.py", "b.py"],
            "effect_verified": True,
        },
    }]
    step = DesktopTaskStep(
        action="write_text_file",
        target=json.dumps({
            "path": "~/Documents/out.txt",
            "content": "Count: {{last.result.count}}\nNames: {{last.result.names}}",
            "overwrite": True,
        }),
        reason="r",
        expect="e",
        critical=True,
    )

    ok, resolved, error = DesktopTaskSkill._resolve_step_target(step, receipts)

    assert ok is True, error
    content = json.loads(resolved.target)["content"]
    assert "Count: 9" in content
    assert "a.py" in content


def test_an_unverified_read_does_not_get_written_up_as_fact() -> None:
    """A failed read must not become a confident zero in a file."""
    receipts = [{
        "index": 0,
        "action": "list_directory",
        "ok": False,
        "effect_verified": False,
        "result": {"ok": False, "error": "Not a directory"},
    }]
    step = DesktopTaskStep(
        action="write_text_file",
        target=json.dumps({
            "path": "~/Documents/out.txt",
            "content": "Count: {{last.result.count}}",
            "overwrite": True,
        }),
        reason="r",
        expect="e",
        critical=True,
    )

    ok, _resolved, error = DesktopTaskSkill._resolve_step_target(step, receipts)

    assert ok is False
    assert "did not verify" in error

def test_the_objective_planner_plans_read_then_write() -> None:
    skill = DesktopTaskSkill.__new__(DesktopTaskSkill)

    steps = skill._derive_steps_from_objective(OBJECTIVE, {})

    assert [step.action for step in steps] == ["list_directory", "write_text_file"]
    assert json.loads(steps[0].target)["path"] == (
        _INTROSPECTION
    )
    assert json.loads(steps[1].target)["path"] == "~/Documents/aura_probe_count.txt"


def test_a_read_verifies_on_the_reading_itself() -> None:
    """LIVE: "unsupported effect evidence for desktop action list_directory".

    Every prior action changed the world, so effect verification only knew how
    to confirm changes. A read's effect IS the reading, and without a branch
    saying so, a directory that had been read correctly reported 0/2 steps and
    the write that depended on it never ran.
    """
    step = DesktopTaskStep(
        action="list_directory",
        target=json.dumps({"path": _INTROSPECTION}),
        reason="r",
        expect="e",
        critical=True,
    )
    verified, evidence = DesktopTaskSkill._verify_step_effect(
        step,
        {
            "ok": True,
            "path": _INTROSPECTION,
            "pattern": "*.py",
            "count": 9,
            "names": ["a.py"],
        },
    )

    assert verified is True
    assert "count=9" in evidence


def test_a_failed_read_does_not_verify() -> None:
    step = DesktopTaskStep(
        action="list_directory",
        target=json.dumps({"path": "/nope"}),
        reason="r",
        expect="e",
        critical=True,
    )
    verified, evidence = DesktopTaskSkill._verify_step_effect(
        step, {"ok": False, "error": "Not a directory"}
    )

    assert verified is False
    assert "Not a directory" in evidence


def test_the_reply_reports_what_the_read_found() -> None:
    """LIVE: 2/2 steps, correct count in the correct file, and a reply that
    said only that it had run.

    The person asked "Tell me the number" and the semantic verifier reported
    requested_source_count_found incomplete — correctly. The number was in the
    receipt the whole time. A read produces a finding, and the finding is the
    answer.
    """
    from interface.routes.chat import _desktop_deliverable_text

    text = _desktop_deliverable_text({
        "receipts": [{
            "action": "list_directory",
            "ok": True,
            "result": {
                "ok": True,
                "path": "/x/core/introspection",
                "pattern": "*.py",
                "count": 9,
                "names": ["a.py", "b.py"],
            },
        }],
    })

    assert "9 file(s)" in text
    assert "a.py" in text


def test_an_unverified_read_reports_nothing() -> None:
    """Quoting a finding from a step that did not verify is the false claim
    wearing a friendlier face."""
    from interface.routes.chat import _desktop_deliverable_text

    assert _desktop_deliverable_text({
        "receipts": [{"action": "list_directory", "ok": False, "result": {"count": 9}}],
    }) == ""


def test_a_partial_task_still_reports_what_it_verified() -> None:
    """LIVE: 2/2 steps, count 9 read and written, and the reply said only
    "semantic completion incomplete: requested_source_count_found".

    A checker correctly reported that the number was missing from the reply,
    while the number sat in a verified receipt one function away. The person
    was told the task failed and never told the answer it had found. Both facts
    belong in the reply: what is verified, and what did not finish.
    """
    import inspect

    from interface.routes import chat

    source = inspect.getsource(chat._execute_desktop_objective_from_chat)
    # The failure's own wording changed ("This did not complete:", "The rest did not complete:", 3ed4138b7); what is
    # protected is that the verified part is read from the receipts and leads the reply.
    marker = "partial = _desktop_deliverable_text(result)"
    assert marker in source
    window = source[source.find(marker) : source.find(marker) + 1600]

    assert "That much is verified" in window
    assert "did not complete:" in window


# ── A path is a name, not prose ────────────────────────────────────────────

def test_a_path_containing_source_is_not_a_research_request() -> None:
    """LIVE: the marker "source" matched inside "live-source".

    "count how many .py files are in /Users/bryan/.aura/live-source/..." was
    classified as a research-document objective, so completion required
    research SOURCES, and a filesystem task that had fully succeeded reported
    "semantic completion incomplete: requested_source_count_found".

    Every marker in that classifier is a substring test over whatever the user
    typed, so any objective naming a path with "source", "report", "news" or
    "article" in it inherited a contract about citations.
    """
    assert DesktopTaskSkill._objective_requests_research_document(OBJECTIVE) is False


@pytest.mark.parametrize(
    "objective",
    [
        "Research the latest news on fusion and write a report with three sources "
        "into ~/Documents/fusion.md",
        "summarize three articles about otters into a document",
        "look up two sources on tides and write them up",
    ],
)
def test_genuine_research_requests_still_require_sources(objective: str) -> None:
    """The direction that matters: this contract exists for a reason."""
    assert DesktopTaskSkill._objective_requests_research_document(objective) is True


@pytest.mark.parametrize(
    "objective",
    [
        "write hello into ~/Documents/x.txt",
        "list the files in ~/Documents/reports",
        f"read {Path.home() / 'news-archive' / 'index.txt'} and tell me the first line",
    ],
)
def test_paths_with_research_words_in_them_stay_filesystem_tasks(objective: str) -> None:
    assert DesktopTaskSkill._objective_requests_research_document(objective) is False
