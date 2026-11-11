"""What one of her tools gives back reaches her model as something she can read.

On 9 and 10 November 2026, 80 of the 161 calls of her goal "Find something productive to work on" came back cut at
4,000 characters and escaped twice: a read of 120 lines of source ran to 5,000 or more, and the bound re-encoded the
whole result as one string in a "preview" field, so the code arrived with its line ends as a backslash and an n and
stopped mid-line. Twice she wrote her call with its opening tag doubled, and it was read as prose. None of her 14 goals
reached an edit.
"""
from __future__ import annotations

import asyncio
import inspect
import json

from core.brain.llm import tool_result_room
from core.brain.llm.mlx_client import MLXLocalClient, _serialize_tool_result_for_model, _truncate_tool_result
from core.skills import her_draft
from tests.test_her_goal_is_worked_by_her import _git, her_source  # noqa: F401

_SOURCE = "".join(f'    value_{n} = compute("field", {n})  # a line of real-looking source\n' for n in range(200))
_RECEIPT = {
    "authority_closure": {"closed": True, "errors": [], "intent_closed": True, "mode": "authority_gateway",
                          "standing_authority_closed": True, "success": True, "token_revoked": True},
    "duration_ms": 1.5, "retries": 0, "skill": "read_source",
}
_A_LONG_READ = {
    **_RECEIPT, "ok": True, "path": "core/memory/cold_store.py",
    "text": "".join(f"{n + 1}: {line}\n" for n, line in enumerate(_SOURCE.splitlines())),
}
_ESCAPED_TWICE = "\\n"  # a backslash and an n, which is what a line end became


def test_a_long_result_is_cut_inside_its_longest_value_and_reads_as_written():
    out = _truncate_tool_result(json.dumps(_A_LONG_READ), limit=4000)
    assert len(out) <= 4000
    parsed = json.loads(out)
    assert parsed["truncated"] is True and parsed["original_chars"] == len(json.dumps(_A_LONG_READ))
    assert parsed["path"] == "core/memory/cold_store.py", "the result keeps its own fields"
    text = parsed["text"]
    assert _ESCAPED_TWICE not in text and '\\"' not in text, "escaped once, as JSON is"
    kept, marker = text.rsplit("\n", 1)
    assert marker.startswith("[... ") and marker.endswith(" more characters]")
    assert kept.splitlines()[-1].endswith("# a line of real-looking source"), "cut at a line end"


def test_a_result_no_single_value_can_carry_still_bounds_as_before():
    payload = json.dumps({"rows": [{"i": i, "text": "x" * 40} for i in range(400)]})
    out = _truncate_tool_result(payload, limit=2000)
    assert json.loads(out)["truncated"] is True and len(out) <= 2000


def test_the_room_is_the_one_the_model_client_bounds_to():
    for function in (_truncate_tool_result, _serialize_tool_result_for_model):
        assert inspect.signature(function).parameters["limit"].default == tool_result_room.TOOL_RESULT_CHARS


def test_a_call_with_its_opening_tag_written_twice_is_still_a_call():
    tools = {"read_source": {"function": {"name": "read_source", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}, "from_line": {"type": "integer"}}}}}}
    said = ("Let me look at the method more carefully.\n\n<tool_call> <tool_call> <function=read_source> "
            "<parameter=path> core/memory/memory_facade.py </parameter> <parameter=from_line> 634 </parameter> "
            "</function> </tool_call>")
    call = MLXLocalClient._extract_tool_call_payload(said, allowed_tools={"read_source"}, tool_definitions=tools)
    assert call is not None and call["tool"] == "read_source"
    assert call["args"]["from_line"] == 634


def test_a_read_of_a_long_file_fits_one_result_and_says_where_to_read_on(her_source):
    from core.agency import her_goal_at_work as work

    workspace = work.workspaces_dir() / "a-goal-of-hers"
    (workspace / "core").mkdir(parents=True)
    _git("init", "-q", cwd=workspace)
    (workspace / "core" / "long.py").write_text(_SOURCE)
    context = {"her_goal_workspace": str(workspace)}
    result = asyncio.run(her_draft.ReadSourceSkill().execute({"path": "core/long.py"}, context))
    shown = result["text"].splitlines()
    served = _serialize_tool_result_for_model("read_source", {**_RECEIPT, **result})
    assert "truncated" not in json.loads(served), "it fits, receipt and all, so nothing cuts it"
    after = len(shown) + 1
    assert f"read on from line {after}" in result["summary"]
    more = asyncio.run(her_draft.ReadSourceSkill().execute({"path": "core/long.py", "from_line": after}, context))
    assert more["text"].splitlines()[0].startswith(f"{after}: ")


def test_her_model_reads_what_the_tool_found_and_not_the_receipts_of_its_going():
    """LIVE 10 November 2026: each of her goal's read_source results opened with 200 characters of authority receipt."""
    shown = json.loads(_serialize_tool_result_for_model(
        "read_source", {**_RECEIPT, "ok": True, "path": "core/memory/cold_store.py", "text": "1: x = 1\n"}))
    for receipt in ("authority_closure", "duration_ms", "retries"):
        assert receipt not in shown
    assert shown == {"ok": True, "path": "core/memory/cold_store.py", "skill": "read_source", "text": "1: x = 1\n"}


def test_a_call_authority_refused_still_says_so():
    shown = json.loads(_serialize_tool_result_for_model(
        "read_source", {**_RECEIPT, "ok": False, "status": "authority_closure_failed", "error": "token not revoked"}))
    assert shown["status"] == "authority_closure_failed" and shown["error"] == "token not revoked"
