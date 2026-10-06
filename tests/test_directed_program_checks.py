"""Rare code paths are exercised explicitly, independent of incidental play."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.agency.when_motion_breaks_a_rule import MotionChecks
from core.self_modification.checking_code_paths import boundary_checks, check_code_paths
from core.self_modification.checking_python import FunctionExample, check_python, examples_in
from core.self_modification.code_that_looks_wrong import applied, what_looks_wrong


@pytest.fixture
async def browser():
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        yield browser
        await browser.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("axis,body,limit,size", [("x", "parcel", "SPAN", "SIDE"), ("y", "packet", "HEIGHT", "LENGTH")])
async def test_directed_reflection_checks_transfer_across_names_and_axes(browser, axis, body, limit, size):
    source = f"""<script>
const {limit}=100,{size}=10;
const {body}={{{axis}:50,v:30}};
let total=0;
function advance(dt){{
 if({body}.{axis}+{size}>{limit}){{{body}.{axis}={limit}-{size};{body}.v=-Math.abs({body}.v);}}
}}
throw Error('the application entry point must not run during a function check');
</script>"""
    [check] = boundary_checks(source)
    results = await check_code_paths(browser, source)
    assert [r["verdict"] for r in results] == ["wrong", "right", "right"]
    [suspicion] = [s for s in what_looks_wrong(source, ".html") if s.pattern == "one-sided boundary"]
    results = await check_code_paths(browser, applied(source, suspicion.edits), checks=[check])
    assert all(r["verdict"] == "right" for r in results)
    assert all(r["unchanged"] for r in results)


@pytest.mark.asyncio
async def test_directed_checks_reject_scoring_instead_of_reflecting(browser):
    source = """<script>const HEIGHT=100,SIZE=10;const object={y:50,vy:30};let score=0;
function move(dt){if(object.y<0){object.y=0;object.vy=Math.abs(object.vy);score++;}
if(object.y+SIZE>HEIGHT){object.y=HEIGHT-SIZE;object.vy=-Math.abs(object.vy);}}</script>"""
    result = await check_code_paths(browser, source)
    assert result[0]["bounded"] and result[0]["directed"]
    assert result[0]["verdict"] == "wrong" and not result[0]["unchanged"]


@pytest.mark.asyncio
async def test_fixture_ceiling_is_measured_without_waiting_for_a_random_trajectory(browser):
    source = (Path(__file__).parent / "fixtures/pong_repair/pong.html").read_text()
    before = await check_code_paths(browser, source)
    assert before[0]["verdict"] == "wrong"
    suspicion = next(s for s in what_looks_wrong(source, ".html") if s.pattern == "one-sided boundary")
    after = await check_code_paths(browser, applied(source, suspicion.edits))
    assert all(r["verdict"] == "right" for r in after)


def test_runtime_contract_separates_open_edges_from_closed_edges():
    moves = SimpleNamespace(shape=(100, 160))
    event = [{"what": "gone", "thing": 3, "x": 60, "y": -1, "vx": 20, "vy": -50}]
    assert not MotionChecks().see(moves, event, 1, 1)
    monitor = MotionChecks(frozenset({"top", "bottom"}), "source reflection")
    assert monitor.see(moves, event, 1, 1)[0]["edge"] == "top"
    assert not monitor.see(moves, event, 1, 3)
    assert not monitor.see(moves, [*event, {"what": "new screen"}], 1, 1)


@pytest.mark.asyncio
async def test_python_examples_execute_in_os_sandbox_and_are_compared_outside_the_candidate():
    source = '''def double(value):
    """>>> double(3)
    6
    >>> double(-2)
    -4
    """
    return value * 2
'''
    examples = examples_in(source)
    good = await check_python(source, examples)
    assert good["verified"] and len(good["cases"]) == 2
    assert all(c["sandboxed"] and c["boundary"] != "none" for c in good["cases"])
    bad = await check_python(source.replace("value * 2", "value * 0"), examples)
    assert not bad["verified"] and all(c["verdict"] == "wrong" for c in bad["cases"])


@pytest.mark.asyncio
async def test_a_python_draft_without_examples_is_unverified():
    assert (await check_python("def step(x): return x"))["verified"] is False
    assert (await check_python("def step(:"))["syntax"] is False


@pytest.mark.asyncio
async def test_python_repair_uses_pinned_examples(tmp_path, monkeypatch):
    from core.self_modification.repairing_python import repair_python
    from core.runtime.file_write_gateway import get_file_write_gateway

    path = tmp_path / "calculation.py"
    original = 'def area(width, height):\n    """>>> area(3, 4)\n    12\n    """\n    return width * height * 0\n'
    path.write_text(original)
    gateway = get_file_write_gateway()

    async def write(path, text, **kwargs):
        path.write_text(text)

    monkeypatch.setattr(gateway, "write_text_async", write)
    repair = await repair_python(path)
    assert repair.checked == ["example 1"] and not repair.after and not repair.unseen
    assert path.with_name(path.name+".before-repair").read_text() == original
    assert "* 0" not in path.read_text()


def test_structural_python_edit_preserves_decorators_newlines_and_refuses_stale_undo(tmp_path, monkeypatch):
    from core.self_modification import coding_aci

    path = tmp_path / "unit.py"
    original = b"class Unit:\r\n    @staticmethod\r\n    def value():\r\n        return 1"
    path.write_bytes(original)
    monkeypatch.setattr(coding_aci, "_write", lambda p, s: p.write_bytes(s.encode()))
    surface = coding_aci.CodingSurface(tmp_path)
    edit = surface.replace_definition("unit.py", "Unit.value", "    @staticmethod\n    def value():\n        return 2")
    assert path.read_bytes().count(b"@staticmethod") == 1
    assert b"\r\n" in path.read_bytes() and not path.read_bytes().endswith(b"\n")
    surface.revert(edit)
    assert path.read_bytes() == original
    edit = surface.replace_definition("unit.py", "Unit.value", "    def value():\n        return 3")
    path.write_bytes(path.read_bytes()+b"\r\n# newer work")
    with pytest.raises(ValueError, match="newer work"):
        surface.revert(edit)
