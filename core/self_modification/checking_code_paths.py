"""Directed function checks derived from a program's boundary branches.

These are isolated repair experiments, never observations or inputs used to
play. A mirrored reflection suggests a bounded axis; each side and the
interior are exercised directly. Unsupported functions remain unmeasured.
"""
from __future__ import annotations

import json
import re
import asyncio
from dataclasses import asdict, dataclass
from typing import Any

from core.self_modification.code_that_looks_wrong import _parser, _text, _walk, scripts_in


@dataclass(frozen=True)
class BoundaryCheck:
    function: str
    position: str
    velocity: str
    extent: str
    limit: str
    axis: str


def boundary_checks(source: str, suffix: str = ".html") -> list[BoundaryCheck]:
    """Reflection branches with a named position, size and bound, independent of identifiers."""
    found = []
    for _offset, script in scripts_in(source, suffix):
        code = script.encode()
        root = _parser("javascript").parse(code).root_node
        for node in _walk(root):
            if node.type != "if_statement":
                continue
            test, body = node.child_by_field_name("condition"), node.child_by_field_name("consequence")
            if test is None or body is None:
                continue
            match = re.fullmatch(r"\s*([\w]+\.([xy]))\s*\+\s*(\w+)\s*>\s*(\w+)\s*", _text(test, code).strip("() "))
            reflected = re.search(r"(\w+\.\w+)\s*=\s*-\s*Math\.abs\(\s*\1\s*\)", _text(body, code))
            if match is None or reflected is None:
                continue
            parent = node.parent
            while parent is not None and parent.type != "function_declaration":
                parent = parent.parent
            if parent is None:
                continue
            name = parent.child_by_field_name("name")
            parameters = parent.child_by_field_name("parameters")
            if name is None or parameters is None or len(parameters.named_children) > 1:
                continue
            check = BoundaryCheck(_text(name, code), match[1], reflected[1], match[3], match[4], match[2])
            if check not in found:
                found.append(check)
    return found


def _declarations(source: str, suffix: str) -> tuple[str, list[str], list[str]]:
    """Keep declarations; omit top-level event registration, scheduling and application entry calls."""
    declarations, mutable, functions = [], [], []
    for _offset, script in scripts_in(source, suffix):
        code = script.encode()
        root = _parser("javascript").parse(code).root_node
        for node in root.named_children:
            if node.type not in ("lexical_declaration", "variable_declaration", "function_declaration", "class_declaration"):
                continue
            declarations.append(_text(node, code))
            if node.type == "function_declaration":
                parameters = node.child_by_field_name("parameters")
                name = node.child_by_field_name("name")
                if parameters is not None and not parameters.named_children and name is not None:
                    functions.append((_text(name, code), _text(node, code)))
            if node.type == "lexical_declaration" and _text(node, code).startswith("let "):
                for part in node.named_children:
                    name = part.child_by_field_name("name")
                    if name is not None and name.type == "identifier":
                        mutable.append(_text(name, code))
    # A reset/factory that assigns several module variables can prepare an
    # uninitialised object without running the event loop or application entry.
    initializers = sorted(functions, key=lambda f: -sum(bool(re.search(rf"\b{re.escape(n)}\s*=", f[1])) for n in mutable))
    initializers = [name for name, body in initializers if sum(bool(re.search(rf"\b{re.escape(n)}\s*=", body)) for n in mutable) >= 2]
    return "\n".join(declarations), mutable, initializers


async def check_code_paths(browser: Any, source: str, *, suffix: str = ".html",
                           checks: list[BoundaryCheck] | None = None) -> list[dict[str, Any]]:
    """Exercise functions in a fresh document, with a receipt for every input and result."""
    from playwright.async_api import Error as BrowserError

    checks = boundary_checks(source, suffix) if checks is None else checks
    if not checks:
        return []
    declarations, mutable, initializers = _declarations(source, suffix)
    markup = re.sub(r"<script\b[^>]*>.*?</script>", "", source, flags=re.I | re.S) if suffix in (".html", ".htm") else ""
    receipts = []
    for check in checks[:8]:
        for label, location, sign in (("near", "-1", -1), ("inside", f"({check.limit}-{check.extent})/2", 1),
                                      ("far", f"{check.limit}-{check.extent}+1", 1)):
            page = await browser.new_page()
            result: dict[str, Any] = {"check": asdict(check), "case": label, "verdict": "unmeasured"}
            numbers = ",".join(f"{json.dumps(n)}:typeof {n}==='number'?{n}:null" for n in mutable)
            root = check.position.split(".")[0]
            initialize = f"if(typeof {root}==='undefined'){{{initializers[0]}();}}" if initializers else ""
            harness = f"""
try {{
  {initialize}
  {check.position}={location}; {check.velocity}={sign}*Math.max(30,Math.abs({check.velocity}));
  const initial={{position:{check.position},velocity:{check.velocity},numbers:{{{numbers}}}}};
  {check.function}(0);
  const final={{position:{check.position},velocity:{check.velocity},numbers:{{{numbers}}}}};
  const bounded=Number.isFinite(final.position)&&final.position>=0&&final.position+{check.extent}<={check.limit};
  const directed={json.dumps(label)}==='near'?final.velocity>=0:{json.dumps(label)}==='far'?final.velocity<=0:final.velocity===initial.velocity;
  const unchanged=JSON.stringify(initial.numbers)===JSON.stringify(final.numbers);
  window.__auraCodeCheck={{verdict:bounded&&directed&&unchanged?'right':'wrong',initial,final,bounded,directed,unchanged}};
}} catch(error) {{ window.__auraCodeCheck={{verdict:'unmeasured',error:String(error).slice(0,400)}}; }}
"""
            try:
                async with asyncio.timeout(10):
                    await page.set_content(markup, timeout=5000)
                    await page.add_script_tag(content=declarations + "\n" + harness)
                    measured = await page.evaluate("window.__auraCodeCheck || null")
                if isinstance(measured, dict):
                    result.update(measured)
            except (RuntimeError, TimeoutError, BrowserError) as exc:
                result["error"] = str(exc)[:400]
            finally:
                await page.close()
            receipts.append(result)
    return receipts
