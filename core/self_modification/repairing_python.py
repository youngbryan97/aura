"""External Python repairs judged by pinned examples, using the shared sandbox."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from core.self_modification.checking_python import FunctionExample, check_python, examples_in
from core.self_modification.code_that_looks_wrong import applied, what_looks_wrong


async def repair_python(path: Path, *, say: Any = None, examples: list[FunctionExample] | None = None) -> Any:
    from core.self_modification.saving_a_verified_repair import save_repair
    from core.self_modification.edits_her_model_proposes import edits_her_model_proposes
    from core.self_modification.repairing_by_behaviour import Repair, _candidates

    source = await asyncio.to_thread(path.read_text)
    pinned = examples_in(source) if examples is None else examples
    repair = Repair(path=str(path))

    def tell(line: str) -> None:
        repair.said.append(line)
        if say is not None:
            say(line)

    before = await check_python(source, pinned)
    repair.code_checks.append({"stage": "before", "results": before})
    repair.before = {c["name"]: f"expected {c['expected']!r}, observed {c['actual']!r} ({c['verdict']})"
                     for c in before.get("cases", []) if c["verdict"] != "right"}
    if not before.get("syntax"):
        repair.before["syntax"] = before.get("error", "syntax failed")
    tell(f"Read the Python source and checked {len(pinned)} independent example(s) in the OS sandbox.")
    current, state = source, before
    deadline = time.monotonic() + 120
    for _round in range(4):
        if state.get("verified"):
            break
        if not pinned:
            break
        candidates = _candidates(what_looks_wrong(current, ".py"))
        chosen = None
        right_before = {c["name"] for c in state.get("cases", []) if c["verdict"] == "right"}
        for phase in range(2):
            if phase:
                if chosen is not None or time.monotonic() >= deadline:
                    break
                findings = {c["name"]: f"expected {c['expected']!r}, got {c['actual']!r}; {c.get('error', '')}"
                            for c in state.get("cases", []) if c["verdict"] != "right"}
                if not state.get("syntax"):
                    findings["syntax"] = state.get("error", "syntax failed")
                candidates = _candidates(await edits_her_model_proposes(current, findings))
            for suspicion, edits in candidates[:8]:
                if time.monotonic() >= deadline:
                    break
                candidate = applied(current, edits)
                measured = await check_python(candidate, pinned, timeout_s=min(5, max(0.1, deadline-time.monotonic())))
                right = {c["name"] for c in measured.get("cases", []) if c["verdict"] == "right"}
                if measured.get("syntax") and right_before <= right and len(right) > len(right_before):
                    if chosen is None or len(right) > chosen[0]:
                        chosen = (len(right), suspicion, edits, candidate, measured)
        if chosen is None:
            break
        _score, suspicion, edits, candidate, measured = chosen
        repair.kept.append({"where": suspicion.function, "line": suspicion.line, "pattern": suspicion.pattern,
                            "why": suspicion.why, "change": ", ".join(e.says(current) for e in edits),
                            "shown": [c["name"] for c in measured["cases"] if c["verdict"] == "right"]})
        tell(f"Line {suspicion.line}: {suspicion.why}. The retained edit passed more examples and preserved every passing example.")
        current, state = candidate, measured
    final = await check_python(current, pinned)
    repair.code_checks.append({"stage": "final", "results": final})
    repair.checked = [c["name"] for c in final.get("cases", []) if c["verdict"] == "right"]
    repair.after = {c["name"]: f"the example is {c['verdict']}: {c.get('error') or c['actual']!r}"
                    for c in final.get("cases", []) if c["verdict"] != "right"}
    repair.unseen = ["behaviour"] if not pinned else []
    if not final.get("syntax"):
        repair.after["syntax"] = final.get("error", "syntax failed")
    if current != source:
        repair.backup = await save_repair(path, source, current)
        repair.written_to = str(path)
    tell(f"Kept {len(repair.kept)} Python edit(s); {len(repair.checked)} examples pass. "
         + ("The supplied examples verify the repaired functions." if final.get("verified") else "Behaviour remains unverified."))
    return repair
