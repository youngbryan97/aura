"""Whether a reply answers a problem, by each benchmark's own rule. Nothing here imports her code."""
from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import Any

from outside_bar import planbench

BENCH = Path("~/.aura/benchmarks").expanduser()
#: How long MATH-500's grader may take on one answer, in seconds: the bound G12 gave it.
_GRADER_SECONDS = 10


def last_boxed(text: str) -> str | None:
    """The contents of the last ``\\boxed{...}`` or ``\\fbox{...}``, braces matched."""
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    opening = text.find("{", start) if start >= 0 else -1
    if opening < 0:
        return None
    depth = 0
    for index in range(opening, len(text)):
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        if depth == 0:
            return text[opening + 1:index]
    return None


def _math(reply: str, truth: str) -> tuple[bool, str]:
    body = reply.strip()
    answer = last_boxed(reply)
    if answer is None and body and "\n" not in body and len(body) <= 40:
        answer = body
    if answer is None:
        return False, "no answer"
    if str(BENCH / "math500") not in sys.path:
        sys.path.insert(0, str(BENCH / "math500"))
    from grading.grader import grade_answer

    def _too_long(*_: Any) -> None:
        raise TimeoutError

    before = signal.signal(signal.SIGALRM, _too_long)
    signal.alarm(_GRADER_SECONDS)
    try:
        return bool(grade_answer(answer, truth)), answer[:120]
    except Exception:  # noqa: BLE001 - an answer the grader cannot read is a wrong answer
        return False, answer[:120]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, before)


def _bbeh_answer(sample: str) -> str:
    """BBEH's extraction and preprocessing (bbeh/evaluate.py), restated."""
    answer = sample.strip()
    for prefix in ("The answer is:", "The final answer is ", "The final answer is: ", "The answer is "):
        if prefix in answer:
            answer = answer.split(prefix)[-1].strip()
    if answer.endswith("."):
        answer = answer[:-1]
    if answer.startswith("$") and answer.endswith("$"):
        answer = answer[1:-1]
    for wrapper in ("boxed{", "text{", "texttt{"):
        if wrapper in answer and answer.endswith("}"):
            answer = answer[0:-1].split(wrapper)[1]
    answer = answer.lower().replace(", ", ",").replace("**", "").split("\n")[0]
    return answer[0:-1] if answer.endswith(".") else answer


def _bbeh_match(prediction: str, reference: str) -> bool:
    """BBEH's fuzzy_match, restated."""
    if prediction == reference:
        return True
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True
    return prediction.endswith("?") and prediction[:-1] == reference


def _plan(reply: str, truth: dict[str, Any]) -> tuple[bool, str]:
    written = reply if "[PLAN END]" in reply else planbench.numbered_plan(reply)
    if planbench.grade(written, truth)[0]:
        return True, "valid plan"
    steps = planbench.action_lines_plan(reply, str(truth["domain"]))
    return (bool(steps) and bool(planbench.grade(steps, truth)[0])), "plan read from its action lines"


def grade(domain: str, reply: str, truth: Any) -> tuple[bool, str]:
    """(right, what was read as the answer)."""
    reply = str(reply or "")
    if domain == "math500":
        return _math(reply, str(truth))
    if domain == "bbeh_mini":
        answer = _bbeh_answer(reply)
        return _bbeh_match(answer, str(truth).strip().lower().replace(", ", ",")), answer[:120]
    if domain in ("blocksworld", "mystery_blocksworld"):
        return _plan(reply, truth)
    raise KeyError(domain)
