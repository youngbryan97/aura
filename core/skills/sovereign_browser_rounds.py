"""The record of each round of a browser pursuit, and the account of the whole of it.

Lifted whole out of `sovereign_browser`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

from typing import Any


def _account_of_the_pursuit(
    *,
    completed: Any,
    concluded: Any,
    final: Any,
    goal: Any,
    observation: Any,
    self: Any,
    steps: Any,
) -> Any:
    """What a pursuit reports: whether it landed anything, what stopped it, and where it ended.

    Moved out of ``_handle_pursue`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 7 name(s) and hands back
    0.
    """
    def _block() -> Any:
        self._retain_stated_positions(goal, steps)
        landed_total = sum(int(step.get("landed") or 0) for step in steps)
        # Work that landed is work that happened. A run that answered forty
        # questions and then hit a slow round is not a failed run, and
        # reporting it as one is what made a timeout look like nothing had
        # been done at all.
        ok = completed or landed_total > 0 or bool(steps and not steps[-1].get("error"))
        # And where it is not ok, what stopped it.
        #
        # Every round that failed wrote its reason into `steps`, and the result
        # carried no `error` field at all, so `BaseSkill` filled the silence
        # with "sovereign_browser reported failure without a cause
        # (status=failed_recoverable)" — LIVE 2026-09-28 00:45, over a run whose
        # own log said `empty_decision` one line earlier. The surprise engine
        # banked the causeless version.
        # Named most recent first, and every distinct one of them: the round
        # that ended the run says the stall limit was reached, and the round
        # before it says why nothing was landing. Reporting only the last gives
        # "no_progress", which is the fact the caller already has.
        seen: list[str] = []
        for step in reversed(steps):
            reason = str(step.get("error") or "").strip()
            if reason and reason not in seen:
                seen.append(reason)
        stopped_by = ":".join(seen[:3])
        return {
            "ok": ok,
            "error": "" if ok else (stopped_by or "pursuit_made_no_progress"),
            "landed_total": landed_total,
            "goal": goal,
            "completed": completed,
            "steps": steps,
            "rounds": len(steps),
            # `observed_url` is the name the effect verifier reads. Returning
            # only `final_url` meant a completed pursuit presented no evidence
            # it had ever been anywhere, and verification failed on a run that
            # had worked — a new mode conforming to its own vocabulary instead
            # of the one the transaction already speaks.
            "observed_url": (final or observation).get("url", ""),
            "final_url": (final or observation).get("url", ""),
            "result_text": str((final or observation).get("text") or "")[:4000],
            "concluded": concluded,
        }

    _seam_early_response = _block()
    return _seam_early_response


def _record_the_round(
    *,
    asked: Any,
    decision: Any,
    elements: Any,
    observation: Any,
    report: Any,
    steps: Any,
) -> None:
    """Put one round on the pursuit's record: what was asked, what she chose, what she said, and whether it landed.

    Moved out of ``_handle_pursue`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 6 name(s) from the turn and hands back
    0.
    """
    steps.append(
        {
            "asked": next(
                (line for line in asked if "?" in line or line.lower().startswith("question")),
                (asked[0] if asked else ""),
            )[:180],
            "why": str(decision.get("why") or ""),
            "chose": [
                str(item.get("name") or "")
                for item in (decision.get("resolved_actions") or [])
                if isinstance(item, dict)
            ]
            or [
                f"{elements[int(item['index'])].get('name')}"
                for item in (decision.get("actions") or [])
                if isinstance(item, dict)
                and str(item.get("index", "")).lstrip("-").isdigit()
                and 0 <= int(item["index"]) < len(elements)
            ],
            # What she SAID for each choice, one line per answer.
            #
            # The account in the reply was built from the round: the
            # first line of the page with a question mark in it, the
            # names of every control pressed, and one reason for the
            # lot. On a screen of eight answers that is a heading
            # repeated eight times and eight control names, and her
            # eight sentences — the whole point of answering as
            # herself — reached nobody. They are already written, for
            # the narration that speaks them as each move is made.
            "said": [
                str(item.get("said") or "")
                for item in (decision.get("resolved_actions") or [])
                if isinstance(item, dict) and str(item.get("said") or "").strip()
            ],
            "ok": bool(report.get("ok")),
            "url": observation.get("url"),
            # What kind of move each was (click, type, scroll, go), so what the round did can vouch for her saying so.
            "kinds": sorted(
                {"click" for item in (decision.get("resolved_actions") or []) if isinstance(item, dict)}
                | {
                    str(item.get("type") or "click").lower()
                    for item in (decision.get("actions") or [])
                    if isinstance(item, dict)
                }
            ),
        }
    )


