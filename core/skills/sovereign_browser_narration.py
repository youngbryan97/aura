"""What she says about the browsing as she does it: each step, and each decision with its reason.

Lifted whole out of `sovereign_browser`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class _NarratesTheBrowsing:
    """Lifted whole out of SovereignBrowserSkill; see sovereign_browser.py."""

    def _narrate(self, step: Mapping[str, Any]) -> None:
        """Put one round of a pursuit into the thought stream as it happens.

        What she chose and why, against the question she was reading. This is
        the same record the trace keeps, said out loud at the time.
        """
        from .sovereign_browser import (
            get_emitter,
            record_degradation,
        )

        chose = [str(name) for name in (step.get("chose") or []) if name]
        asked = str(step.get("asked") or "").strip()
        why = str(step.get("why") or "").strip()
        said = " / ".join(chose) if chose else "reading the page"
        if asked:
            said = f"{asked} -> {said}"
        if why:
            said = f"{said}. {why}"
        try:
            from core.thought_stream import get_emitter

            get_emitter().emit(
                "Browsing",
                said[:400],
                level="info",
                category="ToolExecution",
            )
        except Exception as exc:  # narration must never break the pursuit
            record_degradation("sovereign_browser", exc, action="pursuit narration skipped")

    def _narrate_decision(
        self, decision: Mapping[str, Any], observation: Mapping[str, Any], goal: str = ""
    ) -> str:
        """Say every decision as it is made, including the ones that do nothing.

        Narration fired once, after an action report, so the rounds that
        produced no action said nothing at all: a decision that could not be
        read, a claim of "done" before anything was done, an answer naming a
        control that was not on the page. Those are the rounds a person
        watching most needs to hear, because they are the ones where a run
        stops making sense — and from outside they looked like a browser
        sitting still.
        """
        # Ranked by the same goal the decision was shown, or index 3 names one
        # control on screen and another in the sentence. LIVE 2026-09-28: "I
        # need to click the Open Jungian Type Scales link ... I am going to
        # about." — the page's nav link, read out of a differently ordered list.
        elements = self._controls_worth_offering(
            list(observation.get("elements") or []), goal
        )
        why = " ".join(str(decision.get("why") or "").split())
        parts: dict[str, str] | None = None
        if decision.get("error"):
            raw = " ".join(str(decision.get("raw") or "").split())
            said = f"I could not use my own answer here ({decision['error']})"
            if raw:
                said = f"{said}. What I said was: {raw[:160]}"
        elif decision.get("done") is True and not decision.get("actions"):
            said = f"I think this is finished. {why}" if why else "I think this is finished."
        elif any(
            str(item.get("said") or "").strip() or callable(item.get("think"))
            for item in (decision.get("resolved_actions") or [])
            if isinstance(item, dict)
        ):
            # Every move in this round carries its own words and will say them
            # as it is made. A decision-level line on top of those is the same
            # content twice, which is what a watcher saw: "sceptical ... wants
            # to believe — 3 of 5" and then the identical sentence again.
            return ""
        else:
            naming = [
                str(elements[int(item["index"])].get("name") or "").strip()
                for item in (decision.get("actions") or [])
                if isinstance(item, dict)
                and str(item.get("index", "")).lstrip("-").isdigit()
                and 0 <= int(item["index"]) < len(elements)
            ]
            doing = ", ".join(name for name in naming if name)
            # Nothing to report is not worth a line. "I am deciding what to do
            # here" told a watcher that a decision had been made and nothing
            # about it.
            said = f"{why} I am going to {doing}." if doing and why else (
                why or (f"I am going to {doing}." if doing else "")
            )
            # Laid out, the reason is hers and the act is a footer under it;
            # the joined line stays for every surface that reads only text.
            parts = {"said": why, "doing": doing} if why else None
        if not said:
            return ""
        self._narrate({"why": said, "asked": str(observation.get("title") or "")})
        self._say_out_loud(said, parts)
        return said

