"""General terminal parser — the NetHack glyph/threat idea, generalized to any terminal.

NetHackParser scores danger from a game-specific glyph→threat table. But the same shape —
read the raw text frame, recognize what's on screen, score how dangerous the situation is —
applies to *any* terminal the agent is looking at: a shell, a REPL, a build log, an SSH
session. The danger just comes from terminal *semantics* instead of game monsters.

This parser turns arbitrary terminal text into the shared EnvironmentState:

    * self_state   — cwd, shell/process, last exit status, prompt kind, line/char counts
    * entities     — notable lines (errors, tracebacks, warnings, destructive commands),
                     each with a threat_score, mirroring NetHack's per-glyph threat scoring
    * active_prompts — password / [y/N] confirmation / sudo / pager / REPL prompts that
                     block or gate standard interaction
    * uncertainty  — how confidently we read the frame
    * a frame-level threat in self_state["threat_score"] (max of line threats + prompt risk)

It stays fast and LLM-free, so it can run on the perception hot path and feed the same
belief/risk/action-gating systems the other parsers feed.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from core.perception.environment_parser import EnvironmentParser, EnvironmentState


# ── threat lexicon: terminal semantics → [0,1] danger, the generalization of GLYPH_THREAT ──

# (compiled regex, threat, label). First strong match per line wins for classification, but
# the line's threat is the max over all matches.
_THREAT_PATTERNS: List[Tuple[re.Pattern, float, str]] = [
    # destructive / irreversible commands — the highest terminal danger
    (re.compile(r"\brm\s+-[a-z]*r[a-z]*f|\brm\s+-[a-z]*f[a-z]*r", re.I), 0.97, "destructive_rm"),
    (re.compile(r"\b(mkfs|dd\s+if=|shred|fdisk|:\s*>\s*/dev/|>\s*/dev/sd)", re.I), 0.95, "destructive_disk"),
    (re.compile(r"\bgit\s+(reset\s+--hard|clean\s+-[a-z]*f|push\s+--force)", re.I), 0.8, "destructive_git"),
    (re.compile(r"\b(chmod\s+-R\s+777|curl\s+[^|]*\|\s*(sudo\s+)?sh|wget\s+[^|]*\|\s*sh)", re.I), 0.9, "risky_pipe_exec"),
    (re.compile(r"\bsudo\b", re.I), 0.55, "privilege_escalation"),
    # crashes / fatal runtime failures
    (re.compile(r"\b(segmentation fault|segfault|core dumped|kernel panic|panic:)", re.I), 0.9, "crash"),
    (re.compile(r"\b(fatal|fatal error)\b", re.I), 0.8, "fatal"),
    (re.compile(r"Traceback \(most recent call last\)", re.I), 0.75, "python_traceback"),
    (re.compile(r"\b[A-Za-z_]+(Error|Exception):", ), 0.7, "exception"),
    # ordinary errors / failures
    (re.compile(r"\b(permission denied|access denied|not permitted)\b", re.I), 0.65, "permission_denied"),
    (re.compile(r"\b(command not found|no such file or directory)\b", re.I), 0.5, "not_found"),
    (re.compile(r"\b(connection refused|timed out|timeout|unreachable)\b", re.I), 0.55, "network_failure"),
    (re.compile(r"\b(error|failed|failure)\b", re.I), 0.45, "error"),
    (re.compile(r"\b(warning|deprecated)\b", re.I), 0.25, "warning"),
]

# ── prompt detection: what is blocking / gating interaction right now ──
_PROMPT_PATTERNS: List[Tuple[re.Pattern, float, str]] = [
    (re.compile(r"(password.*:|passphrase.*:)\s*$", re.I), 0.7, "password_prompt"),
    (re.compile(r"\[sudo\]\s+password", re.I), 0.7, "sudo_prompt"),
    (re.compile(r"\(yes/no\)|\[y/n\]|\[Y/n\]|\[y/N\]|\?\s*$", re.I), 0.4, "confirmation_prompt"),
    (re.compile(r"^(>>>|\.\.\.)\s*$"), 0.1, "python_repl"),
    (re.compile(r"--More--|\(END\)|press q to quit|:\s*$", re.I), 0.15, "pager"),
]

# cwd / prompt line, e.g. "user@host:~/proj$ ", "(venv) bryan@mac proj %", "/path #"
_SHELL_PROMPT = re.compile(
    r"(?:\((?P<venv>[^)]+)\)\s*)?"
    r"(?:(?P<user>[\w.-]+)@(?P<host>[\w.-]+)[:\s]+)?"
    r"(?P<cwd>(?:~|/)[\w./~ -]*)?\s*"
    r"(?P<sigil>[#$%❯➜])\s*(?P<cmd>.*)?$"
)


def _line_threat(line: str) -> Tuple[float, str]:
    threat, label = 0.0, ""
    for pat, t, lbl in _THREAT_PATTERNS:
        if pat.search(line):
            if t > threat:
                threat, label = t, lbl
    return threat, label


class GeneralTerminalParser(EnvironmentParser):
    """Parses arbitrary terminal text into a shared EnvironmentState with grounded threat."""

    def __init__(self, *, max_entities: int = 40) -> None:
        self._max_entities = max_entities

    def parse(self, raw_input: str, *, context_id: str = "terminal") -> EnvironmentState:
        text = raw_input if isinstance(raw_input, str) else str(raw_input)
        lines = text.splitlines()
        non_empty = [ln for ln in lines if ln.strip()]

        entities: List[Dict[str, object]] = []
        active_prompts: List[str] = []
        messages: List[str] = []
        max_line_threat = 0.0

        self_state: Dict[str, object] = {
            "line_count": len(lines),
            "char_count": len(text),
        }

        # Threat-scan every line; keep the notable ones as entities (NetHack-style).
        for idx, line in enumerate(lines):
            threat, label = _line_threat(line)
            if threat > 0.0:
                max_line_threat = max(max_line_threat, threat)
                if len(entities) < self._max_entities:
                    entities.append({
                        "type": "terminal_line",
                        "label": label,
                        "line_no": idx,
                        "text": line.strip()[:200],
                        "tags": ["threat"] if threat >= 0.6 else ["notice"],
                        "threat_score": round(threat, 3),
                        "hostile": threat >= 0.6,
                    })

        # Prompt detection on the last few non-empty lines (that's where the cursor is).
        prompt_risk = 0.0
        for line in non_empty[-3:]:
            for pat, risk, label in _PROMPT_PATTERNS:
                if pat.search(line):
                    active_prompts.append(label)
                    prompt_risk = max(prompt_risk, risk)

        # cwd / shell / current command from the last shell-prompt line.
        for line in reversed(non_empty[-5:] if len(non_empty) >= 5 else non_empty):
            m = _SHELL_PROMPT.search(line)
            if m and (m.group("sigil") or m.group("cwd")):
                if m.group("cwd"):
                    self_state["cwd"] = m.group("cwd").strip()
                if m.group("user"):
                    self_state["user"] = m.group("user")
                if m.group("host"):
                    self_state["host"] = m.group("host")
                if m.group("venv"):
                    self_state["venv"] = m.group("venv")
                cmd = (m.group("cmd") or "").strip()
                if cmd:
                    self_state["current_command"] = cmd[:200]
                    # a dangerous command typed at the prompt is itself a threat
                    cmd_threat, cmd_label = _line_threat(cmd)
                    if cmd_threat > 0:
                        max_line_threat = max(max_line_threat, cmd_threat)
                break

        # Last meaningful output line as the headline message.
        if non_empty:
            messages.append(non_empty[-1].strip()[:200])

        frame_threat = max(max_line_threat, prompt_risk)
        self_state["threat_score"] = round(frame_threat, 3)
        self_state["prompt_blocking"] = bool(active_prompts)

        confidence = 0.4 if not non_empty else 0.9
        return EnvironmentState(
            domain="terminal",
            context_id=context_id,
            confidence=confidence,
            self_state=self_state,
            messages=messages,
            entities=entities,
            active_prompts=active_prompts,
            uncertainty={"parse": round(1.0 - confidence, 3)},
            modalities={"text_lines": len(lines)},
        )

    def threat_score(self, raw_input: str) -> float:
        """Convenience: frame-level terminal threat in [0,1] without the full state."""
        return float(self.parse(raw_input).self_state.get("threat_score", 0.0))
