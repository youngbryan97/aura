"""Objective-specific effect contracts for governed macOS automation.

The transport receipt from ``osascript`` proves that a script exited. It does
not prove that the desktop objective happened. This module turns bounded
natural-language objectives into explicit observable requirements and compares
pre/post desktop snapshots against those requirements.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum


class EffectKind(StrEnum):
    """Observable desktop effects supported by the deterministic verifier."""

    APP_FRONTMOST = "app_frontmost"
    APP_NOT_RUNNING = "app_not_running"
    WINDOW_REGION = "window_region"
    WINDOW_MINIMIZED = "window_minimized"
    WINDOW_GEOMETRY_CHANGED = "window_geometry_changed"
    TEXT_VISIBLE = "text_visible"
    BROWSER_URL_CONTAINS = "browser_url_contains"
    CALCULATION_RESULT = "calculation_result"
    INTERACTION_CHANGED_VISIBLE_STATE = "interaction_changed_visible_state"
    FILE_EXISTS = "file_exists"
    FILE_NONEMPTY = "file_nonempty"
    FILE_CONTAINS = "file_contains"
    # LIVE, 2026-08-10: "Put the text ORION-7 on my clipboard" was refused with
    # "the objective has no complete observable acceptance contract" — because
    # no clipboard effect kind existed, so a clipboard goal could never carry a
    # strong required check and was unverifiable by construction. The
    # observation already read the clipboard; nothing could ask about it.
    CLIPBOARD_CONTAINS = "clipboard_contains"


_APP_ALIASES = {
    "arc": "Arc",
    "brave": "Brave Browser",
    "brave browser": "Brave Browser",
    "browser": "browser",
    "calculator": "Calculator",
    "chrome": "Google Chrome",
    "default browser": "browser",
    "finder": "Finder",
    "firefox": "Firefox",
    "google chrome": "Google Chrome",
    "microsoft edge": "Microsoft Edge",
    "microsoft word": "Microsoft Word",
    "note": "Notes",
    "notes": "Notes",
    "pages": "Pages",
    "preview": "Preview",
    "safari": "Safari",
    "textedit": "TextEdit",
    "web browser": "browser",
}
_GENERIC_APP_TARGETS = frozenset(
    {
        "any",
        "application",
        "app",
        "current",
        "default",
        "document",
        "existing",
        "new",
        "page",
        "some",
        "tab",
        "target",
        "visible",
        "window",
    }
)
_WEB_SURFACE_TARGETS = frozenset(
    {
        "google doc",
        "google docs",
        "google drive",
        "google presentation",
        "google presentations",
        "google sheet",
        "google sheets",
        "google slide",
        "google slides",
        "google spreadsheet",
        "google spreadsheets",
    }
)
_UNVERIFIED_CONTROL_OPERATIONS = (
    (
        re.compile(r"\b(?:delete|erase|remove|trash)\b", re.IGNORECASE),
        "deletion lacks a durable target-specific postcondition",
    ),
    (
        re.compile(r"\b(?:save|export|download|upload|print)\b", re.IGNORECASE),
        "persistence or transfer lacks a durable artifact postcondition",
    ),
    (
        re.compile(r"\b(?:email|message|post|publish|send)\b", re.IGNORECASE),
        "external communication lacks recipient and delivery verification",
    ),
    (
        re.compile(r"\b(?:install|uninstall)\b", re.IGNORECASE),
        "software installation is outside the desktop automation authority",
    ),
    (
        # Not when the destination is the clipboard. "copy 'hello' to my
        # clipboard" mutates no filesystem and has an exact postcondition —
        # the text is on the clipboard — but the bare verb read as a file
        # copy and made the objective permanently unverifiable.
        re.compile(
            r"\b(?:copy|move|rename)\b(?![^.?!]{0,60}\b(?:clip\s?board|pasteboard)\b)",
            re.IGNORECASE,
        ),
        "filesystem mutation lacks source, destination, and artifact verification",
    ),
    (
        re.compile(r"\b(?:log\s*in|sign\s*in|password|purchase|pay)\b", re.IGNORECASE),
        "authentication or transaction workflows require a dedicated governed contract",
    ),
)


_HOME_ANCHORS = {
    "desktop": "~/Desktop",
    "documents": "~/Documents",
    "downloads": "~/Downloads",
    "home folder": "~",
    "home directory": "~",
}
_FILENAME_RE = re.compile(r"(?<![\w./~-])([\w-][\w.-]{0,60}\.[A-Za-z0-9]{1,8})(?![\w])")
_EXPLICIT_PATH_RE = re.compile(r"(?<![\w])((?:~|/)[\w./-]*[\w.])")
# Every verb here is one that gives a sentence a file to point at. The
# modification verbs were missing, so "add the line X to the end of notes.txt"
# named no path at all and the objective fell through to a generic summary
# folder — the file the request was entirely about was never located.
_FILE_INTENT_RE = re.compile(
    r"\b(?:file|folder|script|program|app|document|note|report|save|write|create|"
    r"place|put|generate|build|export|drop|"
    r"append|prepend|add|attach|insert|edit|update|modify|amend)\b",
    re.IGNORECASE,
)
_FILE_CONTENT_INTENT_RE = re.compile(
    r"\b(?:containing|contains|with|write|writing|wrote|text|content|contents|"
    r"sentence|line|code|says?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FileFact:
    """What is true of one path on disk, read without touching it."""

    path: str
    exists: bool = False
    size: int = 0
    excerpt: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "exists": self.exists,
            "size": self.size,
            "excerpt": self.excerpt[:400],
            "error": self.error,
        }


def observe_paths(paths: Sequence[str]) -> tuple[FileFact, ...]:
    """Stat and excerpt each path. Read-only, bounded, never raises.

    A file on disk is the strongest desktop postcondition available — far
    stronger than any pixel — and it costs a stat call. Text is excerpted so a
    content requirement can be checked without loading an arbitrary artifact
    into memory.
    """
    import os

    facts: list[FileFact] = []
    for raw in dict.fromkeys(str(item or "").strip() for item in paths):
        if not raw:
            continue
        try:
            expanded = os.path.expanduser(raw)
            if not os.path.exists(expanded):
                facts.append(FileFact(path=raw, exists=False))
                continue
            size = int(os.path.getsize(expanded)) if os.path.isfile(expanded) else 0
            excerpt = ""
            if os.path.isfile(expanded) and size <= 4_000_000:
                try:
                    with open(expanded, encoding="utf-8", errors="replace") as handle:
                        excerpt = handle.read(8000)
                except OSError as exc:
                    facts.append(
                        FileFact(path=raw, exists=True, size=size, error=type(exc).__name__)
                    )
                    continue
            facts.append(FileFact(path=raw, exists=True, size=size, excerpt=excerpt))
        except (OSError, ValueError) as exc:
            facts.append(FileFact(path=raw, error=type(exc).__name__))
    return tuple(facts)


@dataclass(frozen=True)
class DesktopSnapshot:
    """Bounded observable desktop state captured without mutating the host."""

    frontmost_app: str = ""
    frontmost_window: str = ""
    window_frame: tuple[int, int, int, int] | None = None
    desktop_frame: tuple[int, int, int, int] | None = None
    window_minimized: bool | None = None
    focused_value_excerpt: str = ""
    browser_url: str = ""
    screen_text: str = ""
    clipboard_excerpt: str = ""
    running_apps: tuple[str, ...] = ()
    files: tuple[FileFact, ...] = ()

    def file_fact(self, path: str) -> FileFact | None:
        wanted = str(path or '').strip()
        for fact in self.files:
            if fact.path == wanted:
                return fact
        return None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> DesktopSnapshot:
        source = value or {}
        running_raw = source.get("running_apps", ())
        if isinstance(running_raw, str):
            running_apps = tuple(
                item.strip()
                for item in re.split(r"[,\n]", running_raw)
                if item.strip()
            )
        elif isinstance(running_raw, Sequence):
            running_apps = tuple(str(item).strip() for item in running_raw if str(item).strip())
        else:
            running_apps = ()
        return cls(
            frontmost_app=_bounded_text(source.get("frontmost_app"), 160),
            frontmost_window=_bounded_text(source.get("frontmost_window"), 240),
            window_frame=_coerce_frame(
                source.get("window_frame") or source.get("frontmost_window_bounds")
            ),
            desktop_frame=_coerce_frame(source.get("desktop_frame") or source.get("desktop_bounds")),
            window_minimized=_coerce_optional_bool(source.get("window_minimized")),
            focused_value_excerpt=_bounded_text(source.get("focused_value_excerpt"), 1200),
            browser_url=_bounded_text(source.get("browser_url"), 2000),
            screen_text=_bounded_text(source.get("screen_text"), 4000),
            clipboard_excerpt=_bounded_text(source.get("clipboard_excerpt"), 1200),
            running_apps=running_apps,
            files=_coerce_file_facts(source.get("files")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "frontmost_app": self.frontmost_app,
            "frontmost_window": self.frontmost_window,
            "window_frame": list(self.window_frame) if self.window_frame else None,
            "desktop_frame": list(self.desktop_frame) if self.desktop_frame else None,
            "window_minimized": self.window_minimized,
            "focused_value_excerpt": self.focused_value_excerpt,
            "browser_url": self.browser_url,
            "screen_text": self.screen_text,
            "clipboard_excerpt": self.clipboard_excerpt,
            "running_apps": list(self.running_apps),
            "files": [fact.to_dict() for fact in self.files],
        }

    def visible_fingerprint(self) -> str:
        payload = {
            "app": self.frontmost_app,
            "window": self.frontmost_window,
            "frame": self.window_frame,
            "minimized": self.window_minimized,
            "focus": _normalize_text(self.focused_value_excerpt),
            "url": self.browser_url,
            "screen": _normalize_text(self.screen_text),
            "running": sorted(_normalize_app_name(app) for app in self.running_apps),
            "files": sorted(
                (fact.path, fact.exists, fact.size) for fact in self.files
            ),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EffectRequirement:
    requirement_id: str
    kind: EffectKind
    expected: str
    description: str
    required: bool = True
    strong: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "requirement_id": self.requirement_id,
            "kind": self.kind.value,
            "expected": self.expected,
            "description": self.description,
            "required": self.required,
            "strong": self.strong,
        }


@dataclass(frozen=True)
class EffectContract:
    contract_id: str
    goal: str
    requirements: tuple[EffectRequirement, ...]
    unsupported_reasons: tuple[str, ...] = ()

    @property
    def verifiable(self) -> bool:
        return (
            not self.unsupported_reasons
            and any(requirement.required and requirement.strong for requirement in self.requirements)
        )

    @property
    def needs_clipboard(self) -> bool:
        """Whether the snapshot has to read the clipboard back.

        LIVE, 2026-08-10: the deterministic script SET the clipboard, the text
        was really there, and the check failed anyway — nothing had captured
        clipboard_excerpt, so the verifier compared its expectation against an
        empty string and went looking for a repair. An effect that happened and
        cannot be observed is indistinguishable from one that did not.
        """
        return any(
            requirement.kind is EffectKind.CLIPBOARD_CONTAINS
            for requirement in self.requirements
        )

    @property
    def needs_screen_text(self) -> bool:
        return any(
            requirement.kind
            in {
                EffectKind.TEXT_VISIBLE,
                EffectKind.CALCULATION_RESULT,
                EffectKind.INTERACTION_CHANGED_VISIBLE_STATE,
            }
            for requirement in self.requirements
        )

    @property
    def observed_paths(self) -> tuple[str, ...]:
        """Paths whose on-disk state this contract is judged against."""
        return tuple(
            dict.fromkeys(
                requirement.expected.split("::", 1)[0]
                for requirement in self.requirements
                if requirement.kind
                in {EffectKind.FILE_EXISTS, EffectKind.FILE_NONEMPTY, EffectKind.FILE_CONTAINS}
            )
        )

    @property
    def needs_browser_url(self) -> bool:
        return any(
            requirement.kind == EffectKind.BROWSER_URL_CONTAINS
            for requirement in self.requirements
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_id": self.contract_id,
            "goal": self.goal,
            "verifiable": self.verifiable,
            "requirements": [requirement.to_dict() for requirement in self.requirements],
            "unsupported_reasons": list(self.unsupported_reasons),
        }


@dataclass(frozen=True)
class EffectCheck:
    requirement_id: str
    kind: EffectKind
    passed: bool
    expected: str
    observed: str
    detail: str
    required: bool
    strong: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "requirement_id": self.requirement_id,
            "kind": self.kind.value,
            "passed": self.passed,
            "expected": self.expected,
            "observed": self.observed,
            "detail": self.detail,
            "required": self.required,
            "strong": self.strong,
        }


@dataclass(frozen=True)
class EffectVerdict:
    contract_id: str
    verified: bool
    checks: tuple[EffectCheck, ...]
    failure_reasons: tuple[str, ...]

    @property
    def evidence(self) -> tuple[str, ...]:
        return tuple(check.detail for check in self.checks if check.passed and check.detail)

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_id": self.contract_id,
            "verified": self.verified,
            "checks": [check.to_dict() for check in self.checks],
            "evidence": list(self.evidence),
            "failure_reasons": list(self.failure_reasons),
        }


def extract_target_paths(
    goal: str,
    *,
    require_file_intent: bool = True,
) -> tuple[str, ...]:
    """Paths a desktop objective names, as a user would name them.

    "on my Desktop called aura_hello.txt" and "~/Desktop/2048.py" both resolve
    to one concrete path. Without this, an objective whose entire point is a
    file had no observable postcondition at all, and the automation lane
    refused it as unverifiable — the exact refusal seen live 2026-07-27 for
    "create a file on my Desktop called aura_hello.txt".
    """
    text = " ".join(str(goal or "").split())
    if not text or (require_file_intent and not _FILE_INTENT_RE.search(text)):
        return ()

    paths: list[str] = []
    for match in _EXPLICIT_PATH_RE.finditer(text):
        candidate = match.group(1).strip().rstrip(".,;:")
        # A bare "/" or a sentence fragment is not a path; require a real segment.
        if len(candidate) > 2 and ("/" in candidate.lstrip("~/") or candidate.startswith("~/")):
            paths.append(candidate)
    if paths:
        return tuple(dict.fromkeys(paths))

    lowered = text.lower()
    anchor_dir = ""
    for keyword, target in _HOME_ANCHORS.items():
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            anchor_dir = target
            break
    if not anchor_dir:
        return ()
    for match in _FILENAME_RE.finditer(text):
        name = match.group(1).strip()
        # "2048.py" yes; "e.g" and version numbers no.
        if re.fullmatch(r"\d+\.\d+", name):
            continue
        paths.append(f"{anchor_dir}/{name}")
    return tuple(dict.fromkeys(paths))



_CLIPBOARD_GOAL_RE = re.compile(r"\b(?:clip\s?board|pasteboard)\b", re.IGNORECASE)
#: The literal a person put in quotes, or an unquoted token that reads like an
#: identifier — "put ORION-7 on my clipboard" names its own acceptance test.
_CLIPBOARD_LITERAL_RE = re.compile(
    r"[\"'\u201c\u2018](?P<quoted>[^\"'\u201d\u2019]{1,200})[\"'\u201d\u2019]"
    r"|\b(?:text|string|value|word|token)\s+(?P<named>[\w.\-/]{2,80})"
    # A bare identifier-shaped token. "put ORION-7 on my clipboard" names its
    # payload without quoting it or saying the word "text", and requiring
    # either produced NO acceptance criterion at all — so the contract had
    # nothing to verify and the snapshot never read the clipboard back.
    r"|\b(?P<bare>[A-Z0-9][A-Z0-9]*(?:[-_][A-Z0-9]+)+)\b",
)


def _clipboard_payload(goal: str, text_payload: str) -> str:
    """The exact text a clipboard objective asks to be there afterwards."""

    if not _CLIPBOARD_GOAL_RE.search(goal or ""):
        return ""
    explicit = " ".join(str(text_payload or "").split())
    if explicit:
        return explicit[:200]
    match = _CLIPBOARD_LITERAL_RE.search(goal or "")
    if match is None:
        return ""
    return (
        match.group("quoted") or match.group("named") or match.group("bare") or ""
    ).strip()[:200]


def build_effect_contract(
    goal: str,
    *,
    text_payload: str = "",
    expected_url: str = "",
    target_apps: Sequence[str] | None = None,
) -> EffectContract:
    """Compile a bounded desktop objective into causal acceptance criteria."""

    normalized_goal = " ".join(str(goal or "").split())
    lowered = normalized_goal.lower()
    requirements: list[EffectRequirement] = []
    unsupported: list[str] = []
    seen: set[tuple[EffectKind, str]] = set()

    def add(kind: EffectKind, expected: str, description: str) -> None:
        key = (kind, _normalize_text(expected))
        if not expected.strip() or key in seen:
            return
        seen.add(key)
        requirements.append(
            EffectRequirement(
                requirement_id=f"effect-{len(requirements) + 1}",
                kind=kind,
                expected=expected.strip(),
                description=description,
            )
        )

    apps = tuple(
        extract_target_apps(normalized_goal)
        if target_apps is None
        else target_apps
    )
    quit_targets = _extract_action_targets(normalized_goal, ("quit",))
    close_targets = _extract_action_targets(normalized_goal, ("close",))
    for app in apps:
        if any(_apps_match(app, target) for target in quit_targets):
            add(
                EffectKind.APP_NOT_RUNNING,
                app,
                f"{app} is no longer a running visible application.",
            )
        elif any(_apps_match(app, target) for target in close_targets):
            continue
        else:
            add(
                EffectKind.APP_FRONTMOST,
                app,
                f"{app} is the foreground application after the action.",
            )

    if re.search(r"\bminimi[sz](?:e|ed|ing)?\b", lowered):
        add(EffectKind.WINDOW_MINIMIZED, "true", "The targeted window is minimized.")
    elif re.search(r"\bmaximi[sz](?:e|ed|ing)?\b", lowered):
        add(EffectKind.WINDOW_REGION, "maximized", "The targeted window fills the usable desktop.")
    elif re.search(r"\b(?:arrange|resize|drag|organize|tile|snap)\b", lowered):
        region = ""
        for candidate in ("left", "right", "top", "bottom"):
            if re.search(rf"\b{candidate}\b", lowered):
                region = candidate
                break
        if region:
            requires_half_region = bool(
                re.search(r"\b(?:resize|tile|snap)\b", lowered)
                or re.search(rf"\b{region}\s+(?:hand\s+)?side\b", lowered)
            )
            expected_region = f"{region}_half" if requires_half_region else region
            add(
                EffectKind.WINDOW_REGION,
                expected_region,
                f"The targeted window occupies the {expected_region.replace('_', ' ')} region of the desktop.",
            )
        else:
            add(
                EffectKind.WINDOW_GEOMETRY_CHANGED,
                "changed",
                "The targeted window geometry differs from its pre-action geometry.",
            )

    requests_text = bool(
        re.search(r"\b(?:type|paste|write|fill|insert|compose|draft|enter)\b", lowered)
        or "google docs" in lowered
    )
    resolved_text = str(text_payload or "").strip() or _extract_inline_text(normalized_goal)
    if requests_text:
        witness = _text_witness(resolved_text)
        if witness:
            add(
                EffectKind.TEXT_VISIBLE,
                witness,
                "The requested text is visibly present in the target editing surface.",
            )
        else:
            unsupported.append("requested text entry has no concrete text payload to verify")

    resolved_url = str(expected_url or "").strip() or _extract_explicit_url(normalized_goal)
    requests_search = bool(re.search(r"\b(?:search|google|look\s+up)\b", lowered))
    if resolved_url:
        add(
            EffectKind.BROWSER_URL_CONTAINS,
            resolved_url,
            "The active browser URL corresponds to the requested destination or search.",
        )
    elif requests_search:
        unsupported.append("browser search has no concrete destination or query to verify")

    target_paths = extract_target_paths(normalized_goal)
    wants_content = bool(_FILE_CONTENT_INTENT_RE.search(lowered))
    for path in target_paths:
        add(
            EffectKind.FILE_EXISTS,
            path,
            f"{path} exists on disk after the action.",
        )
        if wants_content:
            add(
                EffectKind.FILE_NONEMPTY,
                path,
                f"{path} has content, not just an empty file.",
            )
    witness_for_file = _text_witness(resolved_text) if target_paths and resolved_text else ""
    if witness_for_file and len(witness_for_file) >= 4:
        add(
            EffectKind.FILE_CONTAINS,
            f"{target_paths[0]}::{witness_for_file}",
            f"{target_paths[0]} contains the requested text.",
        )

    # A clipboard objective names the text it wants there, so the acceptance
    # criterion is exact: that text is on the clipboard afterwards.
    clipboard_payload = _clipboard_payload(normalized_goal, text_payload)
    if clipboard_payload:
        add(
            EffectKind.CLIPBOARD_CONTAINS,
            clipboard_payload,
            f"The clipboard contains {clipboard_payload[:80]} after the action.",
        )

    calculation_result = _calculation_result(normalized_goal)
    if calculation_result:
        add(
            EffectKind.CALCULATION_RESULT,
            calculation_result,
            f"The visible calculator result equals {calculation_result}.",
        )

    requests_interaction = bool(
        re.search(r"\b(?:click|press|choose|select)\b", lowered)
    )
    if requests_interaction and not calculation_result:
        interaction_target = _extract_interaction_target(normalized_goal)
        if interaction_target:
            add(
                EffectKind.INTERACTION_CHANGED_VISIBLE_STATE,
                interaction_target,
                "The named visible control is present and the interaction changes observable UI state.",
            )
        else:
            unsupported.append("requested interaction has no concrete visible target to verify")

    if close_targets:
        unsupported.append(
            "closing an app window lacks targeted window-closure observation; "
            "use an explicit quit objective for process termination"
        )
    if target_paths:
        unsupported = [
            reason
            for reason in unsupported
            if reason
            not in {
                "requested interaction has no concrete visible target to verify",
                "requested text entry has no concrete text payload to verify",
            }
        ]
    control_caveats = _unsupported_control_operations(normalized_goal)
    if target_paths:
        # These caveats say "we cannot see the artifact". Now we can: the
        # requirement above stats the exact path. Deletion stays unsupported —
        # absence is not the same evidence as presence.
        control_caveats = [
            reason
            for reason in control_caveats
            if "durable artifact postcondition" not in reason
            and "source, destination, and artifact verification" not in reason
        ]
    unsupported.extend(control_caveats)

    if not requirements:
        unsupported.append("objective has no supported observable desktop postcondition")

    deduped_unsupported = tuple(dict.fromkeys(unsupported))
    digest_payload = {
        "goal": normalized_goal,
        "requirements": [requirement.to_dict() for requirement in requirements],
        "unsupported": deduped_unsupported,
    }
    contract_id = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    return EffectContract(
        contract_id=contract_id,
        goal=normalized_goal,
        requirements=tuple(requirements),
        unsupported_reasons=deduped_unsupported,
    )


def evaluate_effect_contract(
    contract: EffectContract,
    before: DesktopSnapshot,
    after: DesktopSnapshot,
) -> EffectVerdict:
    """Evaluate every required effect and require strong causal evidence."""

    checks = tuple(
        _evaluate_requirement(requirement, before, after)
        for requirement in contract.requirements
    )
    required_passed = all(
        check.passed
        for requirement, check in zip(contract.requirements, checks, strict=True)
        if requirement.required
    )
    strong_passed = any(check.passed and check.strong for check in checks)
    verified = contract.verifiable and required_passed and strong_passed
    failures = list(contract.unsupported_reasons)
    failures.extend(check.detail for check in checks if not check.passed)
    if not strong_passed and not failures:
        failures.append("no strong objective-specific effect was observed")
    return EffectVerdict(
        contract_id=contract.contract_id,
        verified=verified,
        checks=checks,
        failure_reasons=tuple(failures),
    )


def _evaluate_requirement(
    requirement: EffectRequirement,
    before: DesktopSnapshot,
    after: DesktopSnapshot,
) -> EffectCheck:
    kind = requirement.kind
    expected = requirement.expected
    passed = False
    observed = ""
    detail = ""

    if kind in {EffectKind.FILE_EXISTS, EffectKind.FILE_NONEMPTY, EffectKind.FILE_CONTAINS}:
        path, _, witness = expected.partition("::")
        after_fact = after.file_fact(path)
        before_fact = before.file_fact(path)
        if after_fact is None:
            return EffectCheck(
                requirement_id=requirement.requirement_id,
                kind=kind,
                passed=False,
                expected=expected,
                observed="",
                detail=f"{path} was never observed",
                required=requirement.required,
                strong=False,
            )
        observed = f"exists={after_fact.exists};size={after_fact.size}"
        if kind == EffectKind.FILE_EXISTS:
            passed = after_fact.exists
            detail = f"{path} exists" if passed else f"{path} does not exist"
        elif kind == EffectKind.FILE_NONEMPTY:
            passed = after_fact.exists and after_fact.size > 0
            detail = (
                f"{path} has {after_fact.size} bytes"
                if passed
                else f"{path} is missing or empty"
            )
        else:
            passed = _normalize_text(witness) in _normalize_text(after_fact.excerpt)
            detail = (
                f"{path} contains {witness!r}"
                if passed
                else f"{path} does not contain {witness!r}"
            )
        # Causal strength: the objective made this true, rather than finding it
        # true. An unchanged file is honest evidence of nothing.
        changed = before_fact is None or (
            before_fact.exists != after_fact.exists
            or before_fact.size != after_fact.size
            or before_fact.excerpt != after_fact.excerpt
        )
        return EffectCheck(
            requirement_id=requirement.requirement_id,
            kind=kind,
            passed=passed,
            expected=expected,
            observed=observed,
            detail=detail,
            required=requirement.required,
            strong=bool(requirement.strong and changed),
        )

    if kind == EffectKind.APP_FRONTMOST:
        observed = after.frontmost_app
        passed = _apps_match(expected, observed)
        detail = (
            f"frontmost_app={observed}"
            if passed
            else f"expected frontmost app {expected!r}; observed {observed or 'none'!r}"
        )
    elif kind == EffectKind.APP_NOT_RUNNING:
        observed = ", ".join(after.running_apps)
        passed = bool(after.running_apps) and not any(
            _apps_match(expected, running) for running in after.running_apps
        )
        detail = (
            f"app_not_running={expected}"
            if passed
            else f"expected {expected!r} absent from running apps; observed {observed or 'unknown'}"
        )
    elif kind == EffectKind.WINDOW_MINIMIZED:
        observed = str(after.window_minimized)
        passed = after.window_minimized is True and before.window_minimized is not True
        detail = (
            "window_minimized=true"
            if passed
            else f"expected a newly minimized window; observed {observed}"
        )
    elif kind == EffectKind.WINDOW_GEOMETRY_CHANGED:
        observed = _format_frame(after.window_frame)
        passed = (
            before.window_frame is not None
            and after.window_frame is not None
            and before.window_frame != after.window_frame
        )
        detail = (
            f"window_frame={observed};previous={_format_frame(before.window_frame)}"
            if passed
            else "window geometry did not observably change"
        )
    elif kind == EffectKind.WINDOW_REGION:
        observed = _format_frame(after.window_frame)
        passed = _window_matches_region(after.window_frame, after.desktop_frame, expected)
        detail = (
            f"window_region={expected};window_frame={observed};desktop_frame={_format_frame(after.desktop_frame)}"
            if passed
            else (
                f"expected window region {expected!r}; window={observed or 'unknown'}; "
                f"desktop={_format_frame(after.desktop_frame) or 'unknown'}"
            )
        )
    elif kind == EffectKind.CLIPBOARD_CONTAINS:
        observed = _bounded_text(after.clipboard_excerpt, 260)
        passed = _contains_normalized(after.clipboard_excerpt, expected)
        detail = (
            f"clipboard_contains={expected[:120]}"
            if passed
            else "requested text was not on the clipboard after the action"
        )
    elif kind == EffectKind.TEXT_VISIBLE:
        visible_text = "\n".join(
            value
            for value in (after.focused_value_excerpt, after.screen_text)
            if value
        )
        observed = _bounded_text(visible_text, 260)
        passed = _contains_normalized(visible_text, expected)
        detail = (
            f"visible_text_contains={expected[:120]}"
            if passed
            else "requested text was not visible in focused-value or screen read-back"
        )
    elif kind == EffectKind.BROWSER_URL_CONTAINS:
        observed = after.browser_url
        passed = _url_corresponds(after.browser_url, expected)
        detail = (
            f"browser_url={observed[:300]}"
            if passed
            else f"expected browser destination {expected!r}; observed {observed or 'none'!r}"
        )
    elif kind == EffectKind.CALCULATION_RESULT:
        display_text = after.focused_value_excerpt or after.screen_text
        observed = _bounded_text(display_text, 260)
        passed = _contains_number(display_text, expected)
        detail = (
            f"calculation_result={expected}"
            if passed
            else f"expected visible calculation result {expected}; observed {observed or 'none'}"
        )
    elif kind == EffectKind.INTERACTION_CHANGED_VISIBLE_STATE:
        pre_visible = "\n".join(
            value for value in (before.focused_value_excerpt, before.screen_text) if value
        )
        post_visible = "\n".join(
            value for value in (after.focused_value_excerpt, after.screen_text) if value
        )
        target_seen = _contains_normalized(pre_visible, expected) or _contains_normalized(
            post_visible, expected
        )
        changed = before.visible_fingerprint() != after.visible_fingerprint()
        observed = f"target_seen={target_seen};visible_state_changed={changed}"
        passed = target_seen and changed
        detail = (
            f"interaction_target={expected};visible_state_changed=true"
            if passed
            else (
                f"interaction target {expected!r} was not causally verified "
                f"(target_seen={target_seen}, visible_state_changed={changed})"
            )
        )
    else:  # pragma: no cover - exhaustive defensive guard
        detail = f"unsupported effect kind: {kind.value}"

    return EffectCheck(
        requirement_id=requirement.requirement_id,
        kind=kind,
        passed=passed,
        expected=expected,
        observed=observed,
        detail=detail,
        required=requirement.required,
        strong=requirement.strong,
    )


def _bounded_text(value: object, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _coerce_optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    lowered = str(value or "").strip().lower()
    if lowered in {"true", "yes", "1"}:
        return True
    if lowered in {"false", "no", "0"}:
        return False
    return None


def _coerce_frame(value: object) -> tuple[int, int, int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        raw_values = value
    else:
        matches = re.findall(r"-?\d+", str(value or ""))
        if len(matches) != 4:
            return None
        raw_values = matches
    try:
        frame = tuple(int(float(item)) for item in raw_values)
    except (TypeError, ValueError):
        return None
    return (frame[0], frame[1], frame[2], frame[3])


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _contains_normalized(haystack: str, needle: str) -> bool:
    normalized_needle = _normalize_text(needle)
    return bool(normalized_needle) and normalized_needle in _normalize_text(haystack)


def _normalize_app_name(value: str) -> str:
    normalized = _normalize_text(value)
    if normalized.endswith(".app"):
        normalized = normalized[:-4]
    aliases = {
        "chrome": "google chrome",
        "microsoft edge": "edge",
    }
    return aliases.get(normalized, normalized)


def _apps_match(expected: str, observed: str) -> bool:
    expected_name = _normalize_app_name(expected)
    observed_name = _normalize_app_name(observed)
    if expected_name == "browser":
        return observed_name in {
            "arc",
            "brave browser",
            "edge",
            "firefox",
            "google chrome",
            "safari",
        }
    return bool(expected_name and observed_name) and expected_name == observed_name


def extract_target_apps(goal: str) -> tuple[str, ...]:
    """Extract only concrete app targets from objective-control language."""
    normalized_goal = " ".join(str(goal or "").split())
    lowered = normalized_goal.casefold()
    apps: list[str] = []
    for target in _extract_action_targets(
        normalized_goal,
        ("open", "launch", "switch to", "focus", "close", "quit"),
    ):
        if target not in apps:
            apps.append(target)

    phrase_prefix = r"(?:current|in|inside|into|the|using|with)\s+(?:the\s+)?"
    for alias in sorted(_APP_ALIASES, key=len, reverse=True):
        canonical = _APP_ALIASES[alias]
        phrase = rf"\b{phrase_prefix}{re.escape(alias)}\b"
        window_phrase = rf"\b{re.escape(alias)}\s+(?:app|application|window)\b"
        if (
            re.search(phrase, lowered)
            or re.search(window_phrase, lowered)
        ) and canonical not in apps:
            apps.append(canonical)

    if "browser" in apps and any(
        app in apps
        for app in (
            "Arc",
            "Brave Browser",
            "Firefox",
            "Google Chrome",
            "Microsoft Edge",
            "Safari",
        )
    ):
        apps.remove("browser")
    return tuple(apps[:5])


def extract_direct_application_targets(goal: str) -> tuple[str, ...]:
    """Return application referents that a lifecycle directive acts on.

    The broad effect-contract extractor above is intentionally permissive once
    a request has entered desktop execution. Routing needs a stronger claim:
    the direct object must identify an application, including one that is not
    installed yet. Explicit app nouns, known application aliases, and proper
    product names provide that evidence. Ordinary objects such as ``your
    mind`` or ``the gap`` do not.

    This keeps the route and executor on one target grammar while preserving
    the useful failure for an unknown product name: execution can report that
    no installed application matches it instead of turning the command into a
    web search.
    """

    candidates: list[str] = []
    for _action, target, explicitly_typed in _extract_action_target_records(
        goal,
        ("open", "launch", "switch to", "focus", "close", "quit"),
    ):
        lowered = target.casefold()
        known_application = lowered in _APP_ALIASES or target in _APP_ALIASES.values()
        if not (explicitly_typed or known_application or _looks_like_product_name(target)):
            continue
        if target not in candidates:
            candidates.append(target)
    return tuple(candidates[:5])


def _looks_like_product_name(value: str) -> bool:
    """Whether a direct object has the orthography of a named product."""

    candidate = str(value or "").strip()
    words = re.findall(r"[A-Za-z][A-Za-z0-9._-]*", candidate)
    if not words:
        return False
    if any(word.isupper() and len(word) > 1 for word in words):
        return True
    if any(any(char.isupper() for char in word[1:]) for word in words):
        return True
    return all(word[0].isupper() for word in words)


def _extract_action_targets(goal: str, actions: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            target
            for _action, target, _explicitly_typed in _extract_action_target_records(
                goal, actions
            )
        )
    )


def _extract_action_target_records(
    goal: str,
    actions: Sequence[str],
) -> tuple[tuple[str, str, bool], ...]:
    action_pattern = "|".join(re.escape(action) for action in actions)
    pattern = (
        rf"\b(?P<action>{action_pattern})\s+(?:up\s+)?"
        r"(?:a\s+|an\s+|my\s+|the\s+)?"
        r"(?P<type_prefix>(?:app|application)\s+(?:(?:called|named)\s+)?)?"
        r"(?P<target>[A-Za-z][A-Za-z0-9 &._-]{1,60}?)"
        r"(?:\s+(?P<type_suffix>app|application))?"
        r"(?=\s*(?:,|\.|;|!|\?|\b(?:and|then|before|after|for|to)\b|$))"
    )
    records: list[tuple[str, str, bool]] = []
    for match in re.finditer(pattern, goal, flags=re.IGNORECASE):
        candidate = _normalize_app_target(match.group("target"))
        if not candidate:
            continue
        record = (
            " ".join(match.group("action").casefold().split()),
            candidate,
            bool(match.group("type_prefix") or match.group("type_suffix")),
        )
        if record not in records:
            records.append(record)
    return tuple(records)


def canonical_app_target(value: str) -> str:
    """Return the installed-app identity implied by ordinary user wording.

    Planning and execution both use this boundary. That prevents a planner's
    harmless singular/plural variation (for example ``Note app``) from being
    handed verbatim to LaunchServices as a nonexistent application.
    """
    candidate = re.sub(r"\s+", " ", str(value or "")).strip(" ._-'\"")
    candidate = re.sub(
        r"^(?:a|an|my|the)\s+|\s+(?:app|application)$",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip()
    lowered = candidate.casefold()
    if not lowered:
        return ""
    tokens = set(re.findall(r"[a-z]+", lowered))
    if tokens and tokens <= _GENERIC_APP_TARGETS:
        return ""
    if lowered in _WEB_SURFACE_TARGETS:
        return ""
    return _APP_ALIASES.get(lowered, candidate[:80])


def _normalize_app_target(value: str) -> str:
    """Compatibility wrapper for older internal callers."""
    return canonical_app_target(value)


def _coerce_file_facts(raw: object) -> tuple[FileFact, ...]:
    """Only real FileFacts, from something that is actually a sequence.

    ``source.get(...)`` is typed ``object``; iterating it directly is a
    runtime TypeError waiting for the first caller who passes a dict or a
    scalar, and mypy strict says so.
    """
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return ()
    return tuple(item for item in raw if isinstance(item, FileFact))


def _unsupported_control_operations(goal: str) -> list[str]:
    control_text = re.sub(r"([\"']).*?\1", " ", str(goal or ""))
    control_text = re.sub(r"https?://\S+", " ", control_text)
    reasons: list[str] = []
    for operation, reason in _UNVERIFIED_CONTROL_OPERATIONS:
        if operation.search(control_text):
            reasons.append(reason)
    return reasons


def _extract_inline_text(goal: str) -> str:
    quoted = re.search(
        r"\b(?:type|paste|write|fill|insert|enter)\s+(?:the\s+text\s+)?[\"']([^\"']{1,500})[\"']",
        goal,
        flags=re.IGNORECASE,
    )
    if quoted:
        return quoted.group(1).strip()
    direct = re.search(
        r"\b(?:type|paste|enter)\s+(.+?)(?=\s+\b(?:into|in|to|and then|then)\b|[.;]|$)",
        goal,
        flags=re.IGNORECASE,
    )
    return direct.group(1).strip(" \"'") if direct else ""


def _text_witness(text: str) -> str:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return ""
    return normalized[:160].rstrip()


def _extract_explicit_url(goal: str) -> str:
    match = re.search(r"https?://[^\s<>\"']+", goal, flags=re.IGNORECASE)
    return match.group(0).rstrip(".,);]") if match else ""


def _extract_interaction_target(goal: str) -> str:
    match = re.search(
        r"\b(?:click|press|choose|select)\s+(?:on\s+)?(?:the\s+)?"
        r"[\"']?(.+?)[\"']?(?=\s+\b(?:button|menu|item|control)\b|\s+\band\b|[.;]|$)",
        goal,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    target = " ".join(match.group(1).strip(" \"'").split())[:120]
    if _normalize_text(target) in {
        "around",
        "around until it works",
        "anything",
        "it",
        "something",
        "somewhere",
    }:
        return ""
    return target


def _calculation_result(goal: str) -> str:
    expression = goal.lower()
    replacements = {
        "divided by": "/",
        "multiplied by": "*",
        "times": "*",
        "plus": "+",
        "minus": "-",
    }
    for phrase, symbol in replacements.items():
        expression = expression.replace(phrase, f" {symbol} ")
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*([+*/-])\s*(-?\d+(?:\.\d+)?)", expression)
    if not match:
        return ""
    left = float(match.group(1))
    right = float(match.group(3))
    operator = match.group(2)
    if operator == "+":
        result = left + right
    elif operator == "-":
        result = left - right
    elif operator == "*":
        result = left * right
    elif right != 0:
        result = left / right
    else:
        return ""
    if result.is_integer():
        return str(int(result))
    return f"{result:.10f}".rstrip("0").rstrip(".")


def _window_matches_region(
    window: tuple[int, int, int, int] | None,
    desktop: tuple[int, int, int, int] | None,
    region: str,
) -> bool:
    if window is None or desktop is None:
        return False
    wx, wy, ww, wh = window
    dx, dy, dw, dh = desktop
    if min(ww, wh, dw, dh) <= 0:
        return False
    tolerance_x = max(24.0, dw * 0.08)
    tolerance_y = max(24.0, dh * 0.08)
    window_mid_x = wx + ww / 2.0
    window_mid_y = wy + wh / 2.0
    desktop_mid_x = dx + dw / 2.0
    desktop_mid_y = dy + dh / 2.0
    normalized = region.lower()
    if normalized == "left":
        return wx <= dx + tolerance_x and window_mid_x <= desktop_mid_x + tolerance_x
    if normalized == "right":
        return wx + ww >= dx + dw - tolerance_x and window_mid_x >= desktop_mid_x - tolerance_x
    if normalized == "top":
        return wy <= dy + tolerance_y and window_mid_y <= desktop_mid_y + tolerance_y
    if normalized == "bottom":
        return wy + wh >= dy + dh - tolerance_y and window_mid_y >= desktop_mid_y - tolerance_y
    if normalized == "maximized":
        return (
            abs(wx - dx) <= tolerance_x
            and abs(wy - dy) <= tolerance_y
            and ww >= dw * 0.85
            and wh >= dh * 0.80
        )
    if normalized == "left_half":
        return (
            wx <= dx + tolerance_x
            and 0.40 * dw <= ww <= 0.60 * dw
            and wh >= 0.75 * dh
            and window_mid_x <= desktop_mid_x + tolerance_x
        )
    if normalized == "right_half":
        return (
            wx + ww >= dx + dw - tolerance_x
            and 0.40 * dw <= ww <= 0.60 * dw
            and wh >= 0.75 * dh
            and window_mid_x >= desktop_mid_x - tolerance_x
        )
    if normalized == "top_half":
        return (
            wy <= dy + tolerance_y
            and ww >= 0.75 * dw
            and 0.35 * dh <= wh <= 0.65 * dh
            and window_mid_y <= desktop_mid_y + tolerance_y
        )
    if normalized == "bottom_half":
        return (
            wy + wh >= dy + dh - tolerance_y
            and ww >= 0.75 * dw
            and 0.35 * dh <= wh <= 0.65 * dh
            and window_mid_y >= desktop_mid_y - tolerance_y
        )
    return False


def _url_corresponds(observed: str, expected: str) -> bool:
    if not observed or not expected:
        return False
    observed_decoded = urllib.parse.unquote_plus(observed).casefold()
    expected_decoded = urllib.parse.unquote_plus(expected).casefold()
    if expected_decoded in observed_decoded:
        return True
    expected_parts = urllib.parse.urlparse(expected)
    observed_parts = urllib.parse.urlparse(observed)
    if expected_parts.netloc:
        expected_host = expected_parts.netloc.removeprefix("www.").casefold()
        observed_host = observed_parts.netloc.removeprefix("www.").casefold()
        if expected_host != observed_host:
            return False
        expected_tail = urllib.parse.unquote_plus(
            f"{expected_parts.path}?{expected_parts.query}"
        ).casefold().strip("?/")
        return not expected_tail or expected_tail in observed_decoded
    expected_terms = [term for term in re.findall(r"[a-z0-9]+", expected_decoded) if len(term) > 1]
    return bool(expected_terms) and all(term in observed_decoded for term in expected_terms)


def _contains_number(text: str, expected: str) -> bool:
    if not text:
        return False
    pattern = rf"(?<![\d.]){re.escape(expected)}(?![\d.])"
    return re.search(pattern, text.replace(",", "")) is not None


def _format_frame(frame: tuple[int, int, int, int] | None) -> str:
    return ",".join(str(value) for value in frame) if frame else ""
