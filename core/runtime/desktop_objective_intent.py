from __future__ import annotations

import re

from core.runtime.skill_task_bridge import (
    looks_like_capability_inventory_dialogue_request,
    strip_negated_action_spans,
)
from core.utils.intent_normalization import normalize_memory_intent_text
from core.utils.occluded_view_intent import asks_about_occluded_view
from core.utils.own_source_intent import asks_for_own_source
from core.utils.screen_judgement_intent import asks_for_screen_judgement

_WEB_SEARCH_REQUEST_SPAN_RE = re.compile(
    r"\b(?:search|google|look\s*up|research)\b[^.?!]{0,48}?"
    r"\b(?:the\s+)?(?:internet|web|online)\b",
    re.IGNORECASE,
)
_CANONICAL_RESEARCH_TOOL_SPAN_RE = re.compile(
    r"\b(?:use|run|call|invoke|route\s+through|with|via)?\s*"
    r"(?:web_search|grounded_search|search_web|free_search)\b"
    r"[^.?!]{0,80}",
    re.IGNORECASE,
)

_DESKTOP_OBJECTIVE_ACTION_TERMS = (
    "attach",
    "arrange",
    # "paste" and "set" were here and "put" and "copy" were not, so
    # "put the text on my clipboard" and "copy that to my clipboard" reached
    # nothing while "paste it" reached the lane. The same everyday act, named
    # the way people name it, fell outside an enumeration.
    "copy",
    "put",
    "browse",
    "click",
    "compose",
    "close",
    "create",
    # ...and the ordinary synonyms for it. "create" was here alone, so "make a
    # file on my desktop called X with one sentence about what you are doing
    # right now" named no action at all: the reading phrase at the end won,
    # the objective went to the screen-observation lane, and 2ms later the
    # reply was "Done — the desktop steps completed and their effects
    # verified" with nothing written. Same shape as "put"/"copy" above — an
    # everyday word for the act, missing from the enumeration.
    "make",
    "build",
    "generate",
    "draft",
    "produce",
    "record",
    "download",
    "export",
    "find",
    "focus",
    "google",
    "insert",
    "look up",
    "maximize",
    "minimize",
    "move",
    "navigate",
    "open",
    "organize",
    "paste",
    "pdf",
    "resize",
    "save",
    "search",
    "select",
    "show me",
    "switch",
    "tab",
    "timestamp",
    "type",
    "write",
    # Changing a machine setting is a desktop objective too. The list had
    # every verb for moving files and windows and none for altering the
    # system itself, so "change my desktop background to an orca" — which
    # desktop_task can do through system_control — did not route at all and
    # was answered conversationally. Measured live 2026-07-27.
    "change",
    "set",
    "turn on",
    "turn off",
    "enable",
    "disable",
    "adjust",
    "increase",
    "decrease",
    "mute",
    "unmute",
)

_DESKTOP_OBJECTIVE_SURFACE_TERMS = (
    # System surfaces a setting verb acts on, so "set the volume" and "change
    # my wallpaper" reach the lane that can actually do them.
    "background",
    "wallpaper",
    "brightness",
    "volume",
    "dark mode",
    "do not disturb",
    "night shift",
    "setting",
    "settings",
    "app",
    "application",
    # LIVE, 2026-08-10: "Put the text ORION-7 on my clipboard" did not route,
    # nothing ran, and she said "The text ORION-7 is now on your clipboard"
    # while the clipboard was empty. set_clipboard and get_clipboard are
    # declared desktop actions and the word never appeared in this module, so
    # no clipboard request could reach the lane that performs them.
    "clipboard",
    "pasteboard",
    "browser",
    "chrome",
    "computer",
    "desktop",
    "doc",
    "docs",
    "document",
    "drive",
    "file",
    "finder",
    "folder",
    "google",
    "notes",
    "pages",
    "pdf",
    "safari",
    "screen",
    "tab",
    "textedit",
    "web",
    "website",
    "window",
    "word",
    # Plurals and the bulk quantifiers a person actually uses. Every surface
    # above was singular only, so "minimize the window" reached the desktop
    # lane and "minimize all windows" did not — the same request, declined for
    # its grammatical number. Found 2026-08-10.
    "apps",
    "applications",
    "documents",
    "files",
    "folders",
    "tabs",
    "windows",
    "everything",
    "all of them",
)

_DIRECT_DESKTOP_ACTION_RE = re.compile(
    r"\b(?:please\s+)?(?:open|create|write|save|export|search|google|look\s+up|"
    r"type|paste|compose|download|navigate|click|show\s+me|arrange|resize|drag|"
    r"focus|select|switch|close|minimi[sz]e|maximi[sz]e|organize|"
    # Setting verbs, but only when they act on a machine surface — "change my
    # wallpaper" is a desktop objective, "change your mind" is not. The verb
    # alone is far too common in ordinary speech to admit on its own.
    r"(?:change|set|adjust|increase|decrease|turn\s+(?:on|off)|enable|disable|"
    r"mute|unmute)\s+(?:the\s+|my\s+|your\s+)?(?:desktop\s+)?"
    r"(?:background|wallpaper|brightness|volume|dark\s+mode|night\s+shift|"
    r"do\s+not\s+disturb|setting|settings|screen\s+saver))\b",
    re.IGNORECASE,
)
_EXPLANATORY_DESKTOP_QUESTION_RE = re.compile(
    r"^\s*(?:how|what|why)\s+(?:would|could|should|do|does|can)\s+(?:you\s+)?"
    r"(?:open|create|write|save|export|search|google|look\s+up|type|paste|"
    r"compose|download|navigate|click|arrange|resize|drag|focus|select|switch|"
    r"close|minimi[sz]e|maximi[sz]e|organize)\b",
    re.IGNORECASE,
)
# Screen-observation requests need the desktop BODY (read_screen_text), but
# they carry no action+surface verb pair, so the generic classifier missed
# them and "what's on my screen" silently did nothing. Treat them as desktop
# objectives directly. Kept in sync with desktop_task's observation markers.
# Talking ABOUT a past observation is not asking for a new one.
#
# Live 2026-07-27: a message that began "Earlier you described what was on
# his screen and I decided you had made it up" was routed to the governed
# desktop lane and refused, because "described ... screen" reads exactly like
# "describe my screen". Recounting what already happened, or discussing the
# faculty itself, is conversation.
_PAST_SCREEN_NARRATION_RE = re.compile(
    r"\b(?:earlier|previously|before|a\s+moment\s+ago|last\s+time|"
    r"you\s+(?:described|said|told|reported|showed|mentioned|claimed|were)|"
    r"i\s+(?:decided|thought|assumed|concluded|said))\b",
    re.IGNORECASE,
)

#: Surfaces that mean "the thing Bryan is looking at".
_SCREEN_SURFACE = r"(?:screen|display|monitor|window|windows|desktop)"

#: Ways of asking to be told what is there. Deliberately a CUE class rather
#: than a phrase list.
#:
#: The phrase-list approach has now failed twice on the same request, weeks
#: apart. First "the screen" missed a list wanting "my screen". Then, live on
#: 2026-08-03, "can you tell me what you see on my screen currently?" missed
#: this regex, because it accepted "what DO YOU see" but not "what you see" —
#: one auxiliary verb away from a flat "I can't see your screen", which is
#: worse than a refusal because it is false.
#:
#: An enumeration of phrasings will always be one phrasing behind. The rule is
#: structural instead: a perception or report cue anywhere near a screen noun.
_PERCEPTION_CUE = (
    r"(?:read|look(?:ing)?\s+at|inspect|describe|check|examine|capture|view"
    r"|see|seeing|watch|showing|shows|shown|display(?:ed|ing)?|tell\s+me"
    # Question words, with the apostrophe optional. People type "whats on my
    # screen" constantly, and \bwhat\b cannot match inside "whats" — one
    # missing punctuation mark was the whole difference between looking and
    # not. This is the third time this cue class has been one phrasing short.
    r"|what(?:'?s)?|which|where(?:'?s)?|frontmost|in\s+front)"
)

_SCREEN_OBSERVATION_RE = re.compile(
    # cue ... screen  ("tell me what you see on my screen", "what's on screen")
    rf"\b{_PERCEPTION_CUE}\b[^.?!]{{0,60}}\b{_SCREEN_SURFACE}\b"
    # screen ... cue  ("my screen — what do you see", "the display shows what?")
    rf"|\b{_SCREEN_SURFACE}\b[^.?!]{{0,60}}\b{_PERCEPTION_CUE}\b"
    r"|\bscreenshot\b",
    re.IGNORECASE,
)


# Surfaces whose ordinary English meaning is far more common than the app that
# shares the name. A bare word-boundary match on these turns any sentence
# containing an everyday noun into a desktop-control request.
#
# Measured live: "Aura, it's Bryan. Remember the WORD lantern ... show me the
# real output. Run a Python snippet that prints the PID and CPU cores." The
# action term "show me" came from one sentence and the surface term "word" —
# meaning Microsoft Word — came from another, so a code-execution request was
# routed into desktop OS automation, which then correctly refused for lack of an
# observable acceptance contract. The user got a failure for a request the
# sandbox could have answered.
#
# These now require actual app context: a vendor name, an app/document noun, or
# a preposition/verb that only makes sense against an application.
_APP_CONTEXT_SURFACE_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "word",
        r"(?:\b(?:microsoft|ms)\s+word\b"
        r"|\bword\s+(?:doc|docs|document|documents|file|files|app)\b"
        r"|\b(?:open|in|into|to|from|using|with|launch|quit|close|switch\s+to)\s+word\b)",
    ),
    (
        "pages",
        r"(?:\bapple\s+pages\b"
        r"|\bpages\s+(?:doc|docs|document|documents|file|files|app)\b"
        r"|\b(?:open|into|using|launch|quit|close|switch\s+to)\s+pages\b)",
    ),
    (
        "drive",
        r"(?:\bgoogle\s+drive\b"
        r"|\bdrive\s+(?:folder|folders|file|files)\b"
        r"|\b(?:in|into|on|from|to)\s+(?:my\s+|the\s+)?drive\b)",
    ),
)

_PLAIN_ENGLISH_APP_NAMES = frozenset(term for term, _ in _APP_CONTEXT_SURFACE_PATTERNS)


def _contains_desktop_objective_term(text: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        if term in _PLAIN_ENGLISH_APP_NAMES:
            continue
        escaped = re.escape(term)
        if re.search(rf"\b{escaped}\b", text, flags=re.IGNORECASE):
            return True
    requested = set(terms)
    for term, pattern in _APP_CONTEXT_SURFACE_PATTERNS:
        if term in requested and re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False



#: An absolute or home-relative path with a real separator. Deliberately not
#: matching bare words like "core/introspection" — a relative fragment is
#: ambiguous with ordinary prose ("the input/output problem").
_CONCRETE_PATH_RE = re.compile(r"(?<![\w/])(?:~|/)[\w.\-]+(?:/[\w.\-]+)*/?")

#: What a request does with a path it names.
_PATH_OPERATION_RE = re.compile(
    r"\b(?:count|how\s+many|list|show|read|open|write|save|create|append|"
    r"delete|remove|copy|move|rename|check|inspect|look\s+at|what(?:'s| is)\s+in|"
    r"contents?\s+of|files?\s+in|find)\b",
    re.IGNORECASE,
)


#: A path a person spelled out instead of typing: "on my Desktop called
#: aura_haiku.txt". LIVE, 2026-08-10 — "Make me a file on my Desktop called
#: aura_haiku.txt with a haiku you wrote yourself" did not route to the body,
#: so nothing was written and the reply claimed "file writing was successful".
#: The planner could plan it perfectly; it was never asked.
_NAMED_ON_SURFACE_RE = re.compile(
    r"\b(?:on|in|to)\s+(?:my\s+|the\s+)?(?:desktop|documents|downloads)\b"
    r"(?:[^.?!]|\.(?=[A-Za-z0-9])){0,40}?\b(?:called|named)\s+[\w.\-]+\.[A-Za-z0-9]{1,8}",
    re.IGNORECASE,
)


#: A clause reporting a request that was already made. The verbs inside it
#: are history being described, not work being asked for.
_REPORTED_REQUEST_SPAN_RE = re.compile(
    r"\b(?:earlier|before|yesterday|previously|already|last\s+time|"
    r"a\s+(?:moment|minute|while)\s+ago)?\s*"
    r"(?:i|you|we)\s+(?:had\s+)?(?:asked|told|requested|wanted|said)\b"
    # Stops at a new instruction rather than running to the sentence end.
    # "I asked you to do that already, please actually write hello into
    # ~/Documents/x.txt now" reports history AND makes a request, and a greedy
    # span swallowed the request — refusing to act on an explicit retry is a
    # worse failure than the litter this exists to prevent.
    r"(?:[^.?!]|\.(?=[A-Za-z0-9]))*?(?=(?:[,;—–-]*\s*\b(?:please|now|actually|go\s+ahead|just)\b)|[.?!]|$)",
    re.IGNORECASE,
)

#: A question about what she DID, which is answered from memory rather than by
#: doing it again. "what did you write to my Desktop earlier?" carries a write
#: verb and a surface and is not a request to write anything.
_PAST_ACTION_QUESTION_RE = re.compile(
    r"\b(?:what|which|where|when|how\s+many)\b(?:[^.?!]|\.(?=[A-Za-z0-9])){0,80}?"
    r"\b(?:did|have)\s+you\b|\bdo\s+you\s+remember\b|"
    r"\bwhat\s+was\s+the\b",
    re.IGNORECASE,
)

#: An instruction meant for right now, which outranks any recall framing around
#: it.
_PRESENT_INSTRUCTION_RE = re.compile(
    r"\b(?:please|now|actually|go\s+ahead|do\s+it|again)\b",
    re.IGNORECASE,
)


def _asks_about_a_concrete_path(text: str) -> bool:
    """True when the turn names a real file and does something with it.

    Named two ways, because people name files two ways: written out with
    separators, or described — "a file on my Desktop called notes.txt". Both
    are a specific file on this machine, which is the only thing that matters
    for deciding whether the body is required.
    """

    body = text or ""
    if _NAMED_ON_SURFACE_RE.search(body):
        return True
    if not _CONCRETE_PATH_RE.search(body):
        return False
    return bool(_PATH_OPERATION_RE.search(body))


def looks_like_desktop_objective(user_message: str) -> bool:
    """Return true for user requests that need visible desktop/computer action.

    This is intentionally shared by typed chat and voice so Aura does not have
    two drifting definitions of "this request needs the desktop body." The
    function only classifies the objective; actual execution still goes through
    CognitiveEngine, CapabilityEngine, desktop_task, computer_use, and the
    permission/governance gates.
    """

    text = normalize_memory_intent_text(user_message).lower()
    if not text:
        return False
    # Acting on a web page is acting on the world.
    #
    # LIVE 2026-08-18: "go take it for real: <url> — work through the whole
    # thing, answer every question as yourself" classified False here, so it
    # never reached the execution lane at all; it was treated as a lookup,
    # fetched, summarised into nothing, and answered "I couldn't get to an
    # answer I'd stand behind."
    #
    # A questionnaire, a checkout and a signup wizard are the same request as
    # "open Notes and write something": a thing to be DONE, whose result does
    # not exist until it is done. Reading a page is not, and stays a lookup.
    try:
        from core.conversation.page_interaction import asks_to_act_on_a_page

        if asks_to_act_on_a_page(user_message):
            return True
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    sanitized_text = strip_negated_action_spans(text).lower()
    # An action verb inside REPORTED history is not an instruction.
    #
    # LIVE, 2026-08-10: "Earlier today I asked you to count files in one of
    # your own directories, and separately to write a haiku. Without
    # guessing: what was the count, and what was the haiku about?" — a
    # recall question — routed here and created
    # ~/Desktop/Aura Desktop Task 1786465767/ with a summary file in it.
    #
    # Both verbs belong to requests she was being TOLD ABOUT rather than
    # requests being made. Asking what she did is not asking her to do it
    # again, and answering it by doing it again leaves litter on a Desktop.
    sanitized_text = _REPORTED_REQUEST_SPAN_RE.sub(' ', sanitized_text)
    # And a question about what she did is answered from memory, not by doing
    # it again — unless the same turn also asks for it now.
    if _PAST_ACTION_QUESTION_RE.search(sanitized_text) and not _PRESENT_INSTRUCTION_RE.search(
        sanitized_text
    ):
        return False
    # Explicit web-search phrasing ("search the internet/web for X") is a
    # research request, not a desktop objective; classifying it as desktop
    # let the response contract suppress requires_search while desktop_task
    # had nothing visible to do — the search silently went dark on both
    # lanes. Strip the span so only OTHER action/surface terms ("...and
    # save it to Notes") can still classify the request as desktop.
    sanitized_text = _WEB_SEARCH_REQUEST_SPAN_RE.sub(" ", sanitized_text)
    # A named canonical research tool is not itself a visible desktop
    # objective. "Use web_search from the desktop lane" should stay in the
    # research/tool path; "open Chrome and search" still routes to desktop
    # because the visible app action remains after this span is stripped.
    sanitized_text = _CANONICAL_RESEARCH_TOOL_SPAN_RE.sub(" ", sanitized_text)
    # A concrete filesystem path in the request IS the routing signal.
    #
    # LIVE, 2026-08-10: "Count how many .py files are in
    # /Users/bryan/.aura/live-source/core/introspection, then write that number
    # and the file names into ~/Documents/aura_probe_count.txt" routed here as
    # ordinary conversation. She answered from nothing — 3 instead of 9, three
    # invented filenames, and a report of a write that never happened.
    #
    # "write hello into ~/Documents/x.txt" routed correctly, so the action+
    # surface pair works when the action verb LEADS. It missed this because the
    # sentence opens with "count how many", and it missed the pure read
    # "how many .py files are in /abs/path?" for the same reason.
    #
    # The path is the part that settles it. Nothing in the model can answer a
    # question about the contents of a real path, and nothing can write to one
    # without the body. Asked about a path she has not read, the only honest
    # options are to look or to decline, and she did neither.
    if _asks_about_a_concrete_path(sanitized_text):
        return True
    # Ordered ahead of the inventory check on purpose: "count how many .py
    # files are in /abs/path" was being read as a question about her own
    # capability inventory, which is where the live failure was decided.
    if looks_like_capability_inventory_dialogue_request(user_message):
        return False
    # Being shown her own source is not desktop work. "Show me a piece of your
    # own code and tell me which file it lives in" carries an action word
    # ("show me") and a surface word ("file"), so the generic action+surface
    # test called it a desktop objective, sent it to os_automation, and it
    # came back "refused to act because the objective has no complete
    # observable acceptance contract" — correctly, because reading her source
    # aloud has no observable desktop effect to verify. Measured live
    # 2026-08-03: the conversational floor had a real 1999-character excerpt
    # ready for that exact sentence and the person never saw it, because this
    # lane answered first.
    if asks_for_own_source(user_message):
        return False
    # "Ignore your own window — what else is on the screen?" is a question
    # about the ARRANGEMENT, and a screen capture reads what is visible. Sent
    # down the capture lane it came back as a raw OCR dump of whichever window
    # was readable; the floor answers it from the window layout, naming each
    # window, how much of it shows, and saying plainly what it cannot read
    # while it is covered. Measured live 2026-08-04.
    if asks_about_occluded_view(user_message):
        return False
    # An opinion about the screen is not an action on it. "of everything you can
    # see open right now, which window would you close first if you were me, and
    # why that one? I want your actual judgement, not a list" matched the screen
    # observation branch below, went to the desktop lane, and came back
    # "os_automation refused to act because the objective has no complete
    # observable acceptance contract. Completed 0/1 steps." os_automation was
    # right — nothing was asked to happen. Nobody asked her to close a window;
    # they asked which one she WOULD close. Measured live 2026-08-10.
    #
    # Her agency rules already said it — "Hypotheticals, quoted requests,
    # negated actions, and recalled evidence are not execution requests merely
    # because they name a tool" — but only in the identity contract, where the
    # router could not act on it.
    if asks_for_screen_judgement(user_message):
        return False
    # Screen observation ("read my screen", "what's on my screen") needs the
    # desktop body even though it carries no action+surface verb pair.
    if _SCREEN_OBSERVATION_RE.search(sanitized_text) and not (
        _PAST_SCREEN_NARRATION_RE.search(sanitized_text)
    ):
        return True
    if _EXPLANATORY_DESKTOP_QUESTION_RE.search(text):
        return False
    if not _contains_desktop_objective_term(sanitized_text, _DESKTOP_OBJECTIVE_ACTION_TERMS):
        return False
    if not _contains_desktop_objective_term(sanitized_text, _DESKTOP_OBJECTIVE_SURFACE_TERMS):
        return False

    try:
        from core.phases.action_intent import detect_action_intent

        intent = detect_action_intent(user_message)
        if bool(getattr(intent, "should_execute", False)):
            return True
        if bool(getattr(intent, "has_action_request", False)) and re.search(
            r"\b(?:can|could|will|would)\s+you\b",
            sanitized_text,
            flags=re.IGNORECASE,
        ):
            return True
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        pass

    return bool(_DIRECT_DESKTOP_ACTION_RE.search(sanitized_text))


#: Verbs that change something. A screen request carrying one of these is
#: "look, then act", not a read. Drawn from the action terms above, minus the
#: ones that ARE ways of asking to be shown something.
_MUTATING_ACTION_TERMS: tuple[str, ...] = tuple(
    term
    for term in _DESKTOP_OBJECTIVE_ACTION_TERMS
    if term not in {"find", "search", "google", "look up", "show me", "tab", "pdf"}
)


def looks_like_screen_observation(user_message: str) -> bool:
    """True when the request is to READ the screen and report, not to act on it.

    Observation and actuation need different lanes. os_automation is the
    actuation lane: it refuses any objective without an observable acceptance
    contract, which is correct, because a description is not an effect it can
    verify. Sending a read there produces "OS automation refused to act
    because the objective has no complete observable acceptance contract" for
    a question the perception lane answers in one screenshot.

    This existed twice and the copies disagreed. desktop_task carried a
    literal-substring list containing "what's on my screen" and "look at the
    screen"; live on 2026-08-03, "Can you see what's on the screen and tell me
    what you see?" matched none of them — "the screen", not "my screen" — so
    the read escalated to os_automation and was refused. The regex here
    already matched it. One definition now, shared by both callers.
    """

    text = normalize_memory_intent_text(user_message).lower()
    if not text:
        return False
    sanitized_text = strip_negated_action_spans(text).lower()
    if not _SCREEN_OBSERVATION_RE.search(sanitized_text):
        return False
    if _PAST_SCREEN_NARRATION_RE.search(sanitized_text):
        # Recounting what was on screen earlier is conversation, not a look.
        return False
    # "Look at the screen and close the window" observes AND acts; only the
    # actuation lane can finish it.
    return not _contains_desktop_objective_term(sanitized_text, _MUTATING_ACTION_TERMS)


#: Ways of asking to be SHOWN where something is, rather than told. The
#: capture is the thing to point at, and it is deliberately taken verbatim:
#: the accessibility tree is searched for this literal text, so paraphrasing
#: it here would move the rectangle off the thing the person named.
_POINT_REQUEST_RE = re.compile(
    r"\b(?:"
    r"where\s+(?:is|are|'s)\s+"
    r"|show\s+me\s+(?:where\s+)?"
    r"|point\s+(?:at|to)\s+"
    r"|highlight\s+"
    r"|which\s+one\s+is\s+"
    r")(?:the\s+|my\s+|that\s+|a\s+)?(?P<needle>[^,.?!;]{2,60})",
    re.IGNORECASE,
)

#: Trailing phrases that are part of the ASKING, not part of the thing. "the
#: submit button on my screen" is a request to point at "the submit button";
#: searching the accessibility tree for the whole phrase finds nothing.
_POINT_NEEDLE_TAILS: tuple[str, ...] = (
    "on my screen",
    "on the screen",
    "on screen",
    "in the window",
    "right now",
    "please",
    "is",
    "at",
)


def asks_to_be_shown_where(user_message: str) -> str:
    """The thing she is being asked to POINT at, or "" when she is not.

    "Which one is the failing test?" has a better answer than a paragraph
    describing where to look, and the difference between the two is a
    question this predicate answers.

    Returns the needle rather than a bool because the caller needs it: the
    overlay is placed by searching the accessibility tree for this literal
    text, and a bool would leave every caller re-deriving it differently.
    """
    text = str(user_message or "").strip()
    if not text:
        return ""
    match = _POINT_REQUEST_RE.search(text)
    if not match:
        return ""
    needle = str(match.group("needle") or "").strip()
    # Strip the asking off the tail, repeatedly: "the submit button on my
    # screen please" sheds two.
    changed = True
    while changed and needle:
        changed = False
        lowered = needle.lower()
        for tail in _POINT_NEEDLE_TAILS:
            if lowered.endswith(" " + tail) or lowered == tail:
                needle = needle[: len(needle) - len(tail)].strip()
                changed = True
                break
    # Two characters is the floor locate_on_screen enforces anyway; returning
    # a one-character needle would match half the window.
    return needle if len(needle) >= 2 else ""


__all__ = [
    "asks_to_be_shown_where",
    "looks_like_desktop_objective",
    "looks_like_screen_observation",
]
