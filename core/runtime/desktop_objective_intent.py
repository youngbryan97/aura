from __future__ import annotations

import logging
import re
from typing import Any

from core.runtime.os_automation_effects import extract_direct_application_targets
from core.runtime.skill_task_bridge import (
    looks_like_capability_inventory_dialogue_request,
    strip_negated_action_spans,
)
from core.utils.intent_normalization import normalize_memory_intent_text
from core.utils.occluded_view_intent import asks_about_occluded_view
from core.utils.own_source_intent import asks_for_own_source
from core.utils.screen_judgement_intent import asks_for_screen_judgement

logger = logging.getLogger("Aura.DesktopIntent")

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
    r"|\bscreenshot\b"
    # Which app is in front IS a question about the screen, and it is asked
    # without the word. LIVE 2026-08-18: "can you tell what app I'm using
    # right now?" reached no reading and was answered "I can capture and read
    # this screen. I'm in my own little computational world, not connected to
    # your device's sensors or UI" — a denial of the capability she has, and a
    # self-contradiction inside two sentences.
    r"|\bwhat\s+(?:app|application|program|window|document|tab|file)\b"
    r"[^.?!]{0,30}\b(?:am\s+i|are\s+we|is\s+(?:open|active|frontmost|in\s+front)|"
    r"i(?:'m| am)\s+(?:using|in|on|looking\s+at))"
    r"|\b(?:which|what)\s+(?:app|application|program|window)\s+(?:is|am|are)\b"
    r"|\bwhat\s+am\s+i\s+(?:looking\s+at|working\s+(?:on|in)|reading)\b"
    r"|\bam\s+i\s+(?:in|using|on)\s+(?:\w+\s+){0,2}(?:app|application|window)\b",
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
    # A question about how she works is not work.
    #
    # It cannot be answered by doing anything, so routing it to a lane that
    # acts makes her act on the world to answer a question about herself.
    # LIVE 2026-08-27: "What are the three biggest weaknesses in how you
    # currently decide what to do on a screen? Be blunt." went to the desktop
    # lane, tried to read the screen, was refused by the executive for want of
    # scoped authority, and came back as a failure report instead of an answer.
    if asks_about_screens_in_general(text):
        return False
    # A goal to be kept at until a condition holds is a desktop objective by
    # definition: it names something to keep doing on the machine and the
    # thing on screen that means it is finished.
    #
    # LIVE 2026-08-19: "Go find a 2048 game online and play it until you get
    # a 128 tile. Tell me what you are doing as you go." classified False
    # here, was routed to identity grounding by the rider on the end, and
    # answered "I'm Aura. I'm a local stateful cognitive-agent runtime" while
    # the browser sat on a blank page.
    #
    # The recogniser for these already exists and is already the thing that
    # plans them, so asking it here means one definition rather than two.
    try:
        from core.runtime.watched_goal import read_watched_goal

        if read_watched_goal(user_message) is not None:
            return True
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
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
    # Reading something on disk is an observation, and the actuation lane
    # cannot perform it. Live 2026-08-19, "read the code at <path> and work out
    # why the test fails" was routed here and came back "os_automation failed:
    # TimeoutError ... Completed 0/1 steps" — the screen driver spending a turn
    # trying to verify an effect no screen would ever show, while
    # file_operation sat READY with a read action. Asked first, because a path
    # read matches several of the terms below.
    if looks_like_filesystem_observation(user_message):
        return False

    # Building software is not driving the screen.
    #
    # LIVE 2026-08-20: "build me a small web app: a single HTML page that
    # tracks how long I've been sitting… tell me where you put it" was routed
    # here and came back "os_automation refused to act because the objective
    # has no complete observable acceptance contract. Completed 0/1 steps."
    # It never could have one: nothing appears on a screen when a file is
    # written. build_app was ranked FIRST by capability selection and never
    # asked.
    #
    # Same rule as the read above, in the other direction: the actuation lane
    # is for what a screen shows, and a program that does not exist yet shows
    # nothing.
    if asks_to_build_software(user_message):
        return False

    # Working something out from a named file is not driving the screen.
    #
    # Same rule as the two above, a third time: the actuation lane is for what
    # a screen shows, and an average over a spreadsheet shows nothing on one.
    #
    # LIVE 2026-08-27: "Since West came out top on average approved deal size
    # in <path>/deals.csv, what's West doing that the other regions should
    # copy?" was routed here and came back "OS automation refused to act
    # because the objective has no complete observable acceptance contract.
    # Completed 0/1 steps." It never could have one — and the tabular reader
    # settles the premise of that question exactly.
    #
    # A turn that also asks for a change is not one of these: "count the .py
    # files in <dir> and write the number into <file>" needs both, and only
    # the lane that can write can finish it. The read guard above draws the
    # same line for the same reason.
    try:
        # Only what the words settle. Routing away from the lane that can act
        # is expensive when it is wrong, so a learned maybe is not enough:
        # "open a browser window, search for climate news, and show me the
        # articles" was held back from the lane that can do it.
        from core.intent.needs_computation import needs_computation_plainly

        if needs_computation_plainly(user_message) and not _asks_to_change_a_file(
            user_message
        ):
            return False
    except (ImportError, AttributeError, TypeError, ValueError):
        pass

    try:
        from core.conversation.page_interaction import asks_to_act_on_a_page

        if asks_to_act_on_a_page(user_message):
            return True
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    # OS settings are declared in one affordance registry that already owns
    # their language-to-goal-state translation. The registry establishes WHAT
    # goal-state the clause names; the shared request-mood substrate establishes
    # WHETHER the person asked Aura to reach it. Keeping both predicates is what
    # separates "use /path as wallpaper" from "why would someone use X as
    # wallpaper?" without teaching each setting its own permission grammar.
    try:
        from core.conversation.request_mood import assess_request_mood
        from core.skills.os_affordances import detect_os_settings

        if detect_os_settings(user_message):
            setting_mood = assess_request_mood(user_message)
            if setting_mood.asks_for_action:
                return True
            if setting_mood.is_about_rather_than_asking:
                return False
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
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
    # An application does not stop being a desktop surface because it is new,
    # uncommon, or absent from this machine. The executor already parses
    # lifecycle verbs and their direct objects; use that same typed target here
    # after the shared mood and temporal exclusions above. This covers "Open
    # ProductName" and "Could you launch ProductName?" without teaching the
    # router every product name, and without admitting metaphors such as "open
    # your mind" whose object is not app-shaped.
    try:
        from core.conversation.request_mood import assess_request_mood

        if (
            assess_request_mood(user_message).asks_for_action
            and extract_direct_application_targets(user_message)
        ):
            return True
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        pass
    # The vocabulary check is where a phrasing nobody enumerated dies.
    #
    # A request with no term from either list returns False here, before
    # anything else looks at it — and that is precisely the case the learned
    # surface exists for. Consulted after this point it was never reached on
    # the turns that needed it.
    if not _contains_desktop_objective_term(
        sanitized_text, _DESKTOP_OBJECTIVE_ACTION_TERMS
    ) or not _contains_desktop_objective_term(
        sanitized_text, _DESKTOP_OBJECTIVE_SURFACE_TERMS
    ):
        return _learned_actuation_decision(user_message) is True

    try:
        from core.conversation.request_mood import assess_request_mood

        # The action and machine surface are already established above. What
        # remains is grammatical: is the person asking for that action now, or
        # merely discussing it? ``ActionIntent`` used to be queried here for a
        # ``should_execute`` attribute its result type does not define, then a
        # modal-verb regex rescued only "can/could/will/would you" requests.
        # Imperatives such as "find an image and make it my background" were
        # therefore discarded even though the language substrate had already
        # split them into present actionable clauses. Use that shared typed
        # judgement directly so chat, voice, and every desktop action inherit
        # the same request semantics.
        if assess_request_mood(user_message).asks_for_action:
            return True
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        pass

    by_pattern = bool(_DIRECT_DESKTOP_ACTION_RE.search(sanitized_text))

    # What actually ran, for requests like this one.
    #
    # The patterns above are seventeen enumerations of screen work. The
    # intention log holds a hundred and ten distinct requests a person made
    # and the capability that succeeded for each, and that decision measures
    # AUROC 0.979 on a held-out third.
    #
    # Additive in one direction only. Where the patterns already say yes they
    # keep saying yes: sending real screen work somewhere else is the worse
    # error, and flipping a yes to a no needs the surface to beat the patterns
    # on their own examples first. Where the patterns say no and the surface
    # is confident, the phrasing is one nobody enumerated — which is how every
    # one of these rules has been wrong before.
    if not by_pattern:
        if _learned_actuation_decision(user_message) is True:
            return True
    elif _learned_actuation_decision(user_message) is False:
        logger.info(
            "🧭 [ROUTING] the patterns call this screen work and the learned "
            "surface does not; keeping the patterns."
        )
    return by_pattern


def _learned_actuation_decision(user_message: str) -> bool | None:
    """Whether requests like this one have needed the screen. None if unsure."""
    try:
        from core.language.desktop_actuation import actuation_surface

        return actuation_surface().decide_without_waiting(str(user_message or ""))
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return None


#: Verbs that change something. A screen request carrying one of these is
#: "look, then act", not a read. Drawn from the action terms above, minus the
#: ones that ARE ways of asking to be shown something.
_MUTATING_ACTION_TERMS: tuple[str, ...] = tuple(
    term
    for term in _DESKTOP_OBJECTIVE_ACTION_TERMS
    if term not in {"find", "search", "google", "look up", "show me", "tab", "pdf"}
)


#: Reading a file is an observation. Writing one is an actuation.
#:
#: LIVE DEFECT, 2026-08-19. "there's a python project at /private/tmp/.../ledger
#: - one of its tests is failing. read the code, work out why, and tell me
#: exactly which line is wrong" was routed to the actuation lane and came back:
#:
#:     os_automation failed: Skill error: TimeoutError ... Completed 0/1 steps.
#:
#: os_automation drives the screen. It cannot read a file, so it spent the turn
#: failing to verify an effect that was never going to happen, while
#: file_operation sat READY with a read action that is pure observation.
#:
#: This module already draws exactly this line for the screen — see
#: :func:`looks_like_screen_observation`, written after a screen READ was sent
#: to the actuation lane and refused. The same mistake, one surface over.
_FILESYSTEM_OBSERVATION_RE = re.compile(
    r"\b(?:read|list|show|display|inspect|examine|count|check|look\s+at|"
    r"cat|view|open|go\s+through|walk\s+through|step\s+through|trace|"
    r"review|explain|diagnose|figure\s+out|work\s+out)\b"
    # Asking somebody to TELL you about a thing is asking them to look at it,
    # and it needs no anchor because the form is unambiguous. The opening
    # question forms below have to sit at the start of the turn or "what"
    # would match nearly everything; this one carries its own subject.
    r"|\b(?:tell|show|walk)\s+me\b[^.?!]{0,40}?"
    r"\b(?:what|where|why|how|which|through)\b",
    re.IGNORECASE,
)

#: Anything that changes what is on disk.
_FILESYSTEM_MUTATION_RE = re.compile(
    r"\b(?:write|save|create|make|append|edit|modify|patch|fix|update|"
    r"delete|remove|rename|move|copy|touch|mkdir|chmod)\b",
    re.IGNORECASE,
)


#: Making a program, as opposed to running one that already exists.
_BUILDS_SOFTWARE_RE = re.compile(
    r"\b(?:build|make|write|create|code|implement|generate|scaffold|knock\s+up|"
    r"put\s+together)\b[^.?!]{0,80}?"
    r"\b(?:web\s*app|webapp|app|site|website|web\s*page|html|page|game|tool|"
    r"script|program|widget|dashboard|prototype|demo|utility)\b",
    re.IGNORECASE,
)

#: What the request is about is the screen, not a program: "open the app",
#: "close that window". These share the nouns above and are actuation.
_DRIVES_THE_SCREEN_RE = re.compile(
    r"\b(?:open|close|quit|switch\s+to|click|type\s+into|scroll|drag|"
    r"minimi[sz]e|maximi[sz]e|focus)\b",
    re.IGNORECASE,
)


def asks_to_build_software(user_message: str) -> bool:
    """True when the request is to CONSTRUCT a program rather than drive one.

    Narrow: it has to name making something AND name the kind of thing, and it
    must not also be asking for the screen to be operated — "open the app and
    build a new project in it" is still desktop work.
    """
    text = normalize_memory_intent_text(user_message)
    if not text.strip():
        return False
    if _DRIVES_THE_SCREEN_RE.search(text):
        return False
    return bool(_BUILDS_SOFTWARE_RE.search(text))


#: Asking what is at a path, with no verb at all: "what's in /etc/hosts",
#: "how many .py files are in <path>?", "is there a README in <path>". A
#: question about a path is a request to look at it.
_ASKS_ABOUT_A_PATH_RE = re.compile(
    r"^\s*(?:so\s+|and\s+|ok(?:ay)?,?\s+|hey,?\s+)?"
    r"(?:what|what's|whats|which|how\s+many|how\s+much|how\s+big|is\s+there|"
    r"are\s+there|does\s+\w+\s+(?:have|contain)|do\s+you\s+see|anything)\b",
    re.IGNORECASE,
)

#: Whether this turn is asking to look at something on disk.
#:
#: The verbs and the question openings above are the floor. This is the
#: mechanism, because which phrasings mean "look" is a judgement about meaning
#: and a verb list is always the verbs one person thought of.
_WANTS_TO_LOOK: Any = None


def _looking_surface() -> Any:
    """The learned surface, built once and registered on first consultation."""
    global _WANTS_TO_LOOK
    if _WANTS_TO_LOOK is not None:
        return _WANTS_TO_LOOK
    try:
        from core.language.learned_matcher import LearnedMatcher, embed_sentences

        _WANTS_TO_LOOK = LearnedMatcher(
            name="filesystem_observation",
            positives=(
                "what's in /etc/hosts",
                "how many .py files are in that directory?",
                "list the contents of ~/Documents",
                "read the config file and tell me what it says",
                "is there a README in there?",
                "show me what that folder holds",
                "have a look at ~/Downloads and tell me what's there",
            ),
            negatives=(
                "write hello into ~/Documents/x.txt",
                "delete everything in ~/Downloads",
                "move that file to the desktop",
                "what should I do with ~/Documents?",
                "open Notes and write something",
                "rename the folder to archive",
            ),
            features=embed_sentences,
        )
    except (ImportError, RuntimeError, TypeError, ValueError):
        _WANTS_TO_LOOK = None
    return _WANTS_TO_LOOK


#: A place a change could land, close enough after the verb to be its object.
_SOMETHING_TO_CHANGE = re.compile(
    r"(?:~?/[\w.\-~/]+|\b\w+\.[a-z0-9]{1,5}\b|\bfile\b|\bfolder\b|"
    r"\bdirectory\b|\bnote\b|\bdocument\b)",
    re.IGNORECASE,
)


def _asks_to_change_a_file(user_message: str) -> bool:
    """Whether a mutation verb here has something to mutate.

    LIVE, 2026-08-27: "what's West doing that the other regions should COPY?"
    counted as a file mutation, because "copy" is on the list of things that
    change what is on disk. It is also an ordinary English verb, and the same
    shape as `\boff\b` matching inside "off-the-shelf" — a word read out of
    prose that was never about a file.

    A change needs somewhere to land, so the verb has to be followed by one.
    """
    text = strip_negated_action_spans(normalize_memory_intent_text(user_message)).lower()
    for match in _FILESYSTEM_MUTATION_RE.finditer(text):
        after = text[match.end() : match.end() + 48]
        if _SOMETHING_TO_CHANGE.search(after):
            return True
    return False


#: How far after a verb its object can sit before it belongs to another clause.
#: A verb and its object are adjacent or nearly so; past that the reader is
#: joining two clauses, which is the mistake being fixed.
_WITHIN_REACH = 6

#: Standing in for the named thing once it has been named: "read the log, then
#: delete it" changes the log, and says so with a pronoun.
_STANDS_FOR_THE_THING = re.compile(
    r"\b(?:it|them|that|this|those|these)\b"
    r"|\bthe\s+(?:file|files|directory|folder|log|logs|script|copy)\b",
    re.IGNORECASE,
)


#: A word introducing where something goes or comes from. When one of these
#: sits between the verb and its object, the operation is aimed somewhere other
#: than the thing that was named.
_AIMED_ELSEWHERE = re.compile(
    r"^\s*(?:from|into|onto|towards?|out\s+of|across|between)\b",
    re.IGNORECASE,
)

#: Whose verb it is. A request asks HER to do something, so a verb whose
#: subject is the person speaking or a third party is a report of what someone
#: else does — "what approach THEY copy", "what I should copy", "I might copy
#: their layout". Modal words are here because they are how a subject reaches
#: across one: "I might copy" is still I doing it.
_SOMEBODY_ELSE_IS_DOING_IT = re.compile(
    r"\b(?:i|we|they|he|she|someone|somebody|people|users?)\b"
    r"(?:\s+(?:might|may|could|would|should|will|can|do|did|have|had|often|"
    r"usually|already|just|then)){0,3}\s*$",
    re.IGNORECASE,
)

#: What turns the next word into a thing rather than an action. A word after
#: one of these is being named, not done.
_A_THING_FOLLOWS = re.compile(r"(?:^|\s)(?:the|a|an|these|those|this|that|its|their|my|your)\s*$", re.IGNORECASE)


def _asks_to_change_the_named_thing(sanitized: str, original: str) -> bool:
    """Whether a change is being asked FOR, rather than mentioned.

    Four facts about the sentence, none of them a list of words.

    A word after a determiner is a thing, not an action: "the WRITE targets"
    names some targets. A word with no determiner after it is part of a fixed
    expression rather than an operation on something: "does it MAKE sense" acts
    on nothing. A preposition of direction aims the operation somewhere other
    than what was named: "COPY from elsewhere", "COPY into my own project". And
    what is left — a verb with an object and no other place to send it — is a
    change, whether the object is the file itself, a pronoun for it, or
    something inside it: "read it and FIX the bug" changes the file.
    """

    for hit in _FILESYSTEM_MUTATION_RE.finditer(sanitized):
        before = sanitized[: hit.start()]
        if _A_THING_FOLLOWS.search(before[-24:]):
            continue
        if _SOMEBODY_ELSE_IS_DOING_IT.search(before[-40:]):
            continue
        after = sanitized[hit.end() :]
        # A verb does not reach past a clause boundary to find its object.
        stop = re.search(r"[.?!;]|\bthen\b|\bbut\b|\bwhile\b", after)
        reach = after[: stop.start()] if stop else after
        if _AIMED_ELSEWHERE.match(reach):
            continue
        words = reach.split()[:_WITHIN_REACH]
        if not words:
            continue
        window = " ".join(words)
        if _CONCRETE_PATH_RE.search(window) or _NAMED_ON_SURFACE_RE.search(window):
            return True
        if _STANDS_FOR_THE_THING.search(window) and _CONCRETE_PATH_RE.search(before):
            return True
        # An object that is a named thing rather than a bare word. "fix THE
        # bug" acts on something; "make sense" does not.
        if re.match(
            r"\s*(?:the|a|an|this|that|these|those|its|their|my|your|all|every)\b",
            reach,
        ):
            return True
    return False


def looks_like_filesystem_observation(user_message: str) -> bool:
    """True when the turn asks to READ something on disk and report back.

    The read lane is file_operation, whose read/list/exists actions are pure
    observation. Sending that to os_automation asks the screen driver to
    verify an effect no screen will ever show.

    A turn that also asks for a change ("read it and fix the bug") is not an
    observation: only the lane that can write can finish it.

    2026-08-22: this required a verb somebody had listed, so "what's in
    /etc/hosts" and "how many .py files are in <path>?" — both plain reads with
    no verb in them at all — went to the screen driver, while "list the
    contents of ~/Documents" did not. A question about a path is a request to
    look at it, whether or not it names the looking.

    LIVE, 2026-08-28: "Something's off in <path> ... Go through the code and
    tell me what's actually happening, with the file and line" was routed to
    the desktop lane, which planned it as WRITING A DOCUMENT and came back "I
    could not write the words you asked for, so I have not made the file."
    Nothing had asked for a file. The question forms above are anchored to the
    start of the turn, and this one asks in its last clause; none of the listed
    verbs covered going through code and reporting back. The diagnosis engine
    that owns this had the answer — invoice.py:4, a mutable default, with the
    remedy — and was never reached.
    """
    text = normalize_memory_intent_text(user_message).lower()
    if not text:
        return False
    if not _CONCRETE_PATH_RE.search(text) and not _NAMED_ON_SURFACE_RE.search(text):
        return False
    sanitized = strip_negated_action_spans(text).lower()
    # Asking for a change is not observing, whichever way it is phrased — but
    # the verb alone does not say a change was asked for. It says a change was
    # MENTIONED.
    #
    # LIVE, 2026-08-28: four ordinary read requests were classified as changes
    # by one word each. "read <path>/README.md and tell me what approach they
    # COPY from elsewhere"; "go through <path>/Makefile and explain what the
    # WRITE targets do"; "what's in <path>/CLAUDE.md that I should COPY into my
    # own project"; "look at <path>/README.md — does it MAKE sense?". In every
    # one the verb belongs to something other than the thing named.
    #
    # What separates a request to change from a mention of changing is what the
    # verb acts ON. A mutation counts when the thing named is its object.
    if _asks_to_change_the_named_thing(sanitized, text):
        return False
    settled = bool(
        _FILESYSTEM_OBSERVATION_RE.search(sanitized)
        or _ASKS_ABOUT_A_PATH_RE.match(sanitized)
    )
    surface = _looking_surface()
    if surface is None:
        return settled
    if settled:
        try:
            surface.observe(user_message, holds=True)
        except (RuntimeError, TypeError, ValueError):
            pass
        return True
    try:
        return bool(surface.decide_without_waiting(user_message))
    except (RuntimeError, TypeError, ValueError):
        return False


#: A screen referred to as a KIND of thing rather than as the one in front of
#: her. "A screen" and "screens" are the class; "my screen", "the screen" and
#: "this screen" are the one she could look at.
_SCREENS_IN_GENERAL_RE = re.compile(
    r"\b(?:a|an|any|some|every|each)\s+(?:\w+\s+){0,2}"
    r"(?:screen|display|monitor|window|desktop)\b"
    r"|\b(?:screens|displays|monitors|windows|desktops)\b",
    re.IGNORECASE,
)

#: Ways of asking about her method rather than about the world. The subject is
#: her — how she does a thing, what her approach is, where she is weak — and
#: none of them can be answered by looking at anything.
_ABOUT_HER_METHOD_RE = re.compile(
    r"\bhow\s+(?:do|does|did|would|will|can|are)\s+you\b"
    r"|\bhow\s+you\s+(?:currently\s+)?(?:decide|choose|work|handle|approach|do)\b"
    r"|\bwhat(?:'s| is| are)?\s+your\s+(?:approach|method|process|strategy|weakness|strength)"
    r"|\b(?:weaknesses|strengths|limitations|shortcomings)\b[^.?!]{0,40}\byou\b"
    r"|\byou\b[^.?!]{0,40}\b(?:weaknesses|strengths|limitations|shortcomings)\b"
    r"|\bin\s+general\b",
    re.IGNORECASE,
)


#: Ways of putting a CLAIM in front of her rather than a job.
#:
#: A closed class of discourse acts, not topic words: the object of the
#: sentence is an assertion to be discussed, and no amount of doing anything
#: settles it. "Someone claims you're just a chatbot with a screenshot tool"
#: mentions a screenshot and is not a request for one.
_ABOUT_A_CLAIM_RE = re.compile(
    r"\b(?:someone|somebody|people|they|he|she|critics?|a\s+friend|my\s+\w+)\s+"
    r"(?:claims?|says?|said|argues?|reckons?|thinks?|insists?|told\s+me)\b"
    r"|\bi(?:'ve| have)\s+been\s+told\b"
    r"|\bi(?:'m| am)\s+told\b"
    r"|\b(?:rebut|refute|push\s+back\s+on|argue\s+against|respond\s+to\s+(?:the\s+)?claim)\b"
    r"|\bis\s+it\s+true\s+that\b"
    r"|\bdo\s+you\s+(?:agree|disagree)\b"
    r"|\bwhat\s+would\s+you\s+say\s+to\s+(?:that|this|them)\b",
    re.IGNORECASE,
)


def puts_a_claim_to_her(user_message: str) -> bool:
    """True when the message hands her an assertion to discuss, not a job.

    The object of the sentence is a proposition. Nothing she does to the world
    settles it, so a lane that acts is the wrong lane however many capability
    words the claim happens to contain.

    LIVE 2026-08-27: "Someone claims you're just a chatbot with a screenshot
    tool. Rebut that in a few sentences." went to the desktop lane, tried to
    read the screen, and was refused by the executive — because the sentence
    contains the word screenshot.

    A closed class of discourse acts rather than a list of topics, so it holds
    for any capability somebody makes a claim about.
    """
    text = str(user_message or "")
    return bool(text.strip()) and bool(_ABOUT_A_CLAIM_RE.search(text))


def asks_about_screens_in_general(user_message: str) -> bool:
    """True when the request is about screens as a kind, not about this one.

    A question about how she works cannot be answered by looking at anything,
    and routing it to a lane that looks makes her act on the world to answer a
    question about herself.

    LIVE 2026-08-27: "What are the three biggest weaknesses in how you
    currently decide what to do on a screen? Be blunt." was routed to the
    desktop lane, tried to read the screen, was refused by the executive for
    want of scoped authority, and came back as a failure report instead of an
    answer.

    Two things mark it, and either will do. The article: "a screen" and
    "screens" name the class, where "my screen", "the screen" and "this
    screen" name the one she could look at. And the subject: "how do you...",
    "what is your approach", "your weaknesses" are questions about her, which
    no reading answers.
    """
    text = str(user_message or "").lower()
    if not text:
        return False
    if puts_a_claim_to_her(text):
        return True
    if _ABOUT_HER_METHOD_RE.search(text):
        return True
    # An instruction is not a question about a class.
    #
    # The article rule reads "a screen" and "a window" as naming the kind, and
    # that holds for questions — which is what this predicate is for, per the
    # docstring above: "a question about how she works cannot be answered by
    # looking at anything". It does not hold for an imperative. LIVE
    # 2026-08-27: "Open A BROWSER WINDOW, search for climate news, and show me
    # the articles" was read as a question about windows in general and held
    # back from the only lane that could do it.
    try:
        from core.conversation.request_mood import assess_request_mood

        if assess_request_mood(user_message).asks_for_action:
            return False
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        pass
    # The class, and nothing definite alongside it to look at.
    if not _SCREENS_IN_GENERAL_RE.search(text):
        return False
    # A plural can still be definite: "my monitors" and "these windows" are
    # the ones in front of her, however many there are.
    return not re.search(
        r"\b(?:my|the|this|that|these|those|current|your)\s+(?:\w+\s+){0,2}"
        r"(?:screens?|displays?|monitors?|windows?|desktops?)\b",
        text,
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
    if asks_about_screens_in_general(sanitized_text):
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
    "asks_to_build_software",
    "looks_like_filesystem_observation",
    "looks_like_screen_observation",
]


#: Ways of asking to be kept company WHILE something happens.
#:
#: Distinct from asking to be told afterwards. "Tell me when you're done" is a
#: report and can wait; "tell me what you're doing as you go" is company, and
#: company cannot be delivered later.
ACCOMPANIED = (
    r"\bas\s+you\s+(?:go|work|play|do)\b",
    r"\bwhile\s+you\s+(?:go|work|play|do|are)\b",
    r"\bnarrat(?:e|ing)\b",
    r"\bcommentary\b",
    r"\btalk\s+me\s+through\b",
    r"\bwalk\s+me\s+through\b",
    r"\bkeep\s+me\s+posted\b",
    r"\b(?:say|tell\s+me|call\s+out|announce)\s+[^.]{0,40}\b(?:each|every|before\s+each|as\s+each)\b",
    r"\bbefore\s+(?:each|every)\s+\w+\b",
    r"\b(?:step|move|turn)\s+by\s+(?:step|move|turn)\b",
    r"\bwatch\s+you\b",
    r"\blet\s+me\s+(?:see|watch)\b",
    r"\bshow\s+me\s+(?:as|while)\b",
    r"\bout\s+loud\b",
    r"\bthink(?:ing)?\s+out\s+loud\b",
)
_ACCOMPANIED_RE = tuple(re.compile(pattern, re.IGNORECASE) for pattern in ACCOMPANIED)


def asks_to_be_accompanied(user_message: Any) -> bool:
    """Whether the person asked to be with her while she does it.

    A long task is normally handed to the background and answered with a
    receipt, because a person should not sit and wait on something that takes
    minutes. That reasoning inverts the moment they ask to be told what is
    happening as it happens: the telling IS the thing they asked for, and it
    cannot be delivered afterwards.

    LIVE 2026-08-26: "Find 2048 online, play it, and get to a 256 tile. Say
    what you are about to do before each move, and tell me here when you have
    it." came back as "Task accepted into governed background execution. Task
    id: 3f5e2b3b. Commitment id: 2832f808." — a ticket, for a request whose
    whole point was watching her do it.

    General to anything anyone asks to watch: a build, a search, a form, a
    game. Being told afterwards is a report and can wait; being told as it
    happens is company and cannot.
    """
    text = str(user_message or "").strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in _ACCOMPANIED_RE)
