"""Finding out how a thing is done, before and while doing it.

A goal loop that only ever reads the screen in front of it can play a game
badly forever. It has the board, the moves, and what its last few moves led
to, and none of that contains the thing a person would go and look up: that
this kind of task has a known way of being done well.

So a pursuit asks two questions it could not answer from the screen. Have I
done this before, and what happened? And if that is thin — what does anyone
know about doing this? The first comes from her own record, the second from
the search skills she already has.

What comes back is evidence, not instruction. It joins the goal, the reading
and the consequence history as another line the decision rests on, and it is
attributed, so a strategy that turns out to be wrong is traceable to where it
came from rather than folded invisibly into her judgement. Nothing here
phrases anything to steer an answer; a retrieved fact is a fact.

Looking things up is also something she should be able to say she is doing.
Findings go to the workspace, so the narrator can tell you what she read and
what she means to try before she tries it.
"""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.TaskKnowledge")

#: How long a finding stays worth reusing inside one session. Long enough
#: that a run does not search twice for the same goal, short enough that a
#: strategy is re-checked when a task is returned to much later.
KNOWLEDGE_TTL_S = 1800.0
#: How many findings are worth carrying into a decision. Past a handful they
#: stop being evidence and start being a wall of text.
FINDINGS_KEPT = 4
#: Sentences longer than this are prose about a topic rather than a usable
#: statement of how to do it.
MAX_FINDING_CHARS = 240

#: Where an answer came from, because a strategy and a definition are not
#: interchangeable and the difference should survive into the evidence.
BACKGROUND = "background"
TACTIC = "tactic"

#: A question about what something IS. Her offline corpus is a full Wikipedia
#: snapshot, so these are answered instantly, privately, and without a
#: network — and asking the web first would be going outside for something
#: already in the building.
_BACKGROUND_RE = re.compile(
    r"\b(?:what\s+is|what\s+are|who\s+(?:is|was)|when\s+(?:did|was)|"
    r"define|meaning\s+of|history\s+of|background\s+on|tell\s+me\s+about)\b",
    re.IGNORECASE,
)
#: A question about what to DO. An encyclopedia entry describes a game; it
#: does not say what to do when your board is locked. These go to the web,
#: because that is where people write down how things are actually done.
_TACTIC_RE = re.compile(
    r"\b(?:how\s+(?:do|to|can)|what\s+(?:do|should)\s+i|stuck|strategy|"
    r"tactic|technique|tips?|best\s+way|get\s+past|not\s+working|"
    r"do\s+nothing|what\s+to\s+do)\b",
    re.IGNORECASE,
)


def kind_of_question(question: str) -> str:
    """Whether this asks what something is, or what to do about it.

    The distinction decides where to ask. Her offline encyclopedia holds
    6.5 million articles and answers the first kind instantly; it holds
    almost nothing about the second, because an article about a game says
    what the game is and not how to get out of a locked board.
    """
    text = str(question or "")
    if _TACTIC_RE.search(text):
        return TACTIC
    if _BACKGROUND_RE.search(text):
        return BACKGROUND
    return TACTIC


#: Words that mark a sentence as saying HOW to do something, rather than
#: describing what the thing is. A search returns both, and only one of them
#: helps somebody who is mid-task.
#: Modal words that make a sentence advice whatever its shape.
DIRECTIVE = re.compile(
    r"\b(?:keep|hold|always|never|avoid|prefer|try to|make sure|start by|"
    r"the key is|best to|should|must|instead of|rather than|priority|"
    r"strategy|technique|rule of thumb)\b",
    re.IGNORECASE,
)

#: Words that can begin an English sentence WITHOUT it being an instruction.
#:
#: The closed classes — determiners, pronouns, prepositions, conjunctions,
#: subordinators. Anything else at the front of a sentence is almost always a
#: verb, and a sentence that opens with a bare verb is an imperative: someone
#: telling you how to do something. That is a property of the grammar rather
#: than of any subject, which is the point — a list of strategy words knows
#: about games and knows nothing about baking, tax forms or a deployment
#: runbook.
NOT_A_VERB_OPENER = frozenset(
    """a an the this that these those it he she they we i you them us him her
    my your his its our their there here what which who whom whose how why
    where when while if unless until although though because since after
    before during as so but and or yet nor for with from in on at by about
    into onto over under between among against through despite per each every
    most many some all both few several other another such no not only just
    also then thus hence therefore however moreover furthermore meanwhile
    one two three four five six seven eight nine ten first second third last
    next previous new old good best worst more less much little own same
    something anything nothing everything someone anyone everyone""".split()
)

#: Words that only appear after a subject. Their presence in second place
#: means the sentence has one, which an imperative does not.
_AGREES_WITH_A_SUBJECT = frozenset(
    """is are was were be been being has have had can could will would shall
    should may might must does do did seems seemed appears appeared""".split()
)

#: Openers that introduce a rule about what follows from what.
_RULE_OPENER = re.compile(r"^(?:when|if|once|after|whenever|as soon as)\b", re.IGNORECASE)


def says_how(sentence: str) -> bool:
    """Whether this sentence tells you how something is done.

    Three shapes, all grammatical rather than topical. An imperative — a
    sentence that opens with a bare verb — is somebody telling you what to
    do. A directive modality says the same thing inside a longer sentence. A
    conditional with a consequence states a rule of the thing.

    LIVE 2026-08-26: "Use arrow keys to move the tiles" and "When two tiles
    having the same number touch, they join into one" are the whole answer to
    how a game is played, and a list of strategy words matched neither. She
    opened the right page, read four hundred characters of instructions, and
    came away with nothing.
    """
    said = " ".join(str(sentence or "").split())
    if not said:
        return False
    if DIRECTIVE.search(said):
        return True
    if _RULE_OPENER.match(said) and "," in said:
        return True
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", said)
    if not words:
        return False
    if words[0].lower() in NOT_A_VERB_OPENER:
        return False
    # An imperative has no subject, so nothing agrees with anything. A copula
    # or auxiliary in second place means the first word was the subject:
    # "Sourdough is a bread made by fermentation" describes, and "Feed the
    # starter twice a day" instructs, and they are the same shape until here.
    return len(words) < 2 or words[1].lower() not in _AGREES_WITH_A_SUBJECT


@dataclass(frozen=True)
class Finding:
    """One thing she learned about how to do this, and where it came from."""

    says: str
    source: str = ""

    def as_evidence(self) -> str:
        return f"Known about this task — {self.says} ({self.source})" if self.source else f"Known about this task — {self.says}"


@dataclass
class TaskKnowledge:
    """What she knows about doing this, and how she came to know it."""

    goal: str
    findings: list[Finding] = field(default_factory=list)
    from_memory: int = 0
    searched: str = ""
    #: True when this was looked up because the run had stopped getting
    #: anywhere, rather than at the start.
    stuck: bool = False
    #: Which kind of question was asked, and therefore which source answered.
    asking: str = ""
    at: float = field(default_factory=time.time)

    @property
    def known(self) -> bool:
        return bool(self.findings)

    def as_evidence(self) -> list[str]:
        return [finding.as_evidence() for finding in self.findings[:FINDINGS_KEPT]]

    def narrate(self) -> str:
        """What she found and what she means to do with it."""
        if not self.findings:
            if self.stuck:
                return f"This is not working and I could not find out why. I looked up: {self.searched}"
            return f"I could not find anything about how {self.goal[:80]} is usually done."
        lead = self.findings[0].says
        rest = len(self.findings) - 1
        tail = f", and {rest} other thing{'s' if rest > 1 else ''}" if rest > 0 else ""
        opening = "This stopped working, so I looked it up" if self.stuck else "I read that"
        joiner = ": " if self.stuck else " "
        return f"{opening}{joiner}{lead}{tail}. I am going to try that."


_remembered: dict[str, TaskKnowledge] = {}


def _is_a_heading(line: str) -> bool:
    """Whether this is a title or a navigation label rather than a sentence.

    Typography rather than vocabulary, so it holds on any site: headings are
    title-cased and do not end in a full stop, and sentences are neither.
    LIVE 2026-08-26: "2048 - Play the Free Online Game Privacy Policy" was
    read as an instruction, because it opens with a verb and every rule about
    grammar agreed that it did.
    """
    said = " ".join(str(line or "").split())
    if not said:
        return True
    if said[-1] in ".!?:":
        return False
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'\-]*", said) if len(word) > 2]
    if len(words) < 2:
        return True
    capitalised = sum(1 for word in words if word[0].isupper())
    return capitalised >= max(2, len(words) // 2)


def usable_sentences(text: str) -> list[str]:
    """The sentences in a body of text that say how to do something."""
    found: list[str] = []
    for raw in re.split(r"(?<=[.!?])\s+|\n+", str(text or "")):
        sentence = " ".join(raw.split()).strip(" -•*")
        if not sentence or len(sentence) > MAX_FINDING_CHARS or len(sentence) < 20:
            continue
        if _is_a_heading(sentence):
            continue
        if says_how(sentence):
            found.append(sentence)
    return found


def _from_her_own_record(goal: str, *, graph: Any = None) -> list[Finding]:
    """What her consequence graph already says about this kind of task."""
    try:
        if graph is None:
            from core.world_model.acg import acg as graph  # noqa: PLC0415
        rows = graph.query_consequences(goal) or []
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("task_knowledge", exc, severity="info", action="read her own record of this task")
        return []
    findings: list[Finding] = []
    for row in list(rows)[:FINDINGS_KEPT]:
        if not isinstance(row, dict):
            continue
        outcome = str(row.get("outcome") or "").strip()
        if outcome:
            worked = "worked" if row.get("success") else "did not work"
            findings.append(Finding(says=f"last time this {worked}: {outcome}"[:MAX_FINDING_CHARS], source="my own record"))
    return findings


async def _from_her_own_shelf(question: str, *, engine: Any = None) -> list[Finding]:
    """What her offline encyclopedia says, which is instant and needs no network.

    Asked first for a question about what something IS. Going to the web for
    that is going outside for something already in the building — she carries
    a full Wikipedia snapshot and answers from it in tens of milliseconds.
    """
    engine = engine if engine is not None else _capability_engine()
    if engine is None:
        return []
    try:
        result = await engine.execute(
            "local_reference_search",
            {"query": question, "limit": 3},
            {"requested_via": "task_knowledge", "purpose": "background on a task"},
        )
    except (AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
        record_degradation("task_knowledge", exc, severity="info", action="read her own reference shelf")
        return []
    if not isinstance(result, Mapping):
        return []
    findings: list[Finding] = []
    for row in list(result.get("results") or [])[:FINDINGS_KEPT]:
        if not isinstance(row, Mapping):
            continue
        title = str(row.get("title") or "").strip()
        for sentence in usable_sentences(str(row.get("snippet") or ""))[:2]:
            findings.append(Finding(says=sentence, source=f"my own reference shelf: {title}"))
    return findings


def _capability_engine() -> Any:
    try:
        from core.container import get_container  # noqa: PLC0415
        from core.exceptions import ContainerError  # noqa: PLC0415

        try:
            return get_container().get("capability_engine")
        except (ContainerError, KeyError):
            return None
    except (ImportError, AttributeError, RuntimeError):
        return None



#: Words that carry no weight in a query. Every question contains them, so
#: they identify nothing and an engine that ranks on them ranks on noise.
_COMMON = {
    "a", "an", "and", "the", "to", "of", "in", "on", "at", "for", "with", "how",
    "what", "when", "why", "well", "do", "does", "is", "it", "this", "that",
    "my", "your", "get", "go", "play", "make", "keep", "you", "me", "i", "be",
    "can", "will", "should", "best", "good", "way", "ways", "tips", "guide",
}


def _identifying_words(text: str) -> tuple[str, ...]:
    """The words in a question that identify what it is about.

    Not a corpus and not a model — the words everybody's question contains
    cannot be the words that identify anybody's. What is left is the name of
    the thing.
    """
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-]{1,}", str(text or "").lower())
    return tuple(dict.fromkeys(word for word in words if word not in _COMMON))


def _answers_about(result: Mapping[str, Any], marks: Sequence[str]) -> int:
    """How many of the identifying words this result actually mentions."""
    if not marks:
        return 0
    body = f"{result.get('title', '')} {result.get('url', '')} {result.get('snippet', '')}".lower()
    return sum(1 for mark in marks if mark in body)


def _asked_more_than_one_way(question: str) -> list[str]:
    """The same question, phrased the ways a person would retype it.

    One phrasing is a bet on how an engine weights words, and the bet is not
    reliably winnable. LIVE 2026-08-26: "how to play 2048" returned four
    general games portals, because "play" is the commonest word in it and
    "2048" is one token among five. "2048 game strategy" returned the game.
    But a descriptive subject goes the other way — "how to make sourdough
    starter" returns recipes and "sourdough starter how to make well" returns
    an encyclopedia article — so neither order is the right order.

    So she asks both ways and keeps whichever actually came back about the
    thing, which is what a person does when the first search misses.
    """
    plain = " ".join(str(question or "").split())
    if not plain:
        return []
    marks = _identifying_words(plain)
    asked = [plain]
    if marks:
        lead = " ".join(marks)
        if lead.lower() != plain.lower():
            asked.append(lead)
        asked.append(f"{marks[0]} {' '.join(marks[1:])} guide".strip())
    return list(dict.fromkeys(asked))[:3]


async def _search_results_for(question: str, *, engine: Any = None, browser: Any = None) -> list[dict[str, str]]:
    """Candidates for a question, fetched without navigating anywhere.

    Asked more than one way, and kept by whether the answers mention the
    thing being asked about rather than by which phrasing produced them.
    """
    if browser is None:
        try:
            from core.capabilities.browser_controller import get_browser_controller  # noqa: PLC0415

            browser = get_browser_controller()
        except (ImportError, AttributeError, RuntimeError):
            return []
    marks = _identifying_words(question)
    seen: dict[str, dict[str, str]] = {}
    best_score = 0
    for phrasing in _asked_more_than_one_way(question):
        try:
            found = await browser.search_results(phrasing, count=FINDINGS_KEPT)
        except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("task_knowledge", exc, severity="info", action="look for an answer")
            continue
        for result in found or []:
            if not isinstance(result, Mapping):
                continue
            url = str(result.get("url", "") or "")
            if not url or url in seen:
                continue
            score = _answers_about(result, marks)
            if score:
                seen[url] = dict(result)
                best_score = max(best_score, score)
        if best_score >= len(marks) and len(seen) >= FINDINGS_KEPT:
            # This phrasing answered about everything asked. Stop retyping it.
            break
    if seen:
        return sorted(seen.values(), key=lambda row: _answers_about(row, marks), reverse=True)[
            : FINDINGS_KEPT
        ]
    # Nothing came back about the thing. Better to hand back what the plain
    # question returned than to invent relevance that is not there.
    try:
        return await browser.search_results(question, count=FINDINGS_KEPT)
    except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("task_knowledge", exc, severity="info", action="look for an answer")
        return []


#: How many pages she will open before giving up on a question. One page is
#: a bet that the top result is the answer; a person opens the next one.
PAGES_READ = 3


async def _read_the_best_answer(
    results: Sequence[Mapping[str, Any]], question: str, *, browser: Any = None
) -> list[Finding]:
    """Open the most promising results in turn and read what they say.

    A snippet is an advertisement for an answer. Getting the answer means
    opening the page and reading the sentences in it that tell you how the
    thing is done — and, when that page turns out to be a landing page or a
    privacy notice, opening the next one.

    LIVE 2026-08-26: asked how a game is played, the top result was the game
    itself, and the only sentence she came away with was the title of its
    privacy policy. She stopped there because one page was all she read.
    """
    if not results:
        return []
    wanted = _distinctive(question)
    if browser is None:
        try:
            from core.capabilities.browser_controller import get_browser_controller  # noqa: PLC0415

            browser = get_browser_controller()
        except (ImportError, AttributeError, RuntimeError):
            return []
    ranked = sorted(
        results,
        key=lambda row: len(wanted & _distinctive(f"{row.get('title', '')} {row.get('snippet', '')}")),
        reverse=True,
    )
    nothing_readable: list[str] = []
    for candidate in ranked[:PAGES_READ]:
        url = str(candidate.get("url") or "").strip()
        if not url:
            continue
        try:
            extract = await browser.extract_article_text(url)
        except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "task_knowledge", exc, severity="info", action="read a page for an answer"
            )
            continue
        # The field the extractor actually fills. It writes `body`; reading
        # `text` — which is not a field of the type — meant every page she
        # opened came back empty and she carried on knowing nothing, with no
        # error anywhere because nothing had failed.
        body = str(getattr(extract, "body", "") or "")
        if not body or body.startswith("[Extraction failed"):
            nothing_readable.append(url)
            continue
        # The sentences that tell you how this is done, ranked by how much of
        # the question they address — but not required to address it in the
        # question's own words, because a sentence that answers a question
        # rarely repeats it.
        scored = sorted(
            ((len(wanted & _distinctive(line)), line) for line in usable_sentences(body)),
            key=lambda row: row[0],
            reverse=True,
        )
        where = str(getattr(extract, "source_domain", "") or url)
        found = [
            Finding(says=line, source=f"read on {where}")
            for _score, line in scored[:FINDINGS_KEPT]
            if line.strip()
        ]
        if found:
            return found
        nothing_readable.append(url)
    if nothing_readable:
        record_degradation(
            "task_knowledge",
            RuntimeError(f"nothing that says how, on {', '.join(nothing_readable[:3])}"),
            severity="info",
            action="looked for an answer on pages that gave none",
        )
    return []


def _distinctive(text: str) -> set[str]:
    from core.agency.deliberate_action import _distinctive as words  # noqa: PLC0415

    return words(text)


async def _from_search(question: str, *, engine: Any = None) -> tuple[list[Finding], str]:
    """What anyone knows about doing this, through the search skills she has."""
    try:
        if engine is None:
            from core.container import get_container  # noqa: PLC0415
            from core.exceptions import ContainerError  # noqa: PLC0415

            try:
                engine = get_container().get("capability_engine")
            except (ContainerError, KeyError):
                engine = None
        if engine is None:
            return [], ""
        result = await engine.execute(
            "web_search",
            {"query": question, "num_results": 5},
            {"requested_via": "task_knowledge", "purpose": "how a task is done"},
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
        record_degradation("task_knowledge", exc, severity="info", action="looked up how a task is done")
        return [], ""

    text = ""
    if isinstance(result, dict):
        for key in ("summary", "answer", "content", "text", "result"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                text = value
                break
        if not text:
            items = result.get("results") or result.get("items") or []
            if isinstance(items, list):
                text = " ".join(
                    str(item.get("snippet") or item.get("content") or item.get("title") or "")
                    for item in items
                    if isinstance(item, dict)
                )
    findings = [Finding(says=sentence, source="what I read") for sentence in usable_sentences(text)]
    return findings[:FINDINGS_KEPT], question


#: Parts of an instruction that are about the telling rather than the doing.
#:
#: A request carries more than the task: when to stop, what to say while
#: working, who to tell afterwards. None of it is what she needs to look up,
#: and all of it poisons the question she asks.
RIDERS = (
    r"\b(?:and\s+)?(?:then\s+)?tell\s+me\b[^.]*",
    r"\b(?:and\s+)?(?:then\s+)?let\s+me\s+know\b[^.]*",
    r"\b(?:and\s+)?say\s+what\s+you\b[^.]*",
    r"\b(?:and\s+)?narrat\w*\b[^.]*",
    r"\b(?:while|as)\s+you\s+go\b[^.]*",
    r"\bbefore\s+(?:each|every)\s+\w+[^.]*",
    r"\bout\s+loud\b[^.]*",
    r"\bstep\s+by\s+step\b[^.]*",
)
_RIDER_RE = tuple(re.compile(pattern, re.IGNORECASE) for pattern in RIDERS)

#: Where the finishing condition starts. Everything from here on says when to
#: stop, which is hers to check and no use to anyone she asks.
_STOPPING_RE = re.compile(
    r"\b(?:until|till|til|as\s+soon\s+as|once\s+(?:you|it)|when\s+(?:you|it)|"
    r"and\s+get\s+to|and\s+reach|and\s+get\s+me\s+to|up\s+to)\b",
    re.IGNORECASE,
)


def what_it_is_about(goal: str) -> str:
    """The subject of a request, with the instructions to her taken off.

    A request is written to her, and most of it is addressed to her: when to
    stop, what to say while she works, who to tell at the end. Asked as
    written it is not a question anybody can answer.

    LIVE 2026-08-26: researching "play 2048 until you get a 256 tile" asked
    "how to do this well: play 2048 until you get a 256 tile" and came back
    with four dictionary definitions of the word "do". The search had matched
    the only common word in it. She played the whole game knowing nothing.

    General to any task: what is left after the instructions to her is the
    thing itself, and the thing itself is what anyone would look up.
    """
    text = " ".join(str(goal or "").split())
    if not text:
        return ""
    # Her own goal reader has already worked out what this is and where it
    # happens, so use that rather than parsing the sentence a second time.
    # "Find 2048 online, play it, and get to a 256 tile" is a request to play
    # 2048; the finding and the telling are instructions to her.
    try:
        from core.runtime.watched_goal import read_watched_goal  # noqa: PLC0415

        watched = read_watched_goal(text)
    except (ImportError, AttributeError, TypeError, ValueError):
        watched = None
    if watched is not None and watched.where:
        doing = str((watched.detail or {}).get("continuation") or "").strip()
        return " ".join(part for part in (doing, watched.where) if part)[:120]
    cut = _STOPPING_RE.search(text)
    if cut:
        text = text[: cut.start()]
    for pattern in _RIDER_RE:
        text = pattern.sub(" ", text)
    text = re.sub(r"\b(?:go|please|now|then|and|just)\b", " ", text, flags=re.IGNORECASE)
    text = " ".join(text.split()).strip(" ,.;:—-")
    return text[:120]


def how_is_this_done(goal: str) -> str:
    """The question to go and ask about a goal.

    Built from the subject of the goal rather than the instruction, and
    phrased the way a person searching would phrase it. Asking "how to do
    this well: <the whole request>" put every word of the instruction into
    the query, and the engine answered the commonest one.
    """
    subject = what_it_is_about(goal)
    return f"how to {subject}" if subject else ""


def why_is_this_stuck(goal: str, situation: str = "", history: Sequence[Any] = ()) -> str:
    """The question to ask when what she is doing has stopped working.

    A different question from :func:`how_is_this_done`, and the difference is
    the point. "How is this played" returns the beginner's answer she already
    has. What a stuck run needs is the answer to the position it is actually
    in — which moves it has tried, what they did, and what is in front of it.
    That is the question a person types when they are stuck, and it is the
    one that returns something they did not already know.

    Every part is measured: the moves come from the graded history, the
    situation from the last reading. Nothing is characterised or embellished.
    """
    stripped = what_it_is_about(goal)[:100]
    if not stripped:
        return ""
    parts = [f"{stripped} stuck"]

    dead: list[str] = []
    for attempt in list(history)[-6:]:
        option = str(getattr(attempt, "option", "") or "")
        verdict = getattr(attempt, "verdict", None)
        if option and getattr(verdict, "held", True) is False and option not in dead:
            dead.append(option)
    if dead:
        parts.append(f"{' and '.join(dead)} do nothing")

    seen = _salient(situation)
    if seen:
        parts.append(f"showing {seen}")
    parts.append("what to do")
    return ", ".join(parts)[:200]


def _salient(situation: str) -> str:
    """The part of a reading worth putting in a question.

    A whole screen dump makes a search useless. What identifies a position is
    the values on it, so the numbers and short labels are kept and the prose
    around them is dropped.
    """
    text = " ".join(str(situation or "").split())
    if not text:
        return ""
    tokens = re.findall(r"\b[0-9]{1,6}\b", text)
    if tokens:
        # Most recent first would be arbitrary; the largest values are what
        # actually characterise a position.
        largest = sorted({int(token) for token in tokens}, reverse=True)[:6]
        return " ".join(str(value) for value in largest)
    return text[:60]


async def learn_about(
    goal: str,
    *,
    engine: Any = None,
    graph: Any = None,
    search: bool = True,
    remember: bool = True,
    situation: str = "",
    history: Sequence[Any] = (),
    because_stuck: bool = False,
) -> TaskKnowledge:
    """What she knows about doing this, from her own record and from looking.

    Her own record comes first because it is about her, doing this, here.
    Searching happens when that is thin — a run should not go to the network
    to be told something it already learned the hard way.
    """
    goal = str(goal or "").strip()
    knowledge = TaskKnowledge(goal=goal)
    if not goal:
        return knowledge

    held = _remembered.get(goal) if remember else None
    if held is not None and time.time() - held.at < KNOWLEDGE_TTL_S:
        return held

    mine = _from_her_own_record(goal, graph=graph)
    knowledge.findings.extend(mine)
    knowledge.from_memory = len(mine)

    # Being stuck asks a different question, and skips her own record.
    #
    # Her record is what got her here. The whole reason to look again is that
    # what she knows is not enough for the position she is in.
    if because_stuck:
        knowledge.findings.clear()
        knowledge.from_memory = 0
        knowledge.stuck = True

    if search and (because_stuck or len(knowledge.findings) < FINDINGS_KEPT):
        question = (
            why_is_this_stuck(goal, situation, history) if because_stuck else how_is_this_done(goal)
        )
        knowledge.asking = kind_of_question(question)
        knowledge.searched = question
        if knowledge.asking == BACKGROUND:
            knowledge.findings.extend(await _from_her_own_shelf(question, engine=engine))
        if knowledge.asking == TACTIC or not knowledge.findings:
            read, _asked = await _from_search(question, engine=engine)
            if read:
                knowledge.findings.extend(read)
            else:
                # A search that returned only headlines is not an answer.
                # Open the best match and read it.
                results = await _search_results_for(question, engine=engine)
                knowledge.findings.extend(await _read_the_best_answer(results, question))

    if remember:
        _remembered[goal] = knowledge
    _announce(knowledge)
    return knowledge


def forget_everything() -> None:
    """Drop what is held, so a later run learns again."""
    _remembered.clear()


def _announce(knowledge: TaskKnowledge) -> None:
    """Offer what she found to the workspace, so she can say it."""
    if not knowledge.known:
        return
    try:
        from core.consciousness.global_workspace import ContentType  # noqa: PLC0415
        from core.container import ServiceContainer  # noqa: PLC0415

        workspace = ServiceContainer.get("global_workspace", default=None)
        if workspace is None:
            return
        publish = getattr(workspace, "publish", None)
        if publish is None:
            return
        import asyncio  # noqa: PLC0415

        coroutine = publish(
            # Worth interrupting for: she is about to do something differently
            # from how she was doing it, and the person should hear why.
            priority=0.75,
            source="agency.learned",
            payload={
                "schema": "aura.decision.v1",
                "learned": {
                    "goal": knowledge.goal,
                    "findings": [finding.says for finding in knowledge.findings[:FINDINGS_KEPT]],
                    "from_memory": knowledge.from_memory,
                    "searched": knowledge.searched,
                },
            },
            reason=knowledge.narrate(),
            content_type=ContentType.MEMORIAL,
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            coroutine.close()
            return
        from core.utils.task_tracker import get_task_tracker

        task = get_task_tracker().track(coroutine, name="task_knowledge.learn")
        task.add_done_callback(lambda done: done.exception())
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("task_knowledge", exc, severity="info", action="learned without saying so")



@dataclass(frozen=True)
class Implication:
    """What a thing she read means for the position she is actually in."""

    finding: str
    means: str
    favours: str = ""
    #: What she has already measured that bears on this advice. Empty when
    #: nothing does.
    against: str = ""

    @property
    def holds_up(self) -> bool:
        """Whether her own evidence supports acting on this."""
        return not self.against

    def as_evidence(self) -> str:
        line = f"What that means here — {self.means}"
        if self.favours:
            line = f"{line} (so: {self.favours})"
        if self.against:
            line = f"{line}. I am not taking that at face value: {self.against}"
        return line


def _favoured_option(finding: str, options: Sequence[Any]) -> str:
    """Which available move a finding points at, when it points at one.

    An option's own name appearing in the advice is the strongest signal
    there is, and it is checked directly rather than through the distinctive
    words — move names are short, and a filter that drops words under three
    letters drops "up" and "go" along with "the".
    """
    text = str(finding or "")
    if not text or not options:
        return ""
    # The same job choose_named already does, and for the same reason: a
    # sentence that works through the moves before settling on one mentions
    # several, so the last one named is the one it settled on. "Pressing up
    # would dislodge it — go left instead" favours left.
    from core.agency.deliberate_action import choose_named  # noqa: PLC0415

    named = choose_named(text, options)
    if named is not None:
        return str(getattr(named, "name", ""))
    said = _distinctive(text)
    if not said:
        return ""
    best_name, best_score = "", 0
    for option in options:
        described = _distinctive(f"{getattr(option, 'name', '')} {getattr(option, 'detail', '')}")
        score = len(said & described)
        if score > best_score:
            best_name, best_score = str(getattr(option, "name", "")), score
    return best_name


def _structural_meaning(finding: str, situation: str, options: Sequence[Any]) -> Implication:
    """What a finding means here, worked out without asking anything.

    Weaker than reasoning about it, and honest about that: it says which
    available move the advice names, and otherwise says it does not fit what
    is on screen. A floor, so that losing language costs her the quality of
    the thinking and not the step itself.
    """
    favours = _favoured_option(finding, options)
    shared = _distinctive(finding) & _distinctive(situation)
    if favours:
        means = f"it names {favours}, which is available now"
    elif shared:
        means = "it is about " + ", ".join(sorted(shared)[:3]) + ", which is on screen"
    else:
        means = "nothing on screen matches what it describes"
    return Implication(finding=finding, means=means, favours=favours)


def _meaning_question(finding: str, situation: str, options: Sequence[Any]) -> str:
    names = ", ".join(str(getattr(option, "name", "")) for option in options)
    return (
        "Given what is on screen, say in one sentence what this means for the "
        f"move to make next, and name one of: {names}."
    )


def _measured_against(favours: str, history: Sequence[Any]) -> str:
    """What she has already watched happen that bears on this advice.

    Advice is somebody else's experience of a task in general. Her own
    measurements are of this task, now, and when the two disagree the
    measurement wins — it is about the position she is actually in.

    A move she has watched change nothing twice does not become a good move
    because a page recommends it. Saying so is the difference between reading
    advice and following it.
    """
    if not favours:
        return ""
    broke = [
        attempt
        for attempt in history
        if str(getattr(attempt, "option", "")) == favours
        and getattr(getattr(attempt, "verdict", None), "held", True) is False
    ]
    if len(broke) < 2:
        return ""
    verdict = getattr(broke[-1], "verdict", None)
    why = verdict.why() if verdict is not None and hasattr(verdict, "why") else ""
    return f"{favours} has already been tried {len(broke)} times here and {why or 'did nothing'}"


def _appraised(meaning: Implication, history: Sequence[Any]) -> Implication:
    """The same reading, with her own evidence about it attached."""
    against = _measured_against(meaning.favours, history)
    if not against:
        return meaning
    return Implication(
        finding=meaning.finding, means=meaning.means, favours=meaning.favours, against=against
    )


async def work_out_what_it_means(
    knowledge: TaskKnowledge,
    situation: str,
    options: Sequence[Any] = (),
    *,
    think: Any = None,
    history: Sequence[Any] = (),
) -> list[Implication]:
    """Work out what she read against the position she is actually in.

    Retrieving advice is not applying it. "Keep your largest tile in a corner"
    is a fact about the game; what it means here depends on where the tiles
    actually are, and that comparison is the step between reading something
    and playing differently.

    Reasoned when language is reachable and derived structurally when it is
    not, so the step always happens and only its quality varies.
    """
    if not knowledge.known:
        return []
    meanings: list[Implication] = []
    for finding in knowledge.findings[:FINDINGS_KEPT]:
        said = finding.says
        if think is None:
            meanings.append(_appraised(_structural_meaning(said, situation, options), history))
            continue
        try:
            reply = await think(
                _meaning_question(said, situation, options),
                [f"What I read — {said}", f"What is on screen — {situation}"]
                + [f"Available move — {getattr(option, 'label', lambda: '')()}" for option in options],
            )
        except (RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as exc:
            record_degradation(
                "task_knowledge", exc, severity="info", action="worked out what a finding meant without language"
            )
            meanings.append(_appraised(_structural_meaning(said, situation, options), history))
            continue
        spoken = " ".join(str(reply or "").split())[:MAX_FINDING_CHARS]
        if not spoken:
            meanings.append(_appraised(_structural_meaning(said, situation, options), history))
            continue
        meanings.append(
            _appraised(
                Implication(finding=said, means=spoken, favours=_favoured_option(spoken, options)),
                history,
            )
        )
    return meanings


def stuck(history: Sequence[Any], *, run_of: int = 3) -> bool:
    """Whether the last few moves have stopped getting anywhere.

    A run of predictions that all broke means what she is doing is not
    working, whatever the reason. That is the moment to go and find out how
    the task is done rather than to keep pressing.
    """
    recent = list(history)[-run_of:]
    if len(recent) < run_of:
        return False
    return all(getattr(getattr(attempt, "verdict", None), "held", True) is False for attempt in recent)
