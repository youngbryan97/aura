"""Match a request to a skill from what the skill itself says it does.

LIVE DEFECT, 2026-08-19. Asked "run a tiny bit of python and give me the
actual number it printed", with ``code_repl`` READY in the registry, nothing
was dispatched and the answer was invented. The reason was not the model. The
router asks ``CapabilityEngine.detect_intent``, which matches a hand-written
list of phrases, and ``code_repl``'s list reads::

    run (?:this )?(?:python )?code
    execute (?:this )?(?:python )?(?:code|script)

Four of five ordinary ways to ask missed. "run this code" hits; "run a tiny
bit of python", "can you actually run some python for me", "use your
interpreter and tell me what 2**40 is" and "execute a quick script" all miss.
Two separate faults, both structural:

* the modifier slot is hard-coded as ``(?:this )?``, so any real adjective
  between the verb and its object breaks the match, and
* ``python`` is written as an OPTIONAL modifier of a REQUIRED ``code``, so
  naming the language instead of the word "code" cannot match at all.

Adding more phrases is the same bug again with a longer list, and it leaves
the 37 registered skills that have no patterns whatsoever unreachable by
intent under any phrasing. The skill already declares what it does — its name
and its description — and that declaration is the only thing that must stay
true when a skill changes. Reading the request against the declaration means
a new skill is reachable the moment it is registered, with nothing to
maintain in a second place.

The match needs a verb AND an object, because either alone is noise: "run"
appears in "my code doesn't run" and "python" appears in "pythons are
constrictors". Verbs are compared by CLASS, so "run" reaches a skill that
declared "execute" — a genuine closed lexical class, not a per-skill phrase
list. Objects are compared by distinctiveness measured over the live
catalogue, so "code" selects and "time" does not, without anyone tuning it.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from functools import lru_cache

__all__ = [
    "declared_vocabulary",
    "distinctive_objects",
    "rank_declaration_matches",
    "request_matches_declaration",
    "verb_class_of",
]

#: Words that carry no intent. Kept small on purpose: an over-eager stop list
#: silently deletes the one distinctive noun a niche skill owns.
_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "any", "are", "as", "at", "be", "been", "by",
        "can", "could", "do", "does", "for", "from", "has", "have", "how",
        "in", "into", "is", "it", "its", "me", "my", "of", "on", "one", "or",
        "our", "out", "over", "own", "per", "so", "some", "such", "than",
        "that", "the", "their", "them", "then", "there", "these", "they",
        "this", "those", "to", "up", "us", "use", "used", "using", "via",
        "was", "were", "what", "when", "where", "which", "while", "who",
        "will", "with", "without", "you", "your",
        # Marketing adjectives that appear in a third of the descriptions and
        # therefore separate nothing.
        "advanced", "full", "general", "new", "real", "simple", "unified",
    }
)

#: Verbs that mean the same act. A closed class in the linguistic sense: each
#: set is one thing a person can ask for, spelled the several ways English
#: spells it. This is the ONLY place a synonym is written down, and it is
#: indexed by act rather than by skill, so it does not grow with the catalogue.
_VERB_CLASSES: tuple[frozenset[str], ...] = (
    frozenset(
        {
            "run", "runs", "running", "execute", "executes", "executing",
            "eval", "evaluate", "evaluates", "evaluating", "interpret",
            "interprets", "compute", "computes", "computing", "calculate",
            "calculates", "repl", "exec",
        }
    ),
    frozenset(
        {
            "search", "searches", "searching", "find", "finds", "finding",
            "look", "looks", "looking", "google", "browse", "browses",
            "browsing", "lookup", "research", "researches", "researching",
            "fetch", "fetches", "fetching", "retrieve", "retrieves",
        }
    ),
    frozenset(
        {
            "open", "opens", "opening", "launch", "launches", "launching",
            "start", "starts", "starting", "boot", "boots", "booting",
        }
    ),
    frozenset(
        {
            "write", "writes", "writing", "create", "creates", "creating",
            "make", "makes", "making", "generate", "generates", "generating",
            "produce", "produces", "producing", "build", "builds", "building",
            "draft", "drafts", "drafting", "compose", "composes", "render",
            "renders", "rendering", "draw", "draws", "drawing", "paint",
        }
    ),
    frozenset(
        {
            "read", "reads", "reading", "show", "shows", "showing",
            "display", "displays", "displaying", "print", "prints",
            "printing", "see", "sees", "seeing", "watch",
            "observe", "observes", "observing", "inspect", "inspects",
        }
    ),
    frozenset(
        {
            "remember", "remembers", "remembering", "recall", "recalls",
            "recalling", "store", "stores", "storing", "save", "saves",
            "saving", "memorise", "memorize", "log", "logs", "logging",
        }
    ),
    frozenset(
        {
            "send", "sends", "sending", "email", "emails", "emailing",
            "message", "messages", "messaging", "post", "posts", "posting",
            "reply", "replies", "notify", "notifies", "notifying",
        }
    ),
    frozenset(
        {
            "install", "installs", "installing", "add", "adds", "adding",
            "enable", "enables", "enabling", "register", "registers",
        }
    ),
    frozenset(
        {
            "stop", "stops", "stopping", "close", "closes", "closing",
            "quit", "quits", "quitting", "kill", "kills", "killing",
            "disable", "disables", "disabling", "remove", "removes",
        }
    ),
    frozenset(
        {
            "change", "changes", "changing", "edit", "edits", "editing",
            "modify", "modifies", "modifying", "update", "updates",
            "updating", "fix", "fixes", "fixing", "improve", "improves",
            "improving", "refactor", "refactors", "rewrite", "rewrites",
        }
    ),
    frozenset(
        {
            "plan", "plans", "planning", "schedule", "schedules",
            "scheduling", "organise", "organize", "arrange", "arranges",
        }
    ),
    frozenset(
        {
            "analyse", "analyze", "analyses", "analyzes", "analysing",
            "analyzing", "check", "checks", "checking", "test", "tests",
            "testing", "verify", "verifies", "verifying", "audit", "audits",
            "measure", "measures", "measuring", "diagnose", "diagnoses",
        }
    ),
)

#: Domains, spelled the several ways English spells them. Same shape and same
#: justification as the verb classes: indexed by the domain rather than by the
#: skill, so it does not grow as the catalogue does. Nouns are an open class in
#: general, but the domains a machine can act IN are few and stable, and
#: without this a declaration saying "code" cannot hear a request saying
#: "script" — which was half the live misses.
_OBJECT_CLASSES: tuple[frozenset[str], ...] = (
    frozenset({"code", "script", "snippet", "program", "programme", "python",
               "repl", "interpreter", "sandbox", "expression", "function"}),
    frozenset({"web", "online", "internet", "google", "browser", "site",
               "website", "url", "page"}),
    frozenset({"image", "images", "picture", "photo", "illustration",
               "artwork", "drawing", "painting", "diagram"}),
    frozenset({"screen", "display", "desktop", "window", "monitor"}),
    frozenset({"file", "files", "document", "folder", "directory", "path"}),
    frozenset({"memory", "memories", "recollection", "note", "notes"}),
    frozenset({"time", "clock", "date", "hour", "day"}),
    frozenset({"email", "mail", "message", "messages", "text", "dm"}),
    frozenset({"voice", "speech", "audio", "sound", "microphone"}),
    frozenset({"package", "library", "dependency", "module"}),
)

#: Asking for a thing, rather than mentioning it.
#:
#: "my code doesn't run anymore" contains the verb and the object of a request
#: to execute code, in one clause, and is a complaint. "I use python at work"
#: is a fact about a person. Mood is what separates these from a request, and
#: mood is readable: a request puts its verb at the head of the clause, or
#: addresses the listener before it, or reaches it through an infinitive.
#:
#: Past tense is deliberately absent from every verb class above for the same
#: reason. "I ran the numbers" reports; "run the numbers" asks.
_ADDRESSES_THE_LISTENER = frozenset({"you", "your", "yours", "please", "u"})

#: Verbs with no content of their own, which borrow the act from their object:
#: "use the web" asks for whatever the web is for.
_PRO_VERBS = frozenset({"use", "using"})

_NEGATION = frozenset(
    {
        "not", "no", "never", "cant", "cannot", "wont", "dont", "doesnt",
        "didnt", "isnt", "arent", "wasnt", "werent", "couldnt", "wouldnt",
        "shouldnt", "hasnt", "havent", "aint", "failed", "fails", "failing",
        "broken", "broke", "stopped",
    }
)

_WORD_RE = re.compile(r"[a-z0-9_+#.]+")
#: Sentence and clause boundaries. A verb in one clause does not govern an
#: object in the next: "I ran a marathon, then wrote some code" is not a
#: request to execute anything.
_CLAUSE_SPLIT_RE = re.compile(r"[.!?;\n]+|,\s+(?:then|and then|after|before)\b|\bthen\b")


@lru_cache(maxsize=4096)
def verb_class_of(word: str) -> frozenset[str]:
    """Every spelling of the act this word names, or empty if it names none."""
    lowered = str(word or "").strip().lower()
    if not lowered:
        return frozenset()
    for members in _VERB_CLASSES:
        if lowered in members:
            return members
    return frozenset()


@lru_cache(maxsize=4096)
def object_class_of(word: str) -> frozenset[str]:
    """Every word for the domain this one names, or empty if it names none."""
    lowered = str(word or "").strip().lower()
    if not lowered:
        return frozenset()
    for members in _OBJECT_CLASSES:
        if lowered in members:
            return members
    return frozenset()


@lru_cache(maxsize=4096)
def _act_named_by(word: str) -> frozenset[str]:
    """The act a word names, reading an agentive noun back to its verb.

    An "interpreter" is the thing that interprets and a "renderer" is the
    thing that renders, so a request that names the instrument has named the
    act. Live, "use your interpreter and tell me what 2**40 is" named the
    instrument and nothing else, and matched nothing.
    """
    lowered = str(word or "").strip().lower()
    direct = verb_class_of(lowered)
    if direct:
        return direct
    for suffix in ("ers", "ors", "er", "or"):
        if lowered.endswith(suffix) and len(lowered) > len(suffix) + 2:
            stem = lowered[: -len(suffix)]
            found = verb_class_of(stem) or verb_class_of(stem + "e")
            if found:
                return found
    return frozenset()


def _words(text: object) -> list[str]:
    return _WORD_RE.findall(str(text or "").lower().replace("_", " "))


@lru_cache(maxsize=8192)
def _fold(word: str) -> str:
    """One spelling for a noun's singular and plural.

    A declaration saying "circuits" could not hear a request saying "circuit",
    and the same gap sat between images/image, files/file and memories/memory.
    Deliberately light: an aggressive stemmer merges words that mean different
    things, and every merge here is a chance to dispatch the wrong skill.
    """
    lowered = str(word or "").strip().lower()
    if len(lowered) > 4 and lowered.endswith("ies"):
        return lowered[:-3] + "y"
    if len(lowered) > 3 and lowered.endswith("es") and lowered[-3] in "sxzh":
        return lowered[:-2]
    if len(lowered) > 3 and lowered.endswith("s") and not lowered.endswith("ss"):
        return lowered[:-1]
    return lowered


def _asks_rather_than_mentions(clause_words: list[str], verb_positions: set[int]) -> bool:
    """True when one of these verbs sits where a request puts its verb."""
    for index in sorted(verb_positions):
        preceding = clause_words[:index]
        if any(word.replace("'", "") in _NEGATION for word in preceding):
            continue
        if index == 0:
            return True
        if any(word in _ADDRESSES_THE_LISTENER for word in preceding):
            return True
        if clause_words[index - 1] == "to":
            return True
    return False


@lru_cache(maxsize=2048)
def declared_vocabulary(name: str, description: str) -> tuple[frozenset[str], frozenset[str]]:
    """Split a skill's own declaration into (the acts it does, what it acts on).

    The name counts as declaration too, and carries more than the prose does:
    ``code_repl`` names its object twice before the description says anything.
    """
    verbs: set[str] = set()
    objects: set[str] = set()
    described = _words(description)
    for word in _words(name) + described:
        if len(word) < 2 or word in _STOP_WORDS:
            continue
        if verb_class_of(word):
            verbs.add(word)
        else:
            objects.add(word)
    # A skill description is a verb phrase — "Execute Python code", "Search
    # the web", "Simulate quantum circuits" — so its first content word is the
    # act even when no class covers that act. Without this, a skill whose verb
    # is simply uncommon declares no verbs at all and stays unreachable, which
    # is the failure this module exists to remove.
    for word in described:
        if len(word) < 2 or word in _STOP_WORDS:
            continue
        if not verb_class_of(word):
            verbs.add(word)
            objects.discard(word)
        break
    return frozenset(verbs), frozenset(objects)


def distinctive_objects(
    catalogue: Mapping[str, tuple[frozenset[str], frozenset[str]]],
    *,
    keep_fraction: float = 0.5,
) -> dict[str, frozenset[str]]:  # noqa: D401 - described below
    """Keep the nouns that actually separate one skill from the rest.

    Measured over the catalogue that is really registered rather than chosen
    by hand, so a word stops selecting as soon as enough skills claim it. Every
    skill keeps at least its most distinctive noun: a skill whose whole
    vocabulary is common words would otherwise be unreachable, which is the
    failure this module exists to remove.
    """
    if not catalogue:
        return {}
    document_count = len(catalogue)
    frequency: Counter[str] = Counter()
    for _verbs, objects in catalogue.values():
        frequency.update(objects)

    def weight(word: str) -> float:
        return math.log(document_count / (1 + frequency[word]))

    cutoff = math.log(document_count / (1 + document_count * keep_fraction))
    kept: dict[str, frozenset[str]] = {}
    for skill, (_verbs, objects) in catalogue.items():
        selective = {word for word in objects if weight(word) > cutoff}
        if not selective and objects:
            # Every skill needs something to be selected by. A skill's own
            # name is unique and therefore always scores highest, which would
            # make the fallback pick "lab" out of `quantum_lab` and ignore
            # "quantum" and "circuits" — and the name is already matched by
            # name elsewhere. Prefer a word the skill said about itself.
            own_name = set(_words(skill))
            described = objects - own_name
            selective = {max(described or objects, key=weight)}
        kept[skill] = frozenset(selective)
    return kept


def request_matches_declaration(
    message: object,
    *,
    verbs: Iterable[str],
    objects: Iterable[str],
) -> bool:
    """True when one clause asks for one of these acts on one of these things.

    Both halves are required. A verb alone matches "my code doesn't run" and
    an object alone matches "pythons are constrictors"; together, inside one
    clause, they are a request.
    """
    body = str(message or "").strip().lower()
    if not body:
        return False
    wanted_verbs: set[str] = set()
    for verb in verbs:
        wanted_verbs |= verb_class_of(verb) or {str(verb).strip().lower()}
    wanted_objects: set[str] = set()
    for item in objects:
        word = str(item).strip().lower()
        if word:
            wanted_objects |= {
                _fold(member) for member in (object_class_of(word) or {word})
            }
    if not wanted_verbs or not wanted_objects:
        return False

    for clause in _CLAUSE_SPLIT_RE.split(body):
        present = _words(clause)
        if not present:
            continue
        verb_positions = {
            index
            for index, word in enumerate(present)
            if word in wanted_verbs
            or word in _PRO_VERBS
            or (_act_named_by(word) & wanted_verbs)
        }
        if not verb_positions:
            continue
        # An agentive noun names the instrument as well as the act, so it
        # satisfies both halves on its own; anything else needs a separate
        # object, since a bare verb matches far too much.
        names_the_instrument = any(
            _act_named_by(present[index]) & wanted_verbs
            and not verb_class_of(present[index])
            for index in verb_positions
        )
        if not names_the_instrument and not ({_fold(w) for w in present} & wanted_objects):
            continue
        if _asks_rather_than_mentions(present, verb_positions):
            return True
    return False

def rank_declaration_matches(
    message: object,
    catalogue: Mapping[str, tuple[frozenset[str], frozenset[str]]],
    selective: Mapping[str, frozenset[str]],
) -> list[tuple[str, float]]:
    """Matching skills, most specific first.

    Several skills can honestly match one request — "run some python" reaches
    the REPL, the sandbox and anything else declaring code. Picking the first
    by dictionary order would make the choice depend on registration order, so
    rank by how much of the request each skill actually accounts for.
    """
    body = str(message or "").strip().lower()
    if not body:
        return []
    present = {_fold(word) for word in _words(body)}
    scored: list[tuple[str, float]] = []
    for name, (verbs, _declared) in catalogue.items():
        objects = selective.get(name, frozenset())
        if not request_matches_declaration(body, verbs=verbs, objects=objects):
            continue
        covered = len({_fold(word) for word in objects} & present)
        # A skill that declared the very act being asked for accounts for more
        # of the request than one that merely shares a noun with it.
        acts = {member for verb in verbs for member in (verb_class_of(verb) or {verb})}
        said = set(_words(body))
        # Naming the instrument names the act: a request that says
        # "interpreter" has said as much about a REPL as one saying "run".
        spoken = len(acts & said) + sum(
            1 for word in said if not verb_class_of(word) and (_act_named_by(word) & acts)
        )
        named = 1.5 if all(part in present for part in _words(name)) else 0.0
        scored.append((name, float(covered) + named + (0.75 * spoken)))
    scored.sort(key=lambda row: (-row[1], row[0]))
    return scored
