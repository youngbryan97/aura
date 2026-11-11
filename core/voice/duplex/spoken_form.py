"""Text a person would say, from text a model wrote.

A synthesiser pronounces characters. It has no idea that "$1.5B" is four words,
that "e.g." is two, that "2026-07-27" is a date, or that "CPU" is spelled out
while "NASA" is not. Kokoro in particular takes plain text and no SSML at all —
there is no markup channel to say "pause here" or "read this as a number" — so
everything the listener hears is decided by the characters handed over.

That makes this the highest-leverage audible layer in the lane, and the one that
was missing. Prosody carries her mood, paralinguistics reads the listener's, and
both are wasted on a clause that says "colon slash slash".

Two jobs.

**Say it the way it is said.** Numbers, money, ranges, times, dates, units,
symbols, abbreviations and URLs become the words a person uses for them. This
is not decoration: a mispronounced figure is a wrong answer that happens to be
audible.

**Pace it with the only instrument available.** With no SSML, punctuation IS
the prosody control — a comma is a short breath to the model, a full stop is a
longer one. So a clause that would be read as one flat run gets the punctuation
a speaker's pauses would have put there. Used sparingly: inserted punctuation
changes intonation, and too much makes her sound halting rather than thoughtful.

Everything here is reversible in the sense that matters — it changes only how a
clause is *pronounced*, never what it claims. Nothing in this module may alter a
number's value, drop a negation, or reorder a sentence.
"""
from __future__ import annotations

import re

# Spelled out letter by letter. Anything not here that is all-caps is assumed to
# be a word ("NASA", "RAM"), which is the commoner case.
_INITIALISMS: frozenset[str] = frozenset(
    {
        "AI", "API", "CPU", "GPU", "RAM", "SSD", "HDD", "URL", "HTTP", "HTTPS",
        "PDF", "CSV", "JSON", "XML", "HTML", "CSS", "SQL", "USB", "PC", "OS",
        "UI", "UX", "ID", "IP", "TV", "DVD", "FBI", "CIA", "NBA", "NFL", "UK",
        "USA", "EU", "PhD", "CEO", "CTO", "CFO", "HR", "PR", "QA", "LLM", "GPT",
        "TTS", "ASR", "MLX", "RSI", "IO", "AM", "PM", "DNA", "RNA", "GDP",
    }
)

_UNITS: dict[str, tuple[str, str]] = {
    "GB": ("gigabyte", "gigabytes"),
    "MB": ("megabyte", "megabytes"),
    "KB": ("kilobyte", "kilobytes"),
    "TB": ("terabyte", "terabytes"),
    "kg": ("kilogram", "kilograms"),
    "km": ("kilometre", "kilometres"),
    "cm": ("centimetre", "centimetres"),
    "mm": ("millimetre", "millimetres"),
    "ms": ("millisecond", "milliseconds"),
    "Hz": ("hertz", "hertz"),
    "kHz": ("kilohertz", "kilohertz"),
    "GHz": ("gigahertz", "gigahertz"),
    "mph": ("mile per hour", "miles per hour"),
    "%": ("percent", "percent"),
}

_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("&", " and "),
    ("@", " at "),
    ("#", " number "),
    ("~", " about "),
    ("=", " equals "),
    ("+", " plus "),
    ("/", " slash "),
    ("|", " "),
    ("*", " "),
    ("_", " "),
    ("^", " to the power of "),
    ("<", " less than "),
    (">", " greater than "),
)

_ABBREVIATIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\be\.g\.", re.IGNORECASE), "for example"),
    (re.compile(r"\bi\.e\.", re.IGNORECASE), "that is"),
    (re.compile(r"\betc\.?", re.IGNORECASE), "and so on"),
    (re.compile(r"\bvs\.?", re.IGNORECASE), "versus"),
    (re.compile(r"\bapprox\.?", re.IGNORECASE), "approximately"),
    (re.compile(r"\bw/(?=\s)"), "with"),
    (re.compile(r"\bI/O\b"), "I O"),
    (re.compile(r"\baka\b", re.IGNORECASE), "also known as"),
    (re.compile(r"\bFYI\b"), "just so you know"),
    (re.compile(r"\bASAP\b"), "as soon as possible"),
)

_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS = (
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
)
_SCALES = ((1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand"))

_ORDINALS = {
    1: "first", 2: "second", 3: "third", 5: "fifth", 8: "eighth",
    9: "ninth", 12: "twelfth",
}


def _small_number_words(value: int) -> str:
    if value < 20:
        return _ONES[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")
    hundreds, rest = divmod(value, 100)
    head = f"{_ONES[hundreds]} hundred"
    return f"{head} and {_small_number_words(rest)}" if rest else head


def number_words(value: int) -> str:
    """An integer as it is spoken. Negative and large values included."""
    if value < 0:
        return "minus " + number_words(-value)
    if value < 100:
        return _small_number_words(value)
    for scale, name in _SCALES:
        if value >= scale:
            count, rest = divmod(value, scale)
            head = f"{number_words(count)} {name}"
            return f"{head} {number_words(rest)}" if rest else head
    return _small_number_words(value)


def _year_words(year: int) -> str:
    """Years are said in pairs — "twenty twenty-six", not "two thousand and…"."""
    if 1100 <= year < 2000 or 2010 <= year < 3000:
        high, low = divmod(year, 100)
        if low == 0:
            return f"{number_words(high)} hundred"
        return f"{number_words(high)} {_small_number_words(low)}"
    return number_words(year)


def _ordinal_words(value: int) -> str:
    if value in _ORDINALS:
        return _ORDINALS[value]
    if value < 20 or value % 10 == 0 or value % 10 not in _ORDINALS:
        base = number_words(value)
        if base.endswith("y"):
            return base[:-1] + "ieth"
        return base + "th"
    head, tail = divmod(value, 10)
    return f"{_TENS[head]}-{_ORDINALS[tail]}"


def _decimal_words(text: str) -> str:
    whole, _, fraction = text.partition(".")
    spoken = number_words(int(whole or 0))
    if fraction:
        digits = " ".join(_ONES[int(digit)] for digit in fraction)
        spoken = f"{spoken} point {digits}"
    return spoken


def _money(match: re.Match[str]) -> str:
    amount = match.group("amount").replace(",", "")
    suffix = (match.group("suffix") or "").lower()
    currency = {"$": "dollar", "£": "pound", "€": "euro"}[match.group("symbol")]
    scale = {"k": "thousand", "m": "million", "b": "billion", "t": "trillion"}.get(suffix)
    spoken = _decimal_words(amount)
    plural = "" if amount in {"1", "1.0"} and not scale else "s"
    if scale:
        return f"{spoken} {scale} {currency}{plural}"
    return f"{spoken} {currency}{plural}"


def _clock(match: re.Match[str]) -> str:
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    meridiem = (match.group("meridiem") or "").strip().lower().replace(".", "")
    if minute == 0:
        spoken = f"{number_words(hour)} o'clock" if not meridiem else number_words(hour)
    elif minute < 10:
        spoken = f"{number_words(hour)} oh {_ONES[minute]}"
    else:
        spoken = f"{number_words(hour)} {_small_number_words(minute)}"
    if meridiem:
        spoken += " a.m." if meridiem == "am" else " p.m."
    return spoken


def _iso_date(match: re.Match[str]) -> str:
    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return match.group(0)
    return f"the {_ordinal_words(day)} of {_MONTHS[month - 1]}, {_year_words(year)}"


def _url(match: re.Match[str]) -> str:
    """A spoken URL is its host. Nobody reads a path aloud."""
    host = match.group("host").removeprefix("www.")
    return host.replace(".", " dot ")


_MONEY_RE = re.compile(
    r"(?P<symbol>[$£€])(?P<amount>\d[\d,]*(?:\.\d+)?)(?P<suffix>[kKmMbBtT])?\b"
)
# The space before a.m. or p.m. belongs to it: matched on its own, it was eaten when none followed, and LIVE 11 November
# 2026 "just past 9:45 on Wednesday" was spoken "nine forty-fiveon Wednesday". "am" must also end there ("9:45 amazing").
_CLOCK_RE = re.compile(
    r"\b(?P<hour>[01]?\d|2[0-3]):(?P<minute>[0-5]\d)(?:\s*(?P<meridiem>[ap]\.?m\.?)(?![a-z]))?",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_URL_RE = re.compile(r"\bhttps?://(?P<host>[\w.-]+)(?:/\S*)?", re.IGNORECASE)
_BARE_URL_RE = re.compile(r"\b(?P<host>(?:www\.)[\w.-]+)(?:/\S*)?", re.IGNORECASE)
_RANGE_RE = re.compile(r"\b(\d+)\s*[-–]\s*(\d+)\b")
# A trailing \b never matches after "%", which is not a word character — so
# "45%" survived every pass and reached the listener as "forty-five percent
# sign". The boundary is only required for alphabetic units.
_UNIT_RE = re.compile(
    r"(?P<value>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>"
    + "|".join(re.escape(unit) for unit in sorted(_UNITS, key=len, reverse=True))
    + r")(?![A-Za-z0-9])"
)
_ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
_DECIMAL_RE = re.compile(r"\b\d[\d,]*\.\d+\b")
_INTEGER_RE = re.compile(r"\b\d[\d,]*\b")
_INITIALISM_RE = re.compile(r"\b[A-Z]{2,6}\b")
_VERSION_RE = re.compile(r"\bv?(\d+)\.(\d+)(?:\.(\d+))?\b")


def _strip_markup(text: str) -> str:
    """Markdown read aloud is punctuation soup."""
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    # A bullet becomes a clause, not a bullet character.
    text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)
    return text


def to_spoken_form(text: str) -> str:
    """Rewrite a clause into the words a person would say aloud.

    Order matters: the most specific patterns run first, so a date is not
    eaten by the integer rule and money is not eaten by the decimal rule.
    """
    spoken = str(text or "")
    if not spoken.strip():
        return ""

    spoken = _strip_markup(spoken)
    spoken = _URL_RE.sub(_url, spoken)
    spoken = _BARE_URL_RE.sub(_url, spoken)
    spoken = _ISO_DATE_RE.sub(_iso_date, spoken)
    spoken = _MONEY_RE.sub(_money, spoken)
    spoken = _CLOCK_RE.sub(_clock, spoken)

    for pattern, replacement in _ABBREVIATIONS:
        spoken = pattern.sub(replacement, spoken)

    spoken = _VERSION_RE.sub(
        lambda m: " point ".join(
            number_words(int(part)) for part in m.groups() if part is not None
        ),
        spoken,
    )

    def _unit(match: re.Match[str]) -> str:
        raw = match.group("value").replace(",", "")
        singular, plural = _UNITS[match.group("unit")]
        value_words = _decimal_words(raw) if "." in raw else number_words(int(raw))
        word = singular if raw in {"1", "1.0"} else plural
        return f"{value_words} {word}"

    spoken = _UNIT_RE.sub(_unit, spoken)
    spoken = _RANGE_RE.sub(
        lambda m: f"{number_words(int(m.group(1)))} to {number_words(int(m.group(2)))}", spoken
    )
    spoken = _ORDINAL_RE.sub(lambda m: _ordinal_words(int(m.group(1))), spoken)
    spoken = _DECIMAL_RE.sub(lambda m: _decimal_words(m.group(0).replace(",", "")), spoken)

    def _integer(match: re.Match[str]) -> str:
        original = match.group(0)
        raw = original.replace(",", "")
        value = int(raw)
        # Only a bare four-digit run is a year. "1,250" is a quantity — it was
        # being read as "twelve fifty", which is a different number entirely.
        if "," not in original and len(raw) == 4 and 1100 <= value < 3000:
            return _year_words(value)
        return number_words(value)

    spoken = _INTEGER_RE.sub(_integer, spoken)
    spoken = _INITIALISM_RE.sub(
        lambda m: " ".join(m.group(0)) if m.group(0) in _INITIALISMS else m.group(0),
        spoken,
    )
    for symbol, replacement in _SYMBOLS:
        spoken = spoken.replace(symbol, replacement)

    spoken = re.sub(r"\s{2,}", " ", spoken)
    spoken = re.sub(r"\s+([,.;:!?])", r"\1", spoken)
    return spoken.strip()


# Conjunctions a speaker pauses before. Kokoro has no SSML, so a comma is the
# only instrument available for a breath — and it is a real one: the model
# lengthens the preceding syllable and inserts a short rest.
_BREATH_BEFORE = re.compile(
    r"(?<=[a-z0-9])\s+(but|because|so that|which means|although|whereas|"
    r"and then|except|unless)\b",
    re.IGNORECASE,
)
# A long run with no punctuation is read as one flat sweep.
_LONG_RUN_WORDS = 14
# Everything a spoken figure can be made of, so a breath is never inserted
# inside one.
_NUMBER_WORDS: frozenset[str] = frozenset(
    set(_ONES)
    | {tens for tens in _TENS if tens}
    | {name for _, name in _SCALES}
    | {"hundred", "point", "minus", "oh", "percent", "dollars", "dollar"}
)


def add_breathing_room(text: str) -> str:
    """Put a speaker's pauses where the punctuation would have been.

    Deliberately light. Inserted punctuation changes intonation as well as
    timing, so too much of it makes her sound halting rather than considered —
    the failure mode is as audible as the one it fixes.
    """
    spoken = str(text or "")
    if not spoken.strip():
        return ""
    spoken = _BREATH_BEFORE.sub(lambda m: f", {m.group(1)}", spoken)

    # Only break a genuinely long unpunctuated run, and only at a conjunction,
    # so the comma lands where a speaker would have taken the breath anyway.
    words = spoken.split()
    if len(words) > _LONG_RUN_WORDS and not re.search(r"[,;:]", spoken):
        for index in range(_LONG_RUN_WORDS // 2, len(words) - 3):
            if words[index].lower() not in {"and", "or", "but", "then", "while", "when"}:
                continue
            # "one thousand two hundred and fifty" is one figure; a comma
            # inside it is a pause in the middle of a number.
            neighbours = {words[index - 1].lower().rstrip(","), words[index + 1].lower()}
            if neighbours & _NUMBER_WORDS:
                continue
            words[index - 1] = words[index - 1].rstrip(",") + ","
            break
        spoken = " ".join(words)
    return re.sub(r",\s*,", ",", spoken).strip()


def prepare_for_speech(text: str) -> str:
    """Everything this module does, in the order it must happen."""
    return add_breathing_room(to_spoken_form(text))


__all__ = [
    "add_breathing_room",
    "number_words",
    "prepare_for_speech",
    "to_spoken_form",
]
