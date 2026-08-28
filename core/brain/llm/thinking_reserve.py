"""What the private reasoning channel costs, measured rather than assumed.

A thinking model spends its output budget twice: once on reasoning nobody
reads, then on the answer. Callers size ``max_tokens`` against the answer,
because that is the part they can picture. The worker enforces it against
everything the model emits.

LIVE, 2026-08-27: a question about a number sequence was planned at
``max_tokens=1536`` and served 1,469 characters — a shade under one character
per token, which no prose reaches. The reasoning channel had taken the rest,
and the answer stopped mid-paragraph before it reached the part the person had
actually asked for. The failure lands on hard questions only, because those are
the ones that reason.

The worker already records ``native_thinking_private_chars`` on every
generation and nothing read it. This is the reader. It holds no assumed
constants: the characters-per-token ratio comes from the generation it is
measuring, and the reserve is a percentile of what reasoning has cost so far.
With too few observations to express that percentile it reserves nothing,
because a guess dressed as a measurement is worse than an honest zero — and
the runtime generates often enough that the window fills within minutes.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path

#: The smallest window in which a 90th percentile is a real observation rather
#: than a restatement of the largest sample.
_ENOUGH_TO_EXPRESS_A_PERCENTILE = 10

#: How far back reserves are learned from. Long enough to cover a mix of
#: question kinds, short enough that a model swap is forgotten quickly.
_WINDOW = 128

_PERCENTILE = 0.9

_observed: deque[int] = deque(maxlen=_WINDOW)

#: The largest budget a generation has been given and still not finished
#: thinking inside. Unlike the window this is a proof rather than a sample: the
#: channel demonstrably needs more than this, so one observation is enough and
#: no percentile applies.
_proved_insufficient = 0

#: Observed decode rates in tokens per second, so the time a budget needs can
#: be worked out rather than assumed. The rate moves with the model, the
#: quantisation and what else is on the GPU, none of which a constant tracks.
#: Each entry is (tokens generated, tokens per second). The length is kept
#: because a short generation does not predict a long one: it decodes over a
#: shorter context, spends proportionally more of its time on the prompt, and
#: reads faster per token than the run it is being used to size.
_rates: deque[tuple[int, float]] = deque(maxlen=_WINDOW)

_lock = threading.Lock()

#: Taking back earlier measurements happens once, on first use, because the
#: state root is not ready at import time.
_restored = False

#: How many new readings before they are written down again. Every reading is
#: cheap and the write is not, and losing the last few costs a few seconds of
#: cold start rather than anything a person sees.
_WRITE_EVERY = 25
_since_write = 0


def record_reasoning_cost(
    *,
    reasoning_chars: int,
    surface_chars: int,
    generated_tokens: int,
) -> None:
    """Log what one generation spent on its private channel.

    The characters-per-token ratio is taken from this generation rather than
    assumed, so a change of tokenizer needs no edit here.
    """

    try:
        reasoning = max(0, int(reasoning_chars))
        surface = max(0, int(surface_chars))
        tokens = max(0, int(generated_tokens))
    except (TypeError, ValueError):
        return
    total_chars = reasoning + surface
    if total_chars <= 0 or tokens <= 0:
        return
    spent = int(round(tokens * (reasoning / total_chars)))
    with _lock:
        _observed.append(max(0, spent))
    _written_down()


def record_budget_that_ran_out_thinking(*, budget_tokens: int) -> None:
    """A thinking generation spent this budget and still had not finished.

    Two shapes of the same failure. The budget can run out while the model is
    still inside the private channel, leaving no answer at all; or the channel
    can close, the answer start, and the budget die part-way through it. The
    second is much the commoner and was not recorded for a long time, so the
    reserve learned nothing from the failures it exists to prevent.

    Either way this is not a sample of what reasoning costs. It is a proof
    that reasoning cost more than this, and a proof needs no percentile and no
    second opinion — which matters, because the generations that open the
    channel are the ones this reserve is for, and waiting for a window of them
    to accumulate means every one of them fails first.
    """

    global _proved_insufficient
    try:
        spent = max(0, int(budget_tokens))
    except (TypeError, ValueError):
        return
    with _lock:
        _proved_insufficient = max(_proved_insufficient, spent)
    save()


#: Prefill rates, as (prompt_chars, chars_per_second) pairs. Beside the decode
#: rates because they are the same kind of fact about the same generations.
_read_rates: list[tuple[int, float]] = []

#: When the store was last read, so a proof another process wrote is taken
#: back without re-reading a file that has not changed.
_last_seen_store_mtime: int = -1


def _restore_once() -> None:
    """Take back what earlier processes measured, the first time anyone asks."""

    global _restored
    with _lock:
        if _restored:
            return
        _restored = True
    load()


def _written_down() -> None:
    """Write the measurements out every so often, not on every reading."""

    global _since_write
    with _lock:
        _since_write += 1
        due = _since_write >= _WRITE_EVERY
        if due:
            _since_write = 0
    if due:
        save()


def _take_back_any_newer_proof() -> None:
    """Re-read a proof another process has written since this one started.

    The runtime runs more than one worker, each its own process with its own
    copy of this module, and the store was read once at startup and never
    again. So a proof paid for by a failed generation in one worker never
    reached the worker beside it, and never reached anything already running
    — which is every process that matters, since the proof is written the
    moment a turn fails and read on the turn after.

    Cheap and safe to repeat: the file is consulted only when it has changed,
    and what comes back is folded in as a maximum, so the same proof read
    twice says the same thing.
    """

    global _last_seen_store_mtime
    target = _store_path()
    if target is None:
        return
    try:
        stamp = target.stat().st_mtime_ns
    except OSError:
        return
    with _lock:
        if stamp == _last_seen_store_mtime:
            return
        _last_seen_store_mtime = stamp
    try:
        stored = json.loads(target.read_text())
        found = int(stored.get("proved_insufficient") or 0)
    except (OSError, ValueError, TypeError):
        return
    global _proved_insufficient
    with _lock:
        _proved_insufficient = max(_proved_insufficient, found)


def reserve_tokens() -> int:
    """Tokens to add to an answer budget so reasoning does not eat it.

    The larger of what the window has measured and what a generation has
    already proved to be too little. The window needs enough observations to
    carry a percentile; the proof needs one.
    """

    _restore_once()
    _take_back_any_newer_proof()
    with _lock:
        seen = sorted(_observed)
        proved = _proved_insufficient
    measured = 0
    if len(seen) >= _ENOUGH_TO_EXPRESS_A_PERCENTILE:
        index = min(len(seen) - 1, int(_PERCENTILE * len(seen)))
        measured = max(0, seen[index])
    return max(measured, proved)


def record_decode_rate(*, generated_tokens: int, elapsed_s: float) -> None:
    """Log how fast this generation actually decoded."""

    try:
        tokens = int(generated_tokens)
        seconds = float(elapsed_s)
    except (TypeError, ValueError):
        return
    if tokens <= 0 or not (seconds > 0.0) or seconds != seconds:
        return
    with _lock:
        _rates.append((tokens, tokens / seconds))
    _written_down()


#: Below this a generation is mostly its prompt, and its per-token rate says
#: more about prefill than about decoding.
_LONG_ENOUGH_TO_TIME = 32


def seconds_to_decode(tokens: int) -> float:
    """How long a budget of this many tokens takes, or 0.0 when unmeasured.

    Deliberately pessimistic: the tenth-percentile rate, because a deadline
    sized on the typical rate misses every generation slower than typical, and
    those are the long ones a deadline is about.

    Only generations of a comparable length count. Pooling them all lets a
    window full of short background prompts report a rate no long foreground
    turn reaches, which is a deadline sized on the wrong evidence.
    """

    try:
        wanted = int(tokens)
    except (TypeError, ValueError):
        return 0.0
    if wanted <= 0:
        return 0.0
    _restore_once()
    with _lock:
        # Only runs of a comparable size. Half the wanted length is the
        # boundary because below it the prompt dominates the measurement.
        comparable = sorted(
            rate for length, rate in _rates if length * 2 >= wanted
        )
        if len(comparable) < _ENOUGH_TO_EXPRESS_A_PERCENTILE:
            # Nothing comparable, so anything not dominated by its prompt.
            #
            # This returned 0.0 — "unmeasured, extend nothing" — for every
            # budget worth extending. Measured live: forty readings held, and
            # ZERO of them comparable to a 1536-token budget, because this
            # runtime's generations are mostly nineteen to ninety-nine tokens.
            # The estimator was permanently silent for exactly the turns it
            # exists to protect, and they are rare partly BECAUSE the deadline
            # it could not extend kept cancelling them.
            #
            # Safe in the direction a deadline needs. A short run amortises its
            # prefill over fewer tokens, so its per-token rate is the worse one:
            # using it for a long budget over-estimates the time and asks for
            # more deadline, never less.
            comparable = sorted(
                rate for length, rate in _rates if length >= _LONG_ENOUGH_TO_TIME
            )
    if len(comparable) < _ENOUGH_TO_EXPRESS_A_PERCENTILE:
        return 0.0
    index = min(len(comparable) - 1, int((1.0 - _PERCENTILE) * len(comparable)))
    rate = comparable[index]
    if not (rate > 0.0):
        return 0.0
    return wanted / rate





def record_read_rate(*, prompt_chars: int, elapsed_s: float) -> None:
    """Log how fast this generation actually read its prompt."""

    try:
        chars = int(prompt_chars)
        seconds = float(elapsed_s)
    except (TypeError, ValueError):
        return
    if chars <= 0 or not (seconds > 0.0) or seconds != seconds:
        return
    with _lock:
        _read_rates.append((chars, chars / seconds))
    _written_down()


#: Below this a prompt is mostly fixed overhead and its per-character rate says
#: more about starting up than about reading.
_BIG_ENOUGH_TO_TIME = 400


def seconds_to_read(prompt_chars: int) -> float:
    """How long a prompt of this size takes to read, or 0.0 when unmeasured.

    The other half of a generation, and on this hardware the larger half. A
    deadline built from decoding alone gives a turn time to say its answer and
    none to read the question: live on 2026-08-28 a six-kilobyte prompt was
    cancelled at 119.5 seconds of a 120-second prefill, having produced no
    tokens at all.

    Pessimistic in the same way and for the same reason as the decode rate, and
    silent in the same way when it has not been measured: an unmeasured rate
    extends no deadline.
    """

    try:
        wanted = int(prompt_chars)
    except (TypeError, ValueError):
        return 0.0
    if wanted <= 0:
        return 0.0
    _restore_once()
    with _lock:
        comparable = sorted(
            rate for size, rate in _read_rates if size * 2 >= wanted
        )
        if len(comparable) < _ENOUGH_TO_EXPRESS_A_PERCENTILE:
            # Nothing comparable, so anything big enough to time.
            #
            # The same blind spot the decode rate had: readings cluster at the
            # sizes a runtime actually generates, and the prompt sizes worth
            # asking about are far above them. Silence here meant a 47,000
            # character prompt cost nothing as far as any budget could tell.
            #
            # A smaller prompt spreads its fixed overhead over fewer
            # characters, so its per-character rate is the worse one and this
            # over-estimates rather than under-estimates. For a decision about
            # whether to TRIM that is the cautious side while the alternative
            # is three minutes of reading, and the levels are graded rather
            # than all-or-nothing.
            comparable = sorted(
                rate for size, rate in _read_rates if size >= _BIG_ENOUGH_TO_TIME
            )
    if len(comparable) < _ENOUGH_TO_EXPRESS_A_PERCENTILE:
        return 0.0
    index = min(len(comparable) - 1, int((1.0 - _PERCENTILE) * len(comparable)))
    rate = comparable[index]
    if not (rate > 0.0):
        return 0.0
    return wanted / rate


def proved_insufficient() -> int:
    """The largest budget a generation ran out of while still thinking."""

    with _lock:
        return _proved_insufficient


def observations() -> int:
    """How many generations the reserve is learned from."""

    with _lock:
        return len(_observed)


def forget() -> None:
    """Drop what has been learned. For tests and for a model swap."""

    global _proved_insufficient, _restored, _last_seen_store_mtime
    with _lock:
        _observed.clear()
        _rates.clear()
        _read_rates.clear()
        _proved_insufficient = 0
        # And forget having read the store, or a re-read would take it back.
        _last_seen_store_mtime = -1
        # Stops THIS process taking the readings back on the next call.
        _restored = True
    # And removes them, which the line above does not do and the comment here
    # used to claim it did. A test named for forgetting the disk was checking
    # that one process refrained from reloading, while the file sat there for
    # the next process to find.
    target = _store_path()
    if target is None:
        return
    # Through the gateway, like every other consequential write here. A raw
    # unlink is a file mutation nothing owns, and the readings this removes are
    # state the runtime is answerable for.
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope(
            "brain.thinking_reserve", domain="state_mutation"
        ):
            get_file_write_gateway().delete_file(
                target, source="brain.thinking_reserve"
            )
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return


# ------------------------------------------------------------- across restarts
#
# What was measured is about the machine and the model, not about the session,
# and both outlive the process. Held only in memory, the window emptied on every
# restart and the first long generation after a boot was sized on nothing — so
# the deadline that should have covered it was never extended, and the turn that
# needed the measurement most was the one that never had it.
#
# LIVE, 2026-08-28: a diagnosis turn was sized for one generation because the
# rate window had no runs long enough to speak about a 640-token one. The
# runtime had been up eleven minutes and had generated dozens of times.

#: Where the measurements live between processes.
_STORE = "decode_measurements.json"


def _store_path() -> Path | None:
    try:
        from core.runtime.state_ownership import state_root

        return Path(state_root()) / _STORE
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        return None


def save() -> bool:
    """Write the measurements down, through the runtime's own write path."""

    target = _store_path()
    if target is None:
        return False
    with _lock:
        payload = json.dumps(
            {
                "reasoning_tokens": list(_observed),
                "proved_insufficient": _proved_insufficient,
                "rates": [[length, rate] for length, rate in _rates],
                # Reading rates as well as decoding ones.
                #
                # The measurement was added and its persistence was not, so it
                # was learned in one process and could not be read from any
                # other, and every restart began knowing nothing about how long
                # a prompt takes to read. A measurement that does not survive
                # is a measurement nobody has.
                "read_rates": [[size, rate] for size, rate in _read_rates],
            }
        )
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope(
            "llm.thinking_reserve", domain="state_mutation"
        ):
            get_file_write_gateway().write_text(
                target, payload, source="llm.thinking_reserve"
            )
        return True
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False


def load() -> int:
    """Take back what earlier processes measured. Returns how many readings."""

    global _proved_insufficient
    target = _store_path()
    if target is None:
        return 0
    try:
        raw = json.loads(target.read_text())
    except (OSError, ValueError):
        return 0
    if not isinstance(raw, dict):
        return 0
    taken = 0
    with _lock:
        for value in raw.get("reasoning_tokens") or ():
            try:
                _observed.append(max(0, int(value)))
                taken += 1
            except (TypeError, ValueError):
                continue
        for row in raw.get("rates") or ():
            try:
                length, rate = row
                if int(length) > 0 and float(rate) > 0.0:
                    _rates.append((int(length), float(rate)))
                    taken += 1
            except (TypeError, ValueError):
                continue
        for row in raw.get("read_rates") or ():
            try:
                size, rate = row
                if int(size) > 0 and float(rate) > 0.0:
                    _read_rates.append((int(size), float(rate)))
                    taken += 1
            except (TypeError, ValueError):
                continue
        try:
            _proved_insufficient = max(
                _proved_insufficient, int(raw.get("proved_insufficient") or 0)
            )
        except (TypeError, ValueError):
            pass
    return taken


def chars_readable_in(seconds: float, *, ceiling: int = 1_000_000) -> int:
    """How long a prompt may be to be read in this much time.

    The same question as :func:`seconds_to_read` asked from the other end. A
    turn has to read and to answer, and the reserve already works out what
    answering costs; what is left is what reading may cost, and this says how
    many characters that buys.

    Answered by searching the forward function rather than by inverting its
    arithmetic, so the two can never disagree about the same rates, and so
    the graded fallbacks it uses when nothing comparable has been timed
    apply here unchanged. Returns ``ceiling`` only where the forward
    function is silent altogether, since a rate nobody has measured
    constrains nothing.
    """

    try:
        allowed = float(seconds)
    except (TypeError, ValueError):
        return int(ceiling)
    if allowed <= 0:
        return 0
    if seconds_to_read(int(ceiling)) <= allowed:
        return int(ceiling)
    low, high = 0, int(ceiling)
    while low < high:
        middle = (low + high + 1) // 2
        if seconds_to_read(middle) <= allowed:
            low = middle
        else:
            high = middle - 1
    return low
