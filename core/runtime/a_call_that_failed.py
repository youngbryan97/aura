"""The call in her own code that a fault came from, kept so it can be made again.

A degradation record keeps the formatted traceback: three frames and a thousand characters. A fault cannot be
reproduced from that, because what the function was given is gone. This keeps it while the exception is still live:
the innermost frame in her own source, its function, and what it was called with, when each argument can be written
back exactly. A call that needs something it cannot rebuild (the instance a method ran on, a function defined inside
another, an argument that is not a plain value) is not kept, and says why. Neither is a deadline that passed, a
cancellation, or an exception recorded without being raised: none of them is a call that raised because of what it
was given. LIVE 15-16 November 2026, timeouts were 72 of the 150 reasons her stream gave.

Standard library only: every package records its faults through ``core/runtime/errors.py``, and this runs there.
"""
from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import dataclasses
import inspect
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType, FrameType
from typing import Any

#: Her source tree: the directory that holds ``core``.
_ROOT = Path(__file__).resolve().parents[2]

#: As long as an argument written back may be: the length a degradation record keeps of its traceback.
_LONGEST_ARGUMENT = 1000

#: What a call is not: her tests, her virtual environment, anything installed.
_NOT_HERS = ("tests", ".venv", "venv")

#: A deadline that passed. It depends on how long the call took when it ran, not on what it was given.
_RAN_OUT_OF_TIME = (TimeoutError, subprocess.TimeoutExpired)
#: A call stopped by whoever waited on it, not by anything it did.
_CANCELLED = (asyncio.CancelledError, concurrent.futures.CancelledError)


@dataclass(frozen=True)
class FailedCall:
    """A call that raised, as a test can make it again."""

    module: str
    qualname: str
    file: str
    line: int
    error_type: str
    error_message: str
    #: Each parameter's value as Python source, in the order the function takes them.
    positional: tuple[str, ...] = ()
    #: The names of those positional-only parameters, in the same order.
    positional_names: tuple[str, ...] = ()
    keywords: tuple[tuple[str, str], ...] = ()
    star_args: str = ""
    star_kwargs: str = ""
    #: Where the call went on from here before it raised: the frames below it, innermost last.
    below: tuple[str, ...] = field(default=())
    #: For a method: the class its instance was, where that class is defined, and the instance's state, each
    #: attribute as Python source. The test builds the instance from these without running ``__init__``.
    receiver_module: str = ""
    receiver_class: str = ""
    receiver_state: tuple[tuple[str, str], ...] = ()
    #: The exceptions it was raised from, by class, the nearest first: ``RuntimeError`` raised while handling an
    #: ``HTTPError`` is the server's refusal as much as her code's.
    causes: tuple[str, ...] = ()
    #: What the exceptions say of themselves as data rather than words: SQLite's own name and number for its error
    #: (``sqlite_errorname``, ``sqlite_errorcode``: SQLITE_BUSY, 5, when another connection held the database).
    codes: tuple[tuple[str, str], ...] = ()
    #: The call of hers above this one that built an argument this one was given, rather than passing on one it was
    #: given itself, written back the same way: where a refusal was right, what should change is there. None where no
    #: such call of hers was in the traceback, or it could not be written back.
    caller: FailedCall | None = None

    @property
    def fingerprint(self) -> tuple[str, str, str]:
        """The same fault, whatever it was called with: where, and what was raised."""
        return (self.module, self.qualname, self.error_type)

    def call_source(self, name: str) -> str:
        """``name(...)`` with the arguments it was given."""
        parts = [*self.positional, *(f"{key}={value}" for key, value in self.keywords)]
        if self.star_args:
            parts.append(f"*{self.star_args}")
        if self.star_kwargs:
            parts.append(f"**{self.star_kwargs}")
        return f"{name}({', '.join(parts)})"


def _hers(filename: str) -> str:
    """Her source path for a frame's file, relative to her tree; "" when it is not hers."""
    try:
        relative = Path(filename).resolve().relative_to(_ROOT)
    except (OSError, ValueError):
        return ""
    if relative.suffix != ".py" or relative.parts[0] in _NOT_HERS or "site-packages" in relative.parts:
        return ""
    return relative.as_posix()


def path_in_her_source(filename: str) -> str:
    """Her source path for a code object's file, relative to her tree; "" when it is not hers."""
    return _hers(filename)


_PLAIN = (type(None), bool, int, float, str, bytes)


def _plain_within(value: Any, budget: int) -> int:
    """What is left of ``budget`` after ``value``'s characters, or -1 when it is not a plain value or runs over.

    Walked before anything is written, so an argument a megabyte long costs no more than its length check.
    """
    if isinstance(value, _PLAIN):
        size = len(value) if isinstance(value, (str, bytes)) else 8
        return budget - size if budget >= size else -1
    if isinstance(value, dict):
        items: Any = (part for pair in value.items() for part in pair)
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = iter(value)
    else:
        return -1
    budget -= 2
    for item in items:
        if budget < 0:
            return -1
        budget = _plain_within(item, budget)
    return budget


def _a_record_written_back(value: Any) -> str | None:
    """A dataclass instance as its own constructor called with its fields, importing its class by itself; None unless
    that rebuilds it: every field set by the generated ``__init__``, no ``__post_init__`` to change what it is given,
    and each field a value written back in turn. LIVE 19 November 2026 two faults she heard could not be made again
    because an argument was a PhenomenalField, a record of a string and some numbers."""
    kind = type(value)
    params = getattr(kind, "__dataclass_params__", None)
    if not dataclasses.is_dataclass(value) or params is None or not params.init or hasattr(kind, "__post_init__"):
        return None
    if "<" in kind.__qualname__ or kind.__module__ in ("__main__", "builtins"):
        return None
    parts = []
    for one in dataclasses.fields(value):
        if not one.init:
            return None
        written = _written_back(getattr(value, one.name))
        if written is None:
            return None
        parts.append(f"{one.name}={written}")
    source = (f"__import__({kind.__module__!r}, fromlist=[{kind.__qualname__.split('.')[0]!r}])."
              f"{kind.__qualname__}({', '.join(parts)})")
    return source if len(source) <= _LONGEST_ARGUMENT else None


def _written_back(value: Any) -> str | None:
    """``value`` as source that evaluates to an equal value, or None when it is neither a plain value nor a record of
    them. A path is written as the path it is: 4 of the faults she heard on 25 and 26 November 2026 could not be made
    again only because an argument was a PosixPath."""
    from pathlib import PurePath

    if isinstance(value, PurePath) and type(value).__module__ == "pathlib":
        text = str(value)
        return f"__import__('pathlib').{type(value).__name__}({text!r})" if len(text) <= _LONGEST_ARGUMENT else None
    if _plain_within(value, _LONGEST_ARGUMENT) < 0:
        return _a_record_written_back(value)
    try:
        source = repr(value)
    # not a failure: a value whose repr raises cannot be written back.
    except Exception:  # noqa: BLE001 - a repr that raises is a value that cannot be written back
        return None
    if len(source) > _LONGEST_ARGUMENT:
        return None
    try:
        again = ast.literal_eval(source)
    # not a failure: source that does not evaluate back to a plain value cannot be written back.
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    try:
        same = again == value and type(again) is type(value)
    # not a failure: a comparison that raises is not equality, so the value is not written back.
    except Exception:  # noqa: BLE001 - a comparison that raises is not equality
        return None
    return source if same else None


def _names_read(code: CodeType) -> set[str]:
    """Every attribute and global name a function's code reads, its nested functions' included."""
    names = set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            names |= _names_read(constant)
    return names


def _the_instance_written_back(instance: Any, code: CodeType, qualname: str) -> dict[str, str] | str:
    """The instance a method raised on, as its class and its state written back, or why it cannot be.

    A test rebuilds it the way test carving does (Elbaum, Chin, Dwyer and Jorde, FSE 2006): the class's ``__new__``
    with no ``__init__``, then each attribute set as it was. An attribute that is not a plain value is left out when
    the method never names it, and stops the rebuild when it does, since the call would then run on a different
    instance than the one that raised.
    """
    cls = type(instance)
    where, name = str(getattr(cls, "__module__", "") or ""), str(getattr(cls, "__qualname__", "") or "")
    if not where or "<" in name or not _hers(str(getattr(sys.modules.get(where), "__file__", "") or "")):
        return f"{qualname} runs on a {name or type(instance).__name__}, a class outside her source"
    state = getattr(instance, "__dict__", None)
    if not isinstance(state, dict):
        return f"{qualname} runs on a {name} that keeps its state outside a __dict__"
    named = _names_read(code)
    kept: dict[str, str] = {"module": where, "class": name}
    budget = _LONGEST_ARGUMENT * 4
    for attribute, value in state.items():
        if not isinstance(attribute, str) or not attribute.isidentifier():
            continue
        source = _written_back(value)
        if source is None:
            if attribute in named:
                return f"{qualname} reads self.{attribute}, a {type(value).__name__}, not a plain value"
            continue
        budget -= len(source)
        if budget < 0:
            return f"{qualname} runs on a {name} whose state is too large to write back"
        kept[attribute] = source
    return kept


def _innermost_frame_of_hers(error: BaseException) -> tuple[FrameType, int, str, tuple[str, ...]] | None:
    """The frame nearest the raise that is in her source, its line, its path, and where the call went below it."""
    found = _innermost_of_hers(list(traceback.walk_tb(error.__traceback__)))
    return found[1:] if found is not None else None


def _innermost_of_hers(walked: list[tuple[FrameType, int]]) -> tuple[int, FrameType, int, str, tuple[str, ...]] | None:
    """Where in ``walked`` the frame nearest the raise in her source is, and the rest as _innermost_frame_of_hers."""
    for index in range(len(walked) - 1, -1, -1):
        frame, line = walked[index]
        path = _hers(frame.f_code.co_filename)
        if path:
            below = tuple(f"{inner.f_code.co_filename}:{inner_line} {inner.f_code.co_qualname}"
                          for inner, inner_line in walked[index + 1:])
            return index, frame, line, path, below
    return None


def _given(frame: FrameType) -> list[Any]:
    """What ``frame``'s function was given, by its parameters, where the value is an object of its own: an integer or a
    string may be one shared object wherever it is written, so whether it was passed on cannot be told from it."""
    positional_only, either, keyword_only, star_args, star_kwargs = _parameters(frame.f_code)
    names = [*positional_only, *either, *keyword_only, *(n for n in (star_args, star_kwargs) if n)]
    values = frame.f_locals
    return [values[name] for name in names if name not in ("self", "cls") and name in values
            and not isinstance(values[name], _PLAIN)]


def _the_call_that_gave_it(walked: list[tuple[FrameType, int]], index: int) -> FailedCall | None:
    """The nearest call of hers above ``walked[index]`` that built an argument it was given: each call between that
    only passed on what it was given itself is passed through. LIVE 16 November 2026 her notifications built an
    argument vector with an empty subtitle, her subprocess gateway passed it on, and ``_coerce_argv`` refused it; the
    refusal was right, and the fault was in the notification."""
    given = _given(walked[index][0])
    if not given:
        return None
    for outer, line in reversed(walked[:index]):
        if not _hers(outer.f_code.co_filename):
            return None
        passed = _given(outer)
        if all(any(value is other for other in passed) for value in given):
            continue  # it only passed on what it was given
        call = a_call_written_back(outer, raised=True)
        return None if isinstance(call, str) else dataclasses.replace(call, line=int(line or 0))
    return None


def _parameters(code: CodeType) -> tuple[list[str], list[str], list[str], str, str]:
    """Positional-only, positional-or-keyword and keyword-only names, and the names of ``*args`` and ``**kwargs``."""
    names = list(code.co_varnames)
    positional_only = names[: code.co_posonlyargcount]
    either = names[code.co_posonlyargcount: code.co_argcount]
    keyword_only = names[code.co_argcount: code.co_argcount + code.co_kwonlyargcount]
    at = code.co_argcount + code.co_kwonlyargcount
    star_args = names[at] if code.co_flags & inspect.CO_VARARGS else ""
    if star_args:
        at += 1
    star_kwargs = names[at] if code.co_flags & inspect.CO_VARKEYWORDS else ""
    return positional_only, either, keyword_only, star_args, star_kwargs


def where_it_was_raised(error: BaseException) -> tuple[str, str] | None:
    """The function of hers ``error`` was raised in, as (module, qualified name); None outside her source."""
    found = _innermost_frame_of_hers(error) if error.__traceback__ is not None else None
    if found is None:
        return None
    frame = found[0]
    return str(frame.f_globals.get("__name__") or ""), frame.f_code.co_qualname


def the_call_that_failed(error: BaseException) -> FailedCall | str:
    """The call in her own source that ``error`` was raised in, or why it cannot be made again."""
    if isinstance(error, _RAN_OUT_OF_TIME):
        return "it ran past a deadline, which depends on when it ran, not on what it was given"
    if isinstance(error, _CANCELLED):
        return "it was cancelled by what waited on it, not raised by what it was given"
    if error.__traceback__ is None:
        return "it was recorded without being raised in this process, so it carries no call"
    walked = list(traceback.walk_tb(error.__traceback__))
    found = _innermost_of_hers(walked)
    if found is None:
        return "raised outside her source"
    index, frame, line, _path, below = found
    call = a_call_written_back(frame, raised=True)
    if isinstance(call, str):
        return call
    chain: list[BaseException] = []
    held: BaseException | None = error
    while held is not None and len(chain) < 6:
        chain.append(held)
        held = held.__cause__ or (None if held.__suppress_context__ else held.__context__)
    codes = tuple((name, str(value)) for one in chain for name in ("sqlite_errorname", "sqlite_errorcode")
                  if isinstance(value := getattr(one, name, None), (str, int)) and not isinstance(value, bool))
    return dataclasses.replace(call, line=int(line or 0), error_type=f"{type(error).__module__}.{type(error).__qualname__}",
                               error_message=str(error)[:500], below=below,
                               causes=tuple(f"{type(one).__module__}.{type(one).__qualname__}" for one in chain[1:]),
                               codes=codes, caller=_the_call_that_gave_it(walked, index))


#: Wrappers of hers that only hand an error on, passed through to find the call that made it.
_WRAPPERS = 3


def the_call_that_recorded(error: BaseException, frame: FrameType | None) -> FailedCall | str:
    """For a fault recorded without being raised (``record_degradation("x", RuntimeError(...))``), the call of hers
    that recorded it, from ``frame`` outward: making that call again is making the fault again when it records the same
    fault again (core/self_modification/faults_she_can_reproduce.py). On 25 and 26 November 2026, 6 of the faults she
    heard carried no call for that alone."""
    while frame is not None and frame.f_globals.get("__name__") == "core.runtime.errors":
        frame = frame.f_back
    # A wrapper of hers that only hands the error on (``_record_goal_degradation(exc)``) is passed through, a few deep,
    # to what made the error; a frame not of hers never is, since past it lies the machinery that ran her, her event
    # loop and her process's own start among it.
    for _ in range(_WRAPPERS):
        if frame is None or not _hers(frame.f_code.co_filename) \
                or not any(value is error for value in frame.f_locals.values()) or frame.f_back is None \
                or not _hers(frame.f_back.f_code.co_filename):
            break
        frame = frame.f_back
    if frame is None or not _hers(frame.f_code.co_filename):
        return "it was recorded without being raised, from outside her source"
    call = a_call_written_back(frame, raised=True)
    if isinstance(call, str):
        return call
    return dataclasses.replace(call, error_type=f"{type(error).__module__}.{type(error).__qualname__}",
                               error_message=str(error)[:500])


def a_call_written_back(frame: FrameType, *, raised: bool = False) -> FailedCall | str:
    """The call ``frame`` is running, written back as source a test can make again, or why it cannot be.

    A fault's call is this with what it raised. Her calls that raised nothing are kept the same way, as inputs her
    change verifier tries on both sides of a change (core/self_modification/her_real_calls.py).
    """
    code = frame.f_code
    path = _hers(code.co_filename)
    if not path:
        return "raised outside her source" if raised else "outside her source"
    module = str(frame.f_globals.get("__name__") or "")
    if not module or module == "__main__":
        return f"{path} runs as a script, not a module"
    qualname = code.co_qualname
    if "<" in qualname:
        return f"{qualname} is defined inside another function"
    positional_only, either, keyword_only, star_args, star_kwargs = _parameters(code)
    values = frame.f_locals
    leading = (positional_only or either or [""])[0]
    receiver: dict[str, str] = {}
    if leading == "self":
        rebuilt = _the_instance_written_back(values.get("self"), code, qualname)
        if isinstance(rebuilt, str):
            return rebuilt
        receiver = rebuilt
    owner = qualname.split(".")[0]
    if owner not in frame.f_globals:
        return f"{owner} is not a name in {module}"
    written: dict[str, str] = {}
    for name in [*positional_only, *either, *keyword_only, *(n for n in (star_args, star_kwargs) if n)]:
        if name == leading and name in ("cls", "self"):
            continue
        if name not in values:
            return f"{name} was not bound when {qualname} {'raised' if raised else 'started'}"
        source = _written_back(values[name])
        if source is None:
            return f"{name} is a {type(values[name]).__name__}, not a plain value"
        written[name] = source
    return FailedCall(
        module=module,
        qualname=qualname,
        file=path,
        line=int(frame.f_lineno or 0),
        error_type="",
        error_message="",
        positional=tuple(written[name] for name in positional_only if name in written),
        positional_names=tuple(name for name in positional_only if name in written),
        keywords=tuple((name, written[name]) for name in [*either, *keyword_only] if name in written),
        star_args=written.get(star_args, "") if star_args else "",
        star_kwargs=written.get(star_kwargs, "") if star_kwargs else "",
        receiver_module=receiver.get("module", ""),
        receiver_class=receiver.get("class", ""),
        receiver_state=tuple(sorted((k, v) for k, v in receiver.items() if k not in ("module", "class"))),
    )


__all__ = ["FailedCall", "a_call_written_back", "path_in_her_source", "the_call_that_failed", "where_it_was_raised"]
