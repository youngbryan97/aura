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


def _written_back(value: Any) -> str | None:
    """``value`` as source that evaluates to an equal value, or None when it is not a plain value."""
    if _plain_within(value, _LONGEST_ARGUMENT) < 0:
        return None
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
    walked = list(traceback.walk_tb(error.__traceback__))
    for index in range(len(walked) - 1, -1, -1):
        frame, line = walked[index]
        path = _hers(frame.f_code.co_filename)
        if path:
            below = tuple(f"{inner.f_code.co_filename}:{inner_line} {inner.f_code.co_qualname}"
                          for inner, inner_line in walked[index + 1:])
            return frame, line, path, below
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
    found = _innermost_frame_of_hers(error)
    if found is None:
        return "raised outside her source"
    frame, line, path, below = found
    code = frame.f_code
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
            return f"{name} was not bound when {qualname} raised"
        source = _written_back(values[name])
        if source is None:
            return f"{name} is a {type(values[name]).__name__}, not a plain value"
        written[name] = source
    return FailedCall(
        module=module,
        qualname=qualname,
        file=path,
        line=int(line or 0),
        error_type=f"{type(error).__module__}.{type(error).__qualname__}",
        error_message=str(error)[:500],
        positional=tuple(written[name] for name in positional_only if name in written),
        keywords=tuple((name, written[name]) for name in [*either, *keyword_only] if name in written),
        star_args=written.get(star_args, "") if star_args else "",
        star_kwargs=written.get(star_kwargs, "") if star_kwargs else "",
        below=below,
        receiver_module=receiver.get("module", ""),
        receiver_class=receiver.get("class", ""),
        receiver_state=tuple(sorted((k, v) for k, v in receiver.items() if k not in ("module", "class"))),
    )


__all__ = ["FailedCall", "the_call_that_failed", "where_it_was_raised"]
