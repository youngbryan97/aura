"""JSON out of the decoder because the decoder cannot produce anything else.

Ninety-four of the strings this runtime writes to steer a model say some form
of "Return ONLY JSON" or "output only the final answer". A sentence is a
request. The model honours it most of the time, and the parser on the other
side has a fallback for the rest — a fence to strip, a first brace to find,
prose to skip — and every one of those fallbacks is the sentence not having
worked. What the caller wants is a shape, and a shape is something the decoder
can be made to hold.

This is a logits processor that walks a JSON automaton over the text produced
so far and, at every step, allows only the tokens whose characters the
automaton would accept from where it stands. Prose before the value is not
available to sample; a string cannot go unclosed; a brace cannot go
unbalanced; once the top-level value is complete, the only token left is the
end of the turn. The instruction becomes unnecessary because the alternative
is unreachable.

The automaton is JSON's own grammar — objects, arrays, strings with escapes,
numbers, true/false/null — with one stack of open brackets. Its state is small
enough that the set of allowed tokens for each distinct state is computed once
and cached: the first generation in a process pays a few seconds warming
those sets, and every step after that is a mask lookup.

A generation with a private channel is left alone until the channel closes;
the shape applies to the answer, not to the thinking.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.verify.invariants import invariant

logger = logging.getLogger("Aura.Brain.LLM.Shape")

#: Modes of the automaton. The stack of open brackets is beside them.
_START = "start"            # whitespace, then a value
_DONE = "done"              # a complete top-level value; whitespace and EOS only
_OBJ_OPEN = "obj_open"      # after '{': a key string or '}'
_OBJ_KEY = "obj_key"        # inside a key string
_OBJ_COLON = "obj_colon"    # after a key: ':'
_OBJ_VALUE = "obj_value"    # after ':': a value
_OBJ_NEXT = "obj_next"      # after a value: ',' or '}'
_OBJ_KEY_NEXT = "obj_key_next"  # after ',': a key string
_ARR_OPEN = "arr_open"      # after '[': a value or ']'
_ARR_VALUE = "arr_value"    # after ',': a value
_ARR_NEXT = "arr_next"      # after a value: ',' or ']'
_STRING = "string"          # inside a value string
_STRING_ESC = "string_esc"  # after a backslash in a string
_KEY_ESC = "key_esc"
_STRING_UNICODE = "string_unicode"
_KEY_UNICODE = "key_unicode"
_NUMBER = "number"          # inside a number
_LITERAL = "literal"        # inside true/false/null

_LITERALS = ("true", "false", "null")
_WHITESPACE = " \t\n\r"
_DIGITS = "0123456789"
_HEX_DIGITS = "0123456789abcdefABCDEF"
_MASK_CACHE_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class JsonState:
    """Where the automaton stands: its mode, the bracket it is inside, and how
    far into a literal or number it is. Frozen so it can key a cache."""

    mode: str = _START
    stack: tuple[str, ...] = ()
    literal: str = ""          # the literal being spelled, e.g. "tr"
    number: str = ""           # the number so far, for what may follow
    unicode_remaining: int = 0

    @property
    def top(self) -> str:
        return self.stack[-1] if self.stack else ""

    def cache_key(self) -> tuple[str, tuple[str, ...], str, str, int]:
        # A token can close several containers, so the entire stack matters.
        return (self.mode, self.stack, self.literal, _number_class(self.number), self.unicode_remaining)


def _number_class(number: str) -> str:
    """The part of a number's history that decides what may follow it."""
    if not number:
        return ""
    if number in ("-",):
        return "-"
    if number.endswith(("e-", "E-", "e+", "E+")):
        return "exp_sign"
    if number.endswith(("e", "E")):
        return "exp"
    if "." in number and number.endswith("."):
        return "dot"
    if number in ("0", "-0"):
        return "zero"
    has_dot = "." in number
    has_exp = "e" in number or "E" in number
    return f"int{'.' if has_dot else ''}{'e' if has_exp else ''}"


def _after_value(state: JsonState) -> JsonState:
    """A value just completed; where the automaton goes depends on the bracket."""
    if state.top == "{":
        return JsonState(_OBJ_NEXT, state.stack)
    if state.top == "[":
        return JsonState(_ARR_NEXT, state.stack)
    return JsonState(_DONE, ())


def _start_value(state: JsonState, char: str) -> JsonState | None:
    """A value may begin here; what does ``char`` start?"""
    if char in _WHITESPACE:
        return state
    if char == "{":
        return JsonState(_OBJ_OPEN, state.stack + ("{",))
    if char == "[":
        return JsonState(_ARR_OPEN, state.stack + ("[",))
    if char == '"':
        return JsonState(_STRING, state.stack)
    if char == "-" or char in _DIGITS:
        return JsonState(_NUMBER, state.stack, number=char)
    for word in _LITERALS:
        if word[0] == char:
            return JsonState(_LITERAL, state.stack, literal=char)
    return None


def _number_may_end(number: str) -> bool:
    return bool(number) and number[-1] in _DIGITS


def step(state: JsonState, char: str) -> JsonState | None:
    """Feed one character. None means the grammar refuses it here."""
    mode = state.mode
    if mode == _START:
        if state.literal == "object" and char not in _WHITESPACE and char != "{":
            return None
        if state.literal == "array" and char not in _WHITESPACE and char != "[":
            return None
        begun = _start_value(state, char)
        if begun is not None and begun.mode == _START:
            return JsonState(_START, state.stack, literal=state.literal)
        return begun
    if mode == _DONE:
        # Nothing after the value. Whitespace here was a way for a model that
        # never chose the end token to run to its budget on tabs.
        return None
    if mode in (_OBJ_OPEN, _OBJ_KEY_NEXT):
        if char in _WHITESPACE:
            return state
        if char == '"':
            return JsonState(_OBJ_KEY, state.stack)
        if char == "}" and mode == _OBJ_OPEN:
            return _after_value(JsonState(mode, state.stack[:-1]))
        return None
    if mode == _OBJ_KEY:
        if char == "\\":
            return JsonState(_KEY_ESC, state.stack)
        if char == '"':
            return JsonState(_OBJ_COLON, state.stack)
        return state if char >= " " else None
    if mode == _KEY_ESC:
        if char == "u":
            return JsonState(_KEY_UNICODE, state.stack, unicode_remaining=4)
        return JsonState(_OBJ_KEY, state.stack) if char in '"\\/bfnrt' else None
    if mode in (_KEY_UNICODE, _STRING_UNICODE):
        if char not in _HEX_DIGITS:
            return None
        remaining = state.unicode_remaining - 1
        if remaining:
            return JsonState(mode, state.stack, unicode_remaining=remaining)
        return JsonState(_OBJ_KEY if mode == _KEY_UNICODE else _STRING, state.stack)
    if mode == _OBJ_COLON:
        if char in _WHITESPACE:
            return state
        return JsonState(_OBJ_VALUE, state.stack) if char == ":" else None
    if mode in (_OBJ_VALUE, _ARR_VALUE):
        return _start_value(state, char)
    if mode == _ARR_OPEN:
        if char == "]":
            return _after_value(JsonState(mode, state.stack[:-1]))
        return _start_value(state, char)
    if mode == _OBJ_NEXT:
        if char in _WHITESPACE:
            return state
        if char == ",":
            return JsonState(_OBJ_KEY_NEXT, state.stack)
        if char == "}":
            return _after_value(JsonState(mode, state.stack[:-1]))
        return None
    if mode == _ARR_NEXT:
        if char in _WHITESPACE:
            return state
        if char == ",":
            return JsonState(_ARR_VALUE, state.stack)
        if char == "]":
            return _after_value(JsonState(mode, state.stack[:-1]))
        return None
    if mode == _STRING:
        if char == "\\":
            return JsonState(_STRING_ESC, state.stack)
        if char == '"':
            return _after_value(state)
        return state if char >= " " else None
    if mode == _STRING_ESC:
        if char == "u":
            return JsonState(_STRING_UNICODE, state.stack, unicode_remaining=4)
        return JsonState(_STRING, state.stack) if char in '"\\/bfnrt' else None
    if mode == _LITERAL:
        spelled = state.literal + char
        for word in _LITERALS:
            if word == spelled:
                return _after_value(state)
            if word.startswith(spelled):
                return JsonState(_LITERAL, state.stack, literal=spelled)
        return None
    if mode == _NUMBER:
        number = state.number
        cls = _number_class(number)
        if char in _DIGITS:
            if cls == "zero":
                return None  # no leading zeros
            return JsonState(_NUMBER, state.stack, number=number + char)
        if char == "." and cls in ("int", "zero"):
            return JsonState(_NUMBER, state.stack, number=number + char)
        if char in "eE" and cls in ("int", "int.", "zero") and _number_may_end(number):
            return JsonState(_NUMBER, state.stack, number=number + char)
        if char in "+-" and cls == "exp" and number[-1] in "eE":
            return JsonState(_NUMBER, state.stack, number=number + char)
        if not _number_may_end(number):
            return None
        # The number ended; this character belongs to what follows it.
        return step(_after_value(state), char)
    return None


def feed(state: JsonState, text: str) -> JsonState | None:
    for char in text:
        state = step(state, char)
        if state is None:
            return None
    return state


def is_complete(state: JsonState) -> bool:
    """A number at the top level ends only when something follows it, and
    nothing follows it at EOS; a top-level number is complete when it may end."""
    if state.mode == _DONE:
        return True
    return state.mode == _NUMBER and not state.stack and _number_may_end(state.number)


@invariant("decoder.json_token_state", scope="inference",
           owner="core/brain/llm/a_shape_the_decoder_enforces.py", observational=False)
def _complete_token_grammar_state() -> tuple:
    """Mask identities preserve the continuation language of whole tokens."""
    deep, shallow = feed(JsonState(), "[["), feed(JsonState(), "[")
    assert deep is not None and shallow is not None
    assert deep.cache_key() != shallow.cache_key()
    assert feed(deep, "]]") is not None and feed(shallow, "]]") is None
    assert feed(JsonState(), '"\\uZZZZ"') is None
    valid = feed(JsonState(), '{"\\u0061":[1e+2]}')
    assert valid is not None and is_complete(valid)
    return ()


class _Vocabulary:
    """Every token as the text it adds, decoded once per tokenizer."""

    def __init__(self, tokenizer: Any) -> None:
        size = int(len(tokenizer))
        self.texts: list[str] = []
        specials = set(int(i) for i in (getattr(tokenizer, "all_special_ids", None) or ()))
        for token_id in range(size):
            if token_id in specials:
                self.texts.append("")
                continue
            try:
                self.texts.append(str(tokenizer.decode([token_id])))
            except (TypeError, ValueError, RuntimeError):
                self.texts.append("")
        self.specials = specials
        self.size = size
        eos = getattr(tokenizer, "eos_token_id", None)
        self.ends: set[int] = {int(eos)} if eos is not None else set()
        for name in ("<|im_end|>", "<|endoftext|>", "</s>"):
            try:
                ident = tokenizer.convert_tokens_to_ids(name)
            except (TypeError, ValueError, AttributeError, KeyError):
                ident = None
            if isinstance(ident, int) and 0 <= ident < size:
                self.ends.add(ident)


_VOCABULARIES: OrderedDict[int, tuple[Any, _Vocabulary]] = OrderedDict()


def _vocabulary_for(tokenizer: Any) -> _Vocabulary:
    key = id(tokenizer)
    found = _VOCABULARIES.get(key)
    if found is not None and found[0] is tokenizer:
        _VOCABULARIES.move_to_end(key)
        return found[1]
    vocabulary = _Vocabulary(tokenizer)
    # Retain the owner, not only its recyclable id. Keep current and draft
    # tokenizers; a later model swap may rebuild an evicted vocabulary.
    _VOCABULARIES[key] = (tokenizer, vocabulary)
    while len(_VOCABULARIES) > 2:
        _VOCABULARIES.popitem(last=False)
    return vocabulary


def allowed_token_ids(vocabulary: _Vocabulary, state: JsonState) -> list[int]:
    """Every token whose text the automaton accepts from ``state``, whole."""
    allowed: list[int] = []
    for token_id, text in enumerate(vocabulary.texts):
        if not text:
            continue
        if feed(state, text) is not None:
            allowed.append(token_id)
    if is_complete(state) or state.mode == _DONE:
        allowed.extend(sorted(vocabulary.ends))
    return allowed


def enforce_json(
    tokenizer: Any,
    *,
    after_token: int | None = None,
    require: str = "any",
) -> Callable[[Any, Any], Any] | None:
    """A logits processor under which the decoder can only produce JSON.

    ``after_token`` is the id of the token that closes a private channel; the
    shape is enforced only once it has appeared. None enforces from the first
    token. ``require`` narrows the top-level value to an "object" or an
    "array" for callers that will parse one; "any" allows any JSON value.
    Returns None where MLX is not importable, so a caller knows the shape is
    not held rather than assuming it is.

    The first call precedes sampling and carries prompt context in MLX. That
    prefix is not part of the answer. A speculative rewind replays only the
    generated suffix, rather than retaining grammar state from a rejected draft.
    """
    try:
        import mlx.core as mx
    except ImportError:
        return None
    if require not in ("any", "object", "array"):
        raise ValueError(f"require must be any, object or array, not {require!r}")

    vocabulary = _vocabulary_for(tokenizer)
    masks: OrderedDict[tuple[tuple[str, tuple[str, ...], str, str, int], str], Any] = OrderedDict()
    mask_bytes = 0
    opening = JsonState(literal="" if require == "any" else require)
    state = {"json": opening, "seen": 0, "enforcing": after_token is None, "refused": 0}
    prompt: list[int] | None = None
    history: list[int] = []

    def mask_for(json_state: JsonState, dtype: Any) -> Any:
        nonlocal mask_bytes
        key = (json_state.cache_key(), str(dtype))
        found = masks.get(key)
        if found is None:
            ids = allowed_token_ids(vocabulary, json_state)
            found = mx.full((vocabulary.size,), -mx.inf, dtype=dtype)
            if ids:
                found[mx.array(ids)] = 0.0
            while masks and mask_bytes + found.nbytes > _MASK_CACHE_BYTES:
                _, evicted = masks.popitem(last=False)
                mask_bytes -= evicted.nbytes
            masks[key] = found
            mask_bytes += found.nbytes
        else:
            masks.move_to_end(key)
        return found

    def hold_the_shape(tokens: Any, logits: Any) -> Any:
        nonlocal prompt, history
        try:
            produced = len(tokens)
        except TypeError:
            return logits
        token_ids = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
        if prompt is None:
            prompt = token_ids
            state["seen"] = produced
        elif token_ids[:len(prompt)] != prompt:
            raise ValueError("JSON processor reused with a different prompt prefix")
        elif token_ids[:len(history)] != history:
            state["json"] = opening
            state["enforcing"] = after_token is None
            state["seen"] = len(prompt)
        history = token_ids
        # Catch up on tokens produced since the last call.
        while state["seen"] < produced:
            token_id = int(token_ids[state["seen"]])
            state["seen"] += 1
            if not state["enforcing"]:
                if token_id == after_token:
                    state["enforcing"] = True
                continue
            text = vocabulary.texts[token_id] if 0 <= token_id < vocabulary.size else ""
            if token_id in vocabulary.ends:
                continue
            advanced = feed(state["json"], text) if text else state["json"]
            if advanced is None:
                # A token this processor did not allow got through — a
                # sampler bypass, or a tokenizer whose decode differs from
                # its generation. Counted, and the shape is no longer claimed.
                state["refused"] += 1
                state["enforcing"] = False
                logger.warning(
                    "🧠 [WORKER] JSON shape lost at token %d (%r); no longer enforced.",
                    state["seen"], text,
                )
                return logits
            state["json"] = advanced
        if not state["enforcing"]:
            return logits
        width = logits.shape[-1]
        mask = mask_for(state["json"], logits.dtype)
        if width != vocabulary.size:
            # A model whose logits are wider than its tokenizer pads the tail;
            # those ids are never allowed.
            mask = mx.concatenate([mask, mx.full((width - vocabulary.size,), -mx.inf, dtype=logits.dtype)]) if width > vocabulary.size else mask[:width]
        return logits + mask

    hold_the_shape.state = state  # type: ignore[attr-defined]
    hold_the_shape.masks = masks  # type: ignore[attr-defined]
    return hold_the_shape


__all__ = ["JsonState", "allowed_token_ids", "enforce_json", "feed", "is_complete", "step"]
