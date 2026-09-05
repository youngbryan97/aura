from __future__ import annotations

import os
from typing import Any, Mapping

from core.runtime.background_policy import is_user_facing_origin

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _flag_enabled(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in _TRUE_VALUES


def legacy_shortcuts_enabled() -> bool:
    """Escape hatch for older federated behavior.

    Default is strict constitutional mode: legacy shortcut paths are OFF unless
    explicitly re-enabled in the environment.
    """
    return _flag_enabled("AURA_ALLOW_LEGACY_SHORTCUTS", default=False)


def allow_direct_user_shortcut(
    origin: Any,
    *,
    degraded_mode: bool = False,
    explicit_opt_in: bool = False,
) -> bool:
    if legacy_shortcuts_enabled():
        return True
    if degraded_mode or explicit_opt_in:
        return True
    return not is_user_facing_origin(origin)


def allow_intent_hint_bypass(context: Mapping[str, Any] | None, origin: Any) -> bool:
    hint = dict((context or {}).get("intent_hint") or {})
    if not hint or not hint.get("tool"):
        return False
    if legacy_shortcuts_enabled():
        return True
    if bool(hint.get("constitutional_hint")) or bool(hint.get("degraded_mode")):
        return True
    return not is_user_facing_origin(origin)


def allow_simple_query_bypass(query: Any, context: Mapping[str, Any] | None = None) -> bool:
    ctx = dict(context or {})
    if legacy_shortcuts_enabled():
        return True
    if bool(ctx.get("constitutional_simple_query_ok")) or bool(ctx.get("degraded_mode")):
        return True
    origin = ctx.get("origin") or ctx.get("request_origin") or ctx.get("source")
    return not is_user_facing_origin(origin)
