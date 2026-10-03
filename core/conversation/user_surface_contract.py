"""Canonical provenance contract for user-visible validation prompts.

Aura's generation prompt contains system policy, memory evidence, recent
conversation, and live-mind directives.  Reply validation must evaluate only
the person's current request.  This module binds that request at ingress and
lets every downstream layer verify that it has not been substituted.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

USER_SURFACE_PROMPT_BINDING_VERSION = 1
USER_SURFACE_PROMPT_BINDING_KEY = "user_surface_prompt_binding"


def user_surface_prompt_sha256(prompt: Any) -> str:
    return hashlib.sha256(str(prompt or "").encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class UserSurfacePromptResolution:
    prompt: str
    source: str
    sha256: str
    bound: bool
    valid: bool
    error: str = ""


def make_user_surface_prompt_binding(
    prompt: Any,
    *,
    source: str,
) -> dict[str, Any]:
    canonical = str(prompt or "").strip()
    return {
        "version": USER_SURFACE_PROMPT_BINDING_VERSION,
        "prompt": canonical,
        "sha256": user_surface_prompt_sha256(canonical),
        "source": str(source or "unknown")[:120],
    }


def bind_user_surface_prompt(
    context: MutableMapping[str, Any],
    prompt: Any,
    *,
    source: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Bind a canonical visible request into a mutable generation context."""

    existing = context.get(USER_SURFACE_PROMPT_BINDING_KEY)
    if isinstance(existing, Mapping) and not overwrite:
        resolution = resolve_user_surface_prompt(context)
        if resolution.bound and resolution.valid:
            return dict(existing)

    binding = make_user_surface_prompt_binding(prompt, source=source)
    canonical = str(binding["prompt"])
    context["visible_user_message"] = canonical
    context["user_surface_validation_prompt"] = canonical
    context[USER_SURFACE_PROMPT_BINDING_KEY] = binding
    return binding


def resolve_user_surface_prompt(
    payload: Mapping[str, Any] | None,
    *,
    fallback: Any = "",
) -> UserSurfacePromptResolution:
    """Resolve and verify a bound prompt, retaining legacy compatibility.

    A present binding is authoritative.  Any disagreement with the legacy
    prompt fields is a contract violation, not an invitation to guess which
    string is the user's request.
    """

    data = payload if isinstance(payload, Mapping) else {}
    binding = data.get(USER_SURFACE_PROMPT_BINDING_KEY)
    legacy_prompt = str(
        data.get("user_surface_validation_prompt")
        or data.get("visible_user_message")
        or fallback
        or ""
    ).strip()
    # An empty mapping is the absence of a binding, not a binding with no
    # version. The MLX client puts {} in every job whose caller bound nothing,
    # and reading that as a broken binding rejected every draft on the path.
    # LIVE 2026-10-02 20:28: her 1435-character answer to "Bam 83 point game or
    # Kobe 81 point game?" was refused for surface_validation_prompt_binding_version
    # and the 2-bit model's "Kobe's 81." went out in its place, as on every
    # user turn since the 20:07 boot.
    if not isinstance(binding, Mapping) or not binding:
        return UserSurfacePromptResolution(
            prompt=legacy_prompt,
            source="legacy_unbound",
            sha256=user_surface_prompt_sha256(legacy_prompt),
            bound=False,
            valid=bool(legacy_prompt),
            error="" if legacy_prompt else "surface_validation_prompt_missing",
        )

    try:
        version = int(binding.get("version"))
    except (TypeError, ValueError, OverflowError):
        version = -1
    prompt = str(binding.get("prompt") or "").strip()
    source = str(binding.get("source") or "").strip()
    claimed_sha256 = str(binding.get("sha256") or "").strip().lower()
    actual_sha256 = user_surface_prompt_sha256(prompt)

    error = ""
    if version != USER_SURFACE_PROMPT_BINDING_VERSION:
        error = "surface_validation_prompt_binding_version"
    elif not prompt:
        error = "surface_validation_prompt_missing"
    elif not source:
        error = "surface_validation_prompt_binding_source_missing"
    elif claimed_sha256 != actual_sha256:
        error = "surface_validation_prompt_binding_digest_mismatch"
    elif legacy_prompt and legacy_prompt != prompt:
        error = "surface_validation_prompt_binding_value_mismatch"

    return UserSurfacePromptResolution(
        prompt=prompt,
        source=source or "unknown",
        sha256=actual_sha256,
        bound=True,
        valid=not error,
        error=error,
    )
