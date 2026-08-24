"""Reasoning-mode suppression must be proven, not assumed.

Qwen3.5 emits a visible chain of thought unless the chat template is
rendered with ``enable_thinking=False``. Aura's fast lanes (brainstem,
reflex) suppress it, because a model narrating its reasoning inside an
8,000-token budget has less room than the model it replaced.

The failure mode that matters is not a tokenizer that REJECTS the kwarg —
that raises, and the caller falls back to the model default, which is
honest. It is a tokenizer that ACCEPTS the kwarg and ignores it. Jinja
templates never reference undefined variables, so nothing raises, the prompt
is unchanged, and the caller believes it suppressed reasoning while getting
a chain of thought anyway. A failed suppression that looks successful.

These tests pin the probe that catches it.
"""
from __future__ import annotations

import logging

import core.brain.llm.chat_format as chat_format
from core.brain.llm.chat_format import (
    render_chat_continuation_template,
    render_chat_template,
    split_native_thinking_generation,
    template_supports_thinking,
    thinking_enabled_for_generation,
    thinking_enabled_for_model,
    thinking_enabled_for_request,
)


class _Tokenizer:
    """Minimal stand-in with a controllable relationship to the flag."""

    def __init__(self, template: str, *, honours: bool, raises: bool = False) -> None:
        self.chat_template = template
        self._honours = honours
        self._raises = raises
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(dict(kwargs))
        if "enable_thinking" in kwargs:
            if self._raises:
                raise TypeError("unexpected keyword argument 'enable_thinking'")
            if self._honours and kwargs["enable_thinking"] is False:
                return "RENDERED<no-think>"
        return "RENDERED"


def _clear_cache():
    chat_format._THINKING_SUPPORT.clear()


def test_flag_that_changes_the_output_is_supported():
    _clear_cache()
    tok = _Tokenizer("{{ enable_thinking }}", honours=True)
    assert template_supports_thinking(tok) is True


def test_flag_accepted_but_inert_is_NOT_reported_as_supported():
    """The whole point: silently-ignored kwargs must not read as success.

    This tokenizer's template even MENTIONS enable_thinking, so a substring
    check on the template source would have said "supported" and the caller
    would have believed suppression worked.
    """
    _clear_cache()
    tok = _Tokenizer("{{ enable_thinking }} mentioned but unused", honours=False)
    assert template_supports_thinking(tok) is False, (
        "a flag that renders identical output with and without it is inert; "
        "reporting it as supported is a failed suppression that looks "
        "successful"
    )


def test_inert_flag_is_recorded_not_swallowed(monkeypatch):
    """An inert flag must leave a degradation record behind."""
    _clear_cache()
    seen: list[tuple] = []

    import core.runtime.errors as errors

    monkeypatch.setattr(
        errors,
        "record_degradation",
        lambda subsystem, exc, **kw: seen.append((subsystem, str(exc))),
    )
    tok = _Tokenizer("{{ enable_thinking }}", honours=False)
    assert template_supports_thinking(tok) is False
    assert seen, "an inert reasoning flag must be recorded, never swallowed"
    assert "chat_format.thinking_control" == seen[0][0]
    assert "INERT" in seen[0][1]


def test_broken_degradation_recorder_is_visible_without_breaking_render(
    monkeypatch, caplog
):
    _clear_cache()
    import core.runtime.errors as errors

    def _broken_recorder(*_args, **_kwargs):
        raise RuntimeError("recorder unavailable")

    monkeypatch.setattr(errors, "record_degradation", _broken_recorder)
    tok = _Tokenizer("{{ enable_thinking }}", honours=False)

    with caplog.at_level(logging.WARNING, logger="Aura.ChatFormat"):
        assert template_supports_thinking(tok) is False

    assert "Unable to record inert chat-template thinking control" in caplog.text
    assert "recorder unavailable" in caplog.text


def test_rejected_kwarg_is_unsupported_and_not_a_degradation(monkeypatch):
    """A tokenizer that raises is honest — the caller gets the model default."""
    _clear_cache()
    seen: list = []
    import core.runtime.errors as errors

    monkeypatch.setattr(
        errors, "record_degradation", lambda *a, **k: seen.append(a)
    )
    tok = _Tokenizer("no mention here", honours=False, raises=True)
    assert template_supports_thinking(tok) is False
    assert not seen, "a rejected kwarg is not a defect, just an older template"


def test_template_that_never_advertised_the_flag_is_not_a_defect(monkeypatch):
    """Qwen2.5's real behaviour: accepts the kwarg, ignores it, has no
    thinking mode to suppress.

    Jinja never raises on an undefined variable, so the flag lands silently.
    That is inert but CORRECT — the model has no reasoning mode. Warning on
    every load of those models would put noise on a fail-closed subsystem,
    so only a template that ADVERTISES the control and ignores it counts.
    """
    _clear_cache()
    seen: list = []
    import core.runtime.errors as errors

    monkeypatch.setattr(
        errors, "record_degradation", lambda *a, **k: seen.append(a)
    )
    tok = _Tokenizer("a template with no thinking mode at all", honours=False)
    assert template_supports_thinking(tok) is False
    assert not seen, (
        "a template that never claimed to support enable_thinking is an older "
        "model, not a broken control"
    )


def test_render_never_passes_an_unsupported_flag():
    _clear_cache()
    tok = _Tokenizer("no mention here", honours=False, raises=True)
    render_chat_template(tok, [{"role": "user", "content": "hi"}], enable_thinking=False)
    # The probe calls carry the kwarg; the REAL render must not.
    assert "enable_thinking" not in tok.calls[-1], (
        "passing a flag the tokenizer rejects would fail the actual render"
    )


def test_supported_flag_reaches_the_real_render():
    _clear_cache()
    tok = _Tokenizer("{{ enable_thinking }}", honours=True)
    out = render_chat_template(
        tok, [{"role": "user", "content": "hi"}], enable_thinking=False
    )
    assert tok.calls[-1].get("enable_thinking") is False
    assert out == "RENDERED<no-think>"


def test_continuation_template_keeps_the_final_assistant_prefix_open():
    class _ContinuationTokenizer:
        chat_template = "plain"

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["add_generation_prompt"] is False
            assert kwargs["continue_final_message"] is True
            return "rendered-prefix:" + messages[-1]["content"]

    partial = "The function updates the active balances and removes"
    rendered = render_chat_continuation_template(
        _ContinuationTokenizer(),
        [
            {"role": "user", "content": "Explain this function."},
            {"role": "assistant", "content": partial},
        ],
    )

    assert rendered.endswith(partial)


def test_continuation_template_trims_a_legacy_closing_suffix():
    class _LegacyTokenizer:
        chat_template = "plain"

        def apply_chat_template(self, messages, **kwargs):
            if "continue_final_message" in kwargs:
                raise TypeError("unsupported")
            return "rendered-prefix:" + messages[-1]["content"] + "<end>"

    partial = "The function updates the active balances and removes"
    rendered = render_chat_continuation_template(
        _LegacyTokenizer(),
        [
            {"role": "user", "content": "Explain this function."},
            {"role": "assistant", "content": partial},
        ],
    )

    assert rendered.endswith(partial)
    assert "<end>" not in rendered


def test_probe_runs_once_per_template():
    """The probe is deterministic per template — it must not run per render."""
    _clear_cache()
    tok = _Tokenizer("{{ enable_thinking }}", honours=True)
    template_supports_thinking(tok)
    after_first = len(tok.calls)
    for _ in range(5):
        template_supports_thinking(tok)
    assert len(tok.calls) == after_first, "probe result must be memoised"


def test_lane_policy_pins_only_the_fast_lanes():
    assert thinking_enabled_for_model("models/Qwen3.5-9B-4bit") is False
    assert thinking_enabled_for_model("models/Qwen2.5-1.5B-Instruct-4bit") is False
    assert thinking_enabled_for_model("models/Qwen3.8-27B-4bit") is None
    # Deliberation is the point in these two; they keep the artifact default.
    assert thinking_enabled_for_model("models/Qwen2.5-32B-Instruct-8bit") is None
    assert thinking_enabled_for_model("models/Qwen2.5-72B-Instruct-4bit") is None
    assert thinking_enabled_for_model(None) is None


def test_request_cognitive_mode_controls_native_thinking_on_cortex():
    model = "models/Qwen3.8-27B-4bit"

    assert thinking_enabled_for_request(model, cognitive_mode="reactive") is False
    assert thinking_enabled_for_request(model, cognitive_mode="dormant") is False
    assert thinking_enabled_for_request(model, cognitive_mode="fast") is False
    assert thinking_enabled_for_request(model, cognitive_mode="deliberate") is True
    assert thinking_enabled_for_request(model, cognitive_mode="deep") is True
    assert thinking_enabled_for_request(model, cognitive_mode="reflective") is True


def test_clean_user_surface_is_a_direct_render_after_architectural_reasoning():
    model = "models/Qwen3.8-27B-4bit"

    assert (
        thinking_enabled_for_generation(
            model,
            cognitive_mode="deep",
            final_user_surface=True,
        )
        is False
    )
    assert (
        thinking_enabled_for_generation(
            model,
            cognitive_mode="deep",
            final_user_surface=False,
        )
        is True
    )


def test_template_opened_reasoning_is_not_surface_before_typed_boundary():
    opened = split_native_thinking_generation(
        "We need derive the answer and verify each requested part.",
        native_thinking=True,
    )
    assert opened.reasoning.startswith("We need derive")
    assert opened.surface == ""
    assert opened.boundary_closed is False

    closed = split_native_thinking_generation(
        "private work\n</think>\n\nThe complete answer.",
        native_thinking=True,
    )
    assert closed.reasoning == "private work"
    assert closed.surface == "The complete answer."
    assert closed.boundary_closed is True

    direct = split_native_thinking_generation(
        "The complete answer.",
        native_thinking=False,
    )
    assert direct.reasoning == ""
    assert direct.surface == "The complete answer."
    assert direct.boundary_closed is True


def test_request_cognitive_mode_does_not_turn_a_fast_role_into_a_slow_lane():
    assert (
        thinking_enabled_for_request(
            "models/Qwen3.5-9B-Brainstem-4bit",
            cognitive_mode="deliberate",
        )
        is False
    )


def test_absent_or_unknown_request_mode_keeps_artifact_policy():
    model = "models/Qwen3.8-27B-4bit"

    assert thinking_enabled_for_request(model, cognitive_mode=None) is None
    assert thinking_enabled_for_request(model, cognitive_mode="not-a-mode") is None
