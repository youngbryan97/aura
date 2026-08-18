"""The worker's own scaffolding satisfied the contract it was meant to check.

Operator-evidence answers are delivered as a fixed prefix plus the model's
continuation, and that prefix already contains every required term — objective,
governed, tool, receipt, trace, stop, personhood — and clears the word floor on
its own. `_operator_evidence_model_contribution_insufficient` exists precisely
so the model's share has to carry the claim, and the normal lane hands it the
continuation. The cancellation lane handed it the combined text, so on that
path it measured the prefix and passed. A preempted turn could ship scaffolding
plus a fragment of fluff.

Structured mode had the mirror problem at the other end. The first token was
forced to `encode("{")[0]` — one token id. An array-rooted schema could
therefore never begin, and a tokenizer that merges the brace with what follows
(`{"` is a single token in most BPE vocabularies) had its own natural opening
banned, which is how "forced JSON" produced a first token the model had to
fight.
"""
from __future__ import annotations

import ast
import inspect

import pytest

import core.brain.llm.mlx_worker as worker


# ─────────────────────────── the model's share carries the claim


_PREFIX = worker._OPERATOR_EVIDENCE_PREFIX


def test_the_fixed_prefix_contains_every_required_term():
    """This is why the combined text cannot be the thing under test."""
    body = _PREFIX.lower()

    for term in ("objective", "governed", "tool", "receipt", "trace", "stop", "personhood"):
        assert term in body, term


def test_the_prefix_alone_clears_the_word_floor():
    assert len(_PREFIX.split()) >= 16


def test_the_prefix_alone_is_an_insufficient_contribution():
    assert worker._operator_evidence_model_contribution_insufficient(_PREFIX) is True


def test_a_continuation_that_restates_the_scaffolding_is_insufficient():
    """The floor has to be met with words the prefix did not already supply."""
    echo = (
        "Aura should set an objective, use governed tool actions, keep each "
        "receipt and trace, and stop when blocked or unsafe."
    )

    assert worker._operator_evidence_model_contribution_insufficient(echo) is True


def test_a_substantive_continuation_is_sufficient():
    continuation = (
        "I set the objective from the request, called the governed shell action "
        "once, kept its receipt and the trace identifier, and stopped when the "
        "second command needed an approval nobody had granted."
    )

    assert worker._operator_evidence_model_contribution_insufficient(continuation) is False


def test_the_cancellation_path_judges_the_continuation_not_the_whole_answer():
    """The defect: handing the combined text to a check whose docstring says
    it takes the model's own continuation."""
    fluff = "Yes. That is right."
    delivered = f"{_PREFIX}{fluff}"

    refusal = worker._terminal_contract_refusal(
        {},
        delivered,
        operator_evidence_contract=True,
        model_continuation=fluff,
    )

    assert refusal == "operator_evidence_model_contribution_insufficient"


def test_a_real_continuation_survives_the_cancellation_path():
    continuation = (
        "I set the objective from the request, called the governed shell action "
        "once, kept its receipt and the trace identifier, and stopped when the "
        "second command needed an approval nobody had granted."
    )
    delivered = f"{_PREFIX}{continuation}"

    refusal = worker._terminal_contract_refusal(
        {},
        delivered,
        operator_evidence_contract=True,
        model_continuation=continuation,
    )

    assert refusal == ""


def test_the_cancellation_call_site_passes_the_continuation():
    source = inspect.getsource(worker)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "_terminal_contract_refusal"
        ):
            continue
        passed = {keyword.arg for keyword in node.keywords}
        assert "model_continuation" in passed, (
            "the cancellation path still judges the combined answer"
        )
        return
    raise AssertionError("the terminal-refusal call site was not found")


def test_the_combined_text_is_still_checked_for_the_required_terms():
    """Both halves matter: the delivered answer must contain the terms, and
    the model must have earned them."""
    refusal = worker._terminal_contract_refusal(
        {},
        "Sure, happy to help with that whenever you like.",
        operator_evidence_contract=True,
        model_continuation="Sure, happy to help with that whenever you like.",
    )

    assert refusal == "operator_evidence_fragment_incomplete"


def test_an_empty_answer_refuses_nothing_here():
    """Emptiness is not a contract failure to name; the caller handles it."""
    assert worker._terminal_contract_refusal({}, "", operator_evidence_contract=True) == ""


# ─────────────────────────── every legitimate JSON opening


class _MergingTokenizer:
    """BPE-shaped: the brace merges with whatever follows it."""

    _TABLE = {
        "{": 101,
        '{"': 102,
        "{\n": 103,
        '{ "': 104,
        "[": 201,
        "[{": 202,
        "[\n": 203,
        '["': 204,
    }

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [self._TABLE[text]] if text in self._TABLE else [999]


def test_an_object_schema_admits_only_object_openings():
    assert worker._schema_root_openers({"type": "object"}) == ("{",)


def test_an_array_schema_admits_only_array_openings():
    assert worker._schema_root_openers({"type": "array"}) == ("[",)


def test_a_union_type_admits_both():
    assert worker._schema_root_openers({"type": ["object", "array"]}) == ("{", "[")


@pytest.mark.parametrize("schema", [{}, {"properties": {}}, None, "truthy"])
def test_a_schema_that_declares_no_root_type_admits_both(schema):
    assert worker._schema_root_openers(schema) == ("{", "[")


def test_an_array_rooted_schema_can_actually_start():
    """Forcing `encode("{")[0]` made this impossible."""
    ids = worker._json_start_token_ids(_MergingTokenizer(), {"type": "array"})

    assert 201 in ids
    assert 101 not in ids


def test_a_merged_brace_token_is_admissible():
    """`{"` is one token in most BPE vocabularies, and banning it forced the
    model to fight the opening it would naturally produce."""
    ids = worker._json_start_token_ids(_MergingTokenizer(), {"type": "object"})

    assert 102 in ids
    assert 101 in ids


def test_the_admissible_set_has_no_duplicates():
    class _Flat:
        @staticmethod
        def encode(text, add_special_tokens=False):
            del text, add_special_tokens
            return [7]

    assert worker._json_start_token_ids(_Flat(), {}) == (7,)


def test_a_tokenizer_that_cannot_encode_yields_nothing():
    class _Broken:
        @staticmethod
        def encode(text, add_special_tokens=False):
            del text, add_special_tokens
            raise ValueError("no vocabulary")

    assert worker._json_start_token_ids(_Broken(), {}) == ()


def test_the_processor_masks_every_admissible_opening():
    source = inspect.getsource(worker)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.FunctionDef) and node.name == "json_start_processor"
        ):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "for token_id in start_ids:" in rendered
        assert "brace_id" not in rendered
        return
    raise AssertionError("the JSON start processor was not found")


def test_an_unusable_tokenizer_refuses_rather_than_forcing_one_token():
    source = inspect.getsource(worker)

    assert "tokenizer spells no admissible JSON opening" in source


def test_the_supplied_schema_is_actually_enforced_at_the_end():
    """The start processor shapes the first token; the validator is what
    makes the schema mean something."""
    ok, detail, normalized = worker._validate_schema_output(
        '{"answer": 4}', {"type": "object", "required": ["answer"]}
    )

    assert ok is True
    assert detail == ""
    assert normalized == '{"answer": 4}'


def test_a_wrong_shape_draft_is_refused_by_the_schema():
    ok, detail, _ = worker._validate_schema_output(
        '{"other": 4}', {"type": "object", "required": ["answer"]}
    )

    assert ok is False
    assert detail.startswith("schema_violation:")


def test_an_array_draft_is_located_and_validated():
    ok, _detail, normalized = worker._validate_schema_output(
        "Here you go:\n[1, 2, 3]", {"type": "array"}
    )

    assert ok is True
    assert normalized == "[1, 2, 3]"


def test_prose_alone_is_not_structured_output():
    ok, detail, _ = worker._validate_schema_output("no json here", {"type": "object"})

    assert ok is False
    assert detail == "no_json_value_found"


# ─────────────────────────── an assumed window says it is assumed


def test_a_declared_window_needs_no_assumption(tmp_path):
    import json as _json

    (tmp_path / "config.json").write_text(
        _json.dumps({"max_position_embeddings": 16384})
    )

    assert worker._load_effective_context_window(str(tmp_path)) == 16384


def test_a_sentinel_tokenizer_length_is_bounded(tmp_path):
    import json as _json

    (tmp_path / "tokenizer_config.json").write_text(
        _json.dumps({"model_max_length": int(1e30)})
    )

    window = worker._load_effective_context_window(str(tmp_path))

    assert window == 262144


def test_an_unreadable_checkpoint_records_the_assumption(caplog):
    """A guess that decides how much prompt the worker accepts cannot be
    silent — "the model told us" and "we assumed" must be distinguishable."""
    with caplog.at_level("WARNING"):
        window = worker._load_effective_context_window("/nonexistent/checkpoint")

    assert window == worker._ASSUMED_CONTEXT_WINDOW
    assert any("model_path_missing" in record.getMessage() for record in caplog.records)


def test_a_checkpoint_that_declares_nothing_records_the_assumption(tmp_path, caplog):
    (tmp_path / "config.json").write_text("{}")

    with caplog.at_level("WARNING"):
        window = worker._load_effective_context_window(str(tmp_path))

    assert window == worker._ASSUMED_CONTEXT_WINDOW
    assert any(
        "context_window_undeclared" in record.getMessage() for record in caplog.records
    )


# ─────────────────────────── an exemption outlived its reason


def test_a_leaked_backend_code_is_refused_in_proof_mode_too():
    """The exemption was justified by a pattern that no longer exists: the
    markers are exact-case identifiers matched WITHOUT re.IGNORECASE."""
    leaked = "Result: 42 TOOL_ACTION emitted."

    assert worker._sanitize_telemetry_leakage(leaked, is_proof=True) is None
    assert worker._sanitize_telemetry_leakage(leaked, is_proof=False) is None


def test_the_ordinary_english_word_survives_in_both_modes():
    text = "Proceeding with the derivation, the limit equals 3 as required."

    assert worker._sanitize_telemetry_leakage(text, is_proof=True) == text
    assert worker._sanitize_telemetry_leakage(text, is_proof=False) == text


def test_the_markers_never_match_lowercase_prose():
    for probe in (
        "we are proceeding carefully",
        "field coherence is high",
        "the tool action succeeded",
    ):
        assert worker._BACKEND_SYMBOLIC_SURFACE_MARKERS.search(probe) is None


def test_the_path_wall_check_stays_non_proof():
    """Unlike the symbolic markers there is no exact token separating a
    telemetry wall from a path-aware proof answer."""
    source = inspect.getsource(worker)

    assert "if not is_proof:\n        slash_count" in source


def test_corruption_is_refused_in_every_mode():
    """A corrupted sample is thrown away whether or not it is a proof answer.

    This used to read the SOURCE of `_sanitize_telemetry_leakage` and require
    the literal line `if _contains_corrupted_language(text):` inside it. The
    check then moved into `_telemetry_sanitization_failure_reasons`, with the
    old name kept as a delegating adapter — behaviour identical, wording gone,
    and the test failed with nothing wrong.

    A test pinned to where a check is WRITTEN fails on every refactor that
    keeps the guarantee, which teaches the reader to ignore it. So it asserts
    the guarantee instead: corrupted output does not survive, in either mode.
    """
    corrupted = "asdkjfh qwlekjr zxcvmn"
    assert worker._contains_corrupted_language(corrupted), (
        "the sample no longer reads as corrupted, so this proves nothing"
    )

    for is_proof in (True, False):
        assert worker._sanitize_telemetry_leakage(corrupted, is_proof=is_proof) is None
        assert "corrupted_language" in worker._telemetry_sanitization_failure_reasons(
            corrupted, is_proof=is_proof
        )

    # And clean prose is not swept up by the same rule.
    clean = "Hello, this is a normal sentence."
    assert worker._sanitize_telemetry_leakage(clean, is_proof=False) == clean


# ─────────────────────────── a role label inside an exact value


def test_an_inline_role_label_no_longer_truncates_a_proof_value():
    """A transcript, a format description or a test vector can legitimately
    contain "Assistant:"; the bare substring search shortened the answer while
    the envelope still looked contract-compliant."""
    normalized = worker._normalize_strict_answer_response(
        "The transcript reads Assistant: hello there.", envelope_prefixed=True
    )

    assert normalized == "<answer>The transcript reads Assistant: hello there.</answer>"


def test_a_role_label_at_a_line_boundary_still_ends_the_turn():
    normalized = worker._normalize_strict_answer_response(
        "the value\nAssistant: leftover continuation", envelope_prefixed=True
    )

    assert normalized == "<answer>the value</answer>"


def test_chat_control_tokens_still_truncate_anywhere():
    normalized = worker._normalize_strict_answer_response(
        "the value<|im_end|>trailing", envelope_prefixed=True
    )

    assert normalized == "<answer>the value</answer>"


def test_an_escaped_literal_survives_normalization():
    """Windows paths and regexes are exact proof values."""
    literal = "\\".join(["D:", "work", "x", "re", "d+"])

    normalized = worker._normalize_strict_answer_response(
        literal, envelope_prefixed=True
    )

    assert normalized == f"<answer>{literal}</answer>"


def test_no_envelope_is_manufactured_when_the_prompt_did_not_open_one():
    normalized = worker._normalize_strict_answer_response(
        "plain answer", envelope_prefixed=False
    )

    assert normalized == "plain answer"


def test_a_model_emitted_envelope_survives_verbatim():
    body = "\\".join(["D:", "path", "value"])

    normalized = worker._normalize_strict_answer_response(
        f"<answer>{body}</answer>", envelope_prefixed=False
    )

    assert normalized == f"<answer>{body}</answer>"
