"""A syntax success is not a correct or completed public answer."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tools.run_public_shape_canary import assess, load_cases
from tools import run_public_shape_canary as canary


def decoded(text, stop="eos", closed=True):
    return SimpleNamespace(text=text, stop_reason=stop, boundary_closed=closed, receipt=lambda: {})


@pytest.mark.parametrize("text,stop,closed,refused,passed,shape", [
    ('{"product":323}', "eos", True, 0, True, True),
    ('{"product":322}', "eos", True, 0, False, True),
    ('{"product":323}', "token_limit", True, 0, False, True),
    ('{"product":323}', "eos", False, 0, False, True),
    ('{"product":323}', "eos", True, 1, False, True),
    ('[323]', "eos", True, 0, False, False),
    ('{"product":', "eos", True, 0, False, False),
    ('{"product":NaN}', "eos", True, 0, False, False),
])
def test_assessment_separates_shape_semantics_and_completion(text, stop, closed, refused, passed, shape):
    case = {"id": "product", "shape": "object", "expected": {"product": 323}}
    processor = SimpleNamespace(state={"enforcing": True, "refused": refused})
    result = assess(case, decoded(text, stop, closed), processor)
    assert result["passed"] is passed
    assert result["shape_valid"] is shape


def test_cases_must_be_unique_and_shape_consistent(tmp_path):
    path = tmp_path / "cases.json"
    case = {"id": "one", "prompt": "test", "shape": "object", "expected": {}}
    path.write_text(json.dumps([case]))
    assert load_cases(path) == [case]
    for cases in ([], [case, case], [dict(case, expected=[])], [dict(case, prompt="")]):
        path.write_text(json.dumps(cases))
        with pytest.raises(ValueError):
            load_cases(path)


@pytest.mark.parametrize("supported", [True, False])
def test_runner_keeps_expectations_out_of_generation_and_never_claims_qualification(tmp_path, monkeypatch, supported):
    case = {"id": "one", "prompt": "sort 2, 1", "shape": "array", "expected": [1, 2]}
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([case]))
    monkeypatch.setattr("core.brain.llm.model_registry.get_active_cortex_spec",
                        lambda **kwargs: SimpleNamespace(model_path="local-model", descriptor_sha256="basis"))
    monkeypatch.setattr("core.runtime.model_lane_control.standalone_model_lane", lambda **kwargs: nullcontext())
    def template(messages, **kwargs):
        assert messages == [{"role": "user", "content": case["prompt"]}]
        return "sort 2, 1\n<think>"
    tokenizer = SimpleNamespace(apply_chat_template=template)
    monkeypatch.setattr("mlx_lm.load", lambda path: (object(), tokenizer))
    monkeypatch.setattr("core.brain.llm.a_bounded_private_channel._the_token_that_closes_it",
                        lambda tok: 7 if supported else None)
    processor = SimpleNamespace(state={"enforcing": True, "refused": 0})
    def shape(tok, **kwargs):
        assert kwargs == {"after_token": 7, "require": "array"}
        return processor
    monkeypatch.setattr("core.brain.llm.a_shape_the_decoder_enforces.enforce_json", shape)
    calls = []
    def decode(model, tok, prompt, **kwargs):
        calls.append(prompt)
        assert kwargs["logits_processors"] == [processor]
        return decoded("[1, 2]")
    monkeypatch.setattr("core.brain.llm.public_channel_decode.decode_public_sample", decode)
    saved = []
    monkeypatch.setattr(canary, "write_result", lambda path, payload: saved.append(json.loads(json.dumps(payload))))
    args = ["--cases", str(path), "--out", str(tmp_path / "result.json")]
    if not supported:
        with pytest.raises(ValueError, match="private boundary"):
            canary.main(args)
        assert calls == [] and not saved[-1]["complete"]
    else:
        assert canary.main(args) == 0
        assert saved[-1]["complete"] and saved[-1]["passed"]
        assert saved[-2]["results"] and not saved[-2]["complete"]
    assert all(row["qualification"] is False for row in saved)
