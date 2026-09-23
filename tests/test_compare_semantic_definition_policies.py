"""Complete-program comparison must not mistake one matching value for a parse."""

from types import SimpleNamespace

from tools.compare_semantic_definition_policies import _decode


class _Program:
    def __init__(self, name: str, value: int) -> None:
        self.name = name
        self.value = value

    def run(self, _inputs):
        return self.value

    def sha(self):
        return self.name

    def to_dict(self):
        return {"name": self.name}


def test_matching_public_value_does_not_claim_exact_program() -> None:
    target = _Program("target", 3)
    wrong = _Program("wrong", 3)
    item = SimpleNamespace(
        ir=SimpleNamespace(source_token_ids=(1,), source_text_sha256="s", to_program=lambda: target),
        hidden_states=None,
        public_inputs=(3,),
    )
    model = SimpleNamespace(
        model_basis_sha256="basis",
        decode=lambda **_kwargs: SimpleNamespace(ir=SimpleNamespace(to_program=lambda: wrong), reason=""),
    )

    result = _decode(model, item, seconds=1.)

    assert result["decoded"] is True
    assert result["public_value_correct"] is True
    assert result["program_exact"] is False


def test_refusal_is_not_counted_as_correct() -> None:
    item = SimpleNamespace(
        ir=SimpleNamespace(source_token_ids=(1,), source_text_sha256="s"),
        hidden_states=None,
        public_inputs=(3,),
    )
    model = SimpleNamespace(
        model_basis_sha256="basis",
        decode=lambda **_kwargs: SimpleNamespace(ir=None, reason="no_program"),
    )

    assert _decode(model, item, seconds=1.) == {
        "decoded": False,
        "reason": "no_program",
        "program_exact": False,
        "public_value_correct": False,
    }
