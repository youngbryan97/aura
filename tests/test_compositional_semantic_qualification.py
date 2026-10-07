import json
from copy import deepcopy
from pathlib import Path

from core.learning.compositional_semantic_qualification import (
    compositional_semantic_activation_errors,
)
from core.learning.semantic_program_ordinary_baseline import canonical_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVATION_PATH = (
    REPO_ROOT / "artifacts/rlc/semantic_program_27b_frozen_path_v1/activation.json"
)


def _activation():
    return json.loads(ACTIVATION_PATH.read_text(encoding="ascii"))


def _reseal(value):
    body = dict(value)
    body.pop("activation_sha256", None)
    value["activation_sha256"] = canonical_sha256(body)


def test_frozen_path_activation_is_refused_once_its_decoder_changed():
    """The package was qualified against a decoder G03 has since rewritten.

    Its own evidence still reopens; the source it was measured on does not, so
    the runtime refuses it until a reader is qualified on the current code.
    """
    activation = _activation()

    assert compositional_semantic_activation_errors(
        activation,
        repo_root=REPO_ROOT,
        selected_model_path=Path(activation["model"]["path"]),
    ) == ["source_contract_drift"]


def test_activation_authority_cannot_be_granted_by_resealing_the_envelope():
    activation = _activation()
    activation["serving_authority"] = True
    _reseal(activation)

    assert "authority" in compositional_semantic_activation_errors(
        activation,
        repo_root=REPO_ROOT,
    )


def test_activation_rejects_a_missing_evidence_member_after_reseal():
    activation = _activation()
    activation["evidence"] = deepcopy(activation["evidence"])
    activation["evidence"]["mechanism"]["path"] = "artifacts/rlc/absent.json"
    _reseal(activation)

    assert "evidence_invalid:mechanism" in compositional_semantic_activation_errors(
        activation,
        repo_root=REPO_ROOT,
    )


def test_activation_rejects_non_mapping_evidence_without_raising():
    activation = _activation()
    activation["evidence"] = []
    _reseal(activation)

    assert "evidence" in compositional_semantic_activation_errors(
        activation,
        repo_root=REPO_ROOT,
    )


def test_a_reader_qualifies_only_when_every_g10_predicate_holds():
    from core.learning.compositional_semantic_qualification import (
        SEMANTIC_READER_PREDICATES,
        SEMANTIC_READER_QUALIFICATION_SCHEMA,
        semantic_reader_qualification_errors,
    )

    passing = {"schema": SEMANTIC_READER_QUALIFICATION_SCHEMA, "requests": 416,
               "predicates": dict.fromkeys(SEMANTIC_READER_PREDICATES, True)}
    assert semantic_reader_qualification_errors(passing) == []

    lesion_blind = deepcopy(passing)
    lesion_blind["predicates"]["lesion_shows_the_check_can_fail"] = False
    assert semantic_reader_qualification_errors(lesion_blind) == ["failed:lesion_shows_the_check_can_fail"]

    missing = deepcopy(passing)
    del missing["predicates"]["rollback_restores_generation"]
    assert semantic_reader_qualification_errors(missing) == ["predicates"]

    empty = {**passing, "requests": 0}
    assert semantic_reader_qualification_errors(empty) == ["no_requests"]


def test_a_reader_package_is_named_by_its_receipt():
    from core.learning.compositional_semantic_qualification import semantic_reader_package_id

    assert semantic_reader_package_id("033feffa21a4415d" + "0" * 48) == "semantic-reader-27b-033feffa21a4"
