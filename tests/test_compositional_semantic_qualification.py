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


def test_frozen_path_activation_reopens_cleanly():
    activation = _activation()

    assert compositional_semantic_activation_errors(
        activation,
        repo_root=REPO_ROOT,
        selected_model_path=Path(activation["model"]["path"]),
    ) == []


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
