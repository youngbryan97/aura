from types import SimpleNamespace

import pytest

import core.brain.llm.model_registry as model_registry


def test_default_deep_reasoning_role_uses_the_resident_cortex():
    assert model_registry._default_deep_model_name(backend="mlx") == model_registry.ACTIVE_MODEL


def test_no_distinct_solver_is_configured_by_default(monkeypatch):
    monkeypatch.delenv("AURA_DEEP_MODEL", raising=False)
    monkeypatch.delenv("AURA_LLM__MLX_DEEP_MODEL_PATH", raising=False)
    monkeypatch.setattr(model_registry, "get_runtime_setting", lambda *_args, **_kwargs: "")

    assert model_registry.deep_solver_is_distinctly_configured() is False
    assert model_registry.get_deep_model_name() == model_registry.ACTIVE_MODEL
    assert model_registry.get_deep_model_path() == model_registry.get_runtime_model_path(
        model_registry.ACTIVE_MODEL
    )


def test_an_explicit_distinct_solver_keeps_its_own_identity(monkeypatch, tmp_path):
    solver = tmp_path / "specialist"
    solver.mkdir()
    monkeypatch.setenv("AURA_DEEP_MODEL", "Specialist-34B-4bit")
    monkeypatch.setenv("AURA_LLM__MLX_DEEP_MODEL_PATH", str(solver))
    monkeypatch.setattr(model_registry, "get_runtime_setting", lambda *_args, **_kwargs: "")

    assert model_registry.deep_solver_is_distinctly_configured() is True
    assert model_registry.get_deep_model_name() == "Specialist-34B-4bit"
    assert model_registry.get_deep_model_path() == str(solver.resolve())


def test_resident_deep_role_is_not_a_duplicate_model_lane(monkeypatch):
    monkeypatch.setattr(model_registry, "_configured_deep_model_name", lambda: "")
    monkeypatch.setattr(model_registry, "_configured_deep_model_path", lambda: "")
    _reset_lane_audit_cache()
    try:
        audit = model_registry.audit_lane_assignments(force_refresh=True)
    finally:
        _reset_lane_audit_cache()

    deep = audit["lanes"][model_registry.DEEP_ENDPOINT]
    assert deep["active"] is False
    assert deep["role_mode"] == "resident_systems"
    assert not any(
        issue.get("kind") in {"duplicate_model", "duplicate_runtime_path"}
        and model_registry.DEEP_ENDPOINT in issue.get("lanes", ())
        for issue in audit["issues"]
    )


def test_model_lane_role_comes_from_exact_registry_assignment(monkeypatch):
    assignments = {
        model_registry.PRIMARY_ENDPOINT: "/models/resident-renamed",
        model_registry.DEEP_ENDPOINT: "/models/specialist-renamed",
        model_registry.BRAINSTEM_ENDPOINT: "/models/background-renamed",
        model_registry.FALLBACK_ENDPOINT: "/models/emergency-renamed",
    }
    monkeypatch.setattr(
        model_registry,
        "get_lane_runtime_model_path",
        lambda endpoint: assignments[endpoint],
    )
    monkeypatch.setattr(model_registry, "deep_solver_is_distinctly_configured", lambda: True)

    assert model_registry.get_model_lane_role("/models/resident-renamed") == "cortex"
    assert model_registry.get_model_lane_role("/models/specialist-renamed") == "solver"
    assert model_registry.get_model_lane_role("/models/background-renamed") == "brainstem"
    assert model_registry.get_model_lane_role("/models/emergency-renamed") == "reflex"


def test_runtime_assignment_binds_active_cortex_descriptor(monkeypatch, tmp_path):
    model = tmp_path / "resident"
    model.mkdir()
    descriptor_sha = "a" * 64
    pointer_sha = "b" * 64
    monkeypatch.setattr(model_registry, "get_model_path", lambda _value=None: str(model))
    monkeypatch.setattr(model_registry, "get_model_lane_role", lambda _path: "cortex")
    monkeypatch.setattr(
        model_registry,
        "get_active_cortex_spec",
        lambda: SimpleNamespace(
            model_path=model,
            exact_identity=True,
            descriptor_sha256=descriptor_sha,
            pointer_sha256=pointer_sha,
        ),
    )

    assignment = model_registry.get_model_runtime_assignment(str(model))

    assert assignment.model_path == str(model.resolve())
    assert assignment.role == "cortex"
    assert assignment.qos == "guaranteed"
    assert assignment.artifact_identity == descriptor_sha
    assert assignment.artifact_identity_exact is True
    assert assignment.evidence_receipt_id == pointer_sha


def test_runtime_assignment_does_not_promote_unregistered_heavy_model(
    monkeypatch, tmp_path
):
    model = tmp_path / "Definitely-999B-Cortex-Solver"
    model.mkdir()
    monkeypatch.setattr(model_registry, "get_model_path", lambda _value=None: str(model))
    monkeypatch.setattr(model_registry, "get_model_lane_role", lambda _path: None)
    monkeypatch.setattr(model_registry, "get_active_cortex_spec", lambda: None)
    monkeypatch.setattr(
        "core.brain.llm.model_artifact_profile.get_model_artifact_profile",
        lambda _path: SimpleNamespace(
            measured=True,
            fingerprint="c" * 64,
        ),
    )

    assignment = model_registry.get_model_runtime_assignment(str(model))

    assert assignment.role == "auxiliary"
    assert assignment.lane == "auxiliary"
    assert assignment.qos == "best_effort"
    assert assignment.artifact_identity_kind == "artifact_profile_fingerprint"


def test_runtime_assignment_for_training_cannot_inherit_serving_role(
    monkeypatch, tmp_path
):
    model = tmp_path / "resident"
    model.mkdir()
    monkeypatch.setattr(model_registry, "get_model_path", lambda _value=None: str(model))
    monkeypatch.setattr(model_registry, "get_model_lane_role", lambda _path: "cortex")
    monkeypatch.setattr(model_registry, "get_active_cortex_spec", lambda: None)

    assignment = model_registry.get_model_runtime_assignment(str(model), purpose="train")

    assert assignment.role == "trainer"
    assert assignment.lane == "trainer"
    assert assignment.qos == "best_effort"


def test_model_size_or_name_does_not_assign_a_serving_role(monkeypatch):
    monkeypatch.setattr(
        model_registry,
        "get_lane_runtime_model_path",
        lambda endpoint: f"/assigned/{endpoint.lower()}",
    )
    monkeypatch.setattr(model_registry, "deep_solver_is_distinctly_configured", lambda: True)

    assert model_registry.get_model_lane_role("/unassigned/Qwen-72B-Solver") is None
    assert model_registry.get_model_lane_role("/unassigned/Aura-32B-Cortex") is None


def test_model_identity_compatibility_ignores_parameter_count():
    assert model_registry.model_identities_compatible(
        "Qwen2.5-32B-Instruct-4bit",
        "Qwen2.5-32B-Instruct-8bit",
    )
    assert not model_registry.model_identities_compatible(
        "Qwen2.5-32B-Instruct-4bit",
        "QwQ-32B-4bit",
    )
    assert not model_registry.model_identities_compatible(
        "Qwen2.5-32B-Instruct-4bit",
        "DeepSeek-R1-Distill-Qwen-32B-4bit",
    )


def test_unknown_same_size_model_cannot_inherit_a_nonprimary_lane(monkeypatch):
    monkeypatch.setattr(model_registry, "ACTIVE_MODEL", "Resident-27B-4bit")
    monkeypatch.setattr(model_registry, "BRAINSTEM_MODEL", "Background-9B-4bit")
    monkeypatch.setattr(model_registry, "FALLBACK_MODEL", "Reflex-1.5B-4bit")
    monkeypatch.setattr(
        model_registry, "get_deep_model_name", lambda: "Specialist-70B-4bit"
    )
    monkeypatch.setattr(
        model_registry, "deep_solver_is_distinctly_configured", lambda: True
    )

    assert (
        model_registry.get_endpoint_name_for_model("Unknown-70B-4bit")
        == model_registry.PRIMARY_ENDPOINT
    )
    assert (
        model_registry.get_endpoint_name_for_model("Unknown-9B-4bit")
        == model_registry.PRIMARY_ENDPOINT
    )
    assert (
        model_registry.get_endpoint_name_for_model("Unknown-1.5B-4bit")
        == model_registry.PRIMARY_ENDPOINT
    )
    assert (
        model_registry.get_endpoint_name_for_model("Specialist-70B-4bit")
        == model_registry.DEEP_ENDPOINT
    )


def test_specialist_admission_cache_invalidates_on_evidence_or_source_change(
    monkeypatch,
    tmp_path,
):
    from core.learning import specialist_cortex_admission

    certificate = tmp_path / "admission.json"
    trust_root = tmp_path / "admission.pub.pem"
    specialist = tmp_path / "specialist"
    source = tmp_path / "source.py"
    specialist.mkdir()
    (specialist / "weights.safetensors").write_bytes(b"weights")
    certificate.write_text("certificate-v1", encoding="utf-8")
    trust_root.write_text("trust-root", encoding="utf-8")
    source.write_text("source-v1", encoding="utf-8")

    monkeypatch.setattr(model_registry, "BASE_DIR", tmp_path)
    monkeypatch.setattr(model_registry, "deep_solver_is_distinctly_configured", lambda: True)
    monkeypatch.setattr(model_registry, "deep_solver_artifact_is_ready", lambda: True)
    monkeypatch.setattr(
        model_registry,
        "get_active_cortex_spec",
        lambda **_kwargs: SimpleNamespace(
            exact_identity=True,
            promotion_qualified=True,
            descriptor_sha256="a" * 64,
            pointer_sha256="b" * 64,
        ),
    )
    monkeypatch.setattr(
        model_registry,
        "get_deep_specialist_certificate_path",
        lambda: certificate,
    )
    monkeypatch.setattr(
        model_registry,
        "get_deep_specialist_trust_root_path",
        lambda: trust_root,
    )
    monkeypatch.setattr(model_registry, "get_deep_model_path", lambda: str(specialist))
    monkeypatch.setattr(model_registry, "_current_specialist_source_commit", lambda: "c" * 40)
    monkeypatch.setattr(
        specialist_cortex_admission,
        "REQUIRED_SOURCE_CLOSURE",
        frozenset({"source.py"}),
    )
    wall_clock = [1000.0]
    monkeypatch.setattr(model_registry.time, "time", lambda: wall_clock[0])
    calls = []

    def _verify(*_args, **_kwargs):
        calls.append(True)
        return SimpleNamespace(
            admitted=True,
            reason="qualified",
            expires_at=wall_clock[0] + 10.0,
        )

    monkeypatch.setattr(
        specialist_cortex_admission,
        "verify_specialist_qualification_certificate",
        _verify,
    )
    model_registry.reset_deep_solver_admission_cache()
    try:
        model_registry.get_deep_solver_admission_status("general")
        model_registry.get_deep_solver_admission_status("general")
        assert len(calls) == 1

        certificate.write_text("certificate-version-two", encoding="utf-8")
        model_registry.get_deep_solver_admission_status("general")
        assert len(calls) == 2

        source.write_text("source-version-two", encoding="utf-8")
        model_registry.get_deep_solver_admission_status("general")
        assert len(calls) == 3

        wall_clock[0] += 11.0
        model_registry.get_deep_solver_admission_status("general")
        assert len(calls) == 4
    finally:
        model_registry.reset_deep_solver_admission_cache()


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("mlx", "Qwen2.5-72B-Instruct-4bit"),
        ("retired", "Qwen2.5-72B-Instruct-4bit"),
    ],
)
def test_normalize_runtime_model_name_respects_backend(backend, expected):
    assert (
        model_registry.normalize_runtime_model_name(
            "Qwen2.5-72B-Instruct-Q4",
            backend=backend,
        )
        == expected
    )


def test_external_backend_env_is_ignored(monkeypatch):
    monkeypatch.setenv("AURA_LOCAL_BACKEND", "retired")

    assert model_registry.get_local_backend() == "mlx"
    assert model_registry.local_backend_is_mlx() is True


def test_mlx_client_refuses_retired_external_artifact(monkeypatch, tmp_path):
    from core.brain.llm.mlx_client import get_mlx_client

    monkeypatch.setenv("AURA_LOCAL_BACKEND", "mlx")

    gguf_path = tmp_path / "qwen2.5-32b-instruct-q5_k_m.gguf"

    with pytest.raises(RuntimeError, match="external_cortex_disabled"):
        get_mlx_client(model_path=str(gguf_path))


def test_get_model_path_maps_q4_alias_to_existing_mlx_model_dir(monkeypatch, tmp_path):
    model_dir = tmp_path / "models" / "Qwen2.5-72B-Instruct-4bit"
    model_dir.mkdir(parents=True)

    monkeypatch.setattr(model_registry, "BASE_DIR", tmp_path)
    monkeypatch.setattr(model_registry, "LOCAL_BACKEND", "mlx")
    monkeypatch.setitem(
        model_registry.MODEL_PATHS,
        "Qwen2.5-72B-Instruct-4bit",
        model_dir,
    )
    monkeypatch.setitem(
        model_registry.MODEL_PATHS,
        "Qwen2.5-72B-Instruct-Q4",
        tmp_path / "models" / "Qwen2.5-72B-Instruct-Q4",
    )

    resolved = model_registry.get_model_path("Qwen2.5-72B-Instruct-Q4")

    assert resolved == str(model_dir.resolve())


def test_get_model_path_preserves_missing_absolute_paths(monkeypatch, tmp_path):
    missing = tmp_path / "missing-model"
    monkeypatch.setattr(model_registry, "LOCAL_BACKEND", "mlx")

    assert model_registry.get_model_path(str(missing)) == str(missing)


def test_get_model_path_is_idempotent_for_governed_repository_id(monkeypatch, tmp_path):
    monkeypatch.setattr(model_registry, "BASE_DIR", tmp_path)
    monkeypatch.setattr(model_registry, "_cortex_path_cache", None)
    repository_id = model_registry.get_model_path("Qwen2.5-32B-Instruct-8bit")

    assert repository_id == "mlx-community/Qwen2.5-32B-Instruct-8bit"
    assert model_registry.get_model_path(repository_id) == repository_id


def test_explicit_shared_model_root_is_independent_of_source_root(monkeypatch, tmp_path):
    source_root = tmp_path / "worktree"
    shared_models = tmp_path / "primary" / "models"
    model_dir = shared_models / "Qwen2.5-32B-Instruct-8bit"
    model_dir.mkdir(parents=True)
    monkeypatch.setattr(model_registry, "BASE_DIR", source_root)
    monkeypatch.setenv("AURA_MODELS_DIR", str(shared_models))
    monkeypatch.setattr(model_registry, "_cortex_path_cache", None)

    assert model_registry.get_model_path("Qwen2.5-32B-Instruct-8bit") == str(
        model_dir.resolve()
    )


def test_explicit_promotion_root_wins_from_worktree_source(monkeypatch, tmp_path):
    source_root = tmp_path / "worktree"
    promotion_root = tmp_path / "primary" / "training" / "fused-model"
    promoted_model = promotion_root / "promoted-cortex"
    promoted_model.mkdir(parents=True)
    (promotion_root / "active.json").write_text(
        '{"active_model_path": "' + str(promoted_model) + '"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_registry, "BASE_DIR", source_root)
    monkeypatch.setenv("AURA_FUSED_MODEL_ROOT", str(promotion_root))
    monkeypatch.setattr(model_registry, "_cortex_path_cache", None)

    assert model_registry.get_model_path(model_registry.CORTEX_LOGICAL_NAME) == str(
        promoted_model.resolve()
    )


def _reset_lane_audit_cache():
    model_registry._LANE_AUDIT_CACHE.update(key=None, at=0.0, result=None)


def test_audit_lane_assignments_caches_filesystem_work(monkeypatch):
    # The autouse resource_observer fixture zeroes the audit-cache TTL for
    # hermeticity; this test asserts the CACHE, so it pins a real TTL.
    monkeypatch.setenv("AURA_LANE_AUDIT_CACHE_TTL_S", "60")
    _reset_lane_audit_cache()
    calls = {"n": 0}
    real_realpath = model_registry.os.path.realpath

    def _counting_realpath(path, *args, **kwargs):
        calls["n"] += 1
        return real_realpath(path, *args, **kwargs)

    monkeypatch.setattr(model_registry.os.path, "realpath", _counting_realpath)
    try:
        first = model_registry.audit_lane_assignments(force_refresh=True)
        after_first = calls["n"]
        second = model_registry.audit_lane_assignments()

        assert second == first
        assert calls["n"] == after_first, "cached call must not hit the filesystem"
    finally:
        _reset_lane_audit_cache()


def test_audit_lane_assignments_cache_returns_copies(monkeypatch):
    _reset_lane_audit_cache()
    try:
        first = model_registry.audit_lane_assignments(force_refresh=True)
        first["lanes"].clear()
        second = model_registry.audit_lane_assignments()
        assert second["lanes"], "callers mutating a result must not poison the cache"
    finally:
        _reset_lane_audit_cache()


def test_audit_lane_assignments_invalidates_on_assignment_change(monkeypatch):
    _reset_lane_audit_cache()
    try:
        model_registry.audit_lane_assignments(force_refresh=True)
        monkeypatch.setattr(model_registry, "ACTIVE_MODEL", "totally-new-model")
        refreshed = model_registry.audit_lane_assignments()
        assert (
            refreshed["lanes"][model_registry.PRIMARY_ENDPOINT]["model"]
            == "totally-new-model"
        )
    finally:
        _reset_lane_audit_cache()


def test_audit_lane_assignments_ttl_zero_disables_cache(monkeypatch):
    _reset_lane_audit_cache()
    monkeypatch.setenv("AURA_LANE_AUDIT_CACHE_TTL_S", "0")
    calls = {"n": 0}
    real_uncached = model_registry._audit_lane_assignments_uncached

    def _counting_uncached():
        calls["n"] += 1
        return real_uncached()

    monkeypatch.setattr(
        model_registry, "_audit_lane_assignments_uncached", _counting_uncached
    )
    try:
        model_registry.audit_lane_assignments()
        model_registry.audit_lane_assignments()
        assert calls["n"] == 2
    finally:
        _reset_lane_audit_cache()
