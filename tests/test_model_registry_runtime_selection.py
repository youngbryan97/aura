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

    assert model_registry.get_model_path("Qwen2.5-32B-Instruct-8bit") == str(
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
