"""Whole-agent lineage runs inside a boundary or it does not run.

Reproduction is required for a strict evolutionary claim and not for a
cognitive one — a mule is still an organism — so wiring whole-agent
self-reproduction into production merely because it is currently unwired
would not improve anything and would open questions about safety and identity
that nothing in the runtime answers.

The other option is this: an isolated ecology with the boundaries written
down and enforced, so the mechanism can be studied without the live instance
being given reproductive authority over itself. Each test is one boundary.
"""

from __future__ import annotations

import pytest

from core.self_modification.lineage_enclosure import (
    FORBIDDEN_INHERITANCE,
    AuthorityViolationError,
    Budget,
    Enclosure,
    EnclosureError,
    EnclosureExhaustedError,
    check_inheritance,
)


@pytest.fixture
def enclosure(tmp_path):
    with Enclosure(tmp_path / "ecology", budget=Budget(generations=3, population=5)) as e:
        yield e


# ── resource boundary ────────────────────────────────────────────────────


def test_population_is_capped(enclosure):
    parent = enclosure.genesis({"trait": 0.5})
    for _ in range(4):
        enclosure.fork(parent.snapshot_id)
    with pytest.raises(EnclosureExhaustedError):
        enclosure.fork(parent.snapshot_id)


def test_generations_are_capped(enclosure):
    for _ in range(3):
        enclosure.advance_generation()
    with pytest.raises(EnclosureExhaustedError):
        enclosure.advance_generation()


def test_wall_clock_is_capped(tmp_path):
    clock = [0.0]
    with Enclosure(tmp_path / "e", budget=Budget(seconds=1.0), now=lambda: clock[0]) as e:
        e.genesis({"trait": 0.1})
        clock[0] = 2.0
        with pytest.raises(EnclosureExhaustedError):
            e.check()


def test_an_exhausted_enclosure_stays_halted(enclosure):
    for _ in range(3):
        enclosure.advance_generation()
    with pytest.raises(EnclosureExhaustedError):
        enclosure.advance_generation()
    with pytest.raises(EnclosureExhaustedError):
        enclosure.genesis({"trait": 0.2})


def test_a_budget_with_nothing_in_it_is_refused():
    with pytest.raises(EnclosureError):
        Budget(generations=0)
    with pytest.raises(EnclosureError):
        Budget(seconds=0.0)


def test_a_refused_fork_is_not_charged(enclosure):
    parent = enclosure.genesis({"trait": 0.5})
    for _ in range(4):
        enclosure.fork(parent.snapshot_id)
    before = enclosure.spend.population
    with pytest.raises(EnclosureExhaustedError):
        enclosure.fork(parent.snapshot_id)
    assert enclosure.spend.population == before


# ── authority boundary ───────────────────────────────────────────────────


def test_nothing_is_written_outside_the_enclosure(tmp_path):
    with Enclosure(tmp_path / "ecology") as e:
        parent = e.genesis({"trait": 0.5})
        e.fork(parent.snapshot_id)
        assert (e.root / "lineage.sqlite3").exists()
    assert not list(tmp_path.glob("lineage.sqlite3"))


def test_an_enclosure_inside_the_live_state_root_is_refused():
    from core.config import config

    live = getattr(config.paths, "data_dir", None)
    if not live:
        pytest.skip("no live data dir configured in this environment")
    with pytest.raises(AuthorityViolationError):
        Enclosure(str(live) + "/lineage_experiment")


def test_the_manager_never_takes_its_own_default_path(tmp_path):
    """LineageManager's default is the live data directory."""
    with Enclosure(tmp_path / "ecology") as e:
        manager = e.manager()
        assert str(e.root) in str(manager._db_path)


def test_the_enclosure_has_no_way_to_install_a_snapshot():
    """Selection scores configurations. Promotion is a person's decision."""
    forbidden = {"install", "promote", "adopt", "become", "spawn", "launch", "apply_to_live"}
    surface = {name for name in dir(Enclosure) if not name.startswith("_")}
    assert not (surface & forbidden), f"the enclosure grew a promotion path: {surface & forbidden}"


# ── identity boundary ────────────────────────────────────────────────────


@pytest.mark.parametrize("key", sorted(FORBIDDEN_INHERITANCE))
def test_no_identity_bearing_key_may_be_inherited(enclosure, key):
    with pytest.raises(AuthorityViolationError):
        enclosure.genesis({"trait": 0.5, key: "whatever"})


def test_a_nested_identity_key_is_caught(enclosure):
    with pytest.raises(AuthorityViolationError):
        enclosure.genesis({"aura.entity_key": "x"})


def test_an_identity_key_is_refused_not_stripped(enclosure):
    """Dropping it silently makes a child that looks like it inherited it."""
    with pytest.raises(AuthorityViolationError):
        enclosure.genesis({"entity_key": "abc", "trait": 0.5})
    assert enclosure.report().refusals, "the refusal left no record"


def test_an_ordinary_configuration_is_allowed(enclosure):
    snapshot = enclosure.genesis({"trait": 0.5, "curiosity_weight": 0.2})
    assert snapshot.generation == 0
    assert check_inheritance({"trait": 0.5}) == ()


# ── the record ───────────────────────────────────────────────────────────


def test_the_report_says_what_stopped_the_run(enclosure):
    for _ in range(3):
        enclosure.advance_generation()
    with pytest.raises(EnclosureExhaustedError):
        enclosure.advance_generation()
    report = enclosure.report()
    assert "generations" in report.halted_by
    assert report.budget.generations == 3


def test_dispose_removes_the_ecology(tmp_path):
    e = Enclosure(tmp_path / "ecology")
    e.genesis({"trait": 0.5})
    root = e.root
    assert root.exists()
    e.dispose()
    assert not root.exists()


def test_the_enclosure_writes_through_the_governed_path(tmp_path):
    """A write nobody can audit is not isolated, it is unobserved."""
    import inspect

    from core.self_modification import lineage_enclosure

    source = inspect.getsource(lineage_enclosure)
    assert "shutil.rmtree" not in source
    assert "root.mkdir" not in source
    assert "get_file_write_gateway" in source
    assert "local_internal_governed_scope" in source


def test_it_refuses_to_exist_rather_than_write_ungoverned(tmp_path, monkeypatch):
    """A fallback that bypasses governance is taken exactly when governance
    is broken, which is the worst moment to create a directory for something
    that reproduces."""
    from core.self_modification import lineage_enclosure

    def broken(*args, **kwargs):
        raise RuntimeError("the write gateway is down")

    monkeypatch.setattr(lineage_enclosure, "_governed_mkdir", broken)
    with pytest.raises(RuntimeError):
        Enclosure(tmp_path / "ecology")
