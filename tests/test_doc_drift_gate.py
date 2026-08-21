"""Every check in the documentation gate fires on the defect that motivated it.

A gate that cannot match reports green forever. ARTIFACT_INDEX.md pointed all
twelve of its links into an ignored directory for months and looked correct in
every review, because a dead relative link renders as ordinary blue text.

Each case below is the real defect the check was written for, and each negative
is a real construction from this tree that a sloppier rule ate: a proposal
naming what it proposes building, a correction log naming what it just
corrected, and an artifact index naming where a proof run writes.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "lint_doc_drift", ROOT / "tools" / "lint_doc_drift.py"
)
assert _spec and _spec.loader
gate = importlib.util.module_from_spec(_spec)
sys.modules["lint_doc_drift"] = gate
_spec.loader.exec_module(gate)


@pytest.fixture
def scan(tmp_path, monkeypatch):
    """Scan one document written into a scratch copy of the repo root.

    The gate resolves paths and git state against its own ROOT, so the fixture
    moves ROOT rather than the document — anything else would test the real
    tree and pass for the wrong reason.
    """
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "_ignored_cache", {})
    monkeypatch.setattr(gate, "_published_cache", {})

    def run(body, *, targets=(), published=(), env=((), ()), suite=None,
            ignored=(), files=(), symbols=(), routes=(), labels=()):
        for rel in files:
            f = tmp_path / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("x\n")
        doc = tmp_path / "DOC.md"
        doc.write_text(body)
        monkeypatch.setattr(gate, "is_output_path", lambda rel: rel in ignored)
        monkeypatch.setattr(
            gate, "is_published",
            lambda rel: rel in published or any(
                p.startswith(rel.rstrip("/") + "/") for p in published),
        )
        found = gate.scan("DOC.md", set(targets), {}, (set(env[0]), tuple(env[1])),
                          suite, set(symbols), set(routes), set(labels))
        return {f["kind"] for f in found}

    return run


# ---- a path written as a path ------------------------------------------


def test_a_path_that_is_not_there_is_reported(scan):
    assert "missing_path" in scan("See `core/gone/module.py` for the detail.")


def test_a_path_that_is_there_is_not(scan):
    assert scan("See `core/here/module.py`.", files=["core/here/module.py"]) == set()


def test_a_word_with_a_slash_is_not_a_path(scan):
    assert scan("The read/write split is in the gateway.") == set()


def test_a_destination_the_repository_ignores_is_not_a_broken_path(scan):
    """ARTIFACT_INDEX.md names where `make final-proof` writes. That is its job."""
    body = "`make final-proof` writes `artifacts/current/enterprise_gate.json`."
    assert scan(body, targets=["final-proof"],
                ignored=["artifacts/current/enterprise_gate.json"]) == set()


def test_a_placeholder_names_a_shape_not_a_file(scan):
    assert scan("Each frame is `frames/NNNNNNNN.json`.") == set()


def test_a_paragraph_that_says_the_file_is_gone_is_not_corrected(scan):
    """docs/DOC_STATUS.md: naming an absent file is the point."""
    body = (
        "These specs never existed. Git history contains no commit adding\n"
        "`docs/ABUSE_GAUNTLET.md` or `docs/DEPTH_AUDIT.md`.\n"
    )
    assert scan(body) == set()


def test_the_cue_covers_the_paragraph_and_not_the_next_one(scan):
    body = (
        "These were never built: `core/a/gone.py`.\n"
        "\n"
        "The live path is `core/b/also_gone.py`.\n"
    )
    assert "missing_path" in scan(body)


# ---- links ---------------------------------------------------------------


def test_a_link_git_does_not_publish_is_broken(scan):
    """The reader gets a 404 even when the author's working copy has the file."""
    body = "[the report](artifacts/current/FINAL_CLOSURE_REPORT.md)"
    assert "dead_link" in scan(body, ignored=["artifacts/current/FINAL_CLOSURE_REPORT.md"])


def test_a_file_on_disk_that_git_ignores_is_still_an_unpublished_link(scan):
    body = "[the report](build/report.md)"
    assert "unpublished_link" in scan(body, files=["build/report.md"])


def test_a_published_link_passes(scan):
    body = "[the guide](docs/GUIDE.md)"
    assert scan(body, files=["docs/GUIDE.md"], published=["docs/GUIDE.md"]) == set()


def test_an_external_link_is_left_alone(scan):
    assert scan("[docs](https://example.com/x.md)") == set()


# ---- headings ------------------------------------------------------------


def test_a_truncated_anchor_is_reported(scan):
    """ARCHITECTURE.md linked #0-the-unified-will at a heading GitHub slugs longer."""
    body = "# Doc\n\n[Will](#0-the-unified-will)\n\n## 0. The Unified Will: decision authority\n"
    assert "bad_anchor" in scan(body)


def test_the_full_anchor_resolves(scan):
    body = ("# Doc\n\n[Will](#0-the-unified-will-decision-authority)\n\n"
            "## 0. The Unified Will: decision authority\n")
    assert scan(body) == set()


# ---- make ----------------------------------------------------------------


def test_a_make_target_that_does_not_exist_is_reported(scan):
    assert "missing_make_target" in scan("Run `make nonexistent-target`.")


def test_make_as_an_english_verb_is_not_a_target(scan):
    assert scan("This will make an ordinary reader stop and check.") == set()


# ---- environment variables ----------------------------------------------


def test_a_documented_lever_with_no_reader_is_reported(scan):
    """HUMAN_OVERRIDE_POLICY.md offered AURA_TOOLS_ENABLED. Nothing read it."""
    assert "env_var_has_no_reader" in scan("Set `AURA_TOOLS_ENABLED=false`.")


def test_a_name_built_from_a_prefix_at_runtime_counts_as_read(scan):
    """`f"AURA_FLAG_{name.upper()}"` produces names that appear nowhere."""
    body = "Set `AURA_FLAG_WORKSPACE_JAIL_ENABLED=1`."
    assert scan(body, env=((), ("AURA_FLAG_",))) == set()


def test_the_bare_namespace_is_not_a_prefix_that_vouches_for_anything(scan):
    """One f-string yields `AURA_`, which would otherwise allow every name."""
    body = "Set `AURA_MADE_UP_LEVER=1`."
    assert "env_var_has_no_reader" in scan(body, env=((), ("AURA_FLAG_",)))


def test_a_name_mentioned_to_say_it_does_not_exist_is_left_alone(scan):
    """OPERATOR_GUIDE.md warns readers off AURA_MEM_THRESHOLDS."""
    assert scan("There is no `AURA_MEM_THRESHOLDS` variable.") == set()


def test_a_documented_variable_something_reads_passes(scan):
    body = "Set `AURA_FOREGROUND_ONLY=1`."
    assert scan(body, env=({"AURA_FOREGROUND_ONLY"}, ())) == set()


# ---- numbers -------------------------------------------------------------


def test_a_stale_suite_size_is_reported(scan):
    """Eight documents said 34,382 across 2,373 long after the tree passed 40,000."""
    body = "The tree collects **34,382 tests across 2,373 files**."
    assert "stale_suite_size" in scan(body, suite=(40139, 2697))


def test_the_recorded_suite_size_passes(scan):
    body = "The tree collects **40,139 tests across 2,697 files**."
    assert scan(body, suite=(40139, 2697)) == set()


def test_a_quoted_count_is_a_citation_not_a_claim(scan):
    """DOC_STATUS.md's correction log quotes the numbers it replaced."""
    body = '| "24,931 tests across 1,771 files" (8 documents) | corrected |'
    assert scan(body, suite=(40139, 2697)) == set()


def test_a_stale_line_count_is_reported(scan, tmp_path):
    (tmp_path / "core").mkdir(parents=True, exist_ok=True)
    (tmp_path / "core" / "m.py").write_text("a\nb\nc\n")
    assert "stale_line_count" in scan("`core/m.py` (99 lines) does the work.")


def test_an_accurate_line_count_passes(scan, tmp_path):
    (tmp_path / "core").mkdir(parents=True, exist_ok=True)
    (tmp_path / "core" / "m.py").write_text("a\nb\nc\n")
    assert scan("`core/m.py` (3 lines) does the work.") == set()


def test_a_stale_directory_count_is_reported(scan, tmp_path):
    """docs/README.md promised 38 runbooks against 39."""
    for n in ("a", "b"):
        (tmp_path / "rb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "rb" / f"{n}.md").write_text("x\n")
    body = "| [rb/](rb/) | 5 incident procedures, one per failure mode |"
    assert "stale_dir_count" in scan(body, published=["rb/a.md"])


# ---- cited tests ---------------------------------------------------------


def test_a_cited_test_file_with_no_tests_is_reported(scan, tmp_path):
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "test_gone.py").write_text("# everything was renamed\n")
    assert "cited_test_has_no_tests" in scan("Held by `tests/test_gone.py`.")


def test_a_cited_test_file_with_tests_passes(scan, tmp_path):
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "test_real.py").write_text("def test_x():\n    assert True\n")
    assert scan("Held by `tests/test_real.py`.") == set()


def test_a_class_based_test_file_counts(scan, tmp_path):
    """Several suites here put their tests in classes; those are still tests."""
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "test_cls.py").write_text(
        "class TestThing:\n    async def test_x(self):\n        assert True\n"
    )
    assert scan("Held by `tests/test_cls.py`.") == set()


# ---- symbols -------------------------------------------------------------


def test_a_class_the_code_does_not_define_is_reported(scan):
    """OWNERSHIP.md named SelfModificationEngine beside the file holding
    AutonomousSelfModificationEngine: the path resolved, the sentence did not."""
    body = "| Self-modification | `SelfModificationEngine` | owner |"
    assert "symbol_not_defined" in scan(body, symbols={"AutonomousSelfModificationEngine"})


def test_a_class_the_code_defines_passes(scan):
    body = "The owner is `AutonomousSelfModificationEngine`."
    assert scan(body, symbols={"AutonomousSelfModificationEngine"}) == set()


def test_an_acronym_is_not_a_symbol(scan):
    assert scan("Measured with `IIT` and returned as `JSON`.") == set()


def test_a_builtin_quoted_from_a_traceback_is_not_ours_to_define(scan):
    assert scan("The lane raises `TimeoutError` under load.") == set()


def test_a_symbol_named_as_retired_is_left_alone(scan):
    body = "`HaltingState` was removed; the fields live on the receipt now."
    assert scan(body, symbols={"EpisodeReceipt"}) == set()


# ---- routes --------------------------------------------------------------


def test_a_route_missing_its_mount_prefix_is_reported(scan):
    """Three documents said POST /memory/delete for a route served at /api."""
    body = "Delete one memory with `POST /memory/delete`."
    assert "route_not_served" in scan(body, routes={"/api/memory/delete"})


def test_the_served_path_passes(scan):
    body = "Delete one memory with `POST /api/memory/delete`."
    assert scan(body, routes={"/api/memory/delete"}) == set()


def test_a_path_parameter_may_be_named_anything(scan):
    body = "Fetch it with `GET /api/inner-state/will-receipt/{receipt_id}`."
    assert scan(body, routes={"/api/inner-state/will-receipt/{}"}) == set()


def test_a_route_recorded_as_removed_is_left_alone(scan):
    body = "`GET /api/diagnostics/reliability` was removed; use the bundle."
    assert scan(body, routes={"/api/health"}) == set()


def test_routes_resolve_both_prefixes_from_the_real_tree():
    """APIRouter(prefix=...) and include_router(prefix=...) both count."""
    routes = gate.declared_routes()
    assert "/api/memory/delete" in routes
    assert "/api/allostasis/forecasts" in routes, "router-level prefix dropped"
    assert "/api/devices/revoke-scope" in routes


# ---- where a document says to click -------------------------------------


def test_a_settings_group_the_ui_does_not_have_is_reported(scan):
    """USER_GUIDE.md sent a stuck user to Settings -> Models. No such group."""
    body = "Fix it under Settings → Models → Reset cortex."
    assert "settings_group_not_in_ui" in scan(body, labels={"voice", "memory"})


def test_a_real_settings_group_passes(scan):
    assert scan("Settings → Voice holds both toggles.", labels={"voice"}) == set()


def test_the_operating_system_settings_are_not_ours(scan):
    body = "Grant it in System Settings → Privacy → Microphone."
    assert scan(body, labels={"voice"}) == set()


def test_an_arrow_between_pipeline_stages_is_not_navigation(scan):
    body = "User Input → Sanitizer → Working Memory → Model Context."
    assert scan(body, labels={"voice"}) == set()


# ---- the tree itself -----------------------------------------------------


def test_the_baseline_is_a_ratchet_that_only_shrinks():
    """A baseline that can rise launders exactly the debt the gate catches."""
    baseline = json.loads((ROOT / "config" / "doc_drift_baseline.json").read_text())
    assert baseline["total"] == 0, (
        "the documentation baseline is zero; a rise means findings were "
        "recorded rather than fixed"
    )


def test_every_current_document_still_resolves():
    """The gate over the real tree, which is what CI runs."""
    findings = []
    targets = gate.make_targets()
    env_names = gate.readable_env_names()
    suite = gate.recorded_suite_size()
    symbols = gate.defined_symbols()
    routes = gate.declared_routes()
    labels = gate.ui_labels()
    cache: dict[str, set[str]] = {}
    for rel in gate.tracked_docs():
        findings.extend(
            gate.scan(rel, targets, cache, env_names, suite, symbols, routes, labels))
    assert findings == [], "\n".join(
        f"{f['doc']}:{f['line']} {f['kind']}: {f['detail']}" for f in findings
    )
