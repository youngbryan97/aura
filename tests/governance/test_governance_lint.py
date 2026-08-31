"""Effect-ownership lint and baseline-ratchet invariants."""

from __future__ import annotations

import ast
import os
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

from core.runtime.subprocess_gateway import get_subprocess_gateway
from tools.lint_governance import (
    EffectBucket,
    _canonical_owner,
    _scan_tree_scoped,
    _SubprocessDeclarationVisitor,
    compare_inventory,
)

REPO = Path(__file__).resolve().parents[2]


def _run_lint() -> int:
    env = os.environ.copy()
    cmd = [sys.executable, str(REPO / "tools" / "lint_governance.py")]
    proc = get_subprocess_gateway().run(
        cmd,
        cwd=str(REPO),
        env=env,
        capture_output=True,
        timeout=60,
        offline_tooling=True,
        source="certification_tooling:test_governance_lint",
        accelerator_capability="auto",
    )
    return proc.returncode


def test_search_history_owns_governed_writes_not_raw_mutations():
    path = "core/cognition/how_she_learns_to_look.py"
    assert _canonical_owner("file_write_gateway", path)
    assert not _canonical_owner("raw_file_mutation", path)


def test_lint_passes_on_repo():
    rc = _run_lint()
    assert rc in (0,)  # accept 0 — anything else means a real violation in tree


def test_lint_detects_forbidden_call() -> None:
    bad = REPO / "core" / "_governance_lint_test_violator.py"
    try:
        bad.write_text(
            textwrap.dedent('''
            """ephemeral test file: must trigger governance lint"""

            import subprocess

            def use_unsafe() -> None:
                subprocess.run(["echo", "unsafe"], check=False)
        '''),
            encoding="utf-8",
        )
        rc = _run_lint()
        assert rc == 1
    finally:
        bad.unlink(missing_ok=True)


def test_scanner_resolves_aliases_factories_and_path_mutations() -> None:
    tree = ast.parse(
        textwrap.dedent(
            """
            import subprocess as sp
            from pathlib import Path as P
            from core.runtime.network_gateway import get_network_gateway as network

            def perform() -> None:
                target = P("artifact.txt")
                target.write_text("body")
                sp.run(["echo", "hello"], check=False)
                gateway = network()
                gateway.request("POST", "https://example.test")
            """
        )
    )

    buckets = _scan_tree_scoped(tree, "core/synthetic.py")
    categories = {key[0] for key in buckets}
    assert categories == {"network_gateway", "raw_file_mutation", "raw_subprocess"}
    assert all(key[2] == "<module>.perform" for key in buckets)


def test_subprocess_gateway_requires_accelerator_declaration() -> None:
    tree = ast.parse(
        textwrap.dedent(
            """
            from core.runtime.subprocess_gateway import get_subprocess_gateway

            def perform() -> None:
                gateway = get_subprocess_gateway()
                gateway.run(["git", "status"], read_only=True, source="test")
                gateway.run(
                    ["git", "status"],
                    read_only=True,
                    source="test",
                    accelerator_capability="none",
                )
            """
        )
    )
    visitor = _SubprocessDeclarationVisitor(relative_path="core/synthetic.py")
    visitor.visit(tree)

    assert len(visitor.violations) == 1
    assert "accelerator_capability_undeclared" in visitor.violations[0].problem


def test_threaded_subprocess_gateway_requires_accelerator_declaration() -> None:
    tree = ast.parse(
        textwrap.dedent(
            """
            import asyncio
            from core.runtime.subprocess_gateway import get_subprocess_gateway

            async def perform() -> None:
                await asyncio.to_thread(
                    get_subprocess_gateway().run,
                    ["open", "https://example.test"],
                    source="test",
                )
                await asyncio.to_thread(
                    get_subprocess_gateway().run,
                    ["open", "https://example.test"],
                    source="test",
                    accelerator_capability="none",
                )
            """
        )
    )
    visitor = _SubprocessDeclarationVisitor(relative_path="core/synthetic.py")
    visitor.visit(tree)

    assert len(visitor.violations) == 1
    assert "accelerator_capability_undeclared" in visitor.violations[0].problem
    assert "get_subprocess_gateway().run" in visitor.violations[0].problem


def test_scanner_catches_context_bound_multiprocessing_processes() -> None:
    tree = ast.parse(
        textwrap.dedent(
            """
            import multiprocessing as mp

            def perform(target) -> None:
                ctx = mp.get_context("spawn")
                ctx.Process(target=target)
                mp.get_context("spawn").Process(target=target)
            """
        )
    )

    buckets = _scan_tree_scoped(tree, "core/synthetic.py")

    assert sum(buckets.values()) == 2
    assert {key[0] for key in buckets} == {"raw_subprocess"}


def test_python_process_gateway_requires_complete_static_contract() -> None:
    tree = ast.parse(
        textwrap.dedent(
            """
            from core.runtime.subprocess_gateway import (
                PythonProcessSpec,
                get_subprocess_gateway,
            )

            def perform(target) -> None:
                get_subprocess_gateway().spawn_python_process(
                    PythonProcessSpec(target=target, source="test")
                )
            """
        )
    )
    visitor = _SubprocessDeclarationVisitor(relative_path="core/synthetic.py")
    visitor.visit(tree)

    assert len(visitor.violations) == 1
    problem = visitor.violations[0].problem
    assert "python_process_contract_incomplete" in problem
    assert "accelerator_capability" in problem
    assert "requested_privileges" in problem
    assert "start_method" in problem


def test_scanner_recognizes_async_streams_and_their_gateway_owner() -> None:
    tree = ast.parse(
        textwrap.dedent(
            """
            import asyncio
            from core.runtime.network_gateway import get_network_gateway

            async def connect() -> None:
                await asyncio.open_connection("127.0.0.1", 5025)
                await get_network_gateway().connect_stream(
                    "tcp://127.0.0.1:5025",
                    read_only=True,
                )
            """
        )
    )

    buckets = _scan_tree_scoped(tree, "core/synthetic.py")
    assert {key[0] for key in buckets} == {"network_gateway", "raw_network"}
    assert sum(buckets.values()) == 2


def test_inventory_comparison_rejects_growth_and_stale_reductions() -> None:
    base = EffectBucket(
        category="raw_subprocess",
        path="core/example.py",
        scope="<module>.run",
        callee="subprocess.run",
        count=2,
        canonical_owner=False,
    )
    increased = replace(base, count=3)
    reduced = replace(base, count=1)

    regressions, stale = compare_inventory([increased], [base])
    assert len(regressions) == 1 and "INCREASED" in regressions[0]
    assert stale == []

    regressions, stale = compare_inventory([reduced], [base])
    assert regressions == []
    assert len(stale) == 1 and "DECREASED" in stale[0]

    promoted = replace(base, canonical_owner=True)
    regressions, stale = compare_inventory([promoted], [base])
    assert regressions == []
    assert len(stale) == 1 and "OWNER_PROMOTED" in stale[0]

    regressions, stale = compare_inventory([base], [promoted])
    assert len(regressions) == 1 and "OWNER_DEMOTED" in regressions[0]
    assert stale == []


def test_scanner_does_not_count_string_replace_or_read_only_image_open() -> None:
    tree = ast.parse(
        textwrap.dedent(
            """
            from PIL import Image

            def normalize(text: str, image_path: str) -> None:
                text.replace("old", "new")
                Image.open(image_path)
            """
        )
    )

    assert _scan_tree_scoped(tree, "core/synthetic.py") == {}


def test_scanner_uses_replace_arity_to_separate_strings_from_paths() -> None:
    tree = ast.parse(
        textwrap.dedent(
            """
            from pathlib import Path

            def normalize(source: str, path: Path, target: Path) -> None:
                source.replace("_", " ")
                path.replace(target)
            """
        )
    )

    buckets = _scan_tree_scoped(tree, "core/synthetic.py")

    assert sum(buckets.values()) == 1
    assert {key[0] for key in buckets} == {"raw_file_mutation"}


def test_scanner_counts_mutating_path_open_modes() -> None:
    tree = ast.parse(
        textwrap.dedent(
            """
            import tarfile
            import wave
            from pathlib import Path

            def persist(path: Path, dynamic_mode: str) -> None:
                target = Path("artifact.txt")
                target.open("w")
                path.open(mode=dynamic_mode)
                open("read-only.txt", "r")
                tarfile.open("bundle.tar.gz", "w:gz")
                tarfile.open("bundle.tar.gz", "r:gz")
                wave.open("audio.wav", "wb")
            """
        )
    )

    buckets = _scan_tree_scoped(tree, "core/synthetic.py")
    assert sum(buckets.values()) == 4
    assert {key[0] for key in buckets} == {"raw_file_mutation"}


def test_scanner_counts_browser_and_delegated_effects() -> None:
    tree = ast.parse(
        textwrap.dedent(
            """
            import asyncio
            import webbrowser

            async def interact(page, path) -> None:
                await page.goto("https://example.test")
                await page.click("button")
                await asyncio.to_thread(webbrowser.open, "https://example.test")
                await asyncio.to_thread(path.write_text, "artifact")
                page.get("ordinary_mapping_key")
                new_page()
            """
        )
    )

    buckets = _scan_tree_scoped(tree, "core/synthetic.py")
    assert sum(buckets.values()) == 4
    assert {key[0] for key in buckets} == {"raw_browser", "raw_file_mutation"}


def test_scanner_separates_desktop_observation_from_mutation() -> None:
    tree = ast.parse(
        textwrap.dedent(
            """
            import pyautogui

            def observe_and_act() -> None:
                pyautogui.size()
                pyautogui.position()
                pyautogui.screenshot()
                pyautogui.moveTo(10, 10)
                pyautogui.click()
            """
        )
    )

    buckets = _scan_tree_scoped(tree, "core/synthetic.py")
    assert sum(buckets.values()) == 2
    assert {key[0] for key in buckets} == {"raw_desktop"}


def test_flight_recorder_uses_the_canonical_file_gateway_escape_hatch() -> None:
    assert _canonical_owner("raw_file_mutation", "core/runtime/flight_recorder.py") is False
    assert _canonical_owner("file_write_gateway", "core/runtime/flight_recorder.py") is True


def test_stable_file_reader_is_a_canonical_primitive_owner() -> None:
    assert _canonical_owner("raw_file_mutation", "core/runtime/file_read_gateway.py") is True


def test_storage_migrations_are_canonical_file_gateway_owners() -> None:
    migrated = {
        "core/agency/self_repair_backlog.py",
        "core/brain/llm/latent_cortex/persistence.py",
        "core/memory/recall_observations.py",
        "core/runtime/flight_recorder.py",
        "core/security/tls_local.py",
        "core/self_improvement/program_dna.py",
        "infrastructure/rollback.py",
    }

    assert all(_canonical_owner("file_write_gateway", path) for path in migrated)
    assert not any(_canonical_owner("raw_file_mutation", path) for path in migrated)
