from __future__ import annotations

from pathlib import Path

from core.reproducibility import BuildGraph, BuildStep, ContentAddressedStore, EnvPolicy, ProofSubstrate, SourceFile, ToolSpec


def test_cas_latest_graph_ref_is_atomic_and_streams_large_blobs(tmp_path):
    cas = ContentAddressedStore(tmp_path / "cas", max_memory_blob_bytes=4)
    digest = cas.put(b"hello world")
    cas.put_ref_atomic("latest_graph", digest)
    assert cas.get_ref("latest_graph") == digest
    assert b"".join(cas.stream(digest, chunk_size=2)) == b"hello world"
    try:
        cas.get(digest)
    except MemoryError:
        pass
    else:
        raise AssertionError("large CAS get should force streaming")


def test_dirty_detection_checks_sources_tools_artifacts_and_filtered_env(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    src = root / "main.py"
    src.write_text("print('one')\n", encoding="utf-8")
    tool = tmp_path / "tool"
    tool.write_text("#!/bin/sh\necho v1\n", encoding="utf-8")
    cas = ContentAddressedStore(tmp_path / "cas")

    graph = BuildGraph(root=str(root))
    graph.sources["main.py"] = SourceFile.from_path(root, src)
    graph.tools["tool"] = ToolSpec("tool", str(tool), "v1", ToolSpec("tool", str(tool)).refresh_hash())
    policy = EnvPolicy(keys=("PATH",), prefixes=("AURA_",))
    step = BuildStep(
        step_id="step",
        runner="custom",
        command=("tool", "main.py"),
        sources=("main.py",),
        tools=("tool",),
        artifact_hash=cas.put(b"artifact"),
        env_policy=policy,
        env_hash=policy.hash({"PATH": "/bin", "IGNORED": "a", "AURA_MODE": "x"}),
    )
    graph.steps["step"] = step

    assert graph.dirty_nodes(cas, env={"PATH": "/bin", "IGNORED": "changed", "AURA_MODE": "x"}) == []

    tool.write_text("#!/bin/sh\necho v2\n", encoding="utf-8")
    assert any(reason.reason == "tool_hash_changed" for reason in graph.dirty_nodes(cas, env={"PATH": "/bin", "AURA_MODE": "x"}))

    graph.tools["tool"].path_hash = graph.tools["tool"].refresh_hash()
    src.write_text("print('two')\n", encoding="utf-8")
    assert any(reason.reason == "source_hash_changed" for reason in graph.dirty_nodes(cas, env={"PATH": "/bin", "AURA_MODE": "x"}))

    graph.sources["main.py"] = SourceFile.from_path(root, src)
    assert any(reason.reason == "runner_environment_changed" for reason in graph.dirty_nodes(cas, env={"PATH": "/other", "AURA_MODE": "x"}))


def test_build_graph_uses_fine_grained_runner_inputs(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "main.py").write_text("x=1\n", encoding="utf-8")
    (root / "README.md").write_text("docs\n", encoding="utf-8")
    substrate = ProofSubstrate(root, cas_dir=tmp_path / "cas")
    graph = substrate.build_graph(
        runner_specs=[
            {
                "runner": "python",
                "command": ("python", "-m", "pytest"),
                "include": ("src/**/*.py", "tests/**/*.py"),
                "tools": (),
                "env_policy": EnvPolicy(keys=("PATH",)),
            }
        ]
    )
    step = next(iter(graph.steps.values()))
    assert "src/main.py" in step.sources
    assert "README.md" not in step.sources
    digest = substrate.store_graph(graph)
    assert substrate.cas.get_ref("latest_graph") == digest
    assert substrate.load_or_create_graph().steps
