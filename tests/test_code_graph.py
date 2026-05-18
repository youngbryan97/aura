"""Tests for the code graph (AST-based symbol index)."""
import asyncio
import os

import pytest


@pytest.fixture
def sample_codebase(tmp_path):
    """Create a minimal Python codebase for testing."""
    (tmp_path / "module_a.py").write_text("""
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "woof"

    def fetch(self, item):
        self.speak()
        return item

def helper():
    d = Dog()
    d.speak()
    d.fetch("ball")
""")
    (tmp_path / "module_b.py").write_text("""
from module_a import Dog, helper

def main():
    helper()
    d = Dog()
    d.speak()

class Puppy(Dog):
    def play(self):
        self.fetch("toy")
""")
    return tmp_path


@pytest.fixture
def graph(sample_codebase):
    from core.introspection.code_graph import CodeGraph
    db_path = str(sample_codebase / "test_graph.db")
    g = CodeGraph(root=sample_codebase, db_path=db_path)
    return g


@pytest.mark.asyncio
async def test_build_indexes_files(graph, sample_codebase):
    stats = await graph.build(incremental=False)
    assert stats["files"] == 2
    assert stats["symbols"] > 0
    assert stats["relationships"] > 0
    assert stats["errors"] == 0


@pytest.mark.asyncio
async def test_search_symbols(graph):
    await graph.build(incremental=False)
    results = graph.search_symbols("Dog")
    assert any(r["name"] == "Dog" for r in results)


@pytest.mark.asyncio
async def test_search_by_type(graph):
    await graph.build(incremental=False)
    classes = graph.search_symbols("", sym_type="class")
    assert all(r["type"] == "class" for r in classes)
    assert any(r["name"] == "Animal" for r in classes)


@pytest.mark.asyncio
async def test_who_calls(graph):
    await graph.build(incremental=False)
    callers = graph.who_calls("speak")
    assert len(callers) > 0
    caller_names = [c["source_name"] for c in callers]
    assert "fetch" in caller_names or "main" in caller_names or "helper" in caller_names


@pytest.mark.asyncio
async def test_what_calls(graph):
    await graph.build(incremental=False)
    called = graph.what_calls("helper")
    called_names = [c["target_name"] for c in called]
    assert "Dog" in called_names or "speak" in called_names or "fetch" in called_names


@pytest.mark.asyncio
async def test_who_inherits(graph):
    await graph.build(incremental=False)
    children = graph.who_inherits("Animal")
    assert any(c["source_name"] == "Dog" for c in children)


@pytest.mark.asyncio
async def test_who_inherits_chain(graph):
    await graph.build(incremental=False)
    children = graph.who_inherits("Dog")
    assert any(c["source_name"] == "Puppy" for c in children)


@pytest.mark.asyncio
async def test_what_does(graph):
    await graph.build(incremental=False)
    info = graph.what_does("fetch")
    assert len(info) > 0
    assert info[0]["type"] in ("function", "method")
    assert "item" in info[0]["signature"]


@pytest.mark.asyncio
async def test_dependencies_of(graph):
    await graph.build(incremental=False)
    deps = graph.dependencies_of("module_b.py")
    dep_names = [d["target_name"] for d in deps]
    assert any("Dog" in n for n in dep_names)


@pytest.mark.asyncio
async def test_hotspots(graph):
    await graph.build(incremental=False)
    hot = graph.hotspots(limit=5)
    assert len(hot) > 0
    # speak should be one of the most-called
    names = [h["target_name"] for h in hot]
    assert "speak" in names


@pytest.mark.asyncio
async def test_get_stats(graph):
    await graph.build(incremental=False)
    stats = graph.get_stats()
    assert stats["files"] == 2
    assert stats["classes"] >= 3  # Animal, Dog, Puppy
    assert stats["functions"] >= 1  # helper, main
    assert stats["call_edges"] > 0
    assert stats["inherit_edges"] >= 2  # Dog->Animal, Puppy->Dog


@pytest.mark.asyncio
async def test_incremental_build(graph, sample_codebase):
    await graph.build(incremental=False)
    stats1 = graph.get_stats()
    assert stats1["files"] == 2

    # Second build should be fast (nothing changed)
    graph._stats = {"files": 0, "symbols": 0, "relationships": 0, "errors": 0}
    stats2 = await graph.build(incremental=True)
    assert stats2["files"] == 0  # Nothing reparsed


@pytest.mark.asyncio
async def test_incremental_detects_changes(graph, sample_codebase):
    await graph.build(incremental=False)

    # Modify a file
    await asyncio.sleep(0.1)
    (sample_codebase / "module_a.py").write_text("""
class Cat:
    def meow(self):
        return "meow"
""")
    os.utime(sample_codebase / "module_a.py")

    graph._stats = {"files": 0, "symbols": 0, "relationships": 0, "errors": 0}
    stats = await graph.build(incremental=True)
    assert stats["files"] >= 1  # module_a.py reparsed

    results = graph.search_symbols("Cat")
    assert any(r["name"] == "Cat" for r in results)


@pytest.mark.asyncio
async def test_real_codebase_build():
    """Integration test: build graph on the actual Aura codebase."""
    from core.introspection.code_graph import CodeGraph
    graph = CodeGraph()
    stats = await graph.build(incremental=False)

    assert stats["files"] > 100  # Should parse 900+ files
    assert stats["symbols"] > 1000
    assert stats["relationships"] > 1000
    assert stats["errors"] < stats["files"]  # Most files parse cleanly

    # Verify we can find real Aura symbols
    kernel = graph.what_does("AuraKernel")
    assert len(kernel) > 0
    assert kernel[0]["type"] == "class"

    callers = graph.who_calls("_commit_vault")
    assert len(callers) > 0

    hot = graph.hotspots()
    assert len(hot) > 0

    graph.close()
