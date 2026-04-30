"""Architecture-quality scorer for the Aura codebase.

Single overall_score in [0, 10000] derived from five metrics, each in [0,1]:
    modularity (0.30) - graph community structure (Louvain when networkx is present)
    acyclicity (0.30) - 1 - cycle_density of the import graph
    depth      (0.10) - normalised DAG depth (deeper layering = better)
    equality   (0.15) - evenness of module size + coupling distribution
    redundancy (0.15) - 1 - duplicate-AST-fragment rate
Weights sum to 1.0; final score is round(weighted * 10000).
"""
from __future__ import annotations

import ast
import hashlib
import logging
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger("ArchitectureQuality.Scorer")

try:  # pragma: no cover
    import networkx as _nx  # type: ignore
    _HAS_NX = True
except Exception:  # pragma: no cover
    _nx = None
    _HAS_NX = False

WEIGHTS: Dict[str, float] = {
    "modularity": 0.30, "acyclicity": 0.30,
    "depth": 0.10, "equality": 0.15, "redundancy": 0.15,
}
DEFAULT_ROOTS: Tuple[str, ...] = ("core", "skills", "training", "scripts")
DEFAULT_EXCLUDES: Tuple[str, ...] = (
    "__pycache__", ".venv", "build", "dist", ".git", "node_modules", ".pytest_cache",
)

@dataclass
class DependencyGraph:
    """Module-level import graph."""
    nodes: Set[str] = field(default_factory=set)
    edges: Set[Tuple[str, str]] = field(default_factory=set)
    sizes: Dict[str, int] = field(default_factory=dict)
    files: Dict[str, str] = field(default_factory=dict)

    def adj(self) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = defaultdict(set)
        for s, d in self.edges:
            out[s].add(d)
        for n in self.nodes:
            out.setdefault(n, set())
        return out

    def reverse_adj(self) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = defaultdict(set)
        for s, d in self.edges:
            out[d].add(s)
        for n in self.nodes:
            out.setdefault(n, set())
        return out

@dataclass
class QualityScore:
    """Snapshot of architectural quality."""
    overall_score: int
    metrics: Dict[str, float]
    module_count: int
    edge_count: int
    cycles: int
    god_files: List[str]
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

# ---------- Graph construction ----------

def _iter_python_files(root: Path, roots: Sequence[str], excludes: Sequence[str]) -> Iterable[Path]:
    for sub in roots:
        base = (root / sub).resolve()
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            sp = str(p)
            if any(ex in sp for ex in excludes):
                continue
            yield p

def _module_name(file_path: Path, repo_root: Path) -> str:
    rel = file_path.relative_to(repo_root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)

def _resolve_relative(module: str, level: int, current: str) -> str:
    if level <= 0:
        return module or ""
    parts = current.split(".")
    if level > len(parts):
        return module or ""
    base = parts[: len(parts) - level]
    if module:
        base = base + module.split(".")
    return ".".join(base)

def _imports_from_ast(tree: ast.AST, current: str) -> List[str]:
    out: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name:
                    out.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            level = node.level or 0
            mod = node.module or ""
            resolved = _resolve_relative(mod, level, current) if level else mod
            if resolved:
                out.append(resolved)
    return out

def parse_dependency_graph(
    root: Path, *,
    roots: Sequence[str] = DEFAULT_ROOTS,
    exclude: Sequence[str] = DEFAULT_EXCLUDES,
) -> DependencyGraph:
    """Walk the codebase and build a module-level import graph."""
    root = Path(root).resolve()
    g = DependencyGraph()
    files = list(_iter_python_files(root, roots, exclude))
    name_to_file: Dict[str, Path] = {}
    for f in files:
        try:
            mod = _module_name(f, root)
        except ValueError:
            continue
        if not mod:
            continue
        name_to_file[mod] = f
        g.nodes.add(mod)
        g.files[mod] = str(f)
    valid_prefixes = {n.split(".")[0] for n in g.nodes}
    for mod, path in name_to_file.items():
        try:
            src = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        try:
            tree = ast.parse(src, filename=str(path))
        except SyntaxError:
            g.sizes[mod] = src.count("\n") + 1
            continue
        g.sizes[mod] = src.count("\n") + 1
        for imp in _imports_from_ast(tree, mod):
            if not imp:
                continue
            top = imp.split(".")[0]
            if top not in valid_prefixes:
                continue
            target = imp
            while target and target not in g.nodes:
                if "." not in target:
                    target = ""
                    break
                target = target.rsplit(".", 1)[0]
            if target and target != mod:
                g.edges.add((mod, target))
    return g

# ---------- Metrics ----------

def _find_cycles(adj: Dict[str, Set[str]]) -> List[List[str]]:
    """Tarjan's SCC; returns non-trivial SCCs (cycles)."""
    idx_counter = [0]
    stack: List[str] = []
    on_stack: Set[str] = set()
    indices: Dict[str, int] = {}
    lowlinks: Dict[str, int] = {}
    sccs: List[List[str]] = []

    def strongconnect(v: str) -> None:
        indices[v] = idx_counter[0]
        lowlinks[v] = idx_counter[0]
        idx_counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in adj.get(v, ()):
            if w not in indices:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif w in on_stack:
                lowlinks[v] = min(lowlinks[v], indices[w])
        if lowlinks[v] == indices[v]:
            comp: List[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1 or any(y == v for y in adj.get(v, ())):
                sccs.append(comp)

    prev = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(max(prev, 10000))
        for v in list(adj.keys()):
            if v not in indices:
                strongconnect(v)
    finally:
        sys.setrecursionlimit(prev)
    return sccs

def _dag_depth(adj: Dict[str, Set[str]], cycle_nodes: Set[str]) -> int:
    """Longest path in the DAG-only view (cycle members ignored as roots)."""
    nodes = [n for n in adj if n not in cycle_nodes]
    memo: Dict[str, int] = {}

    def depth(v: str, seen: Set[str]) -> int:
        if v in memo:
            return memo[v]
        if v in seen:
            return 0
        seen = seen | {v}
        best = 0
        for w in adj.get(v, ()):
            if w in cycle_nodes or w == v:
                continue
            best = max(best, 1 + depth(w, seen))
        memo[v] = best
        return best

    return max((depth(n, set()) for n in nodes), default=0)

def _modularity_louvain_like(adj: Dict[str, Set[str]]) -> float:
    """Modularity in [0,1] via networkx greedy_modularity, or coarse fallback."""
    if not adj:
        return 0.0
    edges = [(s, d) for s, ds in adj.items() for d in ds]
    if not edges:
        return 0.0
    if _HAS_NX:
        try:
            G = _nx.Graph()
            G.add_nodes_from(adj.keys())
            for s, d in edges:
                if s != d:
                    G.add_edge(s, d)
            if G.number_of_edges() == 0:
                return 0.0
            from networkx.algorithms.community import (  # type: ignore
                greedy_modularity_communities, modularity,
            )
            comms = list(greedy_modularity_communities(G))
            if not comms:
                return 0.0
            q = modularity(G, comms)
            return max(0.0, min(1.0, (q + 1.0) / 2.0))
        except Exception as e:  # pragma: no cover
            logger.debug("networkx modularity failed, fallback: %s", e)

    def cluster(n: str) -> str:
        return n.split(".")[0] + "." + (n.split(".")[1] if "." in n else "")

    same = sum(1 for s, d in edges if cluster(s) == cluster(d))
    return same / len(edges)

def _equality(sizes: Dict[str, int], adj: Dict[str, Set[str]]) -> float:
    """1 - normalised gini over (LOC, fan_out) joint vector."""
    if not sizes:
        return 0.0
    vec: List[float] = []
    for n, loc in sizes.items():
        vec.append(float(loc))
        vec.append(float(len(adj.get(n, ()))))
    if not vec:
        return 0.0
    s = sum(vec)
    if s <= 0:
        return 1.0
    vec.sort()
    n = len(vec)
    cum = sum(i * v for i, v in enumerate(vec, 1))
    gini = (2 * cum) / (n * s) - (n + 1) / n
    gini = max(0.0, min(1.0, gini))
    return 1.0 - gini

def _redundancy(graph: DependencyGraph, max_files: int = 600) -> float:
    """Heuristic duplicate-AST-fragment rate; returns 1 - rate."""
    seen: Counter = Counter()
    total = 0
    for fp in list(graph.files.values())[:max_files]:
        try:
            src = Path(fp).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.body or len(node.body) < 3:
                    continue
                sig = "|".join(type(s).__name__ for s in node.body)
                h = hashlib.blake2b(sig.encode(), digest_size=12).hexdigest()
                seen[h] += 1
                total += 1
    if total == 0:
        return 1.0
    duplicates = sum(c - 1 for c in seen.values() if c > 1)
    return max(0.0, 1.0 - duplicates / total)

def compute_metrics(graph: DependencyGraph) -> Dict[str, float]:
    """Compute the five root-cause metrics in [0, 1]."""
    adj = graph.adj()
    n_nodes = len(graph.nodes)
    sccs = _find_cycles(adj)
    cycle_nodes: Set[str] = set()
    for c in sccs:
        cycle_nodes.update(c)
    cycle_density = (len(cycle_nodes) / n_nodes) if n_nodes else 0.0
    acyclicity = max(0.0, 1.0 - cycle_density)
    depth_raw = _dag_depth(adj, cycle_nodes)
    target = max(2.0, math.log2(max(2, n_nodes)))
    depth = max(0.0, min(1.0, depth_raw / max(target, 1.0)))
    return {
        "modularity": _modularity_louvain_like(adj),
        "acyclicity": acyclicity,
        "depth": depth,
        "equality": _equality(graph.sizes, adj),
        "redundancy": _redundancy(graph),
    }

def _god_files(graph: DependencyGraph, *, loc_thresh: int = 800,
               fan_ratio_thresh: float = 1.5) -> List[str]:
    """Files with > loc_thresh lines and high fan-in/out coupling."""
    radj = graph.reverse_adj()
    adj = graph.adj()
    out: List[str] = []
    for mod, loc in graph.sizes.items():
        if loc <= loc_thresh:
            continue
        fin = len(radj.get(mod, ()))
        fout = len(adj.get(mod, ()))
        coupling = max(fin, fout)
        if coupling >= 5 and (fin + fout) >= fan_ratio_thresh * 5:
            out.append(mod)
    return out

def _score_from(metrics: Dict[str, float], graph: DependencyGraph) -> QualityScore:
    import time as _t
    weighted = sum(metrics[k] * WEIGHTS[k] for k in WEIGHTS)
    overall = int(round(max(0.0, min(1.0, weighted)) * 10000))
    cycles = sum(1 for c in _find_cycles(graph.adj()) if len(c) > 1)
    return QualityScore(
        overall_score=overall, metrics=metrics,
        module_count=len(graph.nodes), edge_count=len(graph.edges),
        cycles=cycles, god_files=_god_files(graph), timestamp=_t.time(),
    )

def score_codebase(
    root: Path, *,
    roots: Sequence[str] = DEFAULT_ROOTS,
    exclude: Sequence[str] = DEFAULT_EXCLUDES,
) -> QualityScore:
    """Score the codebase rooted at *root*. overall_score is in [0, 10000]."""
    g = parse_dependency_graph(root, roots=roots, exclude=exclude)
    return _score_from(compute_metrics(g), g)

def score_from_graph(graph: DependencyGraph) -> QualityScore:
    """Convenience: compute a QualityScore from an already-built graph."""
    return _score_from(compute_metrics(graph), graph)
