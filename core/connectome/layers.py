"""core/connectome/layers.py — the wiring diagram is one layer of several.

The most useful correction to a connectome came from the smallest animal that
has one. *C. elegans* has 302 neurons and its synaptic wiring has been known
since 1986, and when Bentley and colleagues added the monoamine and neuropeptide
layers on top of it, almost none of the new connections were in the old diagram.
96% of monoamine connections exist only in the monoamine layer. 82% of the
neurons carrying a dopamine receptor receive no synapse at all from the neurons
that release dopamine. The layers have different shapes: the monoamine network
is a disassortative star, the neuropeptide network is far more clustered than
anything else, and the neurons at the centre of one are not the neurons at the
centre of another.

A wiring diagram that ignores that is not a small approximation. It is missing
most of the coordination.

Aura's synaptic layer is her calls, and this package has been measuring that
alone. It reported that 65 of 34,348 possible sense-to-action pairs have a path,
which reads as a broken animal and is instead a missing layer. She has two more:

``wired``
    One cell calls another. Directed, fast, local, and the only layer a static
    call graph can see.
``volume``
    One cell publishes on a topic and another subscribes to it. Neither knows
    the other exists. This is volume transmission: the releaser has no synapse
    onto the receiver and the receiver responds anyway.
``gap``
    Two cells touch the same mutable state — a container key, a module global.
    Undirected, because a gap junction is, and because state read by one and
    written by the other couples them in both directions whatever the code
    intended.

What the multilayer view is for is the same thing it was for in the worm:
finding the connections that exist in exactly one layer, and the cells that are
central in one layer and peripheral in another. Both are invisible from any
single layer, and both are where coordination that nobody designed is happening.
"""

from __future__ import annotations

import ast
import logging
import os
import statistics
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .topology import DiGraphView, degree_preserving_rewire, rich_club
from .types import ConnectomeSnapshot, EdgeKind, stable_id

logger = logging.getLogger("Aura.Connectome.Layers")

__all__ = [
    "Layer",
    "ChannelUse",
    "MultilayerConnectome",
    "extract_layers",
    "multilink_census",
    "layer_report",
    "PUBLISH_NAMES",
    "SUBSCRIBE_NAMES",
    "IO_READ_NAMES",
    "IO_WRITE_NAMES",
    "IPC_NAMES",
    "STATE_READ_NAMES",
    "STATE_WRITE_NAMES",
]


class Layer(StrEnum):
    """The ways one cell reaches another.

    ``WIRED`` is a call. ``VOLUME`` is a topic neither end knows the other is
    on. ``GAP`` is shared mutable state in the process. ``IO`` is a durable
    store — a file, a table — where the write and the read can be days apart and
    in different processes. ``IPC`` crosses the process boundary itself.
    """

    WIRED = "wired"
    VOLUME = "volume"
    GAP = "gap"
    IO = "io"
    IPC = "ipc"


#: Calls that put something on a topic without knowing who reads it.
PUBLISH_NAMES: frozenset[str] = frozenset(
    {
        "publish",
        "publish_async",
        "publish_event",
        "emit",
        "emit_event",
        "emit_signal",
        "broadcast",
        "dispatch",
        "fire",
        "post_event",
        "send_signal",
    }
)

#: Calls that register interest in a topic without knowing who writes it.
SUBSCRIBE_NAMES: frozenset[str] = frozenset(
    {
        "subscribe",
        "subscribe_async",
        "on_event",
        "add_listener",
        "add_subscriber",
        "register_handler",
        "register_listener",
        "listen",
        "observe",
        "watch",
    }
)

#: Calls that read shared state by key.
STATE_READ_NAMES: frozenset[str] = frozenset(
    {
        "get",
        "get_service",
        "resolve",
        "require",
        "try_get",
        "lookup",
        "fetch_service",
    }
)

#: Calls that write shared state by key.
STATE_WRITE_NAMES: frozenset[str] = frozenset(
    {
        "set",
        "set_service",
        "register",
        "register_instance",
        "register_factory",
        "provide",
        "bind",
    }
)

#: Calls that write to a durable store. The store outlives the process, so a
#: writer and a reader here can be days and a restart apart, which is a kind of
#: coupling no in-process analysis can see at all.
IO_WRITE_NAMES: frozenset[str] = frozenset(
    {
        "write_text",
        "write_bytes",
        "write_json",
        "atomic_write_text",
        "atomic_write_bytes",
        "atomic_write_json",
        "async_atomic_write_text",
        "async_atomic_write_bytes",
        "async_atomic_write_json",
        "append_text",
        "atomic_append_text",
        "dump",
        "savez",
        "savez_compressed",
        "to_csv",
        "to_parquet",
    }
)
IO_READ_NAMES: frozenset[str] = frozenset(
    {
        "read_text",
        "read_bytes",
        "read_json",
        "load",
        "loads_file",
        "read_csv",
        "read_parquet",
        "drain_text",
    }
)

#: Calls that cross the process boundary. A subprocess command, a socket
#: address, a shared-memory name and a queue name are all channels two cells can
#: meet on without either being in the other's address space.
IPC_NAMES: frozenset[str] = frozenset(
    {
        "Popen",
        "run",
        "check_call",
        "check_output",
        "spawn",
        "connect",
        "bind",
        "SharedMemory",
        "shared_memory",
        "sendto",
        "recvfrom",
    }
)

#: Receivers that make a bare ``get``/``set`` a shared-state access rather than
#: a dictionary lookup. Without this the gap layer would join every cell that
#: ever called ``dict.get`` to every other one.
_STATE_RECEIVERS: frozenset[str] = frozenset(
    {
        "container",
        "services",
        "service_container",
        "registry",
        "ServiceContainer",
        "store",
        "state",
        "cache",
        "_container",
        "_registry",
    }
)


@dataclass(frozen=True)
class ChannelUse:
    """One cell touching one named channel, in one direction."""

    cell: str
    channel: str
    layer: Layer
    direction: str

    def key(self) -> tuple[str, str, str]:
        return (self.channel, self.direction, self.cell)


def _channel_key(node: ast.expr | None) -> str:
    """Render an argument to a stable channel name, or nothing.

    A topic is a machine key, not a sentence, so a string with a space in it is
    a message being shown to somebody and not a channel. ``EventType.THOUGHT``
    keeps both halves because two enums can share a member name. A lowercase
    bare name is a variable or a callback and is refused; a constant imported
    into two modules is the same channel in both, which is exactly the coupling
    this layer exists to find.
    """
    if node is None:
        return ""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        text = node.value.strip()
        if not text or " " in text or len(text) > 120:
            return ""
        return text
    if isinstance(node, ast.Name):
        return node.id if node.id.isupper() else ""
    if isinstance(node, ast.Attribute):
        base = node.value
        if isinstance(base, ast.Name):
            if base.id in {"self", "cls"}:
                return ""
            return f"{base.id}.{node.attr}"
        return ""
    return ""


def _boundary_key(node: ast.Call) -> str:
    """The command, address or name a cross-process call meets another cell at.

    A subprocess call names its binary as the first element of a list, a socket
    names an address, and shared memory names a key. All three are the first
    string literal that appears, so that is what is taken — and a call with no
    literal at all names a boundary nobody else can be matched to, so it is left
    out rather than joined to everything.
    """
    for argument in [*node.args, *(kw.value for kw in node.keywords)]:
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            text = argument.value.strip()
            if text and " " not in text:
                return text
        if isinstance(argument, (ast.List, ast.Tuple)) and argument.elts:
            head = argument.elts[0]
            if isinstance(head, ast.Constant) and isinstance(head.value, str):
                text = head.value.strip()
                if text:
                    return text.rsplit("/", 1)[-1]
            if isinstance(head, ast.Name):
                return head.id
    return ""


def _first_channel(node: ast.Call) -> str:
    """The first argument that looks like a topic rather than like a payload.

    Subscription APIs take the callback in some position and the topic in
    another, and which is which differs between them. Scanning for the first
    topic-shaped argument reads both without a table of signatures per library.
    """
    for argument in node.args:
        channel = _channel_key(argument)
        if channel:
            return channel
    for keyword in node.keywords:
        if keyword.arg in {"topic", "event_type", "event", "channel", "name", "key", "signal"}:
            channel = _channel_key(keyword.value)
            if channel:
                return channel
    return ""


#: Extensions that name a durable store rather than a Python attribute.
_STORE_SUFFIXES: frozenset[str] = frozenset(
    {
        "json", "jsonl", "txt", "md", "csv", "tsv", "parquet", "npz", "npy",
        "db", "sqlite", "sqlite3", "log", "yaml", "yml", "toml", "pkl", "bin",
        "pt", "safetensors", "gguf", "html", "xml", "pdf",
    }
)


def _store_shaped(channel: str) -> bool:
    """Whether a name is a durable store rather than a value.

    A store name carries a path separator or a file extension, or is a constant
    written in capitals. Without this the layer would join every cell that ever
    called ``json.dump`` on anything, which is a bus and not a connection.
    """
    if not channel:
        return False
    if "/" in channel or channel.startswith("."):
        return True
    if "." in channel and channel.rsplit(".", 1)[-1].lower() in _STORE_SUFFIXES:
        return True
    return channel.isupper() and len(channel) > 3


def _store_key(node: ast.Call) -> str:
    """Find the store a call names, wherever in its arguments it appears.

    Paths are built, not written down: ``(out / "report.json").write_text(...)``
    names its store in the receiver, and ``gateway.write_text(out / "report.json")``
    names it in the first argument. Scanning the receiver as well as every
    argument reads both, where taking the first argument alone reads a local
    variable and finds nothing. The filename is the channel rather than the whole
    path, because two cells meeting at the same file usually build the directory
    separately.
    """
    receiver = [node.func.value] if isinstance(node.func, ast.Attribute) else []
    for argument in [*receiver, *node.args, *(kw.value for kw in node.keywords)]:
        for inner in ast.walk(argument):
            if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                text = inner.value.strip()
                if _store_shaped(text) and " " not in text:
                    return text.rsplit("/", 1)[-1]
            elif isinstance(inner, ast.Name) and _store_shaped(inner.id):
                return inner.id
            elif isinstance(inner, ast.Attribute) and _store_shaped(inner.attr):
                return inner.attr
    return ""


def _receiver_name(func: ast.expr) -> str:
    if isinstance(func, ast.Attribute):
        base = func.value
        if isinstance(base, ast.Name):
            return base.id
        if isinstance(base, ast.Attribute):
            return base.attr
        if isinstance(base, ast.Call):
            name = base.func
            if isinstance(name, ast.Name):
                return name.id
            if isinstance(name, ast.Attribute):
                return name.attr
    return ""


class _ChannelVisitor(ast.NodeVisitor):
    """Collect every channel a function touches, and how."""

    def __init__(self) -> None:
        self.uses: list[tuple[str, Layer, str]] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else (
            func.id if isinstance(func, ast.Name) else ""
        )
        if name in IO_WRITE_NAMES or name in IO_READ_NAMES:
            store = _store_key(node)
            if _store_shaped(store):
                self.uses.append(
                    (store, Layer.IO, "write" if name in IO_WRITE_NAMES else "read")
                )
            self.generic_visit(node)
            return
        if name in IPC_NAMES:
            boundary = _boundary_key(node)
            if boundary:
                self.uses.append((boundary, Layer.IPC, "write"))
            self.generic_visit(node)
            return
        if name:
            channel = _first_channel(node)
            if channel:
                receiver = _receiver_name(func)
                if name in PUBLISH_NAMES:
                    self.uses.append((channel, Layer.VOLUME, "out"))
                elif name in SUBSCRIBE_NAMES:
                    self.uses.append((channel, Layer.VOLUME, "in"))
                elif name in STATE_WRITE_NAMES and receiver in _STATE_RECEIVERS:
                    self.uses.append((channel, Layer.GAP, "write"))
                elif name in STATE_READ_NAMES and receiver in _STATE_RECEIVERS:
                    self.uses.append((channel, Layer.GAP, "read"))
                elif name in IO_WRITE_NAMES:
                    store = _store_key(node) or channel
                    if _store_shaped(store):
                        self.uses.append((store, Layer.IO, "write"))
                elif name in IO_READ_NAMES:
                    store = _store_key(node) or channel
                    if _store_shaped(store):
                        self.uses.append((store, Layer.IO, "read"))

        self.generic_visit(node)


@dataclass
class MultilayerConnectome:
    """Every graph over the same cells."""

    wired: dict[tuple[str, str], int]
    volume: dict[tuple[str, str], int]
    gap: dict[tuple[str, str], int]
    io: dict[tuple[str, str], int] = field(default_factory=dict)
    ipc: dict[tuple[str, str], int] = field(default_factory=dict)
    channels: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    unresolved_channels: int = 0

    def layer(self, layer: Layer) -> dict[tuple[str, str], int]:
        return {
            Layer.WIRED: self.wired,
            Layer.VOLUME: self.volume,
            Layer.GAP: self.gap,
            Layer.IO: self.io,
            Layer.IPC: self.ipc,
        }[layer]

    def combined(self) -> dict[tuple[str, str], int]:
        merged: dict[tuple[str, str], int] = dict(self.wired)
        for source in (self.volume, self.gap, self.io, self.ipc):
            for pair, weight in source.items():
                merged[pair] = merged.get(pair, 0) + weight
        return merged

    def unique_fraction(self, layer: Layer) -> float:
        """Share of this layer's pairs that exist in no other layer.

        The worm's monoamine layer scores 0.96 on this. A layer scoring near
        zero is a redrawing of the wiring diagram and adds nothing; a layer
        scoring near one is a set of connections the wiring diagram cannot see.
        """
        target = self.layer(layer)
        if not target:
            return 0.0
        others: set[tuple[str, str]] = set()
        for other in Layer:
            if other is layer:
                continue
            others |= set(self.layer(other))
            others |= {(post, pre) for pre, post in self.layer(other)}
        unique = sum(1 for pair in target if pair not in others)
        return unique / len(target)

    def summary(self) -> dict[str, Any]:
        return {
            "wired_pairs": len(self.wired),
            "volume_pairs": len(self.volume),
            "gap_pairs": len(self.gap),
            "io_pairs": len(self.io),
            "ipc_pairs": len(self.ipc),
            "combined_pairs": len(self.combined()),
            "channels": len(self.channels),
            "unresolved_channels": self.unresolved_channels,
            "unique_fraction": {
                str(layer): round(self.unique_fraction(layer), 4) for layer in Layer
            },
        }


def extract_layers(
    snapshot: ConnectomeSnapshot,
    repo: str | Path | None = None,
    *,
    roots: Sequence[str] = ("core", "interface", "skills", "security", "llm", "executors"),
    max_channel_fanout: int = 400,
) -> MultilayerConnectome:
    """Walk the source again for the two layers a call graph cannot see.

    ``max_channel_fanout`` matters. A channel touched by six hundred cells would
    contribute a third of a million pairs on its own and would say nothing: it
    is a bus, not a connection. Channels above the bound are counted and left
    out, and the count is reported so the omission is visible.
    """
    root = Path(repo) if repo is not None else Path(snapshot.source or ".")
    wired = {
        (conn.pre, conn.post): conn.contacts
        for conn in snapshot.connections.values()
        if conn.kind is EdgeKind.DRIVE
    }
    by_name = {unit.name: uid for uid, unit in snapshot.units.items()}

    channels: dict[str, dict[str, set[str]]] = {}
    skipped = 0
    for path in _iter_files(root, roots):
        try:
            tree = ast.parse(path.read_bytes().decode("utf-8", "replace"), filename=str(path))
        except (SyntaxError, OSError):
            continue
        module = _module_name(path, root)
        for qualname, node in _iter_functions(tree):
            uid = by_name.get(f"{module}:{qualname}")
            if uid is None:
                uid = stable_id(module, qualname)
                if uid not in snapshot.units:
                    continue
            visitor = _ChannelVisitor()
            for statement in node.body:
                visitor.visit(statement)
            for channel, layer, direction in visitor.uses:
                entry = channels.setdefault(
                    f"{layer}:{channel}",
                    {"out": set(), "in": set(), "read": set(), "write": set()},
                )
                entry[direction].add(uid)

    volume: dict[tuple[str, str], int] = {}
    gap: dict[tuple[str, str], int] = {}
    io: dict[tuple[str, str], int] = {}
    ipc: dict[tuple[str, str], int] = {}
    for key, entry in channels.items():
        layer_name, _, _channel = key.partition(":")
        if layer_name == str(Layer.VOLUME):
            writers, readers = entry["out"], entry["in"]
            if len(writers) * len(readers) > max_channel_fanout:
                skipped += 1
                continue
            for pre in writers:
                for post in readers:
                    if pre != post:
                        pair = (pre, post)
                        volume[pair] = volume.get(pair, 0) + 1
        elif layer_name == str(Layer.IO):
            # A durable store is directed: the writer puts something there and
            # the reader finds it, possibly in another process on another day.
            writers, readers = entry["write"], entry["read"]
            if len(writers) * len(readers) > max_channel_fanout:
                skipped += 1
                continue
            for pre in writers:
                for post in readers:
                    if pre != post:
                        pair = (pre, post)
                        io[pair] = io.get(pair, 0) + 1
        elif layer_name == str(Layer.IPC):
            # Everything that names the same boundary meets at it, and which
            # side is writing is not something a static read can tell.
            touchers = entry["write"] | entry["read"]
            if len(touchers) * len(touchers) > max_channel_fanout:
                skipped += 1
                continue
            ordered = sorted(touchers)
            for index, pre in enumerate(ordered):
                for post in ordered[index + 1 :]:
                    pair = (pre, post)
                    ipc[pair] = ipc.get(pair, 0) + 1
        else:
            writers, readers = entry["write"], entry["read"]
            touchers = writers | readers
            if len(touchers) * len(touchers) > max_channel_fanout:
                skipped += 1
                continue
            ordered = sorted(touchers)
            for index, pre in enumerate(ordered):
                for post in ordered[index + 1 :]:
                    pair = (pre, post)
                    gap[pair] = gap.get(pair, 0) + 1

    return MultilayerConnectome(
        wired=wired,
        volume=volume,
        gap=gap,
        io=io,
        ipc=ipc,
        channels=channels,
        unresolved_channels=skipped,
    )


def _iter_files(root: Path, roots: Sequence[str]) -> Iterator[Path]:
    skip = {"__pycache__", ".git", ".venv", "node_modules", ".claude", "artifacts", "data"}
    for name in roots:
        base = root / name
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in skip)
            for filename in sorted(filenames):
                if filename.endswith(".py") and not filename.startswith("test_"):
                    yield Path(dirpath) / filename


def _module_name(path: Path, repo: Path) -> str:
    try:
        rel = path.relative_to(repo).with_suffix("")
    except ValueError:
        return ""
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _iter_functions(
    tree: ast.Module, prefix: str = ""
) -> Iterator[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    for child in ast.iter_child_nodes(tree):
        if isinstance(child, ast.ClassDef):
            yield from _iter_functions(child, prefix=f"{prefix}{child.name}.")
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield f"{prefix}{child.name}", child


#: The worm paper calls the combinations multilink motifs, and its point is that
#: some of them are over-represented: a monoamine link running one way over a
#: reciprocal pair of synapses is a different circuit from either link alone.
#: With five layers there are thirty-one combinations, so the census reports the
#: set of layers joining each pair rather than a name per combination.


def multilink_census(multilayer: MultilayerConnectome) -> dict[str, int]:
    """How many pairs are joined by each combination of layers.

    Undirected layers are counted in both directions before the sets are taken,
    because a pair joined by a call one way and by shared state at all is joined
    by both, and treating the state edge as one-way would split that pair into
    two half-findings.
    """
    membership: dict[tuple[str, str], set[str]] = {}
    for layer in Layer:
        pairs = multilayer.layer(layer)
        undirected = layer in {Layer.GAP, Layer.IPC}
        for pre, post in pairs:
            membership.setdefault((pre, post), set()).add(str(layer))
            if undirected:
                membership.setdefault((post, pre), set()).add(str(layer))
    counts: Counter[str] = Counter()
    for layers in membership.values():
        counts["+".join(sorted(layers))] += 1
    return dict(counts.most_common())


def _as_graph(pairs: Mapping[tuple[str, str], int], nodes: Iterable[str]) -> DiGraphView:
    out: dict[str, set[str]] = {node: set() for node in nodes}
    inbound: dict[str, set[str]] = {node: set() for node in nodes}
    weights: dict[tuple[str, str], int] = {}
    for (pre, post), weight in pairs.items():
        if pre not in out or post not in out or pre == post:
            continue
        out[pre].add(post)
        inbound[post].add(pre)
        weights[(pre, post)] = int(weight)
    return DiGraphView(nodes=tuple(sorted(out)), out=out, inbound=inbound, weights=weights)


def layer_report(
    snapshot: ConnectomeSnapshot,
    multilayer: MultilayerConnectome,
    *,
    top_hubs: int = 12,
) -> dict[str, Any]:
    """Per-layer shape, the layer-unique fractions, and whose hubs are whose.

    The worm's finding that its monoamine rich club is made of different cells
    from its wired rich club is the one worth reproducing, because a cell that
    is central in a layer nobody was looking at is a single point of failure
    nobody has noticed.
    """
    nodes = list(snapshot.units)
    report: dict[str, Any] = {
        "summary": multilayer.summary(),
        "multilink": multilink_census(multilayer),
        "layers": {},
    }
    hubs: dict[str, list[str]] = {}
    for layer in Layer:
        pairs = multilayer.layer(layer)
        graph = _as_graph(pairs, nodes)
        if graph.m == 0:
            report["layers"][str(layer)] = {"pairs": 0}
            continue
        degrees = {node: graph.degree(node) for node in graph.nodes}
        ranked = sorted(degrees.items(), key=lambda item: (-item[1], item[0]))[:top_hubs]
        hubs[str(layer)] = [uid for uid, _ in ranked]
        observed_rich = rich_club(graph)
        null_rich: dict[int, list[float]] = {}
        for index in range(3):
            rewired = degree_preserving_rewire(graph, swaps_per_edge=3, seed=index)
            for cut, value in rich_club(rewired, degrees=sorted(observed_rich)).items():
                null_rich.setdefault(cut, []).append(value)
        report["layers"][str(layer)] = {
            "pairs": graph.m,
            "cells_touched": sum(1 for node in graph.nodes if degrees[node] > 0),
            "mean_degree": round(statistics.fmean(degrees.values()), 4) if degrees else 0.0,
            "unique_fraction": round(multilayer.unique_fraction(layer), 4),
            "rich_club_normalised": {
                str(cut): round(observed_rich[cut] / statistics.fmean(null_rich[cut]), 4)
                for cut in observed_rich
                if null_rich.get(cut) and statistics.fmean(null_rich[cut]) > 0
            },
            "top_hubs": [
                {
                    "cell": snapshot.units[uid].name if uid in snapshot.units else uid,
                    "degree": degrees[uid],
                }
                for uid, _ in ranked
            ],
        }
    overlaps: dict[str, float] = {}
    for left in hubs:
        for right in hubs:
            if left < right:
                shared = set(hubs[left]) & set(hubs[right])
                overlaps[f"{left}|{right}"] = round(shared.__len__() / max(1, top_hubs), 4)
    report["hub_overlap"] = overlaps
    return report
