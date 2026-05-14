"""Deterministic proof substrate for self-improvement and build receipts.

This module closes the reproducibility issues called out in the upgrade brief:

* ``latest_graph`` is updated atomically as a mutable pointer into an immutable
  content-addressed store.
* dirty detection checks source hashes, artifact availability, tool binary
  hashes, and runner-specific filtered environment hashes.
* graph construction can use fine-grained source filters instead of one
  monolithic "all files feed every step" dependency.
* CAS reads verify integrity and support streaming to avoid loading large
  artifacts just to chunk them.
* optional garbage collection keeps old unreferenced blobs bounded.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from core.runtime.atomic_writer import atomic_write_bytes

try:
    import blake3  # type: ignore
except ImportError:  # pragma: no cover - optional acceleration
    blake3 = None


def _hash_bytes(data: bytes) -> str:
    if blake3 is not None:
        return blake3.blake3(data).hexdigest()
    return hashlib.blake2b(data, digest_size=32).hexdigest()


def _hash_file(path: Path) -> str:
    if blake3 is not None:
        hasher = blake3.blake3()
    else:
        hasher = hashlib.blake2b(digest_size=32)
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


@dataclass(frozen=True)
class EnvPolicy:
    """Runner-specific environment hashing and execution policy."""

    mode: str = "whitelist"
    keys: tuple[str, ...] = ("PATH", "HOME")
    prefixes: tuple[str, ...] = ()

    def filter(self, env: Mapping[str, str] | None = None) -> dict[str, str]:
        source = dict(env if env is not None else os.environ)
        if self.mode == "all":
            return {k: source[k] for k in sorted(source)}
        if self.mode == "none":
            return {}
        allowed: dict[str, str] = {}
        for key in self.keys:
            if key in source:
                allowed[key] = source[key]
        for key in sorted(source):
            if any(key.startswith(prefix) for prefix in self.prefixes):
                allowed[key] = source[key]
        return dict(sorted(allowed.items()))

    def hash(self, env: Mapping[str, str] | None = None) -> str:
        return _hash_bytes(_canonical_json(self.filter(env)))


CARGO_ENV = EnvPolicy(
    keys=("PATH", "HOME"),
    prefixes=("RUSTUP_", "CARGO_"),
)
MAKE_ENV = EnvPolicy(keys=("PATH", "HOME", "CC", "CFLAGS", "LDFLAGS"), prefixes=("MAKE",))
CUSTOM_ENV = EnvPolicy(mode="all")


@dataclass
class ToolSpec:
    name: str
    path: str
    version: str = ""
    path_hash: str = ""

    @classmethod
    def discover(cls, name: str) -> "ToolSpec | None":
        path = shutil.which(name)
        if not path:
            return None
        path_hash = _hash_file(Path(path))
        version = f"content:{path_hash[:16]}"
        return cls(name=name, path=path, version=version, path_hash=path_hash)

    def refresh_hash(self) -> str:
        return _hash_file(Path(self.path))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceFile:
    path: str
    hash: str

    @classmethod
    def from_path(cls, root: Path, path: Path) -> "SourceFile":
        return cls(str(path.relative_to(root)), _hash_file(path))


@dataclass
class BuildStep:
    step_id: str
    runner: str
    command: tuple[str, ...]
    sources: tuple[str, ...]
    tools: tuple[str, ...]
    artifact_hash: str = ""
    env_policy: EnvPolicy = field(default_factory=lambda: CARGO_ENV)
    env_hash: str = ""

    def identity_payload(self) -> dict[str, Any]:
        return {
            "runner": self.runner,
            "command": list(self.command),
            "sources": list(self.sources),
            "tools": list(self.tools),
            "env_policy": asdict(self.env_policy),
            "env_hash": self.env_hash,
        }


@dataclass
class DirtyReason:
    step_id: str
    reason: str
    detail: str


@dataclass
class BuildGraph:
    root: str
    sources: dict[str, SourceFile] = field(default_factory=dict)
    tools: dict[str, ToolSpec] = field(default_factory=dict)
    steps: dict[str, BuildStep] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "sources": {k: asdict(v) for k, v in self.sources.items()},
            "tools": {k: v.to_dict() for k, v in self.tools.items()},
            "steps": {
                k: {
                    **asdict(v),
                    "env_policy": asdict(v.env_policy),
                }
                for k, v in self.steps.items()
            },
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BuildGraph":
        graph = cls(root=str(raw["root"]), created_at=float(raw.get("created_at", time.time())))
        graph.sources = {k: SourceFile(**v) for k, v in raw.get("sources", {}).items()}
        graph.tools = {k: ToolSpec(**v) for k, v in raw.get("tools", {}).items()}
        graph.steps = {}
        for key, value in raw.get("steps", {}).items():
            policy_raw = value.get("env_policy", {})
            policy = EnvPolicy(
                mode=policy_raw.get("mode", "whitelist"),
                keys=tuple(policy_raw.get("keys", ())),
                prefixes=tuple(policy_raw.get("prefixes", ())),
            )
            graph.steps[key] = BuildStep(
                step_id=value["step_id"],
                runner=value["runner"],
                command=tuple(value.get("command", ())),
                sources=tuple(value.get("sources", ())),
                tools=tuple(value.get("tools", ())),
                artifact_hash=value.get("artifact_hash", ""),
                env_policy=policy,
                env_hash=value.get("env_hash", ""),
            )
        return graph

    def dirty_nodes(
        self,
        cas: "ContentAddressedStore",
        *,
        env: Mapping[str, str] | None = None,
    ) -> list[DirtyReason]:
        root = Path(self.root)
        dirty: list[DirtyReason] = []
        for step in self.steps.values():
            if step.artifact_hash and not cas.exists(step.artifact_hash):
                dirty.append(DirtyReason(step.step_id, "missing_artifact", step.artifact_hash))
            elif not step.artifact_hash:
                dirty.append(DirtyReason(step.step_id, "artifact_not_recorded", ""))

            for source_name in step.sources:
                source = self.sources.get(source_name)
                path = root / source_name
                if source is None:
                    dirty.append(DirtyReason(step.step_id, "source_missing_from_graph", source_name))
                elif not path.exists():
                    dirty.append(DirtyReason(step.step_id, "source_missing_on_disk", source_name))
                else:
                    current = _hash_file(path)
                    if current != source.hash:
                        dirty.append(DirtyReason(step.step_id, "source_hash_changed", source_name))

            for tool_name in step.tools:
                tool = self.tools.get(tool_name)
                if tool is None:
                    dirty.append(DirtyReason(step.step_id, "tool_missing_from_graph", tool_name))
                    continue
                path = Path(tool.path)
                if not path.exists():
                    dirty.append(DirtyReason(step.step_id, "tool_missing_on_disk", tool_name))
                    continue
                current_hash = _hash_file(path)
                if current_hash != tool.path_hash:
                    dirty.append(DirtyReason(step.step_id, "tool_hash_changed", f"{tool_name}:{tool.path}"))

            current_env = step.env_policy.hash(env)
            if current_env != step.env_hash:
                dirty.append(DirtyReason(step.step_id, "runner_environment_changed", step.runner))
        return dirty


class ContentAddressedStore:
    """File-per-blob CAS with integrity checks, atomic refs, and streaming reads."""

    def __init__(self, root: str | Path, *, max_memory_blob_bytes: int = 8 * 1024 * 1024):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ref_dir = self.root / "refs"
        self.ref_dir.mkdir(exist_ok=True)
        self.max_memory_blob_bytes = max_memory_blob_bytes

    def _blob_path(self, digest: str) -> Path:
        return self.root / digest

    def put(self, payload: bytes) -> str:
        digest = _hash_bytes(payload)
        path = self._blob_path(digest)
        if not path.exists():
            atomic_write_bytes(path, payload)
        return digest

    def put_json(self, payload: Any) -> str:
        return self.put(_canonical_json(payload))

    def get(self, digest: str, *, max_bytes: int | None = None) -> bytes:
        path = self._blob_path(digest)
        if not path.exists():
            raise FileNotFoundError(digest)
        if max_bytes is None:
            max_bytes = self.max_memory_blob_bytes
        size = path.stat().st_size
        if size > max_bytes:
            raise MemoryError(f"CAS blob {digest} is {size} bytes; use stream()")
        data = path.read_bytes()
        actual = _hash_bytes(data)
        if actual != digest:
            raise ValueError(f"CAS integrity error for {digest}: got {actual}")
        return data

    def stream(self, digest: str, *, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        path = self._blob_path(digest)
        if not path.exists():
            raise FileNotFoundError(digest)
        if blake3 is not None:
            hasher = blake3.blake3()
        else:
            hasher = hashlib.blake2b(digest_size=32)
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(chunk_size), b""):
                hasher.update(chunk)
                yield chunk
        actual = hasher.hexdigest()
        if actual != digest:
            raise ValueError(f"CAS integrity error for {digest}: got {actual}")

    def exists(self, digest: str) -> bool:
        return self._blob_path(digest).exists()

    def put_ref_atomic(self, name: str, digest: str) -> None:
        if not self.exists(digest):
            raise FileNotFoundError(f"cannot point ref {name!r} at missing blob {digest}")
        atomic_write_bytes(self.ref_dir / name, digest.encode("ascii"))

    def get_ref(self, name: str) -> str | None:
        path = self.ref_dir / name
        if not path.exists():
            return None
        digest = path.read_text(encoding="ascii").strip()
        if not digest:
            return None
        return digest

    def gc(self, *, keep: Iterable[str], max_age_seconds: float | None = None) -> int:
        keep_set = set(keep)
        now = time.time()
        removed = 0
        for child in self.root.iterdir():
            if child == self.ref_dir or child.is_dir() or child.name in keep_set:
                continue
            if max_age_seconds is not None and now - child.stat().st_mtime < max_age_seconds:
                continue
            child.unlink()
            removed += 1
        return removed


class ProofSubstrate:
    """High-level graph builder/loader for reproducible self-improvement."""

    DEFAULT_IGNORES = (".git/*", "target/*", ".aura/*", "__pycache__/*", "*.pyc")

    def __init__(self, project_root: str | Path, *, cas_dir: str | Path | None = None):
        self.project_root = Path(project_root).resolve()
        self.cas = ContentAddressedStore(cas_dir or (self.project_root / ".aura" / "proof_cas"))

    def load_or_create_graph(self) -> BuildGraph:
        latest = self.cas.get_ref("latest_graph")
        if latest and self.cas.exists(latest):
            return BuildGraph.from_dict(json.loads(self.cas.get(latest, max_bytes=64 * 1024 * 1024).decode("utf-8")))
        graph = self.build_graph()
        self.store_graph(graph)
        return graph

    def store_graph(self, graph: BuildGraph) -> str:
        digest = self.cas.put_json(graph.to_dict())
        self.cas.put_ref_atomic("latest_graph", digest)
        return digest

    def build_graph(
        self,
        *,
        runner_specs: Sequence[Mapping[str, Any]] | None = None,
        ignore_patterns: Sequence[str] = DEFAULT_IGNORES,
    ) -> BuildGraph:
        graph = BuildGraph(root=str(self.project_root))
        for path in self._walk_sources(ignore_patterns):
            source = SourceFile.from_path(self.project_root, path)
            graph.sources[source.path] = source

        for tool_name in ("rustc", "cargo", "make", "cc", "ld", "python", "pytest"):
            tool = ToolSpec.discover(tool_name)
            if tool:
                graph.tools[tool.name] = tool

        specs = list(runner_specs or self._default_runner_specs(graph))
        for spec in specs:
            runner = str(spec["runner"])
            include = tuple(spec.get("include", ("**/*",)))
            exclude = tuple(spec.get("exclude", ()))
            env_policy = spec.get("env_policy") or self._env_policy_for_runner(runner)
            tools = tuple(t for t in spec.get("tools", ()) if t in graph.tools)
            sources = tuple(sorted(self._match_sources(graph.sources, include, exclude)))
            identity = _hash_bytes(
                _canonical_json(
                    {
                        "runner": runner,
                        "command": list(spec.get("command", ())),
                        "sources": sources,
                        "tools": tools,
                        "env_hash": env_policy.hash(),
                    }
                )
            )[:16]
            graph.steps[identity] = BuildStep(
                step_id=identity,
                runner=runner,
                command=tuple(spec.get("command", ())),
                sources=sources,
                tools=tools,
                env_policy=env_policy,
                env_hash=env_policy.hash(),
            )
        return graph

    def mark_artifact(self, graph: BuildGraph, step_id: str, artifact: bytes | Path) -> str:
        payload = artifact.read_bytes() if isinstance(artifact, Path) else bytes(artifact)
        digest = self.cas.put(payload)
        graph.steps[step_id].artifact_hash = digest
        return digest

    def dirty_reasons(self, graph: BuildGraph, *, env: Mapping[str, str] | None = None) -> list[DirtyReason]:
        return graph.dirty_nodes(self.cas, env=env)

    def _walk_sources(self, ignore_patterns: Sequence[str]) -> Iterator[Path]:
        for path in self.project_root.rglob("*"):
            if not path.is_file():
                continue
            rel = str(path.relative_to(self.project_root))
            if any(fnmatch.fnmatch(rel, pattern) or rel.startswith(pattern.rstrip("/*")) for pattern in ignore_patterns):
                continue
            yield path

    def _default_runner_specs(self, graph: BuildGraph) -> list[dict[str, Any]]:
        specs = []
        if "cargo" in graph.tools and (self.project_root / "Cargo.toml").exists():
            specs.append(
                {
                    "runner": "cargo",
                    "command": ("cargo", "test", "--locked"),
                    "include": ("Cargo.toml", "Cargo.lock", "src/**/*.rs", "crates/**/*.rs", "tests/**/*.rs"),
                    "exclude": ("target/*",),
                    "tools": ("cargo", "rustc"),
                    "env_policy": CARGO_ENV,
                }
            )
        if "pytest" in graph.tools and (self.project_root / "tests").exists():
            specs.append(
                {
                    "runner": "pytest",
                    "command": ("python", "-m", "pytest"),
                    "include": ("*.py", "core/**/*.py", "tests/**/*.py", "pyproject.toml", "pytest.ini"),
                    "exclude": ("**/__pycache__/*",),
                    "tools": ("python", "pytest"),
                    "env_policy": EnvPolicy(keys=("PATH", "HOME", "PYTHONPATH"), prefixes=("AURA_", "PYTEST_")),
                }
            )
        return specs

    @staticmethod
    def _env_policy_for_runner(runner: str) -> EnvPolicy:
        if runner == "cargo":
            return CARGO_ENV
        if runner == "make":
            return MAKE_ENV
        if runner == "custom":
            return CUSTOM_ENV
        return EnvPolicy(keys=("PATH", "HOME"), prefixes=("AURA_",))

    @staticmethod
    def _match_sources(
        sources: Mapping[str, SourceFile],
        include: Sequence[str],
        exclude: Sequence[str],
    ) -> list[str]:
        def matches(path: str, pattern: str) -> bool:
            if fnmatch.fnmatch(path, pattern):
                return True
            if "/**/" in pattern and fnmatch.fnmatch(path, pattern.replace("/**/", "/")):
                return True
            return Path(path).match(pattern)

        matched = []
        for source in sources:
            if include and not any(matches(source, pattern) for pattern in include):
                continue
            if exclude and any(matches(source, pattern) for pattern in exclude):
                continue
            matched.append(source)
        return matched
