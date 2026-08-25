#!/usr/bin/env python3
"""Inventory and ratchet consequential effect ownership.

The legacy governance lint matched a small list of historic method names and
therefore returned green while raw subprocess, network, filesystem, desktop,
gateway, and Will calls remained distributed across the runtime. This scanner
does not pretend the existing debt is already gone. It records every recognized
effect call by category, file, lexical scope, and resolved callee, then enforces
an exact checked-in baseline:

* a new bucket or an increased count is a governance regression;
* a removed/decreased bucket makes the baseline stale and requires an explicit
  refresh, preserving an auditable record of debt reduction;
* unreadable or unparsable production files fail the analyzer;
* canonical primitive owners are reported separately from migration debt.

The baseline is a ratchet, not an allow-list. Updating it is an explicit review
act and must not be used to normalize unexplained growth.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.lint_governance_compat import (  # noqa: E402,F401
    ALLOW_LIST,
    CONSEQUENTIAL_CALLS,
)

DEFAULT_BASELINE = ROOT / "config" / "aura_effect_ownership_baseline.json"
BASELINE_SCHEMA_VERSION = 1

SCAN_ROOTS = ("core", "interface", "skills", "tools/longevity", "tools/chaos")
SUBPROCESS_DECLARATION_SCAN_ROOTS = ("core", "interface", "skills", "tools", "training")
SKIP_DIR_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "archive",
    "aura_bench",
    "node_modules",
    "tests",
}

# Compatibility exports for older proof tooling. The scanner does not depend
# on the narrow CONSEQUENTIAL_CALLS or skip-file ALLOW_LIST.

CANONICAL_PRIMITIVE_OWNERS: dict[str, frozenset[str]] = {
    "raw_subprocess": frozenset({"core/runtime/subprocess_gateway.py"}),
    "raw_network": frozenset({"core/runtime/network_gateway.py"}),
    "raw_file_mutation": frozenset(
        {
            "core/runtime/atomic_writer.py",
            "core/brain/llm/latent_cortex/campaign_journal.py",
            # Private paired-action snapshots require no-follow opens,
            # owner/mode/link checks, directory fsync, key-first destruction,
            # and staged crash recovery that the general writer deliberately
            # does not expose. This owner accepts only its fixed internal
            # namespace and schema-bound campaign state, never caller paths.
            "core/brain/llm/latent_cortex/action_state_capture.py",
            # Detached campaign evidence is an immutable, no-follow verifier
            # and staged import is the sole transactional owner of its bounded
            # private arm artifacts. Neither accepts arbitrary user paths.
            "core/brain/llm/latent_cortex/detached_campaign_evidence.py",
            "core/brain/llm/latent_cortex/worker_attempt_import.py",
            # Verified transition evidence owns one owner-private,
            # content-addressed blob namespace plus an externally pinned,
            # OS-append-only attempt ledger. Directory-descriptor-relative,
            # no-follow opens provide create-once publication, inode-stable
            # reads, and actual component-file rehashing; callers cannot
            # choose blob names or mutate production paths.
            "core/learning/verified_transition_episode.py",
            # The canonical physical twin owns one fixed SQLite sibling archive.
            # It accepts no payload path or filename from callers and requires
            # no-follow, owner-only, fsync-backed two-phase segment rotation to
            # preserve lifecycle evidence across database compaction and crash.
            "core/reality_reach/digital_twin.py",
            # Verified-replay SFT publication owns one fixed, owner-private
            # candidate/evaluator namespace. Its sole raw mutation is a
            # no-follow, single-link, inode-bound pair-publication lock; all
            # payload and commit bytes still traverse FileWriteGateway.
            "core/learning/verified_replay_sft_publication.py",
            # Create-once key material. Each of these seals a secret with
            # os.open(O_WRONLY|O_CREAT|O_EXCL, 0o600) — the flags ARE the
            # security property: exclusive creation is what makes the key
            # unforgeable, and 0600-at-creation is what stops a window where it
            # exists world-readable. FileWriteGateway writes content; it cannot
            # express "fail if this already exists, and never be readable by
            # anyone else, from the first instant". Each owns one fixed key
            # path and accepts no caller-supplied filename.
            "core/container.py",
            "core/continuity.py",
            "core/brain/verification/independent_evidence.py",
            # The measurement chain is the sibling of verified_transition_episode
            # above and works the same way: one owner-private, digest-bound
            # generation namespace published two-phase, with os.chmod sealing a
            # generation immutable and rmtree reclaiming only its own staged
            # temporaries. It accepts no caller path and publishes no adapter.
            "core/learning/verified_transition_measurement_chain.py",
            # The recurrent shadow pointer is the sole CAS owner for one fixed,
            # private release namespace. Its no-follow create/replace/fsync and
            # digest-addressed retirement operations are the custody contract;
            # callers cannot select an arbitrary destination or payload.
            "core/brain/llm/unified_recurrent_shadow_pointer.py",
            "core/runtime/file_read_gateway.py",
            "core/runtime/file_write_gateway.py",
            "core/runtime/shutdown_artifact_store.py",
        }
    ),
    "raw_desktop": frozenset({"core/runtime/desktop_action_gateway.py"}),
    "raw_browser": frozenset(),
    "direct_atomic_file_write": frozenset(
        {
            # Latent-cortex persistence owns only schema-bound private artifact
            # directories. Payload bytes and commit markers still cross the
            # governed no-follow FileWriteGateway transaction below.
            "core/brain/llm/latent_cortex/persistence.py",
            "core/memory/memory_write_gateway.py",
            # Immutable recurrence-training generations are a purpose-built
            # atomic evidence store, analogous to campaign_journal. It owns no
            # arbitrary user path and advances only a digest-bound pointer.
            "core/learning/recurrence_training_state.py",
            # Synthetic recurrent-SFT research state owns only immutable,
            # digest-bound quarantine checkpoints and a hash-linked journal
            # under the operator-selected private run root. It cannot publish
            # adapters or mutate production state.
            "core/learning/structured_sft_research_state.py",
            # Paired-transition receipts publish immutable, digest-named
            # evidence blobs under one owner-private store. This owner accepts
            # no caller-selected filename or production mutation surface.
            "core/learning/verified_transition_episode.py",
            # Qualified recurrent authority owns one fixed private activation
            # document and retirement namespace. Directory custody is checked
            # before all gateway-mediated publication and revocation.
            "core/brain/llm/unified_recurrent_qualified_activation_store.py",
            "core/runtime/atomic_writer.py",
            "core/runtime/file_write_gateway.py",
            "core/runtime/post_action_receipt.py",
            "core/runtime/receipts.py",
            "core/runtime/shutdown_artifact_store.py",
            "core/state/state_gateway.py",
        }
    ),
    "subprocess_gateway": frozenset(
        {
            "core/runtime/action_executor.py",
            "core/runtime/desktop_action_gateway.py",
            "core/runtime/skill_catalog_probe.py",
        }
    ),
    "network_gateway": frozenset(
        {
            "core/runtime/action_executor.py",
            # Public HTTP transport fixes method-derived mutability, response
            # bounds, and private-address/redirect pinning before delegating to
            # NetworkGateway. Callers cannot disable those constraints.
            "core/runtime/public_http_transport.py",
        }
    ),
    "file_write_gateway": frozenset(
        {
            # Engineering export owns one fixed artifacts/live_designs tree.
            # Caller-selected subdirectories are confined beneath that root;
            # bundle slugs and filenames are canonical single components, and
            # every byte still crosses FileWriteGateway under a named scope.
            "core/engineering/export.py",
            # Learned-language persistence owns two fixed, schema-bound
            # namespaces: matcher state beneath Aura's configured data root and
            # the frozen measurement receipt beneath the repository artifact
            # root. Callers provide records and identities, never paths.
            "core/language/substrate_store.py",
            "core/agency/tool_orchestrator.py",
            "core/agency/self_repair_backlog.py",
            # Scientific preregistrations publish once beneath a fixed private
            # evidence root using the plan hash as the only filename. Runtime
            # callers cannot choose an artifact name or replace an existing
            # plan after observing results; tests may inject only the root.
            "core/evaluation/preregistration.py",
            # Independently checked verifier outcomes are append-only,
            # schema-bound calibration evidence under Aura's data directory.
            # The ledger accepts no arbitrary runtime action or user path and
            # writes only from its named internal governance scope.
            "core/brain/llm/latent_cortex/verifier_fusion.py",
            # Encrypted session-memory pins own one fixed JSONL filename and
            # schema. The owner performs bounded, inode-stable reads and
            # gateway-mediated atomic replacement inside a named memory-write
            # scope; runtime callers cannot select a filename or plaintext
            # payload.
            "core/memory/session_pin_ledger.py",
            # Recall observations own one fixed, bounded SQLite schema beneath
            # Aura's configured memory root. The payload contains only numeric
            # activation/rank evidence, accepts no caller-selected filename or
            # memory content, and writes inside its named internal scope.
            "core/memory/recall_observations.py",
            # External-effect transactions are digest-sealed, path-derived
            # records under Aura's data directory. The coordinator accepts no
            # caller-selected file path and writes only from its named scope.
            "core/brain/external_execute_coordinator.py",
            "core/brain/llm/latent_cortex/persistence.py",
            # Endogenous vocabulary persistence owns one schema-bound pair of
            # model-specific weights and a digest-binding manifest. Runtime
            # callers provide learned arrays, never an arbitrary file effect;
            # both payloads commit through one rollback-safe gateway batch.
            "core/brain/llm/endogenous_vocab_head.py",
            # Endogenous pair persistence owns one bounded JSONL corpus and
            # its fixed rotation namespace beneath the configured Aura data
            # root. It records only state/text pairs from completed turns and
            # serializes every rotate/append transaction through this owner.
            "core/brain/llm/endogenous_pair_recorder.py",
            # Qualified recurrent authority publishes only its schema-bound,
            # CAS-guarded activation and digest-addressed retirement receipt
            # under a fixed Aura state root and named governance scopes.
            "core/brain/llm/unified_recurrent_qualified_activation_store.py",
            # Ontogeny owns only its schema-bound experience, reservoir,
            # learned-head, and authority records under the configured Aura
            # data root (or an explicitly injected test store). Every write
            # remains inside a named state-mutation governance scope.
            "core/ontogeny/authority.py",
            "core/ontogeny/experience.py",
            "core/ontogeny/service.py",
            "core/ontogeny/state.py",
            # The physical historian owns one fixed SQLite evidence store and
            # its initialization probe. It accepts no caller-selected path or
            # payload; all mutation still traverses named FileWriteGateway
            # scopes before SQLite opens the owner-private database.
            "core/reality_reach/historian.py",
            # Metrology owns one fixed, schema-bound calibration and acquisition
            # evidence journal. It accepts no caller-selected persistence path in
            # production and writes only digest-sealed records from its named
            # internal state-mutation scope.
            "core/reality_reach/metrology.py",
            # The learned world model owns one fixed, schema-bound VRNN
            # checkpoint under Aura's data root. It accepts no caller path,
            # and every publication executes in its named state-mutation
            # scope through FileWriteGateway.
            "core/world_model/learned_world_model.py",
            # Singleton owns one fixed boot-refusal marker under Aura's private
            # run directory. It accepts no caller-selected path or payload and
            # publishes/clears only that bounded launcher coordination record.
            "core/utils/singleton.py",
            # Bus bags and cross-process leases are bounded internal evidence
            # stores. Their paths are runtime-derived, never user-selected,
            # and each write executes inside a named governed scope.
            "core/observability/bus_recorder.py",
            "core/observability/trace_events.py",
            "core/runtime/lease.py",
            # Task-disjoint prefix-stability calibration owns one exact,
            # content-addressed artifact schema. It accepts no runtime action
            # or arbitrary effect and is the sole writer for that evidence.
            "core/learning/prefix_stability.py",
            # Verified replay projection writes only its fixed custody commit
            # and exact candidate/evaluator artifact sets under Aura's private
            # RLC root, inside named internal governance scopes. It grants no
            # trainer authority and accepts no arbitrary artifact filenames.
            "core/learning/verified_replay_sft_publication.py",
            # Combined lineage publication owns one fixed candidate/evaluator
            # custody namespace. All payload and commit bytes traverse
            # FileWriteGateway and the result grants no training authority.
            "core/learning/combined_sft_lineage_publication.py",
            # The safe optimizer writes only its configured adapter state and
            # byte-identical backup through FileWriteGateway; rollback restores
            # that same bounded artifact inside the optimizer's governed lane.
            "core/adaptation/safe_optimizer.py",
            "core/runtime/action_executor.py",
            "core/runtime/detached_subprocess_broker.py",
            "core/runtime/flight_recorder.py",
            "core/security/tls_local.py",
            "core/self_improvement/program_dna.py",
            "infrastructure/rollback.py",
        }
    ),
    "desktop_action_gateway": frozenset({"core/runtime/action_executor.py"}),
    "memory_write_gateway": frozenset({"core/runtime/action_executor.py"}),
    "state_gateway": frozenset({"core/runtime/action_executor.py"}),
    "will_decision": frozenset(
        {
            "core/runtime/action_executor.py",
            "core/executive/authority_gateway.py",
            "core/governance/authority_gateway.py",
        }
    ),
    "action_executor": frozenset(),
}

_SUBPROCESS_CALLS = frozenset(
    {
        "asyncio.create_subprocess_exec",
        "asyncio.create_subprocess_shell",
        "multiprocessing.Process",
        "os.popen",
        "os.posix_spawn",
        "os.posix_spawnp",
        "os.spawnl",
        "os.spawnle",
        "os.spawnlp",
        "os.spawnlpe",
        "os.spawnv",
        "os.spawnve",
        "os.spawnvp",
        "os.spawnvpe",
        "os.system",
        "subprocess.Popen",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.run",
    }
)
_NETWORK_EXACT_CALLS = frozenset(
    {
        "asyncio.open_connection",
        "http.client.HTTPConnection",
        "http.client.HTTPSConnection",
        "socket.create_connection",
        "urllib.request.build_opener",
        "urllib.request.urlopen",
        "websockets.connect",
    }
)
_NETWORK_PREFIXES = (
    "aiohttp.ClientSession().",
    "httpx.",
    "requests.",
    "urllib3.",
)
_BROWSER_EXACT_CALLS = frozenset(
    {
        "selenium.webdriver.Chrome",
        "selenium.webdriver.Edge",
        "selenium.webdriver.Firefox",
        "selenium.webdriver.Safari",
        "webbrowser.open",
        "webbrowser.open_new",
        "webbrowser.open_new_tab",
    }
)
_BROWSER_ACTION_METHODS = frozenset(
    {
        "check",
        "click",
        "evaluate",
        "fill",
        "get",
        "goto",
        "launch",
        "new_page",
        "press",
        "select_option",
        "set_input_files",
        "type",
        "uncheck",
    }
)
_RAW_FILE_EXACT_CALLS = frozenset(
    {
        "os.chflags",
        "os.chmod",
        "os.link",
        "os.makedirs",
        "os.mkdir",
        # os.open is NOT here: it is the one call in this list that can be a
        # pure read. Its second argument is an int flag bitmask, so whether it
        # mutates depends on O_WRONLY/O_RDWR/O_CREAT/O_TRUNC/O_APPEND/O_EXCL
        # being present. Flagging it unconditionally reported the no-follow
        # opens that exist to HASH and VERIFY files as raw file mutations.
        # `_os_open_flags_mutate` decides, and defaults to mutation when the
        # flag expression cannot be read.
        "os.remove",
        "os.removedirs",
        "os.rename",
        "os.renames",
        "os.replace",
        "os.rmdir",
        "os.symlink",
        "os.truncate",
        "os.unlink",
        "shutil.copy",
        "shutil.copy2",
        "shutil.copyfile",
        "shutil.copytree",
        "shutil.move",
        "shutil.rmtree",
    }
)
_PATH_MUTATION_METHODS = frozenset(
    {
        "chmod",
        "hardlink_to",
        "mkdir",
        "rmdir",
        "symlink_to",
        "touch",
        "unlink",
        "write_bytes",
        "write_text",
    }
)
_AMBIGUOUS_PATH_MUTATION_METHODS = frozenset({"rename", "replace"})
_MODED_FILE_OPEN_CALLS = frozenset(
    {
        "aiofiles.open",
        "builtins.open",
        "bz2.open",
        "codecs.open",
        "gzip.open",
        "io.open",
        "lzma.open",
        "open",
        "tarfile.open",
        "wave.open",
    }
)
_ATOMIC_FILE_CALL_SUFFIXES = (
    "atomic_append_text",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "async_atomic_append_text",
    "async_atomic_write_bytes",
    "async_atomic_write_json",
    "async_atomic_write_text",
    "async_durable_replace",
    "async_durable_unlink",
    "durable_replace",
    "durable_unlink",
    "ensure_private_directory",
)
_DESKTOP_MUTATION_METHODS = frozenset(
    {
        "click",
        "doubleClick",
        "dragRel",
        "dragTo",
        "hscroll",
        "hotkey",
        "keyDown",
        "keyUp",
        "leftClick",
        "middleClick",
        "mouseDown",
        "mouseUp",
        "move",
        "moveRel",
        "moveTo",
        "press",
        "release",
        "rightClick",
        "scroll",
        "tripleClick",
        "typewrite",
        "vscroll",
        "write",
    }
)
_GATEWAY_FACTORIES = {
    "get_subprocess_gateway": "subprocess_gateway",
    "get_network_gateway": "network_gateway",
    "get_file_write_gateway": "file_write_gateway",
    "get_desktop_action_gateway": "desktop_action_gateway",
    "get_memory_write_gateway": "memory_write_gateway",
    "get_state_gateway": "state_gateway",
    "get_will": "will_decision",
}
_GATEWAY_METHODS = {
    "subprocess_gateway": frozenset(
        {
            "run",
            "run_async",
            "run_model_blocking",
            "spawn",
            "spawn_async",
            "spawn_python_process",
            "spawn_shell_async",
        }
    ),
    "network_gateway": frozenset(
        {"connect_stream", "connect_websocket", "request", "request_async"}
    ),
    "file_write_gateway": frozenset(
        {
            "append_text",
            "append_text_async",
            "copy_path_async",
            "delete_file",
            "delete_path_async",
            "drain_text",
            "ensure_directory",
            "ensure_directory_async",
            "move_path_async",
            "open_owned_binary",
            "replace_file",
            "write_bytes",
            "write_bytes_batch",
            "write_bytes_async",
            "write_json",
            "write_json_async",
            "write_text",
            "write_text_async",
        }
    ),
    "desktop_action_gateway": frozenset({"run_applescript", "run_applescript_async"}),
    "memory_write_gateway": frozenset({"quarantine", "write"}),
    "state_gateway": frozenset({"mutate"}),
    "will_decision": frozenset({"decide", "decide_async"}),
}


@dataclass(frozen=True, order=True)
class EffectBucket:
    category: str
    path: str
    scope: str
    callee: str
    count: int
    canonical_owner: bool

    def key(self) -> tuple[str, str, str, str]:
        return (self.category, self.path, self.scope, self.callee)


@dataclass(frozen=True)
class ScanProblem:
    path: str
    problem: str


class EffectVisitor(ast.NodeVisitor):
    def __init__(self, *, relative_path: str) -> None:
        self.relative_path = relative_path
        self.aliases: dict[str, str] = {}
        self.binding_scopes: list[dict[str, str]] = [{}]
        self.scope_parts: list[str] = ["<module>"]
        self.calls: list[tuple[str, str, int]] = []

    @property
    def scope(self) -> str:
        return ".".join(self.scope_parts)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name.split(".", 1)[0]
            self.aliases[local] = alias.name if alias.asname else local

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            return
        for alias in node.names:
            if alias.name == "*":
                continue
            local = alias.asname or alias.name
            self.aliases[local] = f"{node.module}.{alias.name}"

    def visit_Assign(self, node: ast.Assign) -> None:
        resolved = self._binding_value(node.value)
        if resolved:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.binding_scopes[-1][target.id] = resolved
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and isinstance(node.target, ast.Name):
            resolved = self._binding_value(node.value)
            if resolved:
                self.binding_scopes[-1][node.target.id] = resolved
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scoped(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scoped(node, node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scoped(node, node.name)

    def _visit_scoped(self, node: ast.AST, name: str) -> None:
        self.scope_parts.append(name)
        self.binding_scopes.append({})
        for child in ast.iter_child_nodes(node):
            self.visit(child)
        self.binding_scopes.pop()
        self.scope_parts.pop()

    def visit_Call(self, node: ast.Call) -> None:
        for category, callee in self._classified_effect_calls(node):
            self.calls.append((category, callee, node.lineno))
        self.generic_visit(node)

    def _classified_effect_calls(self, node: ast.Call) -> list[tuple[str, str]]:
        classified: list[tuple[str, str]] = []
        callee = self._resolve_expr(node.func)
        category = self._classify_call(node, callee)
        if category:
            classified.append((category, callee or "<dynamic>"))
        delegated = _delegated_call(node, callee)
        if delegated is not None:
            delegated_callee = self._resolve_expr(delegated.func)
            delegated_category = self._classify_call(delegated, delegated_callee)
            if delegated_category:
                classified.append((delegated_category, delegated_callee))
        return classified

    def _lookup_binding(self, name: str) -> str | None:
        for scope in reversed(self.binding_scopes):
            if name in scope:
                return scope[name]
        return None

    def _resolve_expr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return self._lookup_binding(node.id) or self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            base = self._resolve_expr(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        if isinstance(node, ast.Call):
            base = self._resolve_expr(node.func)
            return f"{base}()" if base else ""
        return ""

    def _binding_value(self, node: ast.AST) -> str | None:
        if not isinstance(node, ast.Call):
            return None
        callee = self._resolve_expr(node.func)
        factory = _factory_category(callee)
        if factory:
            return f"<{factory}>"
        if _is_multiprocessing_context_factory(callee):
            return "<multiprocessing.context>"
        if callee.endswith("pathlib.Path") or callee == "Path":
            return "<pathlib.Path>"
        if callee.endswith("aiohttp.ClientSession"):
            return "aiohttp.ClientSession()"
        return None

    def _classify_call(self, node: ast.Call, callee: str) -> str | None:
        method = callee.rsplit(".", 1)[-1] if callee else ""

        for category, methods in _GATEWAY_METHODS.items():
            if method not in methods:
                continue
            if f"<{category}>." in callee:
                return category
            if _callee_uses_factory(callee, category):
                return category

        if method in {"execute"} and (
            "ActionExecutor." in callee or callee.endswith("action_executor.ActionExecutor.execute")
        ):
            return "action_executor"

        if _matches_exact(callee, _SUBPROCESS_CALLS):
            return "raw_subprocess"
        if method == "Process" and (
            "<multiprocessing.context>." in callee
            or "multiprocessing.get_context()." in callee
            or _looks_multiprocessing_context_receiver(callee)
        ):
            return "raw_subprocess"
        if _matches_exact(callee, _NETWORK_EXACT_CALLS) or any(
            _strip_project_prefix(callee).startswith(prefix) for prefix in _NETWORK_PREFIXES
        ):
            return "raw_network"
        stripped_callee = _strip_project_prefix(callee)
        if (
            (stripped_callee.startswith("pyautogui.") and method in _DESKTOP_MUTATION_METHODS)
            or (
                stripped_callee.startswith(
                    ("pynput.keyboard.Controller().", "pynput.mouse.Controller().")
                )
                and method in _DESKTOP_MUTATION_METHODS
            )
            or stripped_callee.startswith("Quartz.CGEventPost")
        ):
            return "raw_desktop"
        if _matches_exact(callee, _BROWSER_EXACT_CALLS) or (
            method in _BROWSER_ACTION_METHODS and _looks_browser_receiver(callee)
        ):
            return "raw_browser"
        if callee.endswith(_ATOMIC_FILE_CALL_SUFFIXES):
            return "direct_atomic_file_write"
        if _matches_exact(callee, _RAW_FILE_EXACT_CALLS):
            return "raw_file_mutation"
        if method in _PATH_MUTATION_METHODS:
            return "raw_file_mutation"
        if (
            method in _AMBIGUOUS_PATH_MUTATION_METHODS
            and _looks_path_receiver(callee)
            and _replaces_a_path(node)
        ):
            return "raw_file_mutation"
        if _strip_project_prefix(callee) == "os.open":
            return "raw_file_mutation" if _os_open_flags_mutate(node) else None
        if method == "open" and _open_call_mutates(node, callee):
            return "raw_file_mutation"
        return None


def _strip_project_prefix(value: str) -> str:
    for prefix in ("core.", "interface.", "skills."):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def _matches_exact(value: str, options: Sequence[str] | frozenset[str]) -> bool:
    stripped = _strip_project_prefix(value)
    return stripped in options or any(stripped.endswith(f".{option}") for option in options)


def _factory_category(callee: str) -> str | None:
    stripped = callee.removesuffix("()")
    leaf = stripped.rsplit(".", 1)[-1]
    return _GATEWAY_FACTORIES.get(leaf)


def _is_multiprocessing_context_factory(callee: str) -> bool:
    stripped = _strip_project_prefix(callee.removesuffix("()"))
    return stripped == "multiprocessing.get_context" or stripped.endswith(
        ".multiprocessing.get_context"
    )


def _looks_multiprocessing_context_receiver(callee: str) -> bool:
    if not callee.endswith(".Process"):
        return False
    receiver = callee.rsplit(".", 1)[0].removeprefix("self.")
    leaf = receiver.rsplit(".", 1)[-1].casefold().removeprefix("_")
    return leaf in {"context", "ctx", "mp_context", "selected_context"} or leaf.endswith(
        ("_context", "_ctx")
    )


def _callee_uses_factory(callee: str, category: str) -> bool:
    for factory, factory_category in _GATEWAY_FACTORIES.items():
        if factory_category == category and f"{factory}()." in callee:
            return True
    return False


def _delegated_call(node: ast.Call, callee: str) -> ast.Call | None:
    stripped = _strip_project_prefix(callee)
    if stripped in {"asyncio.to_thread", "anyio.to_thread.run_sync"}:
        callable_index = 0
    elif stripped.endswith(".run_in_executor"):
        callable_index = 1
    else:
        return None
    if len(node.args) <= callable_index:
        return None
    target = node.args[callable_index]
    if not isinstance(target, (ast.Attribute, ast.Name)):
        return None
    return ast.Call(
        func=target,
        args=list(node.args[callable_index + 1 :]),
        keywords=list(node.keywords),
    )


#: POSIX open flags that actually write. `os.open` takes an INT bitmask, not a
#: string mode, so the string-mode logic below cannot read it: literal_eval
#: raises on `os.O_RDONLY | os.O_NOFOLLOW` and the fallback assumed mutation.
#: Every read-only os.open in the tree was therefore reported as a raw file
#: mutation — including the no-follow opens that exist to HASH and VERIFY files
#: without touching them.
_WRITING_OPEN_FLAGS = frozenset(
    {"O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND", "O_EXCL"}
)


def _os_open_flags_mutate(node: ast.Call) -> bool:
    """True when an ``os.open`` flag expression can write.

    Read-only by default: a flag set naming none of the writing flags opens a
    file descriptor and changes nothing. An unreadable expression is treated as
    writing, because an unknown flag set is not evidence of safety.
    """
    if len(node.args) < 2:
        return False  # os.open(path) with no flags defaults to O_RDONLY
    names = {
        child.attr if isinstance(child, ast.Attribute) else child.id
        for child in ast.walk(node.args[1])
        if isinstance(child, (ast.Attribute, ast.Name))
    }
    if not names:
        return True  # a computed flag set nobody can read statically
    return bool(names & _WRITING_OPEN_FLAGS)


def _open_call_mutates(node: ast.Call, callee: str) -> bool:
    if callee not in {"open", "builtins.open"} and not callee.endswith(".open"):
        return False
    if _strip_project_prefix(callee) == "os.open":
        return _os_open_flags_mutate(node)
    mode_node: ast.AST | None = None
    stripped_callee = _strip_project_prefix(callee)
    is_function_open = stripped_callee in _MODED_FILE_OPEN_CALLS
    is_path_method_open = not is_function_open and _looks_path_receiver(callee)
    if not is_function_open and not is_path_method_open:
        return False
    if is_function_open and len(node.args) >= 2:
        mode_node = node.args[1]
    elif is_path_method_open and node.args:
        mode_node = node.args[0]
    for keyword in node.keywords:
        if keyword.arg == "mode":
            mode_node = keyword.value
    if mode_node is None:
        return False
    try:
        mode = ast.literal_eval(mode_node)
    except (ValueError, TypeError):
        return True
    if not isinstance(mode, str):
        return True
    normalized = mode.strip().lower()
    if not normalized:
        return False
    if stripped_callee in {"tarfile.open"}:
        return normalized[0] in {"a", "w", "x"}
    if not _is_file_mode(normalized):
        return False
    return any(marker in normalized for marker in ("a", "w", "x", "+"))


def _is_file_mode(value: str) -> bool:
    return bool(value) and all(character in "rwaxbt+" for character in value)


def _replaces_a_path(node: ast.Call) -> bool:
    """Whether this `replace`/`rename` moves a file rather than editing a string.

    `Path.replace(target)` takes one argument and `str.replace(old, new)` takes
    two, so arity separates them exactly. Without this, `op.source.replace("_",
    " ")` was reported as a raw file mutation because the receiver happened to
    be called `source`, which is on the list of words that look like paths.
    """
    return len(node.args) <= 1 and not node.keywords


def _looks_path_receiver(callee: str) -> bool:
    receiver = callee.rsplit(".", 1)[0].removesuffix("()")
    if "<pathlib.Path>" in receiver or receiver.endswith(".parent"):
        return True
    leaf = receiver.rsplit(".", 1)[-1].casefold().removeprefix("_")
    return leaf in {
        "destination",
        "directory",
        "dir",
        "file",
        "folder",
        "ledger",
        "path",
        "root",
        "source",
        "target",
    } or leaf.endswith(("_dir", "_file", "_path", "_root"))


def _looks_browser_receiver(callee: str) -> bool:
    if "." not in callee:
        return False
    receiver = callee.rsplit(".", 1)[0].removesuffix("()")
    leaf = receiver.rsplit(".", 1)[-1].casefold().removeprefix("_")
    method = callee.rsplit(".", 1)[-1].casefold()
    if method == "get":
        return leaf == "driver" or leaf.endswith("_driver")
    return leaf in {
        "browser",
        "button",
        "driver",
        "element",
        "input",
        "keyboard",
        "link",
        "locator",
        "page",
        "window",
    } or leaf.endswith(
        (
            "_box",
            "_browser",
            "_btn",
            "_button",
            "_driver",
            "_element",
            "_input",
            "_link",
            "_locator",
            "_page",
            "_window",
        )
    )


def _canonical_owner(category: str, relative_path: str) -> bool:
    owners = CANONICAL_PRIMITIVE_OWNERS.get(category, frozenset())
    if category == "action_executor":
        return True
    return relative_path in owners


def _iter_source_files(root: Path) -> Iterable[Path]:
    for top in SCAN_ROOTS:
        base = root / top
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            relative_parts = path.relative_to(root).parts
            if any(part in SKIP_DIR_PARTS for part in relative_parts):
                continue
            yield path


def _iter_subprocess_declaration_source_files(root: Path) -> Iterable[Path]:
    yielded: set[Path] = set()
    for top in SUBPROCESS_DECLARATION_SCAN_ROOTS:
        base = root / top
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            relative_parts = path.relative_to(root).parts
            if any(part in SKIP_DIR_PARTS for part in relative_parts):
                continue
            yielded.add(path)
            yield path
    entrypoint = root / "aura_main.py"
    if entrypoint.is_file() and entrypoint not in yielded:
        yield entrypoint


class _SubprocessDeclarationVisitor(EffectVisitor):
    def __init__(self, *, relative_path: str) -> None:
        super().__init__(relative_path=relative_path)
        self.violations: list[ScanProblem] = []

    def visit_Call(self, node: ast.Call) -> None:
        callee = self._resolve_expr(node.func)
        method = callee.rsplit(".", 1)[-1] if callee else ""
        subprocess_gateway_call = (
            "<subprocess_gateway>." in callee
            or _callee_uses_factory(callee, "subprocess_gateway")
        )
        if subprocess_gateway_call and method == "spawn_python_process":
            required = {
                "accelerator_capability",
                "name",
                "requested_privileges",
                "role",
                "source",
                "start_method",
                "target",
            }
            spec = node.args[0] if node.args else None
            spec_callee = self._resolve_expr(spec.func) if isinstance(spec, ast.Call) else ""
            declared = {
                str(keyword.arg)
                for keyword in getattr(spec, "keywords", ())
                if keyword.arg is not None
            }
            missing = sorted(required - declared)
            if not isinstance(spec, ast.Call) or not spec_callee.endswith(
                "PythonProcessSpec"
            ):
                missing = sorted(required)
            if missing:
                self.violations.append(
                    ScanProblem(
                        self.relative_path,
                        "python_process_contract_incomplete:"
                        f"{node.lineno}:{callee}:missing={','.join(missing)}",
                    )
                )
        if subprocess_gateway_call and method in {
            "run",
            "run_async",
            "spawn",
            "spawn_async",
            "spawn_shell_async",
        }:
            if not any(
                keyword.arg == "accelerator_capability"
                for keyword in node.keywords
            ):
                self.violations.append(
                    ScanProblem(
                        self.relative_path,
                        f"subprocess_accelerator_capability_undeclared:{node.lineno}:{callee}",
                    )
                )
        self.generic_visit(node)


def audit_subprocess_accelerator_declarations(
    root: Path = ROOT,
) -> list[ScanProblem]:
    """Return every production gateway call missing accelerator intent."""

    violations: list[ScanProblem] = []
    for path in _iter_subprocess_declaration_source_files(root):
        relative = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeError, SyntaxError) as exc:
            violations.append(
                ScanProblem(relative, f"declaration_scan_failed:{type(exc).__qualname__}:{exc}")
            )
            continue
        visitor = _SubprocessDeclarationVisitor(relative_path=relative)
        visitor.visit(tree)
        violations.extend(visitor.violations)
    return sorted(violations, key=lambda problem: (problem.path, problem.problem))


class _ScopedEffectVisitor(EffectVisitor):
    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast API
        for category, callee in self._classified_effect_calls(node):
            self.calls.append((category, f"{self.scope}\0{callee}", node.lineno))
        self.generic_visit(node)


def _scan_tree_scoped(
    tree: ast.AST,
    relative_path: str,
) -> dict[tuple[str, str, str, str], int]:
    visitor = _ScopedEffectVisitor(relative_path=relative_path)
    visitor.visit(tree)
    counts: dict[tuple[str, str, str, str], int] = {}
    for category, encoded, _line in visitor.calls:
        scope, callee = encoded.split("\0", 1)
        key = (category, relative_path, scope, callee)
        counts[key] = counts.get(key, 0) + 1
    return counts


def scan_repository(root: Path = ROOT) -> tuple[list[EffectBucket], list[ScanProblem]]:
    counts: dict[tuple[str, str, str, str], int] = {}
    problems: list[ScanProblem] = []
    for path in _iter_source_files(root):
        relative = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            problems.append(ScanProblem(relative, f"read_failed:{type(exc).__qualname__}:{exc}"))
            continue
        try:
            tree = ast.parse(source, filename=relative)
        except SyntaxError as exc:
            problems.append(
                ScanProblem(relative, f"parse_failed:{exc.lineno}:{exc.offset}:{exc.msg}")
            )
            continue
        scoped_counts = _scan_tree_scoped(tree, relative)
        for key, count in scoped_counts.items():
            counts[key] = counts.get(key, 0) + count

    buckets = [
        EffectBucket(
            category=category,
            path=path,
            scope=scope,
            callee=callee,
            count=count,
            canonical_owner=_canonical_owner(category, path),
        )
        for (category, path, scope, callee), count in counts.items()
    ]
    return sorted(buckets), sorted(problems, key=lambda problem: problem.path)


sys.modules.setdefault("tools.lint_governance", sys.modules[__name__])
_CLI_EXPORTS = frozenset(
    {"compare_inventory", "load_baseline", "main", "write_baseline"}
)


def __getattr__(name: str) -> Any:
    if name not in _CLI_EXPORTS:
        raise AttributeError(name)
    from tools import lint_governance_cli

    return getattr(lint_governance_cli, name)


if __name__ == "__main__":
    from tools.lint_governance_cli import main

    raise SystemExit(main(sys.argv[1:]))
