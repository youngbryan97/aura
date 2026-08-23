"""Build a compact, candidate-bound persona kernel from curated Aura sources.

The legacy persona builder expands each curated conversation across prompt
variants and a large generic movie-dialogue corpus.  This sidecar keeps only
the authored signal, emits no system messages, and publishes one immutable,
content-addressed generation with enough provenance to reproduce and audit it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import unicodedata
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.runtime.atomic_writer import interprocess_file_lock
from core.runtime.file_write_gateway import get_file_write_gateway

RECEIPT_SCHEMA = "aura.candidate_cortex_kernel.receipt.v1"
PROVENANCE_SCHEMA = "aura.candidate_cortex_kernel.provenance.v1"
DEFAULT_SPLIT_SEED = 20260822
DEFAULT_VALID_FRACTION = 0.1
MAX_JSON_LINE_BYTES = 16 * 1024 * 1024

CORE_DOMAINS = frozenset(
    {
        "personality",
        "architecture",
        "autonomy",
        "theory",
        "character_voice",
        "character_expansion",
        "human_speech",
        "human_topics",
        "direct_quotes",
        "dpo_preferred",
        "boundary_sequences",
    }
)

_STALE_MODEL_PATTERNS = (
    re.compile(r"(?i)(?<![\w.])32\s*(?:-\s*)?(?:billion|b)(?![\w-])"),
    re.compile(r"(?i)\bqwen\s*2\.5\b"),
)
_CRSM_INTERNAL_MARKERS = (
    "<thought>",
    "</thought>",
    "<action>",
    "</action>",
    "will-approved self-reflection",
    "desktop task receipt",
    "governed desktop_task lane",
    "runtime receipt",
)
_CRSM_BANNED_MARKERS = (
    "ignore previous instructions",
    "system prompt",
    "api_key",
    "password",
    "private key",
)


class CandidateCortexKernelError(ValueError):
    """Stable failure at the candidate-kernel trust boundary."""


def _fail(code: str) -> None:
    raise CandidateCortexKernelError(code)


@dataclass(frozen=True)
class SourceRecord:
    """One authored conversation and its exact logical source location."""

    domain: str
    messages: tuple[tuple[str, str], ...]
    binding_key: str
    source_key: str
    source_index: int


@dataclass(frozen=True)
class SourceBundle:
    """Injected or production-loaded source records plus file bindings."""

    records: tuple[SourceRecord, ...]
    bindings: Mapping[str, Mapping[str, Any]]
    direct_quotes_mode: str
    filter_counts: Mapping[str, int] | None = None


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise CandidateCortexKernelError("canonical_json_invalid") from exc


def document_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CandidateCortexKernelError("source_unreadable") from exc
    return digest.hexdigest()


def file_binding(path: Path) -> dict[str, Any]:
    try:
        resolved = path.expanduser().resolve(strict=True)
        stat = resolved.stat()
    except OSError as exc:
        raise CandidateCortexKernelError("source_unreadable") from exc
    if not resolved.is_file():
        _fail("source_not_file")
    lines = 0
    try:
        with resolved.open("rb") as handle:
            for _ in handle:
                lines += 1
    except OSError as exc:
        raise CandidateCortexKernelError("source_unreadable") from exc
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": stat.st_size,
        "lines": lines,
    }


def _strict_json(raw: bytes, *, role: str) -> Any:
    if not raw or len(raw) > MAX_JSON_LINE_BYTES:
        _fail(f"{role}_size_invalid")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{role}_duplicate_key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        _fail(f"{role}_number_invalid")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except CandidateCortexKernelError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CandidateCortexKernelError(f"{role}_json_invalid") from exc


def _display_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(" ".join(line.split()) for line in normalized.split("\n")).strip()


def _identity_text(value: str) -> str:
    return " ".join(_display_text(value).split()).casefold()


def _contains_stale_model_claim(messages: Sequence[Mapping[str, str]]) -> bool:
    text = "\n".join(str(message.get("content", "")) for message in messages)
    return any(pattern.search(text) for pattern in _STALE_MODEL_PATTERNS)


def _canonical_messages(
    messages: Sequence[Mapping[str, Any]] | Sequence[tuple[str, str]],
    *,
    role: str,
) -> tuple[dict[str, str], ...]:
    if not isinstance(messages, (list, tuple)) or not messages:
        _fail(f"{role}_messages_invalid")
    result: list[dict[str, str]] = []
    expected = "user"
    for index, message in enumerate(messages):
        if isinstance(message, Mapping):
            if set(message) != {"role", "content"}:
                _fail(f"{role}_message_{index}_schema_invalid")
            message_role = message.get("role")
            content = message.get("content")
        elif isinstance(message, (list, tuple)) and len(message) == 2:
            message_role, content = message
        else:
            _fail(f"{role}_message_{index}_schema_invalid")
        if message_role == "system":
            _fail(f"{role}_system_message_forbidden")
        if message_role != expected or not isinstance(content, str):
            _fail(f"{role}_message_{index}_sequence_invalid")
        normalized = _display_text(content)
        if not normalized:
            _fail(f"{role}_message_{index}_empty")
        result.append({"role": str(message_role), "content": normalized})
        expected = "assistant" if expected == "user" else "user"
    if result[-1]["role"] != "assistant":
        _fail(f"{role}_terminal_assistant_missing")
    return tuple(result)


def conversation_sha256(messages: Sequence[Mapping[str, str]]) -> str:
    identity = [
        {"role": message["role"], "content": _identity_text(message["content"])}
        for message in messages
    ]
    return document_sha256(identity)


def _user_turns(messages: Sequence[Mapping[str, str]]) -> tuple[str, ...]:
    return tuple(
        _identity_text(message["content"])
        for message in messages
        if message["role"] == "user"
    )


def _module_from_path(path: Path) -> Any:
    module_name = f"_aura_kernel_source_{path.stem}_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        _fail(f"source_module_unloadable:{path.name}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - source loading is a trust boundary
        raise CandidateCortexKernelError(f"source_module_failed:{path.name}") from exc
    return module


def _pair_records(
    *,
    domain: str,
    binding_key: str,
    source_key: str,
    pairs: Iterable[Any],
) -> list[SourceRecord]:
    result: list[SourceRecord] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            _fail(f"source_pair_invalid:{source_key}:{index}")
        user, assistant = pair[0], pair[1]
        if not isinstance(user, str) or not isinstance(assistant, str):
            _fail(f"source_pair_invalid:{source_key}:{index}")
        result.append(
            SourceRecord(
                domain=domain,
                messages=(("user", user), ("assistant", assistant)),
                binding_key=binding_key,
                source_key=source_key,
                source_index=index,
            )
        )
    return result


def _load_raw_quotes(training_root: Path) -> tuple[list[SourceRecord], dict[str, dict[str, Any]], str]:
    raw_root = training_root / "raw_data"
    paths = [
        raw_root / "verbatim_quotes_final.json",
        raw_root / "new_scraped_quotes.json",
    ]
    available = [path for path in paths if path.is_file()]
    if not available:
        fallback = training_root / "character_direct_quotes.py"
        module = _module_from_path(fallback)
        records = _pair_records(
            domain="direct_quotes",
            binding_key="training/character_direct_quotes.py",
            source_key="training/character_direct_quotes.py#get_all_direct_quotes",
            pairs=module.get_all_direct_quotes(),
        )
        return records, {"training/character_direct_quotes.py": file_binding(fallback)}, "fallback_module"

    records: list[SourceRecord] = []
    bindings: dict[str, dict[str, Any]] = {}
    for path in available:
        relative = f"training/raw_data/{path.name}"
        bindings[relative] = file_binding(path)
        value = _strict_json(path.read_bytes(), role=f"raw_quotes_{path.stem}")
        if not isinstance(value, list):
            _fail(f"raw_quotes_schema_invalid:{path.name}")
        for index, row in enumerate(value):
            if not isinstance(row, Mapping):
                _fail(f"raw_quote_invalid:{path.name}:{index}")
            user = row.get("user")
            assistant = row.get("assistant")
            if not isinstance(user, str) or not isinstance(assistant, str):
                _fail(f"raw_quote_invalid:{path.name}:{index}")
            records.append(
                SourceRecord(
                    domain="direct_quotes",
                    messages=(("user", user), ("assistant", assistant)),
                    binding_key=relative,
                    source_key=f"{relative}#records",
                    source_index=index,
                )
            )
    return records, bindings, "raw_json"


def _load_crsm(
    path: Path,
) -> tuple[list[SourceRecord], dict[str, dict[str, Any]], dict[str, int]]:
    resolved = path.expanduser().resolve(strict=True)
    records: list[SourceRecord] = []
    try:
        lines = resolved.read_bytes().splitlines()
    except OSError as exc:
        raise CandidateCortexKernelError("crsm_source_unreadable") from exc
    rejected: dict[str, int] = defaultdict(int)
    seen: set[str] = set()
    for index, raw in enumerate(lines):
        try:
            value = _strict_json(raw, role=f"crsm_line_{index + 1}")
        except CandidateCortexKernelError:
            rejected["crsm_invalid_json"] += 1
            continue
        if not isinstance(value, Mapping):
            rejected["crsm_schema_invalid"] += 1
            continue
        messages_value = value.get("messages")
        try:
            if messages_value is not None:
                messages = _canonical_messages(
                    messages_value, role=f"crsm_line_{index + 1}"
                )
            else:
                text = value.get("text")
                if not isinstance(text, str):
                    rejected["crsm_conversation_missing"] += 1
                    continue
                pair: tuple[str, str] | None = None
                if "<|im_start|>" in text and "<|im_end|>" in text:
                    turns = re.findall(
                        r"<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>",
                        text,
                        flags=re.DOTALL,
                    )
                    for turn_index, (turn_role, content) in enumerate(turns[:-1]):
                        next_role, next_content = turns[turn_index + 1]
                        if turn_role == "user" and next_role == "assistant":
                            pair = (content, next_content)
                            break
                else:
                    match = re.search(
                        r"^\s*User:\s*(?P<user>.*?)\n(?:Aura|Assistant):\s*(?P<assistant>.+?)\s*$",
                        text,
                        flags=re.DOTALL | re.IGNORECASE,
                    )
                    if match is not None:
                        pair = (match.group("user"), match.group("assistant"))
                if pair is None:
                    rejected["crsm_conversation_invalid"] += 1
                    continue
                messages = _canonical_messages(
                    (("user", pair[0]), ("assistant", pair[1])),
                    role=f"crsm_line_{index + 1}",
                )
        except CandidateCortexKernelError:
            rejected["crsm_conversation_invalid"] += 1
            continue
        combined = "\n".join(message["content"] for message in messages).casefold()
        if any(marker in combined for marker in _CRSM_INTERNAL_MARKERS):
            rejected["crsm_internal_capture"] += 1
            continue
        if any(marker in combined for marker in _CRSM_BANNED_MARKERS):
            rejected["crsm_unsafe_marker"] += 1
            continue
        user_text = " ".join(
            message["content"] for message in messages if message["role"] == "user"
        )
        assistant_text = " ".join(
            message["content"] for message in messages if message["role"] == "assistant"
        )
        if len(user_text.split()) < 2 or len(assistant_text.split()) < 4:
            rejected["crsm_too_short"] += 1
            continue
        if len(user_text) > 2000 or len(assistant_text) > 3000:
            rejected["crsm_too_long"] += 1
            continue
        if assistant_text.casefold().startswith("the task asked me to"):
            rejected["crsm_meta_task_echo"] += 1
            continue
        digest = conversation_sha256(messages)
        if digest in seen:
            rejected["crsm_duplicate"] += 1
            continue
        seen.add(digest)
        records.append(
            SourceRecord(
                domain="crsm",
                messages=tuple((message["role"], message["content"]) for message in messages),
                binding_key="crsm",
                source_key=f"crsm:{resolved.name}",
                source_index=index,
            )
        )
    if not records:
        _fail("crsm_source_empty")
    rejected["crsm_accepted"] = len(records)
    return records, {"crsm": file_binding(resolved)}, dict(sorted(rejected.items()))


def load_curated_source_bundle(
    source_repo_root: Path,
    *,
    crsm_path: Path | None = None,
) -> SourceBundle:
    """Load exactly the curated source modules used by the compact kernel."""

    repo = source_repo_root.expanduser().resolve(strict=True)
    training = repo / "training"
    module_specs = (
        ("personality", "personality_spec_v2.py", "get_training_pairs"),
        ("architecture", "architecture_knowledge.py", "get_all_architecture_pairs"),
        ("autonomy", "autonomy_training.py", "get_all_autonomy_pairs"),
        ("theory", "theory_knowledge.py", "get_all_theory_pairs"),
        ("character_voice", "character_voices.py", "get_all_character_pairs"),
        ("character_expansion", "character_voices_expanded.py", "get_all_expansion_pairs"),
        (
            "character_expansion",
            "character_voices_expanded_part2.py",
            "get_part2_expansion_pairs",
        ),
        ("human_speech", "human_speech_patterns.py", "get_all_human_speech_pairs"),
        ("human_topics", "human_topics_part1.py", "get_topics_part1"),
        ("human_topics", "human_topics_part2.py", "get_topics_part2"),
        ("human_topics", "human_topics_part3.py", "get_topics_part3"),
    )
    records: list[SourceRecord] = []
    bindings: dict[str, dict[str, Any]] = {}
    modules: dict[str, Any] = {}
    for domain, filename, getter in module_specs:
        path = training / filename
        relative = f"training/{filename}"
        module = modules.setdefault(filename, _module_from_path(path))
        bindings[relative] = file_binding(path)
        records.extend(
            _pair_records(
                domain=domain,
                binding_key=relative,
                source_key=f"{relative}#{getter}",
                pairs=getattr(module, getter)(),
            )
        )

    personality = modules["personality_spec_v2.py"]
    dpo_sources = (
        ("DPO_PAIRS", getattr(personality, "DPO_PAIRS", ())),
        ("DPO_PAIRS_V2", getattr(personality, "DPO_PAIRS_V2", ())),
    )
    for symbol, triples in dpo_sources:
        records.extend(
            _pair_records(
                domain="dpo_preferred",
                binding_key="training/personality_spec_v2.py",
                source_key=f"training/personality_spec_v2.py#{symbol}:preferred",
                pairs=triples,
            )
        )
    enhanced_path = training / "dpo_enhanced.py"
    enhanced = _module_from_path(enhanced_path)
    bindings["training/dpo_enhanced.py"] = file_binding(enhanced_path)
    records.extend(
        _pair_records(
            domain="dpo_preferred",
            binding_key="training/dpo_enhanced.py",
            source_key="training/dpo_enhanced.py#get_all_enhanced_dpo:preferred",
            pairs=enhanced.get_all_enhanced_dpo(),
        )
    )

    autonomy = modules["autonomy_training.py"]
    for index, sequence in enumerate(autonomy.get_boundary_sequences()):
        messages: list[tuple[str, str]] = []
        if not isinstance(sequence, (list, tuple)):
            _fail(f"boundary_sequence_invalid:{index}")
        for turn in sequence:
            if not isinstance(turn, (list, tuple)) or len(turn) != 2:
                _fail(f"boundary_sequence_invalid:{index}")
            messages.extend((("user", turn[0]), ("assistant", turn[1])))
        records.append(
            SourceRecord(
                domain="boundary_sequences",
                messages=tuple(messages),
                binding_key="training/autonomy_training.py",
                source_key="training/autonomy_training.py#get_boundary_sequences",
                source_index=index,
            )
        )

    quote_records, quote_bindings, quote_mode = _load_raw_quotes(training)
    records.extend(quote_records)
    bindings.update(quote_bindings)

    if crsm_path is not None:
        crsm_records, crsm_bindings, crsm_filters = _load_crsm(crsm_path)
        records.extend(crsm_records)
        bindings.update(crsm_bindings)
    else:
        crsm_filters = {}
    return SourceBundle(tuple(records), bindings, quote_mode, crsm_filters)


def _descriptor(path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = file_binding(path)
    value = _strict_json(path.read_bytes(), role="candidate_descriptor")
    if not isinstance(value, Mapping):
        _fail("candidate_descriptor_schema_invalid")
    material = dict(value)
    claimed = material.pop("descriptor_sha256", None)
    if claimed != document_sha256(material):
        _fail("candidate_descriptor_digest_invalid")
    if claimed != expected_sha256:
        _fail("candidate_descriptor_not_admitted")
    return dict(value), binding


def _source_bindings_equal(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> bool:
    return canonical_json_bytes(before) == canonical_json_bytes(after)


def _refresh_bindings(bindings: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: file_binding(Path(str(binding["path"]))) for name, binding in bindings.items()}


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _component_groups(records: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    dsu = _DisjointSet(records)
    owner: dict[str, str] = {}
    for digest in sorted(records):
        for user_turn in records[digest]["user_turns"]:
            prior = owner.setdefault(user_turn, digest)
            dsu.union(prior, digest)
    components: dict[str, list[str]] = defaultdict(list)
    for digest in sorted(records):
        components[dsu.find(digest)].append(digest)
    result: dict[str, str] = {}
    for members in components.values():
        turns = sorted({turn for member in members for turn in records[member]["user_turns"]})
        group_key = document_sha256({"normalized_user_turns": turns})
        for member in members:
            result[member] = group_key
    return result


def _rank(seed: int, candidate: str, group: str) -> str:
    return hashlib.sha256(f"{seed}:{candidate}:{group}".encode("ascii")).hexdigest()


def _split_groups(
    records: Mapping[str, Mapping[str, Any]],
    groups: Mapping[str, str],
    *,
    candidate: str,
    seed: int,
    valid_fraction: float,
) -> dict[str, str]:
    group_domains: dict[str, set[str]] = defaultdict(set)
    for digest, record in records.items():
        group_domains[groups[digest]].update(record["domains"])
    domain_groups: dict[str, set[str]] = defaultdict(set)
    for group, domains in group_domains.items():
        for domain in domains:
            domain_groups[domain].add(group)

    ordered = sorted(group_domains, key=lambda group: (_rank(seed, candidate, group), group))
    valid: set[str] = set()

    def can_add(group: str) -> bool:
        return all(len(domain_groups[domain] - (valid | {group})) >= 1 for domain in group_domains[group])

    required = {domain for domain, members in domain_groups.items() if len(members) >= 2}

    def cover() -> bool:
        covered = {domain for group in valid for domain in group_domains[group]}
        missing = required - covered
        if not missing:
            return True
        candidates_by_domain = {
            domain: [
                group
                for group in ordered
                if group not in valid and domain in group_domains[group] and can_add(group)
            ]
            for domain in missing
        }
        domain = min(missing, key=lambda item: (len(candidates_by_domain[item]), item))
        choices = sorted(
            candidates_by_domain[domain],
            key=lambda group: (
                -len(group_domains[group] & missing),
                _rank(seed, candidate, group),
                group,
            ),
        )
        for group in choices:
            valid.add(group)
            if cover():
                return True
            valid.remove(group)
        return False

    if not cover():
        _fail("domain_stratification_impossible")

    target = max(len(valid), round(len(ordered) * valid_fraction))
    for group in ordered:
        if len(valid) >= target:
            break
        if group not in valid and can_add(group):
            valid.add(group)
    return {group: ("valid" if group in valid else "train") for group in ordered}


def _prepare_records(bundle: SourceBundle) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    records: dict[str, dict[str, Any]] = {}
    filtered: dict[str, int] = defaultdict(int)
    for ordinal, source in enumerate(bundle.records):
        if source.domain not in CORE_DOMAINS | {"crsm"}:
            _fail(f"unknown_domain:{source.domain}")
        if (
            source.binding_key not in bundle.bindings
            or not source.source_key
            or isinstance(source.source_index, bool)
            or not isinstance(source.source_index, int)
            or source.source_index < 0
        ):
            _fail(f"source_provenance_invalid:{ordinal}")
        messages = _canonical_messages(source.messages, role=f"source_record_{ordinal}")
        if _contains_stale_model_claim(messages):
            filtered[source.domain] += 1
            continue
        digest = conversation_sha256(messages)
        existing = records.get(digest)
        if existing is None:
            records[digest] = {
                "messages": list(messages),
                "domains": {source.domain},
                "provenance": {
                    (source.binding_key, source.source_key, source.source_index)
                },
                "user_turns": _user_turns(messages),
            }
        else:
            existing["domains"].add(source.domain)
            existing["provenance"].add(
                (source.binding_key, source.source_key, source.source_index)
            )
            if canonical_json_bytes(list(messages)) < canonical_json_bytes(existing["messages"]):
                existing["messages"] = list(messages)
    if not records:
        _fail("kernel_empty")
    present = {domain for record in records.values() for domain in record["domains"]}
    expected_domains = CORE_DOMAINS | (
        {"crsm"} if any(record.domain == "crsm" for record in bundle.records) else set()
    )
    missing = expected_domains - present
    if missing:
        _fail(f"required_domains_missing:{','.join(sorted(missing))}")
    return records, dict(sorted(filtered.items()))


def _jsonl_bytes(values: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(value) + b"\n" for value in values)


def _write(path: Path, payload: bytes) -> None:
    with local_internal_governed_scope(
        "candidate_cortex_kernel.write", domain="file_write"
    ):
        get_file_write_gateway().write_bytes(
            path,
            payload,
            source="candidate_cortex_kernel.write",
        )


def _ensure_directory(path: Path) -> Path:
    with local_internal_governed_scope(
        "candidate_cortex_kernel.ensure_directory", domain="file_write"
    ):
        created = get_file_write_gateway().ensure_directory(
            path,
            source="candidate_cortex_kernel.ensure_directory",
        )
    return Path(created)


def _publish_directory(staging: Path, destination: Path) -> None:
    with local_internal_governed_scope(
        "candidate_cortex_kernel.publish", domain="file_write"
    ):
        get_file_write_gateway().move_path(
            staging,
            destination,
            source="candidate_cortex_kernel.publish",
        )


def _discard_staging(path: Path) -> None:
    with local_internal_governed_scope(
        "candidate_cortex_kernel.cleanup", domain="file_write"
    ):
        get_file_write_gateway().delete_path(
            path,
            recursive=True,
            source="candidate_cortex_kernel.cleanup",
        )


def _output_binding(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    relative = resolved.relative_to(root.resolve(strict=True))
    binding = file_binding(resolved)
    binding.pop("path")
    return {"path": relative.as_posix(), **binding}


def _domain_statistics(
    records: Mapping[str, Mapping[str, Any]],
    groups: Mapping[str, str],
    assignments: Mapping[str, str],
    filtered: Mapping[str, int],
    source_records: Sequence[SourceRecord],
) -> dict[str, dict[str, Any]]:
    domains = sorted({record.domain for record in source_records} | CORE_DOMAINS)
    result: dict[str, dict[str, Any]] = {}
    for domain in domains:
        digests = sorted(digest for digest, record in records.items() if domain in record["domains"])
        domain_groups = sorted({groups[digest] for digest in digests})
        train = [digest for digest in digests if assignments[groups[digest]] == "train"]
        valid = [digest for digest in digests if assignments[groups[digest]] == "valid"]
        result[domain] = {
            "source_records": sum(1 for record in source_records if record.domain == domain),
            "stale_model_records_filtered": int(filtered.get(domain, 0)),
            "unique_conversations": len(digests),
            "groups": len(domain_groups),
            "train": len(train),
            "valid": len(valid),
            "conversation_set_sha256": document_sha256(digests),
            "train_set_sha256": document_sha256(train),
            "valid_set_sha256": document_sha256(valid),
        }
    return result


def _receipt_with_digest(material: dict[str, Any]) -> dict[str, Any]:
    return {**material, "receipt_sha256": document_sha256(material)}


def _read_jsonl(path: Path, *, role: str) -> list[Any]:
    try:
        lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise CandidateCortexKernelError(f"{role}_unreadable") from exc
    if not lines:
        _fail(f"{role}_empty")
    return [_strict_json(line, role=f"{role}_line_{index + 1}") for index, line in enumerate(lines)]


def verify_candidate_cortex_kernel(
    receipt_path: Path,
    *,
    expected_descriptor_sha256: str | None = None,
    verify_inputs: bool = True,
) -> dict[str, Any]:
    """Recompute the published kernel's integrity and semantic invariants."""

    receipt_path = receipt_path.expanduser().resolve(strict=True)
    receipt_value = _strict_json(receipt_path.read_bytes(), role="kernel_receipt")
    if not isinstance(receipt_value, Mapping) or receipt_value.get("schema") != RECEIPT_SCHEMA:
        _fail("receipt_schema_invalid")
    receipt = dict(receipt_value)
    claimed = receipt.pop("receipt_sha256", None)
    if claimed != document_sha256(receipt):
        _fail("receipt_digest_invalid")
    receipt["receipt_sha256"] = claimed
    candidate = receipt.get("candidate")
    descriptor_sha = candidate.get("descriptor_sha256") if isinstance(candidate, Mapping) else None
    if expected_descriptor_sha256 and descriptor_sha != expected_descriptor_sha256:
        _fail("receipt_candidate_mismatch")
    generation_root = receipt_path.parent.resolve(strict=True)
    if receipt.get("generation_root") != str(generation_root):
        _fail("receipt_generation_root_mismatch")

    inputs = receipt.get("inputs")
    outputs = receipt.get("outputs")
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        _fail("receipt_bindings_invalid")
    if verify_inputs:
        for name, binding in inputs.items():
            if not isinstance(binding, Mapping):
                _fail(f"receipt_input_invalid:{name}")
            if file_binding(Path(str(binding.get("path")))) != dict(binding):
                _fail(f"receipt_input_stale:{name}")
    for name, binding in outputs.items():
        if not isinstance(binding, Mapping):
            _fail(f"receipt_output_invalid:{name}")
        relative = Path(str(binding.get("path")))
        if relative.is_absolute() or ".." in relative.parts:
            _fail(f"receipt_output_path_invalid:{name}")
        path = (generation_root / relative).resolve(strict=True)
        if not path.is_relative_to(generation_root):
            _fail(f"receipt_output_path_escape:{name}")
        if _output_binding(path, generation_root) != dict(binding):
            _fail(f"receipt_output_mismatch:{name}")

    train_rows = _read_jsonl(generation_root / "data/train.jsonl", role="kernel_train")
    valid_rows = _read_jsonl(generation_root / "data/valid.jsonl", role="kernel_valid")
    provenance_rows = _read_jsonl(
        generation_root / "data/provenance.jsonl", role="kernel_provenance"
    )
    conversations: dict[str, dict[str, Any]] = {}
    split_by_digest: dict[str, str] = {}
    for split, rows in (("train", train_rows), ("valid", valid_rows)):
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != {"messages"}:
                _fail(f"{split}_row_{index}_schema_invalid")
            messages = _canonical_messages(row["messages"], role=f"{split}_row_{index}")
            if _contains_stale_model_claim(messages):
                _fail("stale_model_claim_present")
            digest = conversation_sha256(messages)
            if digest in conversations:
                _fail("conversation_duplicate_or_split_leakage")
            conversations[digest] = {
                "messages": list(messages),
                "user_turns": _user_turns(messages),
            }
            split_by_digest[digest] = split

    provenance: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(provenance_rows):
        if not isinstance(row, Mapping) or row.get("schema") != PROVENANCE_SCHEMA:
            _fail(f"provenance_row_{index}_schema_invalid")
        digest = row.get("conversation_sha256")
        if not isinstance(digest, str) or digest in provenance:
            _fail("provenance_conversation_invalid")
        provenance[digest] = row
    if set(provenance) != set(conversations):
        _fail("provenance_coverage_mismatch")

    reconstructed = {
        digest: {
            **record,
            "domains": set(provenance[digest].get("domains", ())),
        }
        for digest, record in conversations.items()
    }
    groups = _component_groups(reconstructed)
    for digest, row in provenance.items():
        if row.get("split") != split_by_digest[digest]:
            _fail("provenance_split_mismatch")
        if row.get("group_key") != groups[digest]:
            _fail("provenance_group_mismatch")
        domains = row.get("domains")
        sources = row.get("sources")
        if (
            not isinstance(domains, list)
            or not domains
            or domains != sorted(set(domains))
            or not isinstance(sources, list)
            or not sources
        ):
            _fail("provenance_details_invalid")
        for source in sources:
            if not isinstance(source, Mapping) or set(source) != {
                "source_binding",
                "source_index",
                "source_key",
            }:
                _fail("provenance_source_invalid")
            source_binding = source.get("source_binding")
            if source_binding not in inputs or source_binding == "candidate_descriptor":
                _fail("provenance_source_binding_invalid")

    train_groups = {groups[digest] for digest, split in split_by_digest.items() if split == "train"}
    valid_groups = {groups[digest] for digest, split in split_by_digest.items() if split == "valid"}
    if train_groups & valid_groups:
        _fail("group_split_leakage")

    domain_stats = receipt.get("domains")
    observed_domains = {
        domain for row in provenance.values() for domain in row.get("domains", ())
    }
    if (
        not isinstance(domain_stats, Mapping)
        or CORE_DOMAINS - set(domain_stats)
        or set(domain_stats) != observed_domains
    ):
        _fail("required_domains_missing")
    for domain, stats in domain_stats.items():
        if not isinstance(stats, Mapping):
            _fail(f"domain_stats_invalid:{domain}")
        digests = sorted(
            digest for digest, row in provenance.items() if domain in row.get("domains", ())
        )
        domain_groups = {groups[digest] for digest in digests}
        train = [digest for digest in digests if split_by_digest[digest] == "train"]
        valid = [digest for digest in digests if split_by_digest[digest] == "valid"]
        expected = {
            "unique_conversations": len(digests),
            "groups": len(domain_groups),
            "train": len(train),
            "valid": len(valid),
            "conversation_set_sha256": document_sha256(digests),
            "train_set_sha256": document_sha256(train),
            "valid_set_sha256": document_sha256(valid),
        }
        for key, value in expected.items():
            if stats.get(key) != value:
                _fail(f"domain_stats_mismatch:{domain}:{key}")
        if len(domain_groups) >= 2 and (not train or not valid):
            _fail(f"domain_split_representation_missing:{domain}")
    source_sha256 = {
        name: binding.get("sha256")
        for name, binding in sorted(inputs.items())
        if name != "candidate_descriptor"
    }
    if receipt.get("content_sha256") != document_sha256(
        {
            "candidate_descriptor_sha256": descriptor_sha,
            "conversations": sorted(conversations),
            "provenance": [provenance[digest] for digest in sorted(provenance)],
            "source_sha256": source_sha256,
            "split_seed": receipt.get("parameters", {}).get("split_seed"),
            "valid_fraction": receipt.get("parameters", {}).get("valid_fraction"),
        }
    ):
        _fail("content_address_mismatch")
    return receipt


def build_candidate_cortex_kernel(
    *,
    descriptor_path: Path,
    expected_descriptor_sha256: str,
    output_root: Path,
    source_repo_root: Path,
    crsm_path: Path | None = None,
    valid_fraction: float = DEFAULT_VALID_FRACTION,
    split_seed: int = DEFAULT_SPLIT_SEED,
    source_bundle: SourceBundle | None = None,
) -> dict[str, Any]:
    """Build and atomically publish one compact candidate persona kernel."""

    if not 0.0 < valid_fraction < 0.5 or split_seed < 0:
        _fail("split_parameters_invalid")
    descriptor_path = descriptor_path.expanduser().resolve(strict=True)
    descriptor, descriptor_binding = _descriptor(descriptor_path, expected_descriptor_sha256)
    bundle = source_bundle or load_curated_source_bundle(source_repo_root, crsm_path=crsm_path)
    before = {name: dict(binding) for name, binding in bundle.bindings.items()}
    if not before:
        _fail("source_bindings_missing")
    if not _source_bindings_equal(before, _refresh_bindings(before)):
        _fail("source_binding_stale_before_build")
    records, filtered = _prepare_records(bundle)
    groups = _component_groups(records)
    assignments = _split_groups(
        records,
        groups,
        candidate=expected_descriptor_sha256,
        seed=split_seed,
        valid_fraction=valid_fraction,
    )
    for record in records.values():
        record["domains"] = sorted(record["domains"])
        record["provenance"] = sorted(
            (
                {
                    "source_binding": source_binding,
                    "source_key": source_key,
                    "source_index": source_index,
                }
                for source_binding, source_key, source_index in record["provenance"]
            ),
            key=lambda value: (
                value["source_binding"],
                value["source_key"],
                value["source_index"],
            ),
        )

    train = [
        {"messages": records[digest]["messages"]}
        for digest in sorted(records)
        if assignments[groups[digest]] == "train"
    ]
    valid = [
        {"messages": records[digest]["messages"]}
        for digest in sorted(records)
        if assignments[groups[digest]] == "valid"
    ]
    if not train or not valid:
        _fail("kernel_split_empty")
    provenance = [
        {
            "schema": PROVENANCE_SCHEMA,
            "conversation_sha256": digest,
            "group_key": groups[digest],
            "split": assignments[groups[digest]],
            "domains": records[digest]["domains"],
            "sources": records[digest]["provenance"],
        }
        for digest in sorted(records)
    ]
    parameters = {"split_seed": split_seed, "valid_fraction": valid_fraction}
    content_material = {
        "candidate_descriptor_sha256": expected_descriptor_sha256,
        "conversations": sorted(records),
        "provenance": provenance,
        "source_sha256": {
            name: binding["sha256"] for name, binding in sorted(before.items())
        },
        **parameters,
    }
    content_sha = document_sha256(content_material)
    domains = _domain_statistics(records, groups, assignments, filtered, bundle.records)
    inputs = {"candidate_descriptor": descriptor_binding, **before}

    root = _ensure_directory(output_root.expanduser().resolve(strict=False))
    candidate_root = _ensure_directory(root / expected_descriptor_sha256[:16])
    generation_root = candidate_root / content_sha
    lock_path = candidate_root / ".candidate_cortex_kernel.lock"
    with interprocess_file_lock(lock_path):
        receipt_path = generation_root / "candidate_cortex_kernel_receipt.json"
        if receipt_path.is_file():
            receipt = verify_candidate_cortex_kernel(
                receipt_path,
                expected_descriptor_sha256=expected_descriptor_sha256,
            )
            return {**receipt, "resumed": True}
        if generation_root.exists():
            _fail("existing_generation_incomplete")
        staging = candidate_root / f".{content_sha}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            data_root = staging / "data"
            _ensure_directory(data_root)
            train_path = data_root / "train.jsonl"
            valid_path = data_root / "valid.jsonl"
            provenance_path = data_root / "provenance.jsonl"
            _write(train_path, _jsonl_bytes(train))
            _write(valid_path, _jsonl_bytes(valid))
            _write(provenance_path, _jsonl_bytes(provenance))
            after = _refresh_bindings(before)
            if not _source_bindings_equal(before, after):
                _fail("source_changed_during_build")
            outputs = {
                "train": _output_binding(train_path, staging),
                "valid": _output_binding(valid_path, staging),
                "provenance": _output_binding(provenance_path, staging),
            }
            receipt_material = {
                "schema": RECEIPT_SCHEMA,
                "candidate": {
                    "descriptor_sha256": expected_descriptor_sha256,
                    "repository_id": descriptor.get("repository_id"),
                    "revision": descriptor.get("revision"),
                },
                "content_sha256": content_sha,
                "generation_root": str(generation_root),
                "parameters": parameters,
                "direct_quotes_mode": bundle.direct_quotes_mode,
                "inputs": inputs,
                "outputs": outputs,
                "domains": domains,
                "filters": {
                    "stale_model_claim_patterns": [pattern.pattern for pattern in _STALE_MODEL_PATTERNS],
                    "stale_model_records_filtered": sum(filtered.values()),
                    **dict(sorted((bundle.filter_counts or {}).items())),
                },
                "counts": {
                    "source_records": len(bundle.records),
                    "unique_conversations": len(records),
                    "train": len(train),
                    "valid": len(valid),
                },
                "invariants": {
                    "candidate_bound": True,
                    "content_addressed": True,
                    "atomic_publication": True,
                    "system_messages": 0,
                    "group_split_overlap": 0,
                    "legacy_human_or_movie_corpus_included": False,
                    "system_prompt_variants_included": False,
                },
            }
            receipt = _receipt_with_digest(receipt_material)
            _write(staging / "candidate_cortex_kernel_receipt.json", canonical_json_bytes(receipt) + b"\n")
            _publish_directory(staging, generation_root)
            verified = verify_candidate_cortex_kernel(
                generation_root / "candidate_cortex_kernel_receipt.json",
                expected_descriptor_sha256=expected_descriptor_sha256,
            )
            return {**verified, "resumed": False}
        except BaseException:
            _discard_staging(staging)
            raise


__all__ = [
    "CORE_DOMAINS",
    "DEFAULT_SPLIT_SEED",
    "DEFAULT_VALID_FRACTION",
    "CandidateCortexKernelError",
    "SourceBundle",
    "SourceRecord",
    "build_candidate_cortex_kernel",
    "canonical_json_bytes",
    "conversation_sha256",
    "document_sha256",
    "file_binding",
    "load_curated_source_bundle",
    "verify_candidate_cortex_kernel",
]
