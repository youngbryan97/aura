"""Governed Aura workspace facade.

``MarkdownWorkspace`` is the durable storage engine.  ``AuraWorkspace`` is the
architecture surface Aura should use: canonical mind-filesystem paths, policy
checks, capability-token consumption, receipts, and commit metadata.
"""
from __future__ import annotations

import posixpath
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from core.runtime.errors import record_degradation
from core.workspace.markdown_workspace import MarkdownWorkspace, WorkspaceCommit

CANONICAL_DIRECTORIES: tuple[str, ...] = (
    "aura/identity",
    "aura/observations/video",
    "aura/observations/browser",
    "aura/observations/terminal",
    "aura/observations/user_interactions",
    "aura/scratchpads/active",
    "aura/scratchpads/completed",
    "aura/decisions/approved",
    "aura/decisions/rejected",
    "aura/decisions/deferred",
    "aura/initiatives/proposed",
    "aura/initiatives/in_progress",
    "aura/initiatives/finished",
    "aura/repairs/proposed_patches",
    "aura/repairs/applied_patches",
    "aura/repairs/reverted_patches",
    "aura/runs/video_analysis",
    "aura/runs/self_tests",
    "aura/runs/code_audits",
    "aura/memory/episodic",
    "aura/memory/semantic",
    "aura/memory/procedural",
)


@dataclass(frozen=True)
class WorkspacePolicyDecision:
    approved: bool
    reason: str
    authority_receipt_id: str | None = None
    capability_token_id: str | None = None


@dataclass(frozen=True)
class WorkspaceWriteResult:
    path: str
    content_hash: str
    receipt_id: str
    commit_id: str | None = None
    authority_receipt_id: str | None = None
    capability_token_id: str | None = None


class WorkspacePolicy:
    """Least-privilege policy for workspace writes."""

    WRITE_CAPABILITIES = frozenset({"workspace.write", "memory_write", "agent_workspace.write"})

    def authorize_write(
        self,
        *,
        path: str,
        content: str,
        actor: str,
        purpose: str,
        capability_token: str | None = None,
        authority_receipt: str | None = None,
        allow_bootstrap: bool = False,
    ) -> WorkspacePolicyDecision:
        if allow_bootstrap and actor == "system_boot":
            return WorkspacePolicyDecision(True, "bootstrap", authority_receipt, capability_token)

        if authority_receipt:
            return WorkspacePolicyDecision(
                True,
                "authority_receipt_present",
                authority_receipt_id=authority_receipt,
                capability_token_id=capability_token,
            )

        if capability_token:
            return self._authorize_capability(path=path, capability_token=capability_token)

        decision = self._authorize_via_gateway(path=path, content=content, actor=actor, purpose=purpose)
        if decision is not None:
            return decision

        return WorkspacePolicyDecision(
            False,
            "workspace writes require an authority receipt, valid capability token, or available AuthorityGateway",
        )

    def consume_capability(self, token_id: str | None) -> None:
        if not token_id:
            return
        try:
            from core.runtime.capability_tokens import get_capability_token_store

            if not get_capability_token_store().consume(token_id):
                raise PermissionError("capability_token_consume_failed")
        except Exception as exc:
            record_degradation("aura_workspace", exc)
            raise

    def _authorize_capability(self, *, path: str, capability_token: str) -> WorkspacePolicyDecision:
        try:
            from core.runtime.capability_tokens import TokenStatus, get_capability_token_store

            token = get_capability_token_store().get(capability_token)
            if token is None:
                return WorkspacePolicyDecision(False, "capability_token_unknown")
            if token.status is not TokenStatus.ISSUED:
                return WorkspacePolicyDecision(False, f"capability_token_not_issued:{token.status.value}")
            if token.capability not in self.WRITE_CAPABILITIES:
                return WorkspacePolicyDecision(False, f"capability_token_wrong_capability:{token.capability}")
            scope = str(token.scope or "").strip()
            if scope not in {"", "/", "aura", "/aura"}:
                normalized_scope = _normalize_workspace_scope(scope)
                if path != normalized_scope and not path.startswith(f"{normalized_scope}/"):
                    return WorkspacePolicyDecision(False, "capability_token_scope_mismatch")
            return WorkspacePolicyDecision(
                True,
                "capability_token_valid",
                authority_receipt_id=token.receipt_id,
                capability_token_id=capability_token,
            )
        except Exception as exc:
            record_degradation("aura_workspace", exc)
            return WorkspacePolicyDecision(False, f"capability_token_validation_failed:{exc}")

    def _authorize_via_gateway(
        self,
        *,
        path: str,
        content: str,
        actor: str,
        purpose: str,
    ) -> WorkspacePolicyDecision | None:
        try:
            from core.container import ServiceContainer

            gateway = (
                ServiceContainer.get("authority_gateway", default=None)
                or ServiceContainer.get("executive_authority", default=None)
            )
            if gateway is None or not hasattr(gateway, "authorize_memory_write_sync"):
                return None
            decision = gateway.authorize_memory_write_sync(
                "agent_workspace",
                f"{path}\n{content[:500]}",
                source=actor,
                importance=0.65,
                metadata={"path": path, "purpose": purpose},
            )
            if not getattr(decision, "approved", False):
                return WorkspacePolicyDecision(False, getattr(decision, "reason", "authority_denied"))
            receipt = (
                getattr(decision, "will_receipt_id", None)
                or getattr(decision, "substrate_receipt_id", None)
                or getattr(decision, "executive_intent_id", None)
            )
            return WorkspacePolicyDecision(True, "authority_gateway_approved", receipt)
        except Exception as exc:
            record_degradation("aura_workspace", exc)
            return None


class AuraWorkspace:
    """Governed, versioned workspace for Aura's externalized cognition."""

    def __init__(
        self,
        store: MarkdownWorkspace | None = None,
        *,
        policy: WorkspacePolicy | None = None,
        receipt_store: Any | None = None,
    ) -> None:
        self.store = store or MarkdownWorkspace()
        self.policy = policy or WorkspacePolicy()
        self.receipt_store = receipt_store

    def write_artifact(
        self,
        path: str,
        content: str,
        *,
        actor: str,
        purpose: str,
        capability_token: str | None = None,
        authority_receipt: str | None = None,
        commit: bool = True,
        commit_message: str | None = None,
        bookmark: str = "main",
        metadata: dict[str, Any] | None = None,
        allow_bootstrap: bool = False,
    ) -> WorkspaceWriteResult:
        clean_path = _normalize_workspace_path(path)
        metadata = dict(metadata or {})
        policy = self.policy.authorize_write(
            path=clean_path,
            content=content,
            actor=actor,
            purpose=purpose,
            capability_token=capability_token,
            authority_receipt=authority_receipt,
            allow_bootstrap=allow_bootstrap,
        )
        if not policy.approved:
            raise PermissionError(policy.reason)

        node = self.store.write_file(clean_path, content, user="aura")
        receipt_id = self._emit_receipt(
            path=clean_path,
            content=content,
            actor=actor,
            purpose=purpose,
            authority_receipt_id=policy.authority_receipt_id,
            capability_token_id=policy.capability_token_id,
            metadata=metadata,
        )
        commit_id = None
        if commit:
            workspace_commit = self.store.commit(
                commit_message or f"workspace: {purpose} -> {clean_path}",
                author=actor,
                bookmark=bookmark,
                metadata={
                    "path": clean_path,
                    "purpose": purpose,
                    "receipt_id": receipt_id,
                    "authority_receipt_id": policy.authority_receipt_id,
                    "capability_token_id": policy.capability_token_id,
                    **metadata,
                },
            )
            commit_id = workspace_commit.change_id

        self.policy.consume_capability(policy.capability_token_id)
        return WorkspaceWriteResult(
            path=clean_path,
            content_hash=node.content_hash,
            receipt_id=receipt_id,
            commit_id=commit_id,
            authority_receipt_id=policy.authority_receipt_id,
            capability_token_id=policy.capability_token_id,
        )

    def write_video_evidence(
        self,
        session_id: str,
        evidence: str | Iterable[Any],
        *,
        actor: str,
        purpose: str,
        capability_token: str | None = None,
        authority_receipt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceWriteResult:
        safe_session = _safe_segment(session_id)
        return self.write_artifact(
            f"/aura/observations/video/{safe_session}/evidence.md",
            _render_video_evidence(evidence),
            actor=actor,
            purpose=purpose,
            capability_token=capability_token,
            authority_receipt=authority_receipt,
            metadata=metadata,
        )

    def record_run(
        self,
        run_type: str,
        run_id: str,
        report_markdown: str,
        *,
        actor: str,
        purpose: str,
        capability_token: str | None = None,
        authority_receipt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceWriteResult:
        safe_type = _safe_segment(run_type)
        safe_run = _safe_segment(run_id)
        return self.write_artifact(
            f"/aura/runs/{safe_type}/{safe_run}/result.md",
            report_markdown,
            actor=actor,
            purpose=purpose,
            capability_token=capability_token,
            authority_receipt=authority_receipt,
            metadata=metadata,
        )

    def ensure_scaffold(self) -> WorkspaceCommit:
        for directory in CANONICAL_DIRECTORIES:
            readme = f"{directory}/README.md"
            if readme not in self.store.tree("/"):
                title = directory.split("/")[-1].replace("_", " ").title()
                self.store.write_file(readme, f"# {title}\n\nReserved Aura workspace area.\n", user="aura")
        return self.store.commit(
            "Initialize Aura workspace scaffold",
            author="system_boot",
            metadata={"purpose": "bootstrap_workspace", "allow_bootstrap": True},
        )

    def search(self, text: str, *, path: str = "/aura") -> list[dict[str, Any]]:
        return self.store.grep(text, path=path)

    def log(self, *, limit: int = 50) -> list[WorkspaceCommit]:
        return self.store.log(limit=limit)

    def _emit_receipt(
        self,
        *,
        path: str,
        content: str,
        actor: str,
        purpose: str,
        authority_receipt_id: str | None,
        capability_token_id: str | None,
        metadata: dict[str, Any],
    ) -> str:
        from core.runtime.receipts import MemoryWriteReceipt, get_receipt_store

        store = self.receipt_store or get_receipt_store()
        receipt = store.emit(
            MemoryWriteReceipt(
                cause="aura_workspace.write_artifact",
                family="agent_workspace",
                record_id=path,
                bytes_written=len(content.encode("utf-8")),
                governance_receipt_id=authority_receipt_id,
                metadata={
                    "actor": actor,
                    "purpose": purpose,
                    "path": path,
                    "capability_token_id": capability_token_id,
                    **metadata,
                },
            )
        )
        return receipt.receipt_id


def _normalize_workspace_path(path: str) -> str:
    raw = str(path or "").replace("\\", "/").strip()
    if raw.startswith("/"):
        raw = raw[1:]
    clean = posixpath.normpath(raw)
    if clean in {"", "."}:
        raise ValueError("workspace path cannot be empty")
    if clean.startswith("../") or clean == ".." or "/../" in f"/{clean}/":
        raise ValueError(f"path escapes workspace: {path}")
    if not clean.startswith("aura/"):
        clean = f"aura/{clean}"
    if not clean.endswith(".md"):
        raise ValueError("AuraWorkspace artifacts must be Markdown files")
    return clean


def _normalize_workspace_scope(scope: str) -> str:
    raw = str(scope or "").replace("\\", "/").strip()
    if raw in {"", "/", ".", "aura", "/aura"}:
        return "aura"
    if raw.startswith("/"):
        raw = raw[1:]
    clean = posixpath.normpath(raw)
    if clean in {"", "."}:
        return "aura"
    if clean.startswith("../") or clean == ".." or "/../" in f"/{clean}/":
        raise ValueError(f"scope escapes workspace: {scope}")
    if not clean.startswith("aura/") and clean != "aura":
        clean = f"aura/{clean}"
    return clean.rstrip("/")


def _safe_segment(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip())
    return clean.strip("_") or "unnamed"


def _render_video_evidence(evidence: str | Iterable[Any]) -> str:
    if isinstance(evidence, str):
        return evidence
    lines = ["# Video Evidence", ""]
    for item in evidence:
        data = _evidence_to_dict(item)
        start = _format_timestamp(data.get("start_s", data.get("start", 0.0)))
        end = _format_timestamp(data.get("end_s", data.get("end", 0.0)))
        description = str(data.get("description", data.get("text", ""))).strip() or "Evidence"
        confidence = data.get("confidence")
        confidence_text = ""
        if confidence is not None:
            try:
                confidence_text = f" confidence={float(confidence):.2f}"
            except (TypeError, ValueError):
                confidence_text = f" confidence={confidence}"
        source = str(data.get("source", "aura")).strip() or "aura"
        lines.append(f"- `{start}-{end}` {description} (source={source}{confidence_text})")
        payload = data.get("payload") or data.get("metadata")
        if isinstance(payload, Mapping) and payload:
            for key, value in sorted(payload.items()):
                lines.append(f"  - {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def _evidence_to_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, Mapping):
        return dict(item)
    if is_dataclass(item):
        return asdict(item)
    return {
        key: getattr(item, key)
        for key in ("start_s", "end_s", "description", "confidence", "source", "payload")
        if hasattr(item, key)
    }


def _format_timestamp(value: Any) -> str:
    seconds = max(0.0, float(value or 0.0))
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    if millis == 1000:
        whole += 1
        millis = 0
    minutes, sec = divmod(whole, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}:{sec:02d}.{millis:03d}"


__all__ = [
    "AuraWorkspace",
    "CANONICAL_DIRECTORIES",
    "WorkspacePolicy",
    "WorkspacePolicyDecision",
    "WorkspaceWriteResult",
]
