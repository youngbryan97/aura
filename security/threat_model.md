# Aura Threat Model

## 1. System Overview

Aura is a locally-deployed autonomous AI cognitive runtime with:
- Local LLM inference (MLX on Apple Silicon)
- Optional cloud model fallback
- Tool/skill execution (filesystem, shell, browser, network)
- Persistent memory (SQLite, vector DB)
- Autonomous background behavior (maintenance, learning, self-repair)
- API server surface (local REST/WebSocket)
- Desktop GUI (native macOS)

## 2. Trust Boundaries

| Boundary | Description | Controls |
|----------|-------------|----------|
| **B1: User ↔ Runtime** | User input is untrusted text | Input sanitization, prompt injection defense |
| **B2: Runtime ↔ Tools** | Tool execution outputs are untrusted | Output sanitization, sandbox, timeout |
| **B3: Runtime ↔ Filesystem** | Workspace access must be bounded | Permission matrix, path validation |
| **B4: Runtime ↔ Network** | External content is untrusted | TLS, content filtering, privacy controls |
| **B5: Runtime ↔ Model** | Model outputs are probabilistic, not trusted | Integrity checks, role-continuation defense |
| **B6: Runtime ↔ Memory** | Stored state can be poisoned | Write gating via Will, integrity hashes |
| **B7: Runtime ↔ Cloud** | Cloud provider can observe prompts | Privacy classification, opt-in policy |

## 3. Threat Catalog

### 3.1 AI-Agent-Specific Threats (MITRE ATLAS-aligned)

| ID | Threat | ATLAS Technique | Severity | Control | Status |
|----|--------|-----------------|----------|---------|--------|
| T01 | **Direct prompt injection** | AML.T0051 | Critical | Input sanitizer strips control tokens; integrity check on output | ✅ Mitigated |
| T02 | **Indirect prompt injection** (from files/web/tool output) | AML.T0051.001 | Critical | Tool output sanitization; content boundary markers; Will gating on actions derived from tool output | ✅ Mitigated |
| T03 | **Malicious memory poisoning** | AML.T0018 | High | All memory writes gated by Will; integrity hashes on stored state; memory audit trail | ✅ Mitigated |
| T04 | **Tool-result spoofing** | AML.T0043 | High | Tool results are sandboxed untrusted input; cross-validation for critical actions | ✅ Mitigated |
| T05 | **Model-provider compromise** | AML.T0042 | High | Inference is local only. Every lane the router can fall back to — Cortex, Solver, Brainstem, Reflex — is a local model; `allow_cloud_fallback` is coerced to `False` in `core/brain/request_contract.py` whatever a caller passes | ✅ Mitigated |
| T06 | **Model denial of service** | AML.T0029 | Medium | Resource governor; token budgets; Metal semaphore; timeout enforcement | ✅ Mitigated |
| T07 | **Excessive agency** | OWASP-LLM-08 | Critical | Unified Will; AuthorityGateway; operator permission matrix; fail-closed defaults | ✅ Mitigated |
| T08 | **Unsafe self-modification** | AML.T0044 | Critical | Self-modification disabled in production mode; gated by feature flag + Will | ✅ Mitigated |
| T09 | **Capability-token leakage** | - | High | Secrets never included in model context; env var isolation; log sanitization | ✅ Mitigated |
| T10 | **Workspace escape** | - | Critical | Path validation; sandbox chroot; permission matrix bounds workspace | ✅ Mitigated |
| T11 | **Data exfiltration through generated output** | AML.T0048 | High | Outbound requests pass `core/runtime/network_gateway.py`; `core/security/egress_privacy.py` reads the body before it leaves — credentials never go anywhere, personal identifiers only to a destination that is not a stranger | ✅ Mitigated |
| T12 | **Secret leakage through logs** | - | High | Log sanitizer; structured logging; no secrets in telemetry | ✅ Mitigated |
| T13 | **Unauthorized memory writes** | - | High | Will receipt required; governance lint enforces coverage | ✅ Mitigated |
| T14 | **Corrupted long-term identity/state** | - | High | State snapshots; backup/restore; integrity verification on boot | ✅ Mitigated |
| T15 | **Malicious plugin/skill loading** | - | High | Skill manifest + signature required in production; sandbox policy per skill | ✅ Mitigated |

### 3.2 Traditional Application Threats

| ID | Threat | Severity | Control | Status |
|----|--------|----------|---------|--------|
| T20 | Unauthenticated API access | Medium | Local-only binding; API key for remote mode | ✅ Mitigated |
| T21 | Dependency supply-chain attack | High | Pinned hashes; SBOM; pip-audit; provenance | ✅ Mitigated |
| T22 | Container escape (Docker mode) | Medium | Minimal base image; non-root user; read-only fs | ✅ Mitigated |
| T23 | Denial of service (local) | Low | Resource governor; token budgets; process isolation | ✅ Mitigated |
| T24 | Log injection | Low | Structured logging; parameterized log messages | ✅ Mitigated |
| T25 | Path traversal | Medium | Path validation; workspace boundary enforcement | ✅ Mitigated |

## 4. Threat-Capability Matrix

Every major Aura capability has a threat, a test, a runtime guard, and an incident runbook:

| Capability | Primary Threat | Runtime Guard | Test | Runbook |
|-----------|---------------|---------------|------|---------|
| Chat/conversation | T01, T02 | `ContextGuard` quarantines untrusted content; canaries detect a followed instruction | `tests/test_prompt_sanitizer_cp126.py`, `tests/test_injection_canary.py` | `docs/runbooks/prompt-injection.md` |
| Tool execution | T04, T07, T10 | Sandbox + Will + permission matrix | `tests/test_sandbox_runner_hardening.py` | `docs/runbooks/tool-timeout-storm.md` |
| Memory write | T03, T13 | Will receipt + integrity hash | `tests/test_will_gate_for_plastic_updates.py` | `docs/runbooks/memory-corruption.md` |
| Autonomous action | T07, T08 | Will + AuthorityGateway + feature flag | `tests/test_autonomy_latitude.py` | `docs/runbooks/excessive-agency.md` |
| Local inference boundary | T05 | Request contract pins `allow_cloud_fallback` to `False` | `tests/test_request_contract.py` | `docs/runbooks/local-inference-boundary.md` |
| External interlocutor | T11 | Host allowlist + per-run turn budget + egress body inspection | `tests/test_web_interlocutor_outbound_safety.py` | `docs/runbooks/external-egress.md` |
| Self-repair | T08 | Same Will path as all actions | `tests/test_self_repair_backlog.py` | `docs/runbooks/self-repair-failed.md` |
| Model loading | T05, T06 | Identity measured from the safetensors index, not from a path substring; lane ownership released on load failure | `tests/test_mlx_client_admission_integrity.py`, `tests/test_model_loader_consolidation.py` | `docs/runbooks/model-fails-to-load.md` |
| Plugin/skill loading | T15 | Manifest + signature + sandbox | `tests/test_plugin_allowlist.py` | `docs/runbooks/tool-timeout-storm.md` |
| Backup/restore | T14 | State hash verification + audit trail | `tests/test_restore_drill.py` | `docs/runbooks/disaster-recovery.md` |

## 5. Residual Risks

| Risk | Severity | Mitigation Plan |
|------|----------|----------------|
| Novel prompt injection techniques | Medium | Ongoing defense updates; model-level alignment; output integrity checks |
| Local physical access | Low | Deferred to OS-level encryption (FileVault) |
| Adversarial model weights | Low | Model checksum verification; trusted source policy |
| Zero-day in MLX/Python | Low | Dependency monitoring; rapid update path |

## 6. Review Schedule

- Threat model review: every major release
- Penetration test: annually
- Dependency audit: weekly (automated)
- Incident review: per incident, within 72 hours
