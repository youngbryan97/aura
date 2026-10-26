# Aura permission and authority matrix

Status: Reference · Implemented boundaries reconciled 25 October 2026

Aura’s current platform posture is a local, single-operator installation.
The earlier User/Operator/Admin/Research table described a proposed role
hierarchy; it did not correspond to an implemented multi-user RBAC system.
A runtime mode is a capability posture, not an authenticated user role.
See [platform posture](../docs/PLATFORM_POSTURE.md).

## Distinguish the authorities

| Boundary | Actual owner | What it establishes | What it does not establish |
|---|---|---|---|
| Local HTTP/WebSocket access | [Local API boundary](../docs/LOCAL_API_TRUST_BOUNDARY.md) | Checks peer, literal host/origin and supported credentials under the declared policy | General trust from loopback peer or a UI header alone |
| Master API credential | [Request-security code](../interface/server.py) and route checks | Authenticates the configured owner surface where accepted | Per-user tenant isolation or an admin/operator role hierarchy |
| Paired device | [Device routes](../interface/routes/devices.py), [pairing registry](../core/security/device_pairing.py) | Stored, scoped and revocable device access | An unrestricted grant to every owner endpoint |
| Consequential effect | [Authority gateway](../core/executive/authority_gateway.py), [Unified Will](../core/governance/will.py) | Required action-authorisation and outcome contract | Semantic correctness of the proposed task or a universal no-bypass proof |
| Runtime mode | [Mode manifests](../core/runtime/mode.py) | Declared capability/autonomy ceiling for the process | Immediate reconfiguration from an export in another shell |
| Process spawn role | [Process privileges](../core/runtime/process_privilege.py) | Declared admission constraints for supported source roles | An OS sandbox, per-user RBAC or complete coverage of unknown roles |
| Standing denial | [Standing directives](../core/governance/standing_directives.py) | Refuses matching tools/paths on consuming routes | A permission grant or proof every possible effect route consults it |
| Model request | [Local request contract](../core/brain/request_contract.py) | Local inference routing; remote-only model requests are refused | A switch enabling cloud-model fallback |

## Ordinary operations

Conversation, memory export/deletion, model settings, diagnostics and device
administration have route-specific access checks. Do not infer permission
from a column in an aspirational table. Inspect the route and test its
accepted and denied principals against the installed revision.

For example, a paired phone may be allowed conversation access while being
refused a memory-management or administrative path. The paired-device token’s
scope must remain authoritative at that route; asking in chat cannot expand it.
An owner’s authority to request an action also does not establish that the
requested action is safe, permitted by standing directives or successfully done.

## Configuration and live controls

The process reads its own environment and supported settings stores.
`AURA_MODE`, `AURA_AUTONOMY_LEVEL` and `AURA_FOREGROUND_ONLY` affect their
consumers; assigning them in an unrelated shell does not alter a live Aura
process. Use a supported live setting where implemented, otherwise an
authorised future launch. Follow [the override policy](../HUMAN_OVERRIDE_POLICY.md).

Standing directives are deny rules, not a grant mechanism. A directive can
narrow an action’s authority; it cannot by itself mint device credentials,
change an OS permission, bypass a sandbox or authorise remote inference.

## Verify the boundary you intend to claim

Keep the source revision, principal, request, decision and resulting effect
in the evidence. Verify refused requests never reach their protected effect,
and that permitted recovery remains usable. Static lint and a policy table
do not establish the outcome of every live request.

The project has not established a multi-tenant RBAC or OIDC/SAML deployment.
Adding one would require a new identity, storage and per-route authorisation
contract and its own acceptance evidence.
