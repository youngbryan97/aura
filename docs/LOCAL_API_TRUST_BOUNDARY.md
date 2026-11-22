# Aura Local API Browser Trust Boundary

Status: Checkpoint 88 is pushed on `main` at `5a56bea5`. The reported
tokenless DNS-rebinding bypass is covered by deterministic regressions. This
document defines one bounded interface control; it is not a claim that the
complete `SECURITY-001` program is closed.

## Threat Model

Aura may listen on loopback or all local interfaces. A browser visiting an
attacker-controlled origin can later resolve that origin to loopback or a LAN
address. The resulting TCP peer can therefore be local even though the browser
authority remains attacker-controlled. The following values are not standalone
authentication:

- `request.client.host` being loopback;
- `Host` and `Origin` containing equal attacker-controlled text;
- `Sec-Fetch-Site: same-origin` after DNS rebinding;
- `X-Aura-Surface` or `X-Aura-Desktop-Request`;
- a public health or pairing path;
- `internal_only_mode`.

The boundary assumes a valid master API token remains secret and a paired
device token is independently authenticated, scoped, revocable, and protected
by the paired-device path allowlist.

## Required Invariants

1. A tokenless loopback request is owner-capable only when the socket peer is
   loopback and the HTTP `Host` authority names an exact built-in loopback host.
2. Trusted host names are exactly `localhost`, `127.0.0.1`, and `::1`.
   Resolution is never used to expand this set.
3. Missing, duplicate, malformed, user-info-bearing, path-bearing, ambiguous,
   or out-of-range authorities fail closed.
4. DNS aliases, trailing-dot names, integer/octal/hex IPv4 forms, wildcard
   addresses, IPv4-mapped IPv6, and loopback-looking suffixes are not trusted.
5. A browser `Origin` must be an allowed local development origin or match the
   request's literal local host, port, and transport scheme.
6. `ws` is compared with an HTTP origin and `wss` with an HTTPS origin.
7. The host check runs before health/pairing exemptions, internal-only return,
   CSRF exceptions, desktop markers, and tokenless local access.
8. Owner access-profile classification and route-local trust decisions must use
   the same boundary as middleware.
9. A WebSocket is not registered with the broadcast manager before successful
   local-origin, master-token, or paired-device authentication.
10. A paired-device credential never widens its deny-by-default path scope.

## Accepted Paths

| Principal | Peer and authority requirement | Credential | Maximum surface |
|---|---|---|---|
| Local owner UI | loopback peer plus trusted literal Host and allowed Origin context | tokenless local policy | owner surface |
| Master automation | deployment-dependent peer/Host | exact API token | owner surface |
| Paired LAN device | LAN or loopback transport with valid device credential; browser origins must match the request authority | paired token or session cookie | conversation/read-only allowlist |
| Unpaired LAN visitor | LAN transport | pairing ceremony only | public health/pairing paths |
| Unknown browser/site | any | none | denied outside intentionally public paths; rebound loopback Host is denied even on public paths |

## HTTP Decision Order

`validate_runtime_security_request()` evaluates the boundary in this order:

1. Parse the configured master token, supplied token, paired-device credential,
   request path, peer, Host values, scheme, and browser metadata.
2. Reject a loopback peer with an untrusted Host unless a real master or paired
   credential already authenticates the principal.
3. Reject hostile browser origin metadata. Authenticated paired/public LAN
   requests may use a strict same-request authority match; tokenless owner
   requests may not use an arbitrary same-text host.
4. Apply internal-only restrictions.
5. Apply intentionally public health/pairing paths.
6. Apply master-token and paired-device authorization.
7. Apply the bounded tokenless local-owner policy.
8. Deny everything else.

This ordering prevents a later exception from converting transport locality or
attacker-controlled headers into authentication.

## Proxy Contract

A reverse proxy does not become trusted merely because it runs on loopback.
Production proxy deployments should terminate and rewrite `Host`, sanitize
forwarded headers, preserve the effective scheme, and require the master token
for non-loopback authorities. Aura deliberately allows a correctly supplied
master token across a proxy authority; the browser cannot obtain that exception
without the credential.

A complete supported-proxy manifest, proxy-specific integration suite, and
deployment hardening guide remain open under `SECURITY-001` and
`COMPATIBILITY-001`.

## Regression Evidence

`tests/test_dns_rebinding_auth.py` covers 38 adversarial and preservation cases,
including:

- the published memory-export and settings-PATCH requests;
- no-Origin and forged-desktop-marker variants;
- loopback and LAN-address rebinding;
- health and pairing exception ordering;
- internal-only and route-dependency paths;
- missing and duplicate Host headers;
- malformed ports, user info, paths, queries, fragments, and scheme mismatch;
- loopback aliases and alternate numeric encodings;
- IPv4, IPv6, local development Origin, and secure WebSocket preservation;
- owner-profile non-escalation and valid master-token proxy access.

The expanded adjacent suite passes `95/95` across runtime security, paired
devices, audit security, and WebSocket source contracts. The CP87 settings and
governance suite separately passes `139/139`. Python compilation, Ruff
`F/I/B025`, JavaScript syntax, and tracked-diff hygiene pass on the integrated
tree.

## Remaining Security Work

Checkpoint 88 does not close:

- independent penetration testing and parser/API fuzzing;
- CSP and the complete browser response-header policy;
- supported reverse-proxy configuration and forwarded-header attestation;
- secret storage, rotation, revocation, and compromise recovery;
- complete privacy/data-lifecycle and authenticated export controls;
- dependency scanning, SBOM, license policy, provenance, reproducible builds,
  signing, notarization, and update-channel security;
- hostile prompt, tool, desktop, model, plugin, and supply-chain campaigns;
- clean-machine, installed-app, network-partition, and sustained adversarial
  live proof.

Those remain release-blocking in `SECURITY-001`; the narrow bypass is fixed
without relabeling the entire security program complete.
