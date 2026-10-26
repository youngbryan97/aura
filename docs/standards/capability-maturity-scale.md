# Capability maturity scale

**Status:** live and observing. `core/runtime/capability_maturity.py`, gated in
`CapabilityEngine.execute`. **Not enforcing** — see Rollout.

## Upstream

| | |
|---|---|
| Project | Home Assistant integration quality scale |
| License | Apache-2.0 |
| Adopted | The idea that an integration carries a graded maturity, and that the grade is a first-class fact rather than a README note |
| Code copied | **None.** The tiers, rules, and gate are Aura's own. |

## The idea adopted

Home Assistant grades every integration Bronze → Silver → Gold → Platinum
against published rules, with explicit per-rule exemptions. "It's in the
registry" and "it's ready for unattended use" are different claims.

## Why Aura needed it

Aura has a large capability surface and acts on it **autonomously**. A
background initiative can reach a connector nobody has exercised in months with
no human present to notice it misbehaving. Registration meant "the import
succeeded" — not that the skill validates inputs, bounds its timeout, is safe to
retry, verifies its postconditions, or reports a usable error.

So maturity is a **gate on reach, not on capability**:

| Context | Minimum tier |
|---|---|
| Attended (a person asked and is watching) | UNRATED — unrestricted |
| Deferred (person-triggered, result unseen) | BRONZE |
| Autonomous (Aura decided; nobody watching) | SILVER |
| Autonomous + irreversible | GOLD |

UNRATED attended use is unrestricted by this maturity gate; ordinary action
authorisation, sandbox and resource controls still apply. The goal is to stop the
least-exercised code in the registry from being reachable by the most
consequential path — not to shrink what Aura can do when asked.

## Consolidation: derived, not separately declared

Maturity is **derived from metadata skills already carry** (`input_model`,
`schema_override`, `effect_scope`, `authority_class`). A parallel maturity
manifest would be a second source of truth, and the one nobody updates would be
the one gating autonomous action.

Only claims the metadata genuinely supports are inferred:

- a typed input model **is** a typed request schema;
- a declared `read_only` / `pure_compute` scope earns the gate’s retry-safe
  classification. The declaration must match behavior; resource use,
  time-dependent reads and hidden effects still need appropriate checks;
- a classified authority **is** authority scoping.

Bounded timeouts, postcondition verification, effect receipts, recovery and
diagnostics must be declared. Inferring them would be precisely the unearned
claim this module exists to prevent.

## Rollout: observe before enforce

The gate runs live and records every refusal it *would* make, but does not
refuse unless `AURA_ENFORCE_CAPABILITY_MATURITY` is set.

This is a deliberate rollout decision, not an unfinished one. Aura's skill
surface is large and almost entirely ungraded; shipping this enforcing would
have refused most autonomous work on day one, which is how a safety mechanism
gets switched off permanently instead of adopted. Enabling it was tried first
and broke 11 existing tests — that is the evidence, not a hypothetical.

`CapabilityEngine.maturity_backlog()` surfaces the observations. Every entry
names a capability being reached autonomously without the engineering that use
implies. **That list is the grading backlog**, and enforcement becomes a single
flag once the capabilities that matter carry their properties.

## Deviations

- Tiers are **earned, never claimed**. A capability declaring GOLD without
  GOLD's properties is demoted to what it actually satisfies, and the demotion
  names the missing rules.
- Exemptions waive a named rule with a stated reason and remain visible in the
  grade, so a waiver is a decision rather than a silent gap.
- The gate never raises. A gate that cannot evaluate allows execution and
  records the gap; a maturity check must not become an outage.

## Conformance tests

`tests/test_capability_maturity.py` — 30 tests: tier derivation, cumulative
tiers, demotion, exemptions, context monotonicity, the derivation rules, and
that enforcement is opt-in.

## Known unsupported

- The initial rollout found most capabilities ungraded. Read the current
  metadata-derived report before quoting today’s tier distribution; this
  guide is not a fresh inventory of every skill.
- The gate is applied in `CapabilityEngine.execute` only; other execution
  entry points do not yet consult it.
