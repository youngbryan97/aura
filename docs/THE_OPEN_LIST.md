# The open list

Every task given in this session, in one place, with its state. Append-only in
spirit: a row changes state, it does not disappear.

Four sources feed it.

* **N** — `NextSteps.pdf`, the generality review.
* **L** — the 202-finding peer-comparison ledger. Full rows live in
  [MATURITY_LEDGER.md](MATURITY_LEDGER.md); only the clusters are here.
* **E** — `Additional Fixes.pdf`.
* **M** — `MoreLessons.pdf`, the blind maturity comparison against Letta,
  LangGraph, OpenHands, Home Assistant, AutoGPT, CrewAI, AutoGen, Soar and
  AtomSpace.

DONE means a test that fails without the change passes with it, and the commit
is pushed. Nothing is marked DONE on the strength of an argument.

## M — MoreLessons: what the blind comparison found

The review's own headline: Aura does not need prettier code, it needs **lower
causal ambiguity per unit of functionality**. Every M row is a way of paying
that down.

| # | Item | State |
|---|---|---|
| M1 | Resource handoff must transfer ownership, not wake a future and trust scheduling | DONE |
| M2 | Letta-grade turn lifecycle: discriminated state machine, immutable lease, lease-checked mutation, cancelling is not idle | TODO |
| M3 | LangGraph-grade checkpoint semantics: pending writes separate from committed checkpoints, channel versions, per-node seen-versions, durability never ahead of its writes | TODO |
| M4 | OpenHands-grade durable event log: duplicate-ID rejection, parent validity, branch semantics, bounded traversal, stale-index recovery, gap detection | TODO |
| M5 | Home Assistant-grade store: versioned records, minor versions, load serialisation, bounded concurrent loads, migration hooks, corruption quarantine | TODO |
| M6 | Home Assistant-grade root runtime: lifecycle states, thread affinity, typed jobs, task ownership, startup/shutdown stages | TODO |
| M7 | AutoGPT-grade graph execution: legal status transitions, per-node context copies, typed failure classes, transition-applied vs zero-rows-matched | TODO |
| M8 | CrewAI-grade checkpointing: typed trigger events across task/crew/agent/flow/LLM/tool/memory, two providers, restore from checkpoint | TODO |
| M9 | AutoGen-grade messaging: typed envelopes carrying cancellation, ids and trace; a serializer registry keyed by type — and subscription state persisted, which AutoGen does not do | TODO |
| M10 | Soar-grade authority: working-memory changes buffered and committed at phase boundaries with refcount semantics; explicit stopping reasons | TODO |
| M11 | AtomSpace-grade concurrency: two threads discovering absence together, recheck, repair the losing installation | TODO |
| M12 | Classify every state holder as authority, derived projection, or temporary computational state | STARTED — `who_owns_each_field` covers AuraState's 80 nested fields; the eleven runtime state holders the review names are not yet classified |

## L — the peer ledger

24 of 202 done. Clusters, not rows: [MATURITY_LEDGER.md](MATURITY_LEDGER.md)
carries all 202 with their state.

| Cluster | State |
|---|---|
| A. One cancellation token | DONE |
| B. Typed events | DONE |
| C. The phase DAG, compiled | DONE |
| D. State patches and reducers | DONE |
| E. Deterministic laboratory mode | DONE |
| F. Semantic identity | DONE |
| G. Memory provenance (Letta #1 MemoryFS, #16 three-way merge) | PART — #11 done |
| H. Conformance suites (CrewAI #11 provider adapters) | PART — store and graph suites done |
| I. One working memory | DONE |
| J. Runtime protocol and ownership | DONE |
| K. Tool schemas | PART — 78 tools still declare no result |

## N — NextSteps

| # | Item | State |
|---|---|---|
| N1 | ARC-AGI measured against its own null | DONE |
| N2 | Distance gradient vs visit count, both against random | DONE |
| N3 | Growth worth 0.00 on gate 9 — a real negative, published | DONE |
| N4 | Her generator reaches four distinct one-argument behaviours | DONE |
| N5 | A runtime probe beside gate 16's static scan | TODO |
| N7 | GAIA / OSWorld / post-cutoff SWE / hours-long autonomy | TODO |
| N9 | Grounding beyond the seven authored installers | TODO |
| N10 | Generational compounding, currently depth 0 | TODO |
| N11 | Native and source development in one verified loop | TODO |
| N12 | Fan-out and through-path reduction | DONE |
| N13 | 110 modules over the size threshold, 36 size regressions | TODO |
| N15 | The reachability scan's self-enumeration blind spot | DONE |

## E — Additional Fixes

| # | Item | State |
|---|---|---|
| E1 | `answer_sequence_question()` has no production caller outside its own subsystem: ordinary conversation does not route arbitrary reasoning through the self-growing symbolic language | DONE — served in the live chat path since Aug; every answer route is now counted, so "offered but never answers" is visible |
| E2 | Recurrent and compositional semantic routes are shadow- or qualification-gated rather than authoritative for arbitrary production cognition | STARTED — a route offered enough turns that never answers is now named; un-gating follows the measurement |
| E3 | `SkillSynthesizer.synthesize_pending()` has no production caller: the reactive missing-tool loop closes, the general repeated-failure to new-abstraction to forged-capability loop does not | DONE — `recursive_self_improvement` asks the forge about recurring gaps and records `asked_the_forge` |
| E4 | `LIVE_MIND_CONTROL_POLICY_CALIBRATED=False` — internal state actuates decoding, but the mapping is hand-tuned and unvalidated | TODO |
| E5 | `PhenomenalFalsifier.from_live()` calls a channel causal on proxy evidence; the intervention standard is treatment against null | TODO |
| E6 | Every named faculty needs a measured downstream effect, not a wired channel: "wired to consumer" is not "measurable contribution" | STARTED — the fourteen answer routes are measured; the named cognitive faculties are not |
| E7 | The sealed transfer experiment: measure on unseen domains, allow autonomous development, then intervene on the abstractions she built and show it changes performance on transfer domains | TODO |
| E8 | The improver null: improved improver beats a frozen improver on held-out problems across generations | TODO |
| E9 | Organs that know what they own, what they consume, what they promise and how failure propagates — coherent causal organisation over interaction entropy, not fewer organs | TODO |

## Found while working

Defects met on the way that were not on any list. They are here because
leaving them out would make the list a plan rather than a record.

| Item | State |
|---|---|
| Four stores resolved the live state root at import, so a test run wrote into the live mind | DONE |
| Three tests asserted behaviour the code has never had; one returned a status the contract cannot produce | DONE |
| `phenomena_wiring._service` did not catch `ContainerError`; the `samantha` alias was dropped with three live readers | DONE |
| The compiled plan reported its own mode as a write mode, so every published seal was a digest of a mislabelled plan | DONE |
| The sandbox memory-bomb guard read an attribute `AuraState` does not have, and passed a 5,000-item state | DONE |
| A test run could arrange its own replacement, replaying `-m pytest` as a runtime: 5,205 chained processes | DONE |
| The health report cost 18 seconds on first call and 2–4 after, on a route | DONE |
| A module-level `asyncio.Lock` shared across loops, which `hot_reboot` would have hit live | DONE |
