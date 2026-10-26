# Aura architecture reference

Status: Reference · Reconciled 25 October 2026

Use [the technical architecture](../ARCHITECTURE.md) for the system design and
[the generated map](../docs/ARCHITECTURE_MAP.md) for the source inventory.
This shorter page states the boundaries a reader must preserve when explaining it.

## Processing and persistence

Foreground conversation and background work have distinct paths. The
[legacy engine](../core/brain/cognitive_engine.py) and
[kernel](../core/kernel/aura_kernel.py) must be identified in a receipt before
attributing a result to one of them. A drawing of a kernel tick is not proof
that a particular chat request traversed it.

A state commit can be atomic without the whole tick being atomic. Phases,
model work and external actions do not form one rollback transaction; an
external action cannot generally be undone by abandoning a state write.
Failure handling depends on the phase and declared criticality. Optional
failure recovery does not permit a mock to stand in for unavailable critical
infrastructure. See the [runtime contract](../docs/RUNTIME_CONTRACT.md).

## Authority and resources

The required action contract is governance approval, an authority check and a
traceable outcome. Passing the declared gateway tests supports those paths
and cases; it is not proof that every possible bypass is impossible.
Model roles are subject to memory admission. The current Cortex target is
27B; older 32B records retain their dates and model identities. No second
resident Cortex is implied by a component map.

Context size, retention and cadence are owned by their source configuration
and request policy. There is no universal 40-turn memory guarantee, fixed
1 Hz processing promise, 5,000-token prompt bound or zero-cost idle state
established by this reference. A character cap also does not determine an
exact token count.

## Integration, affect and learning

The [16-node phi model](../core/consciousness/phi_core.py) is a bounded
IIT-inspired estimator with spectral approximation; it is not exact IIT 4.0
of Aura. The [whole-system estimator](../docs/WHOLE_SYSTEM_PHI.md) and
[ISC experiment](../docs/ISC_STATUS.md) have separate methods and evidence.

Activation steering, sampling control and context assembly are distinct
ways internal state can affect generation. The measured effect belongs to
the artifact, scale, configuration and task used in its evidence; a route
in source alone does not establish that the current process used it.

Kept procedures can change later answers without changing the backbone’s
weights. G09 passed its registered broad-gain criterion on math and planning;
L09’s broader ordinary-use comparison is still open. Those outcomes do not
establish backbone recurrence gain or frontier performance.

## Where to look next

- [Source architecture and entry points](../ARCHITECTURE.md)
- [Model roles and artifact scope](../MODEL_CARD.md)
- [Memory mechanisms](../MEMORY_CARD.md)
- [Reasoning results](../docs/RECURSIVE_LATENT_CORTEX.md)
- [Tests, resource budgets and dated catalogues](../TESTING.md)
