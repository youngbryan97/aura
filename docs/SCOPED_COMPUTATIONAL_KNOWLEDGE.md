# Scoped Computational Knowledge

Aura can calculate consequences of an explicit model and scoped premises, then
use those consequences to constrain an executable candidate portfolio. The
calculation does not establish that the model describes the source request.
That mapping remains an upstream proposal which needs evidence.

## Connected Path

`run_semantic_computation_loop` connects these existing authorities:

1. A `KnowledgeContext` identifies observations, givens, estimates and assumptions
   for one source digest, with provenance and optional validity intervals.
2. A `Formulation` binds those premises to versioned calculation cells. Cells
   share derived values through a dependency graph; input identity survives a
   change in graph order. Computations run off the event loop.
3. Calculation checks dimensions, applicability premises and numeric domains.
   Derived values retain the uncertainty and freshness of every ancestor.
   Caller-bound observation ports can supply absent inputs as `ScopedPremise`
   records. Each record must name a requested input, its provenance, the current
   scope and a valid time interval. Derived values and untyped text are refused.
   Conflicting ports leave the input unresolved. New input data triggers another
   graph calculation; the loop ends when complete or when no new input arrives.
4. The semantic program portfolio replays each consequence before selection.
   Contradicted candidates are removed. The incumbent survives when the evidence
   cannot distinguish candidates. If every candidate is refuted, there is no
   selected answer.
5. Missing or contradicted premises produce explicit gaps. Supplied knowledge
   providers receive queries. Their passages remain retrieved material; this
   path does not turn a passage into a measured quantity.
   An optional `memory_retriever` uses the existing intentional-retrieval router,
   including registered local-reference and episodic adapters. Missing stores
   remain visible in its result. No second memory store is created.
6. The state gateway retains the equation graph, model hashes, bindings and
   semantic definition. Restoration verifies those hashes. New observations
   require a new calculation; retained recipes contain no cached answers.

`run_grounded_semantic_computation_loop` also accepts the existing
`SemanticCandidateUnion`. It checks the source identity and union receipt,
executes every distinct program in the common source-input coordinates, then
uses this same graph, observation, retrieval and selection path. Equal programs
retain all method origins without receiving a correctness vote. The named
incumbent must identify exactly one program. The v3 receipt distinguishes that
original incumbent from an executable fallback selected after a real failure.

A formulation with several outputs must name its `selection_output`. Distinct
quantities such as pressure and energy cannot silently constrain the same answer.
An unresolved cell leaves the original portfolio selection untouched.
Observation ports do not overwrite existing evidence or count repeated reports
as new facts. The v2 loop receipt binds each admission and recalculation to its
before-and-after context. Port authority comes from the caller's wiring, not
from a string in retrieved content. Estimates remain conditional after admission.

## Calculation Boundary

`QuantityBounds` uses rational enclosing intervals in the shared SI dimension
system. Addition, subtraction, multiplication, division and bounded integer
powers are available. An interval containing a zero denominator is unresolved.
The expression grammar contains no Python calls, attribute access or generated
scripts. Offset units require conversion to absolute SI quantities first.

`derive_scoped_fact` uses the existing propositional proof kernel and its
independent checker. Inconsistent premises cannot establish arbitrary facts by
explosion. A derived fact can satisfy a model guard, but keeps the assumptions
and validity limits of its ancestors.

`solve_model_unknown` solves a uniquely determined rational affine numerator
and checks the result against the original dimensioned expression. This admits
some inverse formulas, including resistance from voltage and current. It
refuses nonlinear or underdetermined problems. Solving a formula does not
establish its applicability premises.

## Model Catalog

The first catalog has 23 cells, each with input and output units, a formula,
numeric domain checks, applicability premises and a source:

- Rational sum and difference; rectangle area; squared Euclidean distance.
- Distance from average speed; constant-acceleration position; Newtonian force;
  kinetic energy; force from pressure; mechanical work.
- Hydrostatic pressure; buoyancy; flow speed; Reynolds number; hydraulic power;
  mass from uniform density.
- Sensible heat; Ohmic current; electrical power; ideal-gas pressure.
- Molar concentration; dilution; Michaelis-Menten initial reaction rate.

Physical and chemical cells refer to OpenStax's
[University Physics](https://openstax.org/books/university-physics-volume-1)
and [Chemistry](https://openstax.org/books/chemistry-2e).
The kinetic cell refers to the
[Michaelis-Menten translation](https://pmc.ncbi.nlm.nih.gov/articles/PMC3381512/).
These models require their declared regimes; they do not cover entire fields.
Fluid flow and Ohmic current, for example, require the meanings and conditions
specified in [fluid dynamics](https://openstax.org/books/university-physics-volume-1/pages/14-5-fluid-dynamics)
and [Ohm's law](https://openstax.org/books/university-physics-volume-2/pages/9-4-ohms-law).
No cell supplies a default measurement of gravity, density or temperature.

## Proof Status

The executable API and portfolio connection are tested. The tests include a
three-cell fluid/mechanical graph, premise ablation, changed observations,
counterevidence, invalid units, provider failure, observation-driven recomputation,
real local-corpus retrieval and state-gateway restoration.
Mixed-entry tests use actual candidate-bank receipts and actual floor execution;
they check complementary methods, missing premises, observation-driven revision,
complete refutation, changed public inputs and a failing incumbent.
They establish conditional computation and integration on those fixtures.

The autonomous language-to-equation mapper is not supplied by this catalog.
Knowledge providers and the state gateway are explicit dependencies supplied
by the caller. No live chat activation, native checkpoint promotion, fusion,
500-request clearance or general-transfer result follows from these tests.
G03 remains open.

## Code

- `core/reasoning/computational_knowledge.py`: scoped intervals and replay.
- `core/reasoning/computational_models.py`: reviewed formula declarations.
- `core/reasoning/scoped_logical_computation.py`: checked logical consequences.
- `core/reasoning/computation_sandbox.py`: graph solving, inversion and retention.
- `core/learning/semantic_computed_constraints.py`: portfolio constraints.
- `core/learning/semantic_computation_loop.py`: the connected async path.
- `tests/test_computational_knowledge.py` and
  `tests/test_semantic_computed_constraints.py`: executable checks.
- `tests/test_semantic_mixed_computation_loop.py`: mixed-entry checks.
