# Engineering design: how a drawing gets its numbers

Aura can take a design brief and produce checked engineering drawings from
it — an assembly with labelled callouts, an exploded view keyed to a parts
list, a wiring or piping schematic, a section, dimensioned views — plus the
files a printer, a machine shop and a CAD package each want.

The point of the package is not the picture. It is that every number on the
picture was computed, and that a number which cannot say where it came from
never reaches the page.

## The split

The cortex decides **what** to build: the parts, what each is for, what it
is made of, roughly how big, what connects to what, and what the thing has
to achieve. It is never asked for a mass, a stress, a pressure drop, a
safety factor or a coordinate.

Everything downstream is arithmetic in [`core/engineering/`](../core/engineering/).
Mass comes from the geometry and the material density. Stress comes from
the pressure and the wall. Where the parts sit comes from which of them
enclose the others. That division is enforced by
[`core/engineering/DEPS`](../core/engineering/DEPS), which forbids this
package from importing `core.brain`, `core.llm`, `core.conversation` or
`core.search`: an import of any of them would let a generated figure onto a
drawing.

## Why the numbers mean anything

[`validation.py`](../core/engineering/validation.py) holds 29 textbook and
handbook problems with published answers — Euler buckling, Colebrook
friction, molar volume at STP, the 16.3% overshoot of a second-order system
at damping ratio 0.5 — and runs them against this engine. Each names its
source and its tolerance, and a correlation gets the tolerance the
correlation itself is good to rather than a slack one.

Writing that battery caught three wrong expectations and one real bug:
`rss_stack` was converting its own SI result a second time, so a 0.03 and a
0.04 millimetre tolerance stacked to 5 x 10^-8 metres. All 29 now reproduce
their published answers, and the whole battery runs inside
[`verify.py`](../core/engineering/verify.py) before any drawing is made. If
it is failing, the drawing says so on its face rather than implying
otherwise.

`coverage()` reports validated cases per discipline. A discipline with none
has no validated formula, and the capability statement says that out loud.

## The gate

[`verify.py`](../core/engineering/verify.py) runs before anything is drawn:

- **Provenance.** A value reaches a drawing only as a `Finding` carrying its
  formula, its inputs and its reference. Anything else is dropped and
  reported as dropped.
- **Physical plausibility.** No negative mass, no efficiency above one, no
  temperature below absolute zero, no infinities.
- **Units and domains.** A port that says it carries current has to carry
  amperes. A 48 V port wired to a 5 V one is a fault, not a wire.
- **Requirement coverage.** A requirement whose check did not run is
  reported unverified, never met.
- **Interference.** Two parts in the same space is blocking.

## One conservation law

Connections carry the two variables Modelica settled on: an *across*
variable equal at every node, and a *through* variable that sums to zero
there. Kirchhoff's current law, a mass balance around a tee and a heat
balance are then the same statement, so
[`analysis/conservation.py`](../core/engineering/analysis/conservation.py)
is one function covering all of them. Adding a domain to
[`domains.py`](../core/engineering/domains.py) extends it without a line
being written there.

A node where any port has not declared its direction comes back **unchecked**
rather than passed.

## Aerospace and subsea discipline

[`assurance.py`](../core/engineering/assurance.py) carries the habits, as
data rather than as assumptions buried in a formula:

- Margin of safety against a named factor set — NASA-STD-5001B for
  spaceflight, ABS Rules for Underwater Vehicles for pressure hulls, ASME
  BPVC VIII for industrial vessels — reported as
  `allowable / (limit x factor) - 1`, not as a bare ratio.
- Mass growth allowance by design maturity, per ANSI/AIAA S-120A.
- GUM uncertainty propagation with a per-input budget, so the answer to
  "how wrong could this be" names which input to pin down first.
- MIL-STD-1629A criticality, and single points of failure found by walking
  the connection graph.
- NASA EEE-INST-002 derating.

## Reading it without an engineering degree

Every finding carries a `plain` sentence written from the numbers, not
about them. [`explain.py`](../core/engineering/explain.py) holds a glossary
of about eighty terms, and generates the "how it works" narrative by
walking the connection graph — so it cannot describe a link the model does
not have, and cannot omit one it does.

## Acting on it

[`build.py`](../core/engineering/build.py) produces what to buy and from
what sort of supplier, what to make out of what stock size by what process,
and the assembly order — derived from the explode vectors, so nothing has
to be fitted through something already in place. A process that cannot work
the material it was given says so.

## Where it is wired

| Connection | Where |
|---|---|
| Skill on the capability surface | [`core/skills/design_engineering.py`](../core/skills/design_engineering.py) |
| Live routes and panel | [`interface/routes/engineering.py`](../interface/routes/engineering.py), `interface/static/schematics.html` |
| A stated figure checked against the computed one | [`core/brain/verifiers/engineering_engine.py`](../core/brain/verifiers/engineering_engine.py) |
| A physical subject externalises as a schematic | `_externalization_path` in [`core/brain/imagination.py`](../core/brain/imagination.py) |
| Dimensional arithmetic in the sandbox | `RUNNER_PY` in [`core/sandbox/runner.py`](../core/sandbox/runner.py) |
| Measurable faculty | [`core/engineering/faculty.py`](../core/engineering/faculty.py) |
| What a design taught, generalised | [`core/engineering/knowledge.py`](../core/engineering/knowledge.py) |

## What it cannot do

Stated here because the capability statement states it too, and both are
generated from the same registries.

- No finite-element or computational-fluid analysis. Everything is closed
  form or a named correlation, which is what makes it hand-checkable.
- No mesh solver, so no contact, no modal analysis beyond a uniform
  cantilever's first mode, no transient thermal.
- No discipline without a validated case behind it.
- Geometry is parametric primitives placed and combined. It is not a
  boundary-representation CAD kernel: there are no fillets, no booleans and
  no swept profiles beyond a revolve and an extrude.
- The interference check is bounding boxes, so it catches gross clashes and
  never reports one that is not there beyond that approximation.
