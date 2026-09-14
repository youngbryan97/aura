# G09: plan for computed values, then observe them

The common procedure planner refused every exact-value goal when the producing
procedure declared a computed output. It could propose an integer-producing
procedure for "an integer," but never for "the value 42" unless 42 was already
hardcoded into that procedure's signature.

Search now proposes a type-compatible computation and records its unresolved
value obligations on `ProcedurePlan`. A known contradictory constant, an
incompatible type, or inconsistent goal constraints still prevents the plan.
No backend runs during search. An exact value in a goal is never substituted
for a backend's output.

The existing executor checks intermediate preconditions before invoking the
next backend, and final requirements against observed state. A completed
computation that returns the wrong value therefore leaves the goal incomplete.
No procedure success rate is updated from this structural observation.

The change is on `plan_procedure`, already used by the common procedure path
and semantic runtime. It does not add a second planner or an alternate live
dispatch route. Integer outputs also compose into numeric inputs, and integer
sequences into sequence inputs, without permitting the reverse downcasts.
The common signature algebra now recognizes those structural subtype relations.

129 focused tests passed across planning, execution, procedure currency,
semantic procedure currency and semantic runtime. They include a real local
CSV read, universal-floor multiplication, and a separate report backend under
both type-only and exact-value goals. Replacing the report with an unrelated
string completes the execution but fails the exact goal.

This is a reusable planning capability and checked backend integration, not a
learned language-grounding or broad reasoning result. An intended computation
still needs task-grounded selection; type compatibility alone cannot select
its meaning. G09 remains open. The steering campaign owns the model during
these CPU checks, so no desktop live-validation claim is made here.
