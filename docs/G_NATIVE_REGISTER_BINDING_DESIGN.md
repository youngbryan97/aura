# Native register binding

The fresh-schema scorer selected 72/72 correct complete programs from a
target-derived inventory. Target-blind local generation subsequently failed
all three balanced scalar, lookup, and count canaries. In each, its second
operation chose input register 3 instead of computed register 4. Correct
operation choices did not supply correct argument identities.

## Mechanism and evidence

The verified fold-0 source inventory contains 239 three-input, two-step
linear programs and 64 four-input, three-step fork/join programs. It contains
no four-input linear chain. The native wire assigns computed results packed
integer coordinates `n_inputs + step`. Thus the first intermediate is 3 in
every linear fit example and 4 in the fresh linear examples. An arity-sensitive
shortcut is consistent with the trace; its causal contribution is unmeasured.

The shared semantic transducer already encodes role-relative identities as
`input:i` and `result:j`. The new `RegisterIdentity` view uses exactly that
encoding, checks definition availability, and maps back to the same `Program`.
The opt-in native wire includes an explicit `role_relative_v1` register
declaration. The default absolute wire and historical receipts stay unchanged.
The typed decoder supports either representation. Training and replay must
bind the wire version explicitly before the relative path can be measured.

## Exact representation property

For input arity n, define packed-coordinate resolution:

    input:i  -> i,       0 <= i < n
    result:j -> n + j,   0 <= j < current_step.

The two ranges are disjoint. Encoding followed by resolution is identity on
every admitted prior register. Appending unused inputs changes the packed
coordinate of each computed result but not its `result:j` identity. Induction
over the instruction sequence then preserves every primitive argument and
computed value when the input values at the original indices are retained.
This includes independent branches and joins. Forward and self references
remain invalid. The tests check round trips, padding, canonical encodings,
missing definitions, branches, and disconnected graphs.

This proves a coordinate transformation, not semantic interpretation of
language. Source input ordering, choosing the intended role, operation meaning,
and termination remain learned obligations. No expected answer, target graph,
or construction identity may enter runtime scoring.

## Research checks

[ReCOGS](https://aclanthology.org/2023.tacl-1.96/) demonstrates that incidental
logical-form details can substantially affect measured generalization. Its
authors also reject a variable-free representation that loses meanings.
Aura's proposed coordinate change is therefore tested for reversibility;
grading continues to use graph semantics, not wire-string equality.

[Structural generalization through supertagging](https://aclanthology.org/2023.emnlp-main.69/)
uses global structural constraints and graph matching. Aura already has a
typed argument optimizer and operation-chart search. Those mechanisms should
retain jointly coherent alternatives instead of assuming locally likely
references imply a globally correct graph. Their existence does not establish
native search coverage or transfer; those properties require measurements.

[Execution-guided decoding](https://www.microsoft.com/en-us/research/publication/execution-guided-neural-program-decoding/)
uses type and execution checks to exclude invalid programs. Aura's floor already
provides these checks. They cannot distinguish two well-typed programs that
express different plausible meanings without additional source evidence.

## Measurement obligations

1. Finish the frozen complete-epoch fold-2 control without changing its code,
   schedule prefix, loss, or candidate bank mid-run.
2. Bind relative-register training, checkpoint replay, complete-candidate
   scoring, typed generation, and unconstrained generation to one explicit
   encoding contract. Refuse unknown encodings before loading weights.
3. Fit only source examples and select checkpoints only on disjoint source
   calibration. Keep exposed canaries diagnostic; reserve independent families
   for qualification.
4. Measure candidate reach, local binding, whole-graph selection, completion,
   public answers, and incumbent losses separately. Include arity changes,
   unused-input controls, independent branches, depth changes, and sequence
   primitives.
5. Continue G03-G05 closure only on measured non-regression and fresh transfer.
   These representation checks grant no serving, fusion, or frontier claim.
