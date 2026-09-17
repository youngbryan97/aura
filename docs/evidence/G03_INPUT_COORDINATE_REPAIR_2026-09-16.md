# G03 input-coordinate repair

## Completed development measurement

The frozen `semantic-argument-complete-20260915` run finished. Its 500 exposed
validation tasks scored 472 exact programs for the incumbent, 459 for the
joint decoder before fitting, and 436 for the fitted candidate (439 equivalent).
The incumbent remains selected. These are development results, not fresh
transfer evidence; none grants serving authority.

The optimizer reduced wrong or tied retained inequalities from 26 to one with
six accepted updates and no protected-positive inequality regressions. Fresh
decoding nevertheless regressed. Retaining fixed graph witnesses does not
prove retention of autonomous decoding after latent choices change.

## Reproduced contradictory supervision

Source `9ca1b4b99fd721e67bf4e3dce73d63c35f35b4797e06942ee0340037210fa973`
supplied runtime constraint 8939 and binding constraint 8940 with opposed
margins: -14.4544418156 and +14.4544418156.

Its public inputs contain a sequence and two distinct literal occurrences,
both valued 3. Source annotation orders the literal anchors at tokens 54 and
10; runtime grounding orders them at tokens 10 and 54. Both orders preserve
the observed values. Their register identities differ under counterfactual
values, however. The runtime miner compared unaligned register numbers and
scored the source target using the runtime's exchanged anchors. This inverted
the supervision despite a correct source binding.

The miner now re-expresses source input registers in runtime anchor coordinates
before comparison and scoring. Alignment requires identical, unique anchor
sets and a value-preserving permutation. A changed anchor or changed value is
reported as unaligned, never relabelled as correct. Intermediate registers,
operation identities, and source spans remain unchanged. This is training-only;
source annotations do not enter serving.

## Verification

- 47 focused semantic tests passed, including a correct swapped-literal case,
  a genuinely wrong binding, and refused anchor/value substitutions.
- Replaying the actual saved source now reports `equivalent`, no runtime
  contrast, and input map `[0, 2, 1]`.
- The independent binding miner still retains its positive constraint with
  margin +14.4544418156.
- Existing validation score definitions and historical artifacts are unchanged.

A fresh multi-round fit and full paired development measurement are required.
G03-G12 remain open where already open; this repair is not an accuracy claim.
