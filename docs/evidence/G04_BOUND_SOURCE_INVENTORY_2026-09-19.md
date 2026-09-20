# G04 bound source inventory

The natural-transfer preflight rebuilt source schemas from a fixed family
table using each generator's default configuration. It did not establish
which examples the frozen transducer had used. It also could not inventory
the newer counterfactual and natural-alias families.

The preflight now accepts the archived feature manifests and uses the existing
`rebuild_semantic_feature_selection` path. That path verifies manifest and
configuration hashes, selected example identities, and the reconstructed
programs' corpus digest. The inventory additionally checks train/validation
membership and counts against the frozen transducer's receipt. No hidden-state
arrays or model load are needed. Missing manifests leave the legacy default
reconstruction explicitly unverified; they cannot grant a passed structural
preflight. Historical artifacts are unchanged.

## Measured source

The standalone inventory command recovered the current source-fit parent's
eight source families: arithmetic, cataphoric, counterfactual, fork/join,
natural-alias source, natural source, reserved alias, and role binding.

- Archived examples: 1,764.
- Distinct operation/dependency/type schemas: 656.
- Inventory receipt:
  `eb3a601fffdebdd95eb890a5ac6c616e28e55efde151da4130263f1d1945a80c`.
- Artifact:
  `/Users/bryan/.aura/rlc-evidence/semantic-transfer-source-inventory-20260919.json`.
- Hidden arrays loaded: false.

The inventory records all selected source examples, including non-training
splits, conservatively excluding their schemas from a new schema-transfer
claim. It does not prove that every historical exposure is recorded, that a
future task is fresh, or that the compiler succeeds on any transfer task.

## Checks

Twenty-seven focused tests pass across source inventory, natural preflight,
and feature reconstruction. They cover unknown family names, exact nondefault
selection, missing and substituted manifests, source membership changes,
reconstructed program drift, and tampered frozen verification evidence.

G04 remains open. This supplies the actual exclusion inventory for a future
frozen transfer run; it does not replace that run.
