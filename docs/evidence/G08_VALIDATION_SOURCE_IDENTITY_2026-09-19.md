# Validation cache source identity

The validation checkpoint previously bound coefficients, examples, and the
scoring-policy name, but not the implementation producing the scores. Reusing
that checkpoint after changing a decoder or scorer could reuse stale evidence.

Validation now binds a SHA-256 inventory of the core Python source tree, Python
version, and NumPy version. Candidate selection records that identity and checks
it again before publishing its result. A changed implementation invalidates the
measurement cache; this is not a serving restriction or a qualification grant.
Historical receipts are not rewritten.

This is a disk-source snapshot, not loaded-interpreter attestation. Campaigns
must still run from an identified checkout in a fresh process. The conservative
inventory includes dependencies outside the immediate decoder module.

## Checks

`tests/test_semantic_validation_source_identity.py` proves source edits,
additions, and removals change the identity; unchanged runs resume; changed
source cannot reuse cached scores; and mid-evaluation drift prevents selection.
Together with `tests/test_semantic_program_validation_selection.py`, 25 tests
passed on September 19, 2026.

This closes the stale-implementation cache defect, not G08's independent
verification, uncertainty, contamination, or cross-domain outcome obligations.
