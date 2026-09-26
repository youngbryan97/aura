"""Bank arbitration features cannot read target labels until evaluation."""

from tools.calibrate_nested_semantic_bank_choices import bank_pair, construction_support


def _row():
    return {"bank": {"selected_program_sha256": "ordinary", "candidates": [
        {"program_sha256": "ordinary", "joint_score": None,
         "program": {"instructions": [["sub", [0, 1]]]}},
        {"program_sha256": "joint", "joint_score": 2.0,
         "program": {"instructions": [["sub", [1, 0]]]}},
        {"program_sha256": "ordinary", "joint_score": 1.0,
         "program": {"instructions": [["sub", [0, 1]]]}}]},
        "diagnosis": {"comparisons": [
            {"program_sha256": "ordinary", "status": "equivalent"},
            {"program_sha256": "joint", "status": "different"}]}}


def test_bank_pair_features_are_unchanged_when_labels_swap():
    row = _row()
    left, right, ordinary, joint = bank_pair(row)
    assert (ordinary, joint) == (True, False)
    assert left["joint_gap"] == 1.0
    assert right["joint_gap"] == 0.0
    assert left["candidate_count"] == 2.0
    for comparison in row["diagnosis"]["comparisons"]:
        comparison["status"] = ("different" if comparison["status"] == "equivalent"
                                else "equivalent")
    other = bank_pair(row)
    assert other[:2] == (left, right)
    assert other[2:] == (False, True)


def test_bank_pair_requires_an_independent_comparison():
    row = _row()
    row["diagnosis"]["comparisons"].pop()
    try:
        bank_pair(row)
    except ValueError:
        pass
    else:
        raise AssertionError("unverified challenger was accepted")


def test_construction_support_counts_groups_not_paraphrase_rows():
    rows = [("a", None), ("b", None), ("c", None)]
    assert construction_support(rows, {"a": "same", "b": "same", "c": "other"}) == {
        "other": 1, "same": 2}
