"""Context refresh must not run experiments or rewrite stored check results."""
from tools import isc_completion_status as completion


def test_reading_note_is_idempotent_and_preserves_stored_body():
    snapshot = "# The ISC completion list\n\n**3 done, 2 open.**\n\n- [x] kept result\n"
    updated = completion.with_reading_note(snapshot)
    assert completion.with_reading_note(updated) == updated
    assert updated.endswith(snapshot.partition("\n")[2])
    assert "stored counts" in updated


def test_reading_note_command_does_not_evaluate_checks(tmp_path, monkeypatch):
    doc = tmp_path / "snapshot.md"
    original = "# The ISC completion list\n\n- [x] unchanged result\n"
    doc.write_text(original)
    monkeypatch.setattr(completion, "DOC", doc)
    monkeypatch.setattr("sys.argv", ["isc_completion_status", "--refresh-reading-note"])

    def forbidden():
        raise AssertionError("context-only refresh must not execute checks")

    monkeypatch.setattr(completion, "evaluate", forbidden)
    assert completion.main() == 0
    assert doc.read_text().endswith(original.partition("\n")[2])
