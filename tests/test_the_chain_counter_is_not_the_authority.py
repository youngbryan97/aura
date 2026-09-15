"""A counter that had run past the file it writes to.

The chain caches its next sequence and checks the file with a stat — inode,
size, modification time. A fork that puts a state file back also puts its old
modification time back, on purpose, so that a service caching on the stamp
reloads the restored bytes. The chain's own stamp is such a cache, and a rewind
it cannot see leaves the counter ahead of the file.

What that cost was visible in every campaign log: "assigned seq 43, chain head
41, body 42 present", once per arm, recovered each time by re-reading the head
afterwards. The recovery worked and the reading was wrong on the way in.

The file is the authority for the next sequence. These tests hold that the
counter is confirmed against it before a sequence is handed out, so the ghost
line never sees the drift at all.
"""

from __future__ import annotations

import os

from core.ghost.ghost_line import GhostLine, SelfDigest, SubstrateFingerprint
from core.runtime.audit_chain import AuditChain


def _append(chain: AuditChain, i: int):
    return chain.append(
        receipt_id=f"receipt-{i}",
        kind="test_entry",
        body={"i": i},
        timestamp=1000.0 + i,
    )


def test_a_rewind_that_hides_behind_the_old_stamp_is_still_seen(tmp_path) -> None:
    chain = AuditChain(tmp_path / "chain")
    for i in range(4):
        _append(chain, i)
    assert chain.length() == 4

    # The fork's restore: the earlier bytes back, and the earlier stamp with
    # them, so nothing caching on the stamp can tell the file moved.
    lines = chain.path.read_text().splitlines(keepends=True)
    stat = chain.path.stat()
    chain.path.write_text("".join(lines[:2]))
    os.utime(chain.path, ns=(stat.st_mtime_ns, stat.st_mtime_ns))

    entry = _append(chain, 9)
    assert entry.seq == 2, "the sequence came from the counter rather than the file"
    assert chain.last_entry().seq == 2
    assert chain.length() == 3


def test_the_rewound_chain_still_verifies(tmp_path) -> None:
    chain = AuditChain(tmp_path / "chain")
    for i in range(4):
        _append(chain, i)
    lines = chain.path.read_text().splitlines(keepends=True)
    stat = chain.path.stat()
    chain.path.write_text("".join(lines[:2]))
    os.utime(chain.path, ns=(stat.st_mtime_ns, stat.st_mtime_ns))
    _append(chain, 9)

    ok, problems = chain.verify()
    assert ok, problems


def test_the_ghost_line_no_longer_reaches_its_recovery_path(tmp_path, monkeypatch) -> None:
    """The retry stays as the last resort, and nothing normal gets there."""
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path))
    line = GhostLine(root=tmp_path / "ghost")

    recorded: list[str] = []
    monkeypatch.setattr(
        "core.ghost.ghost_line.record_degradation",
        lambda subsystem, exc, **kwargs: recorded.append(f"{subsystem}: {exc}"),
    )

    def step(i: int):
        return line.advance(
            SelfDigest("Aura", f"hash-{i}", f"essence-{i}"),
            SubstrateFingerprint(model_artifact="stub"),
            trigger="tick",
        )

    for i in range(3):
        step(i)
    line._chain._next_seq += 2

    frame = step(9)
    assert frame.seq == 3
    assert not [line_ for line_ in recorded if "GhostAnchorUnavailable" in line_], recorded
