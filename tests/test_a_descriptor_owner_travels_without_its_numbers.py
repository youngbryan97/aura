"""An object holding descriptors as ints does not take its numbers to another process.

The carrier rehearsal of 22 October died at its first restore in two of three
workers: their receipts database held lines of a ghost-frame audit chain. The
chain had travelled in an anchor from the coordinator with the coordinator's
descriptor numbers, and in each worker those numbers named the database.
"""
from __future__ import annotations

import os
import time

import pytest

from core.runtime.audit_chain import AuditChain
from core.runtime.descriptor_owner import OwnsDescriptors
from core.subject.anchor_travel import LiveFunction, dumps, fill_from, loads


class _Holder(OwnsDescriptors):
    """Holds a descriptor and has no way to open it again."""

    def __init__(self) -> None:
        self.fd = 7


def _append(chain: AuditChain, n: int) -> None:
    chain.append(receipt_id=f"r{n}", kind="ghost_frame", body={"n": n}, timestamp=time.time())


def test_a_chain_read_elsewhere_writes_to_its_own_file_and_nothing_else(tmp_path):
    chain = AuditChain(root=tmp_path / "chain")
    _append(chain, 0)
    assert chain._append_fd is not None
    travelled = dumps(chain)
    chain.close()
    # The numbers the writer held now belong to other files here.
    others = [tmp_path / f"other_{n}.bin" for n in range(2)]
    fds = [os.open(str(path), os.O_WRONLY | os.O_CREAT, 0o600) for path in others]
    try:
        read = fill_from(loads(travelled), None)
        assert read._append_fd is None and read._lock_fd is None
        _append(read, 1)
        read.close()
    finally:
        for fd in fds:
            os.close(fd)
    assert [path.read_bytes() for path in others] == [b"", b""]
    assert len((tmp_path / "chain" / "_chain.jsonl").read_text().splitlines()) == 2


def test_an_owner_that_cannot_open_again_travels_as_the_readers_own():
    with pytest.raises(TypeError):
        _Holder().__getstate__()
    assert isinstance(loads(dumps({"held": _Holder()}))["held"], LiveFunction)
