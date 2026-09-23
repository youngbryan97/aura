"""An object holding OS descriptors as ints is never copied into a second owner.

The whole report run of 23 September died with EXC_GUARD (GUARD_TYPE_FD, CLOSE
on descriptor 39): the subject fork deep-copied the latent cortex's private
action-snapshot store, the discarded copy's finalizer closed the store's
descriptors, and a later close landed on a number Metal had taken and guarded.
These pin the fix in core/runtime/descriptor_owner.py and the fork's copy rules,
and a scan keeps any new descriptor-holding class from missing it.
"""

from __future__ import annotations

import ast
import copy
import importlib
from pathlib import Path
from typing import Any

import pytest

from core.runtime.descriptor_owner import OwnsDescriptors

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]

OWNERS = [
    ("core.brain.llm.latent_cortex.campaign_journal", "CampaignJournal"),
    ("core.brain.llm.latent_cortex.action_state_capture", "PrivateActionSnapshotStore"),
    ("core.runtime.audit_chain", "AuditChain"),
    ("core.learning.verified_transition_episode", "TransitionArtifactStore"),
    ("core.runtime.secure_path_custody", "DirectoryCustody"),
    ("core.resilience.resource_arbitrator", "ResourceArbitrator"),
    ("core.conversation.terminal_chat", "TerminalFallbackChat"),
    ("core.bus.shared_mem_bus", "_FileBackedSharedMemory"),
]


def _inert(klass: type) -> Any:
    """An instance built without __init__, whose finalizer has nothing to close."""
    return object.__new__(type(f"Inert{klass.__name__}", (klass,), {"__del__": lambda self: None}))


@pytest.mark.parametrize(("module", "name"), OWNERS)
def test_a_copy_of_a_descriptor_owner_is_the_owner_itself(module: str, name: str) -> None:
    klass = getattr(importlib.import_module(module), name)
    assert issubclass(klass, OwnsDescriptors)
    owner = _inert(klass)
    assert copy.copy(owner) is owner
    assert copy.deepcopy(owner) is owner
    assert copy.deepcopy({"held": [owner]})["held"][0] is owner


def test_the_fork_treats_a_descriptor_owner_as_furniture() -> None:
    from core.subject.copies import _holds_furniture, _is_process_furniture

    klass = importlib.import_module("core.runtime.secure_path_custody").DirectoryCustody
    owner = _inert(klass)
    assert _is_process_furniture(owner)

    class Service:
        def __init__(self) -> None:
            self.custody = owner

    assert _holds_furniture(Service())


def _keeps_a_descriptor(node: ast.ClassDef) -> bool:
    """Whether a method assigns something other than a sentinel to a self.*_fd(s) attribute."""
    for item in ast.walk(node):
        if not isinstance(item, ast.Assign):
            continue
        for target in item.targets:
            attribute = target.value if isinstance(target, ast.Subscript) else target
            if not (isinstance(attribute, ast.Attribute) and isinstance(attribute.value, ast.Name)):
                continue
            if attribute.value.id != "self" or not attribute.attr.endswith(("_fd", "_fds")):
                continue
            value = item.value
            sentinel = (
                (isinstance(value, ast.Constant) and value.value in (None, -1))
                or (isinstance(value, ast.UnaryOp) and isinstance(value.operand, ast.Constant))
                or (isinstance(value, ast.Dict) and not value.keys)
            )
            if not sentinel:
                return True
    return False


def test_every_class_that_keeps_a_descriptor_says_it_owns_one() -> None:
    """A name-based scan, which is what a guard can do: a *_fd attribute set to a live value."""
    missing = []
    for path in sorted((REPO / "core").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or not _keeps_a_descriptor(node):
                continue
            bases = {base.id if isinstance(base, ast.Name) else getattr(base, "attr", "") for base in node.bases}
            if "OwnsDescriptors" not in bases:
                missing.append(f"{path.relative_to(REPO)}:{node.name}")
    assert not missing, f"classes keeping OS descriptors without OwnsDescriptors: {missing}"


def test_a_pipe_a_queue_and_shared_memory_are_furniture() -> None:
    """What a model client holds to talk to its worker is never copied by the fork."""
    import gc
    import multiprocessing

    from core.subject.copies import _holds_furniture, _is_process_furniture
    from core.subject.snapshot import _organ_state

    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    held = [
        reader,
        writer,
        context.Queue(),
        context.SimpleQueue(),
        context.Lock(),
        context.Event(),
        context.Value("i", 0),
        context.Value("b", 0, lock=False),
        context.Array("d", 4, lock=False),
    ]
    for thing in held:
        assert _is_process_furniture(thing), type(thing)

    class Client:
        def __init__(self) -> None:
            self.pipe = writer
            self.channel = held[-1]
            self.generations = 3

    client = Client()
    assert _holds_furniture(client)
    captured = _organ_state(client)
    gc.collect()
    assert "pipe" not in captured and "channel" not in captured
    writer.send("still open")
    assert reader.recv() == "still open"


def test_a_model_client_is_never_copied_or_rewound() -> None:
    """A restore once orphaned its request lock; see mlx_client_worker_lifecycle.py."""
    from core.brain.llm.mlx_client import MLXLocalClient
    from core.subject.copies import _is_process_furniture

    assert issubclass(MLXLocalClient, OwnsDescriptors)
    client = _inert(MLXLocalClient)
    assert copy.deepcopy(client) is client
    assert _is_process_furniture(client)
