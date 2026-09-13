"""A result assembled into a local mapping that nothing ever reads.

The half-wired shape: a function fills in a dictionary whose keys read like a
channel, and drops it. The computation behind every value ran, the tests that
cover that computation pass, and nothing downstream ever sees a number.

The audit has to see a closure read, because that is how the one real mapping
in the subject core is wired: `turn_once` builds `env` and the `capture`
defined inside it hands `env` to every reading of the state.
"""

from __future__ import annotations

import ast

from tools.audit_return_paths_into_local_dicts import (
    MIN_KEYS,
    SCANNED,
    _dict_locals,
    _read,
    _written,
    findings,
)

EXAMPLE = '''
def dropped():
    results = {}
    results["valence"] = compute_valence()
    results["arousal"] = compute_arousal()
    return None


def returned():
    results = {}
    results["valence"] = compute_valence()
    results["arousal"] = compute_arousal()
    return results


def read_by_a_closure():
    env = {"turn": 0.0}
    env["objective_len"] = 12.0
    env.update(extra())

    def capture(tag):
        return read(tag, env)

    return capture


def handed_onward():
    row = {}
    row["phi"] = 0.1
    row["leak"] = 0.2
    publish(row)


def one_key_is_a_lookup():
    table = {}
    table["only"] = 1
'''


def _functions():
    return {node.name: node for node in ast.parse(EXAMPLE).body}


def test_nothing_in_the_tree_drops_a_mapping_it_filled_in() -> None:
    loose = findings()
    assert not loose, "dropped return paths: " + "; ".join(
        f"{item['file']}:{item['line']} {item['function']} builds {item['name']}"
        for item in loose[:6]
    )


def test_the_rule_can_fire() -> None:
    """A rule with no worked example reports green whatever happens."""
    dropped = _functions()["dropped"]
    assert list(_dict_locals(dropped)) == ["results"]
    assert _written(dropped, "results") == ["valence", "arousal"]
    assert _read(dropped, "results") is False


def test_a_returned_mapping_is_wired() -> None:
    returned = _functions()["returned"]
    assert _read(returned, "results") is True


def test_a_closure_read_is_a_read() -> None:
    """`turn_once` builds `env` and `capture`, defined inside it, is what reads
    it. A walk that stops at the nested function calls the one live mapping in
    the driver dead."""
    closure = _functions()["read_by_a_closure"]
    assert "env" in _dict_locals(closure)
    assert _read(closure, "env") is True


def test_an_argument_is_a_read() -> None:
    onward = _functions()["handed_onward"]
    assert _read(onward, "row") is True


def test_a_mapping_with_one_key_is_a_lookup_not_a_result() -> None:
    lookup = _functions()["one_key_is_a_lookup"]
    assert len(_written(lookup, "table")) < MIN_KEYS


def test_the_scan_covers_where_a_dropped_path_costs_something() -> None:
    assert "core/phases/" in SCANNED
    assert "core/subject/" in SCANNED
