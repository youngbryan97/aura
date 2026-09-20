"""An arm records what served it, so two arms can be shown to be the same.

The paired design assumes the arms differ only in the displacement. Through the
offline stub that is nearly free. Through a real cortex it is not: a fallback
lane taking one arm, a chat template edited between them, a pointer swung to a
different checkpoint, a module reloaded — each makes the two arms two systems,
and none of it can be recovered from a report that never wrote it down.

P64.5 asks for the weights and the code to be identical across arms, and P64.6
for a sham-versus-sham floor in the cortex-inclusive campaign as well. Both are
requirements on the protocol rather than results of a run, so both are checked
here and by the preflight.
"""

from __future__ import annotations

from dataclasses import replace

from core.subject.arm_identity import ArmIdentity, differences, pin_arm_identity


def test_arm_commit_probe_uses_the_process_owner(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    from core.subject import arm_identity

    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout='a' * 40 + '\n')
    monkeypatch.setattr(arm_identity, 'get_subprocess_gateway', lambda: SimpleNamespace(run=run))
    assert arm_identity._commit() == 'a' * 40
    assert calls[0][0] == ['git', "-c", "core.fsmonitor=false", 'rev-parse', 'HEAD']
    assert calls[0][1]['read_only'] is True
    assert calls[0][1]['accelerator_capability'] == 'none'
    assert calls[0][1]['cwd'] == Path(arm_identity.__file__).resolve().parents[2]


def test_a_pin_is_stable_when_nothing_has_happened() -> None:
    assert differences(pin_arm_identity(), pin_arm_identity()) == []


def test_a_pin_survives_a_round_trip() -> None:
    pinned = pin_arm_identity()
    assert differences(pinned, ArmIdentity.from_dict(pinned.as_dict())) == []


def test_a_moved_pointer_is_named() -> None:
    first = pin_arm_identity()
    second = replace(first, pointer=first.pointer + "-other")
    assert any("pointer moved" in line for line in differences(first, second))


def test_edited_weights_are_named() -> None:
    first = pin_arm_identity()
    second = replace(first, weights_digest="deadbeef", weights_files=first.weights_files + 1)
    assert any("model directory changed" in line for line in differences(first, second))


def test_an_edited_template_is_named() -> None:
    first = replace(pin_arm_identity(), shaping={"chat_template.jinja": "aaa"})
    second = replace(first, shaping={"chat_template.jinja": "bbb"})
    assert differences(first, second) == ["chat_template.jinja changed between the arms"]


def test_a_free_sampler_between_arms_is_named() -> None:
    first = replace(pin_arm_identity(), decoding={"temperature": 0.0})
    second = replace(first, decoding={"temperature": 0.7})
    assert any("temperature changed" in line for line in differences(first, second))


def test_two_processes_on_two_commits_are_named() -> None:
    first = pin_arm_identity()
    second = replace(first, commit="0" * 40, process=first.process + 1)
    assert any("two processes on two commits" in line for line in differences(first, second))


def test_a_commit_moving_under_one_process_is_not_a_change() -> None:
    """The checkout can move without changing a line of what is executing.

    Committing while a run is in flight refused the run, which is a fact about
    the working tree rather than about the two arms.
    """
    first = pin_arm_identity()
    second = replace(first, commit="0" * 40)
    assert differences(first, second) == []


def test_later_imports_are_not_a_change() -> None:
    """A run imports more of itself as it goes, and that is not two systems."""
    first = pin_arm_identity()
    second = replace(first, modules_loaded=first.modules_loaded + 40)
    assert differences(first, second) == []


def test_a_module_from_somewhere_else_is_named() -> None:
    first = replace(pin_arm_identity(), foreign_modules={})
    second = replace(first, foreign_modules={"core.brain": "/other/checkout/core/brain.py"})
    assert differences(first, second) == [
        "core.brain was loaded from None and then from '/other/checkout/core/brain.py'"
    ]


def test_stub_against_cortex_is_two_systems() -> None:
    first = replace(pin_arm_identity(), stubbed=True)
    second = replace(first, stubbed=False)
    assert any("two systems" in line for line in differences(first, second))


def test_the_preflight_asks_both_questions() -> None:
    """P64.5 and P64.6 each get a row, whatever the answer is."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "cortex_campaign_preflight",
        Path(__file__).resolve().parents[1] / "tools" / "cortex_campaign_preflight.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.preflight()
    names = {row["check"] for row in report["checks"]}
    assert "arms identical" in names
    assert "sham against sham" in names


def test_the_cut_sweep_really_draws_a_second_sham() -> None:
    """The floor P64.6 asks for is in the protocol, not only in the preflight."""
    import inspect

    from core.subject import v25_cut

    source = inspect.getsource(v25_cut)
    assert 'samples["sham_a"]' in source
    assert 'samples["sham_b"]' in source


def test_an_unresolved_pointer_does_not_set_the_walk_loose(monkeypatch) -> None:
    """Path("") is Path("."), and the first version walked the repository.

    A run with no cortex hands back no spec. The pin has to record that as no
    weights rather than as a digest of whatever directory the process happens
    to be standing in, which does not finish.
    """
    from core.subject import arm_identity

    walked: list[str] = []
    real = arm_identity._tree_digest
    monkeypatch.setattr(
        arm_identity,
        "_tree_digest",
        lambda root: (walked.append(str(root)), real(root))[1],
    )
    pinned = arm_identity.pin_arm_identity(spec=None)
    assert pinned.model_path == ""
    assert pinned.weights_files == 0
    assert pinned.stubbed is True
    assert walked == [], f"the pin walked {walked}"


def test_a_deep_directory_is_read_as_far_as_the_bound_and_says_so(tmp_path) -> None:
    from core.subject.arm_identity import MAX_ARTIFACT_DEPTH, _tree_digest

    deep = tmp_path
    for level in range(MAX_ARTIFACT_DEPTH + 3):
        deep = deep / f"level{level}"
        deep.mkdir()
        (deep / "shard.bin").write_bytes(b"x")
    digest, files, truncated = _tree_digest(tmp_path)
    assert truncated is True
    assert files <= MAX_ARTIFACT_DEPTH + 1
    assert digest
