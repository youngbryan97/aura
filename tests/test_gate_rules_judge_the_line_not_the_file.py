"""Four gate rules that were counting sentences instead of defects.

The pattern was the same in each: a text rule matched a WORD, the word turned
up constantly in prose that was doing its job, and the rule was quieted with a
file-name allowlist. A file-name exemption cannot tell a docstring from a
defect, and this repo's did not — it was covering a startup check that marked
itself passed when unimplemented, a "[DUMMY VOICE]" engine that reported
success, and a dead ``mock_hear`` seam.

  * hardcoded_local_path: 37 findings, 35 of them strings nothing opened, and
    the two genuine ones invisible inside the noise.
  * placeholder_stub_mock: 79 findings, its three loudest being an enum
    member documented "Not implemented", a docstring saying in capitals that
    a discovery step is unwritten, and a residual_risk line — the honesty
    mechanism working, punished.
  * pytest_skip_xfail: 31 findings, every one a conditional skip and not one
    xfail, so the count grew each time the suite learned to run somewhere new.
  * potential_secret: fake keys written so the redaction code could be tested.

Each rule now judges what the line IS.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.aura_enterprise_gate import (  # noqa: E402
    docstring_line_numbers,
    file_text_context,
)

# Built here rather than written as literals: a test proving the rule still
# catches hardcoded paths must not itself ship hardcoded paths.
_TMP = "/" + "tmp"
_USERS = "/" + "Users"


def _tree(source: str):
    import ast

    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _context(source: str):
    return file_text_context(_tree(source))


def _scan(source: str) -> set[int]:
    """Lines the gate would report, applying every discriminator."""
    from tools.aura_enterprise_gate import TEXT_PATTERNS, _local_path_is_inert

    pattern = TEXT_PATTERNS["hardcoded_local_path"]
    tree = _tree(source)
    prose = docstring_line_numbers(tree)
    context = file_text_context(tree)

    reported: set[int] = set()
    for line_no, line in enumerate(source.splitlines(), start=1):
        match = pattern.search(line)
        if match is None:
            continue
        if line_no in prose or line.lstrip().startswith("#"):
            continue
        if _local_path_is_inert(match.group(0), line_no, context):
            continue
        reported.add(line_no)
    return reported


class TestTheRuleStillCatchesTheRealThing:
    def test_a_developers_home_directory_is_always_a_finding(self) -> None:
        source = f'DOCS = "{_USERS}/bryan/Documents/report.pdf"\n'
        assert _scan(source) == {1}

    def test_a_home_directory_is_a_finding_even_as_inert_data(self) -> None:
        """No I/O anywhere near it. It is still one person's machine."""
        source = f'CASES = [("{_USERS}/bryan/models/a.bin", "expected")]\n'
        assert _scan(source) == {1}

    def test_writing_to_shared_temp_is_a_finding(self) -> None:
        source = f'with open("{_TMP}/aura-report.json", "w") as fh:\n    fh.write("x")\n'
        assert _scan(source) == {1}

    def test_a_temp_path_wrapped_in_path_is_still_seen(self) -> None:
        source = f'shutil.rmtree(Path("{_TMP}/aura-workdir"))\n'
        assert _scan(source) == {1}

    def test_a_temp_path_as_a_subprocess_cwd_is_seen(self) -> None:
        source = f'subprocess.run(["ls"], cwd="{_TMP}/aura-workdir")\n'
        assert _scan(source) == {1}

    def test_an_interpolated_temp_path_that_gets_opened_is_seen(self) -> None:
        source = f'open(f"{_TMP}/aura-{{pid}}.sock", "w")\n'
        assert _scan(source) == {1}


class TestInertDataIsNotADefect:
    def test_a_rejected_policy_input_is_not_a_finding(self) -> None:
        """The path exists so the policy can refuse it. It is never opened."""
        source = f'assert policy.allows("{_TMP}/attacker/git") is False\n'
        assert _scan(source) == set()

    def test_a_monkeypatched_return_value_is_not_a_finding(self) -> None:
        source = f'monkeypatch.setattr(mod, "_repo", lambda: Path("{_TMP}/model"))\n'
        assert _scan(source) == set()

    def test_a_fake_socket_name_is_not_a_finding(self) -> None:
        source = f'record = {{"socket_path": f"{_TMP}/control-{{n}}.sock"}}\n'
        assert _scan(source) == set()


class TestRedactionFixturesSurvive:
    def test_the_secret_and_its_assertion_are_both_exempt(self) -> None:
        source = (
            f'def test_it():\n'
            f'    err = OSError("{_USERS}/secret/path: token failed")\n'
            f'    assert "{_USERS}/secret" not in str(redact(err))\n'
        )
        assert _scan(source) == set()

    def test_the_exemption_costs_an_actual_assertion(self) -> None:
        """Without the "must not leak" line, the fixture is just a hardcoded
        path again. That is what keeps this from being a way to go quiet."""
        source = f'    err = OSError("{_USERS}/secret/path: token failed")\n'
        assert _scan(source) == {1}

    def test_a_substring_assertion_covers_the_longer_literal(self) -> None:
        source = (
            f'def test_it():\n'
            f'    raise RuntimeError("{_USERS}/secret/path leaked token=abc123")\n'
            f'    assert "secret" not in blob\n'
        )
        assert _scan(source) == set()

    def test_a_short_generic_assertion_does_not_grant_the_exemption(self) -> None:
        """"ok" must not become a licence to hardcode every path in a file."""
        source = (
            f'def test_it():\n'
            f'    p = "{_USERS}/bryan/models/a.bin"\n'
            f'    assert "ok" not in blob\n'
        )
        assert _scan(source) == {2}


class TestProseIsNotCode:
    def test_a_docstring_quoting_an_incident_is_not_a_finding(self) -> None:
        source = (
            f'"""The live error, verbatim:\n'
            f'\n'
            f"    No such file: '{_USERS}/bryan/Desktop/wallpaper.png'\n"
            f'"""\n'
        )
        assert _scan(source) == set()

    def test_a_comment_explaining_the_rule_is_not_a_finding(self) -> None:
        source = f"# a hardcoded {_USERS}/<name> breaks on every other machine\n"
        assert _scan(source) == set()

    def test_a_docstring_does_not_shield_the_code_beneath_it(self) -> None:
        source = (
            f'def f():\n'
            f'    """Explains {_USERS}/bryan/notes."""\n'
            f'    return "{_USERS}/bryan/notes"\n'
        )
        assert _scan(source) == {3}


class TestTheDiscriminatorsAreHonestAboutTheirLimits:
    def test_an_unparseable_file_claims_nothing(self) -> None:
        """Silently returning "no findings" on a syntax error would let a
        broken file suppress its own defects. Empty sets mean the exemptions
        do not apply, so every match is reported."""
        source = f'def broken(\n    x = "{_TMP}/aura-out.json"\n'
        context = _context(source)
        assert docstring_line_numbers(_tree(source)) == set()
        assert context.disk_lines is None
        assert context.redaction_evidence == ()
        assert _scan(source) == {2}

    def test_a_path_bound_to_a_name_first_is_missed(self) -> None:
        """Documented, deliberate: only direct operands are traced. Recording
        the gap here means the next person finds it as a known limit rather
        than as a surprise."""
        source = f'p = "{_TMP}/aura-out.json"\nopen(p, "w")\n'
        assert _scan(source) == set()


def _scan_markers(source: str, rel: str = "core/example.py") -> set[int]:
    """Lines the placeholder/stub rule would report."""
    from tools.aura_enterprise_gate import TEXT_PATTERNS, _marker_is_not_a_claim

    pattern = TEXT_PATTERNS["placeholder_stub_mock"]
    tree = _tree(source)
    prose = docstring_line_numbers(tree)
    context = file_text_context(tree)

    reported: set[int] = set()
    for line_no, line in enumerate(source.splitlines(), start=1):
        if pattern.search(line) is None:
            continue
        if line_no in prose or line.lstrip().startswith("#"):
            continue
        if _marker_is_not_a_claim(line_no, rel, context):
            continue
        reported.add(line_no)
    return reported


class TestAnAdmissionIsNotTheDefect:
    """The rule's three loudest findings were the honesty mechanism working.

    An enum member documented "Not implemented" so a caller cannot mistake a
    digest for a signature; a module docstring saying in capitals that its
    discovery step is unwritten; a residual_risk line in a threat register.
    Flagging those puts the gate's weight behind deleting them.
    """

    def test_a_docstring_admitting_a_gap_is_not_a_finding(self) -> None:
        source = '"""Honest status: the discovery step is NOT IMPLEMENTED."""\n'
        assert _scan_markers(source) == set()

    def test_a_comment_admitting_a_gap_is_not_a_finding(self) -> None:
        source = "# Bytes match a digest signed by a trusted key. Not implemented.\n"
        assert _scan_markers(source) == set()

    def test_code_that_behaves_as_complete_is_still_a_finding(self) -> None:
        source = 'logger.info("[DUMMY VOICE]: %s", text)\n'
        assert _scan_markers(source) == {1}


class TestADetectorMaySpellWhatItHunts:
    def test_a_marker_collection_is_vocabulary(self) -> None:
        source = 'MARKERS = ("not implemented", "placeholder", "stub")\n'
        assert _scan_markers(source) == set()

    def test_a_regex_of_markers_is_vocabulary(self) -> None:
        source = 'SCAFFOLD_RE = re.compile(r"\\b(stub|placeholder|dummy)\\b")\n'
        assert _scan_markers(source) == set()

    def test_a_dict_key_names_a_field(self) -> None:
        source = 'row = {"placeholder": "missing_operand"}\n'
        assert _scan_markers(source) == set()

    def test_a_variable_named_after_the_thing_detected_is_not_a_claim(self) -> None:
        source = "if placeholder:\n    placeholder_detected = True\n"
        assert _scan_markers(source) == set()

    def test_a_marker_returned_as_a_value_is_still_a_claim(self) -> None:
        """Vocabulary names things. A returned value asserts one."""
        source = 'def render():\n    return "placeholder"\n'
        assert _scan_markers(source) == {2}


class TestTestsAreWhereDoublesBelong:
    def test_a_stub_in_a_test_is_the_implementation(self) -> None:
        source = 'model_dir.write_bytes(b"placeholder-bytes")\n'
        assert _scan_markers(source, rel="tests/test_thing.py") == set()

    def test_the_same_line_in_product_code_is_a_finding(self) -> None:
        source = 'model_dir.write_bytes(b"placeholder-bytes")\n'
        assert _scan_markers(source, rel="core/thing.py") == {1}


class TestNotImplementedIsAProtocolNotAConfession:
    def test_the_binary_operator_singleton_is_not_matched(self) -> None:
        """``return NotImplemented`` from __eq__ is the correct answer for an
        unrelated type. All five occurrences in this repo were exactly that."""
        from tools.aura_enterprise_gate import TEXT_PATTERNS

        assert TEXT_PATTERNS["placeholder_stub_mock"].search("return NotImplemented") is None

    def test_the_english_phrase_still_is(self) -> None:
        from tools.aura_enterprise_gate import TEXT_PATTERNS

        assert TEXT_PATTERNS["placeholder_stub_mock"].search('msg = "not implemented"')


def _scan_skips(source: str) -> set[int]:
    """Lines the skip/xfail rule would report."""
    from tools.aura_enterprise_gate import TEXT_PATTERNS, file_text_context

    pattern = TEXT_PATTERNS["pytest_skip_xfail"]
    context = file_text_context(_tree(source))

    reported: set[int] = set()
    for line_no, line in enumerate(source.splitlines(), start=1):
        if pattern.search(line) is None:
            continue
        if (
            "pytest.skip" in line
            and "pytest.mark.skip" not in line
            and line_no not in context.unconditional_skip_lines
        ):
            continue
        reported.add(line_no)
    return reported


class TestAPreconditionIsNotParkedDebt:
    """31 findings, every one a conditional skip, and not one xfail.

    No fork on this platform, no node installed, vm_stat absent, a symlink
    that would not create: the count grew every time the suite learned to run
    somewhere new, which is the opposite of a debt signal.
    """

    def test_a_skip_behind_a_condition_is_not_reported(self) -> None:
        source = (
            "import pytest\n"
            "def test_x():\n"
            "    if not shutil.which('node'):\n"
            "        pytest.skip('node is not installed')\n"
            "    assert run()\n"
        )
        assert _scan_skips(source) == set()

    def test_a_skip_in_an_except_handler_is_not_reported(self) -> None:
        """The capability probe that answers by raising."""
        source = (
            "import pytest\n"
            "def test_x():\n"
            "    try:\n"
            "        link.symlink_to('/etc/passwd')\n"
            "    except OSError:\n"
            "        pytest.skip('symlinks unavailable')\n"
        )
        assert _scan_skips(source) == set()

    def test_skipif_is_the_conditional_form_and_is_not_reported(self) -> None:
        """``pytest.mark.skip`` matched ``skipif`` by prefix before this."""
        source = (
            "import pytest\n"
            "@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS only')\n"
            "def test_x():\n"
            "    assert True\n"
        )
        assert _scan_skips(source) == set()


class TestAnUnguardedSkipIsStillReported:
    def test_a_bare_skip_in_a_test_body_is_reported(self) -> None:
        """Fires every run, so the assertions below never execute anywhere."""
        source = "import pytest\ndef test_x():\n    pytest.skip('later')\n    assert False\n"
        assert _scan_skips(source) == {3}

    def test_a_module_level_skip_is_reported(self) -> None:
        source = "import pytest\npytest.skip('whole file parked')\n"
        assert _scan_skips(source) == {2}

    def test_a_bare_skip_inside_a_class_is_reported(self) -> None:
        source = (
            "import pytest\n"
            "class TestX:\n"
            "    def test_y(self):\n"
            "        pytest.skip('later')\n"
        )
        assert _scan_skips(source) == {4}

    def test_the_unconditional_marker_is_reported(self) -> None:
        source = "import pytest\n@pytest.mark.skip(reason='later')\ndef test_x():\n    pass\n"
        assert _scan_skips(source) == {2}

    def test_xfail_is_always_reported(self) -> None:
        """A parked failure, conditional or not. The repo has none."""
        source = "import pytest\n@pytest.mark.xfail\ndef test_x():\n    assert False\n"
        assert _scan_skips(source) == {2}


class TestTheFileNameAllowlistIsGone:
    """It exempted forty-one files and hid three real defects doing it.

    Every rule it covered now judges the line, so there is nothing left for
    it to do. This asserts it stays gone: reaching for a file-name exemption
    is the move that made a scanner unreadable in the first place.
    """

    def test_the_gate_has_no_file_name_exemption_list(self) -> None:
        from tools import aura_enterprise_gate

        assert not hasattr(aura_enterprise_gate, "SELF_DESCRIPTIVE_PATTERN_FILES")

    def test_the_gate_does_not_report_its_own_pattern_strings(self) -> None:
        """It used to need an exemption for exactly this."""
        from pathlib import Path

        from tools.aura_enterprise_gate import GateReport, rel_path, scan_file

        root = Path(__file__).resolve().parents[1]
        gate = root / "tools" / "aura_enterprise_gate.py"
        report = GateReport(root=str(root), generated_at_unix=0.0)
        scan_file(gate, root, report)

        noisy = [
            f
            for f in report.findings
            if f.kind in {"pytest_skip_xfail", "placeholder_stub_mock", "potential_secret"}
        ]
        assert noisy == [], [(f.kind, f.line, f.detail) for f in noisy]
        assert rel_path(gate, root) == "tools/aura_enterprise_gate.py"

    def test_this_test_file_is_not_reported_either(self) -> None:
        """It is full of sample source containing every marker there is."""
        from pathlib import Path

        from tools.aura_enterprise_gate import GateReport, scan_file

        root = Path(__file__).resolve().parents[1]
        report = GateReport(root=str(root), generated_at_unix=0.0)
        scan_file(Path(__file__).resolve(), root, report)

        noisy = [f for f in report.findings if f.kind != "syntax_error"]
        assert noisy == [], [(f.kind, f.line, f.detail) for f in noisy]


def _scan_subprocess(source: str, rel: str) -> set[int]:
    """Lines the subprocess rule would report."""
    import ast

    from tools.aura_enterprise_gate import AstGate, GateReport

    report = GateReport(root=".", generated_at_unix=0.0)
    AstGate(rel, report, source_lines=source.splitlines()).visit(ast.parse(source))
    return {f.line for f in report.findings if f.kind == "subprocess_usage_review"}


class TestTheOrgansGoThroughTheGateway:
    """A ninety-entry allowlist, and most of its files had no spawn left in
    them at all. What it was really encoding is that core/ and interface/ must
    route through core/runtime/subprocess_gateway.py — which carries the
    source label, the accelerator claim, the shutdown interlock and the
    read-only assertion — and that everything else spawns by design.
    """

    def test_a_direct_spawn_in_core_is_reported(self) -> None:
        source = 'subprocess.run(["git", "status"])\n'
        assert _scan_subprocess(source, "core/thing.py") == {1}

    def test_a_direct_spawn_in_the_interface_is_reported(self) -> None:
        source = 'subprocess.Popen(["node", "x.js"])\n'
        assert _scan_subprocess(source, "interface/thing.py") == {1}

    def test_the_gateway_itself_may_spawn(self) -> None:
        source = 'subprocess.run(command)\n'
        assert _scan_subprocess(source, "core/runtime/subprocess_gateway.py") == set()

    def test_a_test_spawning_a_real_child_is_not_reported(self) -> None:
        """Containment can only be proven against a process you can kill."""
        source = 'subprocess.run([sys.executable, "-c", "import os; os._exit(9)"])\n'
        assert _scan_subprocess(source, "tests/test_containment.py") == set()

    def test_an_operator_driver_is_not_reported(self) -> None:
        source = 'subprocess.run(["make", "smoke"])\n'
        assert _scan_subprocess(source, "tools/release_preflight.py") == set()

    def test_the_launcher_may_spawn_the_sentinels_that_outlive_it(self) -> None:
        source = 'subprocess.Popen([sys.executable, "tools/memory_sentinel.py"])\n'
        assert _scan_subprocess(source, "aura_main.py") == set()

    def test_a_reviewed_line_in_core_is_not_reported(self) -> None:
        """Per-line, like the exec and broad-except markers: blessing a whole
        file also blesses every spawn added to it later."""
        source = 'subprocess.run(argv)  # noqa: S603 - detached verifier, identity-asserted\n'
        assert _scan_subprocess(source, "core/learning/thing.py") == set()

    def test_the_marker_does_not_bless_the_rest_of_the_file(self) -> None:
        source = (
            'subprocess.run(argv)  # noqa: S603 - reviewed\n'
            'subprocess.run(other)\n'
        )
        assert _scan_subprocess(source, "core/learning/thing.py") == {2}

    def test_shell_true_is_still_critical_everywhere(self) -> None:
        import ast

        from tools.aura_enterprise_gate import AstGate, GateReport

        source = 'subprocess.run("rm -rf /", shell=True)  # noqa: S603 - reviewed\n'
        report = GateReport(root=".", generated_at_unix=0.0)
        AstGate("tests/test_x.py", report, source_lines=source.splitlines()).visit(
            ast.parse(source)
        )
        kinds = {(f.kind, f.severity) for f in report.findings}
        assert ("subprocess_shell_true", "critical") in kinds


def _scan_raise_only(source: str, rel: str = "core/thing.py") -> set[str]:
    """Function names the raise-only rule would report."""
    import ast

    from tools.aura_enterprise_gate import AstGate, GateReport

    report = GateReport(root=".", generated_at_unix=0.0)
    AstGate(rel, report, source_lines=source.splitlines()).visit(ast.parse(source))
    return {f.detail for f in report.findings if f.kind == "raise_only_function"}


class TestARefusalIsNotUnfinishedWork:
    """118 findings and not one unwritten function.

    104 were ``_fail(code)`` helpers — the standard way to fail closed with a
    named, greppable code — and the rest were json.loads reject hooks and
    protocol methods that exist to refuse a direct call. The rule was firing
    at the discipline it is supposed to protect.
    """

    def test_a_never_annotated_helper_is_a_declared_contract(self) -> None:
        source = (
            "def _fail(code: str) -> Never:\n"
            "    raise ResumeVerifierError(str(code))\n"
        )
        assert _scan_raise_only(source) == set()

    def test_noreturn_counts_too(self) -> None:
        source = "def _fail(code: str) -> NoReturn:\n    raise VerifierError(code)\n"
        assert _scan_raise_only(source) == set()

    def test_a_named_domain_error_is_a_decision(self) -> None:
        """No annotation, but choosing WHICH failure this is takes a decision.
        The json.loads reject hooks look like this."""
        source = 'def reject_constant(value: str):\n    raise RealityActuationError(value)\n'
        assert _scan_raise_only(source) == set()

    def test_an_unwritten_function_is_still_reported(self) -> None:
        """What the rule was always after."""
        source = "def compute(self):\n    raise NotImplementedError\n"
        assert _scan_raise_only(source) == {"compute"}

    def test_an_abstract_method_may_raise_notimplemented(self) -> None:
        source = (
            "class A:\n"
            "    @abstractmethod\n"
            "    def compute(self):\n"
            "        raise NotImplementedError\n"
        )
        assert _scan_raise_only(source) == set()

    def test_a_bare_reraise_body_is_still_reported(self) -> None:
        """Nothing named, nothing declared: it raises whatever is in flight,
        and outside a handler there is nothing in flight."""
        source = "def guard(self):\n    raise\n"
        assert _scan_raise_only(source) == {"guard"}

    def test_an_async_refusal_is_treated_the_same(self) -> None:
        source = (
            "async def apply(self, effect) -> dict[str, Any]:\n"
            "    raise HomeAssistantRealityError('use the transaction')\n"
        )
        assert _scan_raise_only(source) == set()

    def test_an_async_unwritten_method_is_still_reported(self) -> None:
        source = "async def apply(self, effect):\n    raise NotImplementedError\n"
        assert _scan_raise_only(source) == {"apply"}


def _scan_loops(source: str, rel: str = "core/thing.py") -> set[int]:
    """Lines the unbounded-loop rule would report."""
    import ast

    from tools.aura_enterprise_gate import AstGate, GateReport

    report = GateReport(root=".", generated_at_unix=0.0)
    AstGate(rel, report, source_lines=source.splitlines()).visit(ast.parse(source))
    return {f.line for f in report.findings if f.kind == "unbounded_loop_review"}


class TestWhileTrueIsAnIdiomNotADefect:
    """30 of 31 findings had a break or a return a few lines down.

    ``while True`` is how Python spells a loop whose exit condition is
    computed inside the body. The defect is a loop with no way out at all.
    """

    def test_a_loop_with_a_break_is_bounded(self) -> None:
        source = "while True:\n    if done():\n        break\n"
        assert _scan_loops(source) == set()

    def test_a_loop_that_returns_is_bounded(self) -> None:
        source = "def f():\n    while True:\n        if done():\n            return 1\n"
        assert _scan_loops(source) == set()

    def test_a_loop_that_raises_is_bounded(self) -> None:
        source = "while True:\n    raise Timeout()\n"
        assert _scan_loops(source) == set()

    def test_an_awaiting_service_loop_is_bounded_by_cancellation(self) -> None:
        """How every service loop in this runtime is actually stopped."""
        source = "async def f():\n    while True:\n        item = await queue.get()\n"
        assert _scan_loops(source) == set()

    def test_a_generator_is_bounded_by_its_consumer(self) -> None:
        source = "def f():\n    while True:\n        yield next_value()\n"
        assert _scan_loops(source) == set()

    def test_a_spin_is_reported(self) -> None:
        source = "while True:\n    counter += 1\n"
        assert _scan_loops(source) == {1}

    def test_a_break_from_a_NESTED_loop_does_not_bound_the_outer_one(self) -> None:
        """The subtlety that makes this an AST question and not a grep."""
        source = "while True:\n    for item in items:\n        break\n"
        assert _scan_loops(source) == {1}

    def test_a_return_inside_a_nested_function_does_not_bound_it(self) -> None:
        source = "while True:\n    def helper():\n        return 1\n    helper()\n"
        assert _scan_loops(source) == {1}

    def test_an_exit_inside_a_handler_still_counts(self) -> None:
        source = "while True:\n    try:\n        work()\n    except OSError:\n        break\n"
        assert _scan_loops(source) == set()


def _scan_blocking_sleep(source: str, rel: str = "core/thing.py") -> set[int]:
    """Lines the blocking-sleep rule would report."""
    import ast

    from tools.aura_enterprise_gate import AstGate, GateReport

    report = GateReport(root=".", generated_at_unix=0.0)
    AstGate(rel, report, source_lines=source.splitlines()).visit(ast.parse(source))
    return {f.line for f in report.findings if f.kind == "blocking_sleep_in_async"}


class TestADeliberateStallIsMarkedOnItsLine:
    """A test whose subject is a stalled event loop has to be able to stall one.

    The only escape the rule had was a file-name allowlist, and that blesses
    every sleep added to the file afterwards — which is the debt the rule
    exists to catch. `# noqa: ASYNC251` is the ecosystem-standard annotation
    for this call and it names the one line a human read.
    """

    def test_a_blocking_sleep_inside_async_is_reported(self) -> None:
        source = "import time\n\n\nasync def go():\n    time.sleep(0.1)\n"
        assert _scan_blocking_sleep(source) == {5}

    def test_the_reviewed_marker_clears_that_line_and_no_other(self) -> None:
        source = (
            "import time\n"
            "\n"
            "\n"
            "async def go():\n"
            "    time.sleep(0.1)  # noqa: ASYNC251 - the stalled loop is the subject\n"
            "    time.sleep(0.2)\n"
        )
        assert _scan_blocking_sleep(source) == {6}

    def test_a_sleep_outside_async_was_never_the_rule(self) -> None:
        source = "import time\n\n\ndef go():\n    time.sleep(0.1)\n"
        assert _scan_blocking_sleep(source) == set()
