"""Source-text semantic context for Aura's enterprise gate."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

try:
    from tools.aura_enterprise_contracts import _TMP_PATH_PREFIX, TEXT_PATTERNS
except ModuleNotFoundError as exc:
    if exc.name not in {"tools", "tools.aura_enterprise_contracts"}:
        raise
    from aura_enterprise_contracts import _TMP_PATH_PREFIX, TEXT_PATTERNS

_FILESYSTEM_CALL_NAMES = frozenset(
    {
        "open", "makedirs", "mkdir", "rmdir", "remove", "unlink", "rename",
        "replace", "chdir", "symlink", "touch", "write_text", "write_bytes",
        "read_text", "read_bytes", "rmtree", "copy", "copy2", "copyfile",
        "copytree", "move", "listdir", "scandir", "walk", "glob", "rglob",
        "run", "Popen", "call", "check_call", "check_output",
        "create_subprocess_exec", "create_subprocess_shell",
        "NamedTemporaryFile", "TemporaryDirectory", "mkstemp", "mkdtemp",
    }
)
_FILESYSTEM_KEYWORDS = frozenset({"cwd", "dir", "path", "filename", "file"})
#: Names that mean a value is being USED as a credential — bound to one, or
#: handed to something that authenticates with it. The same distinction the
#: /tmp/ rule already draws with ``disk_lines``: a credential-shaped literal
#: nothing authenticates with is a fixture, exactly as a path nothing opens
#: is not a world-writable-directory defect.
_CREDENTIAL_NAME_PARTS = (
    "key", "token", "secret", "password", "passwd", "credential",
    "auth", "bearer", "api_key", "apikey", "access", "session",
)
_PASSTHROUGH_CALL_NAMES = frozenset({"Path", "PurePath", "str", "fspath"})
_DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
#: Rules that describe something written in the file rather than something
#: the program DOES, and are therefore exempt inside a docstring or comment.
#:
#: For paths and keys the argument is direct: a literal quoted in prose is not
#: a path the program uses, it is usually the verbatim text of the incident
#: the module exists to prevent.
#:
#: placeholder_stub_mock belongs here for a sharper reason. Its three loudest
#: findings were an enum member documented "Not implemented" so a caller
#: cannot mistake a digest for a signature, a module docstring that says in
#: capitals that its discovery step is NOT IMPLEMENTED (with a False flag and
#: a refusing entry point beneath it), and a ``residual_risk`` line in a
#: threat register. Every one is the honesty mechanism working. Flagging the
#: admission puts the gate's weight behind deleting it, which is how a repo
#: ends up with silent stubs and a clean report — the exact failure this gate
#: exists to prevent. The defect is code that BEHAVES as though it were
#: complete, and that is what the rule now looks for.
#: pytest_skip_xfail is here too: a comment saying "pytest.skip" skips
#: nothing. It was the last thing keeping the file-name allowlist alive.
_PROSE_SENSITIVE_KINDS = frozenset(
    {
        "hardcoded_local_path",
        "potential_secret",
        "placeholder_stub_mock",
        "pytest_skip_xfail",
    }
)


@dataclass
class FileTextContext:
    """What a file's syntax says about the text the line rules matched.

    Built from a single AST walk, because the gate scans several thousand
    files and each extra pass over the tree costs real seconds on the clock
    the pre-commit gate runs against.
    """

    #: Lines where a path literal is handed to something that touches the
    #: disk. ``None`` means the file would not parse, so nothing is known.
    disk_lines: set[int] | None = None
    #: Strings the file asserts must NOT appear in some output.
    redaction_evidence: tuple[str, ...] = ()
    #: Lines where a credential-shaped literal is bound to a credential name
    #: or passed as one. ``None`` means the file would not parse, so nothing
    #: is known and every match is reported.
    credential_use_lines: set[int] | None = None
    #: Lines carrying a stub/placeholder marker inside a string CONSTANT. A
    #: line that matches the text rule and is absent here carries the marker
    #: in an identifier instead.
    marker_string_lines: set[int] = field(default_factory=set)
    #: Lines where every such string is used as a NAME, a KEY or a PATTERN —
    #: a detector's vocabulary rather than a claim about this code.
    marker_vocabulary_lines: set[int] = field(default_factory=set)
    #: Lines inside a branch whose test reads the marker — ``if arm.stubbed:``
    #: — where the string is a detector reporting what it found.
    marker_detection_lines: set[int] = field(default_factory=set)
    #: Lines calling ``pytest.skip()`` with nothing guarding the call.
    unconditional_skip_lines: set[int] = field(default_factory=set)
    #: Lines where a skip marker sits INSIDE a string — sample source in a
    #: test for the rule itself, or the rule's own pattern. Not a skip.
    quoted_skip_lines: set[int] = field(default_factory=set)


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _path_shaped_constants(node: ast.AST) -> Iterator[ast.Constant]:
    """String constants inside `node` that look like a local path.

    Descends through the wrappers that do not themselves touch the disk —
    ``Path(...)``, ``str(...)``, an f-string, a ``/`` join — so that
    ``open(Path("/tmp/x"))`` is recognised while a bare ``Path("/tmp/x")``
    assigned to a name is not.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str) and TEXT_PATTERNS["hardcoded_local_path"].search(
            node.value
        ):
            yield node
        return
    if isinstance(node, ast.JoinedStr):
        for value in node.values:
            yield from _path_shaped_constants(value)
        return
    if isinstance(node, ast.FormattedValue):
        return
    if isinstance(node, ast.Call) and _call_name(node) in _PASSTHROUGH_CALL_NAMES:
        for arg in node.args:
            yield from _path_shaped_constants(arg)
        return
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Div)):
        yield from _path_shaped_constants(node.left)
        yield from _path_shaped_constants(node.right)


def docstring_line_numbers(tree: ast.AST | None) -> set[int]:
    """Lines occupied by docstrings.

    A path or a key quoted inside a docstring is PROSE — usually the verbatim
    text of the incident the module exists to prevent.
    ``tests/test_fetched_image_path_is_resolved.py`` opens by quoting the live
    error, complete with the absolute path that broke. That is the evidence,
    not a dependency on one machine, and flagging it pressures the next person
    to delete the record to quiet the gate.

    Walks statement containers only, never expressions: this runs on every
    file that trips any text rule, and ``ast.walk`` over whole expression
    trees was costing more than the rule it serves.
    """
    lines: set[int] = set()
    if tree is None:
        return lines
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        body = getattr(node, "body", None)
        if isinstance(node, _DOCSTRING_OWNERS) and isinstance(body, list) and body:
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                start = int(getattr(first, "lineno", 0) or 0)
                end = int(getattr(first, "end_lineno", start) or start)
                lines.update(range(start, end + 1))
        for child in ast.iter_child_nodes(node):
            if isinstance(getattr(child, "body", None), list):
                stack.append(child)
    lines.update(_multiline_string_lines(tree))
    return lines


def _multiline_string_lines(tree: ast.AST | None) -> set[int]:
    """Lines inside a string constant that spans more than one line.

    A docstring is a string constant that happens to sit in statement
    position, so the docstring exemption above was always the special case of
    a general rule: text inside quotes is data, not something the program
    does. The general rule matters because the files most likely to quote a
    forbidden pattern are the tests that prove the rule against it works, and
    a gate that reports its own fixtures is a gate on its way to a file-name
    allowlist. ``_quoted_skip_lines`` already reached this conclusion for
    pytest skips; this is the same conclusion, applied once instead of once
    per rule.

    Only MULTI-line literals qualify. ``STATE_ROOT = "/Users/bryan/.aura"`` is
    a single-line constant and stays reported — it is a real path this module
    would really use.
    """
    lines: set[int] = set()
    if tree is None:
        return lines
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        start = int(getattr(node, "lineno", 0) or 0)
        end = int(getattr(node, "end_lineno", start) or start)
        if end > start:
            lines.update(range(start, end + 1))
    return lines


_REGEX_CALL_NAMES = frozenset(
    {"compile", "search", "match", "fullmatch", "findall", "finditer", "sub", "split"}
)
_NAME_LOOKUP_CALL_NAMES = frozenset({"getattr", "hasattr", "setattr", "get", "pop"})


_GUARDING_STATEMENTS = (ast.If, ast.Try, ast.While, ast.For, ast.AsyncFor, ast.Match)


def _quoted_skip_lines(tree: ast.AST) -> set[int]:
    """Lines where a skip marker is inside a string constant.

    A rule that reports its own pattern, and reports the fixtures of the test
    that proves the pattern works, is how a file-name allowlist gets born.
    Sample source quoted in a test is data.
    """
    pattern = TEXT_PATTERNS["pytest_skip_xfail"]
    lines: set[int] = set()
    for node in ast.walk(tree):
        text = _marker_text(node)
        if text and pattern.search(text):
            lines.add(int(getattr(node, "lineno", 0) or 0))
            end = int(getattr(node, "end_lineno", 0) or 0)
            lines.update(range(int(getattr(node, "lineno", 0) or 0), end + 1))
    return lines


def _unconditional_skip_lines(tree: ast.AST) -> set[int]:
    """Lines where ``pytest.skip()`` runs with nothing deciding whether to.

    A skip guarded by a condition is how pytest spells a precondition, and
    the suite is full of honest ones: no fork on this platform, no node
    installed, vm_stat absent, a symlink that would not create. Counting
    those made the rule grow every time the suite learned to run somewhere
    new, which is the opposite of a debt signal.

    An UNGUARDED skip is different. It fires every run, so the assertions
    below it never execute anywhere — a parked failure wearing a precondition
    as a disguise. Same for ``pytest.mark.skip`` (as opposed to ``skipif``)
    and for ``xfail``, both of which the line rule still catches on sight.
    """
    lines: set[int] = set()

    def is_skip_call(node: ast.AST) -> bool:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            return False
        func = node.value.func
        return (
            isinstance(func, ast.Attribute)
            and func.attr == "skip"
            and isinstance(func.value, ast.Name)
            and func.value.id == "pytest"
        )

    def walk(body: list[ast.stmt], guarded: bool) -> None:
        for statement in body:
            if is_skip_call(statement) and not guarded:
                lines.add(int(getattr(statement, "lineno", 0) or 0))
                continue
            inner_guarded = guarded or isinstance(statement, _GUARDING_STATEMENTS)
            for name in ("body", "orelse", "finalbody", "handlers", "cases"):
                child = getattr(statement, name, None)
                if isinstance(child, list) and child and isinstance(child[0], ast.AST):
                    if isinstance(child[0], ast.stmt):
                        walk(child, inner_guarded)
                    else:
                        for sub in child:
                            walk(getattr(sub, "body", []), inner_guarded)

    walk(getattr(tree, "body", []), guarded=False)
    return lines


def _marker_text(node: ast.AST) -> str:
    """The text of a str or bytes constant, or "" for anything else.

    Bytes count. ``b"placeholder-bytes"`` written as a fixture's file content
    is a value the program produces, exactly like the str form, and treating
    it as "not a string" made it look like a bare identifier and slip the
    rule.
    """
    if not isinstance(node, ast.Constant):
        return ""
    value = node.value
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return ""


def _vocabulary_string_lines(tree: ast.AST) -> set[int]:
    """Lines whose marker string is a name, a key or a pattern.

    A scanner has to spell the words it hunts for. ``"placeholder"`` sitting
    in ``MARKERS = (...)``, inside ``re.compile(...)``, or as the key of a
    dict is the detector's vocabulary — it says nothing about whether THIS
    module is finished. Flagging it is how a repo ends up with a file-name
    allowlist, and a file-name allowlist hides the real thing: two genuine
    findings were sitting behind this repo's, a "[DUMMY VOICE]" fallback and
    a "Mock hear" path, both in shipping code.

    A string that carries a ``{slot}`` of its own alongside the word is
    talking about format-template syntax — "command_template must contain
    exactly one {value} placeholder" is a name for a brace pair, not an
    admission. The brace has to be in the SAME literal, so a message that
    merely happens to be an f-string does not qualify.

    A marker used as a VALUE — returned, assigned, or handed to a message —
    is not covered here, because that is the module speaking about itself.
    """
    marker = TEXT_PATTERNS["placeholder_stub_mock"]
    lines: set[int] = set()

    def note(node: ast.AST) -> None:
        if marker.search(_marker_text(node)):
            lines.add(int(getattr(node, "lineno", 0) or 0))

    template_slot = re.compile(r"\{[A-Za-z_][A-Za-z_0-9]*\}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            text = _marker_text(node)
            if text and marker.search(text) and template_slot.search(text):
                lines.add(int(getattr(node, "lineno", 0) or 0))
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for element in node.elts:
                note(element)
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if key is not None:
                    note(key)
        elif isinstance(node, ast.Subscript):
            note(node.slice)
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name in _REGEX_CALL_NAMES or name in _NAME_LOOKUP_CALL_NAMES:
                for arg in node.args:
                    note(arg)
    return lines


def _detection_string_lines(tree: ast.AST) -> set[int]:
    """Lines of marker strings emitted by a branch that tested for the marker.

    ``if first.stubbed != second.stubbed: out.append("one arm ran on the
    stub and the other on a cortex")`` is a detector saying what it found.
    The gate's own rule is that the defect is code which BEHAVES as though
    it were complete; a branch that exists to notice the stand-in and say so
    is the opposite of that, and flagging it puts the gate's weight behind
    deleting the notice. The test has to read the marker as a NAME — an
    attribute or a variable — not merely contain the word in a string.
    """
    marker = TEXT_PATTERNS["placeholder_stub_mock"]
    # A name reads the marker as a stem: ``stubbed``, ``is_mock``,
    # ``placeholder_count``. The word-boundary form is for prose.
    named = re.compile(r"(?:placeholder|stub|mock|dummy)", re.IGNORECASE)
    lines: set[int] = set()

    def names_in(test: ast.AST) -> Iterator[str]:
        for node in ast.walk(test):
            if isinstance(node, ast.Name):
                yield node.id
            elif isinstance(node, ast.Attribute):
                yield node.attr

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not any(named.search(name) for name in names_in(node.test)):
            continue
        for inner in node.body:
            for constant in ast.walk(inner):
                if isinstance(constant, ast.Constant) and marker.search(_marker_text(constant)):
                    lines.add(int(getattr(constant, "lineno", 0) or 0))
    return lines


def _marker_string_lines(tree: ast.AST) -> set[int]:
    marker = TEXT_PATTERNS["placeholder_stub_mock"]
    lines: set[int] = set()
    for node in ast.walk(tree):
        if marker.search(_marker_text(node)):
            lines.add(int(getattr(node, "lineno", 0) or 0))
    return lines


def file_text_context(tree: ast.AST | None) -> FileTextContext:
    """Collect, in one walk, everything the path rule needs to know.

    Two questions, one traversal:

    * **What reaches the disk?** A shared-temp path is a hazard because the
      process WRITES there: a predictable name under a world-writable
      directory is a symlink-attack surface and a collision between two users
      on one host. A literal that is only compared against, rejected by a
      policy, or returned from a monkeypatched stub never becomes a file.
      Only direct operands are traced — a path bound to a name and opened
      three lines later is missed; this filters noise, it does not prove
      absence.
    * **What is redaction evidence?** A scrubber test has to name the secret
      it proves gets removed, and both the fixture and its assertion match
      the path rule. Deleting either destroys the proof.
    """
    context = FileTextContext()
    if tree is None:
        return context
    context.disk_lines = set()
    evidence: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in _FILESYSTEM_CALL_NAMES:
                operands: list[ast.AST] = list(node.args)
                operands.extend(
                    kw.value for kw in node.keywords if kw.arg in _FILESYSTEM_KEYWORDS
                )
                for operand in operands:
                    for constant in _path_shaped_constants(operand):
                        context.disk_lines.add(int(getattr(constant, "lineno", 0) or 0))
            elif name == "assertNotIn" and node.args:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Constant) and isinstance(
                    first_arg.value, str
                ):
                    evidence.append(first_arg.value)
        elif isinstance(node, ast.Compare):
            left = node.left
            if (
                isinstance(left, ast.Constant)
                and isinstance(left.value, str)
                and any(isinstance(op, ast.NotIn) for op in node.ops)
            ):
                evidence.append(left.value)

    context.credential_use_lines = _credential_use_lines(tree)
    context.redaction_evidence = tuple(
        text for text in evidence if len(text) >= 4 and ("/" in text or len(text) >= 6)
    )
    context.marker_string_lines = _marker_string_lines(tree)
    context.marker_vocabulary_lines = _vocabulary_string_lines(tree)
    context.marker_detection_lines = _detection_string_lines(tree)
    context.unconditional_skip_lines = _unconditional_skip_lines(tree)
    context.quoted_skip_lines = _quoted_skip_lines(tree)
    return context



def _is_credential_name(name: str) -> bool:
    lowered = str(name or "").lower()
    return any(part in lowered for part in _CREDENTIAL_NAME_PARTS)


def _credential_use_lines(tree: ast.AST | None) -> set[int]:
    """Lines where a credential-shaped string is USED as a credential.

    A secret matters because something authenticates with it. Bound to
    ``API_KEY``, passed as ``token=``, handed to ``authenticate(...)`` — that
    is a key in use, and it is reported wherever it sits, including tests.

    A credential-shaped literal in a list of parametrize cases is not that.
    ``tests/test_security_scan_false_positives.py`` asserts that exactly such
    literals are never excused by the schema-identifier rule, so the file has
    to contain them; flagging its fixtures made the gate report the test that
    proves the gate works. That is how a secret scanner becomes a list nobody
    reads, and the day it finds a real key the finding arrives in that list.

    Judged by USE, not by file: `API_KEY = "ghp_…"` inside tests/ still fires.
    """
    lines: set[int] = set()
    if tree is None:
        return lines

    def _mark(node: ast.AST) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                lines.add(int(getattr(child, "lineno", 0) or 0))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(
                _is_credential_name(name)
                for target in node.targets
                for name in _assigned_names(target)
            ):
                _mark(node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if any(_is_credential_name(name) for name in _assigned_names(node.target)):
                _mark(node.value)
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg and _is_credential_name(keyword.arg):
                    _mark(keyword.value)
            if _is_credential_name(_call_name(node)):
                for arg in node.args:
                    _mark(arg)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and _is_credential_name(key.value)
                ):
                    _mark(value)
    return lines


def _assigned_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Attribute):
        return [target.attr]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for element in target.elts:
            names.extend(_assigned_names(element))
        return names
    return []


def _local_path_is_inert(matched: str, line_no: int, context: FileTextContext) -> bool:
    """Is this path literal data, rather than somewhere the program goes?

    Two different hazards wear one regex here, and they do not have the same
    answer:

    * ``/Users/<name>``, ``/home/<name>``, ``C:\\Users\\`` name one human's
      account. That is machine-specific wherever it appears, so it stays a
      finding unless the file proves it is redaction evidence.
    * ``/tmp/...`` is portable; what makes it a defect is writing to a
      predictable name in a world-writable directory. A literal nothing ever
      opens is not that.
    """
    if any(text in matched or matched in text for text in context.redaction_evidence):
        return True
    if matched.startswith(_TMP_PATH_PREFIX):
        # disk_lines is None when the file would not parse: unknown, so report.
        return context.disk_lines is not None and line_no not in context.disk_lines
    return False


def _marker_is_not_a_claim(line_no: int, rel: str, context: FileTextContext) -> bool:
    """Does this stub/placeholder marker say this code is unfinished?

    Three ways it does not, all judged by what the line IS rather than by
    which file it sits in — a file-name allowlist was what hid a "[DUMMY
    VOICE]" fallback and a "Mock hear" path in shipping code:

    * The marker is a NAME, a KEY or a regex PATTERN. A scanner has to spell
      the words it hunts for, and ``placeholder_detected`` is a field of the
      report, not a confession.
    * The marker is only in an identifier. Naming a variable after the thing
      you detect is not incompleteness. Class names that ARE the tell —
      ``Mock*``, ``Stub*``, ``Fake*`` in product code — are reported by
      tools/integration_debt.py, which matches on the name shape.
    * The file is under tests/. A stub, a fake and a double are how a unit
      gets isolated there; in product code they are unfinished work. The same
      exclusion, for the same reason, as raise_only_function. Test doubles
      that escape into product paths are caught where it can actually be
      proven: the sys.modules contamination guard in tests/conftest.py and
      the production-only sweep in tests/test_semantic_marker_audit.py.
    * The string is emitted by a branch that tested for the marker by name.
      ``if first.stubbed != second.stubbed:`` followed by "one arm ran on
      the stub" is the detector reporting, which is the honesty mechanism
      the first bullet protects, one level down (2026-09-21).
    """
    if rel.startswith("tests/"):
        return True
    if line_no not in context.marker_string_lines:
        return True
    return (
        line_no in context.marker_vocabulary_lines
        or line_no in context.marker_detection_lines
    )


def _skip_is_not_parked_debt(line: str, line_no: int, context: FileTextContext) -> bool:
    """Is this skip a precondition rather than a test nobody runs?

    Thirty-one findings, every one a conditional skip and not one xfail: no
    fork on this platform, no node installed, vm_stat absent, a symlink that
    would not create. The count grew each time the suite learned to run
    somewhere new, which is the opposite of a debt signal.

    What IS debt is a skip nothing decides — it fires every run, so the
    assertions below it never execute anywhere. That, ``pytest.mark.skip``
    (as against ``skipif``), and any xfail still report.
    """
    if line_no in context.unconditional_skip_lines:
        return False
    if line_no in context.quoted_skip_lines:
        return True
    return "pytest.skip" in line and "pytest.mark.skip" not in line

__all__ = [
    "FileTextContext",
    "_local_path_is_inert",
    "_marker_is_not_a_claim",
    "_multiline_string_lines",
    "_skip_is_not_parked_debt",
    "docstring_line_numbers",
    "file_text_context",
]
