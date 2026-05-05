import compileall
import importlib
import ast
from pathlib import Path


def test_repo_compileall_clean():
    assert compileall.compile_dir("core", quiet=1)
    assert compileall.compile_dir("scripts", quiet=1)
    assert compileall.compile_dir("tests", quiet=1)


def test_core_environment_imports_no_side_effects():
    modules = [
        "core.environment.adapter",
        "core.environment.environment_kernel",
        "core.environment.command",
        "core.environment.modal",
        "core.environment.belief_graph",
        "core.environment.homeostasis",
        "core.environment.simulation",
        "core.environment.outcome_attribution",
        "core.environment.blackbox",
        "core.environment.replay",
        "core.environment.curriculum",
        "core.environment.benchmark",
        "core.governance.will_client",
        "core.executive.authority_gateway",
        "core.agency.agency_orchestrator",
    ]
    for module in modules:
        importlib.import_module(module)


def test_no_direct_effectful_import_side_effects_in_new_environment_layer():
    forbidden_names = {"spawn", "run", "post", "write_text"}
    for path in [*Path("core/environment").rglob("*.py"), *Path("core/environments").rglob("*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
                continue
            for node in ast.walk(stmt):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = ""
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                assert name not in forbidden_names, f"top-level effectful call {name} found in {path}"
