"""core/self_improvement/spec_extractor.py — Deterministic module spec extraction.

Extracts a full ModuleSpec from a live Aura module using only AST inspection
and filesystem scanning. No LLM calls — this is the "methods extraction" step
from the paper, adapted for code-to-code reproduction.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import List, Optional, Set

from core.self_improvement.interface_contract import (
    ClassSignature, FunctionSignature, InterfaceContract,
    ModuleSpec, TestCase,
)

logger = logging.getLogger("Aura.SpecExtractor")


class SpecExtractor:
    """Extracts ModuleSpec from a live module using AST analysis. Purely deterministic."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or ".").resolve()

    def extract(self, module_path: str) -> ModuleSpec:
        abs_path = self.project_root / module_path
        if not abs_path.exists():
            raise FileNotFoundError(f"Module not found: {abs_path}")
        source = abs_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(abs_path))
        module_docstring = ast.get_docstring(tree) or ""
        all_names = self._extract_all_names(tree)
        functions = self._extract_functions(tree, all_names)
        classes = self._extract_classes(tree, all_names)
        constants = self._extract_constants(tree, all_names)
        imports = self._extract_imports(tree)
        dependencies = self._extract_dependencies(tree)
        test_cases = self._find_test_cases(module_path)
        interface = InterfaceContract(
            module_path=module_path, functions=functions, classes=classes,
            constants=constants, all_names=frozenset(all_names) if all_names else frozenset(),
            imports=imports,
        )
        return ModuleSpec(
            module_path=module_path, module_name=Path(module_path).stem,
            interface=interface, module_docstring=module_docstring,
            invariants=[], test_cases=test_cases, trace_examples=[],
            dependencies=dependencies,
        )

    def _extract_all_names(self, tree: ast.Module) -> Set[str]:
        names: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    names.add(elt.value)
        return names

    def _extract_functions(self, tree: ast.Module, all_names: Set[str]) -> List[FunctionSignature]:
        functions: List[FunctionSignature] = []
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_") and node.name not in all_names:
                continue
            if all_names and node.name not in all_names:
                continue
            params = self._extract_params(node)
            decorators = self._extract_decorators(node)
            ret = None
            if node.returns:
                try:
                    ret = ast.unparse(node.returns)
                except Exception:
                    pass
            functions.append(FunctionSignature(
                name=node.name, parameters=tuple(params),
                is_async=isinstance(node, ast.AsyncFunctionDef),
                is_classmethod="classmethod" in decorators,
                is_staticmethod="staticmethod" in decorators,
                is_property="property" in decorators,
                return_annotation=ret, docstring=ast.get_docstring(node),
                decorators=tuple(decorators),
            ))
        return functions

    def _extract_classes(self, tree: ast.Module, all_names: Set[str]) -> List[ClassSignature]:
        classes: List[ClassSignature] = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name.startswith("_") and node.name not in all_names:
                continue
            if all_names and node.name not in all_names:
                continue
            bases = []
            for base in node.bases:
                try:
                    bases.append(ast.unparse(base))
                except Exception:
                    pass
            methods: List[FunctionSignature] = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith("_") and item.name != "__init__":
                        continue
                    params = self._extract_params(item)
                    decs = self._extract_decorators(item)
                    ret = None
                    if item.returns:
                        try:
                            ret = ast.unparse(item.returns)
                        except Exception:
                            pass
                    methods.append(FunctionSignature(
                        name=item.name, parameters=tuple(params),
                        is_async=isinstance(item, ast.AsyncFunctionDef),
                        is_classmethod="classmethod" in decs,
                        is_staticmethod="staticmethod" in decs,
                        is_property="property" in decs,
                        return_annotation=ret, docstring=ast.get_docstring(item),
                        decorators=tuple(decs),
                    ))
            class_decs = self._extract_decorators(node)
            classes.append(ClassSignature(
                name=node.name, bases=tuple(bases), methods=tuple(methods),
                docstring=ast.get_docstring(node), decorators=tuple(class_decs),
            ))
        return classes

    def _extract_constants(self, tree: ast.Module, all_names: Set[str]) -> dict[str, str]:
        constants: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        if name.startswith("_") or name == "__all__":
                            continue
                        if all_names and name not in all_names:
                            continue
                        try:
                            type_str = type(ast.literal_eval(node.value)).__name__
                        except Exception:
                            try:
                                type_str = ast.unparse(node.value)
                            except Exception:
                                type_str = "unknown"
                        constants[name] = type_str
        return constants

    def _extract_imports(self, tree: ast.Module) -> List[str]:
        imports: List[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = ", ".join(a.name for a in node.names)
                imports.append(f"from {module} import {names}")
        return imports

    def _extract_dependencies(self, tree: ast.Module) -> List[str]:
        deps: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    deps.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    deps.add(node.module.split(".")[0])
        return sorted(deps)

    def _extract_params(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[str]:
        params: List[str] = []
        for arg in node.args.args:
            name = arg.arg
            if arg.annotation:
                try:
                    name += f": {ast.unparse(arg.annotation)}"
                except Exception:
                    pass
            params.append(name)
        for arg in node.args.kwonlyargs:
            name = arg.arg
            if arg.annotation:
                try:
                    name += f": {ast.unparse(arg.annotation)}"
                except Exception:
                    pass
            params.append(name)
        return params

    def _extract_decorators(self, node) -> List[str]:
        decorators: List[str] = []
        for dec in node.decorator_list:
            try:
                decorators.append(ast.unparse(dec))
            except Exception:
                if isinstance(dec, ast.Name):
                    decorators.append(dec.id)
        return decorators

    def _find_test_cases(self, module_path: str) -> List[TestCase]:
        test_cases: List[TestCase] = []
        module_name = Path(module_path).stem
        tests_dir = self.project_root / "tests"
        if not tests_dir.exists():
            return test_cases
        patterns = [f"test_{module_name}.py"]
        for test_file in tests_dir.rglob("*.py"):
            if not any(test_file.name == p for p in patterns):
                continue
            try:
                test_source = test_file.read_text(encoding="utf-8")
                test_tree = ast.parse(test_source)
            except Exception:
                continue
            for node in ast.walk(test_tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("test_"):
                        try:
                            test_src = ast.get_source_segment(test_source, node) or ""
                        except Exception:
                            test_src = ""
                        test_cases.append(TestCase(
                            name=node.name, source=test_src,
                            file_path=str(test_file.relative_to(self.project_root)),
                        ))
        return test_cases


__all__ = ["SpecExtractor"]
