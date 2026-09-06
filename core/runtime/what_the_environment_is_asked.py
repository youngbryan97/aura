"""Four hundred and three settings, read from the environment by hand.

`core/config.py` holds a typed settings model with defaults, validation and a
declared shape. Four hundred and three environment variables are read straight
out of `os.environ` in 518 places that never go near it, so precedence,
validation, a default that is written down, and knowing the setting exists at
all apply to none of them.

The cost is not tidiness. A variable read in twelve files is twelve chances to
spell it differently, twelve defaults that can disagree, and no way to answer
"what can be configured" except by grepping. AURA_STRICT_RUNTIME is read in
twelve files. Two of them can already disagree about what its absence means.

What this does is smaller than a migration and answers the question a
migration would: every direct read in the tree, what it is called, where it is
read, whether more than one place disagrees about its default, and which of
them the typed model already knows about.

The ratchet is the count of settings read in more than one place with more
than one default. That is the only shape here that is unambiguously a defect
rather than a style.
"""
from __future__ import annotations

import ast
import functools
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.WhatTheEnvironmentIsAsked")

__all__ = [
    "ASetting",
    "every_setting_read",
    "settings_whose_defaults_disagree",
    "settings_the_model_does_not_know",
    "how_the_settings_stand",
]

ROOTS = ("core", "interface", "skills", "llm", "executors", "security")


@dataclass(frozen=True, slots=True)
class ASetting:
    """One environment variable, and every place that reads it."""

    name: str
    read_in: tuple[str, ...] = ()
    #: Every distinct default given at a read site, as source text. More than
    #: one means two places disagree about what its absence means.
    defaults: tuple[str, ...] = ()
    reads: int = 0

    @property
    def defaults_disagree(self) -> bool:
        return len(self.defaults) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reads": self.reads,
            "read_in": list(self.read_in),
            "defaults": list(self.defaults),
            "defaults_disagree": self.defaults_disagree,
        }


def _asks_the_environment(node: ast.AST) -> ast.expr | None:
    """The name argument of an os.getenv / os.environ.get call, or None."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    func = node.func
    if func.attr == "getenv" and node.args:
        return node.args[0]
    if func.attr != "get" or not node.args:
        return None
    holder = func.value
    if isinstance(holder, ast.Attribute) and holder.attr == "environ":
        return node.args[0]
    if isinstance(holder, ast.Name) and holder.id == "environ":
        return node.args[0]
    return None


def _default_of(node: ast.Call) -> str:
    """The default as written, or "" where none was given."""
    if len(node.args) > 1:
        return ast.unparse(node.args[1])
    for kw in node.keywords:
        if kw.arg == "default":
            return ast.unparse(kw.value)
    return ""


@functools.lru_cache(maxsize=4)
def every_setting_read(repo: str = ".") -> dict[str, ASetting]:
    """Every environment variable read directly, and where."""
    root = pathlib.Path(repo)
    seen: dict[str, dict[str, Any]] = {}
    for name in ROOTS:
        base = root / name
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except (SyntaxError, ValueError, OSError):
                continue
            for node in ast.walk(tree):
                asked = _asks_the_environment(node)
                if not isinstance(asked, ast.Constant) or not isinstance(
                    asked.value, str
                ):
                    continue
                row = seen.setdefault(
                    asked.value,
                    {"files": set(), "defaults": set(), "reads": 0},
                )
                row["files"].add(str(path.relative_to(root)))
                row["reads"] += 1
                given = _default_of(node)  # type: ignore[arg-type]
                if given:
                    row["defaults"].add(given)
    return {
        name: ASetting(
            name=name,
            read_in=tuple(sorted(row["files"])),
            defaults=tuple(sorted(row["defaults"])),
            reads=int(row["reads"]),
        )
        for name, row in sorted(seen.items())
    }


def settings_whose_defaults_disagree(repo: str = ".") -> tuple[str, ...]:
    """Read in more than one place, with more than one default.

    The ratchet. Two places disagreeing about what a setting's absence means
    is the one shape here that is unambiguously a defect: the behaviour then
    depends on which module happened to ask first.
    """
    return tuple(
        sorted(
            name
            for name, one in every_setting_read(repo).items()
            if one.defaults_disagree and len(one.read_in) > 1
        )
    )


def settings_the_model_does_not_know(repo: str = ".") -> tuple[str, ...]:
    """Read directly and absent from the typed settings model."""
    try:
        from core.config import AuraConfig
    except (ImportError, AttributeError):
        return ()
    known: set[str] = set()
    for field_name, spec in getattr(AuraConfig, "model_fields", {}).items():
        known.add(field_name.upper())
        known.add(f"AURA_{field_name.upper()}")
        alias = getattr(spec, "alias", None) or getattr(
            getattr(spec, "validation_alias", None), "__str__", lambda: ""
        )()
        if isinstance(alias, str) and alias:
            known.add(alias.upper())
    return tuple(
        sorted(one for one in every_setting_read(repo) if one.upper() not in known)
    )


def how_the_settings_stand(repo: str = ".") -> dict[str, Any]:
    """For the health report: what is configurable, and where it disagrees."""
    every = every_setting_read(repo)
    disagreeing = settings_whose_defaults_disagree(repo)
    return {
        "settings": len(every),
        "reads": sum(one.reads for one in every.values()),
        "read_in_more_than_one_file": sorted(
            name for name, one in every.items() if len(one.read_in) > 1
        ),
        "defaults_disagree": list(disagreeing),
        "the_disagreements": {
            name: {
                "defaults": list(every[name].defaults),
                "read_in": list(every[name].read_in),
            }
            for name in disagreeing
        },
        "the_model_does_not_know": len(settings_the_model_does_not_know(repo)),
    }
