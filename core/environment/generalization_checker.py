"""Generalization claims for architecture features."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GeneralizationClaim:
    feature_name: str
    general_primitive: str
    primary_environment: str
    secondary_environments: list[str]
    environment_specific_code_paths: list[str]
    shared_code_paths: list[str]
    required_tests: list[str]

    def validate(self) -> None:
        if not self.feature_name:
            raise ValueError("feature_name_required")
        if not self.general_primitive:
            raise ValueError("general_primitive_required")
        if not self.secondary_environments:
            raise ValueError("secondary_environment_required")
        if not self.shared_code_paths:
            raise ValueError("shared_code_paths_required")
        if not self.required_tests:
            raise ValueError("required_tests_required")


__all__ = ["GeneralizationClaim"]
