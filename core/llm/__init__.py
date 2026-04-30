"""Compatibility namespace for LLM-layer services.

Aura's modern runtime keeps model orchestration under ``core.brain.llm``.
This package gives production subsystems a shorter ``core.llm`` import path
without duplicating implementations.
"""
from __future__ import annotations

from core.brain.llm.code_generator import GenerationRequest, LLMCodeGenerator, extract_python_code

__all__ = ["GenerationRequest", "LLMCodeGenerator", "extract_python_code"]
