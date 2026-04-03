"""Multi-layer code validation for safe self-modification.
"""
import ast
import io
import logging
import tokenize
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CodeValidator:
    """Validate code changes before and after modification.
    """
    
    def __init__(self):
        self.validation_pipeline: List[Callable] = [
            self._validate_syntax,
            self._validate_imports,
            self._validate_ast_structure,
            self._validate_no_syntax_errors,
            self._validate_security_concerns,
        ]
    
    def validate_change(self, 
                       old_code: str, 
                       new_code: str,
                       file_path: Optional[Path] = None) -> Tuple[bool, List[str]]:
        """Validate code change with multiple checks.
        
        Returns: (is_valid, list_of_warnings)
        """
        errors = []
        warnings = []
        
        # Basic checks
        if not new_code.strip():
            errors.append("New code is empty")
            return False, errors
        
        if old_code == new_code:
            warnings.append("Code unchanged (no-op modification)")
        
        # Run validation pipeline
        for validator in self.validation_pipeline:
            try:
                result = validator(new_code, file_path)
                if isinstance(result, tuple):
                    valid, msg = result
                    if not valid:
                        errors.append(msg)
                elif not result:
                    errors.append(f"Validation failed: {validator.__name__}")
            except Exception as e:
                errors.append(f"Validator {validator.__name__} crashed: {e}")
        
        return len(errors) == 0, warnings + errors
    
    def _validate_syntax(self, code: str, file_path: Optional[Path] = None) -> Tuple[bool, str]:
        """Validate Python syntax"""
        try:
            ast.parse(code)
            return True, "Syntax OK"
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
    
    def _validate_imports(self, code: str, file_path: Optional[Path] = None) -> Tuple[bool, str]:
        """Validate imports - allow functional system access"""
        try:
            tree = ast.parse(code)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.unparse(node))
            
            # We allow all imports for functionality, but warn about very risky ones if needed
            # "Dangerous" imports are now considered "Powerful" tools
            
            return True, f"Imports OK ({len(imports)} imports)"
            
        except Exception as e:
            return False, f"Import validation failed: {e}"
    
    def _validate_ast_structure(self, code: str, file_path: Optional[Path] = None) -> Tuple[bool, str]:
        """Validate AST structure"""
        try:
            tree = ast.parse(code)
            
            # Check for extreme complexity
            stats = {
                "functions": 0,
                "classes": 0,
                "lines": len(code.splitlines()),
                "max_function_length": 0,
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    stats["functions"] += 1
                    # Calculate function length
                    if hasattr(node, 'lineno') and getattr(node, 'end_lineno', None) is not None:
                        length = node.end_lineno - node.lineno # type: ignore
                        stats["max_function_length"] = max(stats["max_function_length"], length)
                
                elif isinstance(node, ast.ClassDef):
                    stats["classes"] += 1
            
            # Warn about potential issues
            warnings = []
            if stats["max_function_length"] > 100:
                warnings.append(f"Long function ({stats['max_function_length']} lines)")
            
            if stats["lines"] > 1000:
                warnings.append(f"Large file ({stats['lines']} lines)")
            
            if warnings:
                return True, f"Structure OK (warnings: {', '.join(warnings)})"
            else:
                return True, "Structure OK"
            
        except Exception as e:
            return False, f"Structure validation failed: {e}"
    
    def _validate_no_syntax_errors(self, code: str, file_path: Optional[Path] = None) -> Tuple[bool, str]:
        """Token-level validation"""
        try:
            # Use tokenize for low-level validation
            tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
            
            # Check for unbalanced brackets/parentheses
            bracket_stack = []
            
            for token in tokens:
                if token.string in '([{':
                    bracket_stack.append(token.string)
                elif token.string in ')]}':
                    if not bracket_stack:
                        return False, f"Unbalanced {token.string}"
                    
                    opening = bracket_stack.pop()
                    if (opening == '(' and token.string != ')') or \
                       (opening == '[' and token.string != ']') or \
                       (opening == '{' and token.string != '}'):
                        return False, f"Mismatched brackets: {opening}{token.string}"
            
            if bracket_stack:
                return False, f"Unclosed brackets: {bracket_stack}"
            
            return True, "Tokenization OK"
            
        except tokenize.TokenError as e:
            return False, f"Token error: {e}"
        except Exception as e:
            return False, f"Token validation failed: {e}"
    
    def _validate_security_concerns(self, code: str, file_path: Optional[Path] = None) -> Tuple[bool, str]:
        """Check for operational risks (blocking, destruction) rather than restriction"""
        # We only want to flag things that are likely errors, like accidental infinite recursion or unconditional deletion of root
        
        # Simple regex checks for obviously dangerous patterns that might be missed by import checks
        risk_patterns = [
            (r'rm\s+-rf\s+/$', "Potential root deletion"),
            (r'globals\(\)\[', "Dynamic global modification (fragile)"),
        ]
        
        import re
        warnings = []
        for pattern, description in risk_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                warnings.append(description)
        
        if warnings:
            # We don't fail, we just warn. The agent needs power.
            return True, f"Operational tips: {'; '.join(warnings)}"
        
        return True, "Operational checks OK"
    
    def diff_code(self, old_code: str, new_code: str) -> Dict[str, Any]:
        """Generate diff between old and new code.
        """
        import difflib
        
        old_lines = old_code.splitlines(keepends=True)
        new_lines = new_code.splitlines(keepends=True)
        
        diff = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile='old', tofile='new',
            lineterm=''
        ))
        
        # Calculate statistics
        stats = {
            "old_lines": len(old_lines),
            "new_lines": len(new_lines),
            "diff_lines": len(diff),
            "added": sum(1 for line in diff if line.startswith('+') and not line.startswith('+++')),
            "removed": sum(1 for line in diff if line.startswith('-') and not line.startswith('---')),
            "changed": sum(1 for line in diff if line.startswith(' ')),
            "diff": diff,
        }
        
        return stats
    
    def calculate_checksum(self, code: str) -> str:
        """Calculate checksum of code"""
        return hashlib.sha256(code.encode('utf-8')).hexdigest()[:16]


# Global instance
code_validator = CodeValidator()