"""core/plugin_loader.py
Safe dynamic loading mechanism for agent skills.
Replaces insecure 'Genetic Mutation' self-modification.
"""
from core.runtime.errors import record_degradation
import importlib.util
import logging
import os
import sys
from typing import Any, Dict

logger = logging.getLogger("Aura.PluginLoader")

_DANGEROUS_DUNDER_ATTRS = {
    "__bases__",
    "__builtins__",
    "__closure__",
    "__code__",
    "__defaults__",
    "__func__",
    "__globals__",
    "__mro__",
    "__self__",
    "__subclasses__",
}

class PluginManager:
    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = plugin_dir
        # Ensure path is absolute if needed, or relative to cwd
        if not os.path.isabs(plugin_dir):
            self.plugin_dir = os.path.abspath(f"autonomy_engine/{plugin_dir}")
            
        os.makedirs(self.plugin_dir, exist_ok=True)
        self.loaded_plugins: Dict[str, Any] = {}

    def validate_plugin(self, file_path: str) -> bool:
        """Static analysis of plugin code before loading using AST.
        Rejects usage of 'eval', 'exec', and dangerous 'subprocess' patterns.
        """
        import ast
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                tree = ast.parse(content)
            
            unsafe_calls = {'eval', 'exec', '__import__', 'getattr', 'setattr', 'delattr', 'compile', 'input'}
            unsafe_imports = {'subprocess', 'ctypes', 'os.system', 'os.popen', 'shutil', 'pty'}
            
            for node in ast.walk(tree):
                # Check for unsafe function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in unsafe_calls:
                        logger.warning("Plugin %s rejected: unsafe call %s detected.", file_path, node.func.id)
                        return False
                # Check for dangerous attributes (e.g., .__subclasses__, .__globals__)
                if isinstance(node, ast.Attribute):
                    if node.attr in _DANGEROUS_DUNDER_ATTRS:
                        logger.warning("Plugin %s rejected: dunder attribute access %s detected.", file_path, node.attr)
                        return False
                # Check for dangerous imports
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in getattr(node, 'names', []):
                        if any(u in alias.name for u in unsafe_imports):
                            logger.warning("Plugin %s rejected: unsafe import %s detected.", file_path, alias.name)
                            return False
                            
            return True
        except Exception as e:
            record_degradation('plugin_loader', e)
            logger.error("Validation failed for %s: %s", file_path, e)
            return False

    def load_plugin(self, plugin_name: str) -> bool:
        """Dynamically loads a module from the plugins directory.
        """
        file_path = os.path.join(self.plugin_dir, f"{plugin_name}.py")
        if not os.path.exists(file_path):
            logger.warning("Plugin file not found: %s", file_path)
            return False
            
        if not self.validate_plugin(file_path):
            return False

        try:
            spec = importlib.util.spec_from_file_location(plugin_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[plugin_name] = module
                spec.loader.exec_module(module)
                self.loaded_plugins[plugin_name] = module
                logger.info("Plugin %s loaded successfully.", plugin_name)
                return True
        except Exception as e:
            record_degradation('plugin_loader', e)
            logger.error("Failed to load plugin %s: %s", plugin_name, e)
            return False
        return False
