from core.runtime.errors import record_degradation
import os
import tempfile
import subprocess
import json
import sys
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10
DEFAULT_MEM_BYTES = 200 * 1024 * 1024  # 200MB limit
DEFAULT_OUTPUT_LIMIT = 200 * 1024  # 200KB limit for std output

RUNNER_PY = r'''
import sys
import json
import traceback

try:
    import resource
except ImportError:
    resource = None

params = json.loads(sys.stdin.read())
code = params.get("code", "")
mem_bytes = params.get("mem_bytes", None)
cpu_seconds = params.get("cpu_seconds", None)

try:
    if resource:
        if mem_bytes:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        if cpu_seconds:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
except Exception as e:
    # Resource limits may fail on some platforms, continue with best effort
    pass

# Strip dangerous builtins to prevent arbitrary execution or network egress
import builtins
safe_builtins = {
    'abs': builtins.abs, 'all': builtins.all, 'any': builtins.any, 'ascii': builtins.ascii,
    'bin': builtins.bin, 'bool': builtins.bool, 'bytearray': builtins.bytearray, 
    'bytes': builtins.bytes, 'callable': builtins.callable, 'chr': builtins.chr,
    'complex': builtins.complex, 'dict': builtins.dict, 'dir': builtins.dir,
    'divmod': builtins.divmod, 'enumerate': builtins.enumerate, 'filter': builtins.filter,
    'float': builtins.float, 'format': builtins.format, 'frozenset': builtins.frozenset,
    'hash': builtins.hash,
    'hex': builtins.hex, 'id': builtins.id, 'int': builtins.int, 'isinstance': builtins.isinstance,
    'issubclass': builtins.issubclass, 'iter': builtins.iter, 'len': builtins.len,
    'list': builtins.list, 'map': builtins.map, 'max': builtins.max, 'min': builtins.min,
    'next': builtins.next, 'object': builtins.object, 'oct': builtins.oct, 'ord': builtins.ord,
    'pow': builtins.pow, 'print': builtins.print, 'range': builtins.range, 'repr': builtins.repr,
    'reversed': builtins.reversed, 'round': builtins.round, 'set': builtins.set,
    'slice': builtins.slice, 'sorted': builtins.sorted, 'str': builtins.str,
    'sum': builtins.sum, 'tuple': builtins.tuple, 'type': builtins.type, 'zip': builtins.zip,
    'None': None, 'True': True, 'False': False,
    # specifically exclude __import__, open, eval, exec, compile, globals, locals
}

try:
    globals_dict = {"__name__": "__main__", "__builtins__": safe_builtins}
    exec(code, globals_dict, {})
    # Keep prints for JSON IPC channel
    print(json.dumps({"status": "ok"}))
except SystemExit as e:
    print(json.dumps({"status": "exit", "code": int(e.code if isinstance(e.code, int) else 0)}))
except Exception as e:
    import traceback
    tb = traceback.format_exc()
    logger.error(f"Sandbox runner crashed: {e}\n{tb}")
    print(json.dumps({"status": "error", "repr": repr(e), "traceback": tb}))
'''

def _truncate(s: str, limit: int) -> str:
    if not s:
        return s
    if len(s) <= limit:
        return s
    return s[:limit] + "...<truncated>"

def run_untrusted(code: str, timeout: int = DEFAULT_TIMEOUT, mem_bytes: int = DEFAULT_MEM_BYTES) -> Dict:
    """
    Executes an untrusted block of Python code in an isolated subprocess with strict safety limits.
    
    Args:
        code: The Python script string to execute.
        timeout: Maximum execution CPU time allowed.
        mem_bytes: Maximum RAM consumption allowed.
        
    Returns:
        Dict: Structured result payload with stdout and stderr output.
    """
    with tempfile.TemporaryDirectory() as d:
        runner_path = os.path.join(d, "runner.py")
        with open(runner_path, "w", encoding="utf-8") as fh:
            fh.write(RUNNER_PY)
            
        payload = json.dumps({
            "code": code, 
            "mem_bytes": mem_bytes, 
            "cpu_seconds": timeout
        })
        
        cmd = [sys.executable, "-I", runner_path]
        proc = subprocess.Popen(
            cmd, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        try:
            stdout, stderr = proc.communicate(input=payload, timeout=timeout + 2)
        except subprocess.TimeoutExpired:
            proc.kill()
            return {"status": "timeout", "stdout": "", "stderr": "timeout expired"}
            
        out = _truncate(stdout, DEFAULT_OUTPUT_LIMIT)
        err = _truncate(stderr, DEFAULT_OUTPUT_LIMIT)
        
        return {
            "status": "finished", 
            "stdout": out, 
            "stderr": err, 
            "returncode": proc.returncode
        }
