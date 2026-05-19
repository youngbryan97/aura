"""Runtime contract tests for MLX client/worker hardening."""

import importlib
import os
import types
import sys


def test_setup_worker_env_not_called_at_import():
    """_setup_worker_env must NOT run when the module is imported.

    If it does, the parent process inherits Metal-specific env vars that
    conflict with multi-process orchestration.
    """
    # Record current env state for key worker-only vars
    sentinel_keys = ("MLX_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "MLX_FORCE_SERIAL_COMPILE", "METAL_COMPILER_TIMEOUT_MS")
    # Clear them so we can detect if import sets them
    saved = {}
    for key in sentinel_keys:
        saved[key] = os.environ.pop(key, None)

    try:
        # Force-reimport the module
        mod_name = "core.brain.llm.mlx_worker"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        # We can't fully reimport due to heavy deps, but we can verify the
        # module source doesn't call _setup_worker_env() at the top level
        import inspect
        from core.brain.llm import mlx_worker

        source = inspect.getsource(mlx_worker)
        # Find all top-level calls (not inside function bodies)
        # The fix moved _setup_worker_env() inside _mlx_worker_loop
        lines = source.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            # Top-level call (indent == 0) to _setup_worker_env
            if indent == 0 and stripped.startswith("_setup_worker_env("):
                raise AssertionError(
                    f"_setup_worker_env() called at module level (line {i}). "
                    "It must only be called inside _mlx_worker_loop()."
                )
    finally:
        for key, val in saved.items():
            if val is not None:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)


def test_bounded_max_tokens_narrowed_exceptions():
    """_bounded_max_tokens must not swallow arbitrary exceptions."""
    from core.brain.llm.mlx_client import _bounded_max_tokens

    # Normal operation
    assert _bounded_max_tokens(100, 200, 512) == 100
    assert _bounded_max_tokens(None, None, 256) == 256
    assert _bounded_max_tokens("", "", 128) == 128

    # Edge: non-numeric values should fallback, not crash
    assert _bounded_max_tokens("abc", 50, 64) == 50
    assert _bounded_max_tokens(50, object(), 64) == 50


def test_response_listener_shutdown_awareness():
    """The response listener loop condition should check shutdown state."""
    import inspect
    from core.brain.llm.mlx_client import MLXLocalClient

    source = inspect.getsource(MLXLocalClient._response_listener_loop)
    assert "_runtime_shutdown_requested()" in source, (
        "_response_listener_loop must check _runtime_shutdown_requested() "
        "instead of using 'while True'"
    )
    assert "while True" not in source, (
        "_response_listener_loop should not use 'while True' — "
        "it must be shutdown-aware"
    )


def test_worker_docstring_placement():
    """_mlx_worker_loop must have its docstring as the first thing in the body."""
    from core.brain.llm.mlx_worker import _mlx_worker_loop

    assert _mlx_worker_loop.__doc__ is not None, (
        "_mlx_worker_loop is missing its docstring"
    )
    assert "subprocess" in _mlx_worker_loop.__doc__.lower(), (
        "_mlx_worker_loop docstring should mention subprocess isolation"
    )


def test_no_duplicate_consecutive_empty_reset():
    """_consecutive_empty should only be reset once on successful generation."""
    import inspect
    from core.brain.llm.mlx_client import MLXLocalClient

    source = inspect.getsource(MLXLocalClient._generate_inner)

    # The success path is the block between "res.get('status') == 'ok'" and
    # the next except clause. Count resets in the full method — there should
    # be exactly one on the success path, not two adjacent lines.
    # A duplicate was the original bug; verify it stays gone.
    lines_with_reset = [
        line.strip()
        for line in source.splitlines()
        if "self._consecutive_empty = 0" in line
    ]
    assert len(lines_with_reset) == 1, (
        f"Expected exactly 1 reset of _consecutive_empty in _generate_inner, "
        f"found {len(lines_with_reset)}. Duplicate resets obscure intent."
    )


def test_stream_path_no_mid_generation_cache_clear():
    """The stream path must NOT clear MLX cache every N tokens.

    Mid-generation cache clearing forces Metal to reallocate GPU memory,
    creating micro-stalls that degrade token throughput. Post-generation
    cleanup is sufficient.
    """
    import inspect
    from core.brain.llm.mlx_worker import _mlx_worker_loop

    source = inspect.getsource(_mlx_worker_loop)
    # Find the stream action handler
    stream_start = source.find("elif action == \"stream\":")
    assert stream_start > 0, "Could not find stream action handler"
    stream_section = source[stream_start:]

    # The old pattern was: if token_count % 10 == 0: _clear_mlx_cache(mx)
    assert "token_count % 10" not in stream_section, (
        "Stream path still contains mid-generation cache clearing every 10 tokens. "
        "This was identified as harmful to throughput and must be removed."
    )
