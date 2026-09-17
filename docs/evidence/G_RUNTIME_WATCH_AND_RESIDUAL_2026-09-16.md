# Runtime recovery and residual sampling

The live desktop log recorded process 92225 entering orchestrator shutdown
through `SelfHealing._restart_watch` at 07:30:26 on September 16. That method
called `stop()` and then `start()` on the process root. Shutdown had already
latched the process-wide fence, so the attempted restart could not register
actors and the API listener disappeared. The watch was installed during boot,
before the root heartbeat loop started.

Watches now declare a recovery scope. The orchestrator watch waits for its
first heartbeat and has process-root scope. A stale process-root watch records
that external recovery is required, preserves its actual heartbeat age, and
cannot invoke component restart or deep repair. Process replacement remains
owned by the external lifecycle machinery. Component watches retain their
existing restart path.

The same live launch exposed a separate residual-sampling failure: NumPy could
not read the MLX bfloat16 PEP 3118 buffer. The sampler now uses the residual's
public host conversion when direct NumPy materialization is unsupported.
The conversion operates on the selected residual vector, after the existing
prefill exclusion.

Focused lifecycle/runtime and residual tests passed 152 cases. Those include
real MLX float16, bfloat16 and float32 arrays, first-heartbeat activation,
retained stale age, and a positive component-restart control. Smoke passed
164 cases with one skip. Live deployment validation is still required.

These repairs do not establish broad reasoning gain, steering qualification,
fusion, or frontier performance. The G-ledger scientific items remain open.
