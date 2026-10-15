# A requested software renderer stays software

Browser play previously requested Metal on macOS and retried the browser's
default renderer after launch failures. That is unsuitable for a caller
that needs to leave an exclusive GPU job alone.

`PhantomBrowser(graphics_backend="software")` now requests the SwiftShader
OpenGL ES driver through ANGLE, using the switches documented by
[Chromium](https://chromium.googlesource.com/chromium/src/+/main/docs/gpu/swiftshader.md).
It does not enable the unsafe WebGL fallback switch. A software request
retains its driver on every executable attempt. A missing or hardware
renderer report closes that browser and supplies no ready page. Host
rendering keeps its existing default and fallback behavior.

The renderer report comes from browser-process CDP, before any tested page
is loaded. Measuring the report has a finite wait. The archived-game runner
accepts `--graphics software`, retains the requested backend in its campaign
identity and records the measured renderer with each returned result.
This is a graphics-driver policy, not a claim that every browser subsystem
or the operating system avoids all GPU work.

## Verification on this Mac

The unit set has 46 passes in 8.27 seconds. It checks constructor refusal,
software arguments on three host platforms, missing renderer evidence,
rejection and closure of a hardware renderer, and no default-renderer retry
after software startup fails. All those launch objects are fakes.

A separate opted-in test launched one real headless browser through
`PhantomBrowser`. Browser-process CDP reported:

`ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (LLVM 10.0.0) (0x0000C0DE)), SwiftShader driver-5.0.0)`.

Its WebGL2 context cleared a framebuffer and read back RGBA
`[32, 64, 128, 255]`. Physical keyboard events then moved a synthetic
Canvas rectangle in captured screenshots. A screenshot after release and
another after a wait placed it at the same position. The real-browser set
has 16 passes in 6.15 seconds. Raw reports and before/after images are in
this worktree's
`artifacts/flash-campaign/2026-10-15-software-renderer/browser-webgl/`.

Smoke has 165 passes and one skip in 67.17 seconds. Ruff, changed-file
compilation, layering, dependency consistency and changed-method size
passed. The new claim uses a small registration helper to keep the existing
installer below its method-size limit. Existing precise dependency rules
were preserved; the policy imports only the invariant-registration module.

The browser's resource lock was inspected before the real test: it is
process-local. The test used a separate state root, loaded no model,
requested no runtime restart and sent no request to G09. The browser was
closed after the test. No original game or holdout was opened. The result
proves this browser configuration and synthetic input-to-pixel path;
ordinary Aura gameplay and demo readiness still need independent proof.
