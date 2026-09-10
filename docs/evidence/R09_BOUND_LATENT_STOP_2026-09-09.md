# R09: Bound latent Stop and live prefill interruption

## Live observations

Runtime PID 29866 loaded revision
`354f6c4443f91ba771d570edb83076bbda9d447e` with the resident
`Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15` model.

At 02:10 on 2026-09-09, desktop turn
`aura-chat-6a511563-086a-4752-9296-23bb6940f609` began ordinary generation.
The desktop Stop control cancelled job sequence 30 during prefill. The worker
logged interruption at 1088 of 2924 tokens at 02:10:39.573. The desktop
delivered one cancellation acknowledgement at 02:10:40. The 02:10:47 health
observation was healthy, with no intervening model reload.

The earlier latent turn `aura-chat-9a23de31-cc39-44cb-b225-2edc1d51db03`
also accepted Stop, but its caller unconditionally scheduled a worker reboot.
The resulting reload made the conversation lane unavailable. This is a
different cancellation owner from ordinary generation.

## Repair and checks

Caller cancellation now uses the existing latent cleanup-receipt validator,
also used by deadline cancellation. Request ownership remains held while the
worker acknowledges Stop. Only a receipt bound to the request payload and
worker identity, with valid runtime cleanup evidence, preserves the worker.
Missing or invalid acknowledgement retains the existing reboot path.

The focused latent wiring, client ownership, client resilience, and installed
MLX prefill suites passed 278 tests in 161.92 seconds. New cases cover a clean
acknowledgement and wrong request, payload, worker, or cleanup bindings.

The ordinary prefill change is live-proven. The latent caller change still
requires deployment and a resident replay. R09 remains unchecked; follow-up
context retention also has a separate live failure.
