# Conversation latency checkpoint — 22 October 2026

The implementation is saved at `f45c6e2cf`. It has not been activated or
measured in a live conversation. Bryan reserved the machine for the existing
27B reports campaign, so Aura remains closed. The selected faster A voice,
its register and acoustic processing are unchanged.

## Changes

- The ordinary desktop reply can publish checked public sentences into the
  voice turn's existing channel. The parent captures that channel in the
  caller's context; foreign, cancelled and finished requests cannot use it.
  Private reasoning, corrective passes, tools and structured output remain
  buffered. The existing final-reply reconciliation still owns revisions.
- A short opening sentence can leave the speech buffer before its character
  budget fills. Following whitespace confirms the boundary; abbreviations,
  decimal prefixes and list enumerators cannot trigger it alone. Subsequent
  chunks keep the existing larger budget for prosody.
- Unchanged history passages reuse document vectors. The cache retains at
  most 256 vectors of at most 4,096 elements, uses exact-text hashes as keys,
  and expires with the encoder. Failed encodes are retried. A batch keeps its
  encoder through fallback and cannot populate a replacement model's cache.
- Ordinary memory, identity and self-condition turns can carry the existing
  conversation-resume capability. Its worker-side model, authority, session
  and template checks still apply; current evidence is appended anew.
- Streamed and typed voice turns record cognition completion independently
  of audio playback. A superseding turn cannot receive the old measurement.

## Evidence and limits

`voice-streaming-2026-10-22T072932Z.json` binds the source to 158 focused
passing checks. Files ran sequentially, with one BLAS thread and lowered
process priority. The onset test injects worker frames and a speech callback;
it proves that the first checked sentence reaches that callback before the
producer finishes. It does not measure a model, microphone or speaker.

The earlier broad run had 574 passes and one stale token-floor assertion.
The unchanged quick-reply methods at `090672eef` reproduced it: allocation
was 512 tokens and the semantic completion floor was 256. The assertion now
checks those separate quantities; all 13 affected quick-reply cases pass.
No production token budget changed for that test repair.

Changed-file syntax, critical Ruff checks and whitespace checks pass. The
earlier smoke run passed 164 tests and skipped one before the final buffer
and cache-race changes. Final-source smoke, aggregate gates and native
activation remain deferred while the campaign owns the machine.

The previous live recorded-input probe took 25,828 ms from input end to first
reply PCM. It measured about 689 ms for final ASR, 22,340 ms for cognition
and 1,588 ms for the first TTS chunk. Two later typed probes took 43,806 ms
and 33,362 ms under different host load. These are diagnostic baselines,
not interchangeable trials or evidence of a new speedup.

## Next live verification

1. Confirm the reports campaign and its model workers have released the
   machine. Check process ownership and the campaign's completion evidence;
   the fusion subprocess ending alone does not mean the reports run ended.
   Do not start Aura while that campaign is active.
2. Run the deferred gates with the shared virtualenv and a separate
   `AURA_LOG_DIR`. Recheck source and process state before activation.
3. Fast-forward the desktop checkout to the tested checkpoint. Use the
   supported signed native build and lifecycle path. Verify one runtime,
   the expected revision, conversation readiness and the selected voice.
4. Measure cold and repeated warm turns through the production voice socket.
   Keep first response PCM, final ASR, cognition and first synthesis timing
   separate. Record current host load and conversation-resume receipts.
   A voice preview measures only synthesis and is insufficient for this step.
5. Repeat with recorded speech input and real OS playback. State explicitly
   that injected input does not test physical microphone acoustics. Test an
   interruption during playback and reject late audio before the next turn.
6. Verify the spoken prefix against the finished governed reply. Record any
   correction, delivery failure or unexpected silence. Continue profiling
   context preparation and prefill if conversational onset remains slow.

Phone-like responsiveness remains an open objective until those measurements
and ordinary conversation demonstrate it.
