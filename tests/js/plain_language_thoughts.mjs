// Plain English on the card face; raw payload preserved for FULL/COPY.
import { readFileSync } from "node:fs";
const src = readFileSync(process.argv[2], "utf8");
const a = src.indexOf("const PLAIN_LANGUAGE_RULES");
const b = src.indexOf("const NEURAL_CHANNELS");
const mod = await import("data:text/javascript;base64," +
  Buffer.from(src.slice(a, b) + "\nexport { plainLanguageThought };").toString("base64"));

const CASES = [
  ["UNIFIED HEALTH PULSE | System: CPU 0.0% | RAM 71.6% | Uptime: 620s | Runtime: HEALTHY", /Vitals steady .* memory 72%, awake 10 minutes/],
  ["Router: Queueing background inference until admission clears for origin=stream_narrative reason=foreground_headroom_reserved", /conversation keeps priority/],
  ["Phase 'UnitaryResponsePhase' timed out after 12s — skipping", /unitary response.*skipped/i],
  ["Tool Deferred: auto_refactor in 0ms (memory_pressure_71.1)", /scan of her own code.*memory is tight/],
  ["stem cell captured: organ=self_object schema=1 bytes=26", /recovery snapshot of self object/],
  ["Flagged response for distillation (confidence=0.40, queue=55)", /only 40% sure/],
  ["PhiCore exclusion postulate: max-phi complex = full 16-node system (phi=0.61610)", /how unified her mind is.*0\.62/],
  ["Sweep complete: 0 procs reaped, 0.0MB storage reclaimed.", /nothing to clean up/],
];
const UNTOUCHED = [
  "[health_poll] health=not ready; probes blocked; conversation ready; proof integrity degraded",
  "[websocket_heartbeat] health=ready; probes pass; conversation ready; proof integrity degraded: source drift",
  "[health_poll] health=ready; probes pass; conversation ready",
  "I don't think you're right, and I'll tell you why.",
  "It's 1:24 AM, and I know that from my clock.",
];

let bad = 0;
for (const [input, expect] of CASES) {
  const out = mod.plainLanguageThought(input);
  const ok = expect.test(out);
  if (!ok) bad++;
  console.log(`${ok ? "OK  " : "FAIL"} ${out.slice(0, 78)}`);
}
for (const text of UNTOUCHED) {
  const ok = mod.plainLanguageThought(text) === text;
  if (!ok) bad++;
  console.log(`${ok ? "OK  " : "FAIL"} untouched: ${text.slice(0, 50)}`);
}
console.log(bad === 0 ? "\nall checks passed" : `\n${bad} FAILED`);
process.exit(bad === 0 ? 0 : 1);
