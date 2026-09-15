// One occurrence of a thought is one card, however many paths carried it.
//
// LIVE, 2026-09-15: the health pulse rendered twice — as SYS and as
// Aura.Core.Orchestrator — both faces "Vitals steady — processor 24%...".
// The face goes through both rule tables; the coalescing key went through
// one, so the two raw shapes never met. The whole script is loaded under a
// stubbed window so the key is computed by the shipped function, with every
// helper it reaches, rather than by a slice of it.
import { readFileSync } from "node:fs";
import vm from "node:vm";

const src = readFileSync(process.argv[2], "utf8");
const el = () => new Proxy({
  style: new Proxy({}, { get(t, k) { return k in t ? t[k] : (() => {}); }, set(t, k, v) { t[k] = v; return true; } }),
  classList: { add() {}, remove() {}, contains() { return false; }, toggle() {} },
  dataset: {}, children: [], childNodes: [], isConnected: true, textContent: "", innerHTML: "",
  addEventListener() {}, removeEventListener() {}, appendChild() {}, prepend() {}, append() {},
  insertBefore() {}, remove() {}, replaceChildren() {}, cloneNode() { return el(); },
  querySelector() { return null; }, querySelectorAll() { return []; }, closest() { return null; },
  matches() { return false; }, contains() { return false; }, setAttribute() {}, getAttribute() { return null; },
  removeAttribute() {}, focus() {}, scrollIntoView() {}, getBoundingClientRect() { return { width: 0, height: 0 }; },
  getContext() { return new Proxy({}, { get() { return () => ({ data: [] }); } }); },
}, { get(t, k) { return k in t ? t[k] : undefined; }, set(t, k, v) { t[k] = v; return true; } });
const document = {
  addEventListener() {}, removeEventListener() {}, dispatchEvent() { return true; },
  getElementById() { return el(); }, querySelector() { return el(); }, querySelectorAll() { return []; },
  createElement() { return el(); }, body: el(), documentElement: el(), hidden: false,
  visibilityState: "visible", hasFocus() { return false; }, readyState: "complete", title: "",
};
const window = {
  document, console, Date, Math, JSON, Number, String, Array, Object, RegExp, Map, Set, Promise,
  Error, TypeError, URL, URLSearchParams, TextEncoder, TextDecoder, Intl, AbortController,
  CustomEvent: class { constructor(t, o) { this.type = t; this.detail = o && o.detail; } },
  Event: class { constructor(t) { this.type = t; } },
  addEventListener() {}, removeEventListener() {}, dispatchEvent() { return true; },
  location: { href: "http://localhost:8000/", protocol: "http:", host: "localhost:8000", pathname: "/", search: "" },
  localStorage: { getItem() { return null; }, setItem() {}, removeItem() {} },
  sessionStorage: { getItem() { return null; }, setItem() {}, removeItem() {} },
  matchMedia() { return { matches: false, addEventListener() {}, addListener() {} }; },
  requestAnimationFrame() { return 0; }, setTimeout() { return 0; }, clearTimeout() {},
  setInterval() { return 0; }, clearInterval() {},
  navigator: { onLine: true, userAgent: "node", language: "en" },
  performance: { now: () => Date.now() }, WebSocket: function () {}, fetch: () => new Promise(() => {}),
  crypto: { randomUUID() { return "x"; } }, innerWidth: 1440, innerHeight: 900, devicePixelRatio: 1,
};
window.window = window; window.self = window; window.globalThis = window;
const ctx = vm.createContext(window);
const quiet = console.log; console.log = () => {};
try { vm.runInContext(src, ctx, { filename: "aura.js" }); } catch (e) { /* the definitions we need precede the boot code */ }
console.log = quiet;
const fingerprint = vm.runInContext("buildThoughtFingerprint", ctx);

const pulse = [
  { name: "SYS", level: "info", timestamp: 1789446661,
    message: "═══ UNIFIED HEALTH PULSE ═══\nSystem: CPU 15.8% | RAM 55.3% | Uptime: 482s\nRuntime: HEALTHY | Required probes: PASS" },
  { name: "Aura.Core.Orchestrator", level: "info", timestamp: 1789446661.2,
    message: "🫀 ═══ UNIFIED HEALTH PULSE ═══ | System: CPU 15.8% | RAM 55.3% | Uptime: 482s | Runtime: HEALTHY | Required probes: PASS" },
];
const keys = pulse.map(fingerprint);
let bad = 0;
if (keys[0] !== keys[1]) { bad++; console.log(`FAIL the two pulse shapes key differently:\n  ${keys[0]}\n  ${keys[1]}`); }
else console.log(`OK   one key for both pulse shapes: ${keys[0].slice(0, 70)}`);
// Readings are normalised out of the key on purpose: two pulses twelve
// seconds apart with different percentages are one card with a ×2 badge.
const fault = fingerprint({ ...pulse[0], level: "error" });
if (fault === keys[0]) { bad++; console.log("FAIL a fault keyed the same as the routine note"); }
else console.log("OK   a fault with the same words is a different card");
console.log(bad === 0 ? "\nall checks passed" : `\n${bad} FAILED`);
process.exit(bad === 0 ? 0 : 1);
