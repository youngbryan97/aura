import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';

const source = readFileSync(process.argv[2], 'utf8');
const start = source.indexOf('function scheduleBootstrapPoll(');
const end = source.indexOf('// ── Tab switching', start);
assert(start >= 0 && end > start);
let now = 0;
let delay = 30000;
let nextId = 1;
let refreshes = 0;
const timers = new Map();
const state = {};
const document = { hidden: false };
const schedule = new Function('state', 'performance', 'optionalSurfacePollDelay',
    'BOOTSTRAP_POLL_MS', 'setTimeout', 'clearTimeout', 'document', 'hydrateBootstrap',
    source.slice(start, end) + '\nreturn scheduleBootstrapPoll;'
)(state, { now: () => now }, () => delay, 30000,
    (callback, ms) => {
        const id = nextId++;
        timers.set(id, { callback, due: now + ms });
        return id;
    }, id => timers.delete(id), document, async () => { refreshes++; });

schedule();
const original = state.bootstrapTimer;
for (now = 1000; now < 30000; now += 1000) {
    delay = now % 2000 === 0 ? 30000 : 90000;
    schedule();
    assert.equal(state.bootstrapTimer, original);
    assert.equal(timers.size, 1);
}
assert.equal(timers.get(original).due, 30000);
const first = timers.get(original);
timers.delete(original);
await first.callback();
assert.equal(refreshes, 1);
assert.equal(timers.size, 1);

// Explicit immediate refresh can accelerate, but never duplicate the timer.
schedule(0);
assert.equal(timers.size, 1);
assert.equal(timers.get(state.bootstrapTimer).due, now);
const immediate = timers.get(state.bootstrapTimer);
timers.delete(state.bootstrapTimer);
await immediate.callback();
assert.equal(refreshes, 2);
assert.equal(timers.size, 1);
console.log('bootstrap poll fairness checks passed');
