import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const root = process.argv[2];
const context = { window: {}, document: {} };
vm.runInNewContext(readFileSync(`${root}/interface/static/conversation_visuals.js`, 'utf8'), context);
const { numberOf, flowOf } = context.window.AuraConversationVisuals;
assert.equal(numberOf('1,024 ms').value, 1024);
assert.equal(numberOf('−2.5%').value, -2.5);
for (const missing of ['N/A', '—', '~12', '3–5', '54/60', '', 'NaN', 'Infinity', '1,23']) {
    assert.equal(numberOf(missing), null, missing);
}
const flow = flowOf('flowchart LR\nA[Observe] --> B[Act] --> C[Check]');
assert.equal(flow.nodes.size, 3);
assert.equal(flow.edges.length, 2);
assert.equal(flow.levels.get('C'), 2);
assert.equal(flowOf('graph TD\nA{Ready?} -->|yes| B[Continue]').edges[0].label, 'yes');
assert.equal(flowOf('graph TD\nA --> B\nB --> A'), null);
assert.equal(flowOf('graph TD\nA --> B\nclick A "javascript:alert(1)"'), null);
assert.equal(flowOf('graph TD\nA --> B\nstyle A fill:red'), null);
assert.equal(flowOf('graph TD\nA --> B\nsubgraph example'), null);
assert.equal(flowOf('graph TD\nA --> B\nthis is not a graph statement'), null);
console.log('visual data and diagram contracts passed');

// Repeated browser events have no conversation UI to create or announce.
let removed = 0;
let cancelled = 0;
let stale = { expiry: 42, remove() { removed++; stale = null; } };
context.document.getElementById = () => stale;
context.clearTimeout = id => { assert.equal(id, 42); cancelled++; };
for (let i = 0; i < 30; i++) {
    for (const action of ['navigating', 'reading', 'clicking', 'typing', 'scrolling', 'closed']) {
        context.window.AuraConversationVisuals.activity({ action, stage: 'started', visible: false });
    }
}
assert.equal(removed, 1);
assert.equal(cancelled, 1);
console.log('browser events stay out of the conversation');
