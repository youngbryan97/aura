import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(process.argv[2], 'utf8');
const start = source.indexOf('function applyToolEvent(event) {');
const end = source.indexOf('\nfunction applyBootstrapPayload', start);
assert.ok(start >= 0 && end > start);
const skills = [];
const thoughts = [];
const context = vm.createContext({
    state: {},
    toolEventIsDeferred: event => event.stage === 'deferred',
    escText: value => value || 'idle',
    $: () => null,
    updateSkillUI: (...args) => skills.push(args),
    describeToolEvent: event => event.tool,
    queueThought: event => thoughts.push(event),
    updateTypingLabel: () => assert.fail('Global tool event overwrote turn progress'),
    setChatPanelState: () => assert.fail('Global tool event changed chat state'),
    formatToolAction: tool => tool,
});
vm.runInContext(source.slice(start, end), context);
for (const stage of ['started', 'completed', 'failed', 'rejected', 'degraded', 'deferred']) {
    context.applyToolEvent({tool: 'background_task', stage});
}
assert.equal(skills.length, 6);
assert.equal(thoughts.length, 3);
console.log('Global tool activity preserves chat ownership and tool telemetry.');
