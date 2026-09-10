import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';

const source = readFileSync(process.argv[2], 'utf8');
globalThis.window = {setTimeout, clearTimeout};
globalThis.CHAT_DELIVERY_STATUS_TIMEOUT_MS = 500;
const start = source.indexOf('function updateChatStopControl()');
const end = source.indexOf('\nconst chatStopButton', start);
assert(start >= 0 && end > start);
const load = new Function('state', '$', 'chatHandoffScope', 'fetch', 'auraDesktopHeaders', 'updateTypingLabel',
  source.slice(start, end) + '\nreturn {updateChatStopControl, cancelActiveChatRequest};');

for (const status of ['cancellation_requested', 'execution_not_cancellable', 'already_terminal']) {
  const item = {idempotencyKey: 'request-27', handoffScope: 'owner'};
  const state = {activeChatRequest: item, isSubmitting: true};
  const button = {style: {}};
  const calls = [];
  const api = load(state, () => button, () => 'owner', async (url, options) => {
    calls.push({url, options});
    return {ok: true, json: async () => ({cancellation_status: status})};
  }, () => ({'X-Aura-Surface': 'desktop'}), () => {});
  api.updateChatStopControl();
  assert.equal(button.style.display, 'inline-flex');
  await api.cancelActiveChatRequest();
  assert.equal(calls[0].url, '/api/chat/delivery/request-27/cancel');
  assert.equal(calls[0].options.method, 'POST');
  assert.equal(state.activeChatRequest, item);
  assert.equal(state.isSubmitting, true);
  assert.equal(button.disabled, status === 'cancellation_requested');
  if (status === 'cancellation_requested') {
    await api.cancelActiveChatRequest();
    assert.equal(calls.length, 1);
  }
  state.activeChatRequest = null;
  state.isSubmitting = false;
  api.updateChatStopControl();
  assert.equal(button.style.display, 'none');
}

const state = {activeChatRequest: {idempotencyKey: 'old', handoffScope: 'another-owner'}, isSubmitting: true};
let called = false;
const api = load(state, () => ({style: {}}), () => 'owner', async () => {called = true;}, () => ({}), () => {});
await api.cancelActiveChatRequest();
assert.equal(called, false);
console.log('chat cancellation checks passed');

const revisionStart = source.indexOf('function verifiedRuntimeRevision(');
const revisionEnd = source.indexOf('\nfunction runtimeRevisionPolicySatisfied', revisionStart);
const revisionFor = new Function(source.slice(revisionStart, revisionEnd) + '\nreturn verifiedRuntimeRevision;')();
const contract = {schema: 'aura.runtime_revision.v2', required: true, verified: true,
  revision_token: 'a'.repeat(64), shell_revision_token: 'b'.repeat(64)};
assert.equal(revisionFor({runtime_revision: contract}), 'b'.repeat(64));
assert.equal(revisionFor({runtime_revision: {...contract, revision_token: 'c'.repeat(64)}}), 'b'.repeat(64));
assert.equal(revisionFor({runtime_revision: {...contract, shell_revision_token: 'd'.repeat(64)}}), 'd'.repeat(64));
assert.equal(revisionFor({runtime_revision: {...contract, verified: false}}), '');
assert.equal(revisionFor({runtime_revision: {...contract, revision_token: ''}}), '');
assert.equal(revisionFor({runtime_revision: {...contract, shell_revision_token: undefined}}), 'a'.repeat(64));
