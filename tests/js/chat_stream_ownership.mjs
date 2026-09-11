import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';

const source = readFileSync(process.argv[2], 'utf8');
const start = source.indexOf('function ownsChatStreamEvent(');
const end = source.indexOf('\nfunction handleWsEvent(', start);
assert(start >= 0 && end > start);
const state = { activeChatRequest: null };
const owns = new Function('state', source.slice(start, end)
    + '\nreturn ownsChatStreamEvent;')(state);
assert.equal(owns({ idempotency_key: 'a' }), false);
state.activeChatRequest = { idempotencyKey: 'a' };
assert.equal(owns({ idempotency_key: 'a' }), true);
assert.equal(owns({ idempotency_key: 'b' }), false);
assert.equal(owns({}), false);
state.activeChatRequest = { idempotencyKey: 'b' };
assert.equal(owns({ idempotency_key: 'a' }), false);
assert(source.includes("startsWith('chat_stream_') && !ownsChatStreamEvent(data)"));
console.log('chat stream ownership: PASS');
