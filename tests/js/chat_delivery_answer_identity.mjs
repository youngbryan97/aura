import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';

const source = readFileSync(process.argv[2], 'utf8');
const start = source.indexOf('function discardChatDraft(');
const end = source.indexOf('\nfunction handleWsEvent(', start);
assert(start >= 0 && end > start);
const messages = { children: [] };
function appendMsg(role, text, html, metadata) {
    const node = { text, dataset: { historyTurnId: metadata.historyTurnId, historyRole: role } };
    node.remove = () => messages.children.splice(messages.children.indexOf(node), 1);
    messages.children.push(node);
    return node;
}
const marks = [];
const render = new Function('DOM', 'appendMsg', 'markReplyConfidence',
    'let activeStreamDiv = null; let activeStreamContentRaw = "";\n'
    + source.slice(start, end) + '\nreturn renderChatDeliveryAnswer;')(
    { messages }, appendMsg, (node, confidence) => marks.push({ node, confidence }));
render({ idempotencyKey: 'one' }, { turn_id: 'first', response: 'OK.' });
render({ idempotencyKey: 'two' }, { turn_id: 'second', response: 'OK.' });
assert.equal(messages.children.length, 2, 'identical answers belong to different turns');
render({ idempotencyKey: 'two' }, { turn_id: 'second', response: 'OK.', response_confidence: 'high' });
assert.equal(messages.children.length, 2, 'same delivery replay does not duplicate');
assert.equal(marks.at(-1).confidence, 'high');
const draft = appendMsg('aura', 'unfinished draft', false, {});
const item = { idempotencyKey: 'three', streamDiv: draft };
render(item, { turn_id: 'third', response: 'Revised complete answer.' });
assert.equal(messages.children.length, 3);
assert.equal(messages.children.at(-1).text, 'Revised complete answer.');
assert.equal(item.streamDiv, null);
console.log('chat delivery answer identity: PASS');
