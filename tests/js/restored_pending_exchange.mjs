import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';

const source = readFileSync(process.argv[2], 'utf8');
const visibleLimit = Number(source.match(/const VISIBLE_CHAT_EXCHANGES = (\d+);/)[1]);
assert.equal(visibleLimit, 100);
const start = source.indexOf('function hydrateRecentConversation(');
const end = source.indexOf('\nfunction applyVoiceSummary', start);
assert(start >= 0 && end > start);
const messages = { children: [], innerHTML: '' };
const state = { isSubmitting: false, activeChatRequest: null, chatSendQueue: [] };
const appendMsg = (role, text, html, metadata, beforeNode = null) => {
    const node = { role, text, dataset: { historyTurnId: metadata.historyTurnId || '', historyRole: role } };
    messages.children.push(node);
    if (beforeNode) messages.insertBefore(node, beforeNode);
    return node;
};
messages.insertBefore = (node, successor) => {
    messages.children.splice(messages.children.indexOf(node), 1);
    const at = successor ? messages.children.indexOf(successor) : messages.children.length;
    messages.children.splice(at, 0, node);
};
const conversionStart = source.indexOf('function withEntryTimestamp(');
const conversionEnd = source.indexOf('\n// The lane-status text', conversionStart);
const convert = new Function(source.slice(conversionStart, conversionEnd)
    + '\nreturn conversationEntriesToMessages;')();
assert.equal(convert([{ id: 'one', user: 'why?', timestamp: 'original' }])[0].metadata.timestamp, 'original');
const hydrate = new Function('DOM', '$', 'state', 'transcriptIsEmpty',
    'conversationEntriesToMessages', 'appendMsg', 'updateLanePlaceholder', 'VISIBLE_CHAT_EXCHANGES',
    source.slice(start, end) + '\nreturn hydrateRecentConversation;'
)({ messages }, () => messages, state, host => host.children.length === 0,
    convert, appendMsg, () => {}, visibleLimit);

hydrate([{ id: 'one', user: 'why?', aura: '' }]);
hydrate([{ id: 'one', user: 'why?', aura: 'first answer' }]);
assert.deepEqual(messages.children.map(node => node.text), ['why?', 'first answer']);
hydrate([{ id: 'one', user: 'why?', aura: 'first answer' }]);
assert.equal(messages.children.length, 2);

// The same words in another exchange do not identify the first exchange.
appendMsg('user', 'why?', false, { historyTurnId: 'two' });
hydrate([{ id: 'two', user: 'why?', aura: 'second answer' }]);
assert.deepEqual(messages.children.map(node => node.text),
    ['why?', 'first answer', 'why?', 'second answer']);

appendMsg('user', 'pending', false, { historyTurnId: 'three' });
state.isSubmitting = true;
hydrate([{ id: 'three', user: 'pending', aura: 'owned by active delivery' }]);
assert.equal(messages.children.length, 5);
state.isSubmitting = false;
hydrate([{ id: 'three', user: 'pending', aura: 'owned by active delivery' }]);
assert.equal(messages.children.length, 6);

// A completed answer is inserted beside its question, before a later pair.
messages.children.length = 0;
hydrate([{ id: 'early', user: 'early?', aura: '' }, { id: 'later', user: 'later?', aura: 'later answer' }]);
hydrate([{ id: 'early', user: 'early?', aura: 'early answer' }, { id: 'later', user: 'later?', aura: 'later answer' }]);
assert.deepEqual(messages.children.map(node => node.text), ['early?', 'early answer', 'later?', 'later answer']);
hydrate([{ id: 'unknown', user: 'early?', aura: 'unrelated' }]);
assert.equal(messages.children.length, 4);
assert(!source.includes('hydrateConversationHistory: !state.bootstrapLoaded'));

// A timed-out bootstrap can show RAM history first, then receive older disk rows.
hydrate([{ id: 'old', user: 'older question', aura: 'older answer' },
    { id: 'early', user: 'early?', aura: 'early answer' },
    { id: 'later', user: 'later?', aura: 'later answer' }]);
assert.deepEqual(messages.children.map(node => node.text),
    ['older question', 'older answer', 'early?', 'early answer', 'later?', 'later answer']);
hydrate([{ id: 'old', user: 'older question', aura: 'older answer' },
    { id: 'early', user: 'early?', aura: 'early answer' }]);
assert.equal(messages.children.length, 6);

// Never insert older rows ahead of an unbound live delivery or an active reply.
messages.children[0].dataset.historyTurnId = '';
hydrate([{ id: 'older', user: 'not yet', aura: 'not yet' },
    { id: 'early', user: 'early?', aura: 'early answer' }]);
assert.equal(messages.children.length, 6);
messages.children[0].dataset.historyTurnId = 'old';
state.activeChatRequest = {};
hydrate([{ id: 'older', user: 'not yet', aura: 'not yet' },
    { id: 'old', user: 'older question', aura: 'older answer' }]);
assert.equal(messages.children.length, 6);
state.activeChatRequest = null;

messages.children.length = 0;
hydrate(Array.from({ length: 110 }, (_, i) => ({ id: String(i), user: `question ${i}`, aura: `answer ${i}` })));
assert.equal(messages.children.length, 200);
assert.equal(messages.children[0].text, 'question 10');
assert.equal(messages.children.at(-1).text, 'answer 109');
hydrate([{ id: 'older', user: 'too old', aura: 'too old' },
    { id: '10', user: 'question 10', aura: 'answer 10' }]);
assert.equal(messages.children.length, 200);
assert.equal(messages.children[0].text, 'question 10');
const pruneStart = source.indexOf('function pruneVisibleMessages(');
const pruneEnd = source.indexOf('\nfunction renderRetryPanel', pruneStart);
const prune = new Function('VISIBLE_CHAT_EXCHANGES', 'updateLanePlaceholder',
    source.slice(pruneStart, pruneEnd) + '\nreturn pruneVisibleMessages;')(visibleLimit, () => {});
messages.removeChild = node => messages.children.splice(messages.children.indexOf(node), 1);
Object.defineProperty(messages, 'firstChild', { get: () => messages.children[0] });
appendMsg('user', 'new question', false, {});
appendMsg('aura', 'new answer', false, {});
prune(messages);
assert.equal(messages.children.length, 200);
assert.equal(messages.children[0].text, 'question 11');
console.log('restored pending exchange checks passed');
