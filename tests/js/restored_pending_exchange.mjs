import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';

const source = readFileSync(process.argv[2], 'utf8');
const start = source.indexOf('function hydrateRecentConversation(');
const end = source.indexOf('\nfunction applyVoiceSummary', start);
assert(start >= 0 && end > start);
const messages = { children: [], innerHTML: '' };
const state = { isSubmitting: false, activeChatRequest: null, chatSendQueue: [] };
const appendMsg = (role, text, html, metadata) => {
    const node = { role, text, dataset: { historyTurnId: metadata.historyTurnId || '', historyRole: role } };
    messages.children.push(node);
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
    'conversationEntriesToMessages', 'appendMsg', 'updateLanePlaceholder',
    source.slice(start, end) + '\nreturn hydrateRecentConversation;'
)({ messages }, () => messages, state, host => host.children.length === 0,
    convert, appendMsg, () => {});

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
console.log('restored pending exchange checks passed');
