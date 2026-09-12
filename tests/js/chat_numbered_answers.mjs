import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(process.argv[2], 'utf8');
const append = source.slice(source.indexOf('async function appendMsg('));
const renderer = append.slice(
    append.indexOf('const render = (t) => {'),
    append.indexOf('// Build thought toggle HTML'),
);
const render = vm.runInNewContext(`${renderer}\nrender`, {
    isHtml: false,
    escHtml: text => text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'),
});

const loose = render('1. Surface.\n\n2. Temperature.\n\n3. Energy.\n\nExamples follow.');
assert.deepEqual([...loose.matchAll(/<ol start="(\d+)">/g)].map(match => +match[1]), [1, 2, 3]);
assert.equal((loose.match(/<li>/g) || []).length, 3);
assert.ok(loose.endsWith('Examples follow.'));

const continued = render('4. Fourth step.\n5. Fifth step.');
assert.equal(continued, '<ol start="4"><li>Fourth step.</li><li>Fifth step.</li></ol>');
const separate = render('1. First list.\n\nA new section.\n\n1. Second list.');
assert.equal((separate.match(/<ol start="1">/g) || []).length, 2);
assert.equal(render('Version 3. is prose.'), 'Version 3. is prose.');
assert.ok(render('2. <script>alert(1)</script>').includes('&lt;script&gt;'));
