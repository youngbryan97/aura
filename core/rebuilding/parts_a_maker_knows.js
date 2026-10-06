// Parts a maker already knows how to make, each written once and checked by use (parts_a_maker_knows.py).
//
// Each section below is the body of a part: run once when the page loads, given `app`, through which
// it plugs into the frame. A section starts with a line `//== name ==`. Parts edit the document only
// through app.editing (document_editing.js) and save it only through app.formats
// (document_formats.js), so each works in whatever program has a document to edit. Nothing here
// knows which program it is in.

//== document page ==
// The page a document is written on: paper of its size on a desk, with its margins, counted in pages
// as it grows, zoomed, its words counted. The paper is the size the program is said to use, else the
// one usual where the person is.
const PAPER = { Letter: [8.5, 11], A4: [8.27, 11.69], Legal: [8.5, 14] };
const said = "__PAPER__";
const usual = PAPER[said] ? said : (/^(en-US|en-CA|es-MX|es-US|fr-CA|en-PH)$/i.test(navigator.language || "") ? "Letter" : "A4");
const setup = Object.assign({ paper: usual, orientation: "portrait", margin: 1 }, app.kept("page setup", {}) || {});
const css = app.make("style", { text: `
.doc-desk { min-height: 100%; box-sizing: border-box; padding: 28px 24px 56px; background: #e9ebee; display: flex; justify-content: center; align-items: flex-start; }
.doc-page { position: relative; box-sizing: border-box; flex: none; width: var(--page-w); min-height: var(--page-h); padding: var(--margin);
  background: #fff; border-radius: 2px; box-shadow: 0 1px 3px rgba(0,0,0,.12), 0 8px 24px rgba(0,0,0,.08); transform-origin: top center; }
.doc-text { outline: none; min-height: calc(var(--page-h) - 2 * var(--margin)); font: 12pt/1.15 "Aptos", "Calibri", "Helvetica Neue", Helvetica, Arial, sans-serif;
  color: #1b1b1b; overflow-wrap: break-word; caret-color: var(--accent); }
.doc-text p { margin: 0 0 8pt; }
.doc-text h1 { font-size: 20pt; font-weight: 600; margin: 12pt 0 6pt; color: #12263f; }
.doc-text h2 { font-size: 16pt; font-weight: 600; margin: 10pt 0 4pt; color: #12263f; }
.doc-text h3 { font-size: 13pt; font-weight: 600; margin: 8pt 0 4pt; color: #1f3b5c; }
.doc-text ul, .doc-text ol { margin: 0 0 8pt; padding-left: 0.4in; }
.doc-text li { margin: 0 0 2pt; }
.doc-text blockquote { margin: 0 0 8pt 0.3in; padding-left: 10pt; border-left: 3px solid #cfd6de; color: #3b3b3b; font-style: italic; }
.doc-text pre { font: 10.5pt/1.45 Menlo, Consolas, "Courier New", monospace; background: #f5f7f9; padding: 8pt 10pt; border-radius: 4px; white-space: pre-wrap; }
.doc-text a { color: #0b57d0; }
.doc-text hr { border: 0; border-top: 1px solid #c9ced4; margin: 10pt 0; }
.doc-text img { max-width: 100%; height: auto; }
.doc-mark { position: absolute; left: 0; right: 0; height: 0; border-top: 1px dashed #d3d8de; pointer-events: none; }
.doc-mark span { position: absolute; right: 10px; top: 3px; font: 10px var(--font); color: #9aa1a9; }
@media print {
  .doc-desk { padding: 0; background: #fff; display: block; }
  .doc-page { width: auto; min-height: 0 !important; padding: 0; box-shadow: none; zoom: 1 !important; }
  .doc-mark { display: none; }
}` });
document.head.append(css);
const sheet = app.make("style");
document.head.append(sheet);
const desk = app.make("div", { class: "doc-desk" });
const page = app.make("div", { class: "doc-page", role: "document", "aria-label": "Page" });
const doc = app.make("div", { class: "doc-text", contenteditable: "true", spellcheck: "true", role: "textbox", "aria-multiline": "true", "aria-label": "Document text" });
doc.innerHTML = "<p><br></p>";
page.append(doc);
desk.append(page);
app.work.replaceChildren(desk);
app.doc = doc;
app.page = page;
let zoom = 100, queued = false;
const inches = () => { const [w, h] = PAPER[setup.paper] || PAPER.Letter; return setup.orientation === "landscape" ? [h, w] : [w, h]; };
function layout() {
  const [w, h] = inches();
  page.dataset.paper = setup.paper;
  page.dataset.orientation = setup.orientation;
  sheet.textContent = `.doc-page { --page-w: ${w}in; --page-h: ${h}in; --margin: ${setup.margin}in; }\n@page { size: ${w}in ${h}in; margin: ${setup.margin}in; }`;
  count();
}
function count() {
  const tall = inches()[1] * 96;
  page.style.minHeight = "";
  const n = Math.max(1, Math.ceil((page.scrollHeight - 4) / tall));
  page.style.minHeight = `calc(var(--page-h) * ${n})`;
  page.dataset.pages = String(n);
  page.querySelectorAll(".doc-mark").forEach((m) => m.remove());
  for (let i = 1; i < n; i++) page.append(app.make("div", { class: "doc-mark", contenteditable: "false", style: { top: `${i * tall}px` } }, [app.make("span", { text: `Page ${i + 1}` })]));
}
function later() { if (queued) return; queued = true; requestAnimationFrame(() => { queued = false; count(); app.refresh(); }); }
function caretPage() {
  const r = app.editing && app.editing.range();
  if (!r) return 1;
  const box = r.getBoundingClientRect(), top = page.getBoundingClientRect().top;
  return Math.min(Number(page.dataset.pages || 1), Math.max(1, Math.floor((box.top - top) / (inches()[1] * 96 * zoom / 100)) + 1));
}
function zoomTo(z) { zoom = Math.min(300, Math.max(50, Math.round(z / 10) * 10)); page.style.zoom = String(zoom / 100); app.refresh(); }
doc.addEventListener("input", later);
app.on("change", later);
app.on("ready", () => { doc.focus({ preventScroll: true }); later(); });
app.pageSetup = { get: () => ({ ...setup }), set(next) { Object.assign(setup, next); app.keep("page setup", setup); layout(); app.emit && app.emit("page setup"); }, inches };
app.status({ label: "Page", value: () => `Page ${caretPage()} of ${page.dataset.pages || 1}` });
app.status({ label: "Words", value: () => { const n = app.editing ? app.editing.words() : 0; return `${n} ${n === 1 ? "word" : "words"}`; } });
app.status({ label: "Zoom", side: "right", value: () => `${zoom}%` });
app.command({ label: "Zoom in", menu: "View", keys: "Mod+=", run: () => zoomTo(zoom + 10) });
app.command({ label: "Zoom out", menu: "View", keys: "Mod+-", run: () => zoomTo(zoom - 10) });
app.command({ label: "Actual size", menu: "View", keys: "Mod+0", run: () => zoomTo(100), enabled: () => zoom !== 100 });
layout();

//== character styles ==
// Bold, italic, underline, strikethrough, superscript and subscript on the selected words, and their removal.
const E = app.editing;
document.head.append(app.make("style", { text: `
.app-tool[data-command="Bold"] .icon { font-weight: 800; }
.app-tool[data-command="Italic"] .icon { font-style: italic; font-family: Georgia, "Times New Roman", serif; }
.app-tool[data-command="Underline"] .icon { text-decoration: underline; text-underline-offset: 2px; }
.app-tool[data-command="Strikethrough"] .icon { text-decoration: line-through; }
.app-tool[data-command="Superscript"] .icon, .app-tool[data-command="Subscript"] .icon { font-size: 13px; }` }));
const style = (label, name, icon, keys) => app.command({ label, menu: "Format", group: "Font", icon, keys, run: () => E.inline(name), active: () => E.isOn(name) });
style("Bold", "bold", "B", "Mod+B");
style("Italic", "italic", "I", "Mod+I");
style("Underline", "underline", "U", "Mod+U");
style("Strikethrough", "strikeThrough", "S", "Mod+Shift+X");
style("Superscript", "superscript", "x²", "Mod+.");
style("Subscript", "subscript", "x₂", "Mod+,");
app.command({ label: "Clear formatting", menu: "Format", keys: "Mod+\\", run: () => E.clear() });

//== fonts ==
// The typeface and size of the selected words, and making them a step larger or smaller.
const E = app.editing;
const FACES = ["Aptos", "Arial", "Avenir Next", "Calibri", "Cambria", "Courier New", "Garamond", "Georgia", "Helvetica Neue", "Menlo", "Palatino", "Times New Roman", "Trebuchet MS", "Verdana"];
const faces = FACES.filter((f) => E.installed(f) || f === "Arial" || f === "Times New Roman" || f === "Courier New");
app.control({ label: "Font", group: "Font", kind: "select", options: faces, set: (face) => E.style({ fontFamily: face }), value: () => E.styleOf("fontFamily") });
app.control({ label: "Font size", group: "Font", kind: "select", options: E.SIZES_PT.map(String), set: (pt) => E.style({ fontSize: Number(pt) }), value: () => E.styleOf("fontSize") });
const step = (by) => {
  const now = Number(E.styleOf("fontSize")) || 12;
  const sizes = E.SIZES_PT;
  const next = by > 0 ? sizes.find((s) => s > now) || sizes[sizes.length - 1] : [...sizes].reverse().find((s) => s < now) || sizes[0];
  E.style({ fontSize: next });
};
app.command({ label: "Grow font", menu: "Format", keys: "Mod+Shift+.", run: () => step(1) });
app.command({ label: "Shrink font", menu: "Format", keys: "Mod+Shift+,", run: () => step(-1) });

//== text colours ==
// The colour of the selected words, and a highlight behind them.
const E = app.editing;
document.head.append(app.make("style", { text: `
.app-field.colour-pick { position: relative; }
.app-field.colour-pick input[type=color] { width: 30px; height: 26px; padding: 1px; border-radius: 4px; cursor: pointer; }
.colour-pick .tag { font-size: 12px; font-weight: 600; color: var(--ink); padding: 0 2px; border-radius: 2px; }
.colour-pick .tag.marker { background: #fff176; }` }));
const pick = (label, tag, marker, initial, apply) => {
  const c = app.control({ label, group: "Font", kind: "color", set: (v) => apply(v) });
  c.input.value = initial;
  c.input.closest("label").classList.add("colour-pick");
  c.input.closest("label").prepend(app.make("span", { class: marker ? "tag marker" : "tag", text: tag, title: label }));
  return c;
};
pick("Text colour", "A", false, "#c00000", (v) => E.style({ color: v }));
pick("Highlight", "ab", true, "#fff59d", (v) => E.style({ backgroundColor: v }));
const lit = () => { const r = E.range(); if (!r) return false; const s = E.surface(); return [...s.querySelectorAll("[style*=background]")].some((el) => r.intersectsNode(el)); };
app.command({ label: "Remove highlight", menu: "Format", run: () => E.style({ backgroundColor: "transparent" }), enabled: lit });

//== alignment ==
// Paragraphs aligned to the left, the centre, the right, or both edges.
const E = app.editing;
const along = (label, value, svg, keys) => app.command({
  label, menu: "Format", group: "Paragraph", svg, keys,
  run: () => E.blockStyle({ textAlign: value }),
  active: () => { const b = E.blocks()[0]; return !!b && getComputedStyle(b).textAlign.replace("start", "left") === value; },
});
along("Align left", "left", "M2 3h12M2 6.5h8M2 10h12M2 13.5h8", "Mod+L");
along("Center", "center", "M2 3h12M4 6.5h8M2 10h12M4 13.5h8", "Mod+E");
along("Align right", "right", "M2 3h12M6 6.5h8M2 10h12M6 13.5h8", "Mod+R");
along("Justify", "justify", "M2 3h12M2 6.5h12M2 10h12M2 13.5h12", "Mod+J");

//== lists ==
// Paragraphs made a bulleted or a numbered list, and back.
const E = app.editing;
const inList = (tag) => E.blocks().some((b) => b.closest(tag));
app.command({ label: "Bulleted list", menu: "Format", group: "Paragraph", keys: "Mod+Shift+8", svg: "M6 4h8M6 8h8M6 12h8M2.6 4h.01M2.6 8h.01M2.6 12h.01", run: () => E.list("ul"), active: () => inList("ul") });
app.command({ label: "Numbered list", menu: "Format", group: "Paragraph", keys: "Mod+Shift+7", svg: "M6.5 4h7.5M6.5 8h7.5M6.5 12h7.5M2.2 2.8l1-.6v3.4M1.9 9.4c.3-.7 2-.7 2 .3 0 .8-2 1.3-2 2.4h2.1", run: () => E.list("ol"), active: () => inList("ol") });

//== indent ==
// Paragraphs moved in from the margin by half an inch, and back; in a list, a level deeper or shallower.
const E = app.editing;
const STEP = 48;
const move = (by) => {
  if (E.blocks().some((b) => b.closest("li"))) { E.exec(by > 0 ? "indent" : "outdent"); E.changed(true); return; }
  E.blockStyle((b) => ({ marginLeft: `${Math.max(0, (parseFloat(getComputedStyle(b).marginLeft) || 0) + by)}px` }));
};
app.command({ label: "Increase indent", menu: "Format", group: "Paragraph", keys: "Mod+]", svg: "M7 4h7M7 8h7M7 12h7M2 6l2.5 2L2 10", run: () => move(STEP) });
app.command({ label: "Decrease indent", menu: "Format", group: "Paragraph", keys: "Mod+[", svg: "M7 4h7M7 8h7M7 12h7M4.5 6 2 8l2.5 2", run: () => move(-STEP) });

//== spacing ==
// The space between the lines of a paragraph, and after it.
const E = app.editing;
app.control({
  label: "Line spacing", group: "Paragraph", kind: "select", options: [["1", "1.0"], ["1.15", "1.15"], ["1.5", "1.5"], ["2", "2.0"], ["2.5", "2.5"], ["3", "3.0"]],
  set: (v) => E.blockStyle({ lineHeight: String(v) }),
  value: () => { const b = E.blocks()[0]; if (!b) return "1.15"; const s = getComputedStyle(b); const n = parseFloat(s.lineHeight) / parseFloat(s.fontSize);
    return isFinite(n) ? ["1", "1.15", "1.5", "2", "2.5", "3"].reduce((a, c) => Math.abs(c - n) < Math.abs(a - n) ? c : a) : "1.15"; },
});
app.command({ label: "Add space after paragraph", menu: "Format", run: () => E.blockStyle({ marginBottom: "12pt" }) });
app.command({ label: "Remove space after paragraph", menu: "Format", run: () => E.blockStyle({ marginBottom: "0" }) });

//== paragraph styles ==
// A paragraph made a heading, a quotation or code, or plain text again.
const E = app.editing;
const STYLES = [["p", "Normal"], ["h1", "Heading 1"], ["h2", "Heading 2"], ["h3", "Heading 3"], ["blockquote", "Quote"], ["pre", "Code"]];
app.control({
  label: "Style", group: "Styles", kind: "select", options: STYLES,
  set: (tag) => E.formatBlock(tag),
  value: () => { const b = E.blocks()[0]; const tag = b ? (b.closest("h1, h2, h3, blockquote, pre") || b).tagName.toLowerCase() : "p"; return STYLES.some(([t]) => t === tag) ? tag : "p"; },
});
STYLES.forEach(([tag, name], i) => app.command({ label: name, menu: "Styles", keys: i < 4 ? `Mod+Alt+${i}` : "", run: () => E.formatBlock(tag) }));

//== history ==
// Every change undone and redone, typing a run of words at a time.
const E = app.editing;
app.command({ label: "Undo", menu: "Edit", group: "History", keys: "Mod+Z", svg: "M5 3.5 2 6.5l3 3M2.5 6.5H9a4 4 0 0 1 0 8H6", run: () => { if (!E.undo()) app.notify("Nothing to undo"); } });
app.command({ label: "Redo", menu: "Edit", group: "History", keys: "Mod+Shift+Z", svg: "M11 3.5l3 3-3 3M13.5 6.5H7a4 4 0 0 0 0 8h3", run: () => { if (!E.redo()) app.notify("Nothing to redo"); } });
app.command({ id: "Redo (Y)", label: "Redo ", keys: "Mod+Y", run: () => E.redo() });

//== clipboard ==
// Cut, copy and paste, with the program's own clipboard behind the system's: a paste the system
// will not hand over still pastes what was last cut or copied here.
const E = app.editing;
let held = null;
const hold = () => { const r = E.range(); if (!r || r.collapsed) return; const box = document.createElement("div"); box.append(r.cloneContents()); held = { html: box.innerHTML, text: r.toString() }; };
E.surface() && E.surface().addEventListener("copy", hold);
E.surface() && E.surface().addEventListener("cut", hold);
app.command({ label: "Cut", menu: "Edit", group: "Clipboard", svg: "M5 12.5a1.8 1.8 0 1 0 0 .01M11 12.5a1.8 1.8 0 1 0 0 .01M6.2 11.2 11.5 2.5M9.8 11.2 4.5 2.5", run: () => { E.focus(); hold(); if (!document.execCommand("cut")) E.exec("delete"); E.changed(true); } });
app.command({ label: "Copy", menu: "Edit", group: "Clipboard", svg: "M5.5 5.5h7v8h-7zM3.5 10.5V2.5h7", run: () => { E.focus(); hold(); document.execCommand("copy"); } });
app.command({
  label: "Paste", menu: "Edit", group: "Clipboard", svg: "M5.5 3.5h-2v10h9v-10h-2M6 2.5h4v2H6z",
  run: async () => {
    let html = "", text = "";
    try {
      if (navigator.clipboard && navigator.clipboard.read) {
        for (const item of await navigator.clipboard.read()) {
          if (item.types.includes("text/html")) html = await (await item.getType("text/html")).text();
          if (item.types.includes("text/plain")) text = await (await item.getType("text/plain")).text();
        }
      }
    } catch (e) { /* the system kept its clipboard to itself */ }
    if (!html && !text && held) ({ html, text } = held);
    if (!html && !text) { app.notify("Nothing to paste"); return; }
    E.insertHTML(html ? E.cleaned(html) : text.replace(/&/g, "&amp;").replace(/</g, "&lt;"));
  },
});
app.command({ label: "Select all", menu: "Edit", run: () => { const s = E.surface(); if (!s) return; s.focus(); const r = document.createRange(); r.selectNodeContents(s); E.select(r); } });

//== find and replace ==
// Finding words in the document, every one shown, going from one to the next, and replacing them.
const E = app.editing;
const can = typeof CSS !== "undefined" && CSS.highlights && typeof Highlight !== "undefined";
document.head.append(app.make("style", { text: `
::highlight(found) { background-color: #fff2a8; }
::highlight(found-now) { background-color: #ffb74d; }
.find-panel { display: grid; gap: 10px; }
.find-panel h3 { margin: 0; font-size: 14px; }
.find-panel label { display: grid; gap: 4px; font-size: 12px; color: var(--muted); }
.find-panel label.inline { display: flex; align-items: center; gap: 6px; color: var(--ink); }
.find-panel input[type=text] { border: 1px solid var(--line); border-radius: 4px; padding: 6px 8px; font: inherit; }
.find-panel .row { display: flex; gap: 6px; flex-wrap: wrap; }
.find-panel button { padding: 5px 10px; border: 1px solid var(--line); border-radius: 4px; background: #fff; font: inherit; cursor: pointer; }
.find-panel button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
.find-panel .found { font-size: 12px; color: var(--muted); min-height: 16px; }` }));
let ranges = [], at = -1;
const field = (label, type = "text") => { const input = app.make("input", { type, "aria-label": label }); return [input, app.make("label", { class: type === "checkbox" ? "inline" : "" }, type === "checkbox" ? [input, label] : [label, input])]; };
const [what, whatLabel] = field("Find");
const [instead, insteadLabel] = field("Replace with");
const [matchCase, caseLabel] = field("Match case", "checkbox");
const [wholeWord, wholeLabel] = field("Whole words only", "checkbox");
const found = app.make("div", { class: "found", role: "status", "aria-live": "polite" });
const button = (label, fn, primary = false) => app.make("button", { type: "button", class: primary ? "primary" : "", text: label, onclick: fn });
const panel = app.make("div", { class: "find-panel", role: "search", "aria-label": "Find and replace" }, [
  app.make("h3", { text: "Find and replace" }), whatLabel, insteadLabel, caseLabel, wholeLabel, found,
  app.make("div", { class: "row" }, [button("Previous", () => go(-1)), button("Next", () => go(1), true)]),
  app.make("div", { class: "row" }, [button("Replace", replaceOne), button("Replace all", replaceAll), button("Close", () => { app.side(null); paint(); })]),
]);
function search() {
  ranges = E.find(what.value, { matchCase: matchCase.checked, wholeWord: wholeWord.checked });
  if (at >= ranges.length) at = ranges.length - 1;
  paint();
}
function paint() {
  const showing = document.getElementById("app-side").contains(panel);
  if (can) {
    CSS.highlights.set("found", new Highlight(...(showing ? ranges : [])));
    CSS.highlights.set("found-now", new Highlight(...(showing && ranges[at] ? [ranges[at]] : [])));
  }
  found.textContent = !what.value ? "" : ranges.length ? `${at >= 0 ? at + 1 : 0} of ${ranges.length}` : "No matches";
}
function go(by) {
  search();
  if (!ranges.length) return;
  at = at < 0 ? (by > 0 ? 0 : ranges.length - 1) : (at + by + ranges.length) % ranges.length;
  E.select(ranges[at]);
  paint();
}
function replaceOne() {
  search();
  if (!ranges.length) return;
  if (at < 0) at = 0;
  const r = ranges[at];
  r.deleteContents(); r.insertNode(document.createTextNode(instead.value));
  E.surface().normalize(); E.changed(true); app.changed();
  search(); if (ranges.length) { at = Math.min(at, ranges.length - 1); E.select(ranges[at]); }
  paint();
}
function replaceAll() {
  search();
  const n = ranges.length;
  for (const r of [...ranges].reverse()) { r.deleteContents(); r.insertNode(document.createTextNode(instead.value)); }
  E.surface().normalize(); E.changed(true); app.changed();
  at = -1; search();
  found.textContent = n ? `Replaced ${n}` : "No matches";
}
what.addEventListener("input", () => { at = -1; search(); if (ranges.length) { at = 0; paint(); } });
for (const box of [matchCase, wholeWord]) box.addEventListener("change", () => { at = -1; search(); });
what.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); go(e.shiftKey ? -1 : 1); } else if (e.key === "Escape") { app.side(null); paint(); } });
const open = (replacing) => {
  const r = E.range();
  if (r && !r.collapsed && r.toString().length < 80) what.value = r.toString();
  app.side(panel); search();
  (replacing && what.value ? instead : what).focus(); what.select();
};
app.command({ label: "Find", menu: "Edit", group: "Editing", keys: "Mod+F", svg: "M7 3a4 4 0 1 1 0 8 4 4 0 0 1 0-8zM10 10l4 4", run: () => open(false) });
app.command({ label: "Replace", menu: "Edit", group: "Editing", keys: "Mod+Shift+H", svg: "M2 4.5h7M6.5 2l2.5 2.5L6.5 7M14 11.5H7M9.5 9 7 11.5 9.5 14", run: () => open(true) });
app.on("change", () => { if (document.getElementById("app-side").contains(panel)) search(); });

//== tables ==
// Tables put in the document, with rows and columns added and taken away, and Tab going cell to cell.
const E = app.editing;
const cell = (tag = "td") => `<${tag}><br></${tag}>`;
const here = () => { const r = E.range(); const n = r && (r.startContainer.nodeType === 1 ? r.startContainer : r.startContainer.parentElement); return n ? n.closest("td, th") : null; };
const put = (el) => { const r = document.createRange(); r.selectNodeContents(el); r.collapse(true); E.select(r); };
app.command({
  label: "Insert table", menu: "Insert", group: "Insert", svg: "M2 3h12v10H2zM2 6.5h12M2 10h12M6 3v10M10 3v10",
  run: async () => {
    const got = await app.ask({ title: "Insert table", ok: "Insert", fields: [
      { name: "rows", label: "Rows", kind: "number", value: 3, min: 1, max: 50 },
      { name: "columns", label: "Columns", kind: "number", value: 3, min: 1, max: 12 },
      { name: "header", label: "Header row", kind: "checkbox", value: false }] });
    if (!got) return;
    const rows = Math.min(50, Math.max(1, Number(got.rows) || 1)), cols = Math.min(12, Math.max(1, Number(got.columns) || 1));
    const head = got.header ? `<thead><tr>${cell("th").repeat(cols)}</tr></thead>` : "";
    const body = `<tbody>${`<tr>${cell().repeat(cols)}</tr>`.repeat(got.header ? Math.max(1, rows - 1) : rows)}</tbody>`;
    E.insertHTML(`<table>${head}${body}</table><p><br></p>`);
  },
});
const table = (fn) => () => { const c = here(); if (!c) { app.notify("Put the cursor in a table first"); return; } fn(c, c.closest("tr"), c.closest("table")); E.changed(true); };
app.command({ label: "Insert row below", menu: "Table", enabled: () => !!here(), run: table((c, tr) => { const row = tr.cloneNode(true); row.querySelectorAll("td, th").forEach((x) => { x.innerHTML = "<br>"; }); tr.after(row); put(row.cells[c.cellIndex]); }) });
app.command({ label: "Insert row above", menu: "Table", enabled: () => !!here(), run: table((c, tr) => { const row = tr.cloneNode(true); row.querySelectorAll("td, th").forEach((x) => { x.innerHTML = "<br>"; }); tr.before(row); put(row.cells[c.cellIndex]); }) });
app.command({ label: "Insert column right", menu: "Table", enabled: () => !!here(), run: table((c, tr, t) => { const i = c.cellIndex; for (const row of t.rows) { const x = document.createElement(row.cells[0] && row.cells[0].tagName === "TH" ? "th" : "td"); x.innerHTML = "<br>"; (row.cells[i] || row.cells[row.cells.length - 1]).after(x); } }) });
app.command({ label: "Delete row", menu: "Table", enabled: () => !!here(), run: table((c, tr, t) => { if (t.rows.length <= 1) t.remove(); else tr.remove(); }) });
app.command({ label: "Delete column", menu: "Table", enabled: () => !!here(), run: table((c, tr, t) => { const i = c.cellIndex; if (tr.cells.length <= 1) { t.remove(); return; } for (const row of t.rows) if (row.cells[i]) row.cells[i].remove(); }) });
app.command({ label: "Delete table", menu: "Table", enabled: () => !!here(), run: table((c, tr, t) => t.remove()) });
document.addEventListener("keydown", (e) => {
  if (e.key !== "Tab" || e.ctrlKey || e.metaKey || e.altKey) return;
  const c = here();
  if (!c) return;
  e.preventDefault(); e.stopImmediatePropagation();
  const cells = [...c.closest("table").querySelectorAll("td, th")];
  const i = cells.indexOf(c) + (e.shiftKey ? -1 : 1);
  if (i >= 0 && i < cells.length) put(cells[i]);
  else if (!e.shiftKey) { const tr = c.closest("tr"), row = tr.cloneNode(true); row.querySelectorAll("td, th").forEach((x) => { x.innerHTML = "<br>"; }); tr.after(row); put(row.cells[0]); E.changed(true); }
}, true);

//== pictures ==
// A picture from a file put in the document, sized to the page, and its size and description changed.
const E = app.editing;
document.head.append(app.make("style", { text: ".doc-text img.picked { outline: 2px solid var(--accent); outline-offset: 2px; }" }));
let picked = null;
app.command({
  label: "Insert picture", menu: "Insert", group: "Insert", svg: "M2 3h12v10H2zM2 11l3.5-3.5 3 3 2-2L14 12M10.5 6h.01",
  run: async () => {
    const range = E.range();
    const file = await app.pickFile("image/*");
    if (!file) return;
    if (range) E.select(range);
    const alt = file.name.replace(/\.[^.]+$/, "").replace(/[-_]+/g, " ");
    E.insertHTML(`<img src="${file.dataUrl}" alt="${alt.replace(/"/g, "&quot;")}" style="width: 60%">`);
  },
});
document.addEventListener("click", (e) => {
  const img = e.target.closest && e.target.closest(".doc-text img");
  if (picked) picked.classList.remove("picked");
  picked = img || null;
  if (picked) picked.classList.add("picked");
});
app.command({
  label: "Picture size", menu: "Format", enabled: () => !!(picked || (E.surface() && E.surface().querySelector("img"))), run: async () => {
    const img = picked || (E.surface() && E.surface().querySelector("img"));
    if (!img) { app.notify("Click a picture first"); return; }
    const got = await app.ask({ title: "Picture", fields: [
      { name: "width", label: "Width (% of the page)", kind: "number", value: parseInt(img.style.width, 10) || 100, min: 5, max: 100 },
      { name: "alt", label: "Description", value: img.alt || "" }] });
    if (!got) return;
    img.style.width = `${Math.min(100, Math.max(5, Number(got.width) || 100))}%`;
    img.alt = got.alt || "";
    E.changed(true);
  },
});

//== links ==
// A link on the selected words to a web address, and taking it off again.
const E = app.editing;
app.command({
  label: "Insert link", menu: "Insert", group: "Insert", keys: "Mod+K", svg: "M6.5 9.5l3-3M7 4.5l1-1a2.8 2.8 0 0 1 4 4l-1 1M9 11.5l-1 1a2.8 2.8 0 0 1-4-4l1-1",
  run: async () => {
    const r = E.range();
    const words = r ? r.toString() : "";
    const got = await app.ask({ title: "Insert link", fields: [{ name: "text", label: "Text to show", value: words }, { name: "address", label: "Address", value: "https://" }] });
    if (!got || !got.address || got.address === "https://") return;
    const address = /^[a-z][a-z0-9+.-]*:/i.test(got.address) ? got.address : `https://${got.address}`;
    if (r) E.select(r);
    const text = (got.text || words || address).replace(/&/g, "&amp;").replace(/</g, "&lt;");
    E.insertHTML(`<a href="${address.replace(/"/g, "%22")}">${text}</a>`);
  },
});
app.command({ label: "Remove link", menu: "Insert", run: () => { E.exec("unlink"); E.changed(true); },
  enabled: () => { const r = E.range(); return !!r && [...E.surface().querySelectorAll("a[href]")].some((a) => r.intersectsNode(a)); } });
document.addEventListener("click", (e) => {
  const a = e.target.closest && e.target.closest(".doc-text a[href]");
  if (a && (e.metaKey || e.ctrlKey)) { e.preventDefault(); window.open(a.href, "_blank", "noopener"); }
});

//== insertions ==
// A page break, a line across the page, today's date, or a symbol, where the cursor is.
const E = app.editing;
document.head.append(app.make("style", { text: `
.doc-text .page-break { break-after: page; height: 0; margin: 14pt 0; border-top: 1px dashed #aab2bb; position: relative; }
.doc-text .page-break::after { content: "Page break"; position: absolute; left: 50%; top: -8px; transform: translateX(-50%); background: #fff; padding: 0 6px; font: 10px var(--font); color: #8a939c; }
@media print { .doc-text .page-break { border: 0; margin: 0; } .doc-text .page-break::after { display: none; } }
.symbol-grid { display: grid; grid-template-columns: repeat(8, 32px); gap: 4px; }
.symbol-grid button { height: 32px; border: 1px solid var(--line); border-radius: 4px; background: #fff; font-size: 16px; cursor: pointer; }
.symbol-grid button:hover { background: var(--hover); }` }));
app.command({ label: "Page break", menu: "Insert", group: "Insert", keys: "Mod+Enter", svg: "M3 2v4h10V2M3 14v-4h10v4M1.5 8h2M6 8h4M12.5 8h2", run: () => E.insertHTML('<div class="page-break" data-break="page" contenteditable="false"></div><p><br></p>') });
app.command({ label: "Horizontal line", menu: "Insert", run: () => E.insertHTML("<hr><p><br></p>") });
app.command({ label: "Date", menu: "Insert", run: () => E.insertHTML(new Date().toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" })) });
const SYMBOLS = "© ® ™ € £ ¥ ¢ § ¶ † ‡ • … — – ° ± × ÷ ≠ ≤ ≥ ∞ √ π µ ← → ↑ ↓ ½ ¼ ¾ “ ” ‘ ’ « » ✓".split(" ");
app.command({
  label: "Symbol", menu: "Insert", group: "Insert", icon: "Ω",
  run: () => {
    const range = E.range();
    const grid = app.make("div", { class: "symbol-grid" }, SYMBOLS.map((s) => app.make("button", { type: "button", text: s, title: s, "aria-label": s,
      onclick: () => { if (range) E.select(range); E.insertHTML(s); app.side(null); } })));
    app.side(app.make("div", { role: "dialog", "aria-label": "Symbol" }, [app.make("h3", { text: "Insert a symbol", style: { margin: "0 0 10px", fontSize: "14px" } }), grid]));
  },
});

//== proofing ==
// Spelling marked as it is typed, by the system's own speller; and the common slips a reader trips
// on (a word twice, a sentence not begun with a capital, "a" before a vowel sound, stray spaces)
// found, shown where they are, and mended one at a time or all together.
const E = app.editing;
const can = typeof CSS !== "undefined" && CSS.highlights && typeof Highlight !== "undefined";
document.head.append(app.make("style", { text: `
::highlight(slip) { text-decoration: underline wavy #2f6fdf; text-decoration-thickness: 1.5px; text-underline-offset: 3px; }
.proof-panel { display: grid; gap: 8px; }
.proof-panel h3 { margin: 0; font-size: 14px; }
.proof-panel .slip { border: 1px solid var(--line); border-radius: 6px; padding: 8px; display: grid; gap: 6px; }
.proof-panel .slip .what { font-size: 12px; color: var(--muted); }
.proof-panel .slip .row { display: flex; gap: 6px; }
.proof-panel button { padding: 4px 10px; border: 1px solid var(--line); border-radius: 4px; background: #fff; font: inherit; cursor: pointer; }
.proof-panel button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }` }));
const AN = /^(hour|honest|honou?r|heir|herb)/i, A = /^(uni|use|usu|uti|one|once|eu|ewe|u\b)/i;
const vowelSound = (w) => (/^[aeiou]/i.test(w) && !A.test(w)) || AN.test(w);
const RULES = [
  { what: "This word is repeated", find: /(?<![\p{L}'])([\p{L}']+)(\s+)\1(?![\p{L}'])/giu, span: (m) => [0, m[0].length], right: (m) => m[1] },
  { what: 'Use "an" before a vowel sound', find: /(?<![\p{L}])(a)(\s+)(\p{L}+)/giu, when: (m) => m[3] !== m[3].toUpperCase() && vowelSound(m[3]),
    span: (m) => [0, m[1].length], right: (m) => (m[1] === "A" ? "An" : "an") },
  { what: 'Use "a" before a consonant sound', find: /(?<![\p{L}])(an)(\s+)(\p{L}+)/giu, when: (m) => m[3] !== m[3].toUpperCase() && !vowelSound(m[3]),
    span: (m) => [0, m[1].length], right: (m) => m[1][0] },
  { what: "A sentence starts with a capital letter", find: /(^|[.!?]["”’)]*\s+)(\p{Ll})/gu, when: (m, text) => !/\b(e\.g|i\.e|etc|vs|mr|mrs|ms|dr|st|no|p)\.\s+$/i.test(text.slice(Math.max(0, m.index - 6), m.index + m[1].length)),
    span: (m) => [m[1].length, m[1].length + m[2].length], right: (m) => m[2].toUpperCase() },
  { what: '"I" is always a capital', find: /(^|\s)(i)(?=[\s,.!?;:'’]|$)/gu, span: (m) => [m[1].length, m[1].length + 1], right: () => "I" },
  { what: "Two spaces where one would do", find: /(\S)( {2,})(?=\S)/gu, span: (m) => [1, 1 + m[2].length], right: () => " " },
  { what: "No space goes before this mark", find: /(\S)(\s+)([,.;:!?])(?=\s|$)/gu, span: (m) => [1, 1 + m[2].length], right: () => "" },
];
let slips = [];
function paragraphs() { const s = E.surface(); return s ? [...s.querySelectorAll("p, li, h1, h2, h3, blockquote, td, th")].filter((b) => !b.querySelector("p, li, td")) : []; }
function rangeIn(block, start, end) {
  const walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT);
  let seen = 0, began = false;
  const r = document.createRange();
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    const len = n.textContent.length;
    if (!began && start <= seen + len) { r.setStart(n, start - seen); began = true; }
    if (began && end <= seen + len) { r.setEnd(n, end - seen); return r; }
    seen += len;
  }
  return null;
}
function check() {
  slips = [];
  for (const block of paragraphs()) {
    const text = block.textContent.replace(/ /g, " ");
    for (const rule of RULES) {
      for (const m of text.matchAll(rule.find)) {
        if (rule.when && !rule.when(m, text)) continue;
        const [from, to] = rule.span(m);
        const r = rangeIn(block, m.index + from, m.index + to);
        if (r) slips.push({ what: rule.what, range: r, wrong: text.slice(m.index, m.index + m[0].length), right: text.slice(m.index, m.index + from) + rule.right(m) + text.slice(m.index + to, m.index + m[0].length), put: rule.right(m) });
      }
    }
  }
  if (can) CSS.highlights.set("slip", new Highlight(...slips.map((s) => s.range)));
  return slips;
}
function mend(slip) { slip.range.deleteContents(); if (slip.put) slip.range.insertNode(document.createTextNode(slip.put)); E.surface().normalize(); E.changed(true); app.changed(); }
const panel = app.make("div", { class: "proof-panel", role: "region", "aria-label": "Proofing" });
function show() {
  check();
  const list = slips.map((slip) => app.make("div", { class: "slip" }, [
    app.make("div", { class: "what", text: slip.what }),
    app.make("div", {}, [app.make("s", { text: slip.wrong.trim() || "␣" }), " → ", app.make("b", { text: slip.right.trim() || "␣" })]),
    app.make("div", { class: "row" }, [
      app.make("button", { type: "button", class: "primary", text: "Fix", onclick: () => { mend(slip); show(); } }),
      app.make("button", { type: "button", text: "Show", onclick: () => E.select(slip.range) })]),
  ]));
  panel.replaceChildren(app.make("h3", { text: "Proofing" }), app.make("div", { text: slips.length ? `${slips.length} to look at` : "No slips found. Spelling is marked as you type." }),
    ...(slips.length > 1 ? [app.make("button", { type: "button", text: "Fix all", onclick: () => { for (const s of [...check()].reverse()) mend(s); show(); } })] : []), ...list,
    app.make("button", { type: "button", text: "Close", onclick: () => { app.side(null); if (can) CSS.highlights.delete("slip"); } }));
  app.side(panel);
}
app.command({ label: "Check document", menu: "Review", group: "Proofing", keys: "F7", icon: "✓", run: show });
app.command({ label: "Spelling as you type", menu: "Review", run: () => { const s = E.surface(); if (s) s.spellcheck = !s.spellcheck; }, active: () => !!(E.surface() && E.surface().spellcheck) });
let pending = 0;
app.on("change", () => { clearTimeout(pending); pending = setTimeout(() => { if (document.getElementById("app-side").contains(panel)) show(); else if (can && CSS.highlights.has("slip")) check(); }, 700); });
app.command({
  label: "Word count", menu: "Review", run: () => {
    const text = E.text(), words = E.words();
    const paras = paragraphs().filter((p) => p.textContent.trim()).length;
    return app.tell("Word count", `Words: ${words}\nCharacters (with spaces): ${text.replace(/\n/g, "").length}\nCharacters (no spaces): ${text.replace(/\s/g, "").length}\nParagraphs: ${paras}\nPages: ${(app.page && app.page.dataset.pages) || 1}`);
  },
});

//== printing ==
// The document printed as its pages, and seen first as those pages.
const E = app.editing;
document.head.append(app.make("style", { text: `
.print-preview { position: fixed; inset: 0; z-index: 90; background: #5b6168; display: flex; flex-direction: column; }
.print-preview header { display: flex; align-items: center; gap: 10px; padding: 10px 16px; background: #2f3439; color: #fff; }
.print-preview header .count { opacity: .8; margin-right: auto; }
.print-preview header button { padding: 6px 14px; border-radius: 4px; border: 1px solid rgba(255,255,255,.35); background: transparent; color: #fff; font: inherit; cursor: pointer; }
.print-preview header button.primary { background: #fff; color: #222; }
.print-preview .sheets { flex: 1; overflow: auto; display: flex; flex-wrap: wrap; justify-content: center; gap: 24px; padding: 28px; }
.print-preview .sheet { background: #fff; box-shadow: 0 6px 20px rgba(0,0,0,.35); overflow: hidden; position: relative; }
.print-preview .sheet .inner { position: absolute; inset: 0; transform-origin: 0 0; }
.print-preview .sheet .num { position: absolute; bottom: 6px; right: 10px; font: 10px var(--font); color: #9aa1a9; }
@media print { .print-preview { display: none !important; } }` }));
function sheets() {
  const s = E.surface();
  const [w, h] = app.pageSetup ? app.pageSetup.inches() : [8.5, 11];
  const margin = (app.pageSetup ? app.pageSetup.get().margin : 1) * 96;
  const pw = w * 96, ph = h * 96, inner = ph - 2 * margin;
  const measure = app.make("div", { class: "doc-text", style: { position: "absolute", visibility: "hidden", width: `${pw - 2 * margin}px`, left: "-9999px", top: "0" } });
  document.body.append(measure);
  const pages = [[]];
  let used = 0;
  for (const block of s ? [...s.children] : []) {
    if (block.dataset && block.dataset.break === "page") { pages.push([]); used = 0; continue; }
    const copy = block.cloneNode(true);
    measure.append(copy);
    const st = getComputedStyle(copy);
    const tall = copy.getBoundingClientRect().height + parseFloat(st.marginTop) + parseFloat(st.marginBottom);
    if (used + tall > inner && pages[pages.length - 1].length) { pages.push([]); used = 0; }
    pages[pages.length - 1].push(block.cloneNode(true));
    used += tall;
  }
  measure.remove();
  return { pages, pw, ph, margin };
}
function preview() {
  const { pages, pw, ph, margin } = sheets();
  const scale = Math.min(1, 560 / pw);
  const holder = app.make("div", { class: "print-preview", role: "dialog", "aria-label": "Print preview" });
  const close = () => { holder.remove(); document.removeEventListener("keydown", esc, true); };
  const esc = (e) => { if (e.key === "Escape") { e.preventDefault(); close(); } };
  document.addEventListener("keydown", esc, true);
  holder.append(
    app.make("header", {}, [app.make("strong", { text: "Print preview" }), app.make("span", { class: "count", text: `${pages.length} ${pages.length === 1 ? "page" : "pages"}` }),
      app.make("button", { type: "button", class: "primary", text: "Print", onclick: () => { close(); window.print(); } }), app.make("button", { type: "button", text: "Close", onclick: close })]),
    app.make("div", { class: "sheets" }, pages.map((blocks, i) => {
      const text = app.make("div", { class: "doc-text", style: { position: "absolute", left: `${margin}px`, top: `${margin}px`, width: `${pw - 2 * margin}px` } }, blocks);
      const inner = app.make("div", { class: "inner", style: { width: `${pw}px`, height: `${ph}px`, transform: `scale(${scale})` } }, [text]);
      return app.make("div", { class: "sheet", style: { width: `${pw * scale}px`, height: `${ph * scale}px` } }, [inner, app.make("span", { class: "num", text: `${i + 1}` })]);
    })),
  );
  document.body.append(holder);
}
app.command({ label: "Print", menu: "File", group: "File", keys: "Mod+P", svg: "M4 6V2h8v4M4 12H2.5V6.5h11V12H12M4 9.5h8V14H4z", run: () => window.print() });
app.command({ label: "Print preview", menu: "File", run: preview });

//== new document ==
// A new, empty document, after asking what to do with changes not yet saved.
const E = app.editing;
const first = app.name;
app.command({
  label: "New document", menu: "File", keys: "Mod+N", run: async () => {
    if (!app.saved && E.text().trim()) {
      const go = await app.ask({ title: "Start a new document?", text: "The changes to this document have not been saved.", ok: "Discard changes", cancel: "Keep editing" });
      if (!go) return;
    }
    const s = E.surface();
    if (!s) return;
    s.innerHTML = "<p><br></p>";
    app.name = first;
    E.focus();
    E.changed(true);
    app.saved = true;
  },
});

//== save as ==
// The document saved under a name, as whichever kind of file the person picks, PDF among them.
const KINDS = [["docx", "Word Document (.docx)"], ["pdf", "PDF (.pdf)"], ["odt", "OpenDocument Text (.odt)"], ["rtf", "Rich Text (.rtf)"],
  ["html", "Web Page (.html)"], ["md", "Markdown (.md)"], ["txt", "Plain Text (.txt)"]];
let last = app.kept("save as", "docx");
app.command({
  label: "Save as", menu: "File", keys: "Mod+Shift+S", run: async () => {
    const got = await app.ask({ title: "Save as", ok: "Save", fields: [
      { name: "name", label: "File name", value: app.name || "Untitled" },
      { name: "kind", label: "Format", kind: "select", options: KINDS, value: last }] });
    if (!got) return;
    last = got.kind; app.keep("save as", last);
    const name = String(got.name || "Untitled").replace(/\.(docx|pdf|odt|rtf|html?|md|txt)$/i, "").trim() || "Untitled";
    app.name = name;
    await app.formats.save(got.kind, name);
  },
});
app.command({ label: "Export as PDF", menu: "File", group: "File", icon: "PDF", run: () => app.formats.save("pdf", app.name || "Untitled") });

//== page setup ==
// The paper, which way round it is, and the margins.
app.command({
  label: "Page setup", menu: "Layout", group: "Page", svg: "M4 1.5h6l3 3v10H4zM10 1.5v3h3M6 8h5M6 10.5h5", run: async () => {
    if (!app.pageSetup) { app.notify("This program has no pages to set up"); return; }
    const now = app.pageSetup.get();
    const got = await app.ask({ title: "Page setup", fields: [
      { name: "paper", label: "Paper", kind: "select", options: [["Letter", "Letter (8.5 × 11 in)"], ["A4", "A4 (210 × 297 mm)"], ["Legal", "Legal (8.5 × 14 in)"]], value: now.paper },
      { name: "orientation", label: "Orientation", kind: "select", options: [["portrait", "Portrait"], ["landscape", "Landscape"]], value: now.orientation },
      { name: "margin", label: "Margins", kind: "select", options: [["1", "Normal (1 in)"], ["0.75", "Moderate (0.75 in)"], ["0.5", "Narrow (0.5 in)"], ["1.5", "Wide (1.5 in)"]], value: String(now.margin) }] });
    if (got) app.pageSetup.set({ paper: got.paper, orientation: got.orientation, margin: Number(got.margin) });
  },
});

//== tabbed toolbar ==
// Many tools kept in tabs, a ribbon: each group of the toolbar under the tab its tools are for.
const TABS = __TABS__;
const where = (group) => { for (const [tab, groups] of TABS) if (groups.includes(group)) return tab; return TABS[0][0]; };
document.head.append(app.make("style", { text: `
#app-ribbon-tabs { display: flex; gap: 2px; padding: 4px 8px 0; background: var(--panel); flex: none; }
#app-ribbon-tabs button { border: 0; background: none; padding: 6px 14px 7px; font: inherit; color: var(--muted); cursor: pointer; border-radius: 6px 6px 0 0; }
#app-ribbon-tabs button[aria-selected="true"] { color: var(--accent); font-weight: 600; box-shadow: inset 0 -2px 0 var(--accent); }
#app-ribbon-tabs button:hover { background: var(--hover); }
#app-toolbar .app-group[hidden] { display: none; }
#app-toolbar { min-height: 44px; }` }));
const strip = app.make("div", { id: "app-ribbon-tabs", role: "tablist", "aria-label": "Toolbar tabs" });
document.getElementById("app-toolbar").before(strip);
function arrange(chosen) {
  const groups = [...document.querySelectorAll("#app-toolbar > .app-group")];
  const used = TABS.map(([tab]) => tab).filter((tab) => groups.some((g) => where(g.getAttribute("aria-label")) === tab));
  const showing = used.includes(chosen) ? chosen : used[0];
  strip.replaceChildren(...used.map((tab) => app.make("button", { type: "button", role: "tab", text: tab, "aria-selected": String(tab === showing), onclick: () => arrange(tab) })));
  for (const g of groups) g.hidden = where(g.getAttribute("aria-label")) !== showing && g.getAttribute("aria-label") !== "History";
  // Each tab's groups in the order its tab lists them; the history first of all, where a hand finds it.
  const order = ["History", ...TABS.flatMap(([, gs]) => gs)];
  const rank = (g) => { const at = order.indexOf(g.getAttribute("aria-label")); return at < 0 ? order.length : at; };
  const bar = document.getElementById("app-toolbar");
  for (const g of [...groups].sort((a, b) => rank(a) - rank(b))) bar.append(g);
}
app.on("ready", () => arrange());
