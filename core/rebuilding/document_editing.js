// Editing a document in the page: its selection, the paragraphs it touches, styles on text and on
// paragraphs, a history to undo, and finding text. Written once here, as code.
//
// Every program that edits a document needs the same few things done right: a style put on the
// selected words, a paragraph aligned, a change undone, pasted text cleaned of another program's
// markup. Asked of her model part by part, each is a guess at contenteditable's edges; here each is
// one implementation, and a part only has to call it. Nothing here knows which program it is in: the
// document is whatever the work area edits (app.doc, else its editable element).
//
// app.editing.surface()             the editable element, or null
// app.editing.focus()               the caret back in it, where it was
// app.editing.inline(name)          toggle bold | italic | underline | strikeThrough | superscript | subscript
// app.editing.isOn(name)            whether the selection has that
// app.editing.style(css)            put CSS on the selected words: {fontFamily, fontSize (pt), color, backgroundColor}
// app.editing.styleOf(prop)         that property where the selection starts (fontSize in pt, fontFamily its first name)
// app.editing.clear()               the selected words back to plain text
// app.editing.blocks()              the paragraphs, headings and list items the selection touches
// app.editing.blockStyle(css)       put CSS on those ({textAlign, lineHeight, marginLeft, marginBottom}); a function of the block is called
// app.editing.formatBlock(tag)      make them p | h1 | h2 | h3 | blockquote | pre
// app.editing.list(kind)            toggle "ul" | "ol" on them
// app.editing.insertHTML(html)      put HTML where the caret is
// app.editing.undo() / redo()       and canUndo() / canRedo(); every change is in the history, typing by runs of words
// app.editing.find(text, options)   Ranges of the text in the document ({matchCase, wholeWord})
// app.editing.select(range)         select a range and show it
// app.editing.text()                the document as plain text
// app.editing.words()               how many words it has
(function (app) {
  "use strict";
  const BLOCKS = "p, h1, h2, h3, h4, h5, h6, li, blockquote, pre, div";
  const SIZES_PT = [8, 9, 10, 10.5, 11, 12, 14, 16, 18, 20, 22, 24, 26, 28, 36, 48, 72];

  function surface() {
    const d = app.doc;
    if (d instanceof Element) return d;
    for (const k of ["element", "el", "page", "body", "editor"]) if (d && d[k] instanceof Element) return d[k];
    return app.work.querySelector('[contenteditable="true"], [contenteditable=""]');
  }

  // ── where the selection is ─────────────────────────────────────────
  let lastRange = null;
  function inside(node) {
    const s = surface();
    return !!(s && node && (node === s || s.contains(node)));
  }
  function range() {
    const sel = window.getSelection();
    if (sel && sel.rangeCount && inside(sel.getRangeAt(0).commonAncestorContainer)) return sel.getRangeAt(0);
    return null;
  }
  document.addEventListener("selectionchange", () => { const r = range(); if (r) lastRange = r.cloneRange(); });
  function focus() {
    const s = surface();
    if (!s) return null;
    if (range()) return range();
    s.focus({ preventScroll: true });
    const sel = window.getSelection();
    sel.removeAllRanges();
    if (lastRange && inside(lastRange.commonAncestorContainer)) sel.addRange(lastRange);
    else { const r = document.createRange(); r.selectNodeContents(s); r.collapse(false); sel.addRange(r); }
    return range();
  }
  function exec(command, value = null, css = false) {
    focus();
    try { document.execCommand("styleWithCSS", false, css); } catch (e) { /* older engines style with tags */ }
    return document.execCommand(command, false, value);
  }

  // ── styles on words ────────────────────────────────────────────────
  function inline(name) { exec(name); changed(); }
  function isOn(name) { if (!range()) return false; try { return document.queryCommandState(name); } catch (e) { return false; } }
  function elementAt(r) {
    if (!r) return null;
    let n = r.startContainer;
    if (n.nodeType === 1 && n.childNodes[r.startOffset]) n = n.childNodes[r.startOffset];
    return n.nodeType === 1 ? n : n.parentElement;
  }
  let pendingSize = null;
  function sizeFonts(pt) {
    const s = surface();
    if (!s) return;
    for (const font of s.querySelectorAll('font[size="7"]')) {
      const span = document.createElement("span");
      span.style.fontSize = `${pt}pt`;
      while (font.firstChild) span.append(font.firstChild);
      for (const inner of span.querySelectorAll("[style]")) inner.style.fontSize = "";
      font.replaceWith(span);
    }
    for (const big of s.querySelectorAll('span[style*="xxx-large"]')) big.style.fontSize = `${pt}pt`;
  }
  function style(css) {
    const r = focus();
    if (!r) return;
    for (const [prop, value] of Object.entries(css || {})) {
      if (prop === "fontSize") {
        const pt = Number(String(value).replace(/pt$/, "")) || 11;
        exec("fontSize", "7", false);
        sizeFonts(pt);
        pendingSize = r.collapsed ? pt : null;
      } else if (prop === "fontFamily") exec("fontName", value, true);
      else if (prop === "color") exec("foreColor", value, true);
      else if (prop === "backgroundColor") { if (!exec("hiliteColor", value, true)) exec("backColor", value, true); }
    }
    changed();
  }
  // Whether a typeface is on this machine: text set in it measures unlike the same text in a fallback.
  const ruler = document.createElement("canvas").getContext("2d");
  const breadth = (font) => { ruler.font = `20px ${font}`; return ruler.measureText("mmmmmmmmmmlliWWQ@#ag").width; };
  const known = new Map();
  function installed(face) {
    const name = String(face).replace(/["']/g, "").trim();
    if (/^(serif|sans-serif|monospace|system-ui|cursive|fantasy|-apple-system)$/i.test(name)) return true;
    if (!known.has(name)) known.set(name, ["monospace", "serif"].some((fallback) => breadth(`"${name}", ${fallback}`) !== breadth(fallback)));
    return known.get(name);
  }
  function styleOf(prop) {
    const el = elementAt(range() || lastRange) || surface();
    if (!el) return "";
    const v = getComputedStyle(el)[prop];
    if (prop === "fontSize") return String(Math.round(parseFloat(v) * 0.75 * 2) / 2);
    // The face the words are shown in: the first of the ones asked for that is on this machine.
    if (prop === "fontFamily") return String(v).split(",").map((f) => f.replace(/["']/g, "").trim()).find(installed) || "";
    return v;
  }
  function clear() {
    exec("removeFormat");
    const r = range();
    if (r) for (const span of surface().querySelectorAll("span[style], font")) if (r.intersectsNode(span)) span.replaceWith(...span.childNodes);
    changed();
  }

  // ── paragraphs ─────────────────────────────────────────────────────
  function paragraphsFirst() {
    // Text typed straight into the surface is made a paragraph, so it can be styled as one.
    const s = surface();
    if (!s) return;
    const loose = [...s.childNodes].filter((n) => (n.nodeType === 3 && n.textContent.trim()) || (n.nodeType === 1 && !n.matches(BLOCKS + ", ul, ol, table, hr, img, figure")));
    if (loose.length) exec("formatBlock", "<p>");
  }
  function blocks() {
    const s = surface();
    let r = range() || lastRange;
    if (!s || !r) return [];
    paragraphsFirst();
    r = range() || r;
    const found = [...s.querySelectorAll(BLOCKS)].filter((el) => r.intersectsNode(el) && !el.querySelector(BLOCKS));
    if (found.length) return found;
    const at = elementAt(r);
    const own = at && at.closest(BLOCKS);
    return own && s.contains(own) && own !== s ? [own] : [];
  }
  function blockStyle(css) {
    const chosen = blocks();
    for (const b of chosen) {
      const values = typeof css === "function" ? css(b) : css;
      Object.assign(b.style, values || {});
    }
    changed(true);
    return chosen;
  }
  function formatBlock(tag) { exec("formatBlock", `<${String(tag).toLowerCase()}>`); changed(true); }
  function list(kind) { exec(kind === "ol" ? "insertOrderedList" : "insertUnorderedList"); changed(true); }
  function insertHTML(html) { exec("insertHTML", html); changed(true); }

  // ── history ────────────────────────────────────────────────────────
  // Her own record of the document, not the engine's: a paragraph's alignment set from code is not in
  // the engine's undo, and an undo that skips it leaves the document in a state no one made.
  const past = [], future = [];
  let current = null, pending = 0;
  function caret() {
    const s = surface(), r = range();
    if (!s || !r) return null;
    const before = document.createRange();
    before.selectNodeContents(s);
    before.setEnd(r.startContainer, r.startOffset);
    return before.toString().length;
  }
  function placeCaret(offset) {
    const s = surface();
    if (!s || offset === null || offset === undefined) return;
    const walker = document.createTreeWalker(s, NodeFilter.SHOW_TEXT);
    let left = offset, node = walker.nextNode(), last = null;
    while (node) {
      if (left <= node.textContent.length) break;
      left -= node.textContent.length; last = node; node = walker.nextNode();
    }
    const r = document.createRange();
    if (node) r.setStart(node, Math.max(0, left)); else if (last) r.setStart(last, last.textContent.length); else r.setStart(s, 0);
    r.collapse(true);
    const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(r);
  }
  function take() {
    clearTimeout(pending); pending = 0;
    const s = surface();
    if (!s) return;
    const now = { html: s.innerHTML, caret: caret() };
    if (current && now.html === current.html) { current.caret = now.caret ?? current.caret; return; }
    if (current) { past.push(current); if (past.length > 300) past.shift(); }
    current = now; future.length = 0;
  }
  function changed(now = false) {
    if (now) take(); else { clearTimeout(pending); pending = setTimeout(take, 400); }
  }
  function show(state) {
    const s = surface();
    if (!s || !state) return;
    s.innerHTML = state.html;
    s.focus({ preventScroll: true });
    placeCaret(state.caret);
    app.changed();
  }
  function undo() {
    if (pending) take();
    if (!past.length) return false;
    future.push(current); current = past.pop(); show(current); return true;
  }
  function redo() {
    if (pending) take();
    if (!future.length) return false;
    past.push(current); current = future.pop(); show(current); return true;
  }

  // ── finding ────────────────────────────────────────────────────────
  function find(text, options = {}) {
    const s = surface();
    if (!s || !text) return [];
    const flags = options.matchCase ? "g" : "gi";
    const quoted = String(text).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const pattern = new RegExp(options.wholeWord ? `(?<![\\p{L}\\p{N}_])${quoted}(?![\\p{L}\\p{N}_])` : quoted, flags + "u");
    const out = [];
    const walker = document.createTreeWalker(s, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      for (const m of node.textContent.matchAll(pattern)) {
        const r = document.createRange();
        r.setStart(node, m.index); r.setEnd(node, m.index + m[0].length);
        out.push(r);
      }
    }
    return out;
  }
  function select(r) {
    if (!r) return;
    const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(r);
    const box = r.getBoundingClientRect(), view = app.work.getBoundingClientRect();
    if (box.top < view.top + 40 || box.bottom > view.bottom - 40) app.work.scrollTop += box.top - view.top - view.height / 3;
  }
  function text() { const s = surface(); return s ? s.innerText.replace(/ /g, " ") : ""; }
  function words() { const t = text().trim(); return t ? t.split(/\s+/).filter((w) => /[\p{L}\p{N}]/u.test(w)).length : 0; }

  // ── what the surface does by itself ────────────────────────────────
  const KEEP_TAGS = /^(P|BR|B|STRONG|I|EM|U|S|STRIKE|DEL|SUB|SUP|H1|H2|H3|H4|UL|OL|LI|BLOCKQUOTE|PRE|CODE|A|TABLE|THEAD|TBODY|TR|TD|TH|IMG|SPAN|HR)$/;
  const KEEP_STYLE = ["font-weight", "font-style", "text-decoration", "color", "background-color", "font-size", "font-family", "text-align"];
  function cleaned(html) {
    const doc = new DOMParser().parseFromString(html, "text/html");
    const clean = (node) => {
      for (const child of [...node.childNodes]) {
        if (child.nodeType === 8) { child.remove(); continue; }
        if (child.nodeType !== 1) continue;
        clean(child);
        const tag = child.nodeName;
        if (/^(SCRIPT|STYLE|META|LINK|TITLE|IFRAME|OBJECT)$/.test(tag)) { child.remove(); continue; }
        if (!KEEP_TAGS.test(tag)) {
          const block = /^(DIV|SECTION|ARTICLE|HEADER|FOOTER|MAIN|ASIDE|NAV|H5|H6)$/.test(tag);
          if (block) { const p = doc.createElement("p"); p.append(...child.childNodes); child.replaceWith(p); } else child.replaceWith(...child.childNodes);
          continue;
        }
        for (const attr of [...child.attributes]) {
          const keep = (attr.name === "href" && tag === "A" && /^(https?:|mailto:)/i.test(attr.value)) || (attr.name === "src" && tag === "IMG" && /^(data:image\/|https?:)/i.test(attr.value))
            || (attr.name === "alt" && tag === "IMG") || attr.name === "style" || (/^(colspan|rowspan)$/.test(attr.name) && /^T[DH]$/.test(tag));
          if (!keep) child.removeAttribute(attr.name);
        }
        if (child.hasAttribute("style")) {
          const kept = KEEP_STYLE.map((p) => [p, child.style.getPropertyValue(p)]).filter(([, v]) => v);
          child.removeAttribute("style");
          for (const [p, v] of kept) child.style.setProperty(p, v);
          if (!child.getAttribute("style")) child.removeAttribute("style");
        }
        if (tag === "SPAN" && !child.attributes.length) child.replaceWith(...child.childNodes);
      }
    };
    clean(doc.body);
    return doc.body.innerHTML;
  }
  function plainAsParagraphs(text) {
    const esc = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const paras = String(text).replace(/\r\n?/g, "\n").split(/\n{2,}/);
    if (paras.length === 1 && !/\n/.test(paras[0])) return esc(paras[0]);
    return paras.map((p) => `<p>${esc(p).replace(/\n/g, "<br>")}</p>`).join("");
  }
  function wire(s) {
    if (!s || s.dataset.editing) return;
    s.dataset.editing = "on";
    try { document.execCommand("defaultParagraphSeparator", false, "p"); } catch (e) { /* the engine's own */ }
    if (!s.innerHTML.trim()) s.innerHTML = "<p><br></p>";
    s.addEventListener("input", () => {
      if (pendingSize) { sizeFonts(pendingSize); }
      if (app.saved) app.saved = false;
      changed();
    });
    s.addEventListener("paste", (e) => {
      const data = e.clipboardData;
      if (!data) return;
      const html = data.getData("text/html"), plain = data.getData("text/plain");
      if (!html && !plain) return;
      e.preventDefault();
      if (html) exec("insertHTML", cleaned(html)); else exec("insertHTML", plainAsParagraphs(plain));
      changed(true);
    });
    s.addEventListener("keydown", (e) => {
      if (e.key !== "Tab" || e.ctrlKey || e.metaKey || e.altKey) return;
      e.preventDefault();
      const inList = blocks().some((b) => b.closest("li"));
      if (inList) exec(e.shiftKey ? "outdent" : "indent");
      else if (!e.shiftKey) exec("insertText", "\u2003\u2003\u2003");
      changed(true);
    });
    // The menu bar's Edit > Undo, and the keys, come here: one history, whichever asks.
    s.addEventListener("beforeinput", (e) => {
      if (e.inputType === "historyUndo") { e.preventDefault(); undo(); }
      else if (e.inputType === "historyRedo") { e.preventDefault(); redo(); }
    });
    current = { html: s.innerHTML, caret: 0 };
  }
  app.on("ready", () => wire(surface()));
  app.on("change", () => { const s = surface(); if (s && !s.dataset.editing) wire(s); changed(true); });

  app.editing = {
    surface, focus, range, exec, inline, isOn, style, styleOf, clear, blocks, blockStyle, formatBlock, list, insertHTML,
    undo, redo, canUndo: () => past.length > 0 || !!pending, canRedo: () => future.length > 0, changed, find, select, text, words,
    cleaned, installed, SIZES_PT,
  };
})(app);
