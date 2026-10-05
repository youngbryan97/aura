// What a document is, read from the page, and the files people exchange it as: written and read here, as code.
//
// A program that makes documents saves them in the formats other programs open
// (.docx, .odt, .rtf, .html, .md, .txt) and opens them back. Asked of her model
// each time, a part that writes a .docx is a guess at a file format; here it is
// one implementation, tested against the programs that read those files, and a
// part only has to call it. Nothing here knows which program it is in: the
// document is whatever the work area edits.
//
// app.formats.blocks(el)        the document as blocks (paragraphs, headings, list items, tables) of styled runs
// app.formats.write(kind, el)   a Blob of the document as kind: "docx" | "odt" | "rtf" | "html" | "md" | "txt"
// app.formats.read(file)        Promise of the HTML of a picked file ({name, text, dataUrl, file}) of any of those kinds
// app.formats.save(kind, name)  write and download it; app.formats.open(accept) pick a file and put it in the document
(function (app) {
  const KINDS = {
    docx: { ext: "docx", label: "Word Document (.docx)", type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document" },
    odt: { ext: "odt", label: "OpenDocument Text (.odt)", type: "application/vnd.oasis.opendocument.text" },
    rtf: { ext: "rtf", label: "Rich Text (.rtf)", type: "application/rtf" },
    html: { ext: "html", label: "Web Page (.html)", type: "text/html" },
    md: { ext: "md", label: "Markdown (.md)", type: "text/markdown" },
    txt: { ext: "txt", label: "Plain Text (.txt)", type: "text/plain" },
  };
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  const BLOCK = /^(P|DIV|H[1-6]|LI|BLOCKQUOTE|PRE|TABLE|UL|OL|HR|SECTION|ARTICLE)$/;

  function documentElement() {
    const d = app.doc;
    if (d instanceof Element) return d;
    for (const k of ["element", "el", "page", "body", "editor"]) if (d && d[k] instanceof Element) return d[k];
    return app.work.querySelector('[contenteditable="true"], [contenteditable=""]') || app.work;
  }
  function hex(color) {
    const m = String(color || "").match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?/);
    if (!m || (m[4] !== undefined && Number(m[4]) === 0)) return "";
    const h = [m[1], m[2], m[3]].map((n) => Number(n).toString(16).padStart(2, "0")).join("").toUpperCase();
    return h === "000000" ? "" : h;
  }
  function runStyle(el, base) {
    const cs = getComputedStyle(el), deco = cs.textDecorationLine || cs.textDecoration || "";
    const size = parseFloat(cs.fontSize) * 0.75;
    return {
      b: Number(cs.fontWeight) >= 600 || cs.fontWeight === "bold", i: cs.fontStyle === "italic",
      u: /underline/.test(deco), s: /line-through/.test(deco), color: hex(cs.color),
      size: Math.abs(size - base) > 0.4 ? Math.round(size * 2) / 2 : 0,
      font: (cs.fontFamily || "").split(",")[0].replace(/["']/g, "").trim(), href: el.closest("a") ? el.closest("a").href : "",
    };
  }
  function blocks(root = documentElement()) {
    const base = parseFloat(getComputedStyle(root).fontSize) * 0.75, out = [];
    let current = null;
    const start = (type, el, extra = {}) => {
      current = { type, align: el ? (getComputedStyle(el).textAlign.replace("start", "left").replace("end", "right")) : "left", runs: [], ...extra };
      out.push(current);
      return current;
    };
    const inline = (node, el) => {
      if (!current) start("p", el);
      if (node.nodeType === 3) {
        const text = node.textContent.replace(/\s+/g, " ");
        if (text.trim() || (text && current.runs.length)) current.runs.push({ text, ...runStyle(node.parentElement, base) });
      } else if (node.nodeName === "BR") {
        current.runs.push({ br: true, text: "" });
      } else if (node.nodeType === 1) {
        for (const child of node.childNodes) walk(child, node);
      }
    };
    const walk = (node, parentEl, list) => {
      if (node.nodeType !== 1) return inline(node, parentEl);
      const tag = node.nodeName;
      if (getComputedStyle(node).display === "none" || tag === "SCRIPT" || tag === "STYLE") return;
      if (!BLOCK.test(tag)) return inline(node, parentEl);
      if (tag === "UL" || tag === "OL") {
        for (const child of node.children) walk(child, node, { kind: tag === "OL" ? "ol" : "ul", level: (list ? list.level + 1 : 0) });
        current = null; return;
      }
      if (tag === "HR") { start("hr", node); current = null; return; }
      if (tag === "TABLE") {
        const rows = [...node.querySelectorAll(":scope > tr, :scope > * > tr")].map((tr) => [...tr.children].map((td) => blocks(td)));
        start("table", node, { rows }); current = null; return;
      }
      const type = /^H[1-6]$/.test(tag) ? tag.toLowerCase() : tag === "LI" ? "li" : "p";
      start(type, node, type === "li" ? { list: list ? list.kind : "ul", level: list ? list.level : 0 } : {});
      for (const child of node.childNodes) {
        if (child.nodeType === 1 && BLOCK.test(child.nodeName)) { walk(child, node, list); start("p", node); }
        else walk(child, node, list);
      }
      current = null;
    };
    for (const child of root.childNodes) walk(child, root);
    return out.filter((b) => b.type === "hr" || b.type === "table" || b.runs.some((r) => r.text.trim() || r.br) || b.type !== "p" || out.length === 1);
  }

  // ── writing ─────────────────────────────────────────────────────────
  function docxRun(r) {
    if (r.br) return "<w:r><w:br/></w:r>";
    const p = [r.b && "<w:b/>", r.i && "<w:i/>", r.u && '<w:u w:val="single"/>', r.s && "<w:strike/>", r.color && `<w:color w:val="${r.color}"/>`,
      r.size && `<w:sz w:val="${Math.round(r.size * 2)}"/>`, r.font && `<w:rFonts w:ascii="${esc(r.font)}" w:hAnsi="${esc(r.font)}"/>`].filter(Boolean).join("");
    return `<w:r>${p ? `<w:rPr>${p}</w:rPr>` : ""}<w:t xml:space="preserve">${esc(r.text)}</w:t></w:r>`;
  }
  function docxBlock(b) {
    if (b.type === "hr") return '<w:p><w:pPr><w:pBdr><w:bottom w:val="single" w:sz="6" w:space="1" w:color="auto"/></w:pBdr></w:pPr></w:p>';
    if (b.type === "table") {
      return `<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/><w:tblBorders>${["top", "left", "bottom", "right", "insideH", "insideV"]
        .map((s) => `<w:${s} w:val="single" w:sz="4" w:space="0" w:color="auto"/>`).join("")}</w:tblBorders></w:tblPr><w:tblGrid>${(b.rows[0] || [])
        .map(() => `<w:gridCol w:w="${Math.floor(9360 / Math.max(1, (b.rows[0] || []).length))}"/>`).join("")}</w:tblGrid>${b.rows.map((row) => `<w:tr>${row
        .map((cell) => `<w:tc>${(cell.length ? cell : [{ type: "p", runs: [] }]).map(docxBlock).join("")}</w:tc>`).join("")}</w:tr>`).join("")}</w:tbl>`;
    }
    const pPr = [/^h[1-6]$/.test(b.type) && `<w:pStyle w:val="Heading${b.type[1]}"/>`,
      b.type === "li" && `<w:numPr><w:ilvl w:val="${b.level || 0}"/><w:numId w:val="${b.list === "ol" ? 2 : 1}"/></w:numPr>`,
      b.align && b.align !== "left" && `<w:jc w:val="${b.align === "justify" ? "both" : b.align}"/>`].filter(Boolean).join("");
    return `<w:p>${pPr ? `<w:pPr>${pPr}</w:pPr>` : ""}${b.runs.map(docxRun).join("")}</w:p>`;
  }
  function docx(bs) {
    const W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"';
    const heading = (n) => `<w:style w:type="paragraph" w:styleId="Heading${n}"><w:name w:val="heading ${n}"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>` +
      `<w:pPr><w:keepNext/><w:spacing w:before="240" w:after="80"/><w:outlineLvl w:val="${n - 1}"/></w:pPr><w:rPr><w:b/><w:sz w:val="${[40, 32, 28, 26, 24, 22][n - 1]}"/></w:rPr></w:style>`;
    const lvl = (i, fmt, text) => `<w:lvl w:ilvl="${i}"><w:start w:val="1"/><w:numFmt w:val="${fmt}"/><w:lvlText w:val="${text}"/><w:lvlJc w:val="left"/><w:pPr><w:ind w:left="${720 * (i + 1)}" w:hanging="360"/></w:pPr></w:lvl>`;
    return app.zip({
      "[Content_Types].xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' +
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>' +
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>' +
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>' +
        '<Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/></Types>',
      "_rels/.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' +
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>',
      "word/_rels/document.xml.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' +
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>' +
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/></Relationships>',
      "word/styles.xml": `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles ${W}><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>` +
        '<w:sz w:val="22"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>' +
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>' + [1, 2, 3, 4, 5, 6].map(heading).join("") +
        '<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/></w:style></w:styles>',
      "word/numbering.xml": `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:numbering ${W}>` +
        `<w:abstractNum w:abstractNumId="0">${[0, 1, 2].map((i) => lvl(i, "bullet", ["•", "◦", "▪"][i])).join("")}</w:abstractNum>` +
        `<w:abstractNum w:abstractNumId="1">${[0, 1, 2].map((i) => lvl(i, ["decimal", "lowerLetter", "lowerRoman"][i], `%${i + 1}.`)).join("")}</w:abstractNum>` +
        '<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num><w:num w:numId="2"><w:abstractNumId w:val="1"/></w:num></w:numbering>',
      "word/document.xml": `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document ${W}><w:body>${bs.map(docxBlock).join("")}` +
        '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/></w:sectPr></w:body></w:document>',
    }, KINDS.docx.type);
  }
  function odt(bs) {
    const styles = [], styleOf = (r) => {
      const props = [r.b && 'fo:font-weight="bold"', r.i && 'fo:font-style="italic"', r.u && 'style:text-underline-style="solid" style:text-underline-width="auto" style:text-underline-color="font-color"',
        r.s && 'style:text-line-through-style="solid"', r.color && `fo:color="#${r.color}"`, r.size && `fo:font-size="${r.size}pt"`, r.font && `style:font-name="${esc(r.font)}"`].filter(Boolean).join(" ");
      if (!props) return "";
      let at = styles.indexOf(props); if (at < 0) { styles.push(props); at = styles.length - 1; }
      return `T${at + 1}`;
    };
    const span = (r) => r.br ? "<text:line-break/>" : (styleOf(r) ? `<text:span text:style-name="${styleOf(r)}">${esc(r.text)}</text:span>` : esc(r.text));
    const para = (b) => b.type === "table"
      ? `<table:table>${b.rows.map((row) => `<table:table-row>${row.map((cell) => `<table:table-cell office:value-type="string">${cell.map(para).join("") || "<text:p/>"}</table:table-cell>`).join("")}</table:table-row>`).join("")}</table:table>`
      : /^h[1-6]$/.test(b.type) ? `<text:h text:outline-level="${b.type[1]}">${b.runs.map(span).join("")}</text:h>`
        : b.type === "li" ? `<text:list><text:list-item><text:p>${b.runs.map(span).join("")}</text:p></text:list-item></text:list>`
          : `<text:p${b.align && b.align !== "left" ? ` text:style-name="P${b.align}"` : ""}>${b.runs.map(span).join("")}</text:p>`;
    const body = bs.map(para).join("");
    const NS = 'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" ' +
      'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"';
    const automatic = styles.map((p, n) => `<style:style style:name="T${n + 1}" style:family="text"><style:text-properties ${p}/></style:style>`).join("") +
      ["center", "right", "justify"].map((a) => `<style:style style:name="P${a}" style:family="paragraph"><style:paragraph-properties fo:text-align="${a === "right" ? "end" : a}"/></style:style>`).join("");
    return app.zip({
      mimetype: KINDS.odt.type,
      "META-INF/manifest.xml": '<?xml version="1.0" encoding="UTF-8"?><manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.2">' +
        `<manifest:file-entry manifest:full-path="/" manifest:media-type="${KINDS.odt.type}"/><manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/></manifest:manifest>`,
      "content.xml": `<?xml version="1.0" encoding="UTF-8"?><office:document-content ${NS} office:version="1.2"><office:automatic-styles>${automatic}</office:automatic-styles>` +
        `<office:body><office:text>${body}</office:text></office:body></office:document-content>`,
    }, KINDS.odt.type);
  }
  function rtf(bs) {
    const colors = [], fonts = ["Calibri"];
    const u = (s) => [...s].map((ch) => { const c = ch.codePointAt(0); return ch === "\\" || ch === "{" || ch === "}" ? "\\" + ch : c > 127 ? (c > 0xffff ? "?" : `\\u${c > 32767 ? c - 65536 : c}?`) : ch; }).join("");
    const run = (r) => {
      if (r.br) return "\\line ";
      let pre = "";
      if (r.color) { let at = colors.indexOf(r.color); if (at < 0) { colors.push(r.color); at = colors.length - 1; } pre += `\\cf${at + 1}`; }
      if (r.font) { let at = fonts.indexOf(r.font); if (at < 0) { fonts.push(r.font); at = fonts.length - 1; } pre += `\\f${at}`; }
      pre += (r.b ? "\\b" : "") + (r.i ? "\\i" : "") + (r.u ? "\\ul" : "") + (r.s ? "\\strike" : "") + (r.size ? `\\fs${Math.round(r.size * 2)}` : "");
      return pre ? `{${pre} ${u(r.text)}}` : u(r.text);
    };
    const para = (b) => b.type === "table" ? b.rows.map((row) => row.map((cell) => cell.map((c) => c.runs.map(run).join("")).join(" ")).join("\\tab ") + "\\par\n").join("")
      : `\\pard${{ center: "\\qc", right: "\\qr", justify: "\\qj" }[b.align] || ""}${/^h[1-6]$/.test(b.type) ? `\\b\\fs${[40, 32, 28, 26, 24, 22][b.type[1] - 1]} ` : " "}` +
        `${b.type === "li" ? (b.list === "ol" ? "1.\\tab " : "\\bullet\\tab ") : ""}${b.runs.map(run).join("")}${/^h/.test(b.type) ? "\\b0\\fs22" : ""}\\par\n`;
    const body = bs.map(para).join("");
    return new Blob([`{\\rtf1\\ansi\\ansicpg1252\\deff0{\\fonttbl${fonts.map((f, n) => `{\\f${n} ${f};}`).join("")}}{\\colortbl;${colors.map((c) =>
      `\\red${parseInt(c.slice(0, 2), 16)}\\green${parseInt(c.slice(2, 4), 16)}\\blue${parseInt(c.slice(4, 6), 16)};`).join("")}}\\fs22\n${body}}`], { type: KINDS.rtf.type });
  }
  function htmlOf(bs) {
    const run = (r) => {
      if (r.br) return "<br>";
      let t = esc(r.text); const css = [r.color && `color:#${r.color}`, r.size && `font-size:${r.size}pt`, r.font && `font-family:${esc(r.font)}`].filter(Boolean).join(";");
      if (r.b) t = `<strong>${t}</strong>`; if (r.i) t = `<em>${t}</em>`; if (r.u) t = `<u>${t}</u>`; if (r.s) t = `<s>${t}</s>`;
      if (css) t = `<span style="${css}">${t}</span>`; if (r.href) t = `<a href="${esc(r.href)}">${t}</a>`;
      return t;
    };
    let html = "", open = null;
    for (const b of bs) {
      if (b.type !== "li" && open) { html += `</${open}>`; open = null; }
      const align = b.align && b.align !== "left" ? ` style="text-align:${b.align}"` : "";
      if (b.type === "li") { if (open !== b.list) { if (open) html += `</${open}>`; html += `<${b.list}>`; open = b.list; } html += `<li>${b.runs.map(run).join("")}</li>`; }
      else if (b.type === "hr") html += "<hr>";
      else if (b.type === "table") html += `<table border="1" cellspacing="0" cellpadding="4">${b.rows.map((row) => `<tr>${row.map((cell) => `<td>${htmlOf(cell)}</td>`).join("")}</tr>`).join("")}</table>`;
      else html += `<${b.type}${align}>${b.runs.map(run).join("") || "<br>"}</${b.type}>`;
    }
    return html + (open ? `</${open}>` : "");
  }
  function md(bs) {
    const run = (r) => { if (r.br) return "  \n"; let t = r.text.replace(/([*_`#\\])/g, "\\$1"); if (!t.trim()) return t;
      if (r.b) t = `**${t}**`; if (r.i) t = `*${t}*`; if (r.s) t = `~~${t}~~`; return r.href ? `[${t}](${r.href})` : t; };
    // A heading is bold by being a heading; its runs are not marked bold again.
    const said = bs.map((b) => b.type === "hr" ? "---" : b.type === "table" ? b.rows.map((row, n) => `| ${row.map((c) => c.map((x) => x.runs.map(run).join("")).join(" ")).join(" | ")} |` +
      (n === 0 ? `\n|${row.map(() => " --- ").join("|")}|` : "")).join("\n")
      : /^h[1-6]$/.test(b.type) ? "#".repeat(Number(b.type[1])) + " " + b.runs.map((r) => run({ ...r, b: false })).join("").trim()
        : `${b.type === "li" ? "  ".repeat(b.level || 0) + (b.list === "ol" ? "1. " : "- ") : ""}${b.runs.map(run).join("").trim()}`);
    // Items of one list follow one another on the next line; everything else is a paragraph apart.
    return said.map((line, n) => (n === 0 ? "" : bs[n].type === "li" && bs[n - 1].type === "li" ? "\n" : "\n\n") + line).join("") + "\n";
  }
  function txt(bs) {
    let n = 0;
    return bs.map((b) => { if (b.type !== "li" || b.list !== "ol") n = 0;
      return b.type === "hr" ? "----------" : b.type === "table" ? b.rows.map((row) => row.map((c) => txt(c).trim()).join("\t")).join("\n")
        : (b.type === "li" ? "  ".repeat(b.level || 0) + (b.list === "ol" ? `${++n}. ` : "• ") : "") + b.runs.map((r) => (r.br ? "\n" : r.text)).join("").trim(); }).join("\n\n") + "\n";
  }
  function write(kind, el) {
    const bs = blocks(el || documentElement());
    if (kind === "docx") return docx(bs);
    if (kind === "odt") return odt(bs);
    if (kind === "rtf") return rtf(bs);
    if (kind === "html") return new Blob([`<!doctype html><html><head><meta charset="utf-8"><title>${esc(app.name)}</title></head><body>${htmlOf(bs)}</body></html>`], { type: KINDS.html.type });
    if (kind === "md") return new Blob([md(bs)], { type: KINDS.md.type });
    return new Blob([txt(bs)], { type: KINDS.txt.type });
  }

  // ── reading ─────────────────────────────────────────────────────────
  async function unzip(buffer) {
    const bytes = new Uint8Array(buffer), view = new DataView(buffer), files = {};
    let end = bytes.length - 22;
    while (end >= 0 && view.getUint32(end, true) !== 0x06054b50) end--;
    if (end < 0) throw new Error("not a zip");
    let at = view.getUint32(end + 16, true);
    const count = view.getUint16(end + 10, true), dec = new TextDecoder();
    for (let n = 0; n < count; n++) {
      const method = view.getUint16(at + 10, true), size = view.getUint32(at + 20, true), nameLen = view.getUint16(at + 28, true);
      const extra = view.getUint16(at + 30, true), comment = view.getUint16(at + 32, true), local = view.getUint32(at + 42, true);
      const name = dec.decode(bytes.subarray(at + 46, at + 46 + nameLen));
      const start = local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true);
      const data = bytes.subarray(start, start + size);
      files[name] = method === 0 ? data : new Uint8Array(await new Response(new Blob([data]).stream().pipeThrough(new DecompressionStream("deflate-raw"))).arrayBuffer());
      at += 46 + nameLen + extra + comment;
    }
    return files;
  }
  const xml = (bytes) => new DOMParser().parseFromString(new TextDecoder().decode(bytes), "application/xml");
  function fromDocx(doc) {
    const W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main", at = (el, name) => el.getElementsByTagNameNS(W, name);
    const val = (el, name) => { const x = at(el, name)[0]; return x ? (x.getAttributeNS(W, "val") ?? x.getAttribute("w:val") ?? "") : null; };
    const para = (p) => {
      const style = val(p, "pStyle") || "", jc = val(p, "jc"), heading = style.match(/^Heading(\d)/i), numbered = at(p, "numPr").length;
      const runs = [...at(p, "r")].map((r) => {
        const t = [...r.childNodes].map((c) => (c.localName === "t" ? c.textContent : c.localName === "br" ? "<br>" : c.localName === "tab" ? "\t" : "")).join("");
        let h = t.split("<br>").map(esc).join("<br>");
        const rp = at(r, "rPr")[0];
        if (rp) { const on = (n) => at(rp, n).length && val(rp, n) !== "0" && val(rp, n) !== "false";
          if (on("b")) h = `<strong>${h}</strong>`; if (on("i")) h = `<em>${h}</em>`; if (at(rp, "u").length && val(rp, "u") !== "none") h = `<u>${h}</u>`; if (on("strike")) h = `<s>${h}</s>`;
          const color = val(rp, "color"); if (color && color !== "auto") h = `<span style="color:#${color}">${h}</span>`; }
        return h;
      }).join("");
      const align = jc && jc !== "left" && jc !== "start" ? ` style="text-align:${jc === "both" ? "justify" : jc}"` : "";
      return heading ? `<h${heading[1]}${align}>${runs}</h${heading[1]}>` : numbered ? `<li>${runs}</li>` : `<p${align}>${runs || "<br>"}</p>`;
    };
    const body = at(doc, "body")[0];
    return [...(body ? body.children : [])].map((el) => el.localName === "p" ? para(el)
      : el.localName === "tbl" ? `<table border="1">${[...at(el, "tr")].map((tr) => `<tr>${[...at(tr, "tc")].map((tc) => `<td>${[...at(tc, "p")].map(para).join("")}</td>`).join("")}</tr>`).join("")}</table>` : "")
      .join("").replace(/(<li>.*?<\/li>)+/g, (m) => `<ul>${m}</ul>`);
  }
  function fromOdt(doc) {
    const T = "urn:oasis:names:tc:opendocument:xmlns:text:1.0";
    const inner = (el) => [...el.childNodes].map((c) => c.nodeType === 3 ? esc(c.textContent) : c.localName === "line-break" ? "<br>" : c.localName === "s" ? " " : inner(c)).join("");
    return [...doc.getElementsByTagNameNS(T, "*")].filter((el) => el.localName === "p" || el.localName === "h")
      .map((el) => el.localName === "h" ? `<h${el.getAttributeNS(T, "outline-level") || 1}>${inner(el)}</h${el.getAttributeNS(T, "outline-level") || 1}>` : `<p>${inner(el) || "<br>"}</p>`).join("");
  }
  function fromRtf(text) {
    let out = "", depth = 0; const skip = [];
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (ch === "{") { depth++; continue; }
      if (ch === "}") { if (skip.length && skip[skip.length - 1] === depth) skip.pop(); depth--; continue; }
      if (ch === "\\") {
        const m = text.slice(i + 1).match(/^([a-z]+)(-?\d+)? ?|^([\\{}])|^'([0-9a-f]{2})/i);
        if (!m) continue;
        i += m[0].length;  // now on the last character the control took
        if (m[3]) { if (!skip.length) out += m[3]; continue; }
        if (m[4]) { if (!skip.length) out += String.fromCharCode(parseInt(m[4], 16)); continue; }
        const word = m[1];
        if (/^(fonttbl|colortbl|stylesheet|info|pict|header|footer)$/.test(word)) skip.push(depth);
        else if (!skip.length && (word === "par" || word === "line")) out += "\n";
        else if (!skip.length && word === "tab") out += "\t";
        else if (!skip.length && word === "u") { out += String.fromCharCode(((Number(m[2]) % 65536) + 65536) % 65536); if (text[i + 1] === "?") i++; }
        continue;
      }
      if (!skip.length && ch !== "\r" && ch !== "\n") out += ch;
    }
    return fromText(out);
  }
  const fromText = (text) => String(text).replace(/\r\n?/g, "\n").split(/\n{2,}|\n/).map((p) => `<p>${esc(p) || "<br>"}</p>`).join("");
  function fromMd(text) {
    const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\*(.+?)\*/g, "<em>$1</em>").replace(/~~(.+?)~~/g, "<s>$1</s>")
      .replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2">$1</a>');
    return String(text).replace(/\r\n?/g, "\n").split(/\n{2,}/).map((block) => {
      const h = block.match(/^(#{1,6})\s+(.*)/);
      if (h) return `<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`;
      if (/^\s*([-*]|\d+\.)\s/.test(block)) { const ol = /^\s*\d+\./.test(block);
        return `<${ol ? "ol" : "ul"}>${block.split("\n").map((l) => `<li>${inline(l.replace(/^\s*([-*]|\d+\.)\s+/, ""))}</li>`).join("")}</${ol ? "ol" : "ul"}>`; }
      if (/^-{3,}$/.test(block.trim())) return "<hr>";
      return `<p>${inline(block).replace(/\n/g, "<br>")}</p>`;
    }).join("");
  }
  function clean(html) {
    const doc = new DOMParser().parseFromString(String(html), "text/html");
    doc.querySelectorAll("script, style, iframe, object, embed, link, meta").forEach((el) => el.remove());
    doc.querySelectorAll("*").forEach((el) => [...el.attributes].forEach((a) => { if (/^on/i.test(a.name) || /^javascript:/i.test(a.value)) el.removeAttribute(a.name); }));
    return doc.body.innerHTML;
  }
  function kindOf(name) { const ext = String(name || "").toLowerCase().split(".").pop(); return ext === "htm" ? "html" : ext === "markdown" ? "md" : KINDS[ext] ? ext : "txt"; }
  async function read(file) {
    const kind = kindOf(file.name);
    const buffer = file.file ? await file.file.arrayBuffer() : file.dataUrl ? await (await fetch(file.dataUrl)).arrayBuffer() : new TextEncoder().encode(file.text || "").buffer;
    if (kind === "docx") return fromDocx(xml((await unzip(buffer))["word/document.xml"]));
    if (kind === "odt") return fromOdt(xml((await unzip(buffer))["content.xml"]));
    const text = file.text != null ? file.text : new TextDecoder().decode(buffer);
    if (kind === "rtf") return fromRtf(text);
    if (kind === "html") return clean(text);
    if (kind === "md") return fromMd(text);
    return fromText(text);
  }
  function save(kind = "docx", name) {
    const k = KINDS[kind] ? kind : "txt";
    const base = String(name || app.name || "Untitled").replace(/\.[a-z0-9]+$/i, "");
    app.download(`${base}.${KINDS[k].ext}`, write(k), KINDS[k].type);
    app.notify(`Saved ${base}.${KINDS[k].ext}`);
  }
  async function open(accept = Object.values(KINDS).map((k) => "." + k.ext).join(",")) {
    const file = await app.pickFile(accept);
    if (!file) return null;
    documentElement().innerHTML = await read(file);
    app.name = file.name.replace(/\.[a-z0-9]+$/i, "");
    app.changed(); app.saved = true;
    return file.name;
  }
  app.formats = { kinds: KINDS, document: documentElement, blocks, write, read, save, open, kindOf };
})(app);
