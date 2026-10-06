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
// app.formats.write(kind, el)   a Blob of the document as kind: "docx" | "odt" | "rtf" | "html" | "md" | "txt" | "pdf"
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
    pdf: { ext: "pdf", label: "PDF (.pdf)", type: "application/pdf", writeOnly: true },
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
  function highlight(el) {
    for (let e = el; e && e !== documentElement() && e !== document.body; e = e.parentElement) {
      if (/^(P|LI|H[1-6]|TD|TH|DIV|BLOCKQUOTE)$/.test(e.nodeName)) return "";
      const h = hex(getComputedStyle(e).backgroundColor);
      if (h && h !== "FFFFFF") return h;
    }
    return "";
  }
  function paragraphOf(el) {
    const cs = getComputedStyle(el), size = parseFloat(cs.fontSize) || 16, lh = parseFloat(cs.lineHeight);
    return { indent: Math.round((parseFloat(cs.marginLeft) || 0) * 0.75), after: Math.round((parseFloat(cs.marginBottom) || 0) * 0.75 * 2) / 2,
      before: Math.round((parseFloat(cs.marginTop) || 0) * 0.75 * 2) / 2, lineHeight: isFinite(lh) ? Math.round((lh / size) * 100) / 100 : 0 };
  }
  function runStyle(el, base) {
    const cs = getComputedStyle(el), deco = cs.textDecorationLine || cs.textDecoration || "";
    const size = parseFloat(cs.fontSize) * 0.75;
    return {
      b: Number(cs.fontWeight) >= 600 || cs.fontWeight === "bold", i: cs.fontStyle === "italic",
      u: /underline/.test(deco), s: /line-through/.test(deco), color: hex(cs.color),
      size: Math.abs(size - base) > 0.4 ? Math.round(size * 2) / 2 : 0,
      font: (cs.fontFamily || "").split(",")[0].replace(/["']/g, "").trim(), href: el.closest("a") ? el.closest("a").href : "",
      bg: highlight(el),
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
      } else if (node.nodeName === "IMG") {
        current.runs.push({ img: node, text: "", width: node.getBoundingClientRect().width * 0.75, height: node.getBoundingClientRect().height * 0.75 });
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
      if (node.dataset && node.dataset.break === "page") { start("pagebreak", node); current = null; return; }
      if (tag === "TABLE") {
        const rows = [...node.querySelectorAll(":scope > tr, :scope > * > tr")].map((tr) => [...tr.children].map((td) => blocks(td)));
        start("table", node, { rows }); current = null; return;
      }
      const type = /^H[1-6]$/.test(tag) ? tag.toLowerCase() : tag === "LI" ? "li" : "p";
      start(type, node, { ...paragraphOf(node), ...(type === "li" ? { list: list ? list.kind : "ul", level: list ? list.level : 0 } : {}) });
      for (const child of node.childNodes) {
        if (child.nodeType === 1 && BLOCK.test(child.nodeName)) { walk(child, node, list); start("p", node); }
        else walk(child, node, list);
      }
      current = null;
    };
    for (const child of root.childNodes) walk(child, root);
    return out.filter((b) => b.type === "hr" || b.type === "table" || b.type === "pagebreak" || b.runs.some((r) => r.text.trim() || r.br || r.img) || b.type !== "p" || out.length === 1);
  }

  // ── writing ─────────────────────────────────────────────────────────
  function docxRun(r) {
    if (r.br) return "<w:r><w:br/></w:r>";
    const p = [r.b && "<w:b/>", r.i && "<w:i/>", r.u && '<w:u w:val="single"/>', r.s && "<w:strike/>", r.color && `<w:color w:val="${r.color}"/>`,
      r.size && `<w:sz w:val="${Math.round(r.size * 2)}"/>`, r.font && `<w:rFonts w:ascii="${esc(r.font)}" w:hAnsi="${esc(r.font)}"/>`].filter(Boolean).join("");
    return `<w:r>${p ? `<w:rPr>${p}</w:rPr>` : ""}<w:t xml:space="preserve">${esc(r.text)}</w:t></w:r>`;
  }
  function docxBlock(b) {
    if (b.type === "pagebreak") return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>';
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
        `<w:sectPr><w:pgSz w:w="${Math.round(paper().w * 20)}" w:h="${Math.round(paper().h * 20)}"${paper().w > paper().h ? ' w:orient="landscape"' : ""}/>` +
        `<w:pgMar w:top="${Math.round(paper().m * 20)}" w:right="${Math.round(paper().m * 20)}" w:bottom="${Math.round(paper().m * 20)}" w:left="${Math.round(paper().m * 20)}" w:header="720" w:footer="720" w:gutter="0"/></w:sectPr></w:body></w:document>`,
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

  // The page: its size and margins in points, as the program's page setup has them, else US Letter with an inch.
  function paper() {
    const ps = app.pageSetup && app.pageSetup.get ? app.pageSetup.get() : null;
    const [w, h] = ps && app.pageSetup.inches ? app.pageSetup.inches() : [8.5, 11];
    return { w: w * 72, h: h * 72, m: (ps ? Number(ps.margin) || 1 : 1) * 72 };
  }

  // ── PDF ─────────────────────────────────────────────────────────────
  // Laid out here into pages: each paragraph's lines broken where its words would break, in the faces
  // every PDF reader has (Helvetica, Times, Courier, each in bold and italic), every run in its own
  // size and colour, underlined, struck through or highlighted as it is; headings, lists, indents,
  // line spacing, alignment (justified lines spread to both edges), tables, pictures and page breaks.
  const FACES = { sans: ["Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique"],
    serif: ["Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic"], mono: ["Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique"] };
  const SHOWN_AS = { sans: "Helvetica, Arial, sans-serif", serif: "Times, 'Times New Roman', serif", mono: "Courier, 'Courier New', monospace" };
  const faceOf = (font) => { const f = String(font || "").toLowerCase();
    return /mono|courier|menlo|consol|code/.test(f) ? "mono" : /times|georgia|garamond|cambria|palatino|book|minion|serif/.test(f) && !/sans/.test(f) ? "serif" : "sans"; };
  const WIN = { 0x20ac: 0x80, 0x201a: 0x82, 0x0192: 0x83, 0x201e: 0x84, 0x2026: 0x85, 0x2020: 0x86, 0x2021: 0x87, 0x02c6: 0x88, 0x2030: 0x89, 0x0160: 0x8a,
    0x2039: 0x8b, 0x0152: 0x8c, 0x017d: 0x8e, 0x2018: 0x91, 0x2019: 0x92, 0x201c: 0x93, 0x201d: 0x94, 0x2022: 0x95, 0x2013: 0x96, 0x2014: 0x97, 0x02dc: 0x98,
    0x2122: 0x99, 0x0161: 0x9a, 0x203a: 0x9b, 0x0153: 0x9c, 0x017e: 0x9e, 0x0178: 0x9f };
  const plainSpaces = (t) => String(t).replace(/ /g, "    ").replace(/[    ]/g, " ").replace(/[​‌‍﻿]/g, "");
  function pdfText(t) {
    let out = "(";
    for (const ch of plainSpaces(t)) {
      const c = ch.codePointAt(0);
      const b = c < 128 ? c : c >= 0xa0 && c <= 0xff ? c : WIN[c] || 63;
      if (b === 40 || b === 41 || b === 92) out += "\\" + String.fromCharCode(b);
      else if (b < 32 || b > 126) out += "\\" + b.toString(8).padStart(3, "0");
      else out += String.fromCharCode(b);
    }
    return out + ")";
  }
  const ruler = document.createElement("canvas").getContext("2d");
  function measure(text, face, b, i, size) { ruler.font = `${i ? "italic " : ""}${b ? "bold " : ""}${size}px ${SHOWN_AS[face]}`; return ruler.measureText(plainSpaces(text)).width; }
  const num = (n) => (Math.round(n * 100) / 100).toString();
  const rgb = (h) => [0, 2, 4].map((k) => num(parseInt(h.slice(k, k + 2), 16) / 255)).join(" ");
  function pdf(bs) {
    const { w: W, h: H, m: M } = paper();
    const root = documentElement(), rootStyle = getComputedStyle(root);
    const baseSize = Math.round(parseFloat(rootStyle.fontSize) * 0.75 * 2) / 2 || 12, baseFace = faceOf(rootStyle.fontFamily);
    const fonts = [], images = [], pages = [];
    let ops = [], y = 0;
    const fontId = (face, b, i) => { const name = FACES[face][(b ? 1 : 0) + (i ? 2 : 0)]; let k = fonts.indexOf(name); if (k < 0) { fonts.push(name); k = fonts.length - 1; } return `F${k + 1}`; };
    const newPage = () => { ops = []; pages.push(ops); y = H - M; };
    const room = (tall) => { if (y - tall < M && y < H - M - 0.5) newPage(); };
    newPage();
    function words(runs) {
      // A run cut into words and the spaces between, each with its style and width.
      const out = [];
      for (const r of runs) {
        if (r.br) { out.push({ br: true }); continue; }
        if (r.img) { out.push({ img: r.img, width: r.width, height: r.height }); continue; }
        const face = r.font ? faceOf(r.font) : baseFace, size = r.size || baseSize;
        for (const piece of plainSpaces(r.text).split(/(\s+)/)) {
          if (!piece) continue;
          out.push({ text: piece, space: /^\s+$/.test(piece), face, b: r.b, i: r.i, u: r.u, s: r.s, color: r.color, bg: r.bg, size, width: measure(piece, face, r.b, r.i, size) });
        }
      }
      return out;
    }
    function lines(pieces, width) {
      const out = []; let line = [], used = 0;
      const end = () => { while (line.length && line[line.length - 1].space) used -= line.pop().width; out.push({ pieces: line, width: used }); line = []; used = 0; };
      for (let p of pieces) {
        if (p.br) { end(); out[out.length - 1].forced = true; continue; }
        if (p.space && !line.length) continue;
        if (used + p.width > width && line.length && !p.space) end();
        if (p.width > width && !p.space && !p.img) {
          // One word wider than the line: broken by letters.
          let part = "";
          for (const ch of p.text) { const next = part + ch; if (measure(next, p.face, p.b, p.i, p.size) > width && part) { line.push({ ...p, text: part, width: measure(part, p.face, p.b, p.i, p.size) }); end(); part = ch; } else part = next; }
          p = { ...p, text: part, width: measure(part, p.face, p.b, p.i, p.size) };
        }
        line.push(p); used += p.width;
      }
      if (line.length || !out.length) end();
      return out;
    }
    function draw(line, x0, width, align, last, spacing) {
      const tall = Math.max(...line.pieces.map((p) => p.img ? p.height : p.size), baseSize);
      const lead = line.pieces.some((p) => p.img) ? tall + 4 : tall * spacing;
      room(lead);
      const baseline = y - Math.max(...line.pieces.map((p) => (p.img ? p.height : p.size * 0.8)), baseSize * 0.8);
      const gaps = line.pieces.filter((p) => p.space).length;
      const spread = align === "justify" && !last && !line.forced && gaps ? (width - line.width) / gaps : 0;
      let x = x0 + (align === "center" ? (width - line.width) / 2 : align === "right" ? width - line.width : 0);
      for (const p of line.pieces) {
        if (p.img) {
          try {
            const canvas = document.createElement("canvas");
            canvas.width = Math.max(1, Math.round(p.width * 2)); canvas.height = Math.max(1, Math.round(p.height * 2));
            const ctx = canvas.getContext("2d"); ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, canvas.width, canvas.height); ctx.drawImage(p.img, 0, 0, canvas.width, canvas.height);
            const bytes = atob(canvas.toDataURL("image/jpeg", 0.9).split(",")[1]);
            let hexed = ""; for (let k = 0; k < bytes.length; k++) hexed += bytes.charCodeAt(k).toString(16).padStart(2, "0");
            images.push({ w: canvas.width, h: canvas.height, data: hexed + ">" });
            ops.push(`q ${num(p.width)} 0 0 ${num(p.height)} ${num(x)} ${num(baseline)} cm /Im${images.length} Do Q`);
          } catch (e) { /* a picture from elsewhere that the page may not copy is left out */ }
          x += p.width; continue;
        }
        if (p.bg) ops.push(`${rgb(p.bg)} rg ${num(x)} ${num(baseline - p.size * 0.22)} ${num(p.width + (p.space ? spread : 0))} ${num(p.size * 1.1)} re f`);
        if (!p.space) {
          ops.push(`BT /${fontId(p.face, p.b, p.i)} ${num(p.size)} Tf ${p.color ? rgb(p.color) : "0 0 0"} rg 1 0 0 1 ${num(x)} ${num(baseline)} Tm ${pdfText(p.text)} Tj ET`);
          const lineAt = (dy) => ops.push(`${p.color ? rgb(p.color) : "0 0 0"} RG ${num(Math.max(0.5, p.size / 16))} w ${num(x)} ${num(baseline + dy)} m ${num(x + p.width)} ${num(baseline + dy)} l S`);
          if (p.u) lineAt(-p.size * 0.12);
          if (p.s) lineAt(p.size * 0.3);
        }
        x += p.width + (p.space ? spread : 0);
      }
      y -= lead;
    }
    function paragraph(b, x0, width) {
      const heading = /^h[1-6]$/.test(b.type);
      const before = b.before || (heading ? 10 : 0);
      if (before && y < H - M - 0.5) y -= before;
      const left = (b.indent || 0) + (b.type === "li" ? 18 * ((b.level || 0) + 1) : 0);
      const all = lines(words(b.runs), width - left);
      const spacing = b.lineHeight || 1.15;
      all.forEach((line, n) => {
        if (n === 0 && b.type === "li") {
          room(Math.max(...line.pieces.map((p) => p.size || baseSize), baseSize) * spacing);
          const size = line.pieces[0] && line.pieces[0].size || baseSize;
          ops.push(`BT /${fontId(baseFace, false, false)} ${num(size)} Tf 0 0 0 rg 1 0 0 1 ${num(x0 + left - 14)} ${num(y - size * 0.8)} Tm ${pdfText(b.marker)} Tj ET`);
        }
        draw(line, x0 + left, width - left, b.align || "left", n === all.length - 1, spacing);
      });
      y -= b.after !== undefined && b.after !== 0 ? b.after : heading ? 4 : b.type === "li" ? 2 : 8;
    }
    function table(b, x0, width) {
      const cols = Math.max(1, ...b.rows.map((r) => r.length)), cw = width / cols, pad = 4;
      for (const row of b.rows) {
        const cells = row.map((cell) => cell.flatMap((cb) => lines(words(cb.runs), cw - 2 * pad).map((l) => ({ line: l, align: cb.align }))));
        const tall = Math.max(...cells.map((ls) => ls.reduce((n, l) => n + Math.max(...l.line.pieces.map((p) => p.size || baseSize), baseSize) * 1.2, 0)), baseSize * 1.2) + 2 * pad;
        room(tall);
        const top = y;
        cells.forEach((ls, c) => {
          ops.push(`0.6 0.62 0.65 RG 0.6 w ${num(x0 + c * cw)} ${num(top - tall)} ${num(cw)} ${num(tall)} re S`);
          y = top - pad;
          for (const l of ls) draw(l.line, x0 + c * cw + pad, cw - 2 * pad, l.align || "left", true, 1.2);
        });
        y = top - tall;
      }
      y -= 8;
    }
    const counters = [];
    for (let b of bs) {
      if (b.type === "pagebreak") { newPage(); continue; }
      if (b.type === "hr") { room(14); y -= 6; ops.push(`0.75 0.77 0.8 RG 0.75 w ${num(M)} ${num(y)} m ${num(W - M)} ${num(y)} l S`); y -= 8; continue; }
      if (b.type === "table") { table(b, M, W - 2 * M); continue; }
      if (b.type === "li") {
        const level = b.level || 0;
        counters.length = level + 1;
        counters[level] = b.list === "ol" ? (counters[level] || 0) + 1 : 0;
        b = { ...b, marker: b.list === "ol" ? `${counters[level]}.` : "•" };
      } else counters.length = 0;
      paragraph(b, M, W - 2 * M);
    }
    // The file: fonts, pictures, then each page and what is drawn on it.
    const objects = [];
    const add = (body) => { objects.push(body); return objects.length; };
    const catalog = add(""), tree = add("");
    const fontRefs = fonts.map((name) => add(`<< /Type /Font /Subtype /Type1 /BaseFont /${name} /Encoding /WinAnsiEncoding >>`));
    const imageRefs = images.map((im) => add(`<< /Type /XObject /Subtype /Image /Width ${im.w} /Height ${im.h} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter [/ASCIIHexDecode /DCTDecode] /Length ${im.data.length} >>\nstream\n${im.data}\nendstream`));
    const resources = `<< /Font << ${fontRefs.map((r, k) => `/F${k + 1} ${r} 0 R`).join(" ")} >>${imageRefs.length ? ` /XObject << ${imageRefs.map((r, k) => `/Im${k + 1} ${r} 0 R`).join(" ")} >>` : ""} >>`;
    const kids = pages.map((page) => {
      const content = page.join("\n");
      const stream = add(`<< /Length ${content.length} >>\nstream\n${content}\nendstream`);
      return add(`<< /Type /Page /Parent ${tree} 0 R /MediaBox [0 0 ${num(W)} ${num(H)}] /Resources ${resources} /Contents ${stream} 0 R >>`);
    });
    objects[catalog - 1] = `<< /Type /Catalog /Pages ${tree} 0 R >>`;
    objects[tree - 1] = `<< /Type /Pages /Kids [${kids.map((k) => `${k} 0 R`).join(" ")}] /Count ${kids.length} >>`;
    const when = new Date(), two = (n) => String(n).padStart(2, "0");
    const info = add(`<< /Title ${pdfText(app.name || "Untitled")} /Producer ${pdfText(document.title || "")} /CreationDate (D:${when.getFullYear()}${two(when.getMonth() + 1)}${two(when.getDate())}${two(when.getHours())}${two(when.getMinutes())}${two(when.getSeconds())}) >>`);
    let out = "%PDF-1.4\n";
    const offsets = objects.map((body, k) => { const at = out.length; out += `${k + 1} 0 obj\n${body}\nendobj\n`; return at; });
    const xref = out.length;
    out += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n${offsets.map((o) => `${String(o).padStart(10, "0")} 00000 n \n`).join("")}`;
    out += `trailer\n<< /Size ${objects.length + 1} /Root ${catalog} 0 R /Info ${info} 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
    return new Blob([out], { type: "application/pdf" });
  }
  function write(kind, el) {
    const bs = blocks(el || documentElement());
    if (kind === "docx") return docx(bs);
    if (kind === "odt") return odt(bs);
    if (kind === "rtf") return rtf(bs);
    if (kind === "pdf") return pdf(bs);
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
    if (KINDS[kind] && KINDS[kind].writeOnly) throw new Error(`a ${KINDS[kind].label} file is written here, not opened`);
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
  async function open(accept = Object.values(KINDS).filter((k) => !k.writeOnly).map((k) => "." + k.ext).join(",")) {
    const file = await app.pickFile(accept);
    if (!file) return null;
    documentElement().innerHTML = await read(file);
    app.name = file.name.replace(/\.[a-z0-9]+$/i, "");
    app.changed(); app.saved = true;
    return file.name;
  }
  app.formats = { kinds: KINDS, document: documentElement, blocks, write, read, save, open, kindOf };
})(app);
