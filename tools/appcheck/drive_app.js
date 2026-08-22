// Drive a built app through its own controls, in a real DOM.
//
// The compiler emits one file with its own state machine; this opens that
// file, fills the inputs it declares, clicks the buttons it declares, and
// reports what the page then shows. The runtime compares that against its own
// model of the same operations, so a page that renders nothing, throws on
// click, or drops a control is caught before anyone opens it.
//
// Usage: node drive_app.js <path-to-html> '<json>'
//   json: { "runs": [["action", ...], ...], "inputs": {"name": value} }
//   out : { "ok": true, "rendered": [ {field: text, ...}, ... ] }
const fs = require("fs");
const { JSDOM, VirtualConsole } = require("jsdom");

// "-" means the page arrives on stdin, so a check writes no file at all.
const file = process.argv[2];
const plan = JSON.parse(process.argv[3] || '{"runs":[],"inputs":{}}');
const html = fs.readFileSync(file === "-" ? 0 : file, "utf8");

const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on("jsdomError", (e) => errors.push(String((e && e.message) || e)));
virtualConsole.on("error", (e) => errors.push(String(e)));

function shown(doc) {
  const out = {};
  doc.querySelectorAll("[data-value]").forEach((node) => {
    out[node.dataset.value] = node.textContent;
  });
  doc.querySelectorAll("[data-list]").forEach((node) => {
    const items = [];
    node.querySelectorAll("li").forEach((li) => {
      if (li.className === "empty") return;
      const span = li.querySelector("span");
      items.push(span ? span.textContent : li.textContent);
    });
    out[node.dataset.list] = items;
  });
  return out;
}

const rendered = [];
try {
  for (const run of plan.runs) {
    const dom = new JSDOM(html, {
      runScripts: "dangerously",
      pretendToBeVisual: true,
      virtualConsole,
      url: "https://aura.local/app",
    });
    const doc = dom.window.document;
    for (const action of run) {
      doc.querySelectorAll("[data-input]").forEach((node) => {
        const value = plan.inputs[node.dataset.input];
        if (value !== undefined) node.value = String(value);
      });
      // A row action has no control of its own until the list renders one.
      const button =
        doc.querySelector(`[data-action="${action}"]`) ||
        doc.querySelector(`[data-row-action="${action}"][data-index]`);
      if (!button) {
        errors.push(`no control for action ${action}`);
        continue;
      }
      button.dispatchEvent(new dom.window.Event("click", { bubbles: true }));
    }
    rendered.push(shown(doc));
    dom.window.close();
  }
} catch (e) {
  errors.push(String((e && e.stack) || e));
}

console.log(JSON.stringify({ ok: errors.length === 0, errors: errors.slice(0, 6), rendered }));
