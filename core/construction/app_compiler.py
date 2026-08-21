"""Turn an :class:`AppSpec` into a single file that runs in a browser.

Every operation in the spec is emitted here, and only here. The Python
interpreter in :mod:`core.construction.app_model` and the JavaScript this writes come
from the same list of operations, so an app can be tested before it is opened
and the thing tested is the thing that ships.

The output is one file with no network dependency: no CDN, no font host, no
build step. It opens from disk.
"""

from __future__ import annotations

import json

from core.construction.app_model import Action, AppSpec, Control, Field, Op, View

__all__ = ["compile_app", "reducer_js"]

_STYLE = """
:root {
  color-scheme: light dark;
  --ink: #14171a; --paper: #fbfbfa; --line: #d8d6d1; --quiet: #6b6f76;
  --accent: #2f5d50;
}
@media (prefers-color-scheme: dark) {
  :root { --ink: #ecebe8; --paper: #16181a; --line: #33373b; --quiet: #9aa0a6;
          --accent: #7fbfa8; }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2.5rem 1.25rem; background: var(--paper); color: var(--ink);
  font: 16px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
  display: flex; justify-content: center;
}
main { width: 100%; max-width: 34rem; }
h1 { font-size: 1.4rem; font-weight: 600; letter-spacing: -0.01em; margin: 0 0 1.5rem; }
.panel { border: 1px solid var(--line); border-radius: 10px; padding: 1.1rem 1.15rem; margin-bottom: 1rem; }
.label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--quiet); margin-bottom: 0.35rem; }
.value { font-size: 2rem; font-variant-numeric: tabular-nums; font-weight: 600; }
.row { display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center; }
button {
  font: inherit; padding: 0.5rem 0.95rem; border-radius: 7px; cursor: pointer;
  border: 1px solid var(--line); background: transparent; color: var(--ink);
}
button:hover { border-color: var(--accent); color: var(--accent); }
input[type=text], input[type=number] {
  font: inherit; padding: 0.5rem 0.65rem; border-radius: 7px; flex: 1 1 10rem;
  border: 1px solid var(--line); background: transparent; color: var(--ink);
}
ul { list-style: none; margin: 0; padding: 0; }
li { display: flex; justify-content: space-between; gap: 0.75rem; padding: 0.5rem 0;
     border-bottom: 1px solid var(--line); }
li:last-child { border-bottom: 0; }
li button { padding: 0.15rem 0.5rem; font-size: 0.85rem; }
.empty { color: var(--quiet); font-style: italic; }
"""


def _js(value: object) -> str:
    return json.dumps(value)


def _op_js(op: Op) -> str:
    """One operation, in the same words as ``core.construction.app_model._apply_one``."""
    target = _js(op.target)
    read = f"read({_js(op.source)}, {_js(op.value)}, state, inputs)"
    body = {
        "set": f"state[{target}] = {read};",
        "add": f"state[{target}] = num(state[{target}]) + num({read});",
        "toggle": f"state[{target}] = !state[{target}];",
        "append": f"state[{target}] = (state[{target}] || []).concat([{read}]);",
        "remove": (
            f"{{ const items = (state[{target}] || []).slice();"
            f" const at = Math.trunc(num({read}));"
            f" if (at >= 0 && at < items.length) items.splice(at, 1);"
            f" state[{target}] = items; }}"
        ),
        "clear": (
            f"state[{target}] = Array.isArray(state[{target}]) ? []"
            f" : (typeof state[{target}] === 'number' ? 0 : '');"
        ),
        "count": f"state[{target}] = ({read} || []).length;",
        "sum": f"state[{target}] = ({read} || []).reduce((a, b) => a + num(b), 0);",
    }.get(op.op)
    if body is None:  # pragma: no cover - problems() rejects these first
        raise ValueError(f"cannot emit operation {op.op}")
    return "    " + body


def reducer_js(spec: AppSpec) -> str:
    """The state machine, as JavaScript. Pure: state in, new state out."""
    cases = []
    for action in spec.actions:
        lines = "\n".join(_op_js(op) for op in action.ops)
        cases.append(f"  if (action === {_js(action.name)}) {{\n{lines}\n    return state;\n  }}")
    return (
        "function num(v) {\n"
        "  if (typeof v === 'boolean') return v ? 1 : 0;\n"
        "  if (typeof v === 'number') return v;\n"
        "  const n = parseFloat(String(v).trim());\n"
        "  return Number.isFinite(n) ? n : 0;\n"
        "}\n"
        "function read(source, value, state, inputs) {\n"
        "  if (source) return (source in inputs) ? inputs[source] : state[source];\n"
        "  return value;\n"
        "}\n"
        "function reduce(previous, action, inputs) {\n"
        "  const state = Object.assign({}, previous);\n"
        "  inputs = inputs || {};\n"
        + "\n".join(cases)
        + "\n  return state;\n}\n"
    )


def _view_html(view: View, spec: AppSpec) -> str:
    label = view.label or (spec.field_named(view.field).label if spec.field_named(view.field) else "") or view.field
    if view.kind == "list":
        row = (
            f' data-row-action="{_escape(view.row_action)}"'
            f' data-row-label="{_escape(view.row_label)}"'
            if view.row_action
            else ""
        )
        return (
            f'<section class="panel"><div class="label">{_escape(label)}</div>'
            f'<ul data-list="{_escape(view.field)}"{row}></ul></section>'
        )
    return (
        f'<section class="panel"><div class="label">{_escape(label)}</div>'
        f'<div class="value" data-value="{_escape(view.field)}"></div></section>'
    )


def _control_html(control: Control) -> str:
    if control.kind == "button":
        return f'<button data-action="{_escape(control.action)}">{_escape(control.label)}</button>'
    if control.kind == "checkbox":
        return (
            f'<label class="row"><input type="checkbox" data-action="{_escape(control.action)}">'
            f"{_escape(control.label)}</label>"
        )
    kind = "number" if control.kind == "number_input" else "text"
    return (
        f'<input type="{kind}" data-input="{_escape(control.input_name)}" '
        f'placeholder="{_escape(control.label)}">'
    )


def _escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def compile_app(spec: AppSpec) -> str:
    """The whole app, as one file. Raises ValueError if the spec cannot compile."""
    problems = spec.problems()
    if problems:
        raise ValueError("; ".join(problems))
    views = "\n    ".join(_view_html(view, spec) for view in spec.views)
    controls = "\n      ".join(_control_html(control) for control in spec.controls)
    storage_key = "aura.app." + "".join(
        character if character.isalnum() else "_" for character in spec.title.lower()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(spec.title)}</title>
<style>{_STYLE}</style>
</head>
<body>
<main>
  <h1>{_escape(spec.title)}</h1>
    {views}
  <section class="panel"><div class="row">
      {controls}
  </div></section>
</main>
<script>
{reducer_js(spec)}
const START = {_js({item.name: item.start() for item in spec.fields})};
const KEY = {_js(storage_key)};
const PERSIST = {_js(bool(spec.persist))};
let state = Object.assign({{}}, START);
if (PERSIST) {{
  try {{
    const saved = JSON.parse(localStorage.getItem(KEY) || "null");
    if (saved && typeof saved === "object") state = Object.assign({{}}, START, saved);
  }} catch (e) {{ /* a corrupt save is not worth a broken page */ }}
}}
function inputs() {{
  const values = {{}};
  document.querySelectorAll("[data-input]").forEach(function (node) {{
    values[node.dataset.input] = node.type === "number" ? num(node.value) : node.value;
  }});
  return values;
}}
function render() {{
  document.querySelectorAll("[data-value]").forEach(function (node) {{
    const value = state[node.dataset.value];
    node.textContent = typeof value === "boolean" ? (value ? "yes" : "no")
      : (typeof value === "number" ? String(Math.round(value * 100) / 100) : String(value ?? ""));
  }});
  document.querySelectorAll("[data-list]").forEach(function (node) {{
    const items = state[node.dataset.list] || [];
    node.innerHTML = "";
    if (!items.length) {{
      const empty = document.createElement("li");
      empty.className = "empty";
      empty.textContent = "Nothing yet";
      node.appendChild(empty);
      return;
    }}
    items.forEach(function (item, index) {{
      const row = document.createElement("li");
      const text = document.createElement("span");
      text.textContent = typeof item === "object" ? JSON.stringify(item) : String(item);
      row.appendChild(text);
      const rowAction = node.dataset.rowAction;
      if (rowAction) {{
        const drop = document.createElement("button");
        drop.textContent = node.dataset.rowLabel || "Remove";
        drop.setAttribute("data-row-action", rowAction);
        drop.setAttribute("data-index", String(index));
        drop.addEventListener("click", function () {{ run(rowAction, {{ index: index }}); }});
        row.appendChild(drop);
      }}
      node.appendChild(row);
    }});
  }});
  if (PERSIST) {{
    try {{ localStorage.setItem(KEY, JSON.stringify(state)); }} catch (e) {{ }}
  }}
}}
function run(action, extra) {{
  const given = Object.assign(inputs(), extra || {{}});
  state = reduce(state, action, given);
  document.querySelectorAll("[data-input]").forEach(function (node) {{ node.value = ""; }});
  render();
}}
document.querySelectorAll("[data-action]").forEach(function (node) {{
  node.addEventListener("click", function () {{ run(node.dataset.action, {{}}); }});
}});
document.querySelectorAll("[data-input]").forEach(function (node) {{
  node.addEventListener("keydown", function (event) {{
    if (event.key !== "Enter") return;
    const button = document.querySelector("[data-action]");
    if (button) run(button.dataset.action, {{}});
  }});
}});
{"setInterval(function () { run('tick', {}); }, 1000);" if spec.ticking else ""}
render();
</script>
</body>
</html>
"""
