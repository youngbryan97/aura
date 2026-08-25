/* schematics.js — the drawings panel.
 *
 * Reads what the runtime wrote and shows it. It computes nothing: every
 * number here came off the design's own model file, which is the same file
 * the drawings were made from. A figure the panel invented would be exactly
 * the failure the engine exists to prevent, one layer out.
 */
(function () {
  "use strict";

  var state = { designs: [], current: null, model: null, tab: null };

  var listEl = document.getElementById("list");
  var tabsEl = document.getElementById("tabs");
  var viewEl = document.getElementById("view");
  var verdictEl = document.getElementById("verdict");

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (key) {
      if (key === "class") node.className = attrs[key];
      else if (key === "text") node.textContent = attrs[key];
      else if (key === "html") node.innerHTML = attrs[key];
      else node.setAttribute(key, attrs[key]);
    });
    (children || []).forEach(function (child) {
      if (child) node.appendChild(child);
    });
    return node;
  }

  function get(path) {
    return fetch(path, { credentials: "same-origin" }).then(function (response) {
      if (!response.ok) throw new Error(response.status + " on " + path);
      return response.json();
    });
  }

  function quantity(value) {
    if (!value) return "-";
    return value.text || String(value.value);
  }

  // ── the list ───────────────────────────────────────────────────────────

  function renderList() {
    listEl.innerHTML = "";
    if (!state.designs.length) {
      listEl.appendChild(el("p", { class: "empty", text: "No designs yet." }));
      return;
    }
    state.designs.forEach(function (design) {
      var flagClass = "flag";
      if (design.ok === true) flagClass += " ok";
      else if (design.ok === false) flagClass += " bad";
      var button = el("button", {
        class: "design",
        "aria-current": String(design.id === state.current),
      }, [
        el("b", {}, [el("span", { class: flagClass }), document.createTextNode(design.name || design.id)]),
        el("span", { text: (design.parts || 0) + " parts · " + (design.findings || 0) + " results" }),
      ]);
      button.addEventListener("click", function () { open(design.id); });
      listEl.appendChild(button);
    });
  }

  // ── the tabs ───────────────────────────────────────────────────────────

  var CAPTIONS = {
    assembly: "The whole thing put together, labelled part by part.",
    exploded: "Pulled apart in the order it comes off, numbered to the parts list.",
    section: "Cut through, so the inside can be seen.",
    orthographic: "Squared-on views with dimensions, for making from.",
    schematic: "What connects to what, rather than what it looks like.",
  };

  function renderTabs() {
    tabsEl.innerHTML = "";
    if (!state.model) return;
    var kinds = (state.model.sheets || []).map(function (sheet) { return sheet.kind; });
    var tabs = kinds.concat(["how it works", "calculations", "parts", "building it", "files"]);
    tabs.forEach(function (name) {
      var button = el("button", {
        "aria-selected": String(name === state.tab),
        text: name.charAt(0).toUpperCase() + name.slice(1),
      });
      button.addEventListener("click", function () { show(name); });
      tabsEl.appendChild(button);
    });
  }

  // ── the panes ──────────────────────────────────────────────────────────

  function sheetPane(kind) {
    return el("div", {}, [
      el("p", { class: "note", text: CAPTIONS[kind] || "" }),
      el("div", { class: "sheet" }, [
        el("img", {
          src: "/api/engineering/designs/" + encodeURIComponent(state.current) +
               "/sheet/" + encodeURIComponent(kind),
          alt: kind + " drawing of " + (state.model.design.name || state.current),
        }),
      ]),
    ]);
  }

  function tiles() {
    var model = state.model;
    var byId = {};
    (model.findings || []).forEach(function (finding) { byId[finding.id] = finding; });
    var rows = [];
    var mass = byId["assurance.mass_growth"] || byId["mass.total"];
    if (mass) rows.push(["Mass", quantity(mass.value)]);
    if (byId["electrical.total_draw"]) rows.push(["Power", quantity(byId["electrical.total_draw"].value)]);
    if (byId["envelope.size"]) rows.push(["Overall size", quantity(byId["envelope.size"].value)]);
    rows.push(["Results computed", String((model.findings || []).length)]);
    var failed = (model.findings || []).filter(function (f) { return f.verdict === "fail"; }).length;
    rows.push(["Checks failed", String(failed)]);
    return el("div", { class: "tiles" }, rows.map(function (row) {
      return el("div", { class: "tile" }, [
        el("span", { text: row[0] }),
        el("b", { text: row[1] }),
      ]);
    }));
  }

  function howPane() {
    var model = state.model;
    return el("div", {}, [
      tiles(),
      el("p", { text: model.narrative || "No narrative was written." }),
      el("h3", { text: "Part by part" }),
      el("div", {}, (model.design.parts || []).map(function (part) {
        return el("p", {}, [
          el("b", { text: (part.lay_name || part.name) + ". " }),
          document.createTextNode(part.function || ""),
        ]);
      })),
    ]);
  }

  function calculationsPane() {
    var model = state.model;
    var verification = model.verification || {};
    var children = [
      el("p", { class: "note", text: verification.plain || "" }),
      el("p", { class: "note", text: "Checks run: " + (verification.checks_run || []).join(", ") }),
    ];
    (model.findings || []).forEach(function (finding) {
      var block = el("div", { class: "finding " + (finding.verdict || "") }, [
        el("div", {}, [
          el("b", { text: finding.name + " " }),
          el("span", { class: "value", text: quantity(finding.value) }),
        ]),
        el("div", { text: finding.plain || "" }),
        el("div", { class: "work", text: (finding.substituted || finding.formula || "") + "\n" + (finding.method || "") }),
      ]);
      if (finding.advice) {
        block.appendChild(el("div", { class: "advice", text: "To fix: " + finding.advice }));
      }
      children.push(block);
    });
    if ((verification.problems || []).length) {
      children.push(el("h3", { text: "Open items" }));
      children.push(el("table", {}, [
        el("tr", {}, [el("th", { text: "Severity" }), el("th", { text: "Where" }), el("th", { text: "What" })]),
      ].concat(verification.problems.map(function (problem) {
        return el("tr", {}, [
          el("td", { text: problem.severity }),
          el("td", { text: problem.subject }),
          el("td", { text: problem.message + " " + (problem.advice || "") }),
        ]);
      }))));
    }
    return el("div", {}, children);
  }

  function partsPane() {
    var parts = state.model.design.parts || [];
    return el("table", {}, [
      el("tr", {}, ["No", "Part", "Qty", "Material", "Mass", "How obtained"].map(function (head) {
        return el("th", { text: head });
      })),
    ].concat(parts.map(function (part) {
      return el("tr", {}, [
        el("td", { class: "num", text: String(part.balloon) }),
        el("td", {}, [
          el("b", { text: part.lay_name || part.name }),
          el("div", { class: "note", text: part.function || "" }),
        ]),
        el("td", { class: "num", text: String(part.quantity) }),
        el("td", { text: (part.material && part.material.name) || "-" }),
        el("td", { class: "num", text: quantity(part.mass) }),
        el("td", { text: (part.sourcing && (part.sourcing.specification || part.sourcing.method)) || "-" }),
      ]);
    })));
  }

  function buildPane() {
    var plan = state.model.build;
    if (!plan) return el("p", { class: "empty", text: "No build plan was produced." });
    var children = [el("p", { text: plan.plain || "" })];
    if ((plan.buy || []).length) {
      children.push(el("h3", { text: "To buy" }));
      children.push(el("table", {}, [
        el("tr", {}, ["Qty", "Item", "Specification", "Unit cost"].map(function (h) {
          return el("th", { text: h });
        })),
      ].concat(plan.buy.map(function (item) {
        return el("tr", {}, [
          el("td", { class: "num", text: String(item.quantity) }),
          el("td", { text: item.name }),
          el("td", { text: item.specification }),
          el("td", { class: "num", text: quantity(item.unit_cost) }),
        ]);
      }))));
    }
    if ((plan.make || []).length) {
      children.push(el("h3", { text: "To make" }));
      children.push(el("table", {}, [
        el("tr", {}, ["Qty", "Part", "Process", "Stock", "Tolerance"].map(function (h) {
          return el("th", { text: h });
        })),
      ].concat(plan.make.map(function (item) {
        return el("tr", {}, [
          el("td", { class: "num", text: String(item.quantity) }),
          el("td", { text: item.name }),
          el("td", { text: item.process }),
          el("td", { text: item.stock }),
          el("td", { class: "num", text: quantity(item.tolerance) }),
        ]);
      }))));
    }
    children.push(el("h3", { text: "Order of assembly" }));
    (plan.steps || []).forEach(function (step) {
      var block = el("div", { class: "step" }, [
        el("div", { class: "n", text: String(step.number) }),
        el("div", {}, [
          el("b", { text: step.action }),
          el("div", { text: step.detail }),
          step.check ? el("div", { class: "note", text: "Check: " + step.check }) : null,
          (step.tools || []).length
            ? el("div", { class: "note", text: "Tools: " + step.tools.join(", ") })
            : null,
        ]),
      ]);
      children.push(block);
    });
    return el("div", {}, children);
  }

  function filesPane() {
    var bundle = state.model.bundle || { files: [] };
    return el("div", {}, [
      el("p", { class: "note", text: "Everything written for this design." }),
      el("div", { class: "files" }, (bundle.files || []).map(function (file) {
        return el("a", {
          href: "/api/engineering/designs/" + encodeURIComponent(state.current) +
                "/file/" + encodeURIComponent(file.name),
          download: file.name,
        }, [
          el("b", { text: file.name }),
          el("span", { text: file.description || "" }),
        ]);
      })),
    ]);
  }

  var PANES = {
    "how it works": howPane,
    "calculations": calculationsPane,
    "parts": partsPane,
    "building it": buildPane,
    "files": filesPane,
  };

  function show(name) {
    state.tab = name;
    renderTabs();
    viewEl.innerHTML = "";
    var builder = PANES[name];
    viewEl.appendChild(builder ? builder() : sheetPane(name));
  }

  // ── loading ────────────────────────────────────────────────────────────

  function open(id) {
    state.current = id;
    renderList();
    get("/api/engineering/designs/" + encodeURIComponent(id)).then(function (model) {
      state.model = model;
      var verification = model.verification || {};
      verdictEl.textContent = verification.plain || "";
      verdictEl.className = "verdict " + (verification.ok === false ? "bad" : "good");
      var first = (model.sheets || [])[0];
      show(first ? first.kind : "how it works");
    }).catch(function (error) {
      viewEl.innerHTML = "";
      viewEl.appendChild(el("p", { class: "empty", text: String(error) }));
    });
  }

  function load() {
    get("/api/engineering/designs").then(function (payload) {
      state.designs = payload.designs || [];
      renderList();
      if (state.designs.length && !state.current) open(state.designs[0].id);
      else if (!state.designs.length) {
        get("/api/engineering/capability").then(function (capability) {
          verdictEl.textContent = capability.statement || "";
        }).catch(function () { /* the panel works without it */ });
      }
    }).catch(function (error) {
      listEl.innerHTML = "";
      listEl.appendChild(el("p", { class: "empty", text: String(error) }));
    });
  }

  load();
  // A design drawn while the panel is open should appear without a reload,
  // and a poll this slow costs nothing.
  setInterval(function () {
    get("/api/engineering/designs").then(function (payload) {
      var incoming = payload.designs || [];
      if (incoming.length !== state.designs.length) {
        state.designs = incoming;
        renderList();
      }
    }).catch(function () { /* keep what is on screen */ });
  }, 15000);
})();
