/**
 * first_run.js — Aura first-run wizard.
 *
 * Steps 1..10 from the polish spec:
 *   1 Welcome / 2 System check / 3 Model / 4 Memory location /
 *   5 Permissions / 6 Safety / 7 Fallback / 8 Test voice /
 *   9 Test chat / 10 Ready
 *
 * The wizard reads and writes settings via /api/settings; world-channel
 * permissions go through /api/dashboard/world (read) and the per-channel
 * grant endpoints. A successful run leaves a file at
 *   ~/.aura/data/settings/first_run_completed
 * so the wizard never reappears.
 */

(() => {
  const STEPS = [
    "welcome", "system_check", "model", "memory_location", "permissions",
    "safety", "fallback", "test_voice", "test_chat", "ready",
  ];
  let step = 0;
  const state = { settings: {}, permissions: {} };

  const panel = document.getElementById("panel");
  const back = document.getElementById("back");
  const next = document.getElementById("next");
  const rail = document.querySelectorAll(".aura-wizard__rail li");

  function paintRail() {
    rail.forEach((li, i) => li.classList.toggle("active", i === step));
  }

  async function fetchSettings() {
    try {
      const r = await fetch("/api/settings");
      const d = await r.json();
      state.settings = d.values || {};
    } catch {}
  }

  async function patchSettings(patch) {
    try {
      const r = await fetch("/api/settings", {
        method: "PATCH",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(patch),
      });
      const d = await r.json();
      state.settings = d.values || state.settings;
    } catch {}
  }

  function render() {
    paintRail();
    const id = STEPS[step];
    panel.innerHTML = "";
    switch (id) {
      case "welcome":
        panel.innerHTML = `
          <h2>Welcome to Aura</h2>
          <p>This wizard will get you to your first conversation in under five minutes.
          Nothing leaves your machine unless you grant a specific permission.</p>`;
        break;
      case "system_check": {
        panel.innerHTML = `<h2>System check</h2><p>Looking at CPU, RAM, disk, and model availability…</p><pre id="sysout">checking…</pre>`;
        fetch("/api/dashboard/snapshot").then(r => r.json()).then(d => {
          document.getElementById("sysout").textContent = JSON.stringify({
            cpu_pct: d.system?.cpu_pct, ram_pct: d.system?.ram_pct,
            disk_pct: d.system?.disk_pct, viability: d.viability?.state,
          }, null, 2);
        }).catch(() => {});
        break;
      }
      case "model": {
        const cur = state.settings["model.local_path"] || "";
        panel.innerHTML = `
          <h2>Local model</h2>
          <label>Path on disk to the local model directory.</label>
          <input id="model_path" placeholder="/Users/you/Models/Qwen2.5-32B-Instruct-8bit" value="${cur}" />`;
        break;
      }
      case "memory_location":
        panel.innerHTML = `
          <h2>Memory location</h2>
          <p>Aura stores autobiographical memory and scars under <code>~/.aura/data</code>.
          You can override the location later in Settings → Memory.</p>`;
        break;
      case "permissions":
        panel.innerHTML = `
          <h2>Permissions</h2>
          <p>Grant only what you want Aura to use. You can flip these at any time.</p>
          <div><input type="checkbox" id="p_mic" /> <label for="p_mic">Microphone</label></div>
          <div><input type="checkbox" id="p_cam" /> <label for="p_cam">Camera</label></div>
          <div><input type="checkbox" id="p_screen" /> <label for="p_screen">Screen perception</label></div>
          <div><input type="checkbox" id="p_files" checked /> <label for="p_files">Workspace files</label></div>`;
        break;
      case "safety":
        panel.innerHTML = `
          <h2>Safety</h2>
          <label>Privacy mode</label>
          <select id="privacy_mode">
            <option value="standard">standard</option>
            <option value="private">private</option>
            <option value="isolated">isolated</option>
          </select>
          <div><input type="checkbox" id="safe_mode" /> <label for="safe_mode">Safe mode (block destructive primitives)</label></div>`;
        break;
      case "fallback":
        panel.innerHTML = `
          <h2>Fallback</h2>
          <p>If your local cortex is unavailable, should Aura route to a configured cloud provider?</p>
          <div><input type="checkbox" id="cloud_fallback" /> <label for="cloud_fallback">Allow cloud fallback</label></div>`;
        break;
      case "test_voice":
        panel.innerHTML = `
          <h2>Test voice</h2>
          <p>Click the button. You should hear a short hello.</p>
          <button type="button" id="say_hi">Say hi</button>`;
        document.getElementById("say_hi").onclick = () => {
          const u = new SpeechSynthesisUtterance("Hi. I'm Aura.");
          window.speechSynthesis?.speak?.(u);
        };
        break;
      case "test_chat":
        panel.innerHTML = `
          <h2>Test chat</h2>
          <p>Type a quick hello to make sure the chat path is wired up.</p>
          <input id="hello_msg" placeholder="hi" />
          <button id="hello_send" type="button">Send</button>
          <pre id="hello_out"></pre>`;
        document.getElementById("hello_send").onclick = async () => {
          try {
            const r = await fetch("/api/chat", {
              method: "POST",
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify({ message: document.getElementById("hello_msg").value || "hi" }),
            });
            const d = await r.json();
            document.getElementById("hello_out").textContent = (d.response || JSON.stringify(d)).slice(0, 400);
          } catch (e) { document.getElementById("hello_out").textContent = String(e); }
        };
        break;
      case "ready":
        panel.innerHTML = `<h2>Aura is ready</h2><p>You can start talking to her now. Settings are at <code>/settings</code>; the dashboard is at <code>/dashboard</code>.</p>`;
        break;
    }
  }

  async function commitStep() {
    const id = STEPS[step];
    if (id === "model") {
      const v = document.getElementById("model_path").value;
      if (v) await patchSettings({ "model.local_path": v });
    } else if (id === "permissions") {
      // Permission grants flow through /api/dashboard/world or settings.
      const get = id => document.getElementById(id).checked;
      await patchSettings({
        "permissions.camera": get("p_cam"),
        "permissions.screen": get("p_screen"),
        "permissions.files_workspace": get("p_files"),
      });
    } else if (id === "safety") {
      await patchSettings({
        "privacy.mode": document.getElementById("privacy_mode").value,
        "safety.safe_mode": document.getElementById("safe_mode").checked,
      });
    } else if (id === "fallback") {
      await patchSettings({ "model.cloud_fallback_enabled": document.getElementById("cloud_fallback").checked });
    } else if (id === "ready") {
      try { await fetch("/api/settings/auth/fresh", {method: "POST"}); } catch {}
    }
  }

  back.addEventListener("click", () => {
    step = Math.max(0, step - 1);
    render();
  });
  next.addEventListener("click", async () => {
    await commitStep();
    if (step < STEPS.length - 1) { step += 1; render(); }
  });

  fetchSettings().then(render);
})();
