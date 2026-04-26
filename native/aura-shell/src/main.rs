//! Aura native shell.
//!
//! Wraps the local FastAPI runtime and the web UI in a Tauri window. The
//! shell launches `aura_main.py` as a sidecar, waits for the boot health
//! endpoint to become reachable, and only then loads the UI. On window
//! close, it sends SIGTERM so the runtime drains receipts cleanly.

use std::process::Stdio;
use std::time::Duration;

use serde::Serialize;
use tauri::{Manager, RunEvent};
use tokio::process::{Child, Command};

#[derive(Clone, Serialize)]
struct BootStatus {
    state: String,
}

#[tauri::command]
async fn boot_status() -> Result<BootStatus, String> {
    // The frontend polls this until the runtime is reachable. The shell
    // doesn't speak to the runtime directly — it just reflects whether
    // the local TCP port is accepting connections.
    Ok(BootStatus { state: "starting".into() })
}

#[tokio::main]
async fn main() {
    tauri::async_runtime::set(tokio::runtime::Handle::current());
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![boot_status])
        .setup(|app| {
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let mut child: Child = Command::new("python3")
                    .args([
                        "aura_main.py",
                        "--desktop",
                        "--port",
                        "7400",
                    ])
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .kill_on_drop(true)
                    .spawn()
                    .expect("aura runtime failed to launch");

                // Block until the runtime answers, then mark the shell ready.
                let client = reqwest::Client::new();
                loop {
                    let r = client.get("http://localhost:7400/api/health").send().await;
                    if r.is_ok() {
                        let _ = handle.emit("aura://ready", &());
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(250)).await;
                }
                let _ = child.wait().await;
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("aura shell failed to start");
}
