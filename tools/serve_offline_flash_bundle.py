#!/usr/bin/env python3
"""Serve a verified offline bundle on an ephemeral loopback port for a bounded run."""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def check_bundle(directory: Path, *, case_ids: set[int] | None = None) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("cases"), list):
        raise ValueError("unsupported bundle manifest")
    seen = set()
    for case in manifest["cases"]:
        number = case.get("id")
        if type(number) is not int or number < 0 or number in seen:
            raise ValueError("duplicate or invalid case identity")
        seen.add(number)
        if case.get("status") != "asset_verified":
            continue
        if case_ids is not None and number not in case_ids:
            continue
        actual = hashlib.sha256((directory / "programs" / f"{number}.swf").read_bytes()).hexdigest()
        if actual != case["program"]["sha256"]:
            raise ValueError(f"program {number} differs from its retained original")
        for name, expected in (case.get("dependencies") or {}).items():
            folder = (directory / "games" / str(number)).resolve()
            source = (folder / name).resolve()
            if not source.is_relative_to(folder) or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError(f"a file of program {number} ({name}) differs from the one fetched")
    for name, expected in manifest["emulator"]["files"].items():
        source = (directory / "ruffle" / name).resolve()
        if not source.is_relative_to(directory.resolve()):
            raise ValueError("emulator path leaves the bundle")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"emulator member {name} differs from its pinned package")
    return manifest


class BundleHandler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm"}

    def send_head(self):
        root = Path(self.directory).resolve()
        if not Path(self.translate_path(self.path)).resolve().is_relative_to(root):
            self.send_error(404, "outside the retained bundle")
            return None
        return super().send_head()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=7200.)
    args = parser.parse_args()
    if not 0 < args.seconds <= 43200:
        raise ValueError("run duration must be positive and at most twelve hours")
    directory = args.bundle.resolve()
    manifest = check_bundle(directory)
    handler = functools.partial(BundleHandler, directory=str(directory))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        server.timeout = 1.
        print(json.dumps({"url": f"http://127.0.0.1:{server.server_port}/index.html",
                          "cases": len(manifest["cases"]), "duration_s": args.seconds}), flush=True)
        until = time.monotonic() + args.seconds
        while time.monotonic() < until:
            server.handle_request()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
