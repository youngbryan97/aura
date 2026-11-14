# Q11: install, update, uninstall and ordinary desktop launch — 13 November 2026

Each part was run on this Mac, on the commits named, between 19:25 and 20:06. Nothing here touched Bryan's own install
except the last part, which rebuilt `/Applications/Aura.app` from main and opened it, as he allowed ("you can install
whatever").

## Install from a clean checkout

`tools/clean_env_install_proof.py` at `f83c7b257`: `git archive` of HEAD (11,688 files), a new virtualenv,
`requirements/core.txt` and `requirements/dev.txt` from nothing, the runtime skeleton imported in that interpreter, and
test collection. PASSED in 71 s: install 13.4 s core, 21.3 s dev; imports 2.4 s; collection 7.4 s. The record is
`q11-clean-env-install-proof-2026-11-13.json`.

The app itself, from `scripts/bundle_app.sh` with `AURA_INSTALL_PATH` set to a sandbox: the native launcher compiled,
signed and installed in 81 s. `tools/app_bundle_freshness.py check` read it FRESH at `f83c7b257`;
`codesign --verify --deep --strict` passed.

## Update, and back

`tools/release_train.py update` in a throwaway worktree set to this morning's `c23e25c3e`: fetched origin, recorded the
rollback point, fast-forwarded to `f83c7b257`, compiled `core/` clean, in 17 s, with no stash because the tree was
clean. `tools/release_train.py rollback` then returned it to `c23e25c3e`. The worktree and its branch were removed.

## Uninstall

There was none. `tools/uninstall_aura.py` is it: the app and every `com.aura.*` launch agent, each booted out of launchd
first, moved to the Trash; her state under `~/.aura` kept unless `--purge-state`; refused while anything answers on her
port. On the real sandbox bundle above, with a launch agent beside it and a stand-in `launchctl`:

- `--dry-run` listed exactly the app and the agent, and her state as kept;
- with her running on 8000 it refused, exit 2, and moved nothing;
- run for real, it booted the agent out, moved both to the Trash, and left her state's marker file in place.

`tests/test_uninstall_aura.py` holds the same five behaviours. It was not run on her real install, by design: it
removes her app.

## Ordinary desktop launch

`tools/app_bundle_freshness.py ensure` rebuilt `/Applications/Aura.app` from `b0418b037` (it had been built from
`678db53ac` on 8 November); FRESH, signature verified. Her runtime, started earlier by `launch_aura.sh`, took SIGTERM
and exited clean in 14 s, port 8000 free. `open /Applications/Aura.app` at 20:03:48: `aura-launcher` started her
runtime, `/api/health` read ok and healthy 63 s later, her native window "Aura" showed her chat, and a typed turn at
20:05 was answered in 60 s.

## Found along the way

`scripts/install_to_applications.py` writes a light wrapper that runs the system `python3` (Homebrew 3.14), not her
runtime's 3.12. It is not the install path anything uses (`make app` and `tools/app_bundle_freshness.py` build through
`scripts/bundle_app.sh`), and is left as it is.
