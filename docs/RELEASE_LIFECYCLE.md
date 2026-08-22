# Release lifecycle

Status: written 2026-08-21. The GitHub repository has published **no
releases**. The signing, notarization, SBOM and provenance machinery in
`.github/workflows/release.yml` is real and has never produced a public
artifact, which means the lifecycle below is a policy this repository now
holds itself to, not a history it can point at. `tools/check_release_ready.py`
enforces the parts that are checkable before a tag; the rest becomes true the
first time a tag is pushed.

## Versioning

Calendar versioning: `YYYY.M.D`, from `version` in `pyproject.toml`, which is
the one authority — `config/product_facts.json` records it and
`tools/check_product_facts.py` fails when another file disagrees.

A version is a point in the source, not a claim about quality. The channel
says what the quality claim is.

## Channels

| Channel | Tag | What it means | Gates that must be green |
| --- | --- | --- | --- |
| `dev` | none | every commit on `main` | the required checks in `config/branch_protection_policy.json` |
| `alpha` | `vYYYY.M.D-alpha.N` | runs, and the owner uses it | the above, plus `make quality` |
| `beta` | `vYYYY.M.D-beta.N` | someone other than the author has run it for a week | the above, plus a 4-hour soak with no leak regression |
| `stable` | `vYYYY.M.D` | signed, notarized, and supported until the next stable | the above, plus `make release-preflight` and `make certify` |

There is no LTS. One maintainer cannot support two lines, and saying otherwise
would be the kind of claim this repository has a gate against.

## Supported platforms

| Platform | Support | What runs |
| --- | --- | --- |
| macOS 14+ on Apple silicon | supported | everything, including the resident model |
| macOS on Intel | unsupported | no MLX, so no resident model |
| Linux x86-64 in the container | headless only | the API and the offline gates; no model runtime |
| Windows | unsupported | — |

The reason is in `config/product_facts.json`: the substrate is MLX, which is
Metal-only, and `requirements.txt` pins `mlx`, `mlx-lm` and `mlx-metal` with
no environment markers. A document that claims broader operation is
contradicted by the requirements file.

## Upgrade compatibility

Between two stable releases:

- **Schema.** Migrations run forward automatically and are verified against
  their recorded checksums before anything else touches the database
  (`core/db/migrations.py`). A database ahead of the binary is named as a
  downgrade and refused, not silently accepted.
- **Downgrade.** Not supported. Restore from a backup taken before the
  upgrade; `make backup` and `make restore` are the path, and
  `make restore-test` proves the path works.
- **Identity and memory.** Carried across. `make backup` before an upgrade is
  the documented step because the identity record is the thing worth keeping.
- **Configuration.** Settings migrate through the versioned envelope in
  `core/runtime/settings_schema.py`; unknown keys are preserved, never
  dropped.

## What a release must produce

Every stable tag produces, or the workflow fails:

1. a signed and notarized `Aura.app` with the hardened runtime and stapled
   ticket;
2. `artifacts/provenance/sbom.json` and `artifacts/provenance/provenance.json`;
3. dependencies installed from `requirements_lock.txt` with `--require-hashes`
   — no fallback, no `|| true`;
4. a CHANGELOG entry for the version;
5. the required checks green on the tagged commit.

## Security maintenance

Security fixes land on `main` and ship in the next stable. There is no
backport line, because there is no supported older line. A vulnerability in a
published artifact is grounds for an out-of-cycle stable tag.

Reporting: [SECURITY.md](../SECURITY.md).

## What has not happened

- No release has been published. Nothing here has been exercised end to end.
- No artifact has been installed by anyone other than the author.
- The upgrade path between two stable releases has never been run, because
  there has never been one stable release, let alone two.
