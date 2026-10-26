# Release lifecycle

<!-- reader-navigation:start -->
> **In plain English:** Reliability work detects failed or stalled components, preserves useful state and recovers through a controlled lifecycle.
>
> **Why it matters:** A user should not lose work because a slow component was mistaken for a dead one, or because a restart created another large process.
>
> **In everyday use:** If generation is progressing slowly, the system needs a different response from the one appropriate to a worker that has stopped making progress.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#reliability) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#reliability) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#runtime) · [Document map](DOCUMENTATION_INDEX.md)
> [Glossary](GLOSSARY.md) · [Formulas with worked examples](FORMULAS.md)
<!-- reader-navigation:end -->

Status: Guide · GitHub release listing checked 25 October 2026: zero releases.

The signing, notarization, SBOM and provenance workflow describes the release
process. A workflow's existence does not establish that it produced a signed
public release. `tools/check_release_ready.py` checks the pre-tag conditions;
published artifacts need their own provenance and verification. The release
listing check does not audit every historical Actions artifact.

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
| macOS 14+ on Apple silicon | reference desktop target | Declared runtime paths, subject to model, permissions, resources and individual capability qualification |
| macOS on Intel | unsupported | no MLX, so no resident model |
| Linux x86-64 in the container | limited headless/offline surface | Only paths whose dependencies are available; no Apple MLX resident model |
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

## What can be exercised without a tag

`make release-dry-run` runs everything in the release lane that does not need
an Apple account: the readiness gate, the pinned preflight, the SBOM and the
provenance record. It ends by naming the three steps it did not run —
signing, notarization, stapling — so a green dry run is never mistaken for a
release.

That is the honest half. It proves the lane is wired and its inputs exist. It
does not prove the artifact installs, because no artifact has been built from
it.

## What has not happened

- The GitHub Releases API reported zero published releases on 25 October.
- That listing does not establish whether signing was ever attempted, whether
  someone installed an independently built artifact, or whether external
  users depend on a checkout. Those require their own evidence.
- A supported upgrade between published stable releases is not established by
  the current release listing or the presence of a migration tool.

A workflow or local gate cannot substitute for a verified distributed artifact,
its install result and the declared support/upgrade evidence. Comparisons with
other projects need their actual release and support records.
