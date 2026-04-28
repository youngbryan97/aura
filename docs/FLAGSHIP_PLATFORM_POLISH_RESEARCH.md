# Aura Flagship Platform Polish Research

Date: 2026-04-28

This document captures what "Chrome-level", flagship-grade Aura polish should mean in practice. The goal is not to make Aura generic. The goal is to keep the original dark glass, violet/cyan, neural cockpit identity and raise the execution quality until it feels deliberate at every pixel and code path.

## Product Bar

Flagship polish is measurable:

- No horizontal overflow at common desktop, laptop, tablet, and phone sizes.
- No text clipped by fixed-width panels, pills, buttons, headers, or cards.
- Pointer targets meet WCAG 2.2 target-size expectations, with at least 24 by 24 CSS pixels or enough spacing.
- Interactive latency is treated as a product bug. Core Web Vitals now use INP for responsiveness, so Aura's UI should budget work around input, streaming, live telemetry, and neural-feed updates.
- The UI should adapt like a native Mac app: related controls are grouped, essential information gets space, and secondary controls use progressive disclosure instead of crowding the window.
- The existing Aura visual language stays: cosmic background, glass panels, neon violet/cyan accents, neural feed, orb language, and cockpit instrumentation.

## Research Anchors

- Chrome/Web Vitals: INP is the responsiveness metric to watch. Streaming, typing, telemetry, and neural-card updates should avoid long main-thread work. Source: https://web.dev/inp/
- W3C WCAG 2.2: target size minimum is a concrete accessibility baseline for controls. Source: https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html
- Apple HIG layout: adaptive layout, clear grouping, readable hierarchy, safe areas, and stable resizing are what make software feel native. Source: https://developer.apple.com/design/human-interface-guidelines/layout
- Material 3 theming: high-end product UI benefits from semantic tokens for color, type, shape, and state instead of scattered one-off values. Source: https://developer.android.com/codelabs/m3-design-theming
- Microsoft SDL: platform maturity requires security and privacy checks across requirements, design, implementation, verification, and release. Source: https://learn.microsoft.com/en-us/compliance/assurance/assurance-microsoft-security-development-lifecycle
- OpenSSF Scorecard: security posture should be continuously assessed with automated checks, not handled as ad hoc cleanup. Source: https://openssf.org/projects/scorecard/

## What This Takes

1. UI polish system
   - Keep the original shell as the default visual direction.
   - Create a strict responsive contract for desktop, laptop, tablet, and mobile.
   - Treat every fixed min-width, overflowing pill, and uncontrolled scroll container as a defect.
   - Move repeated colors, radii, shadows, spacing, and text sizes into tokens.
   - Add browser screenshot checks for layout overflow and target size.

2. Responsiveness and lag
   - Cap live-feed DOM growth.
   - Batch streaming and neural-feed updates.
   - Skip expensive typewriter rendering for long responses or reduced-motion users.
   - Track INP-style interaction latency locally for send, tab switch, typing, and live-feed bursts.

3. Codebase quality gates
   - Make `make setup` install the tools required by `make quality`.
   - Keep compile, lint, typecheck, governance lint, smoke, and focused runtime slices runnable.
   - Add small tests around every repaired fault so fixes do not decay.

4. Runtime platform maturity
   - Replace silent exception swallowing with structured degradation records where practical.
   - Keep background tasks owned by task trackers.
   - Maintain strict shutdown, actor, state, and event-bus contracts.
   - Add release-readiness gates around health, telemetry, crash loops, and fallback behavior.

5. Security and supply chain
   - Treat sandbox escapes, shell execution, unsafe deserialization, token leaks, and forceful self-modification as release blockers.
   - Add dependency and secret scanning to the quality path.
   - Move toward signed build provenance and reproducible release artifacts.

## Current First-Pass Findings

- Legacy mobile overflows horizontally because the chat panel keeps a desktop `450px` minimum width and the header action row does not collapse.
- Legacy desktop has small neural buttons and copy buttons below the WCAG target-size baseline.
- Long Aura responses re-render message HTML word by word, which can feel laggy as response length grows.
- React `/shell` is cleaner and faster, but the user prefers the original visual system, so it should be treated as a reference, not a replacement.
- The active `.venv` had `pytest` but not `ruff`, while `make quality` requires `ruff`. The setup path needs to install dev tooling consistently.

## Immediate Direction

- Keep `/` as the original Aura shell.
- Polish the existing shell first: header containment, mobile sizing, control targets, text wrapping, reduced-motion behavior, and long-response rendering.
- Then add a repeatable UI audit script and promote it into the quality gate once stable.
