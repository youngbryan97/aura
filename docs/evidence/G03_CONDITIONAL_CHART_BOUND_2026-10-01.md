# Conditional Chart Bound, 2026-10-01

The selected-edge replay still timed out on the remaining development
request. Its [negative receipt](G03_SELECTED_EDGE_EXECUTION_2026-10-01.md)
is retained. Execution version five adds an admissible complete-chart bound
before learned scoring and constrained optimization. The thirty-second
request allowance, fitted tensors, candidate grammar and score policy stay
unchanged.

## Bound

For slot s, let b_si be each original option score and
L_s = log(sum_i exp(b_si)). The declared conditional learned update is

```
u_si = b_si + w*z_si - log(sum_j exp(b_sj + w*z_sj)) + L_s.
```

The normalized factor is at most one, so u_si <= L_s for every finite
learned score. This includes scores not yet computed. Each register's
definition attachment contributes at most its largest positive attachment
score, paid once. The existing optimizer enforces that single-definition
condition. Relaxing all graph, use and overlap constraints gives

```
U(chart) = sum_s L_s
         + sum_register max(0, max_definition attachment_score)
         - original_choice_log_normalizer
         + sum_operation operation_score
         - unchanged_length_penalty * operation_count.
```

The bridge creates the same context as before, with no positive pair bonus
or negative context cost. The bound is specific to that execution contract.
It is not valid for raw additive learned scores; raw replay does not prune.
No answer, target register or construction label appears in U.

A bound incumbent must have a complete accepted assignment. The bridge
reuses the parent's connected topological-order check and the same learned
register-use contract before retaining its joint score. It skips a chart
only if U plus a conservative floating-point guard is strictly below that
score. Equal and potentially better charts remain. Diagnostic receipts
record the upper bound, operation score, incumbent, guard and parent checks.
These records do not turn score margins into probabilities.

## Checks

Focused checks passed 103 tests in 23.57 seconds. The new cases cover
arbitrary learned logits and weights against the bound, optional positive
and negative definitions, strict losing charts, ties, better charts, raw
replay, connectivity rejection, register-use rejection and agreement of
the complete graph selector with unpruned execution. Existing fitted
pointer, offset normalization and native-evaluation checks remain green.

Standard gates passed: smoke 164 passed and one skipped in 56.80 seconds;
lint, compile, governance, layering and writing passed. No gate baseline or
request budget was relaxed. Training and checkpoint custody are unchanged.

The remaining source still needs a real replay. A fixture bound proof does
not establish useful pruning on real charts, language accuracy, broad gain,
general transfer, fusion or frontier performance. G03 remains open.

## Verified Canary And Paired Screen

Frozen `615f14b7a` completed the previously timed-out source
`004d8671268bab2c04320a3fcc1ee2876b130a552a5a7be0588dc4f9ac2ce26b`.
Its measured public decode took 16.012830 seconds under the unchanged
thirty-second allowance. The correct selected chart had a joint score of
71.508866. Each of the other fifteen complete charts had a certified upper
bound below that score, including its floating-point guard. All sixteen
charts were examined; only the first needed fitted scoring and optimization.
Both comparison arms returned the same correct program.

Canary root:
`~/.aura/rlc-evidence/semantic-grounded-joint-profile-partition-bound-v5-20261001`.

- Report: `e7fb26d562432ee96df4c7d68d01a10c0c826efa9821c7fb26e27cefbf2f0e5c`.
- Plan: `83d88b24e5996aab5cb71f95772b02f6f196eef37fdc2be4514b6e619bfb3a4c`.
- Independent verification: `9646808ba4a6440e620bf9fc0e743c44419ba0ae1eb9b7793f56c390be426cae`.
- Contained terminal: `70cefbc70b1f5e1f648fc67b571b3f5faf5bb77d71a745229b4ce7fa91eaa309`.

The supervisor returned two because the one-source profiler cannot authorize
advancement. It completed in 102.097094 seconds, with empty lineage and
process group; supervisor 19932, child 19935 and inhibitor 19938 exited.

The subsequent full paired screen used the same frozen checkout, tensors,
score policy and request allowance. Parent: 19 exact and 20 equivalent;
global chart: 20 exact and 21 equivalent; complete joint path: 20 exact and
21 equivalent. There were zero paired regressions, and every request returned
a program. This passes the existing development-screen advancement rule.
It does not isolate a neural gain: the single gain over the parent also
appeared in the global chart arm.

The joint path considered 307 charts and certified 281 as unable to beat an
accepted incumbent. Its total measured decode time was 48.583047 seconds,
and the longest request took 7.560412 seconds without the profiler. The
contained supervisor returned zero after 148.316713 seconds, with empty
lineage and process group. Supervisor 20414 and child 20418 exited.

Full-screen root:
`~/.aura/rlc-evidence/semantic-grounded-joint-development-partition-bound-v5-20261001`.

- Report: `38348a7992a88e01dabe0a49a1f023547ae30424c40d721fe8062d15cbad4b55`.
- Plan: `0bc40712b5343a925ec2237c0c00880be652b8ccc576f9d3732228ab9f06744c`.
- Independent verification: `99092bb704683b3d98b75042d0b4080b1a2170a389702ef3030bd7d814e56eb3`.
- Contained terminal: `b7c9928b878c51006a0bb51ffb5028c15f19d85fc1dbe4fb394d1cb92d3fbb3f`.

The report is 83,603,337 bytes. The original verifier entry point refused
its sixty-four-MiB file bound. Independent verification therefore used an
explicit 128-MiB stable reader, then the unchanged frozen execution hashes,
fit verifier and report checks. No semantic criterion or search allowance
changed. Future measurements use [complete row archives](G03_DEVELOPMENT_ROW_ARCHIVES_2026-10-01.md)
instead of retaining every edge in one growing root report. The original
report remains unchanged. The full source-development population and later
acceptance obligations remain unmeasured for this artifact; G03 stays open.
