# Flash actions retain their branch addresses

The statement reader prints relative `goto` distances, then discards the
instruction addresses that give those distances meaning. It also writes an
anonymous function body separately from the member assignment that owns it.
Those statements cannot prove which signed movement update a parameter selects.

Two new pure readers preserve that evidence:

- `core/perception/flash_actions_with_source.py` retains immutable instructions,
  encoded operands, byte ranges, exact branch destinations and successor edges.
  Function bodies retain their own ranges. A bounded symbolic stack recovers
  qualified function assignments where a straight sequence proves the target.
- `core/perception/flash_action_sources.py` locates frame, initialization, button
  and clip-event streams. Source names include the original SWF digest. Offsets
  address its uncompressed tag stream, including nested sprite tags.

Unsupported instructions, malformed operands, branches into an operand or a
function body, and exhausted bounds make a stream incomplete. Incomplete streams
withhold qualified assignments. `With` and `Try` remain unsupported. AS3 containers
remain explicit source issues. The parser neither runs the program nor selects
controls from it.

## Checks

The two new pure parser test files passed **37 tests in 3.02 seconds**. The tests
cover forward and backward branches, function boundaries, qualified member
assignments, parameter registers, constant references, action-count skips,
truncation and distinct event sources. Ruff and the writing gate passed for all
four code and test files.

An offline read of the assigned original game 9 returned:

| Measurement | Result |
| --- | ---: |
| Separate action streams | 216 |
| Functions | 81 |
| Instructions, including function bodies | 12,147 |
| Qualified function assignments | 81 |
| Incomplete streams or source issues | 0 |

The file digest was
`ef2498224df3a9bfdea35a1441e5b42ad06538998624e5b4155313595095b93d`.
Recovered assignments included the original program's `Game.mHandleKeyboard`,
`Game.mCheckParking`, `Game.mBall2BallReaction` and `Game.mPlayCurrLevel`.
These names are parsed data; the implementation contains no game names or keys.
This read proves source preservation. It does not prove motion extraction,
perception, physical control, successful play or demo readiness.

## Integration contract

Call `read_flash_action_streams(raw)` alongside the existing text reader. Its
`FlashActionSources.streams` are independent `ActionStream` values. Require a
stream's `complete` property before using its control flow. Check the enclosing
result's issues before claiming coverage of the whole program.

Each `ActionInstruction` exposes `offset`, `next_offset`, `payload_offset`,
`payload_end`, `opcode`, `name`, `operands`, `payload`, `branch_target` and
`successors`. Branch displacements are relative to the end of the encoded action
record. Function definitions advance to the end of their body, so the enclosing
stream never falls through into that body.

Each `ActionFunction` exposes its definition and body ranges, parameter names,
register parameters, register count, preload flags and body stream. A
`QualifiedFunctionAssignment` links a target path to the exact function and
assignment instructions. Such an assignment identifies a possible binding; it
does not assert that the program executed the assignment.

Push operands retain their types. Constant-pool operands remain indices, and
constant-pool instructions retain their strings. A later symbolic evaluator must
carry the actual pool along the enclosing control flow to a function definition.
It must refuse an unresolved pool rather than treating the nearest text block as
the function's environment. Unknown branch predicates must remain alternatives
with their source addresses.
