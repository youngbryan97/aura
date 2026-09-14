# G05: resident public-shape canary

`tools/run_public_shape_canary.py` loads the registry's verified resident model
under the existing standalone model lease. Each fixture supplies an ordinary
task, a requested output shape and an exact JSON expectation. Only the task
text reaches the prompt. The shared worker grammar processor receives the
shape and the current private-channel closing token; it does not receive the
expected answer. The existing public-channel decoder supplies EOS, boundary
and token evidence without retaining private text.

The result separates public completion, JSON parsing, requested shape,
processor continuity and exact expected JSON. A valid wrong answer, a capped
answer and a processor refusal are distinct failures. Each completed case is
written through the governed file gateway before the next case starts.
Source hashes, model identity and fixture contents are retained. The artifact
always says qualification=false: this development canary is not broad gain or
desktop runtime proof.

Seventy tests passed in 22.72 seconds across canary assessment, public-channel
decoding, grammar-state equivalence and the existing decoder-shape suite.
The runner tests check that expected answers never reach generation, that an
unsupported private boundary stops before decoding, and that partial receipts
cannot claim completion. Resident measurements follow this checkpoint.
