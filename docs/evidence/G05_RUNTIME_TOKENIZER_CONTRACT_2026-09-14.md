# G05: use the runtime tokenizer's declared vocabulary

The resident public-shape canary loaded the 27B, then failed before its first
decode. MLX returns a TokenizerWrapper whose ordinary methods delegate to the
underlying tokenizer but whose special methods do not include len(). The
shared grammar builder called len(tokenizer), so the worker's actual tokenizer
could not construct a grammar mask. Tests using the underlying Hugging Face
tokenizer did not expose this boundary.

The grammar builder now reads get_vocab() when available. Maximum declared ID
determines mask width, including added tokens and sparse vocabularies. Gaps
remain unavailable even if a decoder substitutes text for unknown IDs. The
existing sized-tokenizer interface remains supported for simple adapters.

EOS comes from declared tokenizer IDs, in scalar or collection form. The old
guesses for token names could map an absent name to UNK and mistakenly allow
it as a terminating token. The decoder no longer infers EOS that way.

The reproducer used the installed MLX TokenizerWrapper, added ordinary and
special tokens, and sparse declared IDs. Six tests failed before the repair.
The first broader run caught the resident underlying tokenizer exposing a
scalar eos_token_ids; that representation is now covered explicitly. The
final combined suite passed 79 tests in 20.60 seconds.

Both failed launch receipts remain in the experiment directories
`~/.aura/experiments/public-shape-20260914` and
`~/.aura/experiments/public-shape-20260914-v2`. Neither produced a model answer.
The next resident attempt must measure the fixed processor; these CPU tests
do not close G05.
