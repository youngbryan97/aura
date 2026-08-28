# The basis is smaller than it looks

My own answer to the metalanguage problem, reached before reading the
council's and kept separate from it. Written 2026-08-28. Every number here
came from a run on the frozen 120-problem battery, and the code and the tests
are in the repository.

## Where I part company with the council

Eight models answered. They converge, closely: detect insufficiency from
invariant violations (length, multiset, no fixed permutation), propose
`sort_by(key)` / `partition_by(predicate)` / `filter(predicate)`, bound the
search with description length, validate across domains and against a shuffle
null. The detection and validation halves are good and I have taken them.

The proposal half is the question, and all eight answer it the same way: a
human lists the three new kinds. That is exactly the act I performed when I
added `grouping` — propose a missing primitive, predict what it fixes, be
right on the second attempt. Performing it three more times reaches 120/120
and leaves the boundary where it was. The fourth failing family needs a fifth
human.

Four of the eight said so themselves when pushed. Grok: "The meta-basis is
still a designer prior." Perplexity: "I replaced 'invent a representational
capability' with 'have the developer pre-author the next three operators'."
Deepseek: "textbook human-in-the-loop via foresight." That concession is the
most useful thing in the whole council, and it points where I had already
gone.

## The observation

My seven forms are not seven kinds. They are seven members of one space that
somebody enumerated by hand, one at a time, each time a family failed.

    identity      f(i) = i
    mirror        f(i) = n-1-i
    offset k      f(i) = (i+k) mod n
    grouping      deal into residue classes, lay them end to end

Every one of those is an instance of

    f(i) = (a*i + b) mod m,   for i < m,   else i

with `m` stated relative to the length. identity is `(1, 0, n)`. mirror is
`(-1, -1, n)`. offset k is `(1, k, n)`. Dealing six cells into two classes is
`(2, 0, n-1)` — the shuffle a person had to add by hand, and the classical
riffle.

The member is not enumerated. It is **solved for** from the observations, in
O(n²), by trying each modulus and each multiplier and reading the shift off
the first position. Nothing about that loop grows when the family gets wider.

## Measured

| claim | result |
|---|---|
| identity subsumed | 6/6 lengths |
| mirror subsumed | 6/6 lengths |
| offset subsumed | 27/27 (every offset, every length) |
| grouping subsumed | 19/32 directly, 11/32 as a composition, 2 out |
| permutations reached at n=6 | 44, against 15 for the written-down forms |
| forms nobody authored | 34, at that one length |
| battery, family off | 111/120, 110/110 expressible |
| battery, family on | 111/120, 110/110 expressible |

The last two rows are the regression test. The family is reached for only
where the written-down forms fail, so it cannot cost a problem, and the
34 unnamed forms are available where they were not.

## Where the claim was too strong, and how that showed

I predicted four forms would collapse into one family. Three did. Grouping did
not, and the reason is worth more than the prediction would have been: an
**even** deal, where the span divides the length, is affine; an **uneven** one
is genuinely piecewise, because the classes come out different sizes. Widening
the modulus to ±6 reached none of the 58 uneven cases. That is recorded in
`tests/test_the_basis_is_smaller_than_it_looks.py` as an assertion that it does
NOT fit, so the limit cannot quietly disappear.

Two other things broke on the way and both were the same mistake:

- The first version described a member by its absolute modulus. Dealing into
  two classes is mod 5 at length six and mod 7 at length eight, so one shape
  seen at two lengths was two shapes and intersected to nothing. The battery
  fell to 88. This is the identical defect the end-exchanges already carried a
  comment about — an absolute pair is false at another length — made again in
  a new form.
- Offered beside the written-down forms rather than after them, the family
  changed three groupings from found to lost while gaining nothing.

## What this does not reach, said plainly

Nothing above crosses to the three failing families. Every expression here is
still `f(i, n)`, blind to the cells. Sorting, content-dependent position and
variable length are a **type** change, not a wider positional family, and
there the council is right and I agree with it.

The right way to make that crossing is not to author `sort_by`, `partition_by`
and `filter` either. It is to let the same solve range over features of the
cell: Gemini's coordinate basis, where the target index is a sparse integer
combination of position features AND rank/accumulator features, so identity,
mirror, sort and filter are all weight vectors in one space found by one
linear solve. That is this result extended along the axis it cannot see, and
it is the next thing to build.

## What is honestly still a prior

The family is a designer prior, like everything else. The difference is what
kind of prior it is: not "grouping, because this week grouping failed," but a
two-parameter family closed under the operation, fitted rather than listed,
whose reach is 34 forms wider than the list at one length and which nobody has
to extend when the next positional family fails.

That is a smaller claim than "the system invents its own primitives." It is
the one the numbers support.
