# PART IV --- CGL CONSTRUCTION

Part II specifies a learner that enumerates expressions and keeps one representative of each observed behaviour. That is sufficient for a tree-shaped search over a fixed training set. It is not sufficient once a value produced on one branch may be an argument to a node on another, which is the defining move of the Concept Graph Learner.

This part specifies the data structure that replaces the representative-per-behaviour bank of Section 16.2: a **value bank** whose entries are equivalence classes rather than expressions, and which records how each class can be produced rather than which expressions belong to it. Section 21 gives the batch construction, which is normative. Section 22 gives incremental ingestion by partition refinement, which is an optional extension --- unnecessary when the training set is fixed and known in advance, and required as soon as it is not.

Nothing in Part IV changes the type system (Section 11), the operator set (Section 13), the semantics (Section 15), or the second layer of the learning algorithm (Section 16.3). It changes only how the pool of candidate predicates handed to that second layer is built and stored.

# 21. The value bank

## 21.1 Why the tree stops being the right picture

In the Concept Tree Learner each node is a predicate, a path from the root is a conjunction of predicates, and the string threaded along that path is progressively trimmed. The only argument a node ever receives is what its parent hands down. Under that restriction "tree of paths" is an honest description of the search state: every candidate is a root-to-leaf path, and paths are independent.

That restriction is what CGL removes. Once `COUNT(INPUT, '0')` computed for one purpose can serve as the second argument of `TRIMHEAD` somewhere else, candidates stop being paths. Two consequences follow immediately.

The first is that the natural unit of search is no longer a candidate hypothesis but a **derived value**. What the search accumulates is a pool of values obtainable from the input, of every type, each available as an argument anywhere a value of that type is expected. Whether a particular value was originally built "for" a comparison or "for" a trim is not a property of the value and is not recorded.

The second is that deduplication changes status. In Section 4.3 observational equivalence is presented as an optimisation: it makes an otherwise intractable enumeration finish. Here it is structural. The argument pool of a node is the whole bank, so the branching factor of the search is `|Ops| × |Bank|^a` where `a` is the maximum arity. If the bank stores expressions, `|Bank|` grows as fast as the expression space and the search does not merely slow down, it fails to make progress: almost every tuple it forms is a re-derivation of a value it already has. Keying the bank by behaviour is therefore a precondition for cross-branch arguments being usable at all, not a speed-up applied afterwards.

## 21.2 Definition

Fix an ordered list of ingested strings `S = (s₁, …, s_m)`. As in Section 15.3 these are the training strings with labels excluded; the bank never sees a label, so Layer 1 remains label-free and the leakage guard of Section 10.4 stays structural.

A **value bank** over `S` is a tuple `B = (C, τ, β, P, μ)`:

  --------------------------------------------------------------------------------------
  Component     Type                          Meaning
  ------------- ----------------------------- ------------------------------------------
  `C`           finite set                    class identifiers, opaque and stable

  `τ`           `C → Type`                    the type of every expression in the class

  `β`           `C → Value^m`                 the class's behaviour vector

  `P`           `C → set of productions`      how the class can be built

  `μ`           `C → ℕ`                       the size of the smallest expression in it
  --------------------------------------------------------------------------------------

A **production** is a pair `(op, ⟨c₁, …, c_k⟩)` with `k = arity(op)`, `τ(cᵢ) = op.paramType(i)` for every `i`, and `op.resultType = τ(c)` for the class `c` that holds it. Productions of arity zero are the constants of Section 13.2 and `INPUT`.

The **value of a production** at position `j` is obtained by applying the operator to the arguments' behaviours at that position, with the strict propagation of Section 12:

    ev(op, ⟨c₁, …, c_k⟩)[j]
        = ⊥                                    if some β(cᵢ)[j] = ⊥
        = op.apply( β(c₁)[j], …, β(c_k)[j] )   otherwise

A class is therefore an equivalence class of expressions under observational equivalence on `S`, represented not by its members but by the ways of producing it. **The member expressions are never stored.** Recovering one is the job of Section 21.5.

## 21.3 Invariants

Checked on construction, and after every ingestion in Section 22.

- **B1 --- Congruence.** For every class `c` and every production `π ∈ P(c)`, `ev(π) = β(c)`. Every recorded way of building the class agrees with the class's behaviour, coordinate by coordinate. This is the invariant that Section 22 exists to restore.
- **B2 --- Quotient.** Distinct classes have distinct keys: `c ≠ c′ ⟹ (τ(c), β(c)) ≠ (τ(c′), β(c′))`. The bank is a quotient of the expression space, not a selection from it.
- **B3 --- Inhabitation.** Every class has at least one production. Classes are created only by inserting a production, so an empty class cannot arise.
- **B4 --- Non-degeneracy.** No class has `β(c) = (⊥, …, ⊥)`. An expression undefined on every training string cannot contribute to anything, since `⊥` propagates, and admitting such classes floods the bank.
- **B5 --- Minimality of `μ`.** `μ(c) = min over π = (op, ⟨c₁, …, c_k⟩) ∈ P(c) of 1 + Σᵢ μ(cᵢ)`.

Note what is *not* an invariant. The relation "class `c` has a production consuming class `d`" **may contain cycles**, and this is normal rather than a defect. If every training string happens to be a palindrome, `REVERSE(INPUT)` has the same behaviour as `INPUT`, so the class of `INPUT` acquires a production consuming itself. Likewise the class whose behaviour is constantly `true` acquires the production `AND(c, c)`. The structure is an e-graph, and e-graphs are cyclic in general.

What B5 guarantees instead is that the *extraction* sub-graph is acyclic. For the production attaining the minimum, `μ(c) = 1 + Σᵢ μ(cᵢ) > μ(cᵢ)` for every `i`, because every class has `μ ≥ 1`. Choosing the `μ`-minimising production at each class therefore always descends strictly, and extraction terminates. Cyclic productions are recorded, are semantically correct, and are simply never selected.

The palindrome example is also a reminder of the caveat in Section 4.3: equivalence is relative to `S`. A class merges expressions that the training data cannot tell apart, and that is exactly the source of both the collapse in search cost and the risk of picking the wrong representative.

## 21.4 Batch construction

The enumeration order is unchanged from Section 16.2 --- strictly increasing size, which is what makes the first consistent hypothesis provably smallest. What changes is the handling of a behaviour that has been seen before. Section 16.2 discards the candidate. Here the candidate's production is recorded on the existing class, and only the expansion is suppressed.

    BUILD_BANK(S, Ops, maxSize, poolCap, pMax):
        C ← ∅ ; index ← empty map     # (Type, behaviour) → class
        byType ← empty map            # Type → list of classes

        for each op in Ops with arity 0:
            ADD(op, ⟨⟩, 1)

        for size = 2 to maxSize:
            for each op in Ops with arity k ≥ 1:
                for each tuple ⟨c₁, …, c_k⟩ such that
                        cᵢ ∈ byType[ op.paramType(i) ]  and
                        1 + Σᵢ μ(cᵢ) = size:
                    ADD(op, ⟨c₁, …, c_k⟩, size)
            if |C| > poolCap:
                PRUNE(C)              # drop the largest-μ first
        return byType[Bool]

    ADD(op, args, size):
        b ← ev(op, args)
        if b = (⊥, …, ⊥):
            return NONE                    # B4
        c ← index[(op.resultType, b)]
        if c = NONE:
            c ← fresh identifier
            τ(c) ← op.resultType ; β(c) ← b ; μ(c) ← size ; P(c) ← ∅
            index[(op.resultType, b)] ← c
            byType[τ(c)].append(c)
        RECORD(c, (op, args))
        return c

    RECORD(c, π):
        P(c) ← P(c) ∪ { π }
        if |P(c)| > pMax:
            keep the pMax productions of smallest 1 + Σᵢ μ(cᵢ),
            always retaining the one attaining μ(c);
            ties by a fixed deterministic operator order

Four points on why this is the right shape.

**Cross-branch arguments are not a special case.** The tuple loop draws each argument from `byType` --- the whole bank of the required type --- with no notion of where a class came from. `TRIMHEAD(INPUT, COUNT(INPUT, '0'))` is formed by the same rule as any other size-5 expression. There is no wiring step, no branch bookkeeping, and no danger of missing a combination.

**The iteration is over classes, not expressions.** This is the entire economy of the structure. Each class enters the tuple loop once no matter how many expressions it stands for.

**Productions are recorded even when the class already exists.** This is what distinguishes the bank from Section 16.2's `Seen` set, and it costs one set insertion. Three later features need it and none of them can be retrofitted: extraction of *all* minimal hypotheses rather than one (requirement F6, Section 19.4), the identifiability report of Section 19.4, and incremental refinement (Section 22). Suppressing the *expansion* is what keeps the cost down; suppressing the *record* would save almost nothing and lose all three.

**Productions must be capped.** Classes saturate --- Section 21.6 --- but productions do not: a class can be the target of arbitrarily many tuples, and at large `maxSize` the memory of the bank is dominated by productions rather than by classes. `pMax` (default 8) bounds this. The retention rule keeps the `μ`-attaining production, so B5 survives the cap and extraction is unaffected; what is lost is some of the alternatives reported for identifiability, which should be stated in the report as a truncation rather than silently dropped.

## 21.5 Extraction

A class is turned back into a hypothesis by choosing, at each class reached, the production attaining `μ`, and memoising so that a class reached twice yields the same node twice-referenced:

    EXTRACT(B, c):
        memo ← empty map
        n ← NODE(c, memo)
        return the concept graph with output n,
               unreachable nodes stripped

    NODE(c, memo):
        if c ∈ memo:
            return memo[c]
        (op, ⟨c₁, …, c_k⟩) ← the production of P(c) minimising
                              1 + Σᵢ μ(cᵢ), ties broken by a
                              fixed deterministic order
        n ← new node with op(n) = op and
                          args(n) = [ NODE(cᵢ, memo) for i = 1..k ]
        memo[c] ← n
        return n

The memo is the reason the output is a DAG rather than a tree, and hence the reason a hypothesis satisfies Section 14 as written. Sharing is not detected after the fact by a common-subexpression pass; it is what the bank was storing all along.

One discrepancy must be recorded honestly, because it is a real gap between the search order and the objective. `μ` is a **tree** size: the recurrence `1 + Σᵢ μ(cᵢ)` charges for a shared class once per use. Section 14.3 defines `size(G)` as a **DAG** size, charging once in total. So `size(EXTRACT(B, c)) ≤ μ(c)`, with strict inequality exactly when extraction reuses a class. The enumeration is therefore ordered by an upper bound on the quantity the objective minimises.

This is not repaired by changing the extraction rule. Choosing productions so as to minimise DAG size --- where the benefit of a choice depends on what other choices reuse it --- is the standard e-graph extraction problem under a sharing-aware cost, and it is NP-hard. The specification therefore takes the practical position:

1. Search and enumerate by `μ`, which is cheap, monotone, and preserves the smallest-first guarantee *with respect to tree size*.
2. Report `size(G)` from Section 14.3 as the size of the delivered hypothesis, and use it in `COST` (Section 16.4).
3. State in the report that minimality is claimed for tree size. This is the same class of substitution as the one flagged for priority-weighted search: the guarantee that is proved and the guarantee that is claimed must be the same sentence.

## 21.6 Saturation, and what does not saturate

The bound of Section 4.3 carries over unchanged and is the reason the structure is finite. With `m` ingested strings, a `Bool`-valued class is determined by an element of `{true, false, ⊥}^m`, so there are at most `3^m` of them. `Char` classes are bounded by `(|Σ| + 1)^m`. `Int` and `Str` classes have no useful a-priori bound, but in practice are limited by the distinct value tuples the operators actually produce from `m` short strings, which is small.

Three quantities behave differently and should be measured separately in the cost report of Section 19.3:

  ------------------------------------------------------------------------------------
  Quantity                    Growth                        Bounded by
  --------------------------- ----------------------------- --------------------------
  classes `|C|`               saturates                     `3^m` for `Bool`; empirical
                                                            for `Int`, `Str`

  productions `Σ|P(c)|`       grows with `maxSize`          `pMax · |C|`

  tuples examined             `|Ops| · |C|^a` per level     the dominant cost
  ------------------------------------------------------------------------------------

The third is the one that decides whether a run finishes. It depends on `|C|`, which saturates, and not on the number of expressions, which does not --- which is the whole argument for the structure.

## 21.7 Behaviour equivalence is finer than coverage equivalence

The Concept Tree Learner deduplicates its leaf paths by **coverage**: the vector of which positive examples a path accepts. Measured across the twenty-nine test cases at depth 4, that collapses 1,267,369 leaf paths to 367 distinct classes, a factor of 3,453, and reduces the input to set cover from a mean of 2,458 paths per case to 12.7.

That equivalence is sound there for a specific reason: the objective of the tree learner sees a path only through its coverage of the positives and its depth, so two paths with equal coverage are interchangeable *as rules*. It is **not** sound here, and the difference is worth stating explicitly because the two look similar and the weaker one is cheaper.

  --------------------------------------------------------------------------------------------
  Keyed by                     Sound for                          Unsound for
  ---------------------------- ---------------------------------- ----------------------------
  coverage of positives        selecting among finished rules     composition
  (CTL)                        by the Layer-2 objective

  behaviour on all of `S`,     composition, hence use as an       ---
  positives and negatives      argument
  alike, including `⊥` (CGL)
  --------------------------------------------------------------------------------------------

Two `Bool` expressions that agree on every positive but differ on a negative are interchangeable as rules and *not* interchangeable as arguments: `NOT` of the first and `NOT` of the second differ, and either may be the argument that makes some larger expression consistent. Merging them would make the bank unsound as a source of arguments, which is precisely the failure mode recorded in the project history for optimistic subtree sharing in the tree learner.

The practical consequence is a prediction that the evaluation should check rather than assume: **the collapse factor here will be smaller than 3,453**, because the equivalence is finer --- it distinguishes on negatives and on `⊥` as well as on positives. If the measured factor comes out comparable, that is evidence the negatives are not discriminating, which is itself worth reporting.

## 21.8 Configuration additions

Extending the table of Section 18:

  ---------------------------------------------------------------------------------------
  Parameter           Default          Meaning
  ------------------- ---------------- --------------------------------------------------
  `pMax`              8                Productions retained per class (Section 21.4)

  `ingestMode`        `batch`          `batch` or `incremental`; see Section 22

  `bankReport`        `false`          Emit per-level class, production and tuple counts
  ---------------------------------------------------------------------------------------

# 22. Incremental ingestion and partition refinement

**Status: optional extension.** `BUILD_BANK` is normative for stage 3. This section specifies what is required when the training strings are not all known before the bank is built. It is written now, and not later, because two of its requirements are free if they are respected from the start and expensive to retrofit --- Section 22.7 states them.

## 22.1 The conflict

Deduplication and incrementality pull against each other. Two expressions that agree on `s₁, …, s_k` are merged; the next string may distinguish them. The merge was not wrong when it was made and is wrong afterwards, so what is needed is not a way to avoid merging but a way to undo one soundly.

Undoing is impossible if a class is stored as a representative expression, because the discarded members are gone. It is straightforward if a class is stored as its productions, because a production is a *schema*: `(op, ⟨c₁, …, c_k⟩)` denotes the whole set `{ op(e₁, …, e_k) : eᵢ ∈ cᵢ }`. Splitting the argument classes partitions that set without anyone ever having enumerated it. This is why B3 requires productions rather than members, and it is the single design decision that makes the rest of this section possible.

## 22.2 What ingestion must do

Appending a string `s` extends every behaviour vector by one coordinate. A class whose productions all take the same value at the new coordinate simply grows a coordinate. A class whose productions disagree there was never one class under the finer observation, and must be replaced by one class per value.

Splitting an argument class changes the value of every production consuming it, which may split the consumer, and so on. Restoring B1 under the extended observation is therefore a congruence closure, computed by the standard worklist:

    INGEST(B, s):
        children ← map, initially children(c) = {c} for every class c
        W ← all classes, ordered by increasing μ

        while W ≠ ∅:
            c ← W.pop()

            # re-point productions through split arguments
            P′ ← ∅
            for each (op, ⟨c₁, …, c_k⟩) ∈ P(c):
                for each ⟨d₁, …, d_k⟩ ∈
                        children(c₁) × … × children(c_k):
                    P′ ← P′ ∪ { (op, ⟨d₁, …, d_k⟩) }

            groups ← partition P′ by ev_s(·)    # new coordinate

            if |groups| = 1:
                β(c) ← β(c) ++ [ the single value ]
                P(c) ← P′
            else:
                for each group g ∈ groups:
                    cg ← fresh identifier
                    τ(cg) ← τ(c)
                    β(cg) ← β(c) ++ [ value of g ]
                    P(cg) ← g
                    μ(cg) ← recomputed by B5
                children(c) ← { cg : g ∈ groups }
                remove c from C ; add the fresh classes
                W.push( every class consuming c )

        drop any class violating B4
        recompute μ to a fixed point ; return B

The re-pointing step is the part an implementation is most likely to get wrong. A production is *not* redirected to whichever child "the original expression" fell into --- there is no original expression. It is expanded into the product of its arguments' children, and each expansion is evaluated on its own. Expansions that turn out to have equal values simply land in the same group.

## 22.3 Invariants

- **R1 --- Congruence restored.** On return, B1 holds over `m + 1` coordinates.
- **R2 --- Refinement only.** The partition after ingestion refines the partition before it. Two classes distinguished by the first `m` strings are still distinguished after the next one, so **classes never merge**. Nothing in the algorithm needs a merge case, and an implementation that appears to need one has a bug elsewhere.
- **R3 --- Identity is stable except on split.** A class that does not split keeps its identifier. A class that splits ceases to exist, and every external reference to it --- an extracted hypothesis, an entry in the concept store --- is invalidated and must be revisited. This is the hook the store's dependency tracking attaches to.
- **R4 --- `μ` never decreases.** Each child inherits a subset of its parent's productions, so its minimum can only rise. A hypothesis is therefore never made smaller by seeing more data, and a lower bound established before ingestion remains a lower bound after it.

**Termination.** Each pass either settles a class or strictly refines the partition. The partition is bounded below by the finest partition of the expressions present, which is finite because behaviours over `m + 1` coordinates are finite, so only finitely many splits can occur. Cyclic productions --- Section 21.3 --- are handled by the worklist rather than by the ordering, which is why the ordering by `μ` is an efficiency heuristic and not a correctness requirement.

**Cost.** A single split costs the re-evaluation of the productions of the split class and enqueues its consumers, so its cost is proportional to `pMax` times the number of classes consuming it. The amortised statement is the useful one: because the partition only ever refines, the total number of splits across *all* ingestions is bounded by `|C_final| − |C_initial|`. Incremental construction therefore does not repeat the work of batch construction once per example; it pays, in total, for the splits that distinguish the classes the batch build would have created directly.

## 22.4 Identifiability comes out of this for free

A `Bool` class that has survived every ingestion while holding two or more productions of equal `μ` is exactly a set of distinct minimal expressions that the data cannot tell apart. That is the set `C` of Section 16.4 and the count reported in Section 19.4, available by inspection rather than by a separate search.

The disambiguating example of Section 19.4 has an equally direct reading: it is a string `s` on which two productions of one class take different values. Confirming it is one call to `INGEST` --- the class splits, and the report can name which hypotheses went which way. Under batch construction the same question requires rebuilding the bank from scratch with the candidate string appended, which is the reason an active-learning loop wants this section and a fixed-training-set run does not.

## 22.5 When refinement earns its keep

  ----------------------------------------------------------------------------------------
  Situation                             Why batch is not enough
  ------------------------------------- --------------------------------------------------
  Active learning / membership queries   Each query appends a string chosen *because* the
  (Sections 2.7, 19.4)                   bank cannot resolve it; rebuilding per query is
                                         quadratic in the number of queries

  Concept store (stage 4)                Entries outlive the training set they were learned
                                         from and must remain valid, or be explicitly
                                         invalidated, when new data arrives

  Optimistic sharing in the tree         Sharing a subtree between two nodes is sound only
  learner                                if they produce the same residual for every word,
                                         which is unknown during incremental construction;
                                         merge-then-refine is the sound form of it

  Streaming or oversized `S`             Ingest a prefix, prune, continue, rather than
                                         holding every behaviour vector at full width
  ----------------------------------------------------------------------------------------

Where none of these applies, refinement is pure overhead: with `S` fixed and known, `BUILD_BANK` computes the fully refined partition directly, in one pass, and no split ever occurs.

## 22.6 Relation to Section 16.2

Section 16.2's `ENUMERATE` is the special case of `BUILD_BANK` in which productions are discarded on insertion and `pMax = 1`. Section 22 is the general case of ingestion in which `INGEST` is called once per string as it arrives; calling it `m` times before any query, on a bank built from no strings at all, reproduces `BUILD_BANK` --- more slowly, and with no other difference in the result.

## 22.7 What stage 3 must do now

Two requirements, both free in a batch implementation and both expensive later:

1. **Store productions, not members.** `RECORD` must run on every insertion, including the ones whose behaviour already exists. Under batch construction the recorded productions are used only for identifiability reporting, so the temptation is to skip them under a flag. Skipping them makes Section 22 a rewrite rather than an addition.
2. **Class identifiers must be opaque and stable.** No part of the system may index a class by its position in a list, and no part may assume the set of classes only grows. `children` in Section 22.2 replaces one identifier with several; every consumer of the bank must be able to survive that. A dictionary keyed by identifier is fine; an array index into `byType` is not.

Neither costs measurable time. Both are load-bearing for stages 4 and 5.
