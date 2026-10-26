# CGL v0.1 — Concept Graph Learner

First implementation of stage 3. Builds the **value bank** of Part IV §21 of
`SoftwareDescription_Rev2.docx`, then assembles an answer with the Layer-2
sequential covering of §16.3 and the MDL model selection of §16.4.

Python 3.8+, standard library only. No dependencies, no build step.

Versioning: every later implementation gets its own sibling folder —
`CGL/v0.2/`, `CGL/v0.3/`, … — so a version is never edited in place once its
numbers have been reported.

---

## Running it

From this folder:

```
python3 cgl_client.py                      # interactive menu
python3 cgl_client.py --list               # what test cases were found
python3 cgl_client.py --case 24            # run one, by number or folder name
python3 cgl_client.py --file some/case.txt # run one from anywhere
python3 cgl_client.py --all                # every case, one summary table
python3 cgl_client.py --rank               # rank the atomic concepts (menu 4)
python3 tests/test_cgl.py                  # 38 tests
python3 experiments/measure.py all         # the generalisation numbers
```

The client scans `TestCases/` in this folder by default; `--testcases PATH`
points it somewhere else. Relative paths given to `--testcases`, `--file` and
`--dot` are taken relative to this folder, not the working directory, so every
read and write stays under `CGL/v0.1/` wherever the client is started from. Menu option 2 builds a case by typing examples line by line
(`<binary string> +` or `<binary string> -`, blank line to finish, `undo`,
`cancel`), runs it, and offers to save it as a new `example.txt`.

Useful flags: `--profile {core,boolean,full}`, `--max-size N`, `--timeout SEC`,
`--pool-cap N`, `--p-max N`, `--ops LIST`, `--dot FILE`, `--no-str-guard`,
`--no-bank`, `--no-identifiability`, `--no-oracle`, `--no-colour`, `--quiet`.

## What the report shows

- **Answer** — the hypothesis as a formula, its DAG size (§14.3), training
  errors, and MDL cost (§16.4).
- **Value bank** — per size level: classes created, classes by type,
  productions stored, tuples formed, cumulative seconds. Then the collapse
  factor, which is the number §21.7 predicts will come out below CTL's 3,453×.
- **Identifiability** (§19.4) — how many consistent classes exist, how many are
  of minimal size, every distinct smallest expression, and a **disambiguating
  string**: one input on which those minimal hypotheses disagree, which is the
  single most useful example to label next.
- **Generalisation** (§19.2) — when `oracles.py` is present and the case is one
  of the 29, accuracy over Σ^1..6, on unseen strings only, and on strings longer
  than any in training, with the confusion matrix and the first disagreements.
  `--no-oracle` turns it off.
- **Configuration** (§18) — every parameter the run used.

## Layout

```
cgl/lang.py        §11–13   types, bottom, the operator set, profiles
cgl/graph.py       §14–15   concept graph, invariants I1–I4, eval, render, DOT
cgl/bank.py        §21      the value bank: BUILD_BANK, RECORD, EXTRACT
cgl/learn.py       §16.3–4  covering into DNF, MDL selection, identifiability
cgl/testcases.py   §10      parsing, scanning, saving
cgl/report.py      §20.4    terminal report
cgl/oracle.py      §19.1–2  scoring against TestCases/oracles.py
cgl/rank.py                 concept ranking: each atomic concept alone and
                            removed, against the full set; reports/ holds
                            rank_concepts_<date>_<time>.txt and .json
cgl_client.py               menu + flags
tests/test_cgl.py           tests, each naming the section it checks
experiments/measure.py      every number in "Generalisation" below
```

Dependencies point strictly downward (§17.1). `lang` and `graph` know nothing
about learning; `bank` never sees a label.

## Operator profiles

`--profile` selects the enabled operator set, which *is* the hypothesis space
(§2.2, §3.1), so it is the strongest inductive bias in the system and is logged
with every run.

| profile | operators | behaviour |
|---|---|---|
| `core` (default) | INPUT, INT, CHR, STR, TRUE/FALSE, LEN, COUNT, MAXRUN, HEAD, LAST, CHREQ, STARTSWITH, ENDSWITH, CONTAINS, and the six Int comparisons | Str, Char and Int stop growing after size 3, so only Bool expands and the bank genuinely saturates |
| `boolean` | `core` + AND, OR, XOR, NOT | for measuring what it costs to do Layer 2's job inside Layer 1 |
| `full` | everything in §13 except DFA | arithmetic and the trims are back; this is the configuration in which the growth of §21.6 is worth measuring |

`core` omits the boolean connectives on purpose. §16.3's `LEARN_TERM` already
takes conjunctions and negations of pool predicates, so putting AND/OR/NOT into
the enumeration duplicates that work and is exactly where it blows up.

## Results — consistency

All 29 CTL test cases, `core`, `maxSize 8`, on one core:

```
29 of 29 consistent, 1.3 s total (0.01–0.10 s per case)
```

18 answers are a single literal, provably the smallest by tree size; 11 need
Layer 2. Sizes run from 3 to 21 nodes.

## Results — generalisation

Consistency is not recovery. Scored against `TestCases/oracles.py` over
every binary string of length 1–6 (126 strings), which is the universe the CTL
figures were measured on. Reproduce with `python3 experiments/measure.py all`
(35 s).

| | mean generalisation | recovered |
|---|---:|---:|
| **CGL v0.1, as shipped** | **66.4%** | **4 / 29** |
| CGL v0.1, `--no-str-guard` | 75.2% | 9 / 29 |
| CGL v0.1, `--no-str-guard`, committee vote | 77.7% | 10 / 29 |
| CTL v1 (as published) | 95.5% | 26 / 29 |
| CTL v2 (fixed + dedup, d=4) | 97.5% | 26 / 29 |

That is a large loss against the baseline, and the cause is not a defect in the
bank. Three measurements locate it.

**1. §13.10 guard 1 costs 8.8 points and five concepts, and forbids nothing.**
The guard exists to block `STREQ(INPUT, <a training string>)`. The `core`
profile has no `STREQ`, so the guard removes only the short constants that
prefix and suffix concepts need: with `"1"` struck from the pool, "starts with
1" cannot be written, and the smallest consistent answer becomes
`STARTSWITH(STR["100"], INPUT)` at 52%. With the guard off, cases 2, 3, 13, 14
and 23 go to 100%. The guard should be attached to the operators that can
memorise, not to the constant pool.

**2. The minimal version space is tiny, so tie-breaking is not the lever.**
On average only 2.4 distinct smallest consistent hypotheses exist per case.
Voting over all of them — free, since the bank stores every minimal production
— buys 2.5 points. Worth having, not a fix.

**3. The gap is missing data, not the wrong bias.** The right hypotheses are in
the space; the five-example datasets do not identify them, exactly as the
identifiability block says (5 given, ~53 recommended).

| | mean generalisation | recovered | labels added |
|---|---:|---:|---:|
| the data as given | 75.2% | 9 / 29 | 0 |
| active learning, the system's own query | 93.2% | 21 / 29 | **3.6** |
| active learning, shortest unseen string | 90.7% | 19 / 29 | 3.6 |
| + every string up to length 4 | 96.6% | 22 / 29 | ~26 |

**3.6 self-chosen labels get within 3.4 points of what 26 exhaustive ones
achieve**, and beat the same budget spent randomly. The query is the
disambiguating string of §19.4, which falls out of the productions the bank
already stores.

The honest reading of the CTL comparison is therefore not "CGL is worse". CTL
scores 97.5% on this data because its narrow vocabulary *is* the missing
information — the bias supplies what the five examples cannot. CGL's space is
wide enough to contain concepts CTL cannot express, and pays for that with a
worse point estimate on tiny samples; what it gets in return is that it knows
when it does not know, and can say which single label would settle it.

The growth §21.6 predicts, on case 29 with `--profile full`:

| size | classes | Str | Char | Int | Bool | productions | tuples |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 7 | 2 | 9 | 2 | 20 | 20 |
| 2 | 25 | 8 | 4 | 11 | 2 | 66 | 66 |
| 3 | 143 | 39 | 9 | 69 | 26 | 415 | 1,779 |
| 4 | 544 | 131 | 55 | 265 | 93 | 1,185 | 2,723 |
| 5 | 3,988 | 947 | 262 | 2,221 | 558 | 8,480 | 23,625 |
| 6 | 20,179 | 5,221 | 1,196 | 12,148 | 1,614 | 42,994 | 100,860 |

Classes saturate for Bool and grow for Int and Str, as §21.6 says they should.
Measured collapse is **5.0×** there and 13–26× on the `core` runs — far below
CTL's 3,453×, which is what §21.7 predicts, because behaviour equivalence
separates on negatives and on ⊥ as well as on positives.

## Deviations from the specification

Three, all deliberate.

1. **§16.4's `for k = 1 to maxSize: ENUMERATE(S, Ops, k)` rebuilds the bank at
   every level.** Here the bank is built once, incrementally, and the top level
   is called back after each size. The enumeration order — and therefore the
   smallest-first guarantee — is identical.
2. **`PRUNE` on reaching `poolCap` stops creating new classes rather than
   deleting existing ones.** Deleting a class would leave dangling argument
   references in other classes' productions. The report says when the cap was
   reached; it does not pretend the search was exhaustive.
3. **The single-predicate answer of §16.4 admits a negated pool predicate**, at
   one node more, because §16.3's `LEARN_TERM` admits one anyway. Without this,
   "first character differs from the last" would be reported as a one-rule DNF
   rather than as the single literal it is.

Two known gaps, both recorded in the specification itself:

- **Tree size versus DAG size** (§21.5). `μ` charges a shared class once per
  use; `size(G)` charges it once in total. Minimality is claimed for tree size
  and `size(G)` is reported separately. Extraction under a sharing-aware cost
  is NP-hard, so this is not going to be closed by a better extractor.
- **A `STR` constant that appears in the training set is excluded** (§13.10).
  Measured above: 8.8 points of generalisation and five concepts, for no gain,
  because the guard is attached to the constant pool rather than to the
  operators that can memorise. `--no-str-guard` turns it off; the default keeps
  the specified behaviour so the reported numbers stand. **This is a defect in
  the specification, not in the implementation, and §13.10 should be rewritten
  before v0.2.**

## Not in v0.1

Incremental ingestion and partition refinement (§22) — v0.1 is batch, and the
two obligations §22.7 places on a batch implementation are met, so §22 can be
added without a rewrite. Also absent: the concept store (stage 4), learned
priors (stage 5), the evolutionary and RPNI strategies (§16.5, §16.6), and the
`DFA` operator (§13.9).

Active learning is measured in `experiments/measure.py` but is **not** in the
client, where the person would answer the query instead of an oracle. On the
evidence above that is the most valuable thing v0.2 could add — and it is also
the case §22 was written for, since every query currently rebuilds the bank
from scratch.
