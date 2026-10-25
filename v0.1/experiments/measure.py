#!/usr/bin/env python3
"""CGL v0.1 --- generalisation experiments.

Every number in the README's Generalisation section comes from here.  Nothing
should be taken on trust; each claim maps to one subcommand.

    python3 experiments/measure.py baseline    generalisation as shipped
    python3 experiments/measure.py guard       cost of Section 13.10 guard 1
    python3 experiments/measure.py committee   voting over the minimal version space
    python3 experiments/measure.py active      active learning vs a random query
    python3 experiments/measure.py all

`--testcases PATH` points at a TestCases folder containing `oracles.py`;
the default is the one the client uses.

The oracle is used only to *label* a query and to score a finished hypothesis.
It never reaches the learner.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cgl import oracle as O                                       # noqa: E402
from cgl import testcases as TC                                   # noqa: E402
from cgl.graph import classify                                    # noqa: E402
from cgl.learn import (Config, _consistent_classes,               # noqa: E402
                       _literal_graph, learn)

UNIVERSE = None
CASES = None
ORACLES = None


def setup(root):
    global CASES, ORACLES
    CASES = TC.scan(root)
    ORACLES = O.load(root)
    if ORACLES is None:
        sys.exit("no oracles.py under %s" % root)
    if not CASES:
        sys.exit("no test cases under %s" % root)


def pred_for(case):
    return ORACLES.predicates[O.case_key(case.name)]


def accuracy(graph, pred):
    if graph is None:
        return 0.0
    u = ORACLES.universe
    return sum(1 for w in u if classify(graph, w) == bool(pred(w))) / len(u)


def summarise(label, accs, recs):
    n = len(accs)
    print("  %-34s mean %5.1f%%   recovered %2d/%d"
          % (label, 100 * sum(accs) / n, recs, n))


# ------------------------------------------------------------------ baseline

def run_baseline(cfg_kw=None, label="as shipped", quiet=False):
    accs, recs, rows = [], 0, []
    for c in CASES:
        res = learn(c, Config(**(cfg_kw or {})))
        a = accuracy(res.graph, pred_for(c))
        accs.append(a)
        recs += (a == 1.0)
        rows.append((c.name, a, res.formula))
    if not quiet:
        summarise(label, accs, recs)
    return accs, recs, rows


def cmd_baseline(args):
    print("\nGeneralisation over Σ^1..6, %d strings\n" % len(ORACLES.universe))
    _, _, rows = run_baseline()
    print()
    for name, a, formula in rows:
        print("  %-12s %5.1f%%  %s" % (name, 100 * a, formula[:52]))
    print()
    for b in (O.CTL_V2_BASELINE, O.CTL_V1_BASELINE):
        print("  %-34s mean %5.1f%%   recovered %2d/%d"
              % (b["label"], 100 * b["mean_generalisation"],
                 b["recovered"], b["cases"]))


# ---------------------------------------------------------------- guard cost

def cmd_guard(args):
    print("\nSection 13.10 guard 1 --- training strings excluded from the STR "
          "pool\n")
    run_baseline({"str_guard": True}, "guard on (as specified)")
    run_baseline({"str_guard": False}, "guard off (full constant pool)")
    print("\n  The guard exists to stop STREQ(INPUT, <a training string>).")
    print("  The `core` profile has no STREQ, so it forbids nothing and only")
    print("  removes the short constants prefix and suffix concepts need.")


# ----------------------------------------------------------------- committee

def committee(bank, case, res):
    """Every minimal consistent literal, as a graph.  Free from the bank."""
    if res.kind != "single":
        return [res.graph]
    pos = sum(1 << j for j, lab in enumerate(case.labels) if lab)
    cons = _consistent_classes(bank, pos)
    if not cons:
        return [res.graph]
    best = min(c[2] for c in cons)
    out = [_literal_graph(bank, cid, neg, prod)
           for cid, neg, sz in cons if sz == best
           for prod in bank.minimal_productions(cid)]
    return out or [res.graph]


def vote(graphs, s):
    return sum(1 for g in graphs if classify(g, s)) * 2 > len(graphs)


def cmd_committee(args):
    print("\nOne minimal hypothesis versus a majority vote over all of them\n")
    for guard in (True, False):
        a1, r1, a2, r2, sizes = [], 0, [], 0, []
        for c in CASES:
            res = learn(c, Config(str_guard=guard))
            p = pred_for(c)
            x = accuracy(res.graph, p)
            a1.append(x)
            r1 += (x == 1.0)
            gs = committee(res.bank, c, res)
            sizes.append(len(gs))
            u = ORACLES.universe
            y = sum(1 for w in u if vote(gs, w) == bool(p(w))) / len(u)
            a2.append(y)
            r2 += (y == 1.0)
        tag = "guard on" if guard else "guard off"
        summarise("%s, single hypothesis" % tag, a1, r1)
        summarise("%s, committee vote" % tag, a2, r2)
        print("  %-34s mean |minimal version space| = %.1f"
              % ("", sum(sizes) / len(sizes)))


# ------------------------------------------------------------ active learning

def saturate(case, pred, upto):
    c = TC.TestCase(name=case.name, strings=list(case.strings),
                    labels=list(case.labels))
    seen = set(c.strings)
    for w in ORACLES.universe:
        if len(w) <= upto and w not in seen:
            c.strings.append(w)
            c.labels.append(bool(pred(w)))
            seen.add(w)
    return c


def active(case, pred, rounds, smart, cfg_kw):
    c = TC.TestCase(name=case.name, strings=list(case.strings),
                    labels=list(case.labels))
    res = learn(c, Config(**cfg_kw))
    used = 0
    for _ in range(rounds):
        if accuracy(res.graph, pred) == 1.0:
            break
        q = res.disambiguator if (smart and res.disambiguator) else ""
        if not q:
            seen = set(c.strings)
            q = next((w for w in ORACLES.universe if w not in seen), "")
        if not q or q in c.strings:
            break
        c.strings.append(q)
        c.labels.append(bool(pred(q)))
        used += 1
        res = learn(c, Config(**cfg_kw))
    return res, used


def cmd_active(args):
    rounds = args.rounds
    kw = {"str_guard": False}
    print("\nIs the gap missing data or the wrong bias?  (guard off, %d queries "
          "at most)\n" % rounds)
    cols = {k: ([], 0) for k in ("base", "sat", "act", "rnd")}
    used_total = 0
    print("  %-12s %7s %7s %7s %7s" % ("case", "base", "sat≤4", "active", "random"))
    for c in CASES:
        p = pred_for(c)
        b = accuracy(learn(c, Config(**kw)).graph, p)
        s = accuracy(learn(saturate(c, p, 4), Config(**kw)).graph, p)
        ra, used = active(c, p, rounds, True, kw)
        used_total += used
        a = accuracy(ra.graph, p)
        rr, _ = active(c, p, rounds, False, kw)
        r = accuracy(rr.graph, p)
        for k, v in (("base", b), ("sat", s), ("act", a), ("rnd", r)):
            accs, rec = cols[k]
            accs.append(v)
            cols[k] = (accs, rec + (v == 1.0))
        print("  %-12s %6.1f%% %6.1f%% %6.1f%% %6.1f%%"
              % (c.name, 100 * b, 100 * s, 100 * a, 100 * r))
    print()
    for k, label in (("base", "the data as given"),
                     ("sat", "+ every string up to length 4"),
                     ("act", "active: the system's own query"),
                     ("rnd", "active: shortest unseen string")):
        summarise(label, *cols[k])
    print("\n  mean queries the active learner asked: %.1f"
          % (used_total / len(CASES)))


# ---------------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(prog="measure.py")
    p.add_argument("command", choices=("baseline", "guard", "committee",
                                       "active", "all"))
    p.add_argument("--testcases", default=None)
    p.add_argument("--rounds", type=int, default=10)
    args = p.parse_args(argv)
    setup(TC.resolve(args.testcases, ROOT) or TC.default_root(ROOT))
    todo = (["baseline", "guard", "committee", "active"]
            if args.command == "all" else [args.command])
    for name in todo:
        globals()["cmd_" + name](args)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
