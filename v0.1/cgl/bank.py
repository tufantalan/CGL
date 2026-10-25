"""CGL v0.1 --- the value bank.

Section 21 of SoftwareDescription_Rev2.  A bank is B = (C, tau, beta, P, mu):
classes keyed by (type, behaviour), each holding the *productions* that build
it rather than the expressions that inhabit it.

Two requirements from Section 22.7 are respected here even though v0.1 is
batch-only, because retrofitting them later would be a rewrite:

  1. RECORD runs on every insertion, including one whose behaviour already
     exists.  Only the *expansion* is suppressed.
  2. Class identifiers are opaque and stable.  Nothing indexes a class by
     position, and nothing assumes the set of classes only grows.
"""

import itertools
import time

from .graph import ConceptGraph
from .lang import BOT, TYPES


class Bank:
    def __init__(self, strings, ops, p_max=8):
        self.S = tuple(strings)
        self.m = len(self.S)
        self.ops = list(ops)
        self.p_max = p_max

        self._ids = itertools.count(1)
        self.tau = {}                  # id -> Type
        self.beta = {}                 # id -> behaviour tuple
        self.prods = {}                # id -> list of (Operator, arg-id tuple)
        self.mu = {}                   # id -> minimal tree size
        self.index = {}                # (Type, behaviour) -> id
        self.by_ts = {}                # (Type, size) -> list of ids

        self.levels = []               # per-level statistics
        self.tuples = 0                # candidate tuples formed, total
        self.recorded = 0              # productions recorded, total
        self.capped = False            # poolCap reached
        self.timed_out = False

    # ---------------------------------------------------------------- core

    def classes(self):
        return self.tau.keys()

    def of_type(self, t):
        return [c for c in self.tau if self.tau[c] == t]

    def bool_classes(self):
        from .lang import BOOL
        return [c for c, t in self.tau.items() if t == BOOL]

    def ev(self, op, args):
        """Section 21.2 --- the value of a production, coordinate by coordinate."""
        if not args:
            if op.symbol == "INPUT":
                return self.S
            v = op.fn()
            return (v,) * self.m
        betas = [self.beta[a] for a in args]
        out = []
        for j in range(self.m):
            vals = [b[j] for b in betas]
            out.append(BOT if any(v is BOT for v in vals) else op.apply(vals))
        return tuple(out)

    def record(self, cid, prod):
        """Section 21.4 RECORD --- keep the pMax cheapest, always the mu-attaining one."""
        ps = self.prods[cid]
        if prod in ps:
            return
        ps.append(prod)
        self.recorded += 1
        if len(ps) > self.p_max:
            ps.sort(key=lambda p: (self._tree_size(p), p[0].symbol))
            del ps[self.p_max:]

    def _tree_size(self, prod):
        op, args = prod
        return 1 + sum(self.mu[a] for a in args)

    def add(self, op, args, size):
        """Section 21.4 ADD.  Returns (class id, created?) or (None, False)."""
        self.tuples += 1
        b = self.ev(op, args)
        if all(v is BOT for v in b):
            return None, False                       # B4
        key = (op.result, b)
        cid = self.index.get(key)
        created = False
        if cid is None:
            if self.capped:
                return None, False
            cid = next(self._ids)
            self.tau[cid] = op.result
            self.beta[cid] = b
            self.mu[cid] = size
            self.prods[cid] = []
            self.index[key] = cid
            self.by_ts.setdefault((op.result, size), []).append(cid)
            created = True
        self.record(cid, (op, tuple(args)))
        return cid, created

    # ------------------------------------------------------------- building

    def build(self, max_size=8, pool_cap=200000, timeout=60.0, on_level=None):
        """Section 21.4 BUILD_BANK, level by level.

        Deviation from Section 16.4 as written: that pseudocode calls ENUMERATE
        afresh for each k, which repeats every earlier level.  Here the bank is
        built incrementally and `on_level` is called after each size, so the
        top-level learner can test for consistency at each k without the bank
        being rebuilt.  The enumeration order, and therefore the smallest-first
        guarantee, is identical.
        """
        t0 = time.perf_counter()
        for op in self.ops:
            if op.arity == 0:
                self.add(op, (), 1)
        self._close_level(1, t0)
        if on_level and on_level(self, 1):
            return self

        for size in range(2, max_size + 1):
            for op in self.ops:
                k = op.arity
                if k == 0:
                    continue
                for args in self._tuples_of_size(op, size - 1):
                    self.add(op, args, size)
                if time.perf_counter() - t0 > timeout:
                    self.timed_out = True
                    break
            if len(self.tau) > pool_cap:
                self.capped = True
            self._close_level(size, t0)
            if on_level and on_level(self, size):
                break
            if self.timed_out:
                break
        return self

    def _tuples_of_size(self, op, budget):
        """Argument tuples whose sizes sum to `budget`.  Arity is at most 2."""
        if op.arity == 1:
            return [(c,) for c in self.by_ts.get((op.params[0], budget), ())]
        out = []
        p0, p1 = op.params
        for s0 in range(1, budget):
            left = self.by_ts.get((p0, s0))
            if not left:
                continue
            right = self.by_ts.get((p1, budget - s0))
            if not right:
                continue
            for a in left:
                for b in right:
                    out.append((a, b))
        return out

    def _close_level(self, size, t0):
        by_type = {t: 0 for t in TYPES}
        for c in self.tau.values():
            by_type[c] += 1
        self.levels.append(dict(
            size=size,
            new=sum(len(self.by_ts.get((t, size), ())) for t in TYPES),
            total=len(self.tau),
            by_type=by_type,
            productions=sum(len(p) for p in self.prods.values()),
            tuples=self.tuples,
            seconds=time.perf_counter() - t0,
        ))

    # ------------------------------------------------------------ extraction

    def minimal_productions(self, cid):
        """Every production of `cid` attaining mu --- the distinct smallest forms."""
        best = self.mu[cid]
        return [p for p in self.prods[cid] if self._tree_size(p) == best]

    def extract(self, cid, graph=None, memo=None):
        """Section 21.5 EXTRACT.  The memo is what makes the result a DAG."""
        g = ConceptGraph() if graph is None else graph
        memo = {} if memo is None else memo

        def node(c):
            if c in memo:
                return memo[c]
            prods = sorted(self.prods[c],
                           key=lambda p: (self._tree_size(p), p[0].symbol))
            op, args = prods[0]
            n = g.add(op, [node(a) for a in args])
            memo[c] = n
            return n

        root = node(cid)
        if graph is None:
            g.out = root
            g.strip_unreachable()
        return g, root

    # ------------------------------------------------------------ reporting

    def collapse(self):
        """Tuples formed per class kept --- the number Section 21.7 predicts."""
        return self.tuples / max(1, len(self.tau))

    def summary(self):
        by_type = {t: 0 for t in TYPES}
        for c in self.tau.values():
            by_type[c] += 1
        return dict(
            strings=self.m,
            classes=len(self.tau),
            by_type=by_type,
            productions=sum(len(p) for p in self.prods.values()),
            recorded=self.recorded,
            tuples=self.tuples,
            collapse=self.collapse(),
            capped=self.capped,
            timed_out=self.timed_out,
            seconds=self.levels[-1]["seconds"] if self.levels else 0.0,
        )
