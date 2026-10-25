"""CGL v0.1 --- the learner.

Layer 2 (Section 16.3, sequential covering into DNF) and the top level
(Section 16.4, model selection by MDL cost), running over the value bank of
Section 21 instead of the representative-per-behaviour bank of Section 16.2.

Layer 1 never sees a label.  Labels enter this module and nowhere below it.
"""

import math
import time
from dataclasses import dataclass, field

from .bank import Bank
from .graph import ConceptGraph, classify, render
from .lang import BOOL, build_operators, logical_ops


# ---------------------------------------------------------------- config

@dataclass
class Config:
    max_size: int = 8
    pool_cap: int = 200000
    timeout: float = 60.0
    p_max: int = 8
    K: int = 8
    l_str: int = 3
    max_dec_len: int = 64
    sigma: str = "01"
    profile: str = "core"
    str_guard: bool = True         # Section 13.10 guard 1
    operators: tuple = None       # explicit override of the profile
    max_alternatives: int = 10
    epsilon: float = 0.1
    delta: float = 0.05

    def as_rows(self):
        rows = [("maxSize", self.max_size), ("poolCap", self.pool_cap),
                ("timeout", "%.0f s" % self.timeout), ("pMax", self.p_max),
                ("K", self.K), ("L_str", self.l_str),
                ("operators", self.profile if not self.operators
                 else ",".join(self.operators)),
                ("strGuard", "on (§13.10)" if self.str_guard else "off")]
        return rows


# ---------------------------------------------------------------- results

@dataclass
class Result:
    case: str
    config: Config
    bank: Bank = None
    graph: ConceptGraph = None
    formula: str = ""
    kind: str = "none"            # "single", "dnf", or "none"
    rules: list = field(default_factory=list)
    size: int = 0
    errors: int = 0
    cost: float = 0.0
    stopped_at: int = 0
    seconds: float = 0.0
    # identifiability, Section 19.4
    consistent_classes: int = 0
    minimal_classes: int = 0
    alternatives: list = field(default_factory=list)
    alternatives_truncated: bool = False
    disambiguator: str = ""
    disambiguator_split: tuple = ()
    sample_needed: int = 0
    n_examples: int = 0
    note: str = ""


# ---------------------------------------------------------------- layer 2

def _popcount(x):
    return bin(x).count("1")


def _foil_gain(tv, covP, covN):
    P0, N0 = _popcount(covP), _popcount(covN)
    P1, N1 = _popcount(covP & tv), _popcount(covN & tv)
    if P1 == 0:
        return None
    return P1 * (math.log2(P1 / (P1 + N1)) - math.log2(P0 / (P0 + N0)))


def _learn_term(lits, covP0, covN0):
    """Section 16.3 LEARN_TERM.  `lits` is a list of (key, truth-mask, size)."""
    term, covP, covN = [], covP0, covN0
    while covN:
        best, best_gain, best_key = None, 0.0, None
        for key, tv, sz in lits:
            g = _foil_gain(tv, covP, covN)
            if g is None or g <= 0:
                continue
            k = (g, -sz)
            if best is None or k > best_key:
                best, best_gain, best_key = (key, tv), g, k
        if best is None:
            return None, 0
        key, tv = best
        term.append(key)
        covP &= tv
        covN &= tv
        if covP == 0:
            return None, 0
    return term, covP


def _learn_dnf(lits, pos_mask, neg_mask, max_rules=16):
    """Section 16.3 LEARN_DNF."""
    rules, uncovered = [], pos_mask
    while uncovered:
        term, cov = _learn_term(lits, uncovered, neg_mask)
        if term is None:
            return None
        rules.append(term)
        newly = cov & uncovered
        if not newly:
            return None
        uncovered &= ~newly
        if len(rules) > max_rules:
            return None
    return rules


# ------------------------------------------------------- graph assembly

def _graph_from_rules(bank, rules):
    """Build one concept graph for a DNF, sharing every class across terms."""
    logic = logical_ops()
    g, memo = ConceptGraph(), {}
    term_nodes = []
    for term in rules:
        lit_nodes = []
        for cid, negated in term:
            _, n = bank.extract(cid, graph=g, memo=memo)
            if negated:
                n = g.add(logic["NOT"], [n])
            lit_nodes.append(n)
        node = lit_nodes[0]
        for n in lit_nodes[1:]:
            node = g.add(logic["AND"], [node, n])
        term_nodes.append(node)
    out = term_nodes[0]
    for n in term_nodes[1:]:
        out = g.add(logic["OR"], [out, n])
    g.out = out
    g.strip_unreachable()
    return g


def _cost(g, data, n_ops):
    errs = sum(1 for s, lab in data if classify(g, s) != lab)
    return errs, errs * math.log2(max(2, len(data))) + g.size() * math.log2(n_ops)


# ---------------------------------------------------------------- top level

def learn(case, cfg: Config = None, progress=None) -> Result:
    """Section 16.4 LEARN.  `case` is a TestCase (strings and labels separate)."""
    cfg = cfg or Config()
    t0 = time.perf_counter()
    S = case.strings
    m = len(S)
    pos_mask = sum(1 << j for j, lab in enumerate(case.labels) if lab)
    neg_mask = sum(1 << j for j, lab in enumerate(case.labels) if not lab)

    ops = build_operators(sigma=cfg.sigma, K=cfg.K, l_str=cfg.l_str,
                          max_dec_len=cfg.max_dec_len,
                          training=S if cfg.str_guard else (),
                          profile=cfg.profile, explicit=cfg.operators)
    bank = Bank(S, ops, p_max=cfg.p_max)
    res = Result(case=case.name, config=cfg, bank=bank, n_examples=m)

    state = {"done": False, "best": None, "best_cost": float("inf"),
             "best_rules": None, "level": 0}

    def bool_literals():
        lits = []
        full = (1 << m) - 1
        for cid in bank.bool_classes():
            tv = sum(1 << j for j, v in enumerate(bank.beta[cid]) if v is True)
            lits.append(((cid, False), tv, bank.mu[cid]))
            lits.append(((cid, True), full & ~tv, bank.mu[cid] + 1))
        return lits

    def on_level(b, size):
        if state["done"]:
            return True
        state["level"] = size
        if progress:
            progress(b, size)

        # a single literal suffices?  Section 16.4's early return.  Only
        # candidates whose effective size is at most this level may be accepted,
        # or the smallest-first guarantee would be broken by a negation.
        cons = [c for c in _consistent_classes(b, pos_mask) if c[2] <= size]
        if cons:
            state["done"] = True
            state["single"] = cons
            return True

        lits = bool_literals()
        if not lits:
            return False
        rules = _learn_dnf(lits, pos_mask, neg_mask)
        if rules is None:
            return False
        g = _graph_from_rules(b, rules)
        errs, c = _cost(g, list(zip(case.strings, case.labels)), len(ops))
        if c < state["best_cost"]:
            state.update(best=g, best_cost=c, best_rules=rules, best_errs=errs)
        if state["best"] is not None and size >= state["best"].size():
            state["done"] = True
            return True
        return False

    bank.build(max_size=cfg.max_size, pool_cap=cfg.pool_cap,
               timeout=cfg.timeout, on_level=_guard(on_level, state))

    res.stopped_at = state["level"]

    if state.get("single"):
        cons = state["single"]
        best = min(c[2] for c in cons)
        minimal = sorted(((cid, neg) for cid, neg, sz in cons if sz == best),
                         key=lambda t: (t[1], bank.mu[t[0]], t[0]))
        g = _literal_graph(bank, *minimal[0])
        g.check()
        res.kind = "single"
        res.graph = g
        res.formula = render(g)
        res.rules = [[minimal[0]]]
        res.consistent_classes = len(cons)
        res.minimal_classes = len(minimal)
        res.alternatives, res.alternatives_truncated = _alternatives(
            bank, minimal, cfg.max_alternatives)
        res.disambiguator, res.disambiguator_split = _disambiguate(
            bank, minimal, case, cfg)
    elif state.get("best") is not None:
        g = state["best"]
        g.check()
        res.graph = g
        res.formula = render(g)
        res.rules = state["best_rules"]
        res.errors = state.get("best_errs", 0)
        res.kind = "dnf"
        if len(res.rules) == 1 and len(res.rules[0]) == 1:
            # Layer 2 returned a bare literal; that is the single-predicate
            # answer of Section 16.4, so report it as one and say how many
            # other literals of the same size are equally consistent.
            cid, neg = res.rules[0][0]
            eff = bank.mu[cid] + (1 if neg else 0)
            cons = _consistent_classes(bank, pos_mask)
            minimal = sorted(((c, n) for c, n, sz in cons if sz == eff),
                             key=lambda t: (t[1], bank.mu[t[0]], t[0]))
            if minimal:
                res.kind = "single"
                res.consistent_classes = len(cons)
                res.minimal_classes = len(minimal)
                res.alternatives, res.alternatives_truncated = _alternatives(
                    bank, minimal, cfg.max_alternatives)
                res.disambiguator, res.disambiguator_split = _disambiguate(
                    bank, minimal, case, cfg)
    else:
        res.note = ("no consistent hypothesis within the budget"
                    + (" (timed out)" if bank.timed_out else "")
                    + (" (poolCap reached)" if bank.capped else ""))

    if res.graph is not None:
        data = list(zip(case.strings, case.labels))
        res.errors, res.cost = _cost(res.graph, data, len(ops))
        res.size = res.graph.size()

    n_bool = sum(1 for c in bank.tau if bank.tau[c] == BOOL)
    res.sample_needed = _sample_complexity(max(2, n_bool), cfg.epsilon, cfg.delta)
    res.seconds = time.perf_counter() - t0
    return res


def _guard(fn, state):
    def wrapped(b, size):
        if state["done"]:
            return True
        return bool(fn(b, size))
    return wrapped


def _consistent_classes(bank, pos_mask):
    """Pool predicates consistent with the labels, each with its effective size.

    Layer 2 admits the negation of any pool predicate (Section 16.3's
    `Preds u {not p}`), so a class whose negation is consistent counts too, at
    one node more.  Bottom classifies negative (Section 12 rule 3), so a direct
    predicate needs True on every positive and not-True on every negative,
    while a negated one needs False on every positive and not-False on every
    negative.

    Returns a list of (class id, negated, effective size).
    """
    out = []
    for cid, t in bank.tau.items():
        if t != BOOL:
            continue
        direct = negated = True
        for j, v in enumerate(bank.beta[cid]):
            if pos_mask >> j & 1:
                direct &= v is True
                negated &= v is False
            else:
                direct &= v is not True
                negated &= v is not False
            if not (direct or negated):
                break
        if direct:
            out.append((cid, False, bank.mu[cid]))
        if negated:
            out.append((cid, True, bank.mu[cid] + 1))
    return out


def _literal_graph(bank, cid, negated, production=None):
    """One literal as a standalone concept graph."""
    g, memo = ConceptGraph(), {}
    if production is None:
        _, n = bank.extract(cid, graph=g, memo=memo)
    else:
        op, args = production
        n = g.add(op, [bank.extract(a, graph=g, memo=memo)[1] for a in args])
    if negated:
        n = g.add(logical_ops()["NOT"], [n])
    g.out = n
    g.strip_unreachable()
    return g


def _alternatives(bank, minimal, limit):
    """Every distinct smallest consistent expression (Section 19.4 / F6)."""
    out = []
    for cid, negated in minimal:
        for prod in bank.minimal_productions(cid):
            g = _literal_graph(bank, cid, negated, prod)
            out.append((cid, render(g), g.size()))
            if len(out) >= limit:
                return out, True
    return out, False


def _universe(max_len, sigma="01"):
    out = []
    for n in range(1, max_len + 1):
        stack = [""]
        for _ in range(n):
            stack = [p + c for p in stack for c in sigma]
        out.extend(stack)
    return out


def _disambiguate(bank, minimal, case, cfg, cap=8):
    """A string on which two minimal consistent hypotheses disagree (19.4)."""
    graphs = []
    for cid, negated in minimal:
        for prod in bank.minimal_productions(cid):
            graphs.append(_literal_graph(bank, cid, negated, prod))
            if len(graphs) >= cap:
                break
        if len(graphs) >= cap:
            break
    if len(graphs) < 2:
        return "", ()
    seen = set(case.strings)
    span = min(9, max(len(s) for s in case.strings) + 1)
    for s in _universe(span, cfg.sigma):
        if s in seen:
            continue
        verdicts = [classify(g, s) for g in graphs]
        if len(set(verdicts)) > 1:
            yes = sum(1 for v in verdicts if v)
            return s, (yes, len(verdicts) - yes)
    return "", ()


def _sample_complexity(h_size, eps, delta):
    """Section 2.7 --- m >= (1/eps)(ln|H| + ln(1/delta))."""
    return int(math.ceil((math.log(h_size) + math.log(1 / delta)) / eps))
