"""CGL v0.1 --- concept ranking.

Measurement only, like `oracle`: it drives the learner with different operator
sets and compares the outcomes; the learner knows nothing about it.

An *atomic concept* is one operator of the enabled set that reads something
out of the string (LEN, COUNT, MAXRUN, HEAD, LAST, STARTSWITH, ...).  The rest
of the set is *support*: INPUT, the constants, the comparisons and the boolean
connectives.  Support cannot express a concept on its own, so it stays enabled
in every run; only the atomic concepts are varied.

Three configurations are run over every test case:

    all         every operator of the profile         (the reference)
    only X      support + X                           (isolation)
    without X   every operator except X               (ablation)

Isolation answers "how far does X get by itself"; ablation answers "what is
lost if X is taken away", i.e. how necessary X is given the others.
"""

import dataclasses
import json
import os
import time
from dataclasses import dataclass, field

from .lang import build_operators
from .learn import learn
from . import oracle as O
from . import report as R


SUPPORT_FAMILIES = {"input", "int_const", "chr_const", "str_const",
                    "bool_const", "cmp", "bool"}


def _base(symbol):
    return symbol.split("[")[0]


def split_operators(cfg):
    """(support symbols, atomic concept symbols) of the configured set."""
    ops = build_operators(sigma=cfg.sigma, K=cfg.K, l_str=cfg.l_str,
                          profile=cfg.profile, explicit=cfg.operators)
    support, concepts = [], []
    for o in ops:
        b = _base(o.symbol)
        dest = support if o.family in SUPPORT_FAMILIES else concepts
        if b not in dest:
            dest.append(b)
    return support, concepts


# ---------------------------------------------------------------- one run

@dataclass
class CaseRun:
    case: str
    consistent: bool
    kind: str
    size: int
    errors: int
    cost: float
    seconds: float
    classes: int
    tuples: int
    consistent_classes: int
    minimal_classes: int
    alternatives: int
    sample_needed: int
    budget_hit: bool
    formula: str
    ops_used: list
    scored: bool = False
    accuracy: float = 0.0
    unseen_accuracy: float = 0.0
    beyond_accuracy: float = 0.0
    f1: float = 0.0
    recovered: bool = False


@dataclass
class ConfigRun:
    label: str
    operators: list
    runs: list = field(default_factory=list)

    # -- aggregates ------------------------------------------------------

    def _mean(self, xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else 0.0

    @property
    def n(self):
        return len(self.runs)

    @property
    def consistent(self):
        return sum(r.consistent for r in self.runs)

    @property
    def scored(self):
        return [r for r in self.runs if r.scored]

    @property
    def mean_gen(self):
        return self._mean(r.accuracy for r in self.scored)

    @property
    def mean_gen_solved(self):
        """Mean generalisation over the cases this configuration passes; a
        failed case has no hypothesis, so `mean_gen` scores it as all-false."""
        return self._mean(r.accuracy for r in self.scored if r.consistent)

    @property
    def mean_unseen(self):
        return self._mean(r.unseen_accuracy for r in self.scored)

    @property
    def mean_f1(self):
        return self._mean(r.f1 for r in self.scored)

    @property
    def recovered(self):
        return sum(r.recovered for r in self.scored)

    @property
    def seconds(self):
        return sum(r.seconds for r in self.runs)

    @property
    def classes(self):
        return sum(r.classes for r in self.runs)

    @property
    def tuples(self):
        return sum(r.tuples for r in self.runs)

    @property
    def collapse(self):
        return self.tuples / max(1, self.classes)

    @property
    def mean_size(self):
        return self._mean(r.size for r in self.runs if r.consistent)

    @property
    def mean_cost(self):
        return self._mean(r.cost for r in self.runs if r.consistent)

    @property
    def mean_alternatives(self):
        return self._mean(r.alternatives for r in self.runs
                          if r.consistent and r.kind == "single")

    @property
    def budget_hits(self):
        return sum(r.budget_hit for r in self.runs)

    @property
    def gen_per_second(self):
        return self.mean_gen / self.seconds if self.seconds else 0.0

    def as_dict(self):
        d = {"label": self.label, "operators": self.operators}
        for k in ("n", "consistent", "mean_gen", "mean_gen_solved",
                  "mean_unseen", "mean_f1", "recovered", "seconds", "classes",
                  "tuples", "collapse", "mean_size", "mean_cost",
                  "mean_alternatives", "budget_hits",
                  "gen_per_second"):
            d[k] = getattr(self, k)
        d["runs"] = [dataclasses.asdict(r) for r in self.runs]
        return d


def _run_case(case, cfg, oracles):
    res = learn(case, cfg)
    ok = res.kind != "none" and res.errors == 0
    used = []
    if res.graph is not None:
        used = sorted({_base(res.graph.op[n].symbol)
                       for n in res.graph.reachable()})
    s = res.bank.summary()
    run = CaseRun(case=case.name, consistent=ok, kind=res.kind,
                  size=res.size, errors=res.errors, cost=res.cost,
                  seconds=res.seconds, classes=s["classes"],
                  tuples=s["tuples"],
                  consistent_classes=res.consistent_classes,
                  minimal_classes=res.minimal_classes,
                  alternatives=len(res.alternatives),
                  sample_needed=res.sample_needed,
                  budget_hit=bool(s["capped"] or s["timed_out"]),
                  formula=res.formula, ops_used=used)
    sc = O.score_result(case, res, oracles)
    if sc is not None:
        run.scored = True
        run.accuracy = sc.accuracy
        run.unseen_accuracy = sc.unseen_accuracy
        run.beyond_accuracy = sc.beyond_accuracy
        run.f1 = sc.f1
        run.recovered = sc.recovered
    return run


def run_config(label, operators, cases, cfg, oracles, progress=None):
    sub = dataclasses.replace(cfg, operators=tuple(operators))
    cr = ConfigRun(label=label, operators=list(operators))
    for c in cases:
        if progress:
            progress(label, c.name)
        cr.runs.append(_run_case(c, sub, oracles))
    return cr


# ---------------------------------------------------------------- the study

@dataclass
class Ranking:
    profile: str
    support: list
    concepts: list
    reference: ConfigRun
    isolated: dict
    ablated: dict
    started: str
    seconds: float
    has_oracle: bool

    def usage(self, concept):
        """Reference answers that use the concept (consistent ones only)."""
        return sum(1 for r in self.reference.runs
                   if r.consistent and concept in r.ops_used)

    def matches_reference(self, concept):
        """Cases where X alone is as good as the whole set: consistent where
        the reference is, and (with an oracle) at least as accurate."""
        k = 0
        for a, ref in zip(self.isolated[concept].runs, self.reference.runs):
            if ref.consistent and not a.consistent:
                continue
            if self.has_oracle and a.scored and a.accuracy + 1e-9 < ref.accuracy:
                continue
            k += 1
        return k

    def unique(self, concept):
        """Cases the reference solves and the ablation of X does not."""
        return sum(1 for a, ref in zip(self.ablated[concept].runs,
                                       self.reference.runs)
                   if ref.consistent and not a.consistent)

    def order(self):
        def key(c):
            iso = self.isolated[c]
            return (-iso.consistent, -iso.mean_gen, -iso.recovered,
                    iso.seconds)
        return sorted(self.concepts, key=key)

    def as_dict(self):
        return {
            "started": self.started,
            "seconds": self.seconds,
            "profile": self.profile,
            "support": self.support,
            "concepts": self.concepts,
            "has_oracle": self.has_oracle,
            "ranking": [
                {"rank": i, "concept": c,
                 "usage_in_reference": self.usage(c),
                 "matches_reference": self.matches_reference(c),
                 "unique_cases": self.unique(c)}
                for i, c in enumerate(self.order(), 1)],
            "reference": self.reference.as_dict(),
            "isolated": {c: r.as_dict() for c, r in self.isolated.items()},
            "ablated": {c: r.as_dict() for c, r in self.ablated.items()},
        }


def rank_concepts(cases, cfg, oracles=None, progress=None):
    t0 = time.perf_counter()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    support, concepts = split_operators(cfg)
    all_ops = support + concepts
    ref = run_config("all", all_ops, cases, cfg, oracles, progress)
    iso = {c: run_config("only " + c, support + [c], cases, cfg, oracles,
                         progress) for c in concepts}
    abl = {c: run_config("without " + c, [o for o in all_ops if o != c],
                         cases, cfg, oracles, progress) for c in concepts}
    return Ranking(profile=cfg.profile if not cfg.operators else "explicit",
                   support=support, concepts=concepts, reference=ref,
                   isolated=iso, ablated=abl, started=started,
                   seconds=time.perf_counter() - t0,
                   has_oracle=bool(ref.scored))


# ---------------------------------------------------------------- report

def _pct(x):
    return "%.1f" % (100 * x)


def _delta(x, unit=""):
    if abs(x) < 0.05:
        return "0" + unit
    return ("%+.1f" % x) + unit


def render_report(rk):
    ref, n = rk.reference, rk.reference.n
    out = [R.header("CGL v0.1  —  concept ranking",
                    "%d cases, profile %s, %d atomic concepts, %s"
                    % (n, rk.profile, len(rk.concepts), rk.started))]

    out.append(R.section("Setup"))
    out.append(R.table([
        ("atomic concepts", ", ".join(rk.concepts)),
        ("support (always on)", ", ".join(rk.support)),
        ("runs", "%d configurations × %d cases = %d learner runs"
         % (1 + 2 * len(rk.concepts), n, (1 + 2 * len(rk.concepts)) * n)),
        ("oracle", "yes — scored over Σ^1..6" if rk.has_oracle
         else "no — generalisation columns are empty"),
        ("total time", "%.1f s" % rk.seconds),
    ]))

    out.append(R.section("Reference — every concept enabled"))
    out.append(R.table([
        ("consistent", "%d / %d" % (ref.consistent, n)),
        ("mean generalisation", _pct(ref.mean_gen) + "%"),
        ("  on passed cases", _pct(ref.mean_gen_solved) + "%"),
        ("mean unseen accuracy", _pct(ref.mean_unseen) + "%"),
        ("mean F1", "%.3f" % ref.mean_f1),
        ("recovered", "%d / %d" % (ref.recovered, len(ref.scored))),
        ("mean answer size", "%.1f nodes" % ref.mean_size),
        ("mean minimal alternatives", "%.1f" % ref.mean_alternatives),
        ("mean MDL cost", "%.1f bits" % ref.mean_cost),
        ("bank classes", R._thousands(ref.classes)),
        ("tuples formed", R._thousands(ref.tuples)),
        ("collapse", "%.1f×" % ref.collapse),
        ("time", "%.2f s" % ref.seconds),
    ]))

    # -- isolation -------------------------------------------------------
    out.append(R.section("Ranking — each concept in isolation  (support + X)"))
    rows = []
    for i, c in enumerate(rk.order(), 1):
        a = rk.isolated[c]
        rows.append((i, c, "%d/%d" % (a.consistent, n),
                     _pct(a.mean_gen),
                     _pct(a.mean_gen_solved) if a.consistent else "–",
                     _pct(a.mean_unseen),
                     "%.2f" % a.mean_f1, a.recovered,
                     "%d/%d" % (rk.matches_reference(c), n),
                     "%.1f" % a.mean_size if a.consistent else "–",
                     "%.1f" % a.mean_alternatives if a.consistent else "–",
                     R._thousands(a.classes), "%.1f×" % a.collapse,
                     "%.2f" % a.seconds,
                     "%.0f%%" % (100 * a.seconds / ref.seconds)
                     if ref.seconds else "–"))
    out.append(R.table(rows, headers=(
        "#", "concept", "cons", "gen%", "gen%✓", "unseen%", "F1", "rec",
        "=ref", "size", "ambig", "classes", "coll", "sec", "time/ref"),
        aligns=["r", "l", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r",
                "r", "r", "r"]))

    # -- ablation --------------------------------------------------------
    out.append(R.section("Necessity — each concept removed  (all − X)"))
    rows = []
    for c in sorted(rk.concepts,
                    key=lambda c: (rk.ablated[c].mean_gen - ref.mean_gen,
                                   rk.ablated[c].consistent)):
        a = rk.ablated[c]
        rows.append((c, "%d/%d" % (a.consistent, n),
                     _delta(a.consistent - ref.consistent),
                     _pct(a.mean_gen),
                     _delta(100 * (a.mean_gen - ref.mean_gen), " pt"),
                     _delta(a.recovered - ref.recovered),
                     rk.unique(c),
                     "%d/%d" % (rk.usage(c), ref.consistent),
                     "%.2f" % a.seconds,
                     _delta(100 * (a.seconds / ref.seconds - 1), "%")
                     if ref.seconds else "–"))
    out.append(R.table(rows, headers=(
        "concept", "cons", "Δcons", "gen%", "Δgen", "Δrec", "unique",
        "used", "sec", "Δtime"),
        aligns=["l", "r", "r", "r", "r", "r", "r", "r", "r", "r"]))

    # -- per case --------------------------------------------------------
    out.append(R.section("Per case — which isolated concepts pass"))
    rows = []
    for j, ref_run in enumerate(ref.runs):
        passing = [c for c in rk.order() if rk.isolated[c].runs[j].consistent]
        best = max(passing, key=lambda c: rk.isolated[c].runs[j].accuracy,
                   default=None)
        b = rk.isolated[best].runs[j] if best else None
        rows.append((ref_run.case,
                     "✓" if ref_run.consistent else "✗",
                     _pct(ref_run.accuracy) if ref_run.scored else "–",
                     ("%s %s" % (best, _pct(b.accuracy))) if b and b.scored
                     else "–",
                     len(passing),
                     ", ".join(passing) if passing else "—"))
    out.append(R.table(rows, headers=(
        "case", "ref", "ref gen%", "best alone", "#pass", "passing alone"),
        aligns=["l", "l", "r", "l", "r", "l"]))

    # -- glossary --------------------------------------------------------
    out.append(R.section("Metrics"))
    out.append(R.dim("""\
  CGL's own (§16.4, §19.4, §21):
    cons      cases with an answer making 0 training errors
    size      mean tree size of the answer (nodes), passed cases only
    ambig     mean number of distinct smallest consistent expressions
              (§19.4, capped at 10); 1.0 = the data identifies the answer
    classes   value-bank classes built over all cases (§21)
    coll      collapse = tuples formed / classes kept (§21.7)
    sec       wall time over all cases
  Against the oracle (§19.2):
    gen%      mean accuracy over every string of length 1–6; a failed case
              has no hypothesis and is scored as answering false everywhere
    gen%✓     the same, over the passed cases only
    unseen%   the same, on strings not in the training set
    F1        mean F1 of the positive class
    rec       cases recovered exactly (100% on the universe)
  Proposed for ranking:
    =ref      cases where X alone is as good as the whole set
              (consistent where the reference is, and no less accurate)
    time/ref  isolated run time as a share of the reference — efficiency
    Δ*        change against the reference when X is removed
    unique    cases the reference solves that become unsolvable without X
    used      reference answers whose graph contains X
  Order: isolation table by cons, then gen%, rec, sec; necessity by Δgen.
"""))
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- saving

def save(rk, folder):
    """Write `<folder>/rank_concepts_<date>_<time>.txt` and `.json`."""
    os.makedirs(folder, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S",
                          time.strptime(rk.started, "%Y-%m-%d %H:%M:%S"))
    base = os.path.join(folder, "rank_concepts_%s" % stamp)
    was = R.colour_on()
    R.set_colour(False)
    try:
        text = render_report(rk)
    finally:
        R.set_colour(was)
    with open(base + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(rk.as_dict(), fh, indent=2, ensure_ascii=False)
    return base + ".txt", base + ".json"
