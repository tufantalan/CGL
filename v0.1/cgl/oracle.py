"""CGL v0.1 --- generalisation against a ground-truth oracle.

Section 19.1/19.2.  This module is measurement only: it imports nothing from
the learner and the learner imports nothing from it, so scoring can never
influence a hypothesis.

The oracles live in `CTL/TestCases/oracles.py`, which maps a case number to an
executable predicate.  The universe is every binary string of length 1 to 6 ---
126 strings --- which is the universe the CTL figures were measured over, so
the numbers here sit directly beside CTL v2's 97.5% mean generalisation and
26-of-29 recovered.
"""

import importlib.util
import os
import re
from dataclasses import dataclass, field

from .graph import classify


@dataclass
class Oracles:
    predicates: dict
    names: dict
    universe: list
    path: str = ""

    def for_case(self, case_name):
        k = case_key(case_name)
        if k is None or k not in self.predicates:
            return None, ""
        return self.predicates[k], self.names.get(k, "")


def case_key(name):
    m = re.search(r"(\d+)\s*$", str(name))
    return int(m.group(1)) if m else None


def _universe(max_len=6, sigma="01"):
    out = []
    for n in range(1, max_len + 1):
        stack = [""]
        for _ in range(n):
            stack = [p + c for p in stack for c in sigma]
        out.extend(stack)
    return out


def load(root, max_len=6):
    """Import `oracles.py` from a TestCases folder.  Returns None if absent."""
    path = os.path.join(root, "oracles.py")
    if not os.path.isfile(path):
        return None
    try:
        spec = importlib.util.spec_from_file_location("_cgl_oracles", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        preds = dict(getattr(mod, "ORACLES"))
        names = dict(getattr(mod, "NAMES", {}))
        univ = (list(mod.universe(max_len)) if hasattr(mod, "universe")
                else _universe(max_len))
    except Exception:
        return None
    return Oracles(predicates=preds, names=names, universe=univ, path=path)


@dataclass
class Score:
    concept: str = ""
    n: int = 0
    accuracy: float = 0.0
    recovered: bool = False
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    unseen_accuracy: float = 0.0
    unseen_n: int = 0
    beyond_accuracy: float = 0.0
    beyond_n: int = 0
    disagreements: list = field(default_factory=list)

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def score(graph, oracle, universe, seen=(), max_train_len=None,
          concept="", max_disagreements=6):
    """Compare a hypothesis with its oracle over the whole universe.

    `recovered` means agreement on every string of the universe --- concept
    recovery, not consistency with the training sample.
    """
    s = Score(concept=concept, n=len(universe))
    seen = set(seen)
    right = unseen_right = unseen_n = beyond_right = beyond_n = 0
    for w in universe:
        pred = classify(graph, w) if graph is not None else False
        truth = bool(oracle(w))
        ok = pred == truth
        right += ok
        if pred and truth:
            s.tp += 1
        elif pred and not truth:
            s.fp += 1
        elif not pred and truth:
            s.fn += 1
        else:
            s.tn += 1
        if w not in seen:
            unseen_n += 1
            unseen_right += ok
        if max_train_len is not None and len(w) > max_train_len:
            beyond_n += 1
            beyond_right += ok
        if not ok and len(s.disagreements) < max_disagreements:
            s.disagreements.append((w, pred, truth))
    s.accuracy = right / len(universe) if universe else 0.0
    s.recovered = right == len(universe)
    s.unseen_n = unseen_n
    s.unseen_accuracy = unseen_right / unseen_n if unseen_n else 0.0
    s.beyond_n = beyond_n
    s.beyond_accuracy = beyond_right / beyond_n if beyond_n else 0.0
    return s


def score_result(case, res, oracles, max_len=6):
    """Score one learner result, or return None when no oracle applies."""
    if oracles is None:
        return None
    pred, name = oracles.for_case(case.name)
    if pred is None:
        return None
    return score(res.graph, pred, oracles.universe,
                 seen=case.strings,
                 max_train_len=max(len(s) for s in case.strings),
                 concept=name)


# The CTL v2 figures, measured over this same universe and recorded in
# PROJECT_STATE section 4.3/4.5.  Printed as a reference line, not recomputed.
CTL_V2_BASELINE = {"mean_generalisation": 0.975, "recovered": 26, "cases": 29,
                   "label": "CTL v2 (fixed + dedup, d=4)"}
CTL_V1_BASELINE = {"mean_generalisation": 0.955, "recovered": 26, "cases": 29,
                   "label": "CTL v1 (as published)"}
