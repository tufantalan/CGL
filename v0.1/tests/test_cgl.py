#!/usr/bin/env python3
"""CGL v0.1 --- test suite.

    python3 tests/test_cgl.py            (from the v0.1 folder)
    python3 -m unittest discover tests

Each test names the section of SoftwareDescription_Rev2 it checks.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cgl import testcases as TC                                   # noqa: E402
from cgl.bank import Bank                                         # noqa: E402
from cgl.graph import (ConceptGraph, classify, evaluate,          # noqa: E402
                       render)
from cgl.lang import (BOOL, BOT, CHAR, INT, STR, build_operators,  # noqa: E402
                      logical_ops)
from cgl.learn import Config, learn                               # noqa: E402


SAMPLES = ["", "0", "1", "01", "10", "111", "0000", "10110", "000111"]
VALUES = {STR: SAMPLES, CHAR: ["0", "1"], INT: [-3, -1, 0, 1, 2, 8, 64],
          BOOL: [True, False]}


def a_case(text, name="t"):
    return TC.parse(text, name=name)


# ------------------------------------------------------------------ 3.5, 3.6

class TestOperators(unittest.TestCase):
    """Sections 3.5 and 3.6: every operator is total and pure."""

    def setUp(self):
        self.ops = build_operators(profile="full")

    def test_totality(self):
        for op in self.ops:
            if op.arity == 0:
                continue
            grids = [VALUES[t] for t in op.params]
            for a in grids[0]:
                for b in (grids[1] if op.arity == 2 else [None]):
                    args = [a] if op.arity == 1 else [a, b]
                    v = op.apply(args)          # must not raise
                    if v is BOT:
                        continue
                    self.assertIn(type(v).__name__,
                                  ("str", "int", "bool"),
                                  "%s returned %r" % (op.symbol, v))

    def test_purity(self):
        for op in self.ops:
            if op.arity == 0:
                self.assertEqual(op.fn() if op.symbol != "INPUT" else None,
                                 op.fn() if op.symbol != "INPUT" else None)
                continue
            grids = [VALUES[t] for t in op.params]
            args = [grids[0][-1]] + ([grids[1][-1]] if op.arity == 2 else [])
            self.assertEqual(op.apply(args), op.apply(args), op.symbol)

    def test_result_type_declared(self):
        for op in self.ops:
            if op.arity == 0 or op.result != INT:
                continue
            grids = [VALUES[t] for t in op.params]
            for a in grids[0]:
                for b in (grids[1] if op.arity == 2 else [None]):
                    v = op.apply([a] if op.arity == 1 else [a, b])
                    if v is not BOT:
                        self.assertIsInstance(v, int, op.symbol)

    def test_str_pool_excludes_training(self):
        """Section 13.10 guard 1 --- no memorisation constant."""
        train = ("0", "10", "111")
        ops = build_operators(profile="full", training=train)
        pool = {o.fn() for o in ops if o.symbol.startswith("STR[")}
        for w in train:
            self.assertNotIn(w, pool)


# ---------------------------------------------------------------------- 10

class TestParser(unittest.TestCase):
    """Section 10: input format and its edge cases."""

    def test_basic(self):
        c = a_case("1 +\n0 -\n")
        self.assertEqual(c.strings, ["1", "0"])
        self.assertEqual(c.labels, [True, False])

    def test_comments_blank_and_header(self):
        c = a_case("===== Test Case 7\n\n# note\n11 +\n00 -\n")
        self.assertEqual(len(c), 2)

    def test_duplicate_dropped(self):
        self.assertEqual(len(a_case("1 +\n1 +\n0 -\n")), 2)

    def test_contradiction_rejected(self):
        with self.assertRaises(TC.ParseError):
            a_case("1 +\n1 -\n")

    def test_alphabet_enforced(self):
        with self.assertRaises(TC.ParseError):
            a_case("12 +\n0 -\n")

    def test_one_sided_rejected(self):
        with self.assertRaises(TC.ParseError):
            a_case("1 +\n11 +\n")
        with self.assertRaises(TC.ParseError):
            a_case("1 -\n11 -\n")

    def test_labels_are_separate(self):
        """Section 10.4 --- strings and labels are different lists."""
        c = a_case("1 +\n0 -\n")
        self.assertTrue(all(isinstance(s, str) for s in c.strings))
        self.assertTrue(all(isinstance(l, bool) for l in c.labels))


# ---------------------------------------------------------------------- 14

class TestGraphInvariants(unittest.TestCase):
    """Section 14.2 --- I1 to I4."""

    def ops(self):
        return {o.symbol: o for o in build_operators(profile="full")}

    def test_valid_graph_passes(self):
        o = self.ops()
        g = ConceptGraph()
        inp = g.add(o["INPUT"])
        ln = g.add(o["LEN"], [inp])
        k = g.add(o["INT[3]"])
        g.out = g.add(o["EQ"], [ln, k])
        self.assertTrue(g.check())
        self.assertEqual(g.size(), 4)

    def test_output_must_be_bool(self):
        o = self.ops()
        g = ConceptGraph()
        g.out = g.add(o["INPUT"])
        with self.assertRaises(AssertionError):
            g.check()

    def test_type_mismatch_rejected(self):
        o = self.ops()
        g = ConceptGraph()
        inp = g.add(o["INPUT"])
        g.out = g.add(o["NOT"], [inp])            # Str where Bool expected
        with self.assertRaises(AssertionError):
            g.check()

    def test_cycle_rejected(self):
        o = self.ops()
        g = ConceptGraph()
        n = g.add(o["TRUE"])
        m = g.add(o["NOT"], [n])
        g.args[n] = (m,)                          # forge a cycle
        g.op[n] = o["NOT"]
        g.out = m
        with self.assertRaises(AssertionError):
            g.check()

    def test_sharing_counted_once(self):
        """Section 14.3 --- DAG size."""
        o = self.ops()
        g = ConceptGraph()
        inp = g.add(o["INPUT"])
        h = g.add(o["HEAD"], [inp])
        t = g.add(o["LAST"], [inp])
        g.out = g.add(o["CHREQ"], [h, t])
        g.check()
        self.assertEqual(g.size(), 4)             # INPUT counted once, not twice

    def test_bottom_classifies_negative(self):
        """Section 12 rule 3."""
        o = self.ops()
        g = ConceptGraph()
        inp = g.add(o["INPUT"])
        h = g.add(o["HEAD"], [inp])
        c = g.add(o["CHR[1]"])
        g.out = g.add(o["CHREQ"], [h, c])
        g.check()
        self.assertIs(evaluate(g, ""), BOT)
        self.assertFalse(classify(g, ""))


# ---------------------------------------------------------------------- 21

class TestBank(unittest.TestCase):
    """Section 21 --- the value bank and its invariants B1 to B5."""

    def build(self, strings, profile="core", max_size=5):
        ops = build_operators(profile=profile, training=strings)
        return Bank(strings, ops).build(max_size=max_size)

    def test_b1_congruence(self):
        b = self.build(["1", "10", "011", "0000"])
        for cid in b.classes():
            for prod in b.prods[cid]:
                self.assertEqual(b.ev(*prod), b.beta[cid],
                                 "B1 violated at class %d" % cid)

    def test_b2_quotient(self):
        b = self.build(["1", "10", "011", "0000"])
        keys = {(b.tau[c], b.beta[c]) for c in b.classes()}
        self.assertEqual(len(keys), len(list(b.classes())))

    def test_b3_inhabitation(self):
        b = self.build(["1", "10", "011"])
        for cid in b.classes():
            self.assertTrue(b.prods[cid], "B3 violated at class %d" % cid)

    def test_b4_non_degeneracy(self):
        b = self.build(["1", "10", "011"], profile="full")
        for cid in b.classes():
            self.assertFalse(all(v is BOT for v in b.beta[cid]))

    def test_b5_mu_is_minimal(self):
        b = self.build(["1", "10", "011", "0000"])
        for cid in b.classes():
            best = min(1 + sum(b.mu[a] for a in args)
                       for _, args in b.prods[cid])
            self.assertEqual(b.mu[cid], best, "B5 violated at class %d" % cid)

    def test_extraction_reproduces_behaviour(self):
        """Section 21.5 --- an extracted graph evaluates to its class behaviour."""
        strings = ["1", "10", "011", "0000"]
        b = self.build(strings)
        for cid in list(b.classes())[:200]:
            g, _ = b.extract(cid)
            for j, s in enumerate(strings):
                self.assertEqual(evaluate(g, s), b.beta[cid][j])

    def test_dag_size_at_most_tree_size(self):
        """Section 21.5 --- size(EXTRACT(c)) <= mu(c), strict when shared."""
        b = self.build(["1", "10", "011", "0000"])
        strict = 0
        for cid in b.classes():
            g, _ = b.extract(cid)
            self.assertLessEqual(g.size(), b.mu[cid])
            strict += g.size() < b.mu[cid]
        self.assertGreater(strict, 0, "expected at least one shared extraction")

    def test_cyclic_productions_are_tolerated(self):
        """Section 21.3 --- the production graph may contain cycles."""
        # every training string is a palindrome, so REVERSE(INPUT) lands in the
        # class of INPUT and that class acquires a production consuming itself
        b = self.build(["1", "010", "0110", "11011"], profile="full", max_size=3)
        inp = b.index[(STR, tuple(b.S))]
        selfloop = any(inp in args for _, args in b.prods[inp])
        self.assertTrue(selfloop, "expected REVERSE(INPUT) to close a cycle")
        g, _ = b.extract(inp)                     # must still terminate
        self.assertEqual(g.size(), 1)

    def test_bank_is_label_blind(self):
        """Section 15.3 / 16.1 --- Layer 1 never sees a label."""
        strings = ["1", "10", "011", "0000"]
        b1 = self.build(strings)
        b2 = self.build(strings)
        self.assertEqual(sorted((b1.tau[c], b1.beta[c]) for c in b1.classes()),
                         sorted((b2.tau[c], b2.beta[c]) for c in b2.classes()))

    def test_class_ids_are_not_indices(self):
        """Section 22.7 requirement 2 --- identifiers are opaque."""
        b = self.build(["1", "10", "011"])
        ids = sorted(b.classes())
        self.assertNotEqual(ids, list(range(len(ids))))   # they start at 1
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_productions_recorded_for_existing_classes(self):
        """Section 22.7 requirement 1 --- RECORD runs on every insertion."""
        b = self.build(["1", "10", "011", "0000"])
        multi = [c for c in b.classes() if len(b.prods[c]) > 1]
        self.assertTrue(multi, "no class collected an alternative production")


# ---------------------------------------------------------------- 15.4, 16

class TestRendering(unittest.TestCase):

    def test_let_binding_appears_when_shared(self):
        o = {x.symbol: x for x in build_operators(profile="full")}
        g = ConceptGraph()
        inp = g.add(o["INPUT"])
        ln = g.add(o["LEN"], [inp])
        g.out = g.add(o["EQ"], [ln, ln])
        g.check()
        self.assertIn("let ", render(g))

    def test_plain_expression_has_no_let(self):
        o = {x.symbol: x for x in build_operators(profile="full")}
        g = ConceptGraph()
        inp = g.add(o["INPUT"])
        ln = g.add(o["LEN"], [inp])
        k = g.add(o["INT[3]"])
        g.out = g.add(o["EQ"], [ln, k])
        self.assertNotIn("let ", render(g))


class TestLearner(unittest.TestCase):
    """Sections 16.3 and 16.4 end to end."""

    def run_case(self, text, **kw):
        case = a_case(text)
        res = learn(case, Config(**kw))
        return case, res

    def test_single_predicate(self):
        case, res = self.run_case("1 +\n11 +\n0 -\n00 -\n")
        self.assertEqual(res.kind, "single")
        self.assertEqual(res.errors, 0)
        for s, lab in zip(case.strings, case.labels):
            self.assertEqual(classify(res.graph, s), lab)

    def test_conjunction_needs_layer_two(self):
        case, res = self.run_case(
            "1000 +\n1001 +\n1010 +\n0000 -\n0101 -\n100 -\n10000 -\n1 -\n")
        self.assertNotEqual(res.kind, "none")
        self.assertEqual(res.errors, 0)

    def test_disjunction_needs_layer_two(self):
        case, res = self.run_case(
            "1 +\n10 +\n000 +\n010 +\n01 -\n0011 -\n0000 -\n")
        self.assertNotEqual(res.kind, "none")
        self.assertEqual(res.errors, 0)

    def test_answer_satisfies_the_invariants(self):
        _, res = self.run_case("1 +\n11 +\n0 -\n00 -\n")
        self.assertTrue(res.graph.check())

    def test_identifiability_is_reported(self):
        _, res = self.run_case("1 +\n11 +\n0 -\n00 -\n")
        self.assertGreaterEqual(res.consistent_classes, 1)
        self.assertGreaterEqual(len(res.alternatives), 1)
        self.assertGreater(res.sample_needed, 0)

    def test_no_memorising_answer(self):
        """Section 13.10 --- the rote learner must not win."""
        case, res = self.run_case("1001 +\n0110 +\n1010 -\n0101 -\n")
        self.assertNotEqual(res.kind, "none")
        for _, formula, _ in ([(0, res.formula, 0)] + list(res.alternatives)):
            for s in case.strings:
                self.assertNotIn('STR["%s"]' % s, formula)

    def test_str_guard_is_configurable(self):
        """Section 13.10 guard 1, and the option that measures its cost."""
        text = "1 +\n10 +\n0 -\n01 -\n"
        on = learn(a_case(text), Config(str_guard=True))
        off = learn(a_case(text), Config(str_guard=False))
        self.assertNotIn('STR["1"]', on.formula)
        self.assertEqual(off.errors, 0)
        self.assertEqual(on.errors, 0)

    def test_budget_is_respected(self):
        _, res = self.run_case("1001 +\n0110 +\n1010 -\n0101 -\n",
                               max_size=3, profile="full")
        self.assertLessEqual(res.stopped_at, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
