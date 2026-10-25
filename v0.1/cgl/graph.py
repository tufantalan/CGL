"""CGL v0.1 --- hypothesis representation, semantics and rendering.

Sections 14 and 15.  A concept graph is a DAG of operator nodes with a single
Bool output; its denotation is a total function Sigma* -> {true, false}.
"""

from .lang import BOT, BOOL, Operator


class ConceptGraph:
    """Section 14.1: G = (N, op, args, out), with nodes identified by integer."""

    __slots__ = ("op", "args", "out", "_next")

    def __init__(self):
        self.op = {}        # node id -> Operator
        self.args = {}      # node id -> tuple of node ids
        self.out = None
        self._next = 0

    # -- construction ----------------------------------------------------

    def add(self, op: Operator, args=()):
        n = self._next
        self._next += 1
        self.op[n] = op
        self.args[n] = tuple(args)
        return n

    # -- structure -------------------------------------------------------

    def reachable(self, root=None):
        root = self.out if root is None else root
        seen, stack = set(), [root]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(self.args[n])
        return seen

    def size(self):
        """Section 14.3 --- DAG size; a shared node is counted once."""
        return len(self.reachable())

    def strip_unreachable(self):
        keep = self.reachable()
        for n in list(self.op):
            if n not in keep:
                del self.op[n]
                del self.args[n]
        return self

    def parents(self):
        cnt = {}
        for n in self.reachable():
            for a in self.args[n]:
                cnt[a] = cnt.get(a, 0) + 1
        return cnt

    # -- invariants I1..I4 ------------------------------------------------

    def check(self):
        """Raise on any violation of Section 14.2.  Cheap; run it always."""
        assert self.out in self.op, "I3: no output node"
        # I1 acyclicity, by a colouring DFS
        colour = {}

        def visit(n):
            c = colour.get(n, 0)
            if c == 1:
                raise AssertionError("I1: cycle through node %d" % n)
            if c == 2:
                return
            colour[n] = 1
            for a in self.args[n]:
                visit(a)
            colour[n] = 2

        visit(self.out)
        # I2 well-typedness
        for n in self.reachable():
            op = self.op[n]
            a = self.args[n]
            assert len(a) == op.arity, "I2: arity mismatch at %s" % op.symbol
            for i, child in enumerate(a):
                assert self.op[child].result == op.params[i], (
                    "I2: %s argument %d expects %s, got %s"
                    % (op.symbol, i + 1, op.params[i], self.op[child].result))
        # I3 output type
        assert self.op[self.out].result == BOOL, "I3: output is not Bool"
        # I4 reachability
        assert set(self.op) == self.reachable(), "I4: unreachable nodes present"
        return True


# ---------------------------------------------------------------- semantics

def evaluate(g: ConceptGraph, s: str):
    """Section 15.2 --- EVAL with memoisation.  O(size(G)) applications."""
    memo = {}

    def value(n):
        if n in memo:
            return memo[n]
        op = g.op[n]
        if op.symbol == "INPUT":
            v = s
        elif op.arity == 0:
            v = op.fn()
        else:
            vs = [value(a) for a in g.args[n]]
            v = BOT if any(x is BOT for x in vs) else op.apply(vs)
        memo[n] = v
        return v

    return value(g.out)


def classify(g: ConceptGraph, s: str) -> bool:
    """Section 15.1 --- bottom and false both classify negative."""
    return evaluate(g, s) is True


# ---------------------------------------------------------------- rendering

def render(g: ConceptGraph, max_inline_parents=1) -> str:
    """Section 15.4 --- readable formula, with let-bindings for shared nodes."""
    shared = [n for n, c in g.parents().items()
              if c > max_inline_parents and g.op[n].arity > 0]
    order = _topo(g)
    names = {}
    for i, n in enumerate(x for x in order if x in shared):
        names[n] = "r%d" % (i + 1)

    def expr(n, top=False):
        if n in names and not top:
            return names[n]
        op = g.op[n]
        if op.arity == 0:
            return op.symbol
        kids = [expr(a) for a in g.args[n]]
        if op.arity == 2 and op.infix:
            return "(%s %s %s)" % (kids[0], op.infix, kids[1])
        return "%s(%s)" % (op.symbol, ", ".join(_unwrap(k) for k in kids))

    body = expr(g.out, top=(g.out in names))
    if not names:
        return _unwrap(body)
    lets = ", ".join("%s = %s" % (names[n], expr(n, top=True))
                     for n in order if n in names)
    return "let %s in %s" % (lets, _unwrap(body))


def _unwrap(s):
    if s.startswith("(") and s.endswith(")"):
        depth = 0
        for i, ch in enumerate(s):
            depth += (ch == "(") - (ch == ")")
            if depth == 0 and i < len(s) - 1:
                return s
        return s[1:-1]
    return s


def _topo(g):
    out, seen = [], set()

    def visit(n):
        if n in seen:
            return
        seen.add(n)
        for a in g.args[n]:
            visit(a)
        out.append(n)

    visit(g.out)
    return out


def to_dot(g: ConceptGraph) -> str:
    """Section 20.2 --- Graphviz, with argument position on the edge."""
    lines = ["digraph G {", '  node [shape=box, fontname="monospace"];']
    for n in sorted(g.reachable()):
        lines.append('  n%d [label="%s"];' % (n, g.op[n].symbol.replace('"', '\\"')))
    for n in sorted(g.reachable()):
        for i, a in enumerate(g.args[n]):
            lines.append('  n%d -> n%d [label="%d"];' % (n, a, i + 1))
    lines.append('  n%d [peripheries=2];' % g.out)
    lines.append("}")
    return "\n".join(lines)
