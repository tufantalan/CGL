"""CGL v0.1 --- the language layer.

Implements Sections 11 to 13 of SoftwareDescription_Rev2: the four types, the
bottom value, and the operator set.  Nothing in this module knows what a label
is, what a hypothesis is, or how search works (Section 17.1: dependencies point
strictly downward).

Every operator is total and pure (Sections 3.5, 3.6).  Purity is what makes the
observational-equivalence collapse of Section 4.3 --- and therefore the whole
value bank of Section 21 --- sound.  Do not add a stateful or randomised
operator here.
"""

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple


# ---------------------------------------------------------------- bottom

class _Bottom:
    """The undefined value of Section 3.5.  A singleton, so `is` works."""
    __slots__ = ()
    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self):
        return "⊥"

    def __reduce__(self):
        return (_Bottom, ())


BOT = _Bottom()


# ---------------------------------------------------------------- types

STR, CHAR, INT, BOOL = "Str", "Char", "Int", "Bool"
TYPES = (STR, CHAR, INT, BOOL)


# ---------------------------------------------------------------- operators

@dataclass(frozen=True)
class Operator:
    symbol: str
    params: Tuple[str, ...]
    result: str
    fn: Callable          # applied only to non-bottom arguments
    family: str           # for operator profiles and reporting
    infix: str = ""       # rendered infix when binary and set

    @property
    def arity(self) -> int:
        return len(self.params)

    def apply(self, args: Sequence):
        try:
            return self.fn(*args)
        except Exception:
            return BOT


def _maxrun(s, c):
    best = run = 0
    for ch in s:
        run = run + 1 if ch == c else 0
        if run > best:
            best = run
    return best


def _indexof(s, w):
    if w == "":
        return BOT
    i = s.find(w)
    return BOT if i < 0 else i


def _dec(s, max_len):
    if len(s) > max_len:
        return BOT
    return int(s, 2) if s else 0


def _charat(s, n):
    return s[n] if 0 <= n < len(s) else BOT


def _prefix(s, n):
    return BOT if n < 0 else s[:n]


def _suffix(s, n):
    if n < 0:
        return BOT
    return s[len(s) - n:] if n else ""


def _trimhead(s, n):
    return BOT if n < 0 else s[n:]


def _trimtail(s, n):
    if n < 0:
        return BOT
    return s[:len(s) - n] if n else s


def _div(a, b):
    return BOT if b == 0 else a // b


def _mod(a, b):
    return BOT if b == 0 else a % b


# ------------------------------------------------------------ the registry

# Section 17.2: the registry is the extension point.  Adding an operator is one
# entry here and no other edit (requirement N3).

def build_operators(sigma="01", K=8, l_str=3, max_dec_len=64,
                    training=(), profile="core", explicit=None):
    """Return the operator list for one run.

    `training` is the set of training strings; Section 13.10 requires the STR
    constant pool to exclude every one of them, so that memorisation by
    `STREQ(INPUT, <a training string>)` is not available at all.
    """
    banned = set(training)
    ops = []

    def add(symbol, params, result, fn, family, infix=""):
        ops.append(Operator(symbol, tuple(params), result, fn, family, infix))

    # --- 13.1 structural ------------------------------------------------
    add("INPUT", (), STR, lambda: None, "input")        # value supplied by the bank

    # --- 13.2 constants -------------------------------------------------
    for n in range(0, K + 1):
        add("INT[%d]" % n, (), INT, (lambda v: (lambda: v))(n), "int_const")
    for c in sigma:
        add("CHR[%s]" % c, (), CHAR, (lambda v: (lambda: v))(c), "chr_const")
    pool = []
    for length in range(1, l_str + 1):
        stack = [""]
        for _ in range(length):
            stack = [p + c for p in stack for c in sigma]
        pool.extend(stack)
    for w in pool:
        if w in banned:
            continue                                   # 13.10 guard 1
        add('STR["%s"]' % w, (), STR, (lambda v: (lambda: v))(w), "str_const")
    add("TRUE", (), BOOL, lambda: True, "bool_const")
    add("FALSE", (), BOOL, lambda: False, "bool_const")

    # --- 13.3 string to string ------------------------------------------
    add("TRIMHEAD", (STR, INT), STR, _trimhead, "str2str")
    add("TRIMTAIL", (STR, INT), STR, _trimtail, "str2str")
    add("PREFIX", (STR, INT), STR, _prefix, "str2str")
    add("SUFFIX", (STR, INT), STR, _suffix, "str2str")
    add("REVERSE", (STR,), STR, lambda s: s[::-1], "str2str")

    # --- 13.4 string to integer -----------------------------------------
    add("LEN", (STR,), INT, len, "str2int")
    add("COUNT", (STR, CHAR), INT, lambda s, c: s.count(c), "str2int")
    add("COUNTSUB", (STR, STR), INT,
        lambda s, w: BOT if w == "" else s.count(w), "str2int_x")
    add("MAXRUN", (STR, CHAR), INT, _maxrun, "str2int")
    add("INDEXOF", (STR, STR), INT, _indexof, "str2int_x")
    add("DEC", (STR,), INT, lambda s: _dec(s, max_dec_len), "arith")

    # --- 13.5 string to character ---------------------------------------
    add("HEAD", (STR,), CHAR, lambda s: s[0] if s else BOT, "str2chr")
    add("LAST", (STR,), CHAR, lambda s: s[-1] if s else BOT, "str2chr")
    add("CHARAT", (STR, INT), CHAR, _charat, "str2chr_x")

    # --- 13.6 string predicates -----------------------------------------
    add("STARTSWITH", (STR, STR), BOOL, lambda s, w: s.startswith(w), "strpred")
    add("ENDSWITH", (STR, STR), BOOL, lambda s, w: s.endswith(w), "strpred")
    add("CONTAINS", (STR, STR), BOOL, lambda s, w: w in s, "strpred")
    add("STREQ", (STR, STR), BOOL, lambda s, w: s == w, "strpred_x")

    # --- 13.7 arithmetic --------------------------------------------------
    add("ADD", (INT, INT), INT, lambda a, b: a + b, "arith", "+")
    add("SUB", (INT, INT), INT, lambda a, b: a - b, "arith", "-")
    add("MUL", (INT, INT), INT, lambda a, b: a * b, "arith", "*")
    add("DIV", (INT, INT), INT, _div, "arith", "/")
    add("MOD", (INT, INT), INT, _mod, "arith", "%")
    add("MIN", (INT, INT), INT, min, "arith")
    add("MAX", (INT, INT), INT, max, "arith")
    add("ABS", (INT,), INT, abs, "arith")

    # --- 13.8 comparison and boolean --------------------------------------
    add("EQ", (INT, INT), BOOL, lambda a, b: a == b, "cmp", "=")
    add("NEQ", (INT, INT), BOOL, lambda a, b: a != b, "cmp", "!=")
    add("LT", (INT, INT), BOOL, lambda a, b: a < b, "cmp", "<")
    add("LE", (INT, INT), BOOL, lambda a, b: a <= b, "cmp", "<=")
    add("GT", (INT, INT), BOOL, lambda a, b: a > b, "cmp", ">")
    add("GE", (INT, INT), BOOL, lambda a, b: a >= b, "cmp", ">=")
    add("CHREQ", (CHAR, CHAR), BOOL, lambda a, b: a == b, "cmp", "=")
    add("AND", (BOOL, BOOL), BOOL, lambda a, b: a and b, "bool", "and")
    add("OR", (BOOL, BOOL), BOOL, lambda a, b: a or b, "bool", "or")
    add("XOR", (BOOL, BOOL), BOOL, lambda a, b: a != b, "bool", "xor")
    add("NOT", (BOOL,), BOOL, lambda a: not a, "bool")

    if explicit is not None:
        keep = set(explicit)
        return [o for o in ops
                if o.symbol in keep or o.symbol.split("[")[0] in keep]
    return [o for o in ops if o.family in PROFILES[profile]]


# Section 18: the enabled operator set is configuration.  Three profiles ship.
#
# `core` deliberately omits the boolean connectives.  Layer 2 (Section 16.3)
# already builds conjunctions and takes negations of pool predicates, so putting
# AND / OR / NOT into Layer 1 duplicates that work and is where the enumeration
# blows up.  It also omits arithmetic and the string-to-string operators, which
# means Str, Char and Int all stop growing after size 3 and only Bool keeps
# expanding --- the bank genuinely saturates, which is what makes an exhaustive
# search to size 8 finish in seconds.
#
# `boolean` adds the connectives back, for measuring the cost of doing Layer 2's
# job inside Layer 1.  `full` is every operator of Section 13 except DFA, and is
# the configuration in which the growth of Section 21.6 is worth measuring.

def logical_ops():
    """AND / OR / NOT as graph nodes.

    Layer 2 assembles a DNF out of pool predicates, so it needs these whether or
    not they are enabled for Layer 1's enumeration.  They are the same operators
    of Section 13.8, obtained here without going through a profile.
    """
    return {
        "AND": Operator("AND", (BOOL, BOOL), BOOL,
                        lambda a, b: a and b, "bool", "and"),
        "OR": Operator("OR", (BOOL, BOOL), BOOL,
                       lambda a, b: a or b, "bool", "or"),
        "NOT": Operator("NOT", (BOOL,), BOOL, lambda a: not a, "bool"),
    }


PROFILES = {
    "core": {"input", "int_const", "chr_const", "str_const", "bool_const",
             "str2int", "str2chr", "strpred", "cmp"},
    "boolean": {"input", "int_const", "chr_const", "str_const", "bool_const",
                "str2int", "str2chr", "strpred", "cmp", "bool"},
    "full": {"input", "int_const", "chr_const", "str_const", "bool_const",
             "str2str", "str2int", "str2int_x", "str2chr", "str2chr_x",
             "strpred", "strpred_x", "arith", "cmp", "bool"},
}
