"""CGL v0.1 --- input.

Section 10.  Parses a test case and, critically for Section 10.4, keeps the
strings and the labels in separate lists so that nothing below the learner can
see a label even by accident.
"""

import os
import re
from dataclasses import dataclass, field


class ParseError(ValueError):
    pass


@dataclass
class TestCase:
    name: str
    strings: list = field(default_factory=list)
    labels: list = field(default_factory=list)     # True = positive
    path: str = ""
    gloss: str = ""

    @property
    def positives(self):
        return [s for s, l in zip(self.strings, self.labels) if l]

    @property
    def negatives(self):
        return [s for s, l in zip(self.strings, self.labels) if not l]

    def __len__(self):
        return len(self.strings)


_LINE = re.compile(r"^\s*([01]+)\s+([+-])\s*$")


def parse(text, name="case", path="", sigma="01"):
    """Section 10.1/10.3.  Blank lines and `#` comments are ignored; an optional
    `===== Test Case <id>` header line is tolerated."""
    strings, labels, seen = [], [], {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("="):
            continue
        m = _LINE.match(line)
        if not m:
            raise ParseError("line %d: expected '<string> +|-', got %r"
                             % (lineno, raw))
        s, lab = m.group(1), (m.group(2) == "+")
        bad = set(s) - set(sigma)
        if bad:
            raise ParseError("line %d: symbol(s) %s outside the alphabet %r"
                             % (lineno, "".join(sorted(bad)), sigma))
        if s in seen:
            if seen[s] != lab:
                raise ParseError("line %d: %r is labelled both + and -"
                                 % (lineno, s))
            continue                       # 10.3: an exact duplicate is dropped
        seen[s] = lab
        strings.append(s)
        labels.append(lab)
    if not strings:
        raise ParseError("no examples found")
    if not any(labels):
        raise ParseError("no positive examples; nothing to learn")
    if all(labels):
        raise ParseError("no negative examples; every hypothesis is consistent")
    return TestCase(name=name, strings=strings, labels=labels, path=path)


def load(path, sigma="01"):
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    folder = os.path.dirname(os.path.abspath(path))
    name = os.path.basename(folder) or os.path.basename(path)
    case = parse(text, name=name, path=path, sigma=sigma)
    idea = os.path.join(folder, "idea.txt")
    if os.path.isfile(idea):
        with open(idea, "r", encoding="utf-8", errors="replace") as fh:
            case.gloss = " ".join(fh.read().split())
    return case


def _key(name):
    m = re.search(r"(\d+)$", name)
    return (0, int(m.group(1))) if m else (1, 0), name


def scan(root, sigma="01"):
    """Every readable test case under `root`, ordered numerically by name."""
    found = []
    if not os.path.isdir(root):
        return found
    for entry in sorted(os.listdir(root)):
        folder = os.path.join(root, entry)
        if os.path.isdir(folder):
            f = os.path.join(folder, "example.txt")
            if os.path.isfile(f):
                found.append((entry, f))
        elif entry.endswith(".txt") and entry != "idea.txt":
            found.append((os.path.splitext(entry)[0], folder))
    out = []
    for name, path in found:
        try:
            out.append(load(path, sigma=sigma))
        except (OSError, ParseError):
            continue
    out.sort(key=lambda c: _key(c.name))
    return out


def default_root(script_dir):
    """`CGL/<version>/` sits two levels below the project root, next to `CTL/`."""
    guesses = [
        os.path.join(script_dir, "..", "..", "CGL", "TestCases"),
        os.path.join(script_dir, "..", "CGL", "TestCases"),
        os.path.join(script_dir, "TestCases"),
    ]
    for g in guesses:
        g = os.path.normpath(g)
        if os.path.isdir(g):
            return g
    return os.path.normpath(guesses[0])


def save(case, root, name=None):
    """Write a hand-built case as `<root>/<name>/example.txt`."""
    name = name or case.name
    folder = os.path.join(root, name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "example.txt")
    with open(path, "w", encoding="utf-8") as fh:
        for s, lab in zip(case.strings, case.labels):
            fh.write("%s %s\n" % (s, "+" if lab else "-"))
    case.path = path
    return path
