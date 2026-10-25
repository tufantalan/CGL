#!/usr/bin/env python3
"""CGL v0.1 --- client.

Run with no arguments for an interactive menu; run with flags to script it.

    python3 cgl_client.py                       menu
    python3 cgl_client.py --list                list the test cases found
    python3 cgl_client.py --case 24             run one case by number or name
    python3 cgl_client.py --file path/to.txt    run a case from any file
    python3 cgl_client.py --all                 run every case, summary table
    python3 cgl_client.py --case 6 --profile full --max-size 7

Stdlib only.  Colour is switched off automatically when output is redirected.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from cgl import report as R                                    # noqa: E402
from cgl.learn import Config, learn                            # noqa: E402
from cgl.lang import PROFILES                                  # noqa: E402
from cgl import testcases as TC                                # noqa: E402
from cgl import oracle as O                                    # noqa: E402


# ---------------------------------------------------------------- helpers

def find_case(cases, token):
    token = str(token).strip()
    for c in cases:
        if c.name == token:
            return c
    if token.isdigit():
        want = "example_" + token
        for c in cases:
            if c.name == want:
                return c
        idx = int(token)
        if 1 <= idx <= len(cases):
            return cases[idx - 1]
    matches = [c for c in cases if token.lower() in c.name.lower()]
    return matches[0] if len(matches) == 1 else None


def list_cases(cases, root):
    print(R.section("Test cases in %s" % root))
    if not cases:
        print("  " + R.yellow("none found — use --testcases PATH, or build one "
                              "from the menu"))
        return
    rows = []
    for i, c in enumerate(cases, 1):
        gloss = c.gloss[:44] + ("…" if len(c.gloss) > 44 else "")
        rows.append((i, c.name, len(c), len(c.positives), len(c.negatives),
                     "%d–%d" % (min(map(len, c.strings)),
                                max(map(len, c.strings))), gloss))
    print(R.table(rows, headers=("#", "name", "n", "+", "−", "len", "intent"),
                  aligns=["r", "l", "r", "r", "r", "r", "l"]))


def run_case(case, cfg, args, oracles=None):
    live = not args.quiet and sys.stdout.isatty()

    def progress(bank, size):
        sys.stdout.write("\r  " + R.dim("building bank … size %d, %d classes"
                                        % (size, len(bank.tau))))
        sys.stdout.flush()

    res = learn(case, cfg, progress=progress if live else None)
    if live:
        sys.stdout.write("\r" + " " * 60 + "\r")
    sc = None if args.no_oracle else O.score_result(case, res, oracles)
    print(R.full_report(case, res, score=sc,
                        show_bank=not args.no_bank,
                        show_ident=not args.no_identifiability))
    if args.dot:
        if res.graph is not None:
            from cgl.graph import to_dot
            with open(args.dot, "w", encoding="utf-8") as fh:
                fh.write(to_dot(res.graph))
            print("  " + R.dim("graph written to %s" % args.dot))
    return res


def run_all(cases, cfg, args, oracles=None):
    print(R.header("CGL v0.1 — all test cases",
                   "%d cases, profile %s, maxSize %d"
                   % (len(cases), cfg.profile, cfg.max_size)))
    live = not args.quiet and sys.stdout.isatty()
    rows, solved, scores = [], 0, []
    for c in cases:
        if live:
            sys.stdout.write("\r  " + R.dim("running %s …" % c.name))
            sys.stdout.flush()
        res = learn(c, cfg)
        sc = None if args.no_oracle else O.score_result(c, res, oracles)
        if sc is not None:
            scores.append(sc)
        rows.append(R.summary_row(c, res, sc))
        if res.kind != "none" and res.errors == 0:
            solved += 1
    if live:
        sys.stdout.write("\r" + " " * 60 + "\r")
    headers = ["", "case", "form", "size", "err", "sec"]
    aligns = ["l", "l", "l", "r", "r", "r"]
    if scores:
        headers += ["gen%", "rec"]
        aligns += ["r", "l"]
    headers.append("hypothesis")
    aligns.append("l")
    print(R.table(rows, headers=tuple(headers), aligns=aligns))
    print("\n  " + R.bold("%d of %d consistent" % (solved, len(cases))))
    if scores:
        mean = sum(s.accuracy for s in scores) / len(scores)
        rec = sum(1 for s in scores if s.recovered)
        print(R.baseline_line(mean, rec, len(scores),
                              [O.CTL_V2_BASELINE, O.CTL_V1_BASELINE]))
        missed = [s for s in scores if not s.recovered]
        if missed:
            print("\n  " + R.dim("not recovered: ") + ", ".join(
                "%s (%.0f%%)" % (s.concept or "?", 100 * s.accuracy)
                for s in sorted(missed, key=lambda x: x.accuracy)))
    print("")


# ---------------------------------------------------------------- entry input

def build_case_interactively(sigma="01"):
    print(R.section("New test case"))
    print(R.dim("  One example per line: a binary string, a space, then + or -."))
    print(R.dim("  Blank line to finish, 'undo' drops the last, 'cancel' aborts."))
    print("")
    lines = []
    while True:
        try:
            raw = input("  %3d> " % (len(lines) + 1)).strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            break
        if not raw:
            break
        low = raw.lower()
        if low == "cancel":
            return None
        if low == "undo":
            if lines:
                print("      " + R.dim("dropped %r" % lines.pop()))
            continue
        raw = raw.replace("\u2212", "-")
        if TC._LINE.match(raw) is None:
            print("      " + R.red("expected '<string> +' or '<string> -'"))
            continue
        word = raw.split()[0]
        bad = set(word) - set(sigma)
        if bad:
            print("      " + R.red("symbol(s) %s outside the alphabet %r"
                                   % ("".join(sorted(bad)), sigma)))
            continue
        earlier = [l for l in lines if l.split()[0] == word]
        if earlier:
            print("      " + R.yellow("%r already entered as %r; replacing"
                                      % (word, earlier[-1])))
            lines = [l for l in lines if l.split()[0] != word]
        lines.append(raw)
    if not lines:
        return None
    try:
        return TC.parse("\n".join(lines), name="entered")
    except TC.ParseError as e:
        print("  " + R.red(str(e)))
        return None


def menu(cases, root, cfg, args, oracles=None):
    while True:
        print(R.header("CGL v0.1  —  concept graph learner",
                       "%d test cases in %s" % (len(cases), root)))
        print("  1  list and pick a test case")
        print("  2  build a new test case, line by line")
        print("  3  run every test case")
        print("  4  settings   " + R.dim("(profile %s, maxSize %d, timeout %.0fs)"
                                         % (cfg.profile, cfg.max_size, cfg.timeout)))
        print("  5  quit")
        try:
            choice = input("\n  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("")
            return
        if choice in ("5", "q", "quit", "exit"):
            return
        if choice == "1":
            list_cases(cases, root)
            if not cases:
                continue
            try:
                token = input("\n  case (number, name, or blank to go back) > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("")
                continue
            if not token:
                continue
            c = find_case(cases, token)
            if c is None:
                print("  " + R.red("no such case: %r" % token))
                continue
            run_case(c, cfg, args, oracles)
        elif choice == "2":
            case = build_case_interactively(cfg.sigma)
            if case is None:
                print("  " + R.dim("cancelled"))
                continue
            run_case(case, cfg, args, oracles)
            try:
                keep = input("  save this case under %s? name (blank = no) > "
                             % root).strip()
            except (EOFError, KeyboardInterrupt):
                keep = ""
            if keep:
                path = TC.save(case, root, keep)
                print("  " + R.green("saved to %s" % path))
                cases = TC.scan(root, sigma=cfg.sigma)
        elif choice == "3":
            run_all(cases, cfg, args, oracles)
        elif choice == "4":
            cfg = settings(cfg)
        else:
            print("  " + R.red("pick 1–5"))


def settings(cfg):
    print(R.section("Settings  (§18)"))
    print(R.table(cfg.as_rows()))
    print(R.dim("\n  profiles: " + ", ".join(sorted(PROFILES))))
    fields = [("profile", str), ("max_size", int), ("timeout", float),
              ("pool_cap", int), ("p_max", int), ("K", int), ("l_str", int),
              ("str_guard", lambda v: v.strip().lower() in ("1", "on", "true",
                                                            "yes"))]
    print(R.dim("  blank keeps the current value"))
    for name, cast in fields:
        cur = getattr(cfg, name)
        try:
            raw = input("  %-9s [%s] > " % (name, cur)).strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            break
        if not raw:
            continue
        try:
            val = cast(raw)
        except ValueError:
            print("    " + R.red("not a %s" % cast.__name__))
            continue
        if name == "profile" and val not in PROFILES:
            print("    " + R.red("unknown profile"))
            continue
        setattr(cfg, name, val)
    return cfg


# ---------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(
        prog="cgl_client.py",
        description="CGL v0.1 — run the concept graph learner on a test case.")
    p.add_argument("--testcases", default=None,
                   help="folder to scan (default: ../../CTL/TestCases)")
    p.add_argument("--case", help="run one case by number or folder name")
    p.add_argument("--file", help="run one case from a file")
    p.add_argument("--all", action="store_true", help="run every case found")
    p.add_argument("--list", action="store_true", help="list the cases and exit")
    p.add_argument("--profile", default="core", choices=sorted(PROFILES),
                   help="operator profile (§18); default core")
    p.add_argument("--ops", help="explicit operator list, comma separated")
    p.add_argument("--max-size", type=int, default=8)
    p.add_argument("--pool-cap", type=int, default=200000)
    p.add_argument("--p-max", type=int, default=8)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--dot", help="write the answer graph to this .dot file")
    p.add_argument("--no-bank", action="store_true",
                   help="omit the value-bank statistics")
    p.add_argument("--no-identifiability", action="store_true")
    p.add_argument("--no-str-guard", action="store_true",
                   help="keep training strings in the STR pool (§13.10 guard 1 "
                        "off); costs nothing when STREQ is disabled")
    p.add_argument("--no-oracle", action="store_true",
                   help="skip scoring against CTL/TestCases/oracles.py")
    p.add_argument("--universe", type=int, default=6,
                   help="longest string in the scoring universe (default 6)")
    p.add_argument("--no-colour", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.no_colour or not sys.stdout.isatty():
        R.set_colour(False)

    cfg = Config(max_size=args.max_size, pool_cap=args.pool_cap,
                 p_max=args.p_max, timeout=args.timeout, profile=args.profile,
                 str_guard=not args.no_str_guard,
                 operators=tuple(args.ops.split(",")) if args.ops else None)

    root = args.testcases or TC.default_root(HERE)
    cases = TC.scan(root, sigma=cfg.sigma)
    oracles = None if args.no_oracle else O.load(root, max_len=args.universe)

    if args.file:
        try:
            case = TC.load(args.file, sigma=cfg.sigma)
        except (OSError, TC.ParseError) as e:
            print(R.red("cannot read %s: %s" % (args.file, e)))
            return 2
        run_case(case, cfg, args, oracles)
        return 0
    if args.list:
        list_cases(cases, root)
        return 0
    if args.all:
        if not cases:
            print(R.red("no test cases under %s" % root))
            return 2
        run_all(cases, cfg, args, oracles)
        return 0
    if args.case:
        c = find_case(cases, args.case)
        if c is None:
            print(R.red("no such case: %r (try --list)" % args.case))
            return 2
        run_case(c, cfg, args, oracles)
        return 0

    menu(cases, root, cfg, args, oracles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
