"""CGL v0.1 --- the report.

Section 20.4.  Terminal output only; stdlib only.  Colour is ANSI and is
switched off automatically when stdout is not a terminal.
"""

import os
import sys

from .graph import render

_ON = sys.stdout.isatty() and os.environ.get("TERM") not in (None, "dumb")


def set_colour(on):
    global _ON
    _ON = bool(on)


def colour_on():
    return _ON


def _c(code, s):
    return "\x1b[%sm%s\x1b[0m" % (code, s) if _ON else str(s)


def bold(s):    return _c("1", s)
def dim(s):     return _c("2", s)
def cyan(s):    return _c("36", s)
def green(s):   return _c("32", s)
def yellow(s):  return _c("33", s)
def red(s):     return _c("31", s)
def blue(s):    return _c("34", s)


WIDTH = 78


def rule(ch="─"):
    return dim(ch * WIDTH)


def header(title, subtitle=""):
    out = ["", bold(cyan("╭" + "─" * (WIDTH - 2) + "╮"))]
    out.append(bold(cyan("│")) + " " + bold(title).ljust(WIDTH - 3 + len(bold(title)) - len(title))
               + bold(cyan("│")))
    if subtitle:
        pad = WIDTH - 3 - len(subtitle)
        out.append(bold(cyan("│")) + " " + dim(subtitle) + " " * max(0, pad)
                   + bold(cyan("│")))
    out.append(bold(cyan("╰" + "─" * (WIDTH - 2) + "╯")))
    return "\n".join(out)


def section(title):
    return "\n" + bold(title) + "\n" + rule()


def table(rows, headers=None, aligns=None, indent=2):
    rows = [[str(c) for c in r] for r in rows]
    if not rows and not headers:
        return ""
    ncol = max(len(r) for r in ([headers] if headers else []) + rows)
    for r in rows:
        r.extend([""] * (ncol - len(r)))
    widths = [0] * ncol
    if headers:
        headers = [str(h) for h in headers] + [""] * (ncol - len(headers))
        for i, h in enumerate(headers):
            widths[i] = len(h)
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    aligns = aligns or ["l"] * ncol

    def fmt(cells, style=None):
        parts = []
        for i, c in enumerate(cells):
            p = c.rjust(widths[i]) if aligns[i] == "r" else c.ljust(widths[i])
            parts.append(p)
        line = " " * indent + "  ".join(parts).rstrip()
        return style(line) if style else line

    out = []
    if headers:
        out.append(fmt(headers, dim))
    for r in rows:
        out.append(fmt(r))
    return "\n".join(out)


def _thousands(n):
    return "{:,}".format(n) if isinstance(n, int) else n


# ---------------------------------------------------------------- sections

def case_block(case):
    pos, neg = case.positives, case.negatives
    lines = [section("Test case")]
    lines.append(table([
        ("name", case.name),
        ("examples", "%d  (%s positive, %s negative)"
         % (len(case), green(str(len(pos))), red(str(len(neg))))),
        ("lengths", "%d–%d" % (min(map(len, case.strings)),
                               max(map(len, case.strings)))),
        ("source", case.path or "entered by hand"),
    ]))
    shown = []
    for s, lab in zip(case.strings, case.labels):
        shown.append(green(s + " +") if lab else red(s + " -"))
    lines.append("")
    line, buf = [], 0
    for item in shown:
        raw = len(item) if not _ON else len(item) - 9
        if buf + raw + 3 > WIDTH - 4:
            line.append("\n")
            buf = 0
        line.append(item)
        buf += raw + 3
    lines.append("  " + "   ".join(line).replace("\n   ", "\n  "))
    if case.gloss:
        lines.append("")
        lines.append(dim("  intent: " + _wrap(case.gloss, WIDTH - 12, "          ")))
    return "\n".join(lines)


def _wrap(text, width, indent=""):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    out.append(line)
    return ("\n" + indent).join(out)


def answer_block(res):
    lines = [section("Answer")]
    if res.kind == "none":
        lines.append("  " + yellow("no hypothesis found — " + res.note))
        return "\n".join(lines)
    label = {"single": "single predicate (provably smallest, by tree size)",
             "dnf": "disjunctive normal form, from Layer 2"}[res.kind]
    lines.append("  " + bold(green(res.formula)))
    lines.append("")
    lines.append(table([
        ("form", label),
        ("size(G)", "%d nodes (DAG, shared counted once — §14.3)" % res.size),
        ("training errors", res.errors),
        ("MDL cost", "%.2f bits" % res.cost),
    ]))
    lines.append("")
    lines.append(dim("  Consistent with the data, which is not the same as "
                     "correct (§2.4)."))
    lines.append(dim("  The identifiability block below says how far apart "
                     "those two are here."))
    if res.kind == "dnf":
        lines.append("")
        lines.append(dim("  %d rule(s):" % len(res.rules)))
        for i, term in enumerate(res.rules, 1):
            parts = []
            for cid, neg in term:
                g, _ = res.bank.extract(cid)
                parts.append(("NOT " if neg else "") + render(g))
            lines.append("    %d. %s" % (i, " and ".join(parts)))
    return "\n".join(lines)


def bank_block(res):
    b = res.bank
    lines = [section("Value bank  (§21)")]
    rows = []
    for lv in b.levels:
        t = lv["by_type"]
        rows.append((lv["size"], _thousands(lv["new"]), _thousands(lv["total"]),
                     t["Str"], t["Char"], t["Int"], t["Bool"],
                     _thousands(lv["productions"]), _thousands(lv["tuples"]),
                     "%.2f" % lv["seconds"]))
    lines.append(table(rows,
                       headers=("size", "new", "classes", "Str", "Chr", "Int",
                                "Bool", "prods", "tuples", "sec"),
                       aligns=["r"] * 10))
    s = b.summary()
    lines.append("")
    notes = []
    if s["capped"]:
        notes.append(yellow("poolCap reached — no new classes after that point"))
    if s["timed_out"]:
        notes.append(yellow("timed out"))
    lines.append(table([
        ("classes kept", _thousands(s["classes"])),
        ("productions stored", "%s  (of %s recorded, pMax=%d)"
         % (_thousands(s["productions"]), _thousands(s["recorded"]), b.p_max)),
        ("tuples formed", _thousands(s["tuples"])),
        ("collapse", bold("%.1f×" % s["collapse"])
         + dim("   tuples formed per class kept")),
        ("stopped at size", res.stopped_at),
    ] + ([("notes", "; ".join(notes))] if notes else [])))
    lines.append("")
    lines.append(dim("  §21.7 predicts this collapse is smaller than CTL's 3,453×,"))
    lines.append(dim("  because behaviour equivalence also separates on negatives and ⊥."))
    return "\n".join(lines)


def identifiability_block(res):
    lines = [section("Identifiability  (§19.4)")]
    if res.kind != "single":
        lines.append(dim("  Reported for a single-predicate answer only; this run "
                         "returned a DNF,"))
        lines.append(dim("  which Layer 2 selects greedily rather than exhaustively."))
        return "\n".join(lines)
    rows = [("consistent classes", res.consistent_classes),
            ("of minimal size", res.minimal_classes),
            ("distinct minimal expressions",
             "%d%s" % (len(res.alternatives),
                       " (truncated)" if res.alternatives_truncated else ""))]
    lines.append(table(rows))
    if len(res.alternatives) > 1:
        lines.append("")
        lines.append(dim("  equally small, equally consistent:"))
        for _, formula, sz in res.alternatives:
            lines.append("    " + formula + dim("   [%d]" % sz))
    if res.disambiguator:
        yes, no = res.disambiguator_split
        lines.append("")
        lines.append("  " + bold("ask about: ") + bold(yellow(res.disambiguator))
                     + dim("   %d of them say +, %d say −" % (yes, no)))
        lines.append(dim("  Labelling that one string is the single most useful "
                         "example you could add."))
    elif len(res.alternatives) > 1:
        lines.append("")
        lines.append(dim("  no disambiguating string found within length "
                         "max+1 — the alternatives may be equivalent."))
    lines.append("")
    verdict = ("%d examples given; roughly %d recommended for this hypothesis space"
               % (res.n_examples, res.sample_needed))
    if res.n_examples < res.sample_needed:
        lines.append("  " + yellow(verdict))
        lines.append(dim("  This result is not identified by the data (§2.7)."))
    else:
        lines.append("  " + green(verdict))
    return "\n".join(lines)


def generalisation_block(sc):
    """Section 19.2 --- accuracy against the oracle over the whole universe."""
    lines = [section("Generalisation  (§19.2)")]
    if sc is None:
        lines.append(dim("  No oracle for this case; nothing to score against."))
        lines.append(dim("  Consistency with the training data is all that was "
                         "measured."))
        return "\n".join(lines)
    verdict = (green("recovered — agrees with the oracle on all %d strings" % sc.n)
               if sc.recovered else
               red("not recovered — disagrees on %d of %d strings"
                   % (sc.n - sc.tp - sc.tn, sc.n)))
    rows = [("intended concept", sc.concept or "(unnamed)"),
            ("accuracy", "%.1f%%   over Σ^1..6, %d strings"
             % (100 * sc.accuracy, sc.n)),
            ("unseen only", "%.1f%%   %d strings not in training"
             % (100 * sc.unseen_accuracy, sc.unseen_n)),
            ("longer than training", "%.1f%%   %d strings"
             % (100 * sc.beyond_accuracy, sc.beyond_n)),
            ("precision / recall / F1", "%.3f / %.3f / %.3f"
             % (sc.precision, sc.recall, sc.f1)),
            ("confusion", "TP %d   FP %d   TN %d   FN %d"
             % (sc.tp, sc.fp, sc.tn, sc.fn))]
    lines.append(table(rows))
    lines.append("")
    lines.append("  " + verdict)
    if sc.disagreements:
        lines.append("")
        lines.append(dim("  first disagreements (string: learned / intended):"))
        items = ["%s: %s/%s" % (w, "+" if p else "−", "+" if t else "−")
                 for w, p, t in sc.disagreements]
        lines.append("    " + "   ".join(items))
    return "\n".join(lines)


def config_block(res):
    lines = [section("Configuration  (§18)")]
    lines.append(table([(k, v) for k, v in res.config.as_rows()]))
    lines.append(table([("elapsed", "%.2f s" % res.seconds)]))
    return "\n".join(lines)


def full_report(case, res, show_bank=True, show_ident=True, show_config=True,
                score=None, show_score=True):
    out = [header("CGL v0.1  —  concept graph learner",
                  "value bank + sequential covering, §21 / §16.3")]
    out.append(case_block(case))
    out.append(answer_block(res))
    if show_score and score is not None:
        out.append(generalisation_block(score))
    if show_bank:
        out.append(bank_block(res))
    if show_ident:
        out.append(identifiability_block(res))
    if show_config:
        out.append(config_block(res))
    out.append("")
    return "\n".join(out)


def summary_row(case, res, score=None):
    ok = green("✓") if res.kind != "none" and res.errors == 0 else red("✗")
    row = [ok, case.name, res.kind, res.size, res.errors, "%.2f" % res.seconds]
    if score is not None:
        rec = green("✓") if score.recovered else red("✗")
        row += ["%.1f" % (100 * score.accuracy), rec]
    row.append(res.formula[:38])
    return tuple(row)


def baseline_line(mean_gen, recovered, total, baselines):
    """Print the CGL figures next to the recorded CTL ones."""
    lines = ["", "  " + bold("CGL v0.1") + "   mean generalisation %s, %s recovered"
             % (bold("%.1f%%" % (100 * mean_gen)),
                bold("%d of %d" % (recovered, total)))]
    for b in baselines:
        lines.append(dim("  %-24s mean generalisation %.1f%%, %d of %d recovered"
                         % (b["label"], 100 * b["mean_generalisation"],
                            b["recovered"], b["cases"])))
    return "\n".join(lines)
