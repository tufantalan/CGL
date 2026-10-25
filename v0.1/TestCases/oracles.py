"""Executable ground-truth oracles for the 29 CTL concepts.

Derived from each dataset's idea.txt (and cross-checked against the responses
reported in Table 4 of the Applied Sciences paper). Each oracle maps a binary
string to True (positive) / False (negative).

These make concept recovery and generalisation measurable against the *intended*
target, rather than against the learner's own depth-4 output.
"""

ORACLES = {
    1:  lambda s: len(s) != 5,
    2:  lambda s: s.startswith("1"),
    3:  lambda s: s.endswith("0"),
    4:  lambda s: s.count("1") >= 3,
    5:  lambda s: len(s) == 4,
    6:  lambda s: s.count("0") == s.count("1"),
    7:  lambda s: len(s) != 3,
    8:  lambda s: s.startswith("10"),
    9:  lambda s: s.endswith("01"),
    10: lambda s: s.count("0") >= 4,
    11: lambda s: s.startswith("1") or s.endswith("0"),
    12: lambda s: len(s) in (3, 5),
    13: lambda s: s.startswith("00"),
    14: lambda s: s.endswith("11"),
    15: lambda s: s.count("1") >= 3 and s.endswith("0"),
    16: lambda s: s.startswith("0") and len(s) == 4,
    17: lambda s: (s.startswith("1") and s.endswith("0"))
                 or (s.startswith("0") and s.count("1") >= 3),
    18: lambda s: s.startswith("1") or len(s) == 3,
    19: lambda s: s.count("0") >= 3 or s.count("1") >= 3,
    20: lambda s: len(s) >= 1 and s[0] == s[-1],
    21: lambda s: s.endswith("00"),
    22: lambda s: s.endswith("000"),
    23: lambda s: s.endswith("100"),
    24: lambda s: s.endswith("0") and len(s) == 5,
    25: lambda s: s.startswith("1") and len(s) == 4,
    26: lambda s: s.startswith("1") and s.endswith("0") and len(s) == 5,
    27: lambda s: s.startswith("1") and s.endswith("00"),
    28: lambda s: s.endswith("00") or (s.endswith("1") and len(s) == 5),
    29: lambda s: len(s) == 3
                 or (len(s) == 4 and s.endswith("0"))
                 or (len(s) == 5 and s.endswith("1")),
}

# Short English glosses, for reports.
NAMES = {
    1: "length is not 5",                     2: "starts with 1",
    3: "ends with 0",                         4: "at least three 1s",
    5: "length is exactly 4",                 6: "equal numbers of 0s and 1s",
    7: "length is not 3",                     8: "prefix 10",
    9: "suffix 01",                           10: "at least four 0s",
    11: "starts with 1 or ends with 0",       12: "length 3 or 5",
    13: "prefix 00",                          14: "suffix 11",
    15: "at least three 1s and ends with 0",  16: "starts with 0 and length 4",
    17: "(starts 1 and ends 0) or (starts 0 and >=3 ones)",
    18: "starts with 1 or length 3",          19: ">=3 zeros or >=3 ones",
    20: "first and last character are equal", 21: "ends with 00 (multiple of 4)",
    22: "ends with 000 (multiple of 8)",      23: "ends with 100 (4 but not 8)",
    24: "ends with 0 and length 5",           25: "starts with 1 and length 4",
    26: "starts with 1, ends with 0, length 5",
    27: "starts with 1 and ends with 00",     28: "ends with 00, or ends with 1 and length 5",
    29: "length 3, or length 4 ending 0, or length 5 ending 1",
}

# The exhaustive test universe used in the paper: all binary strings of length 1..6.
def universe(max_len=6):
    out = []
    for n in range(1, max_len + 1):
        for k in range(2 ** n):
            out.append(format(k, "0%db" % n))
    return out
