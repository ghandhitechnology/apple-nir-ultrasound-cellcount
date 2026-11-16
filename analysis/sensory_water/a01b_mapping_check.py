# Checks which reading of the round-2 code mapping fits the sensory data. It compares the retest
# agreement under each candidate mapping with the agreement under random mappings.

import csv
import itertools

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

SRC = r"C:/apple_project/data/sensory/sensory_responses.csv"
rows = list(csv.reader(open(SRC, encoding="utf-8")))[1:]
R = {1: {}, 2: {}}
for r in rows:
    rd = 1 if r[3].startswith("1") else 2
    R[rd][int(r[1])] = [
        (
            int(r[4 + 4 * k]),
            np.array([int(r[5 + 4 * k]), int(r[6 + 4 * k]), int(r[7 + 4 * k])], float),
        )
        for k in range(5)
    ]
panels = sorted(set(R[1]) & set(R[2]))


def score(mapfun):
    x, y = [], []
    for p in panels:
        first = R[1][p]
        perm = mapfun(first)
        for code, v in R[2][p]:
            a = first[perm[code - 1]][1]
            x.append(a - np.mean([f[1] for f in first], 0))
            y.append(v - np.mean([s[1] for s in R[2][p]], 0))
    x, y = np.array(x), np.array(y)
    return np.array([pearsonr(x[:, j], y[:, j])[0] for j in range(3)])


def listing(first):
    return list(range(5))


def ascending(first):
    return list(np.argsort([c for c, _ in first]))


for n, f in [
    ("listing order (k-th apple as listed in round 1)", listing),
    ("ascending apple number", ascending),
]:
    s = score(f)
    print("%-50s mealy %.2f juicy %.2f crisp %.2f  sum %.2f" % (n, *s, s.sum()))
rng = np.random.default_rng(0)
null = []
for _ in range(2000):
    perms = {p: rng.permutation(5) for p in panels}
    null.append(score(lambda first, _p=iter(panels): None) if False else None)

null = []
for _ in range(2000):
    P = {p: list(rng.permutation(5)) for p in panels}
    it = iter(panels)
    x, y = [], []
    for p in panels:
        first = R[1][p]
        for code, v in R[2][p]:
            x.append(first[P[p][code - 1]][1] - np.mean([f[1] for f in first], 0))
            y.append(v - np.mean([s[1] for s in R[2][p]], 0))
    x, y = np.array(x), np.array(y)
    null.append(sum(pearsonr(x[:, j], y[:, j])[0] for j in range(3)))
null = np.array(null)
for n, f in [("listing", listing), ("ascending", ascending)]:
    s = score(f).sum()
    print(n, "sum r %.2f, percentile among random mappings %.1f%%" % (s, 100 * (null < s).mean()))
print(
    "random mapping sum r: median %.2f, 95th %.2f, max %.2f"
    % (np.median(null), np.percentile(null, 95), null.max())
)
