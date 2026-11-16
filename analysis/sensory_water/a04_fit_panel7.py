# Recovers which apples panel 7 tasted in round 1, where the codes were not readable, and what the
# two typo codes 061 and 068 really were. It tries every assignment allowed by the serving design
# and keeps the one whose ratings best match the other panelists and panel 7's own retest.

import csv
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
F = OUT / "data/sensory_responses_with_panel7.csv"
rows = list(csv.reader(open(F, encoding="utf-8", newline="")))
hdr, body = rows[0], rows[1:]
rec = []
for r in body:
    for k in range(5):
        rec.append(
            dict(
                panel=int(r[1]),
                rnd=1 if r[3].startswith("1") else 2,
                pos=k + 1,
                code=int(r[4 + 4 * k]) if r[4 + 4 * k] else None,
                v=np.array([int(r[5 + 4 * k]), int(r[6 + 4 * k]), int(r[7 + 4 * k])], float),
            )
        )
d = pd.DataFrame(rec)
for rd in (1, 2):
    for p in d.panel.unique():
        m = (d.panel == p) & (d.rnd == rd)
        if m.any():
            mu = np.mean(np.stack(d.loc[m, "v"]), 0)
            d.loc[m, "vc"] = pd.Series([x - mu for x in d.loc[m, "v"]], index=d.index[m])
CAND = [37, 49, 51, 63, 65, 93, 99]
r1 = d[d.rnd == 1]
r2 = d[d.rnd == 2]


def r2_apples(p, typo_map):
    codes = [typo_map.get(c, c) for c in r1[r1.panel == p].sort_values("pos").code]
    return sorted(codes)


clean = []
for p in sorted(set(r1.panel) - {7}):
    first = r1[r1.panel == p].sort_values("pos")
    codes = list(first.code)
    asc = sorted(codes)
    for _, row in first.iterrows():
        if row.code in (61, 68):
            continue
        clean.append((row.code, row.vc))
    for _, row in r2[r2.panel == p].iterrows():
        a = asc[int(row.code) - 1]
        if a in (61, 68):
            continue
        clean.append((a, row.vc))
cons = {a: np.mean([v for b, v in clean if b == a], 0) for a in CAND}

p7_r1 = [x for x in r1[r1.panel == 7].sort_values("pos").vc]
p7_r2 = {int(row.code): row.vc for _, row in r2[r2.panel == 7].iterrows()}


def typo_rows(p, code):
    first = r1[r1.panel == p].sort_values("pos")
    vr1 = first[first.code == code].vc.iloc[0]
    rank = sorted(first.code).index(code) + 1
    vr2 = r2[(r2.panel == p) & (r2.code == rank)].vc.iloc[0]
    return vr1, vr2


p61 = int(r1[r1.code == 61].panel.iloc[0])
p68 = int(r1[r1.code == 68].panel.iloc[0])
t61 = typo_rows(p61, 61)
t68 = typo_rows(p68, 68)
others61 = set(r1[r1.panel == p61].code) - {61}
others68 = set(r1[r1.panel == p68].code) - {68}

res = []


def sq(a, b):
    return float(np.sum((a - b) ** 2))


for x, y in itertools.permutations(CAND, 2):
    if x in others61 or y in others68:
        continue
    S = [a for a in CAND if a not in (x, y)]
    base = sq(t61[0], cons[x]) + sq(t61[1], cons[x]) + sq(t68[0], cons[y]) + sq(t68[1], cons[y])
    asc = sorted(S)
    base += sum(sq(p7_r2[k + 1], cons[asc[k]]) for k in range(5))
    for order in itertools.permutations(S):
        s = (
            base
            + sum(sq(p7_r1[k], cons[order[k]]) for k in range(5))
            + sum(sq(p7_r1[k], p7_r2[asc.index(order[k]) + 1]) for k in range(5))
        )
        res.append(
            dict(
                t061=x,
                t068=y,
                order=order,
                score=s,
                one_digit=(x in (51, 63, 65)) and (y in (63, 65, 88)),
            )
        )
R = pd.DataFrame(res).sort_values("score").reset_index(drop=True)
R["rel_lik"] = np.exp(
    -(R.score - R.score.min()) / (2 * np.var(np.concatenate([v for _, v in clean])))
)
R["post"] = R.rel_lik / R.rel_lik.sum()
print(R.head(10).to_string())
best = R.iloc[0]
print(
    "\nbest: 061->%d, 068->%d, panel 7 order %s, posterior share %.2f"
    % (best.t061, best.t068, best.order, best.post)
)
for k in range(5):
    print(
        " serving position %d -> apple %d: share of fits agreeing %.2f"
        % (k + 1, best.order[k], R[R.order.map(lambda o: o[k]) == best.order[k]].post.sum())
    )
print("061 marginal:", R.groupby("t061").post.sum().round(2).to_dict())
print("068 marginal:", R.groupby("t068").post.sum().round(2).to_dict())
oned = R[R.one_digit].iloc[0]
print(
    "best among one-digit-typo fits: 061->%d 068->%d order %s (rank %d)"
    % (oned.t061, oned.t068, oned.order, oned.name + 1)
)
R.head(200).to_csv(OUT / "panel7_fit_candidates.csv", index=False)

for r in body:
    if r[1] == "7" and r[3].startswith("1"):
        for k in range(5):
            r[4 + 4 * k] = "%03d" % best.order[k]
        r[24] = "스크린샷 입력. 시료 코드는 자료 기반 최적 대응으로 추정(a04_fit_panel7.py)"
    if r[3].startswith("1"):
        for k in range(5):
            if r[4 + 4 * k] == "061":
                r[4 + 4 * k] = "%03d" % best.t061
                r[24] = (r[24] + " | " if r[24] else "") + "코드 061 -> %03d 추정 수정" % best.t061
            if r[4 + 4 * k] == "068":
                r[4 + 4 * k] = "%03d" % best.t068
                r[24] = (r[24] + " | " if r[24] else "") + "코드 068 -> %03d 추정 수정" % best.t068
with open(OUT / "data/sensory_responses_fitted.csv", "w", encoding="utf-8", newline="") as f:
    csv.writer(f, lineterminator="\r\n").writerows([hdr] + body)
print("wrote data/sensory_responses_fitted.csv")
