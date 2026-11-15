"""Cycle 8, POST HOC (protocol Section 35, R3): leaderboard inheritance robustness.
Placebo (same architecture, different declared base, same sizes within +-10%), depth intervals, within-family triplets.
Data and matching as in checks/cycle8_olb_inheritance.py (Section 32).
    .venv/bin/python checks/cycle8_olb_robustness.py --data data/olb --boot 1000
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cycle8_olb_inheritance as o                   # noqa: E402

SEED = 20261116


def params(name, info):
    try:
        return float((info.get(name) or {}).get("#Params (B)") or "nan")
    except (TypeError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(HERE), "data", "olb"))
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    t0 = time.time()
    info, rows = o.load(args.data)
    P = o.classify(rows, info)
    by = collections.defaultdict(list)
    for k, p in enumerate(P):
        by[p["cls"]].append(k)
    idx0 = by[0]
    ms = o.matched_sets(P, by[1] + by[2] + by[3], idx0)
    s0 = np.array([P[k]["s"] for k in idx0])
    ex = {t: P[t]["s"] - float(s0[ms[t][0]].mean()) for t in ms}
    pa = {k: (params(P[k]["a"], info), params(P[k]["b"], info)) for k in range(len(P))}

    def close(x, y):
        return np.isfinite(x) and np.isfinite(y) and abs(x - y) <= 0.10 * max(x, y)

    l1 = [t for t in by[1] if t in ex]
    l1p = np.array([pa[t] for t in l1])
    l1e = np.array([ex[t] for t in l1])
    placebo = {}
    for c in (2, 3):
        for t in by[c]:
            if t not in ex:
                continue
            x, y = pa[t]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            m = [(close(a, x) and close(b, y)) or (close(a, y) and close(b, x)) for a, b in l1p]
            m = np.array(m)
            if m.sum() >= 5:
                placebo[t] = float(l1e[m].mean())
    fams = sorted({P[t]["fa"] for t in ex} | {P[t]["fb"] for t in ex})
    # triplets: base b with children c1, c2 both in L3
    child_of = collections.defaultdict(dict)        # base -> child -> pair index
    for t in by[3]:
        if t in ex:
            ch = P[t]["child"]
            b = P[t]["b"] if ch == P[t]["a"] else P[t]["a"]
            child_of[b][ch] = t
    pair_idx = {}
    for k, p in enumerate(P):
        pair_idx[(p["a"], p["b"])] = k
        pair_idx[(p["b"], p["a"])] = k
    trip = []
    for b, kids in child_of.items():
        for c1, c2 in __import__("itertools").combinations(sorted(kids), 2):
            k = pair_idx.get((c1, c2))
            if k is None or k not in ex:
                continue
            trip.append((b, ex[k] - 0.5 * (ex[kids[c1]] + ex[kids[c2]])))
    rng = np.random.default_rng(SEED)

    def wmean(items, w):                             # items: list of (family, value)
        num = sum(w.get(f, 0) * v for f, v in items)
        den = sum(w.get(f, 0) for f, _ in items)
        return num / den if den else float("nan")

    groups = {}
    for c in (2, 3):
        ts = [t for t in placebo if P[t]["cls"] == c]
        groups[f"L{c}"] = [(P[t]["fa"], ex[t]) for t in ts]
        groups[f"L{c}_placebo"] = [(P[t]["fa"], placebo[t]) for t in ts]
        groups[f"L{c}-placebo"] = [(P[t]["fa"], ex[t] - placebo[t]) for t in ts]
    depth = collections.defaultdict(list)
    for t in by[3]:
        if t in ex:
            depth[(info.get(P[t]["child"]) or {}).get("Generation")].append((P[t]["fa"], ex[t]))
    for d, v in depth.items():
        groups[f"L3 depth {d}"] = v
    groups["triplets: siblings - mean(child-base)"] = trip
    res = {}
    point = {k: wmean(v, {f: 1 for f in fams}) for k, v in groups.items()}
    bs = {k: [] for k in groups}
    for _ in range(args.boot):
        w = collections.Counter(fams[i] for i in rng.integers(0, len(fams), len(fams)))
        for k, v in groups.items():
            bs[k].append(wmean(v, w))
    L = ["Cycle 8 POST HOC (Section 35, R3): leaderboard inheritance robustness; excess of P(same|both wrong) over matched L0",
         f"family bootstrap {args.boot}, seed {SEED}"]
    for k, v in groups.items():
        b = np.array(bs[k])
        b = b[~np.isnan(b)]
        nf = len({f for f, _ in v})
        lo, hi = (float(np.quantile(b, .025)), float(np.quantile(b, .975))) if len(b) else (None, None)
        res[k] = {"point": point[k], "ci": [lo, hi], "n": len(v), "families": nf}
        L.append(f"  {k}: {point[k]:+.4f} [{lo:+.4f}, {hi:+.4f}] (n {len(v)}, families {nf})" if lo is not None else f"  {k}: {point[k]:+.4f} (n {len(v)})")
    L.append(f"seconds {round(time.time() - t0)}")
    json.dump(res, open(os.path.join(HERE, "cycle8_olb_robustness.json"), "w"), indent=1)
    open(os.path.join(HERE, "cycle8_olb_robustness.out"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
