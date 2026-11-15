"""Cycle 8, POST HOC (protocol Section 40): leaderboard inheritance with lineage from Hugging Face model cards.
Classes C3 direct parent-child, C2 card siblings, C1 same architecture / different card root, C0 unrelated.
Accuracy-matched excess of P(same|both wrong) over C0; family bootstrap over card roots; placebo as in Section 35 R3.
    .venv/bin/python checks/cycle8_olb_cardlineage.py --data data/olb --boot 1000
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
import cycle8_olb_robustness as rb                   # noqa: E402

SEED = 20261117


def parents(name, cards):
    b = (cards.get(name) or {}).get("base_model")
    if isinstance(b, str):
        return [b]
    if isinstance(b, list):
        return [x for x in b if isinstance(x, str)]
    return []


def card_root(name, cards, stats):
    seen, cur = [name], name
    for _ in range(10):
        ps = parents(cur, cards)
        if len(ps) != 1:                              # no base, or a merge: stop here
            return cur
        nxt = ps[0]
        if nxt in seen:
            stats["loops"] += 1
            return cur
        seen.append(nxt)
        cur = nxt
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(HERE), "data", "olb"))
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    t0 = time.time()
    info, rows = o.load(args.data)
    cards = json.load(open(os.path.join(args.data, "card_base_models.json")))
    stats = collections.Counter()
    P = []
    for r in rows:
        a, b = r["name"], r["name_2"]
        ra, rbb = card_root(a, cards, stats), card_root(b, cards, stats)
        pa_, pb_ = parents(a, cards), parents(b, cards)
        if b in pa_ or a in pb_:
            cls = 3
        elif ra == rbb and ra not in (a, b):
            cls = 2
        elif r["architecture"] == r["architecture_2"]:
            cls = 1
        else:
            cls = 0
        merge = cls == 3 and len(pa_ if b in pa_ else pb_) > 1      # child lists several bases: a merge
        P.append(dict(a=a, b=b, fa=ra, fb=rbb, cls=cls, merge=merge, acc1=float(r["accuracy_1"]), acc2=float(r["accuracy_2"]),
                      s=float(r["agreement_rate_when_both_wrong"])))
    by = collections.defaultdict(list)
    for k, p in enumerate(P):
        by[p["cls"]].append(k)
    idx0 = by[0]
    ms = o.matched_sets(P, by[1] + by[2] + by[3], idx0)
    s0 = np.array([P[k]["s"] for k in idx0])
    ex = {t: P[t]["s"] - float(s0[ms[t][0]].mean()) for t in ms}
    pa = {k: (rb.params(P[k]["a"], info), rb.params(P[k]["b"], info)) for k in range(len(P))}

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
            m = np.array([(close(u, x) and close(v, y)) or (close(u, y) and close(v, x)) for u, v in l1p])
            if m.sum() >= 5:
                placebo[t] = float(l1e[m].mean())
    groups = {f"C{c}": [(P[t]["fa"], ex[t]) for t in by[c] if t in ex] for c in (1, 2, 3)}
    for c in (2, 3):
        ts = [t for t in placebo if P[t]["cls"] == c]
        groups[f"C{c} (with size)"] = [(P[t]["fa"], ex[t]) for t in ts]
        groups[f"C{c} placebo"] = [(P[t]["fa"], placebo[t]) for t in ts]
        groups[f"C{c} - placebo"] = [(P[t]["fa"], ex[t] - placebo[t]) for t in ts]
    for tag, flag in (("fine-tune only", False), ("merge only", True)):          # diagnostic split (Section 41)
        ts = [t for t in placebo if P[t]["cls"] == 3 and P[t]["merge"] == flag]
        groups[f"C3 {tag}"] = [(P[t]["fa"], ex[t]) for t in ts]
        groups[f"C3 {tag} - placebo"] = [(P[t]["fa"], ex[t] - placebo[t]) for t in ts]
    fams = sorted({f for v in groups.values() for f, _ in v} | {P[t]["fb"] for t in ex})
    rng = np.random.default_rng(SEED)

    def wmean(items, w):
        num = sum(w.get(f, 0) * v for f, v in items)
        den = sum(w.get(f, 0) for f, _ in items)
        return num / den if den else float("nan")

    point = {k: wmean(v, {f: 1 for f in fams}) for k, v in groups.items()}
    bs = {k: [] for k in groups}
    for _ in range(args.boot):
        w = collections.Counter(fams[i] for i in rng.integers(0, len(fams), len(fams)))
        for k, v in groups.items():
            bs[k].append(wmean(v, w))
    n_card = sum(1 for n in {p["a"] for p in P} | {p["b"] for p in P} if parents(n, cards))
    L = ["Cycle 8 POST HOC (Section 40): leaderboard inheritance with model-card lineage",
         f"models with a card base_model {n_card}; chain loops {stats['loops']}; class sizes C0 {len(by[0])}, C1 {len(by[1])}, "
         f"C2 {len(by[2])}, C3 {len(by[3])}; family bootstrap {args.boot}, seed {SEED}"]
    res = {}
    for k, v in groups.items():
        b = np.array(bs[k])
        b = b[~np.isnan(b)]
        nf = len({f for f, _ in v})
        lo, hi = (float(np.quantile(b, .025)), float(np.quantile(b, .975))) if len(b) else (float("nan"), float("nan"))
        res[k] = {"point": point[k], "ci": [lo, hi], "n": len(v), "families": nf}
        L.append(f"  {k}: {point[k]:+.4f} [{lo:+.4f}, {hi:+.4f}] (n {len(v)}, families {nf})")
    L.append(f"seconds {round(time.time() - t0)}")
    json.dump(res, open(os.path.join(HERE, "cycle8_olb_cardlineage.json"), "w"), indent=1)
    open(os.path.join(HERE, "cycle8_olb_cardlineage.out"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
