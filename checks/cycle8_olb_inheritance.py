"""Cycle 8, inheritance at scale on the Open LLM Leaderboard (protocol Section 32, pre-registered).

Pair statistics: Kim et al. 2025 public table (MMLU-Pro, 349 models, all pairs). Lineage: leaderboard "Base Model".
Classes: L3 parent-child, L2 siblings, L1 same architecture class / different base, L0 unrelated.
Excess = s - mean s over accuracy-matched L0 pairs (+-0.02, doubling to 0.08, >= 10 matches).
Cluster bootstrap over root-base families; comparison means are recomputed with the resampled family weights
(the matched sets and tolerances are those of the full data).

    .venv/bin/python checks/cycle8_olb_inheritance.py --data data/olb [--boot 1000] [--test]
"""
import argparse
import collections
import csv
import gzip
import io
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261115
LARGEST = "Qwen/Qwen2.5-14B"


def load(data):
    meta = json.load(gzip.open(os.path.join(data, "olb_contents.json.gz"), "rt", encoding="utf-8"))
    info = {}
    for m in meta:
        info.setdefault(m["fullname"], m)
    csv.field_size_limit(10 ** 9)
    rows = list(csv.DictReader(io.TextIOWrapper(gzip.open(os.path.join(data, "kim2025_hf_featured_models.csv.gz")), encoding="utf-8")))
    return info, rows


def root(name, info):
    b = (info.get(name) or {}).get("Base Model") or name
    return b


def classify(rows, info):
    P = []
    for r in rows:
        a, b = r["name"], r["name_2"]
        if a not in info or b not in info:
            continue
        ra, rb = root(a, info), root(b, info)
        if ra == b or rb == a:
            cls = 3
        elif ra == rb:
            cls = 2
        elif r["architecture"] == r["architecture_2"]:
            cls = 1
        else:
            cls = 0
        child = a if ra == b else (b if rb == a else None)
        P.append(dict(a=a, b=b, fa=ra, fb=rb, cls=cls, acc1=float(r["accuracy_1"]), acc2=float(r["accuracy_2"]),
                      s=float(r["agreement_rate_when_both_wrong"]), s_either=float(r["agreement_rate_when_either_wrong"]),
                      pdiff=float(r["param_diff"] or "nan"), child=child))
    return P


def matched_sets(P, idx_targets, idx0):
    a1 = np.array([P[k]["acc1"] for k in idx0])
    a2 = np.array([P[k]["acc2"] for k in idx0])
    out = {}
    for t in idx_targets:
        x, y = P[t]["acc1"], P[t]["acc2"]
        tol, m = 0.02, None
        while tol <= 0.08 + 1e-9:
            ok = ((np.abs(a1 - x) <= tol) & (np.abs(a2 - y) <= tol)) | ((np.abs(a1 - y) <= tol) & (np.abs(a2 - x) <= tol))
            if ok.sum() >= 10:
                m = np.where(ok)[0]
                break
            tol *= 2
        if m is not None:
            out[t] = (m, tol)
    return out


def class_means(P, ms, idx0, w_fam, stat, targets_by_class):
    """Weighted mean excess per class; pair weight = product of family multiplicities (same family: m_f)."""
    def pw(k):
        f1, f2 = P[k]["fa"], P[k]["fb"]
        return w_fam.get(f1, 0) if f1 == f2 else w_fam.get(f1, 0) * w_fam.get(f2, 0)
    w0 = np.array([pw(k) for k in idx0], dtype=float)
    s0 = np.array([P[k][stat] for k in idx0])
    res = {}
    for c, ts in targets_by_class.items():
        num = den = 0.0
        for t in ts:
            if t not in ms:
                continue
            wt = pw(t)
            if wt == 0:
                continue
            m = ms[t][0]
            wm = w0[m]
            if wm.sum() == 0:
                continue
            ex = P[t][stat] - float((s0[m] * wm).sum() / wm.sum())
            num += wt * ex
            den += wt
        res[c] = num / den if den else float("nan")
    return res


def run(P, idx_targets_by_class, stat, boot, rng, label):
    idx0 = [k for k, p in enumerate(P) if p["cls"] == 0]
    allt = sorted({t for ts in idx_targets_by_class.values() for t in ts})
    ms = matched_sets(P, allt, idx0)
    fams = sorted({p["fa"] for p in P} | {p["fb"] for p in P})
    point = class_means(P, ms, idx0, {f: 1 for f in fams}, stat, idx_targets_by_class)
    bs = []
    for _ in range(boot):
        draw = rng.integers(0, len(fams), len(fams))
        w = collections.Counter(fams[i] for i in draw)
        bs.append(class_means(P, ms, idx0, w, stat, idx_targets_by_class))
    out = {"label": label, "stat": stat, "n_matched": {c: sum(t in ms for t in ts) for c, ts in idx_targets_by_class.items()},
           "n_pairs": {c: len(ts) for c, ts in idx_targets_by_class.items()},
           "n_families": {c: len({P[t]["fa"] for t in ts if t in ms}) for c, ts in idx_targets_by_class.items()},
           "tol_share_002": float(np.mean([ms[t][1] <= 0.02 + 1e-9 for t in ms])) if ms else None}
    for c in idx_targets_by_class:
        v = np.array([b[c] for b in bs])
        v = v[~np.isnan(v)]
        out[f"L{c}"] = [point[c], float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else [point[c], None, None]
    for hi, lo in ((3, 2), (2, 1)):
        if hi in idx_targets_by_class and lo in idx_targets_by_class:
            v = np.array([b[hi] - b[lo] for b in bs])
            v = v[~np.isnan(v)]
            out[f"L{hi}-L{lo}"] = [point[hi] - point[lo], float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    return out, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(HERE), "data", "olb"))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--test", action="store_true", help="code test on shuffled statistics: results are not reported")
    args = ap.parse_args()
    t0 = time.time()
    info, rows = load(args.data)
    P = classify(rows, info)
    if args.test:                                    # destroy any real signal
        rr = np.random.default_rng(0)
        perm = rr.permutation(len(P))
        svals = [P[k]["s"] for k in perm]
        evals = [P[k]["s_either"] for k in perm]
        for k, p in enumerate(P):
            p["s"], p["s_either"] = svals[k], evals[k]
    by = collections.defaultdict(list)
    for k, p in enumerate(P):
        by[p["cls"]].append(k)
    rng = np.random.default_rng(SEED)
    T = {c: by[c] for c in (1, 2, 3)}
    results = []
    prim, ms = run(P, T, "s", args.boot, rng, "primary")
    results.append(prim)
    T_eq = {c: [t for t in by[c] if P[t]["pdiff"] == 0] for c in (1, 2, 3)}
    results.append(run(P, T_eq, "s", args.boot, rng, "S1 equal parameter count")[0])
    results.append(run(P, T, "s_either", args.boot, rng, "S2 agreement when either wrong")[0])
    T_nq = {1: by[1], 2: [t for t in by[2] if P[t]["fa"] != LARGEST], 3: [t for t in by[3] if P[t]["fa"] != LARGEST and P[t]["fb"] != LARGEST]}
    results.append(run(P, T_nq, "s", args.boot, rng, f"S3 without {LARGEST} family")[0])
    # I3 descriptive: L3 by child's generation and type (full-data excess, no bootstrap)
    idx0 = [k for k, p in enumerate(P) if p["cls"] == 0]
    s0 = np.array([P[k]["s"] for k in idx0])
    dose = collections.defaultdict(list)
    for t in by[3]:
        if t not in ms:
            continue
        ch = P[t]["child"]
        m = info.get(ch, {})
        ex = P[t]["s"] - float(s0[ms[t][0]].mean())
        dose[("generation", m.get("Generation"))].append(ex)
        dose[("type", (m.get("Type") or "?").split(" ", 1)[-1][:40])].append(ex)
    L = [f"Cycle 8 inheritance at scale (protocol Section 32){' (CODE TEST on shuffled statistics, not a result)' if args.test else ''}",
         f"pairs with metadata {len(P)}; class sizes L0 {len(by[0])}, L1 {len(by[1])}, L2 {len(by[2])}, L3 {len(by[3])}; "
         f"bootstrap {args.boot} over root-base families; seed {SEED}"]
    for r in results:
        L.append(f"\n[{r['label']}] statistic {r['stat']}; matched pairs {r['n_matched']} of {r['n_pairs']}; families {r['n_families']}; "
                 f"share matched at tol 0.02: {r['tol_share_002']:.2f}")
        for k in ("L1", "L2", "L3", "L3-L2", "L2-L1"):
            if k in r and r[k][1] is not None:
                L.append(f"  {k}: {r[k][0]:+.4f} [{r[k][1]:+.4f}, {r[k][2]:+.4f}]")
            elif k in r:
                L.append(f"  {k}: {r[k][0]:+.4f} (no interval)")
    p = results[0]
    i1 = {c: (p[f"L{c}"][1] is not None and p[f"L{c}"][1] > 0) for c in (2, 3)}
    i2 = all(p[k][1] is not None and p[k][1] > 0 for k in ("L3-L2", "L2-L1"))
    L.append(f"\nI1 (L3 > 0, L2 > 0): L3 {'SUPPORTED' if i1[3] else 'NOT SUPPORTED'}, L2 {'SUPPORTED' if i1[2] else 'NOT SUPPORTED'}")
    L.append(f"I2 (L3 > L2 > L1): {'SUPPORTED' if i2 else 'NOT SUPPORTED'}")
    L.append("I3 (descriptive, L3 excess by child's generation / type; n, mean):")
    for k in sorted(dose, key=str):
        L.append(f"  {k[0]}={k[1]}: n {len(dose[k])}, mean {np.mean(dose[k]):+.4f}")
    L.append(f"seconds {round(time.time() - t0)}")
    text = "\n".join(L)
    suffix = "_test" if args.test else ""
    json.dump({"results": results, "dose": {f"{k[0]}={k[1]}": dose[k] for k in dose}},
              open(os.path.join(HERE, f"cycle8_olb_inheritance{suffix}.json"), "w"), indent=1)
    open(os.path.join(HERE, f"cycle8_olb_inheritance{suffix}.out"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
