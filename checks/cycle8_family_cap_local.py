"""Cycle 8, protocol Section 61 (F2, exploratory): developer cap vs family cap vs joint cap on the 12 local MMLU runs.

    .venv/bin/python checks/cycle8_family_cap_local.py --data data --local data/ru [--boot 200]
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cycle8_ru_analysis as ru                      # noqa: E402
import cycle8_s1_analysis as s1                      # noqa: E402
import cycle8_tpro_analysis as tp                    # noqa: E402


def pick(acc, groups_list, n):
    taken = [set() for _ in groups_list]
    chosen = []
    for i in np.argsort(-acc):
        labs = [g[i] for g in groups_list]
        if any(l in t for l, t in zip(labs, taken)):
            continue
        chosen.append(int(i))
        for l, t in zip(labs, taken):
            t.add(l)
        if len(chosen) == n:
            break
    return chosen


def err_vote(A, key, members, ev, w, acc):
    votes = np.zeros((int(ev.sum()), 4))
    for i in members:
        a = A[i, ev]
        ok = a >= 0
        votes[np.where(ok)[0], a[ok]] += 1.0 + 1e-6 * acc[i]
    pred = np.where(votes.max(1) > 0, votes.argmax(1), -1)
    return (pred != key[ev]) * w[ev]


def evaluate(A, key, fold, w, dev, fam, n=5):
    rules = {"developer": [dev], "family": [fam], "joint": [dev, fam]}
    res, ports = {}, {}
    accs = {}
    for f in (0, 1):
        fi = fold == f
        accs[f] = np.array([((A[i, fi] == key[fi]) * w[fi]).sum() / max(w[fi].sum(), 1e-9) for i in range(A.shape[0])])
    for k, g in rules.items():
        sets = {f: pick(accs[f], g, n) for f in (0, 1)}
        ports[k] = sets

        def cf_err(drop=set()):
            e = tot = 0.0
            for f in (0, 1):
                mem = [i for i in sets[f] if i not in drop]
                ev = fold != f
                if not mem:
                    e += 0.75 * w[ev].sum()
                else:
                    e += err_vote(A, key, mem, ev, w, accs[f]).sum()
                tot += w[ev].sum()
            return e / tot
        res[f"{k}/full"] = cf_err()
        for sname, labels in (("fam", fam), ("dev", dev)):
            present = sorted({labels[i] for f in (0, 1) for i in sets[f]})
            res[f"{k}/worst_{sname}"] = max(cf_err(set(np.where(labels == l)[0])) for l in present)
    return res, ports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--local", required=True)
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args()
    g = s1.load_real(args.data)
    items = set()
    with gzip.open(os.path.join(args.data, "helm_mmlu_parsed.tsv.gz"), "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            fs = line.split("\t")
            items.add((fs[2], fs[3]))
    order_all = {f"{s}|{i}": j for j, (s, i) in enumerate(sorted(items))}
    sub = [json.loads(l) for l in gzip.open(os.path.join(args.data, "helm_mmlu_prompts.jsonl.gz"), "rt", encoding="utf-8")]
    idx = np.array([order_all[f"{r['subject']}|{r['instance_id']}"] for r in sub])
    key, subj = g["key"][idx], g["subj"][idx]
    labels = list(ru.LOCAL)
    order = {f"{r['subject']}|{r['instance_id']}": j for j, r in enumerate(sub)}
    A = np.vstack([tp.read_local(os.path.join(args.local, f"{l}.jsonl"), order, len(sub)) for l in labels]).astype(np.int8)
    dev = np.array([ru.LOCAL[l][1] for l in labels])
    fam = np.array([ru.LINEAGE[l] for l in labels])
    rng = np.random.default_rng(tp.SEED)
    fold = np.zeros(len(key), dtype=int)
    for s in np.unique(subj):
        ids = np.where(subj == s)[0]
        fold[rng.permutation(ids)[: len(ids) // 2]] = 1
    w1 = np.ones(len(key))
    point, ports = evaluate(A, key, fold, w1, dev, fam)
    L = ["Cycle 8 family-cap experiment on the 12 local MMLU runs (protocol Section 61, F2; exploratory). Errors in %",
         f"developers {len(set(dev))}, families {len(set(fam))}"]
    for k in ("developer", "family", "joint"):
        L.append(f"  {k:9s} cap: full {100*point[k+'/full']:.2f}; worst single-family withdrawal {100*point[k+'/worst_fam']:.2f}; "
                 f"worst single-developer withdrawal {100*point[k+'/worst_dev']:.2f}; portfolios fold0 {[labels[i] for i in ports[k][0]]}, "
                 f"fold1 {[labels[i] for i in ports[k][1]]}")
    out = {"point": point}
    if args.boot:
        rb = np.random.default_rng(tp.SEED + 61)
        nS = int(subj.max()) + 1
        bs = []
        for _ in range(args.boot):
            cnt = np.bincount(rb.integers(0, nS, nS), minlength=nS).astype(float)
            r, _ = evaluate(A, key, fold, cnt[subj], dev, fam)
            bs.append(r)
        for a in ("family", "joint"):
            for m in ("full", "worst_fam", "worst_dev"):
                d = np.array([b[f"{a}/{m}"] - b[f"developer/{m}"] for b in bs])
                pt = point[f"{a}/{m}"] - point[f"developer/{m}"]
                L.append(f"  {a} - developer, {m}: {100*pt:+.2f} [{100*np.quantile(d,.025):+.2f}, {100*np.quantile(d,.975):+.2f}]")
                out[f"{a}-developer/{m}"] = [pt, float(np.quantile(d, .025)), float(np.quantile(d, .975))]
    open(os.path.join(HERE, "cycle8_family_cap_local.out"), "w").write("\n".join(L) + "\n")
    json.dump(out, open(os.path.join(HERE, "cycle8_family_cap_local.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
