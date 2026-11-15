"""Cycle 8, MMLU-Pro replication of H1-H3 and the descriptive outputs (protocol Section 22).

Reuses the frozen S1 functions (cycle8_s1_analysis) with ten options, items as bootstrap clusters, a random
half split for cross-fitting, and the 1PL-g guessing floor at 1/10. The S1b decision code is reused for H2/H3.

    python3 checks/cycle8_pro_analysis.py --data data --mde <S0-Pro MDE> --workers 16          # server
    .venv/bin/python checks/cycle8_pro_analysis.py --data data --mde 0.005 --boot 2 --k-null 5 --workers 4 --test
"""
import argparse
import collections
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cycle8_s0_synthetic_validation as v1          # noqa: E402
import cycle8_s0b_synthetic_validation as v2         # noqa: E402
import cycle8_s1_analysis as s1                      # noqa: E402
import cycle8_s1b_supplementary as s1b               # noqa: E402

K, SEED = 10, s1.SEED
SKIP = ("/eng", "S/open_compute100B")                 # no categories, no parameter counts (Section 22)


def set_k(mde):
    s1.M_OPT = K
    v1.M_OPT = K
    v2.GUESS = 1.0 / K
    s1.MDE_H1 = mde


def load(data_dir, which):
    panel = json.load(open(os.path.join(data_dir, "helm_mmlu_pro_panel.json")))
    models = panel[which]
    items = panel["common_items"]
    im, ii = {m: i for i, m in enumerate(models)}, {it: j for j, it in enumerate(items)}
    A = np.full((len(models), len(items)), -1, dtype=np.int8)
    key = np.full(len(items), -1, dtype=np.int64)
    with gzip.open(os.path.join(data_dir, "helm_mmlu_pro_parsed.tsv.gz"), "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            m, v, iid, ans, kk, nopt, how, hc, tail = line.rstrip("\n").split("\t")
            if m in im and iid in ii:
                A[im[m], ii[iid]] = int(ans)
                key[ii[iid]] = int(kk)
    assert (key >= 0).all(), "missing keys"
    N = len(items)
    dev_names = sorted({m.split("_")[0] for m in models})
    classes = sorted({panel["settings_class"][m] for m in models})
    rng = np.random.default_rng(SEED)
    fold = np.zeros(N, dtype=int)
    fold[rng.permutation(N)[: N // 2]] = 1
    return dict(models=models, A=A, key=key, subj=np.arange(N), eng=np.zeros(N, dtype=bool),
                dev=np.array([dev_names.index(m.split("_")[0]) for m in models]), dev_names=dev_names,
                api=np.array([m not in panel["open_weight"] for m in models]),
                pclass=np.array([classes.index(panel["settings_class"][m]) for m in models]), pclass_names=classes,
                params=np.full(len(models), np.nan), strict=set(panel["strict"]), fold=fold, ones=np.ones(N))


def init_worker(g, mde):
    set_k(mde)
    s1.G.update(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(HERE), "data"))
    ap.add_argument("--mde", type=float, required=True, help="minimal detectable effect from S0-Pro")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--k-null", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--test", action="store_true", help="code test: results are not reported")
    args = ap.parse_args()
    suffix = "_test" if args.test else ""
    set_k(args.mde)
    g = load(args.data, "primary")
    g_all = load(args.data, "dedup_all")
    s1.G.update(g)
    G = s1.G
    t0 = time.time()
    print(f"{len(g['models'])} models, {len(g['key'])} items, {len(g['dev_names'])} developers, "
          f"{int(g['api'].sum())} API-only, {len(g['pclass_names'])} setting classes", flush=True)
    point = s1.analysis(G["ones"])
    print(f"point estimates done in {time.time() - t0:.0f} s", flush=True)
    h1res = s1.h1(np.random.default_rng([SEED, 1]), args.k_null)
    print(f"H1 done in {time.time() - t0:.0f} s: {h1res}", flush=True)
    acc_all = (G["A"] == G["key"][None, :]).mean(1)
    largest = np.bincount(G["dev"]).argmax()
    M = len(G["models"])
    sens = {"strict_settings_class": [i for i, m in enumerate(G["models"]) if m in G["strict"]],
            "without_largest_developer": [i for i in range(M) if G["dev"][i] != largest],
            "without_accuracy_below_40pct": [i for i in range(M) if acc_all[i] >= 0.40],
            "open_weight_only": [i for i in range(M) if not G["api"][i]]}
    h1res["sensitivity"] = {}
    for k, sub in sens.items():
        r = s1.h1(np.random.default_rng([SEED, 2]), max(5, args.k_null // 2), subset=sub)
        h1res["sensitivity"][k] = {kk: r[kk] for kk in ("T", "p", "ci90", "developers_used", "verdict")} | {"models": len(sub)}
    saved = dict(G)
    G.clear()
    G.update(g_all)
    r = s1.h1(np.random.default_rng([SEED, 2]), max(5, args.k_null // 2))
    h1res["sensitivity"]["with_low_agreement_models"] = {kk: r[kk] for kk in ("T", "p", "ci90", "developers_used", "verdict")} \
        | {"models": len(g_all["models"])}
    G.clear()
    G.update(saved)
    h1res["sensitivity_note"] = f"largest developer: {G['dev_names'][largest]}"
    print(f"H1 sensitivity done in {time.time() - t0:.0f} s", flush=True)
    boot_path = os.path.join(HERE, f"cycle8_pro_bootstrap{suffix}.jsonl")
    done = {}
    if os.path.exists(boot_path):
        for line in open(boot_path):
            rr = json.loads(line)
            done[rr["b"]] = rr["res"]
    todo = [b for b in range(args.boot) if b not in done]
    with open(boot_path, "a") as f, Pool(args.workers, initializer=init_worker, initargs=(g, args.mde)) as pool:
        for i, (b, rr) in enumerate(pool.imap_unordered(s1.boot_task, todo), 1):
            f.write(json.dumps({"b": b, "res": rr}) + "\n")
            f.flush()
            done[b] = rr
            if i % 10 == 0 or i == len(todo):
                print(f"  bootstrap {i}/{len(todo)}, {time.time() - t0:.0f} s", flush=True)
    boots = [done[b] for b in range(args.boot)]
    d = s1b.decide(point, boots)
    ci = s1b.make_ci(boots)
    desc = {}
    for k, v in point.items():
        if not k.startswith(("D/", "S/")) or not isinstance(v, float) or any(s in k for s in SKIP):
            continue
        if "/JI/" in k:
            vals = np.array([rr.get(k, np.nan) for rr in boots], dtype=float)
            vals = vals[~np.isnan(vals)]
            n_cl = len(G["key"])
            adj = v + np.sqrt(n_cl / (n_cl - 1)) * (vals - vals.mean()) if len(vals) else vals
            desc[k] = [v] + ([float(np.quantile(adj, 0.025)), float(np.quantile(adj, 0.975))] if len(vals) else [None, None])
        else:
            desc[k] = [v] + ci(lambda r_, kk=k: r_.get(kk, np.nan))
    raw = {}
    for k in sorted(point):
        if k.startswith("D/prim/dE/"):
            vals = [(rr["D/prim/full"] + rr[k]) / rr["D/prim/full"] for rr in boots if k in rr and rr.get("D/prim/full")]
            raw[k.split("/", 3)[3]] = [(point["D/prim/full"] + point[k]) / point["D/prim/full"]] + \
                ([float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))] if vals else [None, None]) + [len(vals)]
    out = {"point": point, "H1": h1res, "H2": d["H2"], "H3": d["H3"], "descriptive": desc, "RAW": raw,
           "n_boot": len(boots), "mde": args.mde, "seconds": round(time.time() - t0)}
    json.dump(out, open(os.path.join(HERE, f"cycle8_pro_analysis{suffix}.json"), "w"), indent=1, default=str)
    L = [f"Cycle 8 MMLU-Pro replication{' (CODE TEST, not a result)' if args.test else ''}",
         f"models {len(G['models'])}, items {len(G['key'])}, bootstrap resamples {len(boots)}, MDE {args.mde}",
         f"\nH1: T = {h1res['T']:.5f}, p = {h1res['p']:.3f}, jackknife 90% CI [{h1res['ci90'][0]:.5f}, {h1res['ci90'][1]:.5f}], "
         f"developers used {h1res['developers_used']}, within pairs {h1res['within_pairs_used']} -> {h1res['verdict']}",
         "H1 sensitivity: " + "; ".join(f"{k} ({v['models']} models): T {v['T']:.5f}, p {v['p']:.3f}, CI [{v['ci90'][0]:.5f}, "
                                        f"{v['ci90'][1]:.5f}] -> {v['verdict']}" for k, v in h1res["sensitivity"].items())
         + f" ({h1res['sensitivity_note']})"]
    for n, v in d["H2"]["by_n"].items():
        L.append(f"H2 n={n}: full winner {v['full_winner']}, worst-case winner {v['worst_winner']}, flip {v['flip']}; "
                 + "; ".join(f"{r} {100*v['full'][r]:.2f}/{100*v['worst'][r]:.2f}" for r in s1.RULES))
    L.append(f"H2 -> {d['H2']['verdict']}")
    h3 = d["H3"]
    L.append(f"H3: held-out worst-case error robust {100*h3['H3/robust/heldout_worst']:.2f}, cap {100*h3['H3/cap/heldout_worst']:.2f}, "
             f"greedy {100*h3['H3/greedy/heldout_worst']:.2f}, MI {100*h3['H3/mi/heldout_worst']:.2f}; gain vs cap "
             f"{100*h3['gain_vs_cap']:.2f} pp, CI [{100*h3['gain_ci'][0]:.2f}, {100*h3['gain_ci'][1]:.2f}]; full-availability "
             f"cost vs cap {100*h3['full_cost_vs_cap']:.2f} pp; full errors cap {100*h3['H3/cap/full']:.2f}, greedy "
             f"{100*h3['H3/greedy/full']:.2f}, robust {100*h3['H3/robust/full']:.2f}, MI {100*h3['H3/mi/full']:.2f} -> {h3['verdict']}")
    L.append("\nDescriptive (error, %; 95% CI):")
    for k, v in sorted(desc.items()):
        L.append(f"  {k}: {100*v[0]:.2f} [{100*v[1]:.2f}, {100*v[2]:.2f}]" if v[1] is not None else f"  {k}: {100*v[0]:.2f}")
    L.append("RAW (primary portfolio): " + "; ".join(f"{k} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}] ({v[3]})" for k, v in raw.items()))
    L.append(f"\nportfolios (fold 0 / fold 1): {json.dumps(point.get('_portfolios'))}")
    L.append(f"seconds {out['seconds']}")
    text = "\n".join(L)
    open(os.path.join(HERE, f"cycle8_pro_analysis{suffix}.out"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
