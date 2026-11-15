"""Cycle 8, amendment A5: targeted re-checks of S0-Pro (protocol Section 23). No real answers used.

1. Size: 400 new null datasets, 200 null panels per H1 test (as in the real analysis). Pass if rejection rate <= 0.06.
2. Coverage: 100 new datasets at sigma = 0.3; true values from 50,000 fresh items drawn from the SAME 14 subject
   shifts as the observed items. Pass if Delta E and JI coverage are each >= 0.90.

Reuses the generator of checks/cycle8_s0pro_validation.py unchanged, except for drawing items with given subject shifts.

    .venv/bin/python checks/cycle8_s0pro_a5.py --data data --smoke --workers 4
    python3 checks/cycle8_s0pro_a5.py --data data --workers 16
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cycle8_s0pro_validation as sp                 # noqa: E402
import cycle8_s1_analysis as s1                      # noqa: E402
import cycle8_s0_synthetic_validation as v1          # noqa: E402

MASTER_SEED = 20261117                               # new seeds, different from S0-Pro (20261116)


def draw_items_shift(rng, n, shift):
    """As sp.draw_items, but with the subject difficulty shifts given (fixed categories)."""
    K, N_SUBJ = sp.K, sp.N_SUBJ
    subj = np.arange(n) % N_SUBJ
    b = shift[subj] + rng.normal(0, 1, n)
    a = np.exp(rng.normal(0, 0.3, n))
    y = rng.integers(0, K, n)
    q = rng.dirichlet(np.ones(K), n)
    key = y.copy()
    err = rng.random(n) < sp.KEY_ERROR
    key[err] = (y[err] + rng.integers(1, K, err.sum())) % K
    return {"subj": subj, "b": b, "a": a, "y": y, "q": q, "key": key}


def run(task):
    kind, rep, cfg = task["kind"], task["rep"], task["cfg"]
    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, 0 if kind == "size" else 1, rep]))
    t0 = time.time()
    sigma = 0.0 if kind == "size" else 0.3
    world = sp.draw_world(rng, sigma)
    shift = rng.normal(0, 0.5, sp.N_SUBJ)
    items = draw_items_shift(rng, sp.N_ITEMS, shift)
    ans = sp.simulate(rng, world, items)
    out = {"kind": kind, "rep": rep}
    if kind == "size":
        s1.G.clear()
        s1.G.update(A=ans, key=items["key"], dev=sp.S["dev"], pclass=sp.S["pclass"], dev_names=sp.S["dev_names"])
        r = s1.h1(rng, cfg["k_null"])
        out.update(T=r["T"], p=r["p"])
    else:
        N = sp.N_ITEMS
        half = np.zeros(N, dtype=bool)
        half[rng.permutation(N)[: N // 2]] = True
        sel, ev = np.where(half)[0], np.where(~half)[0]
        k_sel, k_ev = items["key"][sel], items["key"][ev]
        a_sel, a_ev = ans[:, sel], ans[:, ev]
        acc = (a_sel == k_sel[None, :]).mean(1)
        port = v1.greedy_portfolio(a_sel, k_sel, acc, list(range(len(sp.S["dev"]))), 5)
        scen = [("dev:" + sp.S["dev_names"][d], [i for i in port if sp.S["dev"][i] == d]) for d in sorted(set(sp.S["dev"][port]))]
        api = [i for i in port if sp.S["api"][i]]
        if api:
            scen.append(("api-only", api))
        scen = [(nm, W) for nm, W in scen if len(W) < len(port)]
        fresh = draw_items_shift(rng, sp.N_FRESH, shift)
        a_fr = sp.simulate(rng, world, fresh)
        errs = sp.errs
        e0_ev, e0_fr = errs(a_ev, k_ev, acc, port), errs(a_fr, fresh["key"], acc, port).mean()
        rows = []
        for nm, W in scen:
            rest = [i for i in port if i not in W]
            d = errs(a_ev, k_ev, acc, rest) - e0_ev
            bs = sp.item_boot(d, rng, cfg["boot"])
            rows.append({"scenario": nm, "est": float(d.mean()), "lo": float(np.quantile(bs, 0.025)),
                         "hi": float(np.quantile(bs, 0.975)),
                         "truth": float(errs(a_fr, fresh["key"], acc, rest).mean() - e0_fr)})
        out["dE"] = rows
        if len(scen) >= 2:
            (n1, W1), (n2, W2) = scen[0], scen[1]
            r1, r2 = [i for i in port if i not in W1], [i for i in port if i not in W2]
            r12 = [i for i in port if i not in W1 and i not in W2]
            if r12:
                ji = errs(a_ev, k_ev, acc, r12) - errs(a_ev, k_ev, acc, r1) - errs(a_ev, k_ev, acc, r2) + e0_ev
                bs = sp.item_boot(ji, rng, cfg["boot"])
                adj = ji.mean() + np.sqrt(len(ji) / (len(ji) - 1)) * (bs - bs.mean())
                tr = (errs(a_fr, fresh["key"], acc, r12).mean() - errs(a_fr, fresh["key"], acc, r1).mean()
                      - errs(a_fr, fresh["key"], acc, r2).mean() + e0_fr)
                out["JI"] = {"pair": [n1, n2], "est": float(ji.mean()), "lo": float(np.quantile(adj, 0.025)),
                             "hi": float(np.quantile(adj, 0.975)), "truth": float(tr)}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(HERE), "data"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    panel = json.load(open(os.path.join(args.data, "helm_mmlu_pro_panel.json")))
    cfg = dict(n_size=4, n_cov=3, k_null=20, boot=100) if args.smoke else dict(n_size=400, n_cov=100, k_null=200, boot=300)
    suffix = "_smoke" if args.smoke else ""
    res_path = os.path.join(HERE, f"cycle8_s0pro_a5_results{suffix}.jsonl")
    out_path = os.path.join(HERE, f"cycle8_s0pro_a5{suffix}.out")
    tasks = [{"kind": "size", "rep": i, "cfg": cfg} for i in range(cfg["n_size"])]
    tasks += [{"kind": "cov", "rep": i, "cfg": cfg} for i in range(cfg["n_cov"])]
    done = {}
    if os.path.exists(res_path):
        for line in open(res_path):
            r = json.loads(line)
            done[(r["kind"], r["rep"])] = r
    todo = [t for t in tasks if (t["kind"], t["rep"]) not in done]
    sp.set_globals(panel)
    print(f"{len(tasks)} datasets, {len(todo)} to run", flush=True)
    t0 = time.time()
    with open(res_path, "a") as f, Pool(args.workers, initializer=sp.init_worker, initargs=(panel,)) as pool:
        for i, r in enumerate(pool.imap_unordered(run, todo), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            done[(r["kind"], r["rep"])] = r
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}, {time.time() - t0:.0f} s", flush=True)
    rows = list(done.values())
    size_rows = [r for r in rows if r["kind"] == "size"]
    size = float(np.mean([r["p"] < 0.05 for r in size_rows]))
    de = [x for r in rows if r["kind"] == "cov" for x in r["dE"]]
    ji = [r["JI"] for r in rows if r["kind"] == "cov" and "JI" in r]
    cov_de = float(np.mean([x["lo"] <= x["truth"] <= x["hi"] for x in de]))
    cov_ji = float(np.mean([x["lo"] <= x["truth"] <= x["hi"] for x in ji])) if ji else float("nan")
    sd_err = float(np.std([x["est"] - x["truth"] for x in de]))
    width = float(np.mean([x["hi"] - x["lo"] for x in de]))
    L = [f"Cycle 8 amendment A5{' (SMOKE TEST, not a result)' if args.smoke else ''}: S0-Pro re-checks, master seed {MASTER_SEED}",
         f"config {json.dumps(cfg)}",
         f"1. size: {size:.3f} over {len(size_rows)} null datasets (limit 0.06; binomial SE at 0.05: "
         f"{np.sqrt(0.05 * 0.95 / max(len(size_rows), 1)):.3f}) -> {'PASS' if size <= 0.06 else 'FAIL'}",
         f"2. coverage with fixed subjects: Delta E {cov_de:.3f} over {len(de)} intervals (misses below "
         f"{sum(x['truth'] < x['lo'] for x in de)}, above {sum(x['truth'] > x['hi'] for x in de)}; mean width {width:.4f}, "
         f"sd of error {sd_err:.4f}), JI {cov_ji:.3f} over {len(ji)} -> "
         f"{'PASS' if cov_de >= 0.90 and cov_ji >= 0.90 else 'FAIL'}",
         f"A5 overall: {'PASS' if size <= 0.06 and cov_de >= 0.90 and cov_ji >= 0.90 else 'FAIL'}",
         f"seconds {round(time.time() - t0)}"]
    open(out_path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
