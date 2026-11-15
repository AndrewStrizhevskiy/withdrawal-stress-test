"""Cycle 8, S0-Pro: synthetic validation for the MMLU-Pro replication (protocol Section 22). No real answers used.

Uses the real MMLU-Pro primary-panel structure (developers, API status, adapter-setting classes) from
data/helm_mmlu_pro_panel.json, 1,000 items in 14 subjects, ten options, and the exact H1 statistic of the S1 script
(cycle8_s1_analysis.h1) with the 1PL-g guessing floor set to 1/10.

Criteria (Section 22): size <= 0.06 over 200 null datasets; smallest sigma in {0.1, 0.2, 0.3, 0.5} with power >= 0.8
over 100 datasets (MDE = mean T there); item-bootstrap coverage of Delta E and JI >= 0.90 over 100 datasets at 0.3.

    .venv/bin/python checks/cycle8_s0pro_validation.py --data data --smoke --workers 4     # code test
    python3 checks/cycle8_s0pro_validation.py --data data --workers 16                     # server
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
import cycle8_s0_synthetic_validation as v1          # noqa: E402
import cycle8_s0b_synthetic_validation as v2         # noqa: E402
import cycle8_s1_analysis as s1                      # noqa: E402

K = 10
KEY_ERROR = 0.03
MASTER_SEED = 20261116
N_SUBJ, N_ITEMS, N_FRESH = 14, 1000, 50000
SIGMAS = (0.1, 0.2, 0.3, 0.5)
S = {}                                                # panel structure, per process


def set_globals(panel):
    s1.M_OPT = K
    v1.M_OPT = K
    v2.GUESS = 1.0 / K
    models = sorted(panel["primary"], key=lambda m: (m.split("_")[0], m))
    devs = sorted({m.split("_")[0] for m in models})
    dev = np.array([devs.index(m.split("_")[0]) for m in models])
    lin, lc = np.zeros(len(models), dtype=int), 0
    for d in range(len(devs)):
        idx = np.where(dev == d)[0]
        lin[idx] = lc + np.arange(len(idx)) // 3                # lineages of up to three models, as in S0
        lc += (len(idx) + 2) // 3
    api = np.array([m in panel["api_only"] for m in models])
    cls_names = sorted({panel["settings_class"][m] for m in models})
    pclass = np.array([cls_names.index(panel["settings_class"][m]) for m in models])
    S.update(models=models, dev=dev, lin=lin, n_lin=lc, api=api, pclass=pclass, dev_names=devs, n_dev=len(devs))


def init_worker(panel):
    set_globals(panel)


# ------------------------------------------------------------------ generator (S0 latent model, ten options)
def know_prob_target(acc):
    return np.clip((acc - 1 / K) / ((1 - KEY_ERROR) - 1 / K), 1e-3, 0.999)


def draw_world(rng, sigma):
    m_d = rng.uniform(0.30, 0.85, S["n_dev"])
    acc_target = np.clip(m_d[S["dev"]] + rng.normal(0, 0.07, len(S["dev"])), 0.15, 0.90)
    n = 20000
    b = rng.normal(0, 1, n) + rng.normal(0, 0.5, n)
    a = np.exp(rng.normal(0, 0.3, n))
    noise = rng.normal(0, 1, n) + rng.normal(0, sigma, n) + rng.normal(0, sigma, n)
    pk = know_prob_target(acc_target)
    lo, hi = np.full(len(pk), -8.0), np.full(len(pk), 8.0)
    for _ in range(40):
        mid = (lo + hi) / 2
        p = np.mean((a[None, :] * (mid[:, None] - b[None, :]) + noise[None, :]) > 0, axis=1)
        lo, hi = np.where(p < pk, mid, lo), np.where(p >= pk, mid, hi)
    return {"theta": (lo + hi) / 2, "sigma": sigma, "w": min(sigma, 0.5)}


def draw_items(rng, n):
    subj = np.arange(n) % N_SUBJ
    b = rng.normal(0, 0.5, N_SUBJ)[subj] + rng.normal(0, 1, n)
    a = np.exp(rng.normal(0, 0.3, n))
    y = rng.integers(0, K, n)
    q = rng.dirichlet(np.ones(K), n)
    key = y.copy()
    err = rng.random(n) < KEY_ERROR
    key[err] = (y[err] + rng.integers(1, K, err.sum())) % K
    return {"subj": subj, "b": b, "a": a, "y": y, "q": q, "key": key}


def simulate(rng, world, items):
    sigma, w, dev, lin = world["sigma"], world["w"], S["dev"], S["lin"]
    n, M = len(items["b"]), len(dev)
    u = rng.normal(0, sigma, (S["n_dev"], n)) if sigma > 0 else np.zeros((S["n_dev"], n))
    v = rng.normal(0, sigma, (S["n_lin"], n)) if sigma > 0 else np.zeros((S["n_lin"], n))
    z = items["a"][None, :] * (world["theta"][:, None] - items["b"][None, :]) + u[dev] + v[lin] + rng.normal(0, 1, (M, n))
    if w > 0:
        qd = (1 - w) * items["q"][None, :, :] + w * rng.dirichlet(np.ones(K), (S["n_dev"], n))
        guess = v1.sample_categorical(rng, qd[dev])
    else:
        guess = v1.sample_categorical(rng, np.broadcast_to(items["q"], (M, n, K)))
    return np.where(z > 0, items["y"][None, :], guess).astype(np.int8)


# ------------------------------------------------------------------ one dataset
def errs(ans, key, acc, members):
    return v1.plurality_errors(v1.contributions(ans[members], acc[members]).sum(0), key)


def item_boot(values, rng, B):
    idx = rng.integers(0, len(values), (B, len(values)))
    return values[idx].mean(1)


def run(task):
    sigma, rep, cfg = task["sigma"], task["rep"], task["cfg"]
    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, int(round(sigma * 1000)), rep]))
    t0 = time.time()
    world = draw_world(rng, sigma)
    items = draw_items(rng, N_ITEMS)
    ans = simulate(rng, world, items)
    s1.G.clear()
    s1.G.update(A=ans, key=items["key"], dev=S["dev"], pclass=S["pclass"], dev_names=S["dev_names"])
    r = s1.h1(rng, cfg["k_null"])
    out = {"sigma": sigma, "rep": rep, "T": r["T"], "p": r["p"], "devs": r["developers_used"], "pairs": r["within_pairs_used"]}
    if sigma == 0.3 and rep < cfg["n_recovery"]:
        half = np.zeros(N_ITEMS, dtype=bool)
        half[rng.permutation(N_ITEMS)[: N_ITEMS // 2]] = True
        sel, ev = np.where(half)[0], np.where(~half)[0]
        k_sel, k_ev = items["key"][sel], items["key"][ev]
        a_sel, a_ev = ans[:, sel], ans[:, ev]
        acc = (a_sel == k_sel[None, :]).mean(1)
        port = v1.greedy_portfolio(a_sel, k_sel, acc, list(range(len(S["dev"]))), 5)
        scen = [("dev:" + S["dev_names"][d], [i for i in port if S["dev"][i] == d]) for d in sorted(set(S["dev"][port]))]
        api = [i for i in port if S["api"][i]]
        if api:
            scen.append(("api-only", api))
        scen = [(nm, W) for nm, W in scen if len(W) < len(port)]
        fresh = draw_items(rng, N_FRESH)
        a_fr = simulate(rng, world, fresh)
        e0_ev, e0_fr = errs(a_ev, k_ev, acc, port), errs(a_fr, fresh["key"], acc, port).mean()
        rows = []
        for nm, W in scen:
            rest = [i for i in port if i not in W]
            d = errs(a_ev, k_ev, acc, rest) - e0_ev
            bs = item_boot(d, rng, cfg["boot"])
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
                bs = item_boot(ji, rng, cfg["boot"])
                n_cl = len(ji)
                adj = ji.mean() + np.sqrt(n_cl / (n_cl - 1)) * (bs - bs.mean())
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
    cfg = (dict(n_null=4, n_alt=3, n_recovery=3, k_null=20, boot=100) if args.smoke
           else dict(n_null=200, n_alt=100, n_recovery=100, k_null=100, boot=300))
    suffix = "_smoke" if args.smoke else ""
    res_path = os.path.join(HERE, f"cycle8_s0pro_results{suffix}.jsonl")
    out_path = os.path.join(HERE, f"cycle8_s0pro_validation{suffix}.out")
    tasks = [{"sigma": 0.0, "rep": i, "cfg": cfg} for i in range(cfg["n_null"])]
    tasks += [{"sigma": s, "rep": i, "cfg": cfg} for s in SIGMAS for i in range(cfg["n_alt"])]
    done = {}
    if os.path.exists(res_path):
        for line in open(res_path):
            r = json.loads(line)
            done[(r["sigma"], r["rep"])] = r
    todo = [t for t in tasks if (t["sigma"], t["rep"]) not in done]
    set_globals(panel)
    print(f"panel {len(S['models'])} models, {S['n_dev']} developers, {int(S['api'].sum())} API-only, "
          f"{len(set(S['pclass']))} setting classes; {len(tasks)} datasets, {len(todo)} to run", flush=True)
    t0 = time.time()
    with open(res_path, "a") as f, Pool(args.workers, initializer=init_worker, initargs=(panel,)) as pool:
        for i, r in enumerate(pool.imap_unordered(run, todo), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            done[(r["sigma"], r["rep"])] = r
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}, {time.time() - t0:.0f} s", flush=True)
    rows = list(done.values())
    null = [r for r in rows if r["sigma"] == 0.0]
    size = float(np.mean([r["p"] < 0.05 for r in null]))
    L = [f"Cycle 8 S0-Pro{' (SMOKE TEST, not a result)' if args.smoke else ''}: MMLU-Pro panel structure, "
         f"{N_ITEMS} items, {K} options, master seed {MASTER_SEED}", f"config {json.dumps(cfg)}",
         f"criterion 1, size: {size:.3f} over {len(null)} null datasets (limit 0.06) -> {'PASS' if size <= 0.06 else 'FAIL'}"]
    mde, power = None, {}
    for s in SIGMAS:
        alt = [r for r in rows if r["sigma"] == s]
        power[s] = float(np.mean([r["p"] < 0.05 for r in alt]))
        L.append(f"  sigma {s}: power {power[s]:.2f}, mean T {np.mean([r['T'] for r in alt]):.5f}, "
                 f"developers used {np.mean([r['devs'] for r in alt]):.1f}, within pairs {np.mean([r['pairs'] for r in alt]):.1f}")
        if mde is None and power[s] >= 0.8:
            mde = (s, float(np.mean([r["T"] for r in alt])))
    L.append(f"criterion 2, MDE: " + (f"sigma {mde[0]}, MDE = {mde[1]:.5f} -> PASS" if mde else "no sigma reaches power 0.8 -> FAIL"))
    rec = [r for r in rows if "dE" in r]
    de = [x for r in rec for x in r["dE"]]
    ji = [r["JI"] for r in rec if "JI" in r]
    cov_de = float(np.mean([x["lo"] <= x["truth"] <= x["hi"] for x in de])) if de else float("nan")
    cov_ji = float(np.mean([x["lo"] <= x["truth"] <= x["hi"] for x in ji])) if ji else float("nan")
    ok3 = cov_de >= 0.90 and cov_ji >= 0.90
    L.append(f"criterion 3, item-bootstrap coverage at sigma 0.3: Delta E {cov_de:.3f} over {len(de)} intervals, "
             f"JI {cov_ji:.3f} over {len(ji)} -> {'PASS' if ok3 else 'FAIL'}")
    overall = size <= 0.06 and mde is not None and ok3
    L.append(f"S0-Pro overall: {'PASS' if overall else 'FAIL'}")
    L.append(f"seconds {round(time.time() - t0)}")
    open(out_path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
