"""Cycle 8, S1b: missing pre-registered outputs (protocol Section 20). Imports the frozen S1 script; changes nothing in it.

    python3 checks/cycle8_s1b_supplementary.py --data data --workers 16                     # real data (server)
    .venv/bin/python checks/cycle8_s1b_supplementary.py --synthetic --workers 4 --quick     # code test

Outputs: checks/cycle8_s1b_supplementary[_synthetic].out and .json, checks/cycle8_s1b_sens_bootstrap[_synthetic].jsonl.
Reads (real data): checks/cycle8_s1_analysis.json and checks/cycle8_s1_bootstrap.jsonl.
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
import cycle8_s1_analysis as s1                     # noqa: E402
import cycle8_s0_synthetic_validation as v1         # noqa: E402
import cycle8_s0b_synthetic_validation as v2        # noqa: E402

SEED = s1.SEED
GRID_A, GRID_R, DRAWS = (0.5, 0.7, 0.85), (0.0, 0.3, 0.6), 20
G = s1.G


# ================================================================== decision code, copied from cycle8_s1_analysis.main
def make_ci(boots):
    def ci(keyf, a=0.05):
        v = np.array([keyf(r) for r in boots], dtype=float)
        v = v[~np.isnan(v)]
        return [float(np.quantile(v, a / 2)), float(np.quantile(v, 1 - a / 2))] if len(v) else [np.nan, np.nan]
    return ci


def decide(point, boots):
    ci = make_ci(boots)
    h2 = {}
    for n in (3, 5, 7, 9):
        full = {r: point[f"H2/all/n{n}/{r}/full"] for r in s1.RULES}
        worst = {r: point[f"H2/all/n{n}/{r}/worst"] for r in s1.RULES}
        wf, ww = min(full, key=full.get), min(worst, key=worst.get)
        flip = False
        if wf != ww:
            d_full = ci(lambda r_, a=wf, b=ww: r_[f"H2/all/n{n}/{b}/full"] - r_[f"H2/all/n{n}/{a}/full"])
            d_worst = ci(lambda r_, a=wf, b=ww: r_[f"H2/all/n{n}/{a}/worst"] - r_[f"H2/all/n{n}/{b}/worst"])
            flip = (full[ww] - full[wf] >= 0.01 and d_full[0] > 0 and worst[wf] - worst[ww] >= 0.01 and d_worst[0] > 0)
        same_or_close = (wf == ww) or (worst[wf] - worst[ww] <= 0.005)
        h2[n] = dict(full_winner=wf, worst_winner=ww, flip=bool(flip), close=bool(same_or_close), full=full, worst=worst)
    n_flip = sum(v["flip"] for v in h2.values())
    h2_verdict = "SUPPORTED" if n_flip >= 2 else ("REFUTED" if all(v["close"] for v in h2.values()) else "INCONCLUSIVE")
    gain = point["H3/cap/heldout_worst"] - point["H3/robust/heldout_worst"]
    gain_ci = ci(lambda r_: r_["H3/cap/heldout_worst"] - r_["H3/robust/heldout_worst"])
    cost = point["H3/robust/full"] - point["H3/cap/full"]
    if gain >= 0.01 and gain_ci[0] > 0 and cost <= 0.01:
        h3_verdict = "SUPPORTED"
    elif gain <= 0.005:
        h3_verdict = "REFUTED"
    else:
        h3_verdict = "INCONCLUSIVE"
    return {"H2": {"by_n": h2, "verdict": h2_verdict},
            "H3": {"gain_vs_cap": gain, "gain_ci": gain_ci, "full_cost_vs_cap": cost, "verdict": h3_verdict,
                   **{k: v for k, v in point.items() if k.startswith("H3/")}}}


# ================================================================== workers
def init_worker(g):
    G.update(g)


def restrict(base, sub):
    g = dict(base)
    for k in ("A", "dev", "api", "pclass", "params"):
        g[k] = base[k][sub]
    g["models"] = [base["models"][i] for i in sub]
    return g


def sens_task(job):
    name, sub, b = job
    full = dict(G)
    G.update(restrict(full, sub))
    try:
        if b < 0:
            w = G["ones"]
        else:
            rng = np.random.default_rng([SEED, 1000 + b])
            n_s = G["subj"].max() + 1
            w = np.bincount(rng.integers(0, n_s, n_s), minlength=n_s).astype(float)[G["subj"]]
        res = s1.analysis(w)
        res = {k: v for k, v in res.items() if k.startswith(("H2/", "H3/"))}
    finally:
        G.clear()
        G.update(full)
    return name, b, res


def null_panel(kind, rng):
    A, key = G["A"], G["key"]
    if kind == "N1":
        acc = (A == key[None, :]).mean(1)
        correct = rng.random(A.shape) < acc[:, None]
        wrong = np.ones((len(key), s1.M_OPT))
        wrong[np.arange(len(key)), key] = 0
        wrong /= wrong.sum(1, keepdims=True)
        wa = v1.sample_categorical(rng, np.broadcast_to(wrong, A.shape + (s1.M_OPT,)))
        return np.where(correct, key[None, :], wa).astype(np.int8)
    return v1.simulate_n2(rng, G["fit1plg_p"], G["pooled_wrong"], key)


def null_task(job):
    kind, j = job
    rng = np.random.default_rng([SEED, 3000 if kind == "N1" else 4000, j])
    real = G["A"]
    G["A"] = null_panel(kind, rng)
    try:
        p = s1.Pipe(G["ones"])
        prim = (p.greedy(0, 5), p.greedy(1, 5))
        e0 = p.mean(p.err("plurality", prim))
        groups = s1.scenario_groups()
        out = {"full": e0, "dE": {}, "empty": []}
        for g, W in groups.items():
            if not any(i in W for s in prim for i in s):
                continue
            ws = s1.withdraw(prim, W)
            if ws[0] and ws[1]:
                out["dE"][g] = p.mean(p.err("plurality", ws)) - e0
            else:
                out["empty"].append(g)
    finally:
        G["A"] = real
    return kind, j, out


def full_eng_task(b):
    if b < 0:
        w = G["ones"]
    else:
        rng = np.random.default_rng([SEED, 1000 + b])
        n_s = G["subj"].max() + 1
        w = np.bincount(rng.integers(0, n_s, n_s), minlength=n_s).astype(float)[G["subj"]]
    p = s1.Pipe(w)
    allm = tuple([list(range(len(G["models"])))] * 2)
    return b, p.mean(p.err("plurality", allm), G["eng"])


# ================================================================== admission chart (single process)
def admission(rng):
    p = s1.Pipe(G["ones"])
    open_pool = [i for i in range(len(G["models"])) if not G["api"][i]]
    port = (p.greedy(0, 5, open_pool), p.greedy(1, 5, open_pool))
    key, fold, A = G["key"], G["fold"], G["A"]
    base_err, gains = {}, {(a, r): [] for a in GRID_A for r in GRID_R}
    n_items = {}
    for f in (0, 1):
        fi, ev = fold == f, fold != f
        acc_fit = s1.wacc(A[port[f]][:, fi], key[fi], np.ones(fi.sum()))
        Ae, ke = A[port[f]][:, ev], key[ev]
        s_p = (s1.onehot(Ae) * (1.0 + s1.TIE_EPS * acc_fit[:, None, None])).sum(0)
        plural = np.argmax(s_p, -1)

        def err(s):
            return ((np.argmax(s, -1) != ke) | (s.max(-1) <= 0)).astype(float)
        base = err(s_p)
        base_err[f] = float(base.mean())
        n_items[f] = int(ev.sum())
        Ao = A[open_pool][:, ev]
        fit = v2.fit_1plg((Ao == ke[None, :]).astype(float))
        wrong = v1.pooled_wrong(Ao, ke)
        b_hat = fit[2]
        n = len(ke)
        for a in GRID_A:
            lo, hi = -10.0, 10.0
            for _ in range(60):
                mid = (lo + hi) / 2
                if v2.p_of_theta("1plg", mid, None, b_hat).mean() < a:
                    lo = mid
                else:
                    hi = mid
            pc = v2.p_of_theta("1plg", (lo + hi) / 2, None, b_hat)
            for r in GRID_R:
                for _ in range(DRAWS):
                    copy = rng.random(n) < r
                    own = np.where(rng.random(n) < pc, ke, v1.sample_categorical(rng, wrong))
                    cand = np.where(copy, plural, own)
                    oh = np.zeros((n, s1.M_OPT))
                    oh[np.arange(n), cand] = 1.0 + s1.TIE_EPS * a
                    gains[(a, r)].append((f, float(base.mean() - err(s_p + oh).mean())))
    tot = n_items[0] + n_items[1]
    out = {"portfolio": [[G["models"][i] for i in s] for s in port],
           "base_error": (base_err[0] * n_items[0] + base_err[1] * n_items[1]) / tot, "cells": {}}
    for (a, r), v in gains.items():
        by_f = {f: [g for ff, g in v if ff == f] for f in (0, 1)}
        per_draw = [(by_f[0][d] * n_items[0] + by_f[1][d] * n_items[1]) / tot for d in range(DRAWS)]
        out["cells"][f"a={a},r={r}"] = {"gain": float(np.mean(per_draw)), "sd_over_draws": float(np.std(per_draw))}
    return out


# ================================================================== driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(HERE), "data"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sens-boot", type=int, default=100)
    ap.add_argument("--quick", action="store_true", help="code test: 2 resamples, 2 null panels, 2 draws")
    args = ap.parse_args()
    global DRAWS
    suffix = "_synthetic" if args.synthetic else ""
    n_null, n_sens = (2, 2) if args.quick else (20, args.sens_boot)
    if args.quick:
        DRAWS = 2
    g = s1.load_synthetic() if args.synthetic else s1.load_real(args.data)
    rng = np.random.default_rng(SEED)
    fold = np.zeros(len(g["key"]), dtype=int)
    for s in np.unique(g["subj"]):
        ids = np.where(g["subj"] == s)[0]
        fold[rng.permutation(ids)[: len(ids) // 2]] = 1
    g["fold"] = fold
    g["ones"] = np.ones(len(g["key"]))
    c = (g["A"] == g["key"][None, :]).astype(float)
    g["fit1plg_p"] = v2.fit_1plg(c)[3]
    g["pooled_wrong"] = v1.pooled_wrong(g["A"], g["key"])
    G.update(g)
    t0 = time.time()
    out = {}
    lines = [f"Cycle 8 S1b supplementary outputs{' (SYNTHETIC CODE TEST, not a result)' if args.synthetic else ''}",
             f"models {len(g['models'])}, items {len(g['key'])}"]

    # ---- 0. check of the copied decision code against the stored S1 outputs
    if not args.synthetic:
        s1out = json.load(open(os.path.join(HERE, "cycle8_s1_analysis.json")))
        boots = [json.loads(l)["res"] for l in open(os.path.join(HERE, "cycle8_s1_bootstrap.jsonl"))]
        boots = [r for _, r in sorted(zip([json.loads(l)["b"] for l in open(os.path.join(HERE, "cycle8_s1_bootstrap.jsonl"))], boots))]
        d = decide(s1out["point"], boots)
        same = (d["H2"]["verdict"] == s1out["H2"]["verdict"] and d["H3"]["verdict"] == s1out["H3"]["verdict"]
                and abs(d["H3"]["gain_vs_cap"] - s1out["H3"]["gain_vs_cap"]) < 1e-12
                and np.allclose(d["H3"]["gain_ci"], s1out["H3"]["gain_ci"]))
        out["decision_code_check"] = bool(same)
        lines.append(f"\nDecision-code check against S1 outputs: {'reproduced' if same else 'MISMATCH'} "
                     f"(H2 {d['H2']['verdict']}, H3 {d['H3']['verdict']}, gain {100*d['H3']['gain_vs_cap']:.2f})")
        if not same:
            print("\n".join(lines))
            sys.exit("decision code does not reproduce S1; stopping")
        # ---- 1. RAW intervals from the stored resamples
        pt = s1out["point"]
        raw = {}
        for k in sorted(pt):
            if not k.startswith("D/prim/dE/"):
                continue
            g_ = k.split("/", 3)[3]
            vals = [(r["D/prim/full"] + r[k]) / r["D/prim/full"] for r in boots if k in r and r.get("D/prim/full")]
            raw[g_] = {"RAW": (pt["D/prim/full"] + pt[k]) / pt["D/prim/full"],
                       "ci": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))] if vals else None,
                       "resamples_used": len(vals), "resamples_dropped": len(boots) - len(vals)}
        out["RAW"] = raw
        lines.append("\nRAW (primary portfolio), 95% percentile CI, resamples used:")
        lines += [f"  {g_}: {v['RAW']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}], {v['resamples_used']}" for g_, v in raw.items()]
        out["observed_dE"] = {k.split("/", 3)[3]: v for k, v in pt.items() if k.startswith("D/prim/dE/")}
    print(f"part 0-1 done in {time.time() - t0:.0f} s", flush=True)

    # ---- 2. admission chart
    adm = admission(np.random.default_rng([SEED, 5000]))
    out["admission"] = adm
    lines.append(f"\nAdmission chart (MODEL-BASED, 1PL-g fit to open-weight panel): base = open-weight portfolio "
                 f"{adm['portfolio'][0]} / {adm['portfolio'][1]}, base plurality error {100*adm['base_error']:.2f}")
    for k, v in adm["cells"].items():
        lines.append(f"  {k}: gain {100*v['gain']:.2f} pp (sd over draws {100*v['sd_over_draws']:.2f})")
    print(f"admission done in {time.time() - t0:.0f} s", flush=True)

    with Pool(args.workers, initializer=init_worker, initargs=(g,)) as pool:
        # ---- 3. full panel on the engineering subset
        fe = dict(pool.map(full_eng_task, [-1] + list(range(n_sens if args.quick else 200))))
        vals = [fe[b] for b in fe if b >= 0]
        out["full_panel_eng"] = {"point": fe[-1], "ci": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]}
        lines.append(f"\nFull 70-model plurality, engineering subset, full availability: {100*fe[-1]:.2f} "
                     f"[{100*out['full_panel_eng']['ci'][0]:.2f}, {100*out['full_panel_eng']['ci'][1]:.2f}]")
        print(f"full-panel engineering done in {time.time() - t0:.0f} s", flush=True)

        # ---- 4. delta E under N1 and N2
        res = {"N1": [], "N2": []}
        for kind, j, r in pool.imap_unordered(null_task, [(k, j) for k in ("N1", "N2") for j in range(n_null)]):
            res[kind].append(r)
        out["null_dE"] = res
        lines.append("\nDelta E of the re-selected primary portfolio on simulated null panels (pp; mean [5th, 95th]):")
        for kind in ("N1", "N2"):
            api = [r["dE"]["api-only"] for r in res[kind] if "api-only" in r["dE"]]
            mx = [max([v for g_, v in r["dE"].items() if g_.startswith("dev:")] or [np.nan]) for r in res[kind]]
            mx = [v for v in mx if not np.isnan(v)]
            emp = sum("api-only" in r["empty"] for r in res[kind])
            full = [r["full"] for r in res[kind]]

            def q(v):
                return (f"{100*np.mean(v):.2f} [{100*np.quantile(v, 0.05):.2f}, {100*np.quantile(v, 0.95):.2f}] (n={len(v)})"
                        if v else "none")
            lines.append(f"  {kind}: full error {q(full)}; API-only dE {q(api)}, emptied {emp}; largest developer dE {q(mx)}")
        if "observed_dE" in out:
            od = out["observed_dE"]
            lines.append(f"  observed: full error {100*s1out['point']['D/prim/full']:.2f}; API-only dE {100*od.get('api-only', np.nan):.2f}; "
                         f"largest developer dE {100*max(v for k, v in od.items() if k.startswith('dev:')):.2f}")
        print(f"null panels done in {time.time() - t0:.0f} s", flush=True)

        # ---- 5. H2/H3 sensitivity analyses
        acc_all = (g["A"] == g["key"][None, :]).mean(1)
        largest = np.bincount(g["dev"]).argmax()
        M = len(g["models"])
        subs = {"strict_panel": [i for i, m in enumerate(g["models"]) if m in g["strict"]],
                "without_largest_developer": [i for i in range(M) if g["dev"][i] != largest],
                "without_accuracy_below_40pct": [i for i in range(M) if acc_all[i] >= 0.40]}
        path = os.path.join(HERE, f"cycle8_s1b_sens_bootstrap{suffix}.jsonl")
        done = {}
        if os.path.exists(path):
            for l in open(path):
                r = json.loads(l)
                done[(r["name"], r["b"])] = r["res"]
        jobs = [(nm, sub, b) for nm, sub in subs.items() for b in [-1] + list(range(n_sens)) if (nm, b) not in done]
        with open(path, "a") as fh:
            for i, (nm, b, r) in enumerate(pool.imap_unordered(sens_task, jobs), 1):
                fh.write(json.dumps({"name": nm, "b": b, "res": r}) + "\n")
                fh.flush()
                done[(nm, b)] = r
                if i % 20 == 0 or i == len(jobs):
                    print(f"  sensitivity {i}/{len(jobs)}, {time.time() - t0:.0f} s", flush=True)
    out["sensitivity"] = {}
    lines.append(f"\nH2/H3 sensitivity analyses ({n_sens} resamples each; largest developer: {g['dev_names'][largest]}):")
    for nm, sub in subs.items():
        d = decide(done[(nm, -1)], [done[(nm, b)] for b in range(n_sens)])
        out["sensitivity"][nm] = {"models": len(sub), **d}
        h2 = d["H2"]["by_n"]
        lines.append(f"  {nm} ({len(sub)} models): H2 {d['H2']['verdict']} ("
                     + "; ".join(f"n={n}: {v['full_winner']}->{v['worst_winner']}, plurality {100*v['full']['plurality']:.2f}/"
                                 f"{100*v['worst']['plurality']:.2f}" for n, v in h2.items())
                     + f"); H3 {d['H3']['verdict']} (robust {100*d['H3']['H3/robust/heldout_worst']:.2f}, cap "
                     f"{100*d['H3']['H3/cap/heldout_worst']:.2f}, gain {100*d['H3']['gain_vs_cap']:.2f} "
                     f"[{100*d['H3']['gain_ci'][0]:.2f}, {100*d['H3']['gain_ci'][1]:.2f}], cost {100*d['H3']['full_cost_vs_cap']:.2f})")
    out["seconds"] = round(time.time() - t0)
    lines.append(f"seconds {out['seconds']}")
    json.dump(out, open(os.path.join(HERE, f"cycle8_s1b_supplementary{suffix}.json"), "w"), indent=1, default=str)
    text = "\n".join(lines)
    open(os.path.join(HERE, f"cycle8_s1b_supplementary{suffix}.out"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
