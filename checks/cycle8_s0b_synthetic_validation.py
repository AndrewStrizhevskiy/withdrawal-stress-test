"""Cycle 8, stage S0 re-run under Amendment A2 (protocol Section 13).

Reuses the generator and helpers of checks/cycle8_s0_synthetic_validation.py (unchanged, its S0 result is in
protocol Section 12). New here: H1 as excess same-wrong-answer agreement within against between developers over
N2; two N2 variants (2PL and 1PL with guessing fixed at 1/4); replicate-reliability criterion 3c.
No real data are used. Run from the project folder:

    .venv/bin/python checks/cycle8_s0b_synthetic_validation.py --smoke
    python3 checks/cycle8_s0b_synthetic_validation.py --workers 14       # on the server
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cycle8_s0_synthetic_validation as v1  # noqa: E402

M_OPT, TIE_EPS = v1.M_OPT, v1.TIE_EPS
DEV, API, N_MODELS, N_DEV, STRUCTURE = v1.DEV, v1.API, v1.N_MODELS, v1.N_DEV, v1.STRUCTURE
MASTER_SEED = 20261114          # different from the first S0 run
GUESS = 0.25
VARIANTS = ("2pl", "1plg")
sigmoid = v1.sigmoid

DEV_MULTI = [d for d in range(N_DEV) if (DEV == d).sum() >= 2]
IU = np.triu_indices(N_MODELS, 1)
SAME = (DEV[IU[0]] == DEV[IU[1]])


# ------------------------------------------------------------------ N2 variants
def fit_2pl_warm(c, init=None, iters=25):
    if init is None:
        return v1.fit_2pl(c, iters)
    theta, a, b = (x.copy() for x in init)
    for _ in range(iters):
        p = sigmoid(a[None, :] * (theta[:, None] - b[None, :]))
        g = ((c - p) * a[None, :]).sum(1) - 0.01 * theta
        h = (p * (1 - p) * a[None, :] ** 2).sum(1) + 0.01
        theta = theta + np.clip(g / h, -1, 1)
        theta = (theta - theta.mean()) / theta.std()
        p = sigmoid(a[None, :] * (theta[:, None] - b[None, :]))
        dth = theta[:, None] - b[None, :]
        r, w = c - p, p * (1 - p)
        ga = (r * dth).sum(0) - (a - 1.0)
        gb = (-r * a[None, :]).sum(0) - b / 9.0
        haa = (w * dth ** 2).sum(0) + 1.0
        hbb = (w * a[None, :] ** 2).sum(0) + 1 / 9.0
        hab = -(w * a[None, :] * dth).sum(0) + r.sum(0)
        det = haa * hbb - hab ** 2
        ok = det > 1e-8
        da = np.where(ok, (hbb * ga - hab * gb) / np.where(ok, det, 1), ga / haa)
        db = np.where(ok, (haa * gb - hab * ga) / np.where(ok, det, 1), gb / hbb)
        a = np.clip(a + np.clip(da, -0.5, 0.5), 0.05, 5.0)
        b = np.clip(b + np.clip(db, -1.0, 1.0), -8.0, 8.0)
    p = sigmoid(a[None, :] * (theta[:, None] - b[None, :]))
    return theta, a, b, p


def fit_1plg(c, init=None, iters=30):
    if init is None:
        acc_m = np.clip((c.mean(1) - GUESS) / (1 - GUESS), 0.02, 0.98)
        acc_i = np.clip((c.mean(0) - GUESS) / (1 - GUESS), 0.02, 0.98)
        theta = np.log(acc_m / (1 - acc_m))
        theta = theta - theta.mean()
        b = -np.log(acc_i / (1 - acc_i))
    else:
        theta, b = init[0].copy(), init[2].copy()
    for _ in range(iters):
        for which in ("theta", "b"):
            s = sigmoid(theta[:, None] - b[None, :])
            P = np.clip(GUESS + (1 - GUESS) * s, 1e-6, 1 - 1e-6)
            dP = (1 - GUESS) * s * (1 - s)
            score = (c - P) / (P * (1 - P)) * dP
            info = dP ** 2 / (P * (1 - P))
            if which == "theta":
                theta = theta + np.clip((score.sum(1) - 0.01 * theta) / (info.sum(1) + 0.01), -1, 1)
                theta = theta - theta.mean()
            else:
                b = b + np.clip((-score.sum(0) - b / 9.0) / (info.sum(0) + 1 / 9.0), -1, 1)
                b = np.clip(b, -10, 10)
    p = GUESS + (1 - GUESS) * sigmoid(theta[:, None] - b[None, :])
    return theta, np.ones_like(b), b, p


def fit_variant(variant, c, init=None):
    if variant == "2pl":
        return fit_2pl_warm(c, init, 25 if init is None else 6)
    return fit_1plg(c, init, 30 if init is None else 8)


def p_of_theta(variant, th, a, b):
    if variant == "2pl":
        return sigmoid(a * (th - b))
    return GUESS + (1 - GUESS) * sigmoid(th - b)


# ------------------------------------------------------------------ new H1
def h1_excess(ans, key, p, wrong, matched=True, tol=0.03):
    """Excess same-wrong-answer agreement over N2. matched=True (A2.1): each within-developer pair is compared
    with between-developer pairs whose two accuracies match its two accuracies within tol (widened to 0.12)."""
    n = ans.shape[1]
    S = np.zeros((N_MODELS, N_MODELS))
    for o in range(M_OPT):
        ind = ((ans == o) & (key[None, :] != o)).astype(np.float64)
        S += ind @ ind.T
    q2 = (wrong ** 2).sum(1)
    E = ((1 - p) * q2[None, :]) @ (1 - p).T
    x = ((S - E) / n)[IU]
    if not matched:
        within = np.mean([x[SAME & (DEV[IU[0]] == d)].mean() for d in DEV_MULTI])
        return float(within - x[~SAME].mean())
    acc = (ans == key[None, :]).mean(1)
    ai, aj = acc[IU[0]], acc[IU[1]]
    bk, bl, xb = ai[~SAME], aj[~SAME], x[~SAME]
    per_dev = []
    for d in DEV_MULTI:
        idx = np.where(SAME & (DEV[IU[0]] == d))[0]
        ex = []
        for q in idx:
            t = tol
            while t <= 0.12 + 1e-9:
                m = ((np.abs(bk - ai[q]) <= t) & (np.abs(bl - aj[q]) <= t)) | ((np.abs(bk - aj[q]) <= t) & (np.abs(bl - ai[q]) <= t))
                if m.sum() >= 5:
                    ex.append(x[q] - xb[m].mean())
                    break
                t *= 2
        if ex:
            per_dev.append(np.mean(ex))
    return float(np.mean(per_dev))


def h1_excess_unmatched(ans, key, p, wrong):
    n = ans.shape[1]
    S = np.zeros((N_MODELS, N_MODELS))
    for o in range(M_OPT):
        ind = ((ans == o) & (key[None, :] != o)).astype(np.float64)
        S += ind @ ind.T
    q2 = (wrong ** 2).sum(1)
    E = ((1 - p) * q2[None, :]) @ (1 - p).T
    x = ((S - E) / n)[IU]
    within = np.mean([x[SAME & (DEV[IU[0]] == d)].mean() for d in DEV_MULTI])
    return float(within - x[~SAME].mean())


def h1_test(rng, ans, key, variant, k_null):
    c = (ans == key[None, :]).astype(float)
    fit = fit_variant(variant, c)
    wrong = v1.pooled_wrong(ans, key)
    t_obs = h1_excess(ans, key, fit[3], wrong)
    t_null = []
    for _ in range(k_null):
        a0 = v1.simulate_n2(rng, fit[3], wrong, key)
        c0 = (a0 == key[None, :]).astype(float)
        f0 = fit_variant(variant, c0, init=fit[:3])
        t_null.append(h1_excess(a0, key, f0[3], v1.pooled_wrong(a0, key)))
    t_null = np.array(t_null)
    p = float((1 + (t_null >= t_obs).sum()) / (1 + k_null))
    return t_obs, p, float(t_null.mean()), fit, wrong


# ------------------------------------------------------------------ admission chart with a chosen N2 variant
def admission_gain_variant(rng, ans_ev, key, weights, port, a_t, r, how, items_ev, fit, wrong, world, variant, draws):
    if how == "true":
        return v1.admission_gain(rng, ans_ev, key, weights, port, a_t, r, "true", items_ev, None, world, draws)
    s_p = v1.contributions(ans_ev[port], weights[port]).sum(0)
    base = v1.plurality_errors(s_p, key).mean()
    plural = np.argmax(s_p, axis=-1)
    n = len(key)
    _, a_hat, b_hat, _ = fit
    lo, hi = -10.0, 10.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if p_of_theta(variant, mid, a_hat, b_hat).mean() < a_t:
            lo = mid
        else:
            hi = mid
    pc = p_of_theta(variant, (lo + hi) / 2, a_hat, b_hat)
    gains = []
    for _ in range(draws):
        copy = rng.random(n) < r
        own = np.where(rng.random(n) < pc, key, v1.sample_categorical(rng, wrong))
        cand = np.where(copy, plural, own)
        oh = np.zeros((n, M_OPT))
        oh[np.arange(n), cand] = 1.0 + TIE_EPS * a_t
        gains.append(base - v1.plurality_errors(s_p + oh, key).mean())
    return float(np.mean(gains))


# ------------------------------------------------------------------ one dataset
def run_dataset(task):
    sigma, rep, cfg = task["sigma"], task["rep"], task["cfg"]
    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, int(round(sigma * 1000)), rep]))
    t0 = time.time()
    world = v1.draw_world(rng, sigma)
    items = v1.draw_items(rng, cfg["n_subj"], cfg["per_subj"])
    ans = v1.simulate_answers(rng, world, items)
    half = np.zeros(len(items["b"]), dtype=bool)
    for s in range(cfg["n_subj"]):
        ids = np.where(items["subj"] == s)[0]
        half[rng.permutation(ids)[: len(ids) // 2]] = True
    sel, ev = np.where(half)[0], np.where(~half)[0]
    key_sel, key_ev = items["key"][sel], items["key"][ev]
    ans_sel, ans_ev = ans[:, sel], ans[:, ev]
    acc_sel = (ans_sel == key_sel[None, :]).mean(1)
    items_ev = {k: v[ev] for k, v in items.items()}
    out = {"sigma": sigma, "rep": rep}
    # replicate panel on the same evaluation items (criterion 3c)
    ans_rep = v1.simulate_answers(rng, world, items_ev)
    prop_rep = (ans_rep == key_ev[None, :]).mean(0)
    fits = {}
    for var in VARIANTS:
        t, p, tn, fit, wrong = h1_test(rng, ans_ev, key_ev, var, cfg["k_null"])
        fits[var] = (fit, wrong)
        out[var] = {"T": t, "p": p, "T_null_mean": tn, "rel": v1.spearman(fit[2], -prop_rep)}

    if sigma in cfg["recovery_sigmas"]:
        port = v1.greedy_portfolio(ans_sel, key_sel, acc_sel, list(range(N_MODELS)), 5)
        scen = v1.scenario_sets(port)
        fresh = v1.draw_items(rng, cfg["fresh_subj"], cfg["per_subj"])
        ans_fr = v1.simulate_answers(rng, world, fresh)
        e_full_fr = v1.per_item_errors(ans_fr, fresh["key"], acc_sel, port).mean()
        e_full_ev = v1.per_item_errors(ans_ev, key_ev, acc_sel, port)
        rows = []
        for name, W in scen:
            rest = [i for i in port if i not in W]
            d_ev = v1.per_item_errors(ans_ev, key_ev, acc_sel, rest) - e_full_ev
            lo, hi = v1.cluster_boot_ci(d_ev, items_ev["subj"], rng, cfg["boot"])
            truth = v1.per_item_errors(ans_fr, fresh["key"], acc_sel, rest).mean() - e_full_fr
            row = {"scenario": name, "est": float(d_ev.mean()), "lo": lo, "hi": hi, "truth": float(truth)}
            if sigma == 0.0:
                for var in VARIANTS:
                    fit, wrong = fits[var]
                    pred = []
                    for _ in range(cfg["k_pred"]):
                        a0 = v1.simulate_n2(rng, fit[3], wrong, key_ev)
                        pred.append(v1.per_item_errors(a0, key_ev, acc_sel, rest).mean()
                                    - v1.per_item_errors(a0, key_ev, acc_sel, port).mean())
                    row["pred_" + var] = float(np.mean(pred))
            rows.append(row)
        out["dE"] = rows
        if len(scen) >= 2:
            (n1, W1), (n2, W2) = scen[0], scen[1]
            r1 = [i for i in port if i not in W1]
            r2 = [i for i in port if i not in W2]
            r12 = [i for i in port if i not in W1 and i not in W2]
            if r12:
                pe = v1.per_item_errors
                ji = pe(ans_ev, key_ev, acc_sel, r12) - pe(ans_ev, key_ev, acc_sel, r1) - pe(ans_ev, key_ev, acc_sel, r2) + e_full_ev
                lo, hi = v1.cluster_boot_ci(ji, items_ev["subj"], rng, cfg["boot"])
                tr = (pe(ans_fr, fresh["key"], acc_sel, r12).mean() - pe(ans_fr, fresh["key"], acc_sel, r1).mean()
                      - pe(ans_fr, fresh["key"], acc_sel, r2).mean() + e_full_fr)
                out["JI"] = {"pair": [n1, n2], "est": float(ji.mean()), "lo": lo, "hi": hi, "truth": float(tr)}
        if sigma == cfg["admission_sigma"]:
            port_o = v1.greedy_portfolio(ans_sel, key_sel, acc_sel, [i for i in range(N_MODELS) if not API[i]], 5)
            adm = []
            for a_t in (0.5, 0.7, 0.85):
                for r in (0.0, 0.3, 0.6):
                    row = {"a": a_t, "r": r}
                    row["true"] = admission_gain_variant(rng, ans_ev, key_ev, acc_sel, port_o, a_t, r, "true", items_ev,
                                                         None, None, world, None, cfg["adm_draws"])
                    for var in VARIANTS:
                        fit, wrong = fits[var]
                        row[var] = admission_gain_variant(rng, ans_ev, key_ev, acc_sel, port_o, a_t, r, "n2", items_ev,
                                                          fit, wrong, world, var, cfg["adm_draws"])
                    adm.append(row)
            out["admission"] = adm
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ------------------------------------------------------------------ criteria
def summarise(rows, cfg):
    by = {}
    for r in rows:
        by.setdefault(r["sigma"], []).append(r)
    null, rec = by.get(0.0, []), by.get(cfg["admission_sigma"], [])
    lines = ["datasets: " + ", ".join(f"sigma={s}: {len(v)}" for s, v in sorted(by.items()))]
    cover = [(x["lo"] <= x["truth"] <= x["hi"]) for r in rec for x in r.get("dE", [])]
    cover_ji = [(r["JI"]["lo"] <= r["JI"]["truth"] <= r["JI"]["hi"]) for r in rec if "JI" in r]
    c3a = len(cover) > 0 and np.mean(cover) >= 0.90
    c3b = len(cover_ji) > 0 and np.mean(cover_ji) >= 0.90
    lines.append(f"\nShared by both N2 variants:\n  3a dE interval coverage {np.mean(cover):.3f} over {len(cover)} -> {'pass' if c3a else 'FAIL'}")
    lines.append(f"  3b JI interval coverage {np.mean(cover_ji):.3f} over {len(cover_ji)} -> {'pass' if c3b else 'FAIL'}")
    verdict = {}
    for var in VARIANTS:
        lines.append(f"\nN2 variant {var}")
        size = np.mean([r[var]["p"] <= 0.05 for r in null])
        c1 = size <= 0.06
        lines.append(f"  1  size: rejection rate under sigma=0 {size:.3f} (n={len(null)}), mean T {np.mean([r[var]['T'] for r in null]):.5f} -> {'pass' if c1 else 'FAIL'}")
        mde, mde_s = None, None
        for s in cfg["power_sigmas"]:
            v = by.get(s, [])
            if not v:
                continue
            pw = np.mean([r[var]["p"] <= 0.05 for r in v])
            mt = np.mean([r[var]["T"] for r in v])
            lines.append(f"     sigma={s}: power {pw:.2f}, mean T {mt:.5f} (n={len(v)})")
            if mde is None and pw >= 0.8:
                mde, mde_s = mt, s
        c2 = mde is not None
        lines.append(f"  2  MDE: " + (f"mean T {mde:.5f} at sigma={mde_s} -> pass" if c2 else "no sigma reaches power 0.8 -> FAIL"))
        rel = np.mean([r[var]["rel"] for r in rows])
        c3c = rel >= 0.85
        lines.append(f"  3c replicate reliability (mean Spearman) {rel:.3f} -> {'pass' if c3c else 'FAIL'}")
        ok = []
        for r in null:
            for x in r.get("dE", []):
                if "pred_" + var in x:
                    ok.append(abs(x["pred_" + var] - x["truth"]) <= max(0.005, (x["hi"] - x["lo"]) / 2))
        share = float(np.mean(ok)) if ok else float("nan")
        c3d = share >= 0.90
        lines.append(f"  3d N2-predicted dE within tolerance {share:.3f} over {len(ok)} -> {'pass' if c3d else 'FAIL'}")
        adm = {}
        for r in rec:
            for g in r.get("admission", []):
                adm.setdefault((g["a"], g["r"]), []).append((g[var], g["true"]))
        worst = 0.0
        for (a_t, rr), v in sorted(adm.items()):
            v = np.array(v)
            md = float(np.mean(np.abs(v[:, 0] - v[:, 1])))
            worst = max(worst, md)
            lines.append(f"     admission a={a_t:.2f} r={rr:.1f}: N2 {100*v[:,0].mean():6.2f} | true {100*v[:,1].mean():6.2f} | mean |diff| {100*md:5.2f} pp")
        c3e = len(adm) > 0 and worst <= 0.005
        lines.append(f"  3e admission chart, worst mean |diff| {100*worst:.2f} pp -> {'pass' if c3e else 'FAIL'}")
        allpass = c1 and c2 and c3a and c3b and c3c and c3d and c3e
        verdict[var] = (allpass, share, worst, mde)
        lines.append(f"  -> variant {var}: {'PASS' if allpass else 'FAIL'}")
    passing = [v for v in VARIANTS if verdict[v][0]]
    if not passing:
        lines.append("\nS0 (A2) overall: FAIL (no N2 variant passes all criteria)")
    else:
        best = sorted(passing, key=lambda v: (-verdict[v][1], verdict[v][2]))[0]
        lines.append(f"\nS0 (A2) overall: PASS; selected N2 variant by the pre-stated rule: {best}; "
                     f"minimal detectable effect for H1 = {verdict[best][3]:.5f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--results", default=None)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        cfg = dict(n_subj=12, per_subj=60, fresh_subj=40, k_null=10, k_pred=5, boot=100, adm_draws=3,
                   n_null=4, n_power=2, power_sigmas=[0.3, 0.5], recovery_sigmas=[0.0, 0.3], admission_sigma=0.3)
        suffix = "_smoke"
    else:
        cfg = dict(n_subj=57, per_subj=246, fresh_subj=244, k_null=100, k_pred=20, boot=300, adm_draws=30,
                   n_null=200, n_power=100, power_sigmas=[0.1, 0.2, 0.3, 0.5], recovery_sigmas=[0.0, 0.3],
                   admission_sigma=0.3)
        suffix = ""
    here = os.path.dirname(os.path.abspath(__file__))
    res_path = args.results or os.path.join(here, f"cycle8_s0b_results{suffix}.jsonl")
    out_path = os.path.join(here, f"cycle8_s0b_synthetic_validation{suffix}.out")
    tasks = [{"sigma": 0.0, "rep": i, "cfg": cfg} for i in range(cfg["n_null"])]
    for s in cfg["power_sigmas"]:
        tasks += [{"sigma": s, "rep": i, "cfg": cfg} for i in range(cfg["n_power"])]
    done = {}
    if os.path.exists(res_path):
        for line in open(res_path):
            r = json.loads(line)
            done[(r["sigma"], r["rep"])] = r
    todo = [] if args.summary_only else [t for t in tasks if (t["sigma"], t["rep"]) not in done]
    print(f"{len(tasks)} datasets, {len(done)} done, {len(todo)} to run, {args.workers} workers", flush=True)
    t0 = time.time()
    if todo:
        with open(res_path, "a") as f, Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(run_dataset, todo), 1):
                f.write(json.dumps(r) + "\n")
                f.flush()
                done[(r["sigma"], r["rep"])] = r
                if i % 20 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} done, {time.time() - t0:.0f} s", flush=True)
    rows = [done[(t["sigma"], t["rep"])] for t in tasks if (t["sigma"], t["rep"]) in done]
    if len(rows) < len(tasks):
        print(f"{len(tasks) - len(rows)} datasets missing; no summary written")
        return
    text = (f"Cycle 8 S0 under Amendment A2{' (SMOKE TEST, not a result)' if args.smoke else ''}\n"
            f"config: {json.dumps(cfg)}\nmodels {N_MODELS}, developers {N_DEV}, developers with >=2 models "
            f"{len(DEV_MULTI)}\n" + summarise(rows, cfg))
    open(out_path, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
