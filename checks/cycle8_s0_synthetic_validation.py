"""Cycle 8, stage S0: synthetic validation of the withdrawal stress-test methods.

Protocol: reports/Cycle 8 withdrawal stress test protocol.md, Section 6 and Amendment A1.
No real data are used. Run from the project folder:

    .venv/bin/python checks/cycle8_s0_synthetic_validation.py            # full run (about an hour, 6 processes)
    .venv/bin/python checks/cycle8_s0_synthetic_validation.py --smoke    # tiny run to test the code

Outputs: checks/cycle8_s0_results.jsonl (one row per simulated dataset) and
checks/cycle8_s0_synthetic_validation.out (summary and pass/fail of the three criteria).
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

MASTER_SEED = 20261113
M_OPT = 4                      # answer options (MMLU)
KEY_ERROR = 0.03               # share of wrong keys
TIE_EPS = 1e-6                 # plurality tie-break by the sum of voters' accuracies

# provisional deduplicated panel (Amendment A1): developer, number of models, number of API-only models
STRUCTURE = [("google", 9, 6), ("meta", 8, 0), ("anthropic", 6, 6), ("mistralai", 6, 0), ("openai", 6, 6),
             ("qwen", 6, 0), ("01-ai", 3, 0), ("microsoft", 3, 0), ("ai21", 3, 0), ("amazon", 3, 3),
             ("allenai", 2, 0), ("deepseek-ai", 2, 0), ("cohere", 2, 0), ("writer", 1, 1),
             ("databricks", 1, 0), ("snowflake", 1, 0), ("nvidia", 1, 0), ("upstage", 1, 1)]


def build_structure():
    dev, lin, api = [], [], []
    lcount = 0
    for d, (name, n, n_api) in enumerate(STRUCTURE):
        for j in range(n):
            dev.append(d)
            lin.append(lcount + j // 3)          # lineages of up to three models
            api.append(j < n_api)
        lcount += (n + 2) // 3
    return np.array(dev), np.array(lin), np.array(api, dtype=bool), lcount


DEV, LIN, API, N_LIN = build_structure()
N_MODELS = len(DEV)
N_DEV = len(STRUCTURE)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ------------------------------------------------------------------ generator
def draw_world(rng, sigma):
    """Model-level parameters: target accuracies, abilities calibrated for this sigma."""
    m_d = rng.uniform(0.40, 0.85, N_DEV)
    acc_target = np.clip(m_d[DEV] + rng.normal(0, 0.07, N_MODELS), 0.27, 0.90)
    theta = calibrate_theta(acc_target, sigma, rng)
    return {"theta": theta, "acc_target": acc_target, "sigma": sigma, "w": min(sigma, 0.5)}


def know_prob_target(acc):
    # acc ~ (1-KEY_ERROR)(pk + (1-pk)/4) + KEY_ERROR (1-pk)/4   ->   solve for pk
    return np.clip((acc - 0.25) / ((1 - KEY_ERROR) * 0.75), 1e-3, 0.999)


def calibrate_theta(acc_target, sigma, rng):
    n = 20000
    b = rng.normal(0, 1, n) + rng.normal(0, 0.5, n)            # item difficulty incl. subject shift
    a = np.exp(rng.normal(0, 0.3, n))
    noise = rng.normal(0, 1, n) + rng.normal(0, sigma, n) + rng.normal(0, sigma, n)
    pk = know_prob_target(acc_target)
    lo, hi = np.full(len(pk), -8.0), np.full(len(pk), 8.0)
    for _ in range(40):
        mid = (lo + hi) / 2
        p = np.mean((a[None, :] * (mid[:, None] - b[None, :]) + noise[None, :]) > 0, axis=1)
        lo = np.where(p < pk, mid, lo)
        hi = np.where(p >= pk, mid, hi)
    return (lo + hi) / 2


def draw_items(rng, n_subj, per_subj):
    n = n_subj * per_subj
    subj = np.repeat(np.arange(n_subj), per_subj)
    b = rng.normal(0, 0.5, n_subj)[subj] + rng.normal(0, 1, n)
    a = np.exp(rng.normal(0, 0.3, n))
    y = rng.integers(0, M_OPT, n)
    q = rng.dirichlet(np.ones(M_OPT), n)                       # global guess distribution per item
    key = y.copy()
    err = rng.random(n) < KEY_ERROR
    key[err] = (y[err] + rng.integers(1, M_OPT, err.sum())) % M_OPT
    return {"subj": subj, "b": b, "a": a, "y": y, "q": q, "key": key}


def sample_categorical(rng, probs):
    """probs: (..., K) rows summing to 1."""
    c = np.cumsum(probs, axis=-1)
    u = rng.random(probs.shape[:-1])[..., None]
    return np.minimum((u > c).sum(-1), probs.shape[-1] - 1)


def simulate_answers(rng, world, items, theta=None, dev=DEV, lin=LIN, n_dev=N_DEV, n_lin=N_LIN):
    theta = world["theta"] if theta is None else theta
    sigma, w = world["sigma"], world["w"]
    n = len(items["b"])
    u = rng.normal(0, sigma, (n_dev, n)) if sigma > 0 else np.zeros((n_dev, n))
    v = rng.normal(0, sigma, (n_lin, n)) if sigma > 0 else np.zeros((n_lin, n))
    eps = rng.normal(0, 1, (len(theta), n))
    z = items["a"][None, :] * (theta[:, None] - items["b"][None, :]) + u[dev] + v[lin] + eps
    know = z > 0
    if w > 0:
        dpref = rng.dirichlet(np.ones(M_OPT), (n_dev, n))
        qd = (1 - w) * items["q"][None, :, :] + w * dpref       # (n_dev, n, K)
        guess = sample_categorical(rng, qd[dev])
    else:
        guess = sample_categorical(rng, np.broadcast_to(items["q"], (len(theta), n, M_OPT)))
    return np.where(know, items["y"][None, :], guess).astype(np.int8)


# ------------------------------------------------------------------ aggregation
def contributions(ans, weights):
    onehot = np.zeros(ans.shape + (M_OPT,), dtype=np.float64)
    np.put_along_axis(onehot, ans[..., None].astype(np.int64), 1.0, axis=-1)
    return onehot * (1.0 + TIE_EPS * weights[:, None, None])


def plurality_errors(contrib_sum, key):
    return (np.argmax(contrib_sum, axis=-1) != key).astype(np.float64)


def greedy_portfolio(ans_sel, key_sel, weights, pool, n):
    contrib = contributions(ans_sel[pool], weights[pool])
    chosen, current = [], np.zeros((ans_sel.shape[1], M_OPT))
    for _ in range(n):
        best, best_err = None, 2.0
        for j in range(len(pool)):
            if j in chosen:
                continue
            e = plurality_errors(current + contrib[j], key_sel).mean()
            if e < best_err - 1e-12:
                best, best_err = j, e
        chosen.append(best)
        current = current + contrib[best]
    return [pool[j] for j in chosen]


# ------------------------------------------------------------------ H1 statistic
def h1_statistic(ans, key, acc, rng, draws):
    contrib = contributions(ans, acc)
    s_full = contrib.sum(0)
    e0 = plurality_errors(s_full, key).mean()
    zs = []
    for d in range(N_DEV):
        members = np.where(DEV == d)[0]
        if len(members) < 2:
            continue
        others = np.where(DEV != d)[0]
        e_obs = plurality_errors(s_full - contrib[members].sum(0), key).mean() - e0
        pools = []
        for j in members:
            cand = others[np.abs(acc[others] - acc[j]) <= 0.02]
            if len(cand) == 0:
                cand = others[np.argsort(np.abs(acc[others] - acc[j]))[:3]]
            pools.append(cand)
        de = np.empty(draws)
        for t in range(draws):
            pick = set()
            for cand in pools:
                free = [c for c in cand if c not in pick]
                pick.add(rng.choice(free) if free else rng.choice(cand))
            idx = np.array(sorted(pick))
            de[t] = plurality_errors(s_full - contrib[idx].sum(0), key).mean() - e0
        sd = max(de.std(ddof=1), 1e-4)
        zs.append((e_obs - de.mean()) / sd)
    return float(np.mean(zs))


# ------------------------------------------------------------------ N2: 2PL with pooled wrong answers
def fit_2pl(c, iters=25):
    m, n = c.shape
    acc_m = np.clip(c.mean(1), 0.02, 0.98)
    acc_i = np.clip(c.mean(0), 0.02, 0.98)
    theta = np.log(acc_m / (1 - acc_m))
    theta = (theta - theta.mean()) / theta.std()
    b = -np.log(acc_i / (1 - acc_i))
    a = np.ones(n)
    for _ in range(iters):
        p = sigmoid(a[None, :] * (theta[:, None] - b[None, :]))
        g = ((c - p) * a[None, :]).sum(1) - 0.01 * theta
        h = (p * (1 - p) * a[None, :] ** 2).sum(1) + 0.01
        theta = theta + np.clip(g / h, -1, 1)
        theta = (theta - theta.mean()) / theta.std()
        p = sigmoid(a[None, :] * (theta[:, None] - b[None, :]))
        dth = theta[:, None] - b[None, :]
        r = c - p
        w = p * (1 - p)
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


def pooled_wrong(ans, key):
    n = ans.shape[1]
    counts = np.full((n, M_OPT), 0.5)
    for k in range(M_OPT):
        counts[:, k] += (ans == k).sum(0)
    counts[np.arange(n), key] = 0.0
    return counts / counts.sum(1, keepdims=True)


def simulate_n2(rng, p, wrong, key, rows=None):
    pr = p if rows is None else p[rows]
    correct = rng.random(pr.shape) < pr
    wrong_ans = sample_categorical(rng, np.broadcast_to(wrong, pr.shape + (M_OPT,)))
    return np.where(correct, key[None, :], wrong_ans).astype(np.int8)


# ------------------------------------------------------------------ recovery helpers
def scenario_sets(portfolio):
    sc = []
    for d in sorted(set(DEV[portfolio])):
        sc.append(("dev:" + STRUCTURE[d][0], [i for i in portfolio if DEV[i] == d]))
    api = [i for i in portfolio if API[i]]
    if api:
        sc.append(("api-only", api))
    return [(name, W) for name, W in sc if len(W) < len(portfolio)]


def per_item_errors(ans, key, weights, members):
    if len(members) == 0:
        return None
    s = contributions(ans[members], weights[members]).sum(0)
    return plurality_errors(s, key)


def cluster_boot_ci(values, subj, rng, B):
    n_s = subj.max() + 1
    sums = np.bincount(subj, weights=values, minlength=n_s)
    cnts = np.bincount(subj, minlength=n_s).astype(float)
    idx = rng.integers(0, n_s, (B, n_s))
    stats = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def solve_theta_for_acc(target, a, b, extra_noise_sd=0.0, rng=None):
    """Ability giving mean know/correct probability `target` on these items (2PL mean or latent model)."""
    lo, hi = -10.0, 10.0
    if extra_noise_sd == 0.0:
        for _ in range(50):
            mid = (lo + hi) / 2
            if sigmoid(a * (mid - b)).mean() < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    noise = rng.normal(0, 1, (8, len(b))) + rng.normal(0, extra_noise_sd, (8, len(b))) * math.sqrt(2)
    for _ in range(50):
        mid = (lo + hi) / 2
        if ((a * (mid - b))[None, :] + noise > 0).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def admission_gain(rng, ans_eval, key, weights, portfolio, a_target, r, how, items_eval, n2, world, draws):
    """Gain in plurality accuracy from adding a candidate to the portfolio (eval half)."""
    s_p = contributions(ans_eval[portfolio], weights[portfolio]).sum(0)
    base = plurality_errors(s_p, key).mean()
    plural = np.argmax(s_p, axis=-1)
    n = len(key)
    gains = []
    if how == "n2":
        _, a_hat, b_hat, _, wrong = n2
        pk = a_target                                    # candidate correct probability on keyed items
        th = solve_theta_for_acc(pk, a_hat, b_hat)
        pc = sigmoid(a_hat * (th - b_hat))
    else:
        pk = know_prob_target(np.array([a_target]))[0]
        th = solve_theta_for_acc(pk, items_eval["a"], items_eval["b"], world["sigma"], rng)
    for _ in range(draws):
        copy = rng.random(n) < r
        if how == "n2":
            own = np.where(rng.random(n) < pc, key, sample_categorical(rng, wrong))
        else:
            one_dev = {"theta": np.array([th]), "sigma": world["sigma"], "w": world["w"]}
            own = simulate_answers(rng, one_dev, items_eval, theta=np.array([th]), dev=np.array([0]),
                                   lin=np.array([0]), n_dev=1, n_lin=1)[0]
        cand = np.where(copy, plural, own)
        oh = np.zeros((n, M_OPT))
        oh[np.arange(n), cand] = 1.0 + TIE_EPS * a_target
        gains.append(base - plurality_errors(s_p + oh, key).mean())
    return float(np.mean(gains))


# ------------------------------------------------------------------ one simulated dataset
def run_dataset(task):
    sigma, rep, cfg = task["sigma"], task["rep"], task["cfg"]
    ss = np.random.SeedSequence([MASTER_SEED, int(round(sigma * 1000)), rep])
    rng = np.random.default_rng(ss)
    t0 = time.time()
    world = draw_world(rng, sigma)
    items = draw_items(rng, cfg["n_subj"], cfg["per_subj"])
    ans = simulate_answers(rng, world, items)
    # subject-stratified halves
    half = np.zeros(len(items["b"]), dtype=bool)
    for s in range(cfg["n_subj"]):
        ids = np.where(items["subj"] == s)[0]
        half[rng.permutation(ids)[: len(ids) // 2]] = True
    sel, ev = np.where(half)[0], np.where(~half)[0]
    key_sel, key_ev = items["key"][sel], items["key"][ev]
    ans_sel, ans_ev = ans[:, sel], ans[:, ev]
    acc_sel = (ans_sel == key_sel[None, :]).mean(1)
    acc_ev = (ans_ev == key_ev[None, :]).mean(1)
    out = {"sigma": sigma, "rep": rep, "acc_min": float(acc_ev.min()), "acc_max": float(acc_ev.max())}

    # H1 on the full panel (evaluation half), N2 parametric bootstrap
    t_obs = h1_statistic(ans_ev, key_ev, acc_ev, rng, cfg["draws"])
    c_ev = (ans_ev == key_ev[None, :]).astype(float)
    th_hat, a_hat, b_hat, p_hat = fit_2pl(c_ev)
    wrong = pooled_wrong(ans_ev, key_ev)
    t_null = []
    for _ in range(cfg["k_null"]):
        a_null = simulate_n2(rng, p_hat, wrong, key_ev)
        acc_null = (a_null == key_ev[None, :]).mean(1)
        t_null.append(h1_statistic(a_null, key_ev, acc_null, rng, cfg["draws"]))
    t_null = np.array(t_null)
    out["T"] = t_obs
    out["p"] = float((1 + (t_null >= t_obs).sum()) / (1 + len(t_null)))
    out["T_null_mean"] = float(t_null.mean())
    # secondary (not a pass criterion, see Amendment A1): two-sided p-value around the null mean
    out["p_two"] = float((1 + (np.abs(t_null - t_null.mean()) >= abs(t_obs - t_null.mean())).sum()) / (1 + len(t_null)))
    items_ev = {k: v[ev] for k, v in items.items()}
    out["b_spearman"] = spearman(b_hat, items_ev["b"])

    # recovery: portfolio quantities, population values from fresh items
    if sigma in cfg["recovery_sigmas"]:
        port = greedy_portfolio(ans_sel, key_sel, acc_sel, list(range(N_MODELS)), 5)
        scen = scenario_sets(port)
        fresh = draw_items(rng, cfg["fresh_subj"], cfg["per_subj"])
        ans_fr = simulate_answers(rng, world, fresh)
        e_full_fr = per_item_errors(ans_fr, fresh["key"], acc_sel, port).mean()
        e_full_ev = per_item_errors(ans_ev, key_ev, acc_sel, port)
        rows = []
        for name, W in scen:
            rest = [i for i in port if i not in W]
            d_ev = per_item_errors(ans_ev, key_ev, acc_sel, rest) - e_full_ev
            lo, hi = cluster_boot_ci(d_ev, items_ev["subj"], rng, cfg["boot"])
            truth = per_item_errors(ans_fr, fresh["key"], acc_sel, rest).mean() - e_full_fr
            row = {"scenario": name, "est": float(d_ev.mean()), "lo": lo, "hi": hi, "truth": float(truth)}
            if sigma == 0.0:      # N2-predicted withdrawal loss
                pred = []
                for _ in range(cfg["k_pred"]):
                    a_null = simulate_n2(rng, p_hat, wrong, key_ev)
                    pred.append(per_item_errors(a_null, key_ev, acc_sel, rest).mean()
                                - per_item_errors(a_null, key_ev, acc_sel, port).mean())
                row["n2_pred"] = float(np.mean(pred))
            rows.append(row)
        out["dE"] = rows
        if len(scen) >= 2:
            (n1, W1), (n2_, W2) = scen[0], scen[1]
            r1 = [i for i in port if i not in W1]
            r2 = [i for i in port if i not in W2]
            r12 = [i for i in port if i not in W1 and i not in W2]
            if r12:
                ji_ev = (per_item_errors(ans_ev, key_ev, acc_sel, r12) - per_item_errors(ans_ev, key_ev, acc_sel, r1)
                         - per_item_errors(ans_ev, key_ev, acc_sel, r2) + e_full_ev)
                lo, hi = cluster_boot_ci(ji_ev, items_ev["subj"], rng, cfg["boot"])
                tr = (per_item_errors(ans_fr, fresh["key"], acc_sel, r12).mean()
                      - per_item_errors(ans_fr, fresh["key"], acc_sel, r1).mean()
                      - per_item_errors(ans_fr, fresh["key"], acc_sel, r2).mean() + e_full_fr)
                out["JI"] = {"pair": [n1, n2_], "est": float(ji_ev.mean()), "lo": lo, "hi": hi, "truth": float(tr)}
        if sigma == cfg["admission_sigma"]:
            open_pool = [i for i in range(N_MODELS) if not API[i]]
            port_o = greedy_portfolio(ans_sel, key_sel, acc_sel, open_pool, 5)
            n2pack = (th_hat, a_hat, b_hat, p_hat, wrong)
            adm = []
            for a_t in (0.5, 0.7, 0.85):
                for r in (0.0, 0.3, 0.6):
                    g_n2 = admission_gain(rng, ans_ev, key_ev, acc_sel, port_o, a_t, r, "n2", items_ev, n2pack,
                                          world, cfg["adm_draws"])
                    g_tr = admission_gain(rng, ans_ev, key_ev, acc_sel, port_o, a_t, r, "true", items_ev, n2pack,
                                          world, cfg["adm_draws"])
                    adm.append({"a": a_t, "r": r, "n2": g_n2, "true": g_tr})
            out["admission"] = adm
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ------------------------------------------------------------------ evaluation of criteria
def summarise(rows, cfg):
    lines = []
    by = {}
    for r in rows:
        by.setdefault(r["sigma"], []).append(r)
    null = by.get(0.0, [])
    size = np.mean([r["p"] <= 0.05 for r in null]) if null else float("nan")
    c1 = size <= 0.06
    lines.append(f"datasets: " + ", ".join(f"sigma={s}: {len(v)}" for s, v in sorted(by.items())))
    acc = [r["acc_min"] for r in rows], [r["acc_max"] for r in rows]
    lines.append(f"observed accuracy range across datasets: min {np.mean(acc[0]):.3f}, max {np.mean(acc[1]):.3f} (means)")
    lines.append(f"\nCriterion 1 (size): rejection rate under sigma=0: {size:.3f} (n={len(null)}); "
                 f"mean T under null {np.mean([r['T'] for r in null]):.3f}  -> {'PASS' if c1 else 'FAIL'}")
    power_rows, mde, mde_sigma = [], None, None
    for s in cfg["power_sigmas"]:
        v = by.get(s, [])
        if not v:
            continue
        pw = np.mean([r["p"] <= 0.05 for r in v])
        mt = np.mean([r["T"] for r in v])
        power_rows.append((s, pw, mt, len(v)))
        if mde is None and pw >= 0.8:
            mde, mde_sigma = mt, s
    lines.append("\nCriterion 2 (minimal detectable effect): power of H1 by sigma (one-sided, pre-registered)")
    for s, pw, mt, n in power_rows:
        pw2 = np.mean([r["p_two"] <= 0.05 for r in by[s]])
        lines.append(f"  sigma={s}: power {pw:.2f}, mean T {mt:.2f} (n={n}); secondary two-sided power {pw2:.2f}")
    if null:
        lines.append(f"  secondary: two-sided rejection rate under sigma=0: {np.mean([r['p_two'] <= 0.05 for r in null]):.3f}")
    c2 = mde is not None
    lines.append(f"  -> {'PASS' if c2 else 'FAIL'}: " + (f"MDE = mean T {mde:.2f} at sigma={mde_sigma}" if c2 else "no sigma reaches power 0.8"))

    rec = by.get(cfg["admission_sigma"], [])
    cover = [(x["lo"] <= x["truth"] <= x["hi"]) for r in rec for x in r.get("dE", [])]
    cover_ji = [(r["JI"]["lo"] <= r["JI"]["truth"] <= r["JI"]["hi"]) for r in rec if "JI" in r]
    bsp = np.mean([r["b_spearman"] for r in rec]) if rec else float("nan")
    pred_ok = []
    for r in null:
        for x in r.get("dE", []):
            if "n2_pred" in x:
                tol = max(0.005, (x["hi"] - x["lo"]) / 2)
                pred_ok.append(abs(x["n2_pred"] - x["truth"]) <= tol)
    adm = {}
    for r in rec:
        for g in r.get("admission", []):
            adm.setdefault((g["a"], g["r"]), []).append((g["n2"], g["true"]))
    lines.append("\nCriterion 3 (recovery)")
    c3a = len(cover) > 0 and np.mean(cover) >= 0.90
    c3b = len(cover_ji) > 0 and np.mean(cover_ji) >= 0.90
    c3c = bsp >= 0.85
    c3d = len(pred_ok) > 0 and np.mean(pred_ok) >= 0.90
    lines.append(f"  dE interval coverage: {np.mean(cover):.3f} over {len(cover)} scenario estimates -> {'pass' if c3a else 'fail'}")
    lines.append(f"  JI interval coverage: {np.mean(cover_ji):.3f} over {len(cover_ji)} -> {'pass' if c3b else 'fail'}")
    lines.append(f"  mean Spearman(fitted b, true b): {bsp:.3f} -> {'pass' if c3c else 'fail'}")
    lines.append(f"  N2-predicted dE within tolerance under sigma=0: {np.mean(pred_ok):.3f} over {len(pred_ok)} -> {'pass' if c3d else 'fail'}")
    c3e = True
    lines.append("  admission chart, mean over datasets (gain in accuracy, percentage points): a, r: N2 | true | mean |diff|")
    for (a_t, r), v in sorted(adm.items()):
        v = np.array(v)
        md = np.mean(np.abs(v[:, 0] - v[:, 1]))
        ok = md <= 0.005
        c3e &= ok
        lines.append(f"    a={a_t:.2f} r={r:.1f}: {100*v[:,0].mean():6.2f} | {100*v[:,1].mean():6.2f} | {100*md:5.2f} {'ok' if ok else 'FAIL'}")
    c3e = c3e and len(adm) > 0
    c3 = c3a and c3b and c3c and c3d and c3e
    lines.append(f"  -> {'PASS' if c3 else 'FAIL'}")
    lines.append(f"\nS0 overall: {'PASS' if (c1 and c2 and c3) else 'FAIL'}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--shard", default="0/1", help="k[,k2...]/n: run only tasks whose index mod n is in the list")
    ap.add_argument("--results", default=None, help="results file (default checks/cycle8_s0_results.jsonl)")
    ap.add_argument("--summary-only", action="store_true", help="only summarise existing results")
    args = ap.parse_args()
    if args.smoke:
        cfg = dict(n_subj=12, per_subj=60, fresh_subj=40, draws=10, k_null=10, k_pred=5, boot=100, adm_draws=3,
                   n_null=4, n_power=2, power_sigmas=[0.3, 0.5], recovery_sigmas=[0.0, 0.3], admission_sigma=0.3)
        suffix = "_smoke"
    else:
        cfg = dict(n_subj=57, per_subj=246, fresh_subj=244, draws=100, k_null=100, k_pred=20, boot=300, adm_draws=30,
                   n_null=200, n_power=100, power_sigmas=[0.1, 0.2, 0.3, 0.5], recovery_sigmas=[0.0, 0.3],
                   admission_sigma=0.3)
        suffix = ""
    here = os.path.dirname(os.path.abspath(__file__))
    res_path = args.results or os.path.join(here, f"cycle8_s0_results{suffix}.jsonl")
    out_path = os.path.join(here, f"cycle8_s0_synthetic_validation{suffix}.out")
    tasks = [{"sigma": 0.0, "rep": i, "cfg": cfg} for i in range(cfg["n_null"])]
    for s in cfg["power_sigmas"]:
        tasks += [{"sigma": s, "rep": i, "cfg": cfg} for i in range(cfg["n_power"])]
    done = {}
    if os.path.exists(res_path):                      # resume
        for line in open(res_path):
            r = json.loads(line)
            done[(r["sigma"], r["rep"])] = r
    ks, n_sh = args.shard.split("/")
    keep = {int(k) for k in ks.split(",")}
    todo = [t for i, t in enumerate(tasks) if i % int(n_sh) in keep and (t["sigma"], t["rep"]) not in done]
    if args.summary_only:
        todo = []
    print(f"{len(tasks)} datasets, {len(done)} already done, {len(todo)} to run, {args.workers} workers", flush=True)
    t0 = time.time()
    if not todo:
        pass
    with open(res_path, "a") as f, Pool(max(1, args.workers)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_dataset, todo), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            done[(r["sigma"], r["rep"])] = r
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {time.time() - t0:.0f} s elapsed", flush=True)
    missing = [t for t in tasks if (t["sigma"], t["rep"]) not in done]
    if missing:
        print(f"{len(missing)} datasets not yet available (other shards); no summary written", flush=True)
        return
    rows = [done[(t["sigma"], t["rep"])] for t in tasks]
    header = (f"Cycle 8 S0 synthetic validation{' (SMOKE TEST, not a result)' if args.smoke else ''}\n"
              f"config: {json.dumps(cfg)}\nmodels {N_MODELS}, developers {N_DEV}, "
              f"developers with >=2 models {sum(1 for s in STRUCTURE if s[1] >= 2)}, API-only models {API.sum()}\n")
    text = header + summarise(rows, cfg)
    open(out_path, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
