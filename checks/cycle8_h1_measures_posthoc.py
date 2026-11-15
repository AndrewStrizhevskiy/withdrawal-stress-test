"""Cycle 8, POST-HOC: the H1 developer contrast under alternative agreement measures (HELM MMLU, 70 models).

Same matching as H1 (accuracy within 0.03 widening to 0.12, >= 5 pairs, same prompt classes), same developers.
Measures per pair:
  x     excess same-wrong agreement over N2 (the pre-registered H1 measure)
  raw   same-wrong agreement rate S_ij / N (no null)
  cond  P(same answer | both wrong) (Kim et al. 2025)
  capa  chance-adjusted agreement with discrete answers (Goel et al. 2025):
        c_obs = share of items with equal answers; c_exp = p_i p_j + (1-p_i)(1-p_j)/(m-1); kappa = (c_obs-c_exp)/(1-c_exp)
For each measure: the observed contrast T, and its distribution on panels simulated under N2 (item difficulty and
wrong-option popularity only, no developer dependence). A measure whose contrast is positive under N2 confounds shared
difficulty with shared developer.

    .venv/bin/python checks/cycle8_h1_measures_posthoc.py --data data --k-null 50
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cycle8_s0_synthetic_validation as v1          # noqa: E402
import cycle8_s0b_synthetic_validation as v2         # noqa: E402
import cycle8_s1_analysis as s1                      # noqa: E402

M_OPT = 4


def pair_measures(A, key, p, wrong):
    M, n = A.shape
    S = np.zeros((M, M))
    for o in range(M_OPT):
        ind = ((A == o) & (key[None, :] != o)).astype(float)
        S += ind @ ind.T
    E = ((1 - p) * (wrong ** 2).sum(1)[None, :]) @ (1 - p).T
    W = (A != key[None, :]).astype(float)
    BW = W @ W.T
    EQ = np.zeros((M, M))
    for o in range(-1, M_OPT):
        ind = (A == o).astype(float)
        EQ += ind @ ind.T
    acc = 1 - W.mean(1)
    cexp = np.outer(acc, acc) + np.outer(1 - acc, 1 - acc) / (M_OPT - 1)
    cobs = EQ / n
    return {"x": (S - E) / n, "raw": S / n, "cond": S / np.maximum(BW, 1), "capa": (cobs - cexp) / (1 - cexp)}, acc


def contrast(vals, acc, DEVS, PCLS, devs_multi):
    M = len(acc)
    iu = np.triu_indices(M, 1)
    same = DEVS[iu[0]] == DEVS[iu[1]]
    ai, aj = acc[iu[0]], acc[iu[1]]
    ci, cj = PCLS[iu[0]], PCLS[iu[1]]
    out = {}
    for name, X in vals.items():
        x = X[iu]
        per_dev = []
        for d in devs_multi:
            ex = []
            for q in np.where(same & (DEVS[iu[0]] == d))[0]:
                cls_ok = ((ci == ci[q]) & (cj == cj[q])) | ((ci == cj[q]) & (cj == ci[q]))
                t = 0.03
                while t <= 0.12 + 1e-9:
                    m = (~same) & cls_ok & (((np.abs(ai - ai[q]) <= t) & (np.abs(aj - aj[q]) <= t)) |
                                            ((np.abs(ai - aj[q]) <= t) & (np.abs(aj - ai[q]) <= t)))
                    if m.sum() >= 5:
                        ex.append(x[q] - x[m].mean())
                        break
                    t *= 2
            if ex:
                per_dev.append(np.mean(ex))
        unmatched = float(np.mean([x[same & (DEVS[iu[0]] == d)].mean() for d in devs_multi]) - x[~same].mean())
        out[name] = {"matched": float(np.mean(per_dev)), "unmatched": unmatched}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(HERE), "data"))
    ap.add_argument("--k-null", type=int, default=50)
    args = ap.parse_args()
    t0 = time.time()
    g = s1.load_real(args.data)
    A, key, DEVS, PCLS = g["A"], g["key"], g["dev"], g["pclass"]
    devs_multi = [d for d in range(len(g["dev_names"])) if (DEVS == d).sum() >= 2]
    fit = v2.fit_1plg((A == key[None, :]).astype(float))
    wrong = v1.pooled_wrong(A, key)
    vals, acc = pair_measures(A, key, fit[3], wrong)
    obs = contrast(vals, acc, DEVS, PCLS, devs_multi)
    rng = np.random.default_rng([s1.SEED, 99])
    nulls = {k: {"matched": [], "unmatched": []} for k in obs}
    for b in range(args.k_null):
        a0 = v1.simulate_n2(rng, fit[3], wrong, key)
        f0 = v2.fit_1plg((a0 == key[None, :]).astype(float), init=fit[:3], iters=8)
        v0, acc0 = pair_measures(a0, key, f0[3], v1.pooled_wrong(a0, key))
        r = contrast(v0, acc0, DEVS, PCLS, devs_multi)
        for k in r:
            for mm in ("matched", "unmatched"):
                nulls[k][mm].append(r[k][mm])
        print(f"  null {b + 1}/{args.k_null}, {time.time() - t0:.0f} s", flush=True)
    L = ["Cycle 8 POST-HOC: H1 developer contrast under alternative agreement measures (HELM MMLU, 70 models)",
         f"N2 null panels: {args.k_null}; values: observed contrast; N2 mean [5th, 95th percentile]; z = (obs - null mean) / null sd"]
    res = {}
    for k in ("x", "raw", "cond", "capa"):
        for mm in ("matched", "unmatched"):
            nv = np.array(nulls[k][mm])
            o = obs[k][mm]
            z = (o - nv.mean()) / nv.std() if nv.std() > 0 else float("nan")
            res[f"{k}/{mm}"] = {"obs": o, "null_mean": float(nv.mean()), "null_q": [float(np.quantile(nv, 0.05)), float(np.quantile(nv, 0.95))], "z": float(z)}
            L.append(f"  {k:5s} {mm:9s}: obs {o:+.5f}; N2 {nv.mean():+.5f} [{np.quantile(nv, 0.05):+.5f}, {np.quantile(nv, 0.95):+.5f}]; z {z:.1f}")
    L.append(f"seconds {round(time.time() - t0)}")
    text = "\n".join(L)
    json.dump(res, open(os.path.join(HERE, "cycle8_h1_measures_posthoc.json"), "w"), indent=1)
    open(os.path.join(HERE, "cycle8_h1_measures_posthoc.out"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
