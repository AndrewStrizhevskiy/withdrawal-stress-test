"""Cycle 8, H1 under model-specific letter bias (null N3; protocol Section 58). POST HOC.

    .venv/bin/python checks/cycle8_h1_letterbias.py --data data [--k-null 200]
"""
import argparse
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


def letter_bias(A, key, p, w, iters=20):
    """b[i,k] such that w_i[n,k] ∝ w[n,k] b[i,k] (non-key options) matches model i's wrong-letter counts."""
    M = A.shape[0]
    b = np.ones((M, M_OPT))
    wrong = (A >= 0) & (A != key[None, :])
    O = np.stack([((A == k) & wrong).sum(1) for k in range(M_OPT)], 1).astype(float)
    for _ in range(iters):
        W = model_wrong(w, b)
        Ehat = np.einsum("mn,mnk->mk", wrong.astype(float), W)
        b = b * (O + 0.5) / (Ehat + 0.5)
        b = b / b.mean(1, keepdims=True)
    return b


def model_wrong(w, b):
    W = w[None, :, :] * b[:, None, :]          # w is 0 at the key, so the key stays excluded
    return W / W.sum(2, keepdims=True)


def make_stat(DEVS, PCLS, key, M):
    iu = np.triu_indices(M, 1)
    same = DEVS[iu[0]] == DEVS[iu[1]]
    devs_multi = [d for d in np.unique(DEVS) if (DEVS == d).sum() >= 2]

    def stat(ans, p, W, drop_dev=None):
        n = ans.shape[1]
        S = np.zeros((M, M))
        E = np.zeros((M, M))
        for o in range(M_OPT):
            ind = ((ans == o) & (key[None, :] != o)).astype(float)
            S += ind @ ind.T
            q = (1 - p) * W[:, :, o]
            E += q @ q.T
        x = ((S - E) / n)[iu]
        acc = (ans == key[None, :]).mean(1)
        ai, aj = acc[iu[0]], acc[iu[1]]
        ci, cj = PCLS[iu[0]], PCLS[iu[1]]
        keep = np.ones(len(x), dtype=bool)
        if drop_dev is not None:
            keep = (DEVS[iu[0]] != drop_dev) & (DEVS[iu[1]] != drop_dev)
        btw = (~same) & keep
        per_dev = []
        for d in devs_multi:
            if d == drop_dev:
                continue
            ex = []
            for q_ in np.where(same & (DEVS[iu[0]] == d))[0]:
                cls_ok = ((ci == ci[q_]) & (cj == cj[q_])) | ((ci == cj[q_]) & (cj == ci[q_]))
                t = 0.03
                while t <= 0.12 + 1e-9:
                    m = btw & cls_ok & (((np.abs(ai - ai[q_]) <= t) & (np.abs(aj - aj[q_]) <= t)) |
                                        ((np.abs(ai - aj[q_]) <= t) & (np.abs(aj - ai[q_]) <= t)))
                    if m.sum() >= 5:
                        ex.append(x[q_] - x[m].mean())
                        break
                    t *= 2
            if ex:
                per_dev.append(np.mean(ex))
        return float(np.mean(per_dev))
    return stat, devs_multi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--k-null", type=int, default=200)
    args = ap.parse_args()
    t0 = time.time()
    g = s1.load_real(args.data)
    A, key = g["A"], g["key"]
    M = A.shape[0]
    stat, devs_multi = make_stat(g["dev"], g["pclass"], key, M)
    c = (A == key[None, :]).astype(float)
    fit = v2.fit_1plg(c)
    p = fit[3]
    w = v1.pooled_wrong(A, key)
    W2 = np.broadcast_to(w, (M,) + w.shape)
    t_n2 = stat(A, p, W2)
    b = letter_bias(A, key, p, w)
    W3 = model_wrong(w, b)
    t_n3 = stat(A, p, W3)
    rng = np.random.default_rng([20261114, 9])
    null = []
    for _ in range(args.k_null):
        correct = rng.random(p.shape) < p
        u = rng.random(p.shape)[:, :, None]
        wa = (u > np.cumsum(W3, 2)).sum(2).clip(0, M_OPT - 1)
        a0 = np.where(correct, key[None, :], wa).astype(np.int8)
        f0 = v2.fit_1plg((a0 == key[None, :]).astype(float), init=fit[:3], iters=8)
        w0 = v1.pooled_wrong(a0, key)
        b0 = letter_bias(a0, key, f0[3], w0)
        null.append(stat(a0, f0[3], model_wrong(w0, b0)))
    null = np.array(null)
    pval = float((1 + (null >= t_n3).sum()) / (1 + len(null)))
    jk = np.array([stat(A, p, W3, drop_dev=d) for d in devs_multi])
    k = len(jk)
    se = float(np.sqrt((k - 1) / k * np.sum((jk - jk.mean()) ** 2)))
    spread = np.abs(b - 1).max(1)
    L = ["Cycle 8 H1 under model-specific letter bias (null N3; protocol Section 58). POST HOC",
         f"MMLU, {M} models, {A.shape[1]} items; {len(null)} N3 panels",
         f"Letter-bias vectors b_i: median max |b_ik - 1| {np.median(spread):.3f}, max {spread.max():.3f}",
         f"T under N2 (check, Section 19 value 0.00462): {t_n2:.5f}",
         f"T under N3: {t_n3:.5f}; N3 null mean {null.mean():.5f}, sd {null.std():.5f}; p {pval:.3f}; "
         f"jackknife SE {se:.5f}, 90% interval [{t_n3 - 1.645*se:.5f}, {t_n3 + 1.645*se:.5f}]; MDE 0.0028",
         f"verdict: {'H1 holds under N3' if (pval < 0.05 and t_n3 > 0.0028) else 'H1 weakened under N3'}",
         f"seconds {round(time.time() - t0)}"]
    open(os.path.join(HERE, "cycle8_h1_letterbias.out"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
