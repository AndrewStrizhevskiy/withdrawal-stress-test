"""Cycle 8, protocol Section 61 (C1/C2): calibrated Delta E intervals from synthetic panels. POST HOC.

    python3 checks/cycle8_ci_calibration.py checks/cycle8_s0b_results.jsonl [label] [sigma]
    (sigma: restrict to the datasets used for the original coverage check, e.g. the admission sigma of S0b)
"""
import json
import sys

import numpy as np

SEED = 20261161


def load(path, sigma=None):
    ds = []
    for line in open(path):
        r = json.loads(line)
        if sigma is not None and abs(r.get("sigma", -1) - sigma) > 1e-9:
            continue
        if r.get("kind", "cov") != "cov":
            continue
        iv = [(d["est"], d["lo"], d["hi"], d["truth"]) for d in r.get("dE", []) if d.get("lo") is not None and d.get("truth") is not None]
        if iv:
            ds.append(np.array(iv, dtype=float))
    return ds


def coverage(arr, k):
    est, lo, hi, t = arr.T
    return np.mean((t >= est - k * (est - lo)) & (t <= est + k * (hi - est)))


def kmin(arr, target=0.95):
    lo_k, hi_k = 0.5, 4.0
    for _ in range(60):
        m = 0.5 * (lo_k + hi_k)
        if coverage(arr, m) >= target:
            hi_k = m
        else:
            lo_k = m
    return hi_k


def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    sigma = float(sys.argv[3]) if len(sys.argv) > 3 else None
    ds = load(path, sigma)
    allv = np.vstack(ds)
    rng = np.random.default_rng(SEED)
    ks, covs = [], []
    for _ in range(200):
        idx = rng.permutation(len(ds))
        a = np.vstack([ds[i] for i in idx[: len(ds) // 2]])
        b = np.vstack([ds[i] for i in idx[len(ds) // 2:]])
        k = kmin(a)
        ks.append(k)
        covs.append(coverage(b, k))
    ks, covs = np.array(ks), np.array(covs)
    kmed = float(np.median(ks))
    out = {"label": label, "sigma": sigma, "datasets": len(ds), "intervals": int(len(allv)), "coverage_k1": float(coverage(allv, 1.0)),
           "k_median": kmed, "k_p05": float(np.quantile(ks, .05)), "k_p95": float(np.quantile(ks, .95)),
           "oos_coverage_median": float(np.median(covs)), "oos_coverage_p05": float(np.quantile(covs, .05)),
           "coverage_all_at_kmed": float(coverage(allv, kmed))}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
