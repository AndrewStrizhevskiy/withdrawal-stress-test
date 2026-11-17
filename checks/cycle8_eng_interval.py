"""Cycle 8, protocol Section 62: interval for the API-only withdrawal effect on the engineering-science subset. POST HOC.

Same quantity and the same subject-resample draws as S1 (cycle8_s1_analysis.boot_task: seed [SEED, 1000 + b]), with the
primary portfolio (accuracy-greedy, n = 5, plurality, cross-fitted) re-selected in every resample.

    python3 checks/cycle8_eng_interval.py --data data --boot 200 --workers 18
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cycle8_s1_analysis as s1                      # noqa: E402

K_MMLU = 1.116   # Section 61


def eng_effect(w):
    G = s1.G
    p = s1.Pipe(w)
    prim = (p.greedy(0, 5), p.greedy(1, 5))
    groups = s1.scenario_groups()
    full = p.mean(p.err("plurality", prim), G["eng"])
    ws = s1.withdraw(prim, groups["api-only"])
    wd = p.mean(p.err("plurality", ws), G["eng"]) if (ws[0] and ws[1]) else 0.75
    return full, wd


def task(b):
    rng = np.random.default_rng([s1.SEED, 1000 + b])
    n_s = s1.G["subj"].max() + 1
    counts = np.bincount(rng.integers(0, n_s, n_s), minlength=n_s).astype(float)
    return b, eng_effect(counts[s1.G["subj"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    g = s1.load_real(args.data)
    rng = np.random.default_rng(s1.SEED)
    fold = np.zeros(len(g["key"]), dtype=int)
    for s in np.unique(g["subj"]):
        ids = np.where(g["subj"] == s)[0]
        fold[rng.permutation(ids)[: len(ids) // 2]] = 1
    g["fold"], g["ones"] = fold, np.ones(len(g["key"]))
    s1.G.update(g)
    full, wd = eng_effect(g["ones"])
    with Pool(args.workers, initializer=s1.init_worker, initargs=(g,)) as pool:
        res = dict(pool.map(task, range(args.boot)))
    d = np.array([res[b][1] - res[b][0] for b in range(args.boot)])
    est = wd - full
    lo, hi = float(np.quantile(d, .025)), float(np.quantile(d, .975))
    clo, chi = est - K_MMLU * (est - lo), est + K_MMLU * (hi - est)
    n_eng = int(g["eng"].sum())
    txt = (f"Engineering-science subset ({n_eng} items), primary portfolio, API-only withdrawal (protocol Section 62; POST HOC)\n"
           f"full {100*full:.2f}%, withdrawn {100*wd:.2f}%, dE {100*est:+.2f} points\n"
           f"percentile 95% interval [{100*lo:+.2f}, {100*hi:+.2f}] over {args.boot} S1 subject resamples\n"
           f"widened by k = {K_MMLU} (approximate for this subset): [{100*clo:+.2f}, {100*chi:+.2f}]\n"
           f"share of resamples with dE <= 0: {np.mean(d <= 0):.3f}")
    open(os.path.join(HERE, "cycle8_eng_interval.out"), "w").write(txt + "\n")
    json.dump({"full": full, "withdrawn": wd, "dE": est, "pct": [lo, hi], "calibrated": [clo, chi], "boot": d.tolist()},
              open(os.path.join(HERE, "cycle8_eng_interval.json"), "w"))
    print(txt)


if __name__ == "__main__":
    main()
