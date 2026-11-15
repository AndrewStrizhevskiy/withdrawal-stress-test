"""Cycle 8, POST HOC (protocol Section 42): leave-one-developer-out jackknife of the sovereignty cost C, Delta E (API-only)
and the H3 gain, on MMLU and MMLU-Pro (point estimates; whole pipeline re-selected on each reduced panel).
    python3 checks/cycle8_dev_jackknife.py --data data --workers 16
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
import cycle8_s1_analysis as s1                      # noqa: E402
import cycle8_s1b_supplementary as s1b               # noqa: E402

BENCH = {}


def setup(bench, data):
    if bench == "MMLU":
        g = s1.load_real(data)
        rng = np.random.default_rng(s1.SEED)
        fold = np.zeros(len(g["key"]), dtype=int)
        for s in np.unique(g["subj"]):
            ids = np.where(g["subj"] == s)[0]
            fold[rng.permutation(ids)[: len(ids) // 2]] = 1
        g["fold"], g["ones"] = fold, np.ones(len(g["key"]))
    else:
        import cycle8_pro_analysis as pro
        pro.set_k(0.00358)
        g = pro.load(data, "primary")
    return g


def task(job):
    bench, d, data = job
    if bench not in BENCH:
        if bench == "MMLU-Pro":
            import cycle8_pro_analysis as pro
            pro.set_k(0.00358)
        else:
            import cycle8_s0_synthetic_validation as v1
            import cycle8_s0b_synthetic_validation as v2
            s1.M_OPT, v1.M_OPT, v2.GUESS = 4, 4, 0.25
        BENCH[bench] = setup(bench, data)
    base = BENCH[bench]
    sub = [i for i in range(len(base["models"])) if base["dev"][i] != d] if d >= 0 else list(range(len(base["models"])))
    s1.G.clear()
    s1.G.update(s1b.restrict(base, sub))
    r = s1.analysis(s1.G["ones"])
    return bench, d, {"C": r["S/open5"] - r["H3/cap/full"], "dE_api": r.get("D/prim/dE/api-only"),
                      "h3_gain": r["H3/cap/heldout_worst"] - r["H3/robust/heldout_worst"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    t0 = time.time()
    jobs = []
    names = {}
    for bench in ("MMLU", "MMLU-Pro"):
        if bench == "MMLU-Pro":
            import cycle8_pro_analysis as pro
            pro.set_k(0.00358)
        g = setup(bench, args.data)
        if bench == "MMLU-Pro":
            s1.M_OPT = 4
            import cycle8_s0_synthetic_validation as v1
            import cycle8_s0b_synthetic_validation as v2
            v1.M_OPT = 4
            v2.GUESS = 0.25
        names[bench] = g["dev_names"]
        jobs += [(bench, d, args.data) for d in [-1] + sorted(set(g["dev"].tolist()))]
    with Pool(args.workers, maxtasksperchild=1) as pool:
        res = pool.map(task, jobs)
    json.dump(res, open(os.path.join(HERE, "cycle8_dev_jackknife_raw.json"), "w"), indent=1)   # saved before reporting
    out = {}
    L = ["Cycle 8 POST HOC (Section 42): leave-one-developer-out jackknife (points; full pipeline re-selected)"]
    for bench in ("MMLU", "MMLU-Pro"):
        rows = {d: r for b, d, r in res if b == bench}
        full = rows.pop(-1)
        out[bench] = {"full": full, "loo": {names[bench][d]: r for d, r in rows.items()}}
        L.append(f"\n{bench}: full C {100*full['C']:.2f}, dE(API-only) {100*full['dE_api']:.2f}" if full["dE_api"] is not None
                 else f"\n{bench}: full C {100*full['C']:.2f}")
        for q in ("C", "dE_api", "h3_gain"):
            v = np.array([r[q] for r in rows.values() if r[q] is not None], dtype=float)
            if len(v) == 0 or full[q] is None:
                continue
            k = len(v)
            se = np.sqrt((k - 1) / k * ((v - v.mean()) ** 2).sum())
            worst = min(rows, key=lambda d: rows[d][q] if rows[d][q] is not None else 1e9)
            best = max(rows, key=lambda d: rows[d][q] if rows[d][q] is not None else -1e9)
            L.append(f"  {q}: leave-one-out range [{100*v.min():.2f} (without {names[bench][worst]}), {100*v.max():.2f} "
                     f"(without {names[bench][best]})]; jackknife SE {100*se:.2f}; sign changes {int((np.sign(v) != np.sign(full[q])).sum())} of {k}")
    L.append(f"seconds {round(time.time() - t0)}")
    json.dump(out, open(os.path.join(HERE, "cycle8_dev_jackknife.json"), "w"), indent=1)
    open(os.path.join(HERE, "cycle8_dev_jackknife.out"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
