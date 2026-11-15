"""Cycle 8, POST HOC (referee item: null comparison rested on 20 panels): Delta E of the re-selected primary
portfolio on 100 N1 and 100 N2 panels, same code and seed scheme as S1b (panels 0-19 reproduce S1b).
    python3 checks/cycle8_null_panels_100.py --data data --n 100 --workers 16
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
import cycle8_s1b_supplementary as s1b               # noqa: E402
import cycle8_s1_analysis as s1                      # noqa: E402
import cycle8_s0_synthetic_validation as v1          # noqa: E402
import cycle8_s0b_synthetic_validation as v2         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    t0 = time.time()
    g = s1.load_real(args.data)
    rng = np.random.default_rng(s1b.SEED)
    fold = np.zeros(len(g["key"]), dtype=int)
    for s in np.unique(g["subj"]):
        ids = np.where(g["subj"] == s)[0]
        fold[rng.permutation(ids)[: len(ids) // 2]] = 1
    g["fold"], g["ones"] = fold, np.ones(len(g["key"]))
    g["fit1plg_p"] = v2.fit_1plg((g["A"] == g["key"][None, :]).astype(float))[3]
    g["pooled_wrong"] = v1.pooled_wrong(g["A"], g["key"])
    s1b.G.update(g)
    res = {"N1": {}, "N2": {}}
    with Pool(args.workers, initializer=s1b.init_worker, initargs=(g,)) as pool:
        for kind, j, r in pool.imap_unordered(s1b.null_task, [(k, j) for k in ("N1", "N2") for j in range(args.n)]):
            res[kind][j] = r
            print(f"  {kind} {j} done, {time.time() - t0:.0f} s", flush=True)
    obs = json.load(open(os.path.join(HERE, "cycle8_s1b_supplementary.json")))
    old = obs.get("null_dE", {})
    L = [f"Cycle 8 POST HOC: Delta E on {args.n} null panels each (MMLU, primary pipeline); pp; mean [5th, 95th]; "
         f"share of panels with value <= observed"]
    od = obs["observed_dE"]["api-only"]
    for kind in ("N1", "N2"):
        rs = [res[kind][j] for j in range(args.n)]
        api = [r["dE"]["api-only"] for r in rs if "api-only" in r["dE"]]
        full = [r["full"] for r in rs]
        same20 = None
        if kind in old and len(old[kind]) >= 20:
            m = min(20, args.n)
            a_old = sorted(r["full"] for r in old[kind][:20])
            a_new = sorted(res[kind][j]["full"] for j in range(m))
            a_old = a_old if m == 20 else a_old[:0]
            a_new = a_new if m == 20 else a_new[:0]
            same20 = bool(np.allclose(a_old, a_new))
        L.append(f"  {kind}: full error {100*np.mean(full):.2f} [{100*np.quantile(full,.05):.2f}, {100*np.quantile(full,.95):.2f}] "
                 f"(observed 11.06; share <= observed {np.mean(np.array(full) <= 0.1106):.2f}); API-only dE {100*np.mean(api):.2f} "
                 f"[{100*np.quantile(api,.05):.2f}, {100*np.quantile(api,.95):.2f}] (n {len(api)}; observed {100*od:.2f}; "
                 f"share <= observed {np.mean(np.array(api) <= od):.2f}); first 20 reproduce S1b: {same20}")
    L.append(f"seconds {round(time.time() - t0)}")
    json.dump({k: [res[k][j] for j in range(args.n)] for k in res}, open(os.path.join(HERE, "cycle8_null_panels_100.json"), "w"))
    open(os.path.join(HERE, "cycle8_null_panels_100.out"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
