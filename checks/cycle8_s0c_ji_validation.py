"""Cycle 8, Amendment A3: validation of joint-importance (JI) intervals (protocol Section 14).

Reuses the generator and helpers of checks/cycle8_s0_synthetic_validation.py. No real data.

    .venv/bin/python checks/cycle8_s0c_ji_validation.py --smoke
    python3 checks/cycle8_s0c_ji_validation.py --workers 16          # on the server
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

MASTER_SEED = 20261115
SIGMA = 0.3


def ji_intervals(values, subj, rng, B):
    """Return (old percentile interval, A3 centred-and-inflated interval) for the mean of per-item values."""
    n_s = subj.max() + 1
    sums = np.bincount(subj, weights=values, minlength=n_s)
    cnts = np.bincount(subj, minlength=n_s).astype(float)
    idx = rng.integers(0, n_s, (B, n_s))
    stats = sums[idx].sum(1) / cnts[idx].sum(1)
    est = values.mean()
    old = (float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975)))
    adj = est + np.sqrt(n_s / (n_s - 1)) * (stats - stats.mean())
    new = (float(np.quantile(adj, 0.025)), float(np.quantile(adj, 0.975)))
    return old, new


def run_dataset(task):
    rep, cfg = task["rep"], task["cfg"]
    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, rep]))
    t0 = time.time()
    world = v1.draw_world(rng, SIGMA)
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
    port = v1.greedy_portfolio(ans_sel, key_sel, acc_sel, list(range(v1.N_MODELS)), 5)
    scen = v1.scenario_sets(port)
    out = {"rep": rep, "n_scen": len(scen)}
    if len(scen) < 2:
        out["skip"] = "fewer than two scenarios"
        return out
    (n1, W1), (n2, W2) = scen[0], scen[1]
    r1 = [i for i in port if i not in W1]
    r2 = [i for i in port if i not in W2]
    r12 = [i for i in port if i not in W1 and i not in W2]
    if not r12:
        out["skip"] = "both scenarios empty the portfolio"
        return out
    pe = v1.per_item_errors
    e0 = pe(ans_ev, key_ev, acc_sel, port)
    ji = pe(ans_ev, key_ev, acc_sel, r12) - pe(ans_ev, key_ev, acc_sel, r1) - pe(ans_ev, key_ev, acc_sel, r2) + e0
    old, new = ji_intervals(ji, items["subj"][ev], rng, cfg["boot"])
    fresh = v1.draw_items(rng, cfg["fresh_subj"], cfg["per_subj"])
    af, kf = v1.simulate_answers(rng, world, fresh), fresh["key"]
    truth = pe(af, kf, acc_sel, r12).mean() - pe(af, kf, acc_sel, r1).mean() - pe(af, kf, acc_sel, r2).mean() + pe(af, kf, acc_sel, port).mean()
    out.update({"pair": [n1, n2], "est": float(ji.mean()), "truth": float(truth), "old": old, "new": new,
                "seconds": round(time.time() - t0, 1)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--results", default=None)
    args = ap.parse_args()
    cfg = (dict(n_subj=12, per_subj=60, fresh_subj=40, boot=100, n=6) if args.smoke
           else dict(n_subj=57, per_subj=246, fresh_subj=244, boot=300, n=400))
    suffix = "_smoke" if args.smoke else ""
    here = os.path.dirname(os.path.abspath(__file__))
    res_path = args.results or os.path.join(here, f"cycle8_s0c_results{suffix}.jsonl")
    out_path = os.path.join(here, f"cycle8_s0c_ji_validation{suffix}.out")
    done = {}
    if os.path.exists(res_path):
        for line in open(res_path):
            r = json.loads(line)
            done[r["rep"]] = r
    todo = [{"rep": i, "cfg": cfg} for i in range(cfg["n"]) if i not in done]
    print(f"{cfg['n']} datasets, {len(done)} done, {len(todo)} to run", flush=True)
    if todo:
        with open(res_path, "a") as f, Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(run_dataset, todo), 1):
                f.write(json.dumps(r) + "\n")
                f.flush()
                done[r["rep"]] = r
                if i % 50 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} done", flush=True)
    rows = [done[i] for i in range(cfg["n"])]
    used = [r for r in rows if "skip" not in r]
    cov_new = np.mean([r["new"][0] <= r["truth"] <= r["new"][1] for r in used])
    cov_old = np.mean([r["old"][0] <= r["truth"] <= r["old"][1] for r in used])
    below = sum(r["truth"] < r["new"][0] for r in used)
    above = sum(r["truth"] > r["new"][1] for r in used)
    se = np.sqrt(0.9 * 0.1 / len(used))
    text = (f"Cycle 8 Amendment A3: JI interval validation{' (SMOKE TEST, not a result)' if args.smoke else ''}\n"
            f"config: {json.dumps(cfg)}, sigma={SIGMA}, master seed {MASTER_SEED}\n"
            f"datasets: {len(rows)}; used {len(used)}; skipped {len(rows) - len(used)}\n"
            f"A3 interval coverage: {cov_new:.3f} (binomial SE at 0.90: {se:.3f}); misses below {below}, above {above}\n"
            f"old percentile interval coverage (comparison only): {cov_old:.3f}\n"
            f"mean width A3 {np.mean([r['new'][1] - r['new'][0] for r in used]):.4f}, old "
            f"{np.mean([r['old'][1] - r['old'][0] for r in used]):.4f}\n"
            f"A3 overall: {'PASS' if cov_new >= 0.90 else 'FAIL'}")
    open(out_path, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
