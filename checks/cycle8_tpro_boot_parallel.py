"""Cycle 8, protocol Section 60: the subject-bootstrap interval of cycle8_tpro_analysis.py --items all, computed in parallel.

Draws exactly the same resamples, in the same order and from the same seed, as the serial script (fold permutations
first, then one subject-count vector per resample), so the interval is identical to the serial result.

    python3 checks/cycle8_tpro_boot_parallel.py --data data --local data/ru --tpro data/ru_tpro --workers 18
"""
import argparse
import gzip
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cycle8_ru_analysis as ru                      # noqa: E402
import cycle8_s1_analysis as s1                      # noqa: E402
import cycle8_tpro_analysis as tp                    # noqa: E402

W = {}


def one(cnt):
    w = cnt[W["subj"]]
    A, key, pool, fold, iT = W["A"], W["key"], W["pool"], W["fold"], W["iT"]
    return (tp.crossfit(A, key, pool, fold, w), tp.crossfit(A, key, pool, fold, w, extra=iT),
            tp.crossfit(A, key, pool + [iT], fold, w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--local", required=True)
    ap.add_argument("--tpro", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    g = s1.load_real(args.data)
    items = set()
    with gzip.open(os.path.join(args.data, "helm_mmlu_parsed.tsv.gz"), "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            fs = line.split("\t")
            items.add((fs[2], fs[3]))
    order_all = {f"{s}|{i}": j for j, (s, i) in enumerate(sorted(items))}
    sub = [json.loads(l) for l in gzip.open(os.path.join(args.data, "helm_mmlu_prompts.jsonl.gz"), "rt", encoding="utf-8")]
    idx = np.array([order_all[f"{r['subject']}|{r['instance_id']}"] for r in sub])
    key, subj, H = g["key"][idx], g["subj"][idx], g["A"][:, idx]
    loc_labels = list(ru.LOCAL)
    order = {f"{r['subject']}|{r['instance_id']}": j for j, r in enumerate(sub)}
    Lc = np.vstack([tp.read_local(os.path.join(args.local, f"{l}.jsonl"), order, len(sub)) for l in loc_labels])
    t = tp.read_local(os.path.join(args.tpro, "t-pro-it-2.1.jsonl"), order, len(sub))
    A = np.vstack([H, Lc, t[None, :]]).astype(np.int8)
    nH, nL = H.shape[0], Lc.shape[0]
    rng = np.random.default_rng(tp.SEED)                       # same sequence as the serial script
    fold = np.zeros(len(key), dtype=int)
    for s in np.unique(subj):
        ids = np.where(subj == s)[0]
        fold[rng.permutation(ids)[: len(ids) // 2]] = 1
    nS = int(subj.max()) + 1
    w1 = np.ones(len(key))
    open_all = [i for i in range(nH) if not g["api"][i]]
    e0 = tp.crossfit(A, key, open_all, fold, w1)
    e1 = tp.crossfit(A, key, open_all, fold, w1, extra=nH + nL)
    e2 = tp.crossfit(A, key, open_all + [nH + nL], fold, w1)
    draws = [np.bincount(rng.integers(0, nS, nS), minlength=nS).astype(float) for _ in range(args.boot // 5)]
    W.update(A=A, key=key, pool=open_all, fold=fold, iT=nH + nL, subj=subj)
    with Pool(args.workers) as p:
        bs = np.array(p.map(one, draws, chunksize=1))
    d1, d2 = bs[:, 1] - bs[:, 0], bs[:, 2] - bs[:, 0]
    q = lambda d, a: 100 * float(np.quantile(d, a))
    txt = (f"T-pro on all {len(key):,} items, open-weight portfolio of five (cross-fitted): {100*e0:.2f}%; with T-pro added as sixth "
           f"{100*e1:.2f}% (change {100*(e1-e0):+.2f} [{q(d1,.025):+.2f}, {q(d1,.975):+.2f}]); re-selected with T-pro in the pool "
           f"{100*e2:.2f}% (change {100*(e2-e0):+.2f} [{q(d2,.025):+.2f}, {q(d2,.975):+.2f}]); {len(bs)} subject resamples (parallel, "
           f"same draws as the serial script)")
    open(os.path.join(HERE, "cycle8_tpro_full_boot.out"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
