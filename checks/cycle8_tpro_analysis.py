"""Cycle 8, large domestic model T-pro-it-2.1 (32B) on a 2,850-item HELM MMLU subset (protocol Section 52). POST HOC.

Accuracy vs HELM open-weight models of 27-72B and vs the 8-12B Russian local runs (same items); whether adding T-pro
to a cross-fitted open-weight plurality portfolio of five lowers its error (subject bootstrap, 1,000 resamples);
excess same-wrong agreement x of T-pro with T-lite and Qwen3-8B (same Qwen3 family) vs T-pro with unrelated local models.

    .venv/bin/python checks/cycle8_tpro_analysis.py --data data --local data/ru --tpro <dir with t-pro-it-2.1.jsonl>
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cycle8_ru_analysis as ru                      # noqa: E402
import cycle8_s0_synthetic_validation as v1          # noqa: E402
import cycle8_s0b_synthetic_validation as v2         # noqa: E402
import cycle8_s1_analysis as s1                      # noqa: E402

SEED = 20261119
RUS = ["yandexgpt5-lite-8b", "t-lite-it-2.1", "gigachat3.1-10b", "vikhr-nemo-12b", "saiga-nemo-12b", "saiga-yandexgpt-8b"]
QWEN3_FAMILY = ["t-lite-it-2.1", "qwen3-8b"]


def read_local(path, order, n):
    a = np.full(n, -2, dtype=np.int8)
    op = gzip.open(path + ".gz", "rt") if not os.path.exists(path) else open(path)
    for line in op:
        r = json.loads(line)
        if r["k"] in order and "ans" in r:
            a[order[r["k"]]] = r["ans"]
    return a


def plurality(A, key, members, fit, ev, w):
    acc = np.array([((A[i, fit] == key[fit]) * w[fit]).sum() / max(w[fit].sum(), 1e-9) for i in range(A.shape[0])])
    votes = np.zeros((int(ev.sum()), 4))
    for i in members:
        a = A[i, ev]
        ok = a >= 0
        votes[np.where(ok)[0], a[ok]] += 1.0 + 1e-6 * acc[i]
    pred = np.where(votes.max(1) > 0, votes.argmax(1), -1)
    return float(((pred != key[ev]) * w[ev]).sum() / max(w[ev].sum(), 1e-9))


def greedy(A, key, pool, n, fit, w):
    ch = []
    for _ in range(n):
        ch.append(min((j for j in pool if j not in ch), key=lambda j: plurality(A, key, ch + [j], fit, fit, w)))
    return ch


def crossfit(A, key, pool, fold, w, extra=None):
    e = tot = 0.0
    for f in (0, 1):
        fit, ev = fold == f, fold != f
        mem = greedy(A, key, pool, 5, fit, w)
        if extra is not None:
            mem = mem + [extra]
        e += plurality(A, key, mem, fit, ev, w) * w[ev].sum()
        tot += w[ev].sum()
    return e / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--local", required=True)
    ap.add_argument("--tpro", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--items", choices=["subset", "all"], default="subset", help="subset50 (Section 52) or all items (Section 56)")
    args = ap.parse_args()
    g = s1.load_real(args.data)
    items = set()
    with gzip.open(os.path.join(args.data, "helm_mmlu_parsed.tsv.gz"), "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            fs = line.split("\t")
            items.add((fs[2], fs[3]))
    order_all = {f"{s}|{i}": j for j, (s, i) in enumerate(sorted(items))}
    pf = "helm_mmlu_prompts_subset50.jsonl.gz" if args.items == "subset" else "helm_mmlu_prompts.jsonl.gz"
    sub = [json.loads(l) for l in gzip.open(os.path.join(args.data, pf), "rt", encoding="utf-8")]
    idx = np.array([order_all[f"{r['subject']}|{r['instance_id']}"] for r in sub])
    key = g["key"][idx]
    subj = g["subj"][idx]
    H = g["A"][:, idx]
    loc_labels = list(ru.LOCAL)
    order = {f"{r['subject']}|{r['instance_id']}": j for j, r in enumerate(sub)}
    Lc = np.vstack([read_local(os.path.join(args.local, f"{l}.jsonl"), order, len(sub)) for l in loc_labels])
    tp = read_local(os.path.join(args.tpro, "t-pro-it-2.1.jsonl"), order, len(sub))
    assert (tp > -2).all() and (Lc > -2).all(), "incomplete runs"
    A = np.vstack([H, Lc, tp[None, :]]).astype(np.int8)
    nH, nL = H.shape[0], Lc.shape[0]
    iT = nH + nL
    acc = (A == key[None, :]).mean(1)
    open_mid = [i for i in range(nH) if not g["api"][i] and np.isfinite(g["params"][i]) and 27 <= g["params"][i] <= 72]
    open_all = [i for i in range(nH) if not g["api"][i]]
    rng = np.random.default_rng(SEED)
    fold = np.zeros(len(key), dtype=int)
    for s in np.unique(subj):
        ids = np.where(subj == s)[0]
        fold[rng.permutation(ids)[: len(ids) // 2]] = 1
    w1 = np.ones(len(key))
    L = [f"Cycle 8 T-pro-it-2.1 (32B, Qwen3-32B base) on {len(sub):,} HELM MMLU items (protocol Sections 52/56). POST HOC",
         f"\nT-pro accuracy {acc[iT]:.3f}, invalid {np.mean(tp < 0):.3f}"]
    L.append("HELM open-weight models of 27-72B on the same items: " + "; ".join(
        f"{g['models'][i]} {acc[i]:.3f} ({g['params'][i]:.0f}B)" for i in sorted(open_mid, key=lambda i: -acc[i])))
    L.append(f"  mean {np.mean([acc[i] for i in open_mid]):.3f}; T-pro rank among them {1 + sum(acc[i] > acc[iT] for i in open_mid)} of {len(open_mid) + 1}")
    L.append("Russian 8-12B local runs on the same items: " + "; ".join(f"{l} {acc[nH + loc_labels.index(l)]:.3f}" for l in RUS))
    # portfolio
    e0 = crossfit(A, key, open_all, fold, w1)
    e1 = crossfit(A, key, open_all, fold, w1, extra=iT)
    e2 = crossfit(A, key, open_all + [iT], fold, w1)
    bs = []
    nS = int(subj.max()) + 1
    for _ in range(args.boot // 5):
        cnt = np.bincount(rng.integers(0, nS, nS), minlength=nS).astype(float)
        w = cnt[subj]
        bs.append((crossfit(A, key, open_all, fold, w), crossfit(A, key, open_all, fold, w, extra=iT), crossfit(A, key, open_all + [iT], fold, w)))
    bs = np.array(bs)
    d1, d2 = bs[:, 1] - bs[:, 0], bs[:, 2] - bs[:, 0]
    L.append(f"\nOpen-weight portfolio of five (cross-fitted): {100*e0:.2f}%; with T-pro added as sixth {100*e1:.2f}% "
             f"(change {100*(e1-e0):+.2f} [{100*np.quantile(d1,.025):+.2f}, {100*np.quantile(d1,.975):+.2f}]); "
             f"re-selected with T-pro in the pool {100*e2:.2f}% (change {100*(e2-e0):+.2f} [{100*np.quantile(d2,.025):+.2f}, "
             f"{100*np.quantile(d2,.975):+.2f}]); {len(bs)} subject resamples")
    # family excess
    c = (A == key[None, :]).astype(float)
    fit = v2.fit_1plg(c)
    X = ru.excess_matrix(A, key, fit[3], v1.pooled_wrong(A, key))
    unrel = [nH + loc_labels.index(l) for l in loc_labels if l not in QWEN3_FAMILY and l != "qwen2.5-7b"]
    base = np.mean([X[iT, j] for j in unrel])
    L.append("\nExcess same-wrong agreement x of T-pro with: " + "; ".join(
        f"{l} {X[iT, nH + loc_labels.index(l)]:+.4f}" for l in QWEN3_FAMILY) + f"; mean over unrelated local models {base:+.4f} "
        f"(range {min(X[iT, j] for j in unrel):+.4f} to {max(X[iT, j] for j in unrel):+.4f})")
    text = "\n".join(L)
    open(os.path.join(HERE, "cycle8_tpro_analysis.out" if args.items == "subset" else "cycle8_tpro_full_analysis.out"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
