"""Cycle 8, Russian-language run analysis (protocol Sections 45-46). POST HOC.

M1 accuracy and invalid share per model (Russian ruMMLU vs English MMLU local runs); M2 origin at matched size and the
difference-in-differences (Russian minus English) between Russian and foreign groups, with leave-one-model-out ranges;
M3 cross-fitted plurality trios (Russian pool vs foreign pool); M4 inheritance pairs in Russian (excess over matched
unrelated pairs, N2 null). Domain bootstrap, 1,000 resamples, seed 20261118.

    .venv/bin/python checks/cycle8_mera_analysis.py --prompts data/mera_rummlu_prompts.jsonl.gz --runs <ru_mera dir> \
        --en-runs <dir with English local jsonl> --en-parsed data/helm_mmlu_parsed.tsv.gz
"""
import argparse
import gzip
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cycle8_mera_parse as mp                       # noqa: E402
import cycle8_ru_analysis as ru                      # noqa: E402
import cycle8_s0_synthetic_validation as v1          # noqa: E402
import cycle8_s0b_synthetic_validation as v2         # noqa: E402

SEED = 20261118
LABELS = list(ru.LOCAL)
RUS6 = ["yandexgpt5-lite-8b", "t-lite-it-2.1", "gigachat3.1-10b", "vikhr-nemo-12b", "saiga-nemo-12b", "saiga-yandexgpt-8b"]
OWN3 = ["yandexgpt5-lite-8b", "saiga-yandexgpt-8b", "gigachat3.1-10b"]
FOR4 = ["qwen2.5-7b", "qwen3-8b", "llama3.1-8b", "mistral-nemo-12b"]
FOR3 = ["qwen2.5-7b", "qwen3-8b", "mistral-nemo-12b"]


def load_ru(prompts, runs):
    items = [json.loads(l) for l in gzip.open(prompts, "rt", encoding="utf-8")]
    idx = {f"{r['subject']}|{r['instance_id']}": j for j, r in enumerate(items)}
    key = np.array([r["key"] for r in items])
    subj_names = sorted({r["subject"] for r in items})
    subj = np.array([subj_names.index(r["subject"]) for r in items])
    A = np.full((len(LABELS), len(items)), -1, dtype=np.int8)
    for i, lab in enumerate(LABELS):
        n = 0
        for line in open(os.path.join(runs, f"{lab}.jsonl")):
            r = json.loads(line)
            if r["k"] in idx and "comp" in r:
                A[i, idx[r["k"]]] = mp.parse(r["comp"])
                n += 1
        assert n == len(items), (lab, n)
    return A, key, subj, subj_names


def load_en(en_runs, parsed, subj_names):
    items = set()
    with gzip.open(parsed, "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            fs = line.split("\t")
            items.add((fs[2], fs[3]))
    order = {f"{s}|{i}": j for j, (s, i) in enumerate(sorted(items))}
    subj = np.array([subj_names.index(s) if s in subj_names else -1 for s, _ in sorted(items)])
    C = np.zeros((len(LABELS), len(order)))
    for i, lab in enumerate(LABELS):
        p = os.path.join(en_runs, f"{lab}.jsonl")
        op = gzip.open(p + ".gz", "rt") if not os.path.exists(p) else open(p)
        for line in op:
            r = json.loads(line)
            if r["k"] in order and "ans" in r:
                C[i, order[r["k"]]] = float(r["ans"] == r["key"])
    return C, subj


def wacc(C, w):
    return (C * w[None, :]).sum(1) / w.sum()


def groups_diff(acc, g1, g2):
    return np.mean([acc[LABELS.index(l)] for l in g1]) - np.mean([acc[LABELS.index(l)] for l in g2])


def plurality_err(A, key, members, fit, ev, w):
    acc = np.array([((A[i, fit] == key[fit]) * w[fit]).sum() / w[fit].sum() for i in range(A.shape[0])])
    votes = np.zeros((ev.sum(), 4))
    for i in members:
        a = A[i, ev]
        ok = a >= 0
        votes[np.where(ok)[0], a[ok]] += 1.0 + 1e-6 * acc[i]
    pred = np.where(votes.max(1) > 0, votes.argmax(1), -1)
    return float(((pred != key[ev]) * w[ev]).sum() / w[ev].sum())


def greedy_trio(A, key, pool, fit, w):
    chosen = []
    for _ in range(3):
        best = min((j for j in pool if j not in chosen), key=lambda j: plurality_err(A, key, chosen + [j], fit, fit, w))
        chosen.append(best)
    return chosen


def trio_error(A, key, pool, fold, w):
    e = 0.0
    tot = 0.0
    for f in (0, 1):
        fit, ev = fold == f, fold != f
        mem = greedy_trio(A, key, pool, fit, w)
        e += plurality_err(A, key, mem, fit, ev, w) * w[ev].sum()
        tot += w[ev].sum()
    return e / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--en-runs", required=True)
    ap.add_argument("--en-parsed", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--k-null", type=int, default=200)
    args = ap.parse_args()
    t0 = time.time()
    A, key, subj, subj_names = load_ru(args.prompts, args.runs)
    Cen, subj_en = load_en(args.en_runs, args.en_parsed, subj_names)
    Cru = (A == key[None, :]).astype(float)
    valid = (A >= 0)
    nS = len(subj_names)
    rng = np.random.default_rng(SEED)
    fold = np.zeros(len(key), dtype=int)
    for s in range(nS):
        ids = np.where(subj == s)[0]
        fold[rng.permutation(ids)[: len(ids) // 2]] = 1
    ones_ru, ones_en = np.ones(len(key)), np.ones(Cen.shape[1])
    acc_ru, acc_en = wacc(Cru, ones_ru), wacc(Cen, ones_en)
    acc_valid = (Cru * valid).sum(1) / np.maximum(valid.sum(1), 1)
    L = ["Cycle 8 Russian-language run (MERA ruMMLU public_test; protocol Sections 45-46). POST HOC",
         f"{len(key)} items, {nS} domains; English: local HELM MMLU runs ({Cen.shape[1]} items); bootstrap {args.boot} over domains"]
    L.append("\nM1. Accuracy (Russian / English), invalid share and accuracy on valid answers (Russian):")
    for i, lab in enumerate(LABELS):
        L.append(f"  {lab}: RU {acc_ru[i]:.3f} / EN {acc_en[i]:.3f} (change {acc_ru[i]-acc_en[i]:+.3f}); invalid {1-valid[i].mean():.3f}; "
                 f"valid-only {acc_valid[i]:.3f}")
    # M2 with domain bootstrap
    contrasts = {"RUS6 - FOR3": (RUS6, FOR3), "OWN3 - FOR3": (OWN3, FOR3), "RUS6 - FOR4": (RUS6, FOR4), "OWN3 - FOR4": (OWN3, FOR4)}
    boots = []
    for b in range(args.boot):
        cnt = np.bincount(rng.integers(0, nS, nS), minlength=nS).astype(float)
        w_ru = cnt[subj]
        w_en = np.where(subj_en >= 0, cnt[np.maximum(subj_en, 0)], 0.0)
        ar, ae = wacc(Cru, w_ru), wacc(Cen, w_en)
        av = (Cru * valid * w_ru[None, :]).sum(1) / np.maximum((valid * w_ru[None, :]).sum(1), 1e-9)
        boots.append((ar, ae, av))
    out = {"M1": {lab: {"ru": float(acc_ru[i]), "en": float(acc_en[i]), "invalid": float(1 - valid[i].mean()),
                        "valid_only": float(acc_valid[i])} for i, lab in enumerate(LABELS)}, "M2": {}}
    L.append("\nM2. Origin at matched size (points; 95% domain-bootstrap CI; leave-one-model-out range):")
    for name, (g1, g2) in contrasts.items():
        for qn, f in (("Russian accuracy", lambda ar, ae, av: groups_diff(ar, g1, g2)),
                      ("diff-in-diff (RU-EN)", lambda ar, ae, av: groups_diff(ar - ae, g1, g2)),
                      ("Russian accuracy, valid answers only", lambda ar, ae, av: groups_diff(av, g1, g2))):
            pv = f(acc_ru, acc_en, acc_valid)
            bv = np.array([f(*b) for b in boots])
            loo = []
            for x in g1 + g2:
                h1 = [l for l in g1 if l != x] or g1
                h2 = [l for l in g2 if l != x] or g2
                loo.append(groups_diff(acc_ru, h1, h2) if qn == "Russian accuracy" else
                           groups_diff(acc_ru - acc_en, h1, h2) if qn.startswith("diff") else groups_diff(acc_valid, h1, h2))
            ci = [float(np.quantile(bv, .025)), float(np.quantile(bv, .975))]
            out["M2"][f"{name} / {qn}"] = {"point": float(pv), "ci": ci, "loo": [float(min(loo)), float(max(loo))]}
            L.append(f"  {name} / {qn}: {100*pv:+.2f} [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]; LOO [{100*min(loo):+.2f}, {100*max(loo):+.2f}]")
    # M3 trios
    L.append("\nM3. Cross-fitted plurality trios (error %, 95% domain-bootstrap CI over 200 resamples):")
    pools = {"Russian 8-12B (6)": [LABELS.index(l) for l in RUS6], "all Russian (8)": [i for i, l in enumerate(LABELS) if ru.LOCAL[l][0] == "russian"],
             "foreign (4)": [LABELS.index(l) for l in FOR4], "foreign without Llama (3)": [LABELS.index(l) for l in FOR3]}
    pt = {k: trio_error(A, key, v, fold, ones_ru) for k, v in pools.items()}
    tb = {k: [] for k in pools}
    for b in range(200):
        cnt = np.bincount(rng.integers(0, nS, nS), minlength=nS).astype(float)
        w = cnt[subj]
        for k, v in pools.items():
            tb[k].append(trio_error(A, key, v, fold, w))
    out["M3"] = {}
    for k in pools:
        ci = [float(np.quantile(tb[k], .025)), float(np.quantile(tb[k], .975))]
        out["M3"][k] = {"point": pt[k], "ci": ci}
        L.append(f"  {k}: {100*pt[k]:.2f} [{100*ci[0]:.2f}, {100*ci[1]:.2f}]")
    d = np.array(tb["Russian 8-12B (6)"]) - np.array(tb["foreign without Llama (3)"])
    L.append(f"  Russian 8-12B minus foreign without Llama: {100*(pt['Russian 8-12B (6)']-pt['foreign without Llama (3)']):+.2f} "
             f"[{100*np.quantile(d,.025):+.2f}, {100*np.quantile(d,.975):+.2f}]")
    # M4 inheritance
    fit = v2.fit_1plg(Cru)
    wrong = v1.pooled_wrong(A, key)
    X = ru.excess_matrix(A, key, fit[3], wrong)
    loc = {i: lab for i, lab in enumerate(LABELS)}
    comp, unrel = ru.comparison_sets(acc_ru, list(range(len(LABELS))), loc)
    nulls = {k: [] for k in comp}
    rng2 = np.random.default_rng([SEED, 7])
    for _ in range(args.k_null):
        a0 = v1.simulate_n2(rng2, fit[3], wrong, key)
        f0 = v2.fit_1plg((a0 == key[None, :]).astype(float), init=fit[:3], iters=8)
        X0 = ru.excess_matrix(a0, key, f0[3], v1.pooled_wrong(a0, key))
        for k, (i, j, m, t) in comp.items():
            if m:
                nulls[k].append(X0[i, j] - np.mean([X0[a, b] for a, b in m]))
    L.append("\nM4. Inheritance in Russian: excess x of the pair minus matched unrelated local pairs; N2 p (English value in brackets):")
    en_inh = json.load(open(os.path.join(HERE, "cycle8_ru_analysis.json")))["inheritance"]
    out["M4"] = {}
    for k, (i, j, m, t) in comp.items():
        if not m:
            L.append(f"  {k[0]} ~ {k[1]}: no comparison pairs")
            continue
        dd = float(X[i, j] - np.mean([X[a, b] for a, b in m]))
        pv = float((1 + sum(v >= dd for v in nulls[k])) / (1 + len(nulls[k])))
        en = en_inh.get("|".join(k), {}).get("excess_vs_matched")
        out["M4"]["|".join(k)] = {"excess": dd, "p": pv, "n_matched": len(m), "en": en}
        L.append(f"  {k[0]} ~ {k[1]}: {dd:+.4f} (matched {len(m)}, tol {t}); p {pv:.3f}  [EN {en:+.4f}]" if en is not None else
                 f"  {k[0]} ~ {k[1]}: {dd:+.4f}; p {pv:.3f}")
    L.append(f"seconds {round(time.time() - t0)}")
    json.dump(out, open(os.path.join(HERE, "cycle8_mera_analysis.json"), "w"), indent=1)
    open(os.path.join(HERE, "cycle8_mera_analysis.out"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
