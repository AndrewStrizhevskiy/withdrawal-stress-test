"""Cycle 8, Russian-model run (protocol Section 26): bridge, inheritance and admission analyses on HELM MMLU.

Combines the 70-model HELM panel with the local Ollama runs (data/ru/<label>.jsonl). Reuses the frozen S1 functions.

    python3 checks/cycle8_ru_analysis.py --data data --runs data/ru --workers 16
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
import cycle8_s0_synthetic_validation as v1          # noqa: E402
import cycle8_s0b_synthetic_validation as v2         # noqa: E402
import cycle8_s1_analysis as s1                      # noqa: E402

SEED = s1.SEED
# local label -> (role, developer group, HELM counterpart or None)
LOCAL = {
    "yandexgpt5-lite-8b": ("russian", "yandex", None),
    "t-lite-it-2.1": ("russian", "t-tech", None),
    "gigachat3.1-10b": ("russian", "ai-sage", None),
    "vikhr-nemo-12b": ("russian", "vikhr", None),
    "saiga-nemo-12b": ("russian", "gusev", None),
    "saiga-yandexgpt-8b": ("russian", "gusev", None),
    "gigachat-20b-v1.5": ("russian", "ai-sage", None),
    "cotype-nano": ("russian", "mts", None),
    "qwen3-8b": ("anchor", "qwen", None),
    "qwen2.5-7b": ("anchor", "qwen", "qwen_qwen2.5-7b-instruct-turbo"),
    "llama3.1-8b": ("anchor", "meta", "meta_llama-3.1-8b-instruct-turbo"),
    "mistral-nemo-12b": ("anchor", "mistralai", "mistralai_open-mistral-nemo-2407"),
}
# declared or name-implied lineage pairs tested for inheritance (derivative, base); kind in the label
PAIRS = [("t-lite-it-2.1", "qwen3-8b", "fine-tune of declared base"),
         ("vikhr-nemo-12b", "mistral-nemo-12b", "fine-tune of declared base"),
         ("saiga-nemo-12b", "mistral-nemo-12b", "fine-tune, base by name"),
         ("vikhr-nemo-12b", "saiga-nemo-12b", "siblings on the same base"),
         ("saiga-yandexgpt-8b", "yandexgpt5-lite-8b", "fine-tune, base by name"),
         ("gigachat3.1-10b", "gigachat-20b-v1.5", "same developer, different models")]
LINEAGE = {"t-lite-it-2.1": "qwen3", "qwen3-8b": "qwen3", "vikhr-nemo-12b": "nemo", "saiga-nemo-12b": "nemo",
           "mistral-nemo-12b": "nemo", "saiga-yandexgpt-8b": "yagpt", "yandexgpt5-lite-8b": "yagpt",
           "gigachat3.1-10b": "giga", "gigachat-20b-v1.5": "giga", "qwen2.5-7b": "qwen2.5", "llama3.1-8b": "llama3.1",
           "cotype-nano": "cotype"}


def load(data_dir, runs_dir):
    g = s1.load_real(data_dir)
    order = {}
    # item order of load_real: sorted (subject, instance_id)
    import gzip
    items = set()
    with gzip.open(os.path.join(data_dir, "helm_mmlu_parsed.tsv.gz"), "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            fs = line.split("\t")
            items.add((fs[2], fs[3]))
    for j, it in enumerate(sorted(items)):
        order[f"{it[0]}|{it[1]}"] = j
    labels, rows = [], []
    for lab in LOCAL:
        p = os.path.join(runs_dir, f"{lab}.jsonl")
        if not os.path.exists(p):
            print(f"missing run: {lab}", flush=True)
            continue
        a = np.full(len(order), -1, dtype=np.int8)
        n = 0
        for line in open(p):
            r = json.loads(line)
            if "ans" in r and r["k"] in order:
                a[order[r["k"]]] = r["ans"]
                n += 1
        if n < len(order):
            print(f"incomplete run: {lab} {n}/{len(order)}", flush=True)
            continue
        labels.append(lab)
        rows.append(a)
    n_helm = len(g["models"])
    g["A"] = np.vstack([g["A"]] + [r[None, :] for r in rows]).astype(np.int8)
    g["models"] = g["models"] + labels
    devs = g["dev_names"] + sorted({"local:" + LOCAL[l][1] for l in labels})
    g["dev"] = np.concatenate([g["dev"], [devs.index("local:" + LOCAL[l][1]) for l in labels]]).astype(int)
    g["dev_names"] = devs
    g["api"] = np.concatenate([g["api"], np.zeros(len(labels), dtype=bool)])
    g["pclass"] = np.concatenate([g["pclass"], np.full(len(labels), -1)])
    g["params"] = np.concatenate([g["params"], np.full(len(labels), np.nan)])
    rng = np.random.default_rng(SEED)
    fold = np.zeros(len(g["key"]), dtype=int)
    for s in np.unique(g["subj"]):
        ids = np.where(g["subj"] == s)[0]
        fold[rng.permutation(ids)[: len(ids) // 2]] = 1
    g["fold"], g["ones"] = fold, np.ones(len(g["key"]))
    return g, n_helm, labels


def excess_matrix(A, key, p, wrong):
    n = A.shape[1]
    S = np.zeros((A.shape[0], A.shape[0]))
    for o in range(4):
        ind = ((A == o) & (key[None, :] != o)).astype(float)
        S += ind @ ind.T
    E = ((1 - p) * (wrong ** 2).sum(1)[None, :]) @ (1 - p).T
    return (S - E) / n


def comparison_sets(acc, idx_local, labels_local):
    """For each tested pair, unrelated local-local pairs matched on both accuracies (+-0.03 doubling to 0.12, >= 5)."""
    out = {}
    unrel = [(a, b) for ai, a in enumerate(idx_local) for b in idx_local[ai + 1:]
             if LINEAGE[labels_local[a]] != LINEAGE[labels_local[b]] and LOCAL[labels_local[a]][1] != LOCAL[labels_local[b]][1]]
    for d, b, _ in PAIRS:
        if d not in labels_local.values() or b not in labels_local.values():
            continue
        i = [k for k, v in labels_local.items() if v == d][0]
        j = [k for k, v in labels_local.items() if v == b][0]
        t, m = 0.03, []
        while t <= 0.12 + 1e-9:
            m = [(x, y) for x, y in unrel if
                 (abs(acc[x] - acc[i]) <= t and abs(acc[y] - acc[j]) <= t) or (abs(acc[y] - acc[i]) <= t and abs(acc[x] - acc[j]) <= t)]
            if len(m) >= 5:
                break
            t *= 2
        out[(d, b)] = (i, j, m, t if len(m) >= 5 else None)
    return out, unrel


def boot_task(b):
    rng = np.random.default_rng([SEED, 1000 + b])
    n_s = s1.G["subj"].max() + 1
    w = np.bincount(rng.integers(0, n_s, n_s), minlength=n_s).astype(float)[s1.G["subj"]]
    return b, admission(w)


def admission(w):
    G = s1.G
    p = s1.Pipe(w)
    n_helm = G["n_helm"]
    open_helm = [i for i in range(n_helm) if not G["api"][i]]
    rus = [i for i in range(n_helm, len(G["models"])) if LOCAL[G["models"][i]][0] == "russian"]
    res = {}
    base = (p.greedy(0, 5, open_helm), p.greedy(1, 5, open_helm))
    res["open5"] = p.mean(p.err("plurality", base))
    for r in rus:
        res[f"open5+{G['models'][r]}"] = p.mean(p.err("plurality", (base[0] + [r], base[1] + [r])))
        res[f"single/{G['models'][r]}"] = p.mean(p.err("best", ([r], [r])))
    reopen = (p.greedy(0, 5, open_helm + rus), p.greedy(1, 5, open_helm + rus))
    res["open5_with_russian_pool"] = p.mean(p.err("plurality", reopen))
    res["n_russian_in_reselected"] = float(np.mean([sum(i in rus for i in s) for s in reopen]))
    dom = (p.greedy(0, 5, rus), p.greedy(1, 5, rus))
    res["domestic5"] = p.mean(p.err("plurality", dom))
    res["best_single_domestic"] = p.mean(p.err("best", (rus, rus)))
    res["best_single_open_helm"] = p.mean(p.err("best", (open_helm, open_helm)))
    if w is G["ones"]:
        res["_portfolios"] = {"open5": [[G["models"][i] for i in s] for s in base],
                              "open5_with_russian_pool": [[G["models"][i] for i in s] for s in reopen],
                              "domestic5": [[G["models"][i] for i in s] for s in dom]}
    return res


def init_worker(g):
    s1.G.update(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(HERE), "data"))
    ap.add_argument("--runs", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--k-null", type=int, default=200)
    args = ap.parse_args()
    t0 = time.time()
    g, n_helm, labels = load(args.data, args.runs)
    g["n_helm"] = n_helm
    s1.G.update(g)
    A, key = g["A"], g["key"]
    acc = (A == key[None, :]).mean(1)
    L = [f"Cycle 8 Russian-model run: {n_helm} HELM models + {len(labels)} local runs ({', '.join(labels)}), {len(key)} items"]
    loc = {n_helm + k: lab for k, lab in enumerate(labels)}
    L.append("\nLocal accuracy and invalid share:")
    for i, lab in loc.items():
        L.append(f"  {lab} ({LOCAL[lab][0]}): accuracy {acc[i]:.4f}, invalid {(A[i] < 0).mean():.4f}")
    # ---- 1. bridge
    c = (A == key[None, :]).astype(float)
    fit = v2.fit_1plg(c)
    wrong = v1.pooled_wrong(A, key)
    X = excess_matrix(A, key, fit[3], wrong)
    L.append("\n1. Bridge: local run against its HELM counterpart")
    hi_ = {m: i for i, m in enumerate(g["models"][:n_helm])}
    for i, lab in loc.items():
        cp = LOCAL[lab][2]
        if cp is None or cp not in hi_:
            continue
        h = hi_[cp]
        agree = float((A[i] == A[h]).mean())
        others = [k for k in range(n_helm) if k != h]
        best_other = max(others, key=lambda k: (A[k] == A[h]).mean())
        L.append(f"  {lab} vs {cp}: answer agreement {agree:.4f}; accuracy local {acc[i]:.4f}, HELM {acc[h]:.4f}; "
                 f"excess x {X[i, h]:.5f}; most similar other HELM model to {cp}: {g['models'][best_other]} "
                 f"(agreement {(A[best_other] == A[h]).mean():.4f}, x {X[best_other, h]:.5f})")
    # deployment check: unrelated local-local pairs vs unrelated local-HELM pairs (similar accuracy)
    idx_local = list(loc)
    comp, unrel = comparison_sets(acc, idx_local, loc)
    x_ll = [X[a, b] for a, b in unrel]
    x_lh = [X[i, h] for i in idx_local for h in range(n_helm) if abs(acc[i] - acc[h]) <= 0.05
            and not (LOCAL[loc[i]][2] == g["models"][h])]
    L.append(f"  deployment check: mean x over unrelated local-local pairs {np.mean(x_ll):.5f} ({len(x_ll)} pairs); "
             f"over local-HELM pairs within 0.05 accuracy {np.mean(x_lh):.5f} ({len(x_lh)} pairs)")
    # ---- 2. inheritance
    L.append("\n2. Inheritance: excess same-wrong agreement x of the pair minus the mean over matched unrelated local pairs")
    rng = np.random.default_rng([SEED, 7])
    nulls = {k: [] for k in comp}
    for _ in range(args.k_null):
        a0 = v1.simulate_n2(rng, fit[3], wrong, key)
        f0 = v2.fit_1plg((a0 == key[None, :]).astype(float), init=fit[:3], iters=8)
        X0 = excess_matrix(a0, key, f0[3], v1.pooled_wrong(a0, key))
        for k, (i, j, m, t) in comp.items():
            if m:
                nulls[k].append(X0[i, j] - np.mean([X0[a, b] for a, b in m]))
    inher = {}
    for k, (i, j, m, t) in comp.items():
        kind = [p_[2] for p_ in PAIRS if (p_[0], p_[1]) == k][0]
        if not m:
            L.append(f"  {k[0]} ~ {k[1]} ({kind}): no comparison pairs")
            continue
        d = float(X[i, j] - np.mean([X[a, b] for a, b in m]))
        pv = float((1 + sum(v >= d for v in nulls[k])) / (1 + len(nulls[k])))
        bw = ((A[i] != key) & (A[j] != key))
        same = float(((A[i] == A[j]) & bw).sum() / max(bw.sum(), 1))
        inher["|".join(k)] = {"x_pair": float(X[i, j]), "excess_vs_matched": d, "p": pv, "n_matched": len(m), "tol": t,
                              "same_wrong_given_both_wrong": same, "kind": kind}
        L.append(f"  {k[0]} ~ {k[1]} ({kind}): x {X[i, j]:.5f}, excess over {len(m)} matched pairs (tol {t}) {d:.5f}, "
                 f"N2 p = {pv:.3f}; P(same answer | both wrong) {same:.3f}; accuracies {acc[i]:.3f}/{acc[j]:.3f}")
    L.append(f"  (for scale: S1 H1 mean within-developer excess T = 0.00462)")
    # ---- 3. admission and sovereignty
    point = admission(g["ones"])
    with Pool(args.workers, initializer=init_worker, initargs=(g,)) as pool:
        boots = [r for _, r in sorted(pool.map(boot_task, range(args.boot)))]

    def ci(k):
        v = np.array([b_[k] for b_ in boots if k in b_], dtype=float)
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    L.append(f"\n3. Admission and sovereignty (error %, 95% subject-bootstrap CI, {args.boot} resamples)")
    adm = {}
    for k, v in point.items():
        if k.startswith("_") or k == "n_russian_in_reselected":
            continue
        adm[k] = [v] + ci(k)
        line = f"  {k}: {100 * v:.2f} [{100 * adm[k][1]:.2f}, {100 * adm[k][2]:.2f}]"
        if k.startswith("open5+"):
            dd = [b_[k] - b_["open5"] for b_ in boots]
            line += f"; change vs open5 {100 * (v - point['open5']):+.2f} [{100 * np.quantile(dd, 0.025):+.2f}, {100 * np.quantile(dd, 0.975):+.2f}]"
        L.append(line)
    L.append(f"  Russian models in the re-selected open portfolio: {point['n_russian_in_reselected']:.1f} of 5 (mean over folds)")
    L.append(f"  portfolios: {json.dumps(point['_portfolios'])}")
    L.append(f"seconds {round(time.time() - t0)}")
    out = {"labels": labels, "accuracy": {lab: float(acc[i]) for i, lab in loc.items()}, "inheritance": inher,
           "admission": adm, "portfolios": point["_portfolios"]}
    json.dump(out, open(os.path.join(HERE, "cycle8_ru_analysis.json"), "w"), indent=1)
    text = "\n".join(L)
    open(os.path.join(HERE, "cycle8_ru_analysis.out"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
