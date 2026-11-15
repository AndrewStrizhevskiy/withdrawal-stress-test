"""Cycle 8, post-hoc robustness analyses of the Russian-model run (protocol Section 29). POST HOC.

P1: size-matched origin comparison (Russian 8-12B vs foreign 7-12B anchors), with and without Llama-3.1-8B.
P2: own-base domestic portfolio (YandexGPT-5-Lite, saiga_yandexgpt, GigaChat3.1, GigaChat-20B).
Same data, folds, plurality rule, cross-fitting and subject-bootstrap seeds as checks/cycle8_ru_analysis.py.

    python3 checks/cycle8_ru_posthoc.py --data data --runs data/ru --workers 16
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
import cycle8_ru_analysis as ru                      # noqa: E402
import cycle8_s1_analysis as s1                      # noqa: E402

SEED = s1.SEED
RUS = ["yandexgpt5-lite-8b", "t-lite-it-2.1", "gigachat3.1-10b", "vikhr-nemo-12b", "saiga-nemo-12b", "saiga-yandexgpt-8b"]
FOR = ["qwen2.5-7b", "qwen3-8b", "llama3.1-8b", "mistral-nemo-12b"]
OWN = ["yandexgpt5-lite-8b", "saiga-yandexgpt-8b", "gigachat3.1-10b", "gigachat-20b-v1.5"]
DOMESTIC5_POOL = ["yandexgpt5-lite-8b", "t-lite-it-2.1", "gigachat3.1-10b", "vikhr-nemo-12b", "saiga-nemo-12b",
                  "saiga-yandexgpt-8b", "gigachat-20b-v1.5", "cotype-nano"]


def stats(w):
    G = s1.G
    p = s1.Pipe(w)
    idx = {m: i for i, m in enumerate(G["models"])}
    n_helm = G["n_helm"]
    open_helm = [i for i in range(n_helm) if not G["api"][i]]
    base = (p.greedy(0, 5, open_helm), p.greedy(1, 5, open_helm))
    e_open5 = p.mean(p.err("plurality", base))
    corr = (G["A"] == G["key"][None, :]).astype(float)
    acc = (corr * w[None, :]).sum(1) / w.sum()
    res = {"open5": e_open5}
    for lab in RUS + FOR:
        i = idx[lab]
        res[f"acc/{lab}"] = float(acc[i])
        res[f"adm/{lab}"] = p.mean(p.err("plurality", (base[0] + [i], base[1] + [i]))) - e_open5
    for tag, drop in (("", []), ("_nollama", ["llama3.1-8b"])):
        r = [l for l in RUS]
        f = [l for l in FOR if l not in drop]
        for k in ("acc", "adm"):
            res[f"P1/{k}/russian{tag}"] = float(np.mean([res[f"{k}/{l}"] for l in r]))
            res[f"P1/{k}/foreign{tag}"] = float(np.mean([res[f"{k}/{l}"] for l in f]))
            res[f"P1/{k}/diff{tag}"] = res[f"P1/{k}/russian{tag}"] - res[f"P1/{k}/foreign{tag}"]
        pr, pf = [idx[l] for l in r], [idx[l] for l in f]
        sr, sf = (p.greedy(0, 3, pr), p.greedy(1, 3, pr)), (p.greedy(0, 3, pf), p.greedy(1, 3, pf))
        res[f"P1/port3/russian{tag}"] = p.mean(p.err("plurality", sr))
        res[f"P1/port3/foreign{tag}"] = p.mean(p.err("plurality", sf))
        res[f"P1/port3/diff{tag}"] = res[f"P1/port3/russian{tag}"] - res[f"P1/port3/foreign{tag}"]
        if w is G["ones"]:
            res[f"_port3{tag}"] = {"russian": [[G["models"][i] for i in s] for s in sr],
                                  "foreign": [[G["models"][i] for i in s] for s in sf]}
    own = [idx[l] for l in OWN]
    dom_pool = [idx[l] for l in DOMESTIC5_POOL]
    dom5 = p.mean(p.err("plurality", (p.greedy(0, 5, dom_pool), p.greedy(1, 5, dom_pool))))
    so = (p.greedy(0, 3, own), p.greedy(1, 3, own))
    res["P2/domestic5"] = dom5
    res["P2/own_best_single"] = p.mean(p.err("best", (own, own)))
    res["P2/own_greedy3"] = p.mean(p.err("plurality", so))
    res["P2/own_all4"] = p.mean(p.err("plurality", (own, own)))
    for k in ("own_best_single", "own_greedy3", "own_all4"):
        res[f"P2/{k}_minus_domestic5"] = res[f"P2/{k}"] - dom5
    if w is G["ones"]:
        res["_own_greedy3"] = [[G["models"][i] for i in s] for s in so]
    return res


def boot_task(b):
    rng = np.random.default_rng([SEED, 1000 + b])          # same resamples as cycle8_ru_analysis.py
    n_s = s1.G["subj"].max() + 1
    w = np.bincount(rng.integers(0, n_s, n_s), minlength=n_s).astype(float)[s1.G["subj"]]
    return b, stats(w)


def init_worker(g):
    s1.G.update(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(HERE), "data"))
    ap.add_argument("--runs", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--test", action="store_true", help="code test: results are not reported")
    args = ap.parse_args()
    suffix = "_test" if args.test else ""
    t0 = time.time()
    g, n_helm, labels = ru.load(args.data, args.runs)
    g["n_helm"] = n_helm
    missing = [l for l in set(RUS + FOR + OWN + DOMESTIC5_POOL) if l not in labels]
    assert not missing, missing
    s1.G.update(g)
    point = stats(g["ones"])
    with Pool(args.workers, initializer=init_worker, initargs=(g,)) as pool:
        boots = [r for _, r in sorted(pool.map(boot_task, range(args.boot)))]

    def ci(k):
        v = np.array([b_[k] for b_ in boots], dtype=float)
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]

    out = {k: [v] + ci(k) for k, v in point.items() if not k.startswith("_")}

    def f(k, scale=100, sign=False):
        v, lo, hi = out[k]
        fmt = "{:+.2f}" if sign else "{:.2f}"
        return f"{fmt.format(scale * v)} [{fmt.format(scale * lo)}, {fmt.format(scale * hi)}]"

    L = [f"Cycle 8 POST-HOC robustness analyses (protocol Section 29){' (CODE TEST, not a result)' if args.test else ''}",
         f"{n_helm} HELM models + {len(labels)} local runs, {len(g['key'])} items, {args.boot} subject-bootstrap resamples; "
         f"values in %, 95% CI",
         "\nP1. Size-matched origin comparison: Russian 8-12B (6) vs foreign 7-12B anchors (4)"]
    for lab in RUS + FOR:
        L.append(f"  {lab}: accuracy {f('acc/' + lab)}; change in open5 error when added {f('adm/' + lab, sign=True)}")
    for tag, name in (("", "all anchors"), ("_nollama", "without Llama-3.1-8B")):
        L.append(f"  [{name}] accuracy: Russian {f('P1/acc/russian' + tag)}, foreign {f('P1/acc/foreign' + tag)}, "
                 f"difference {f('P1/acc/diff' + tag, sign=True)}")
        L.append(f"  [{name}] admission change: Russian {f('P1/adm/russian' + tag, sign=True)}, foreign "
                 f"{f('P1/adm/foreign' + tag, sign=True)}, difference {f('P1/adm/diff' + tag, sign=True)}")
        L.append(f"  [{name}] greedy-3 portfolio error: Russian {f('P1/port3/russian' + tag)}, foreign "
                 f"{f('P1/port3/foreign' + tag)}, difference {f('P1/port3/diff' + tag, sign=True)}; "
                 f"portfolios {json.dumps(point['_port3' + tag])}")
    L.append(f"  open5 error {f('open5')}")
    L.append("\nP2. Own-base domestic portfolio (YandexGPT-5-Lite, saiga_yandexgpt, GigaChat3.1, GigaChat-20B)")
    L.append(f"  domestic5 (Section 28.5, recomputed): {f('P2/domestic5')}")
    for k, name in (("own_best_single", "best single"), ("own_greedy3", "greedy n=3"), ("own_all4", "plurality of all 4")):
        L.append(f"  {name}: {f('P2/' + k)}; minus domestic5 {f('P2/' + k + '_minus_domestic5', sign=True)}")
    L.append(f"  greedy-3 members (fold 0 / fold 1): {json.dumps(point['_own_greedy3'])}")
    L.append(f"seconds {round(time.time() - t0)}")
    text = "\n".join(L)
    json.dump({"point_and_ci": out, "portfolios": {k: v for k, v in point.items() if k.startswith("_")}},
              open(os.path.join(HERE, f"cycle8_ru_posthoc{suffix}.json"), "w"), indent=1)
    open(os.path.join(HERE, f"cycle8_ru_posthoc{suffix}.out"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
