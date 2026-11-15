"""Cycle 8, POST HOC (protocol Section 35, R2): origin vs scale with own-base Russian models, and leave-one-model-out.
Reuses stats() of checks/cycle8_ru_posthoc.py (same data, folds, seeds); earlier outputs are untouched.
    .venv/bin/python checks/cycle8_ru_posthoc_r2.py --data data --runs <dir with local jsonl>
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cycle8_ru_posthoc as ph                       # noqa: E402
import cycle8_ru_analysis as ru                      # noqa: E402
import cycle8_s1_analysis as s1                      # noqa: E402

OWN8 = ["yandexgpt5-lite-8b", "saiga-yandexgpt-8b", "gigachat3.1-10b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args()
    g, n_helm, labels = ru.load(args.data, args.runs)
    g["n_helm"] = n_helm
    s1.G.update(g)
    point = ph.stats(g["ones"])
    boots = [ph.boot_task(b)[1] for b in range(args.boot)]
    L = ["Cycle 8 POST HOC (Section 35, R2): origin vs scale; values in points (accuracy and admission change, %)"]
    out = {}
    for rname, rus in (("all six Russian 8-12B", ph.RUS), ("own-base Russian 8-12B (3)", OWN8)):
        for fname, fo in (("all anchors", ph.FOR), ("without Llama", [l for l in ph.FOR if l != "llama3.1-8b"])):
            for k in ("acc", "adm"):
                d = lambda r: np.mean([r[f"{k}/{l}"] for l in rus]) - np.mean([r[f"{k}/{l}"] for l in fo])
                pv = d(point)
                bv = np.array([d(r) for r in boots])
                loo = [np.mean([point[f"{k}/{l}"] for l in rus if l != x]) - np.mean([point[f"{k}/{l}"] for l in fo]) for x in rus if len(rus) > 1] + \
                      [np.mean([point[f"{k}/{l}"] for l in rus]) - np.mean([point[f"{k}/{l}"] for l in fo if l != x]) for x in fo if len(fo) > 1]
                key = f"{rname} vs {fname} / {k}"
                out[key] = {"diff": float(pv), "ci": [float(np.quantile(bv, .025)), float(np.quantile(bv, .975))], "loo": [float(min(loo)), float(max(loo))]}
                L.append(f"  {key}: {100*pv:+.2f} [{100*np.quantile(bv,.025):+.2f}, {100*np.quantile(bv,.975):+.2f}]; leave-one-model-out range [{100*min(loo):+.2f}, {100*max(loo):+.2f}]")
    json.dump(out, open(os.path.join(HERE, "cycle8_ru_posthoc_r2.json"), "w"), indent=1)
    open(os.path.join(HERE, "cycle8_ru_posthoc_r2.out"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
