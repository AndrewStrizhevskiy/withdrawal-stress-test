"""Cycle 8, protocol Section 61 (F1): developer cap vs family cap vs joint cap on MMLU (70 models). POST HOC.

Same data, folds and seed as cycle8_s1_analysis.py. Each cap takes the most accurate model per group on the fitting fold,
five models, plurality vote, cross-fitted. Outcomes: full-availability error, held-out worst case over single-developer
withdrawals, worst case over single-family withdrawals; 200-resample subject bootstrap for the differences.

    .venv/bin/python checks/cycle8_family_cap.py --data data [--boot 200]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cycle8_s1_analysis as s1                      # noqa: E402

PALM2 = {"google_text-bison@001", "google_text-unicorn@001"}


def family_labels(models, variant):
    fam = []
    for m in models:
        dev = m.split("_")[0]
        f = dev
        if dev == "google":
            if m in PALM2:
                f = "google-palm2"
            elif "gemma" in m:
                f = "google-gemini" if variant == "gemma_in_gemini" else "google-gemma"
            else:
                f = "google-gemini"
        if m.startswith("upstage_solar") and variant != "no_solar_merge":
            f = "microsoft"                                 # Solar Pro scales Phi-3-medium (Solar Pro Preview card)
        fam.append(f)
    return np.array(fam)


def pick(acc, groups_list, n):
    """Most accurate first; accept a model if none of its group labels (one label per grouping) is already taken."""
    taken = [set() for _ in groups_list]
    chosen = []
    for i in np.argsort(-acc):
        labs = [g[i] for g in groups_list]
        if any(l in t for l, t in zip(labs, taken)):
            continue
        chosen.append(int(i))
        for l, t in zip(labs, taken):
            t.add(l)
        if len(chosen) == n:
            break
    return chosen


def evaluate(w, fam):
    p = s1.Pipe(w)
    G = s1.G
    dev = np.array([G["dev_names"][d] for d in G["dev"]])
    rules = {"developer": [dev], "family": [fam], "joint": [dev, fam]}
    port = {k: (pick(p.acc_fold(0), g, 5), pick(p.acc_fold(1), g, 5)) for k, g in rules.items()}
    dev_groups = {d: set(np.where(dev == d)[0]) for d in np.unique(dev)}
    fam_groups = {f: set(np.where(fam == f)[0]) for f in np.unique(fam)}
    res = {}
    for k, sets in port.items():
        res[f"{k}/full"] = p.mean(p.err("plurality", sets))
        for sname, groups in (("dev", dev_groups), ("fam", fam_groups)):
            present = [g for g, W in groups.items() if any(i in W for s in sets for i in s)]
            worst = -1.0
            for g in present:
                ws = s1.withdraw(sets, groups[g])
                e = p.mean(p.err("plurality", ws)) if (ws[0] and ws[1]) else 0.75
                worst = max(worst, e)
            res[f"{k}/worst_{sname}"] = worst
    res["_portfolios"] = {k: [[G["models"][i] for i in s] for s in v] for k, v in port.items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args()
    t0 = time.time()
    g = s1.load_real(args.data)
    rng = np.random.default_rng(s1.SEED)
    fold = np.zeros(len(g["key"]), dtype=int)
    for s in np.unique(g["subj"]):
        ids = np.where(g["subj"] == s)[0]
        fold[rng.permutation(ids)[: len(ids) // 2]] = 1
    g["fold"], g["ones"] = fold, np.ones(len(g["key"]))
    s1.G.update(g)
    L = ["Cycle 8 family-cap experiment (protocol Section 61, F1). POST HOC", "MMLU, 70 models, same folds as S1; errors in %"]
    out = {}
    keys = ["full", "worst_dev", "worst_fam"]
    for variant in ("primary", "no_solar_merge", "gemma_in_gemini"):
        fam = family_labels(g["models"], variant)
        point = evaluate(g["ones"], fam)
        L.append(f"\nvariant {variant}: families {len(set(fam))} vs developers {len(g['dev_names'])}")
        for k in ("developer", "family", "joint"):
            L.append(f"  {k:9s} cap: full {100*point[k+'/full']:.2f}; worst single-developer withdrawal {100*point[k+'/worst_dev']:.2f}; "
                     f"worst single-family withdrawal {100*point[k+'/worst_fam']:.2f}; fold portfolios {point['_portfolios'][k]}")
        out[variant] = {"point": point}
        if variant == "primary" and args.boot:
            rb = np.random.default_rng(s1.SEED + 61)
            nS = int(g["subj"].max()) + 1
            bs = []
            for _ in range(args.boot):
                cnt = np.bincount(rb.integers(0, nS, nS), minlength=nS).astype(float)
                r = evaluate(cnt[g["subj"]], fam)
                bs.append({f"{a}-developer/{m}": r[f"{a}/{m}"] - r[f"developer/{m}"] for a in ("family", "joint") for m in keys})
            for a in ("family", "joint"):
                for m in keys:
                    d = np.array([b[f"{a}-developer/{m}"] for b in bs])
                    pt = point[f"{a}/{m}"] - point[f"developer/{m}"]
                    L.append(f"  {a} - developer, {m}: {100*pt:+.2f} [{100*np.quantile(d,.025):+.2f}, {100*np.quantile(d,.975):+.2f}]")
                    out[variant][f"{a}-developer/{m}"] = [pt, float(np.quantile(d, .025)), float(np.quantile(d, .975))]
    L.append(f"seconds {round(time.time() - t0)}")
    json.dump(out, open(os.path.join(HERE, "cycle8_family_cap.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, "cycle8_family_cap.out"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
