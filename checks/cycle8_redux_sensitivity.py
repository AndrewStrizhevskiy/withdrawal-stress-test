"""Cycle 8, MMLU-Redux exclusion sensitivity (protocol Section 39; pre-specified in Section 8).

Flags HELM MMLU items whose MMLU-Redux 2.0 label is not "ok", then repeats the S1 analyses with those items given
weight 0 (fitting, selection and evaluation), H1 on the remaining items, and a subject bootstrap with the S1 seeds.

    python3 checks/cycle8_redux_sensitivity.py --data data --workers 16 [--boot 200] [--k-null 200]
    .venv/bin/python checks/cycle8_redux_sensitivity.py --data data --match-only     # matching report only
"""
import argparse
import gzip
import json
import os
import re
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cycle8_s1_analysis as s1                      # noqa: E402

KEY_WRONG = {"wrong_groundtruth", "no_correct_answer"}


def norm(t):
    return re.sub(r"\s+", " ", t).strip().lower()


def target_question(prompt):
    q = prompt.rsplit("Question: ", 1)[1]
    return q.split("\nA. ", 1)[0]


def flags(data_dir):
    order = {}
    items = set()
    with gzip.open(os.path.join(data_dir, "helm_mmlu_parsed.tsv.gz"), "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            fs = line.split("\t")
            items.add((fs[2], fs[3]))
    for j, it in enumerate(sorted(items)):
        order[it] = j
    helm = {}
    for line in gzip.open(os.path.join(data_dir, "helm_mmlu_prompts.jsonl.gz"), "rt", encoding="utf-8"):
        r = json.loads(line)
        helm.setdefault((r["subject"], norm(target_question(r["prompt"]))), []).append(order[(r["subject"], r["instance_id"])])
    redux = json.load(gzip.open(os.path.join(data_dir, "redux", "mmlu_redux_2.0.json.gz"), "rt", encoding="utf-8"))
    matched, flagged, keywrong = 0, set(), set()
    for r in redux:
        hits = helm.get((r["subject"], norm(r["question"])), [])
        if hits:
            matched += 1
            if r["error_type"] != "ok":
                flagged.update(hits)
            if r["error_type"] in KEY_WRONG:
                keywrong.update(hits)
    return len(order), len(redux), matched, sorted(flagged), sorted(keywrong)


def boot_task(args):
    b, mask = args
    rng = np.random.default_rng([s1.SEED, 1000 + b])          # S1 seeds
    n_s = s1.G["subj"].max() + 1
    counts = np.bincount(rng.integers(0, n_s, n_s), minlength=n_s).astype(float)
    return b, s1.analysis(counts[s1.G["subj"]] * mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(HERE), "data"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--k-null", type=int, default=200)
    ap.add_argument("--match-only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    n_items, n_redux, matched, flagged, keywrong = flags(args.data)
    L = ["Cycle 8 MMLU-Redux exclusion sensitivity (protocol Section 39)",
         f"HELM items {n_items}; Redux items {n_redux}; matched {matched} ({matched / n_redux:.3f}); "
         f"flagged HELM items {len(flagged)}; key-wrong subset {len(keywrong)}"]
    if args.match_only:
        print("\n".join(L))
        return
    g = s1.load_real(args.data)
    rng = np.random.default_rng(s1.SEED)
    fold = np.zeros(len(g["key"]), dtype=int)
    for s in np.unique(g["subj"]):
        ids = np.where(g["subj"] == s)[0]
        fold[rng.permutation(ids)[: len(ids) // 2]] = 1
    g["fold"], g["ones"] = fold, np.ones(len(g["key"]))
    s1.G.update(g)
    mask = np.ones(len(g["key"]))
    mask[flagged] = 0
    mask_kw = np.ones(len(g["key"]))
    mask_kw[keywrong] = 0
    point = s1.analysis(mask)
    point_kw = s1.analysis(mask_kw)
    print(f"point estimates done, {time.time() - t0:.0f} s", flush=True)
    # H1 on the remaining items
    keep = mask > 0
    saved = dict(s1.G)
    s1.G.update(A=g["A"][:, keep], key=g["key"][keep])
    h1 = s1.h1(np.random.default_rng([s1.SEED, 1]), args.k_null)
    s1.G.clear()
    s1.G.update(saved)
    print(f"H1 done, {time.time() - t0:.0f} s: T {h1['T']:.5f} p {h1['p']:.3f} -> {h1['verdict']}", flush=True)
    with Pool(args.workers, initializer=s1.init_worker, initargs=(g,)) as pool:
        boots = [r for _, r in sorted(pool.map(boot_task, [(b, mask) for b in range(args.boot)]))]

    def ci(f):
        v = np.array([f(r) for r in boots], dtype=float)
        v = v[~np.isnan(v)]
        return [float(np.quantile(v, .025)), float(np.quantile(v, .975))] if len(v) else [None, None]

    orig = json.load(open(os.path.join(HERE, "cycle8_s1_analysis.json")))["point"]
    rows = [("primary portfolio error", lambda r: r["D/prim/full"]),
            ("dE API-only", lambda r: r.get("D/prim/dE/api-only", np.nan)),
            ("sovereignty cost C", lambda r: r["S/open5"] - r["H3/cap/full"]),
            ("best single open", lambda r: r["S/best_single_open"]),
            ("open5", lambda r: r["S/open5"]),
            ("H3 cap held-out worst", lambda r: r["H3/cap/heldout_worst"]),
            ("H3 robust held-out worst", lambda r: r["H3/robust/heldout_worst"]),
            ("H3 gain robust vs cap", lambda r: r["H3/cap/heldout_worst"] - r["H3/robust/heldout_worst"])]
    for k in sorted(point):
        if k.startswith("D/prim/dE/dev:"):
            rows.append((f"dE {k.split('/')[-1]}", lambda r, kk=k: r.get(kk, np.nan)))
    L.append(f"\nH1 without flagged items ({int(keep.sum())} items): T {h1['T']:.5f} (S1: 0.00462), p {h1['p']:.3f}, "
             f"jackknife 90% CI [{h1['ci90'][0]:.5f}, {h1['ci90'][1]:.5f}] -> {h1['verdict']}")
    L.append("\nDescriptives (%, 95% subject-bootstrap CI with exclusion; S1 value; key-wrong-only exclusion point):")
    out = {"h1": {k: h1[k] for k in ("T", "p", "ci90", "verdict", "developers_used")}, "n_flagged": len(flagged),
           "n_keywrong": len(keywrong), "matched": matched, "desc": {}}
    for name, f in rows:
        try:
            pv, ov, kv = f(point), f(orig), f(point_kw)
        except KeyError:
            continue
        lo, hi = ci(f)
        out["desc"][name] = {"point": pv, "ci": [lo, hi], "s1": ov, "keywrong_only": kv}
        L.append(f"  {name}: {100*pv:.2f} [{100*lo:.2f}, {100*hi:.2f}]; S1 {100*ov:.2f}; key-wrong-only {100*kv:.2f}")
    for n in (3, 5, 7, 9):
        rules = [r for r in s1.RULES if r != "best"]
        fv = [point.get(f"H2/all/n{n}/{r}/full") for r in rules]
        wv = [point.get(f"H2/all/n{n}/{r}/worst") for r in rules]
        if all(v is not None for v in fv + wv):
            L.append(f"  H2 n={n}: voting-rule spread full {100*(max(fv)-min(fv)):.2f}, worst {100*(max(wv)-min(wv)):.2f}")
    L.append(f"seconds {round(time.time() - t0)}")
    json.dump(out, open(os.path.join(HERE, "cycle8_redux_sensitivity.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, "cycle8_redux_sensitivity.out"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
