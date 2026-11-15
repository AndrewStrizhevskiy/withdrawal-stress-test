"""Cycle 8, S1: primary analysis of the withdrawal stress test (protocol Sections 5, 13, 18).

    python3 checks/cycle8_s1_analysis.py --data data --workers 16 --boot 200        # real data (server)
    .venv/bin/python checks/cycle8_s1_analysis.py --synthetic --boot 4 --workers 4   # code test on simulated data

Numpy only. Writes checks/cycle8_s1_analysis[_synthetic].out and checks/cycle8_s1_bootstrap[_synthetic].jsonl.
"""
import argparse
import collections
import gzip
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
M_OPT, TIE_EPS, SEED = 4, 1e-6, 20261114
MDE_H1 = 0.0028
PARAMS_B = {  # total parameters, billions (model names and cards)
    "01-ai_yi-34b": 34, "01-ai_yi-6b": 6, "ai21_jamba-1.5-large": 398, "ai21_jamba-1.5-mini": 52, "allenai_olmo-7b": 7,
    "allenai_olmo-1.7-7b": 7, "cohere_command-r": 35, "cohere_command-r-plus": 104, "databricks_dbrx-instruct": 132,
    "deepseek-ai_deepseek-llm-67b-chat": 67, "deepseek-ai_deepseek-v3": 671, "google_gemma-2-27b": 27, "google_gemma-2-9b": 9,
    "google_gemma-7b": 8.5, "meta_llama-2-7b": 7, "meta_llama-2-13b": 13, "meta_llama-2-70b": 70, "meta_llama-3-8b": 8,
    "meta_llama-3-70b": 70, "meta_llama-3.1-8b-instruct-turbo": 8, "meta_llama-3.1-70b-instruct-turbo": 70,
    "meta_llama-3.1-405b-instruct-turbo": 405, "meta_llama-3.3-70b-instruct-turbo": 70, "microsoft_phi-2": 2.7,
    "microsoft_phi-3-small-8k-instruct": 7, "microsoft_phi-3-medium-4k-instruct": 14, "mistralai_mistral-7b-v0.1": 7,
    "mistralai_mistral-7b-instruct-v0.3": 7, "mistralai_mistral-large-2407": 123, "mistralai_mixtral-8x22b": 141,
    "mistralai_mixtral-8x7b-32kseqlen": 47, "mistralai_open-mistral-nemo-2407": 12, "nvidia_nemotron-4-340b-instruct": 340,
    "qwen_qwen1.5-7b": 7, "qwen_qwen1.5-14b": 14, "qwen_qwen1.5-32b": 32, "qwen_qwen1.5-72b": 72, "qwen_qwen1.5-110b-chat": 110,
    "qwen_qwen2-72b-instruct": 72, "qwen_qwen2.5-7b-instruct-turbo": 7, "qwen_qwen2.5-72b-instruct-turbo": 72,
    "snowflake_snowflake-arctic-instruct": 480}
ENGINEERING = ["electrical_engineering", "college_physics", "high_school_physics", "conceptual_physics",
               "college_chemistry", "high_school_chemistry"]
RULES = ("best", "plurality", "npw", "ds", "stack")
G = {}   # global data for worker processes


# ================================================================== data
def load_real(data_dir):
    panel = json.load(open(os.path.join(data_dir, "helm_mmlu_panel.json")))
    models = panel["dedup_all_prompts"]
    idx_m = {m: i for i, m in enumerate(models)}
    items, rows, first_hash = {}, [], {}
    with gzip.open(os.path.join(data_dir, "helm_mmlu_parsed.tsv.gz"), "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            m, v, s, iid, ans, key, ph, comp = line.rstrip("\n").split("\t")
            if m not in idx_m:
                continue
            k = items.setdefault((s, iid), len(items))
            rows.append((idx_m[m], k, int(ans), int(key)))
            first_hash.setdefault(m, {})[(s, iid)] = ph
    order = sorted(items, key=lambda x: (x[0], x[1]))
    remap = {items[it]: j for j, it in enumerate(order)}
    N, M = len(order), len(models)
    A = np.full((M, N), -1, dtype=np.int8)
    key = np.zeros(N, dtype=np.int64)
    for i, k, a, kk in rows:
        A[i, remap[k]] = a
        key[remap[k]] = kk
    subjects = sorted({it[0] for it in order})
    subj = np.array([subjects.index(it[0]) for it in order])
    eng = np.array([it[0] in ENGINEERING for it in order])
    ref = order[0]
    share = panel["prompt_share"]
    classes = {}
    for m in models:
        if share[m] > 0.9:
            classes[m] = "modal"
        else:
            classes[m] = "nova" if m.startswith("amazon_") else "none:" + first_hash[m][ref]
    names = sorted(set(classes.values()))
    pclass = np.array([names.index(classes[m]) for m in models])
    dev_names = sorted({m.split("_")[0] for m in models})
    dev = np.array([dev_names.index(m.split("_")[0]) for m in models])
    api = np.array([m not in panel["open_weight"] for m in models])
    params = np.array([PARAMS_B.get(m, np.nan) for m in models])
    return dict(models=models, A=A, key=key, subj=subj, eng=eng, dev=dev, dev_names=dev_names, api=api,
                pclass=pclass, pclass_names=names, params=params, strict=set(panel["primary"]))


def load_synthetic():
    import cycle8_s0_synthetic_validation as v1
    rng = np.random.default_rng(7)
    world = v1.draw_world(rng, 0.3)
    items = v1.draw_items(rng, 57, 246)
    A = v1.simulate_answers(rng, world, items)
    models = [f"{v1.STRUCTURE[d][0]}_m{i}" for i, d in enumerate(v1.DEV)]
    eng = np.isin(items["subj"], [0, 1, 2, 3, 4, 5])
    params = np.where(rng.random(len(models)) < 0.5, 7.0, 70.0)
    return dict(models=models, A=A.astype(np.int8), key=items["key"], subj=items["subj"], eng=eng, dev=v1.DEV.copy(),
                dev_names=[s[0] for s in v1.STRUCTURE], api=v1.API.copy(), pclass=np.zeros(len(models), dtype=int),
                pclass_names=["modal"], params=params, strict=set(models[:30]))


# ================================================================== aggregation rules
def onehot(ans):
    oh = np.zeros(ans.shape + (M_OPT,))
    valid = ans >= 0
    np.put_along_axis(oh, np.where(valid, ans, 0)[..., None].astype(np.int64), 1.0, axis=-1)
    return oh * valid[..., None]


def wacc(ans, key, w):
    return ((ans == key[None, :]) * w[None, :]).sum(1) / max(w.sum(), 1e-12)


def predict(rule, Af, kf, wf, Ae):
    """Af/Ae: (n, items) answers on fitting / evaluation items. Returns predictions for evaluation items (-1 = none)."""
    n = Af.shape[0]
    acc = wacc(Af, kf, wf)
    if rule == "best" or n == 1:
        return Ae[int(np.argmax(acc))].astype(np.int64)
    if rule == "plurality" or (rule == "ds" and n < 3):
        s = (onehot(Ae) * (1.0 + TIE_EPS * acc[:, None, None])).sum(0)
        out = np.argmax(s, -1)
        return np.where(s.max(-1) > 0, out, -1)
    if rule == "npw":
        p = np.clip(acc, 1 / M_OPT + 0.01, 0.99)
        wt = np.log((M_OPT - 1) * p / (1 - p))
        s = (onehot(Ae) * (wt + TIE_EPS * acc)[:, None, None]).sum(0)
        return np.where(s.max(-1) > 0, np.argmax(s, -1), -1)
    if rule == "ds":
        return ds_predict(Af, wf, Ae, acc)
    if rule == "stack":
        return stack_predict(Af, kf, wf, Ae)
    raise ValueError(rule)


def ds_predict(Af, wf, Ae, acc, iters=25):
    n = Af.shape[0]
    ohf = onehot(Af)                                        # (n, items, K)
    s = (ohf * (1.0 + TIE_EPS * acc[:, None, None])).sum(0)
    post = np.full(s.shape, 0.05)
    post[np.arange(len(s)), np.argmax(s, -1)] += 1
    post /= post.sum(1, keepdims=True)
    for _ in range(iters):
        pw = post * wf[:, None]
        pi = pw.sum(0) + 0.5
        pi /= pi.sum()
        conf = np.einsum("ic,nik->nck", pw, ohf) + 0.5          # (n, true c, answer k)
        conf /= conf.sum(2, keepdims=True)
        lp = np.log(pi)[None, :] + np.einsum("nik,nck->ic", ohf, np.log(conf))
        lp -= lp.max(1, keepdims=True)
        post = np.exp(lp)
        post /= post.sum(1, keepdims=True)
    ohe = onehot(Ae)
    lp = np.log(pi)[None, :] + np.einsum("nik,nck->ic", ohe, np.log(conf))
    return np.argmax(lp, 1)


def stack_predict(Af, kf, wf, Ae, iters=200, lam=0.01, lr=0.1):
    Xf = onehot(Af).transpose(1, 0, 2).reshape(Af.shape[1], -1)
    Xe = onehot(Ae).transpose(1, 0, 2).reshape(Ae.shape[1], -1)
    Xf = np.hstack([Xf, np.ones((len(Xf), 1))])
    Xe = np.hstack([Xe, np.ones((len(Xe), 1))])
    Y = np.zeros((len(kf), M_OPT))
    Y[np.arange(len(kf)), kf] = 1
    W = np.zeros((Xf.shape[1], M_OPT))
    m, v = np.zeros_like(W), np.zeros_like(W)
    sw = wf / max(wf.sum(), 1e-12)
    for t in range(1, iters + 1):
        Z = Xf @ W
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        g = Xf.T @ ((P - Y) * sw[:, None]) + lam * W
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        W -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return np.argmax(Xe @ W, 1)


# ================================================================== pipeline with item weights
class Pipe:
    def __init__(self, w):
        self.w = w
        self.cache = {}

    def err(self, rule, sets):
        """sets: (members on fold 0 as fitting fold, members on fold 1 as fitting fold). Per-item error, all items."""
        out = np.zeros(len(G["key"]))
        for f in (0, 1):
            mem = tuple(sorted(sets[f]))
            ev = G["fold"] != f
            if not mem:
                out[ev] = np.nan
                continue
            ck = (rule, f, mem)
            if ck not in self.cache:
                fi = G["fold"] == f
                Af, Ae = G["A"][list(mem)][:, fi], G["A"][list(mem)][:, ev]
                pred = predict(rule, Af, G["key"][fi], self.w[fi], Ae)
                self.cache[ck] = (pred != G["key"][ev]).astype(float)
            out[ev] = self.cache[ck]
        return out

    def mean(self, e, mask=None):
        w = self.w if mask is None else self.w * mask
        ok = ~np.isnan(e)
        return float((e[ok] * w[ok]).sum() / max(w[ok].sum(), 1e-12))

    def acc_fold(self, f):
        fi = G["fold"] == f
        return wacc(G["A"][:, fi], G["key"][fi], self.w[fi])

    # ---- portfolio selection on the fitting fold f
    def contrib_fold(self, f):
        ck = ("contrib", f)
        if ck not in self.cache:
            fi = G["fold"] == f
            acc = self.acc_fold(f)
            self.cache[ck] = (onehot(G["A"][:, fi]) * (1.0 + TIE_EPS * acc[:, None, None]), G["key"][fi], self.w[fi])
        return self.cache[ck]

    def fit_err_plurality(self, f, members, C=None):
        C, kf, wf = C or self.contrib_fold(f)
        if not members:
            return 1.0
        s = C[list(members)].sum(0)
        wrong = (np.argmax(s, -1) != kf) | (s.max(-1) <= 0)
        return float((wrong * wf).sum() / max(wf.sum(), 1e-12))

    def top_n(self, f, n, pool=None):
        acc = self.acc_fold(f)
        pool = list(range(len(acc))) if pool is None else pool
        return sorted(pool, key=lambda i: -acc[i])[:n]

    def greedy(self, f, n, pool=None):
        pool = list(range(G["A"].shape[0])) if pool is None else pool
        chosen = []
        for _ in range(n):
            best = min((j for j in pool if j not in chosen), key=lambda j: self.fit_err_plurality(f, chosen + [j]))
            chosen.append(best)
        return chosen

    def cap(self, f, n):
        acc = self.acc_fold(f)
        best_per_dev = {}
        for i in np.argsort(-acc):
            best_per_dev.setdefault(G["dev"][i], i)
        return sorted(best_per_dev.values(), key=lambda i: -acc[i])[:n]

    def greedy_mi(self, f, n):
        fi = G["fold"] == f
        A, k, w = G["A"][:, fi].astype(np.int64) + 1, G["key"][fi], self.w[fi]
        code = np.zeros(A.shape[1], dtype=np.int64)
        chosen = []

        def mi(c):
            joint = collections.defaultdict(float)
            for cc, kk, ww in zip(c, k, w):
                joint[(cc, kk)] += ww
            tot = sum(joint.values())
            pc, pk = collections.defaultdict(float), collections.defaultdict(float)
            for (cc, kk), v in joint.items():
                pc[cc] += v
                pk[kk] += v
            return sum(v / tot * np.log((v / tot) / (pc[cc] / tot * pk[kk] / tot)) for (cc, kk), v in joint.items() if v > 0)
        for _ in range(n):
            best = max((j for j in range(A.shape[0]) if j not in chosen), key=lambda j: mi(code * 5 + A[j]))
            chosen.append(best)
            code = code * 5 + A[best]
        return chosen

    def robust(self, f, n, lam, scen):
        """scen: list of sets of model indices withdrawn together (training scenarios)."""
        C = self.contrib_fold(f)

        def obj(P):
            full = self.fit_err_plurality(f, P, C)
            # a scenario that empties the portfolio leaves the system guessing (error 0.75): penalised, not skipped
            worst = max((self.fit_err_plurality(f, [i for i in P if i not in W], C) if any(i not in W for i in P)
                         else 0.75) for W in scen) if scen else full
            return (1 - lam) * full + lam * worst
        pool = list(range(G["A"].shape[0]))
        P = []
        for _ in range(n):
            P.append(min((j for j in pool if j not in P), key=lambda j: obj(P + [j])))
        cur = obj(P)
        improved = True
        while improved:
            improved = False
            for a_i in range(n):
                for j in pool:
                    if j in P:
                        continue
                    Q = P.copy()
                    Q[a_i] = j
                    v = obj(Q)
                    if v < cur - 1e-12:
                        P, cur, improved = Q, v, True
        return P


def scenario_groups():
    groups = {"dev:" + G["dev_names"][d]: set(np.where(G["dev"] == d)[0]) for d in range(len(G["dev_names"]))}
    groups["api-only"] = set(np.where(G["api"])[0])
    return groups


def withdraw(sets, W):
    return tuple([i for i in s if i not in W] for s in sets)


# ================================================================== one full analysis for given item weights
def analysis(w, with_jackknife_items=False):
    p = Pipe(w)
    groups = scenario_groups()
    res = {}
    # ---------- H2
    for n in (3, 5, 7, 9):
        sets = (p.top_n(0, n), p.top_n(1, n))
        present = [g for g, W in groups.items() if any(i in W for s in sets for i in s)]
        for rule in RULES:
            for mask_name, mask in (("all", None), ("eng", G["eng"])):
                e_full = p.mean(p.err(rule, sets), mask)
                wc, empt = -1.0, []
                for g in present:
                    ws = withdraw(sets, groups[g])
                    if not ws[0] or not ws[1]:
                        empt.append(g)
                        continue
                    wc = max(wc, p.mean(p.err(rule, ws), mask))
                res[f"H2/{mask_name}/n{n}/{rule}/full"] = e_full
                res[f"H2/{mask_name}/n{n}/{rule}/worst"] = wc
        res[f"H2/n{n}/emptying"] = ";".join(empt)
    # ---------- H3 (n=5, plurality, lambda=0.5)
    scen_all = list(groups.values())
    port = {"cap": (p.cap(0, 5), p.cap(1, 5)), "greedy": (p.greedy(0, 5), p.greedy(1, 5)),
            "mi": (p.greedy_mi(0, 5), p.greedy_mi(1, 5)),
            "robust": (p.robust(0, 5, 0.5, scen_all), p.robust(1, 5, 0.5, scen_all))}
    for k, sets in port.items():
        res[f"H3/{k}/full"] = p.mean(p.err("plurality", sets))
    devs_in = sorted({G["dev"][i] for sets in port.values() for s in sets for i in s})
    held = collections.defaultdict(list)
    for d in devs_in:
        W = groups["dev:" + G["dev_names"][d]]
        tr = [S for S in scen_all if S != W]
        rob = (p.robust(0, 5, 0.5, tr), p.robust(1, 5, 0.5, tr))
        for k, sets in [("robust", rob)] + [(k, port[k]) for k in ("cap", "greedy", "mi")]:
            ws = withdraw(sets, W)
            e = p.mean(p.err("plurality", ws)) if (ws[0] and ws[1]) else 0.75
            held[k].append(e)
    for k, v in held.items():
        res[f"H3/{k}/heldout_worst"] = float(max(v))
        res[f"H3/{k}/heldout_mean"] = float(np.mean(v))
    # ---------- descriptives on the primary portfolio (accuracy-greedy n=5, plurality)
    prim = port["greedy"]
    e0 = p.err("plurality", prim)
    res["D/prim/full"] = p.mean(e0)
    present = [g for g, W in groups.items() if any(i in W for s in prim for i in s)]
    for g in present:
        ws = withdraw(prim, groups[g])
        if ws[0] and ws[1]:
            res[f"D/prim/dE/{g}"] = p.mean(p.err("plurality", ws)) - res["D/prim/full"]
    for g1, g2 in itertools.combinations(present, 2):
        ws1, ws2 = withdraw(prim, groups[g1]), withdraw(prim, groups[g2])
        ws12 = withdraw(prim, groups[g1] | groups[g2])
        if all(x[0] and x[1] for x in (ws1, ws2, ws12)):
            res[f"D/prim/JI/{g1}+{g2}"] = (p.mean(p.err("plurality", ws12)) - p.mean(p.err("plurality", ws1))
                                          - p.mean(p.err("plurality", ws2)) + res["D/prim/full"])
    # sovereignty comparison
    open_pool = [i for i in range(len(G["models"])) if not G["api"][i]]
    p_open = (p.greedy(0, 5, open_pool), p.greedy(1, 5, open_pool))
    res["S/prim_api_withdrawn"] = p.mean(p.err("plurality", withdraw(prim, groups["api-only"]))) \
        if all(withdraw(prim, groups["api-only"])) else 0.75
    res["S/open5"] = p.mean(p.err("plurality", p_open))
    allm = tuple([list(range(len(G["models"])))] * 2)
    res["S/full_panel"] = p.mean(p.err("plurality", allm))
    res["S/full_panel_api_withdrawn"] = p.mean(p.err("plurality", (open_pool, open_pool)))
    res["S/best_single_open"] = p.mean(p.err("best", (open_pool, open_pool)))
    res["S/best_single_any"] = p.mean(p.err("best", allm))
    small = [i for i in open_pool if G["params"][i] <= 100]
    cons = []
    for f in (0, 1):
        chosen = []
        for _ in range(5):
            cand = [j for j in small if j not in chosen and G["params"][chosen + [j]].sum() <= 100]
            if not cand:
                break
            chosen.append(min(cand, key=lambda j: p.fit_err_plurality(f, chosen + [j])))
        cons.append(chosen)
    res["S/open_compute100B"] = p.mean(p.err("plurality", tuple(cons)))
    res["S/robust_full"] = res["H3/robust/full"]
    rw = withdraw(port["robust"], groups["api-only"])
    res["S/robust_api_withdrawn"] = p.mean(p.err("plurality", rw)) if (rw[0] and rw[1]) else 0.75
    for k in ("S/open5", "S/prim_api_withdrawn", "S/best_single_open", "S/full_panel_api_withdrawn"):
        res[k + "/eng"] = None
    res["S/open5/eng"] = p.mean(p.err("plurality", p_open), G["eng"])
    res["S/prim_full/eng"] = p.mean(e0, G["eng"])
    res["S/prim_api_withdrawn/eng"] = p.mean(p.err("plurality", withdraw(prim, groups["api-only"])), G["eng"]) \
        if all(withdraw(prim, groups["api-only"])) else 0.75
    res["S/best_single_open/eng"] = p.mean(p.err("best", (open_pool, open_pool)), G["eng"])
    res["S/full_panel_api_withdrawn/eng"] = p.mean(p.err("plurality", (open_pool, open_pool)), G["eng"])
    if w is G["ones"]:
        res["_portfolios"] = {k: [[G["models"][i] for i in s] for s in v] for k, v in
                              list(port.items()) + [("open5", p_open), ("open_compute100B", tuple(cons))]}
    return res


# ================================================================== H1 (no subject bootstrap)
def h1(rng, k_null=200, subset=None):
    import cycle8_s0_synthetic_validation as v1
    import cycle8_s0b_synthetic_validation as v2
    sel = np.arange(G["A"].shape[0]) if subset is None else np.asarray(subset)
    A, key = G["A"][sel], G["key"]
    DEVS, PCLS = G["dev"][sel], G["pclass"][sel]
    M = A.shape[0]
    iu = np.triu_indices(M, 1)
    same = DEVS[iu[0]] == DEVS[iu[1]]
    devs_multi = [d for d in range(len(G["dev_names"])) if (DEVS == d).sum() >= 2]

    def stat(ans, p, wrong, drop_dev=None):
        n = ans.shape[1]
        S = np.zeros((M, M))
        for o in range(M_OPT):
            ind = ((ans == o) & (key[None, :] != o)).astype(float)
            S += ind @ ind.T
        q2 = (wrong ** 2).sum(1)
        E = ((1 - p) * q2[None, :]) @ (1 - p).T
        x = ((S - E) / n)[iu]
        acc = (ans == key[None, :]).mean(1)
        ai, aj = acc[iu[0]], acc[iu[1]]
        ci, cj = PCLS[iu[0]], PCLS[iu[1]]
        keep = np.ones(len(x), dtype=bool)
        if drop_dev is not None:
            keep = (DEVS[iu[0]] != drop_dev) & (DEVS[iu[1]] != drop_dev)
        btw = (~same) & keep
        per_dev, n_pairs = [], 0
        for d in devs_multi:
            if d == drop_dev:
                continue
            ex = []
            for q in np.where(same & (DEVS[iu[0]] == d))[0]:
                cls_ok = ((ci == ci[q]) & (cj == cj[q])) | ((ci == cj[q]) & (cj == ci[q]))
                t = 0.03
                while t <= 0.12 + 1e-9:
                    m = btw & cls_ok & (((np.abs(ai - ai[q]) <= t) & (np.abs(aj - aj[q]) <= t)) |
                                        ((np.abs(ai - aj[q]) <= t) & (np.abs(aj - ai[q]) <= t)))
                    if m.sum() >= 5:
                        ex.append(x[q] - x[m].mean())
                        break
                    t *= 2
            if ex:
                per_dev.append(np.mean(ex))
                n_pairs += len(ex)
        return float(np.mean(per_dev)), len(per_dev), n_pairs
    c = (A == key[None, :]).astype(float)
    fit = v2.fit_1plg(c)
    wrong = v1.pooled_wrong(A, key)
    t_obs, n_dev, n_pairs = stat(A, fit[3], wrong)
    null = []
    for _ in range(k_null):
        a0 = v1.simulate_n2(rng, fit[3], wrong, key)
        f0 = v2.fit_1plg((a0 == key[None, :]).astype(float), init=fit[:3], iters=8)
        null.append(stat(a0, f0[3], v1.pooled_wrong(a0, key))[0])
    null = np.array(null)
    pval = float((1 + (null >= t_obs).sum()) / (1 + k_null))
    jk = [stat(A, fit[3], wrong, drop_dev=d)[0] for d in devs_multi]
    k = len(jk)
    se = float(np.sqrt((k - 1) / k * np.sum((np.array(jk) - np.mean(jk)) ** 2)))
    lo, hi = t_obs - 1.645 * se, t_obs + 1.645 * se
    if pval < 0.05 and t_obs > MDE_H1:
        verdict = "SUPPORTED"
    elif hi < MDE_H1:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"
    # strict-panel sensitivity: same statistic restricted to strict-panel models is reported only via note
    return dict(T=t_obs, p=pval, null_mean=float(null.mean()), null_sd=float(null.std()), jk_se=se, ci90=[lo, hi],
                developers_used=n_dev, within_pairs_used=n_pairs, verdict=verdict)


# ================================================================== driver
def boot_task(b):
    rng = np.random.default_rng([SEED, 1000 + b])
    n_s = G["subj"].max() + 1
    counts = np.bincount(rng.integers(0, n_s, n_s), minlength=n_s).astype(float)
    return b, analysis(counts[G["subj"]])


def init_worker(g):
    G.update(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(HERE), "data"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--k-null", type=int, default=200)
    args = ap.parse_args()
    suffix = "_synthetic" if args.synthetic else ""
    g = load_synthetic() if args.synthetic else load_real(args.data)
    rng = np.random.default_rng(SEED)
    fold = np.zeros(len(g["key"]), dtype=int)
    for s in np.unique(g["subj"]):
        ids = np.where(g["subj"] == s)[0]
        fold[rng.permutation(ids)[: len(ids) // 2]] = 1
    g["fold"] = fold
    g["ones"] = np.ones(len(g["key"]))
    G.update(g)
    t0 = time.time()
    print(f"{len(g['models'])} models, {len(g['key'])} items, {len(g['dev_names'])} developers, "
          f"{int(g['api'].sum())} API-only, prompt classes {g['pclass_names']}", flush=True)
    point = analysis(G["ones"])
    print(f"point estimates done in {time.time() - t0:.0f} s", flush=True)
    h1res = h1(np.random.default_rng([SEED, 1]), args.k_null)
    print(f"H1 done in {time.time() - t0:.0f} s: {h1res}", flush=True)
    # pre-registered sensitivity analyses of H1 (protocol Sections 8 and 18)
    acc_all = (G["A"] == G["key"][None, :]).mean(1)
    largest = np.bincount(G["dev"]).argmax()
    sens = {"strict_panel": [i for i, m in enumerate(G["models"]) if m in G["strict"]],
            "without_largest_developer": [i for i in range(len(G["models"])) if G["dev"][i] != largest],
            "without_accuracy_below_40pct": [i for i in range(len(G["models"])) if acc_all[i] >= 0.40],
            "open_weight_only": [i for i in range(len(G["models"])) if not G["api"][i]]}
    h1res["sensitivity"] = {}
    for k, sub in sens.items():
        r = h1(np.random.default_rng([SEED, 2]), max(50, args.k_null // 2), subset=sub)   # half the panels, to save time
        h1res["sensitivity"][k] = {kk: r[kk] for kk in ("T", "p", "ci90", "developers_used", "verdict")} | {"models": len(sub)}
    h1res["sensitivity_note"] = f"largest developer: {G['dev_names'][largest]}"
    print(f"H1 sensitivity done in {time.time() - t0:.0f} s", flush=True)
    boot_path = os.path.join(HERE, f"cycle8_s1_bootstrap{suffix}.jsonl")
    done = {}
    if os.path.exists(boot_path):
        for line in open(boot_path):
            r = json.loads(line)
            done[r["b"]] = r["res"]
    todo = [b for b in range(args.boot) if b not in done]
    with open(boot_path, "a") as f, Pool(args.workers, initializer=init_worker, initargs=(g,)) as pool:
        for i, (b, r) in enumerate(pool.imap_unordered(boot_task, todo), 1):
            f.write(json.dumps({"b": b, "res": r}) + "\n")
            f.flush()
            done[b] = r
            if i % 10 == 0 or i == len(todo):
                print(f"  bootstrap {i}/{len(todo)}, {time.time() - t0:.0f} s", flush=True)
    boots = [done[b] for b in range(args.boot)]
    out = {"point": point, "H1": h1res, "n_boot": len(boots)}

    def ci(keyf, a=0.05):
        v = np.array([keyf(r) for r in boots], dtype=float)
        v = v[~np.isnan(v)]
        return [float(np.quantile(v, a / 2)), float(np.quantile(v, 1 - a / 2))] if len(v) else [np.nan, np.nan]
    # H2 decision per size (all items)
    h2 = {}
    for n in (3, 5, 7, 9):
        full = {r: point[f"H2/all/n{n}/{r}/full"] for r in RULES}
        worst = {r: point[f"H2/all/n{n}/{r}/worst"] for r in RULES}
        wf, ww = min(full, key=full.get), min(worst, key=worst.get)
        flip = False
        if wf != ww:
            d_full = ci(lambda r_, a=wf, b=ww: r_[f"H2/all/n{n}/{b}/full"] - r_[f"H2/all/n{n}/{a}/full"])
            d_worst = ci(lambda r_, a=wf, b=ww: r_[f"H2/all/n{n}/{a}/worst"] - r_[f"H2/all/n{n}/{b}/worst"])
            flip = (full[ww] - full[wf] >= 0.01 and d_full[0] > 0 and worst[wf] - worst[ww] >= 0.01 and d_worst[0] > 0)
        else:
            d_full = d_worst = [0.0, 0.0]
        same_or_close = (wf == ww) or (worst[wf] - worst[ww] <= 0.005)
        h2[n] = dict(full_winner=wf, worst_winner=ww, flip=bool(flip), close=bool(same_or_close),
                     full=full, worst=worst, ci_full_gap=d_full, ci_worst_gap=d_worst)
    n_flip = sum(v["flip"] for v in h2.values())
    h2_verdict = "SUPPORTED" if n_flip >= 2 else ("REFUTED" if all(v["close"] for v in h2.values()) else "INCONCLUSIVE")
    out["H2"] = {"by_n": h2, "verdict": h2_verdict}
    # H3 decision
    gain = point["H3/cap/heldout_worst"] - point["H3/robust/heldout_worst"]
    gain_ci = ci(lambda r_: r_["H3/cap/heldout_worst"] - r_["H3/robust/heldout_worst"])
    cost = point["H3/robust/full"] - point["H3/cap/full"]
    if gain >= 0.01 and gain_ci[0] > 0 and cost <= 0.01:
        h3_verdict = "SUPPORTED"
    elif gain <= 0.005:
        h3_verdict = "REFUTED"
    else:
        h3_verdict = "INCONCLUSIVE"
    out["H3"] = {"gain_vs_cap": gain, "gain_ci": gain_ci, "full_cost_vs_cap": cost, "verdict": h3_verdict,
                 **{k: v for k, v in point.items() if k.startswith("H3/")}}
    # intervals for descriptives
    desc = {}
    for k, v in point.items():
        if k.startswith(("D/", "S/")) and isinstance(v, float):
            if "/JI/" in k:
                vals = np.array([r.get(k, np.nan) for r in boots], dtype=float)
                vals = vals[~np.isnan(vals)]
                Gs = G["subj"].max() + 1
                adj = v + np.sqrt(Gs / (Gs - 1)) * (vals - vals.mean()) if len(vals) else vals
                desc[k] = [v] + ([float(np.quantile(adj, 0.025)), float(np.quantile(adj, 0.975))] if len(vals) else [None, None])
            else:
                desc[k] = [v] + ci(lambda r_, kk=k: r_.get(kk, np.nan))
    out["descriptive"] = desc
    out["seconds"] = round(time.time() - t0)
    json.dump(out, open(os.path.join(HERE, f"cycle8_s1_analysis{suffix}.json"), "w"), indent=1, default=str)
    lines = [f"Cycle 8 S1 primary analysis{' (SYNTHETIC CODE TEST, not a result)' if args.synthetic else ''}",
             f"models {len(G['models'])}, items {len(G['key'])}, bootstrap resamples {len(boots)}",
             f"\nH1: T = {h1res['T']:.5f}, p = {h1res['p']:.3f}, jackknife 90% CI [{h1res['ci90'][0]:.5f}, {h1res['ci90'][1]:.5f}], "
             f"MDE {MDE_H1}, developers used {h1res['developers_used']}, within pairs {h1res['within_pairs_used']} -> {h1res['verdict']}",
             "H1 sensitivity: " + "; ".join(f"{k} ({v['models']} models): T {v['T']:.5f}, p {v['p']:.3f} -> {v['verdict']}"
                                           for k, v in h1res["sensitivity"].items()) + f" ({h1res['sensitivity_note']})"]
    for n, v in h2.items():
        lines.append(f"H2 n={n}: full winner {v['full_winner']}, worst-case winner {v['worst_winner']}, flip {v['flip']}; "
                     + "; ".join(f"{r} {100*v['full'][r]:.2f}/{100*v['worst'][r]:.2f}" for r in RULES))
    lines.append(f"H2 -> {h2_verdict}")
    lines.append(f"H3: held-out worst-case error robust {100*point['H3/robust/heldout_worst']:.2f}, cap {100*point['H3/cap/heldout_worst']:.2f}, "
                 f"greedy {100*point['H3/greedy/heldout_worst']:.2f}, MI {100*point['H3/mi/heldout_worst']:.2f}; "
                 f"gain vs cap {100*gain:.2f} pp, CI [{100*gain_ci[0]:.2f}, {100*gain_ci[1]:.2f}]; full-availability cost vs cap {100*cost:.2f} pp -> {h3_verdict}")
    lines.append("\nDescriptive (error, %; 95% CI):")
    for k, v in sorted(desc.items()):
        lines.append(f"  {k}: {100*v[0]:.2f} [{100*v[1]:.2f}, {100*v[2]:.2f}]" if v[1] is not None else f"  {k}: {100*v[0]:.2f}")
    lines.append(f"\nportfolios (fold 0 / fold 1): {json.dumps(point.get('_portfolios'))}")
    lines.append(f"seconds {out['seconds']}")
    text = "\n".join(lines)
    open(os.path.join(HERE, f"cycle8_s1_analysis{suffix}.out"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
