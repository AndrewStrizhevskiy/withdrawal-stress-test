"""Cycle 8, MMLU-Pro replication Step 0 checks (protocol Section 22). Standard library only (server).

Item coverage, key consistency, parsing agreement with HELM's own correctness, adapter-setting classes,
deduplication and access classification (Hugging Face model API). Writes the panel file and a report.
No hypothesis statistic is computed here.

    python3 cycle8_pro_step0_checks.py --data data --out checks/cycle8_pro_step0_checks.out
"""
import argparse
import collections
import gzip
import json
import os
import time
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (research-reader)"}

# --- deduplication (Section 16 rules, plus the reasoning-budget rule of Section 22) -----------------------------
DROP = {
    "anthropic_claude-3-5-sonnet-20240620": ("anthropic_claude-3-5-sonnet-20241022", "earlier snapshot"),
    "openai_gpt-4o-2024-11-20": ("openai_gpt-4o-2024-08-06", "later snapshot has no scenario_state.json in HELM; "
                                 "the latest snapshot with data is kept"),
    "anthropic_claude-opus-4-20250514-thinking-10k": ("anthropic_claude-opus-4-20250514", "same weights, reasoning budget"),
    "anthropic_claude-sonnet-4-20250514-thinking-10k": ("anthropic_claude-sonnet-4-20250514", "same weights, reasoning budget"),
}

# --- open-weight candidates: HELM model -> Hugging Face repository (verified below against the HF model API) -----
OPEN = {
    "allenai_olmo-2-0325-32b-instruct": "allenai/OLMo-2-0325-32B-Instruct",
    "allenai_olmo-2-1124-13b-instruct": "allenai/OLMo-2-1124-13B-Instruct",
    "allenai_olmo-2-1124-7b-instruct": "allenai/OLMo-2-1124-7B-Instruct",
    "allenai_olmoe-1b-7b-0125-instruct": "allenai/OLMoE-1B-7B-0125-Instruct",
    "deepseek-ai_deepseek-r1-0528": "deepseek-ai/DeepSeek-R1-0528",
    "deepseek-ai_deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "ibm_granite-3.3-8b-instruct": "ibm-granite/granite-3.3-8b-instruct",
    "ibm_granite-4.0-h-small": "ibm-granite/granite-4.0-h-small",
    "ibm_granite-4.0-micro": "ibm-granite/granite-4.0-micro",
    "marin-community_marin-8b-instruct": "marin-community/marin-8b-instruct",
    "meta_llama-3.1-405b-instruct-turbo": "meta-llama/Llama-3.1-405B-Instruct",
    "meta_llama-3.1-70b-instruct-turbo": "meta-llama/Llama-3.1-70B-Instruct",
    "meta_llama-3.1-8b-instruct-turbo": "meta-llama/Llama-3.1-8B-Instruct",
    "meta_llama-4-maverick-17b-128e-instruct-fp8": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "meta_llama-4-scout-17b-16e-instruct": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "mistralai_mistral-7b-instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai_mistral-large-2411": "mistralai/Mistral-Large-Instruct-2411",
    "mistralai_mistral-small-2503": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "mistralai_mixtral-8x22b-instruct-v0.1": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "mistralai_mixtral-8x7b-instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "moonshotai_kimi-k2-instruct": "moonshotai/Kimi-K2-Instruct",
    "openai_gpt-oss-120b": "openai/gpt-oss-120b",
    "openai_gpt-oss-20b": "openai/gpt-oss-20b",
    "qwen_qwen2.5-72b-instruct-turbo": "Qwen/Qwen2.5-72B-Instruct",
    "qwen_qwen2.5-7b-instruct-turbo": "Qwen/Qwen2.5-7B-Instruct",
    "qwen_qwen3-235b-a22b-fp8-tput": "Qwen/Qwen3-235B-A22B-FP8",
    "qwen_qwen3-235b-a22b-instruct-2507-fp8": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    "qwen_qwen3-next-80b-a3b-thinking": "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "writer_palmyra-fin": "Writer/Palmyra-Fin-70B-32K",
    "writer_palmyra-med": "Writer/Palmyra-Med-70B-32K",
    "zai-org_glm-4.5-air-fp8": "zai-org/GLM-4.5-Air-FP8",
}
FLAG = {"writer_palmyra-fin": "open release exists; HELM may have queried a different hosted version",
        "writer_palmyra-med": "open release exists; HELM may have queried a different hosted version"}


def hf_check(repo):
    try:
        with urllib.request.urlopen(urllib.request.Request("https://huggingface.co/api/models/" + repo, headers=UA),
                                    timeout=30) as r:
            d = json.loads(r.read())
        lic = [t.split(":", 1)[1] for t in d.get("tags", []) if t.startswith("license:")]
        return {"status": "ok", "license": lic[0] if lic else None, "gated": d.get("gated")}
    except Exception as e:
        return {"status": f"error: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.data, "helm_mmlu_pro_runs.json")))
    rows = collections.defaultdict(dict)
    with gzip.open(os.path.join(args.data, "helm_mmlu_pro_parsed.tsv.gz"), "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            m, v, iid, ans, key, nopt, how, hc, tail = line.rstrip("\n").split("\t")
            rows[m][iid] = (int(ans), int(key), int(nopt), int(how), None if hc == "" else float(hc))
    L = []
    models = sorted(meta["models"])
    L.append(f"MMLU-Pro Step 0 checks ({time.strftime('%Y-%m-%d')})\nmodels listed: {len(models)}; with parsed rows: {len(rows)}")
    # --- keys consistent across models
    keys = collections.defaultdict(set)
    for m in rows:
        for iid, r in rows[m].items():
            keys[iid].add((r[1], r[2]))
    bad = [i for i, s in keys.items() if len(s) > 1]
    L.append(f"item ids seen: {len(keys)}; ids with inconsistent key or option count across models: {len(bad)}")
    # --- parsing agreement with HELM
    L.append("\nper model: items, invalid share, parse rule 1/2/0 shares, agreement with HELM correctness (items compared)")
    agree = {}
    for m in models:
        r = rows.get(m, {})
        if not r:
            L.append(f"  {m}: NO DATA ({meta['models'][m].get('error')})")
            continue
        hows = collections.Counter(x[3] for x in r.values())
        comp = [(x[0] == x[1]) == (x[4] >= 0.5) for x in r.values() if x[4] is not None]
        agree[m] = sum(comp) / len(comp) if comp else None
        L.append(f"  {m}: {len(r)}, invalid {sum(x[0] < 0 for x in r.values()) / len(r):.3f}, "
                 f"rules {hows[1] / len(r):.3f}/{hows[2] / len(r):.3f}/{hows[0] / len(r):.3f}, "
                 f"agreement {'n/a' if agree[m] is None else f'{agree[m]:.3f}'} ({len(comp)})")
    # --- deduplication
    dedup = [m for m in models if m in rows and rows[m] and m not in DROP]
    missing_keep = [k for k, (keep, _) in DROP.items() if keep not in dedup]
    L.append(f"\ndeduplication: {len(DROP)} drops -> {len(dedup)} models; kept models missing: {missing_keep}")
    for k, (keep, why) in DROP.items():
        L.append(f"  drop {k} -> keep {keep}: {why}")
    # --- agreement rule
    low = [m for m in dedup if agree.get(m) is None or agree[m] < 0.95]
    primary = [m for m in dedup if m not in low]
    L.append(f"agreement < 0.95 or not available (excluded from primary, kept for sensitivity): {low}")
    # --- common items
    common = set.intersection(*(set(rows[m]) for m in primary))
    L.append(f"items common to all {len(primary)} primary models: {len(common)}")
    # --- adapter-setting classes
    def cls(m):
        s = meta["models"][m]
        return f"T={s.get('temperature')},max_tokens={s.get('max_tokens')},n={s.get('num_outputs')},p={s.get('prompt_parts_sha1')}"
    classes = collections.Counter(cls(m) for m in primary)
    L.append("adapter-setting classes (primary): " + "; ".join(f"{k}: {v}" for k, v in classes.most_common()))
    modal = classes.most_common(1)[0][0]
    strict = [m for m in primary if cls(m) == modal]
    # --- access
    L.append("\naccess classification (Hugging Face model API, " + time.strftime("%Y-%m-%d") + "):")
    access = {}
    for m in dedup:
        if m in OPEN:
            c = hf_check(OPEN[m])
            access[m] = {"open": c["status"] == "ok", "repo": OPEN[m], **c, **({"flag": FLAG[m]} if m in FLAG else {})}
            L.append(f"  {m}: {OPEN[m]} -> {c}" + (f"  FLAG: {FLAG[m]}" if m in FLAG else ""))
        else:
            access[m] = {"open": False}
    unverified = [m for m in dedup if m in OPEN and not access[m]["open"]]
    open_w = [m for m in primary if access[m]["open"]]
    api = [m for m in primary if not access[m]["open"]]
    devs = collections.Counter(m.split("_")[0] for m in primary)
    L.append(f"\nunverified open-weight candidates (treated as API-only until resolved): {unverified}")
    L.append(f"primary panel: {len(primary)} models, {len(devs)} developers ({sum(v >= 2 for v in devs.values())} with >= 2), "
             f"{len(open_w)} open-weight, {len(api)} API-only; strict (modal settings) panel: {len(strict)}")
    L.append("developers: " + ", ".join(f"{d} {n}" for d, n in devs.most_common()))
    panel = {"primary": primary, "strict": strict, "dedup_all": dedup, "low_agreement": low, "open_weight": open_w,
             "api_only": api, "common_items": sorted(common), "settings_class": {m: cls(m) for m in dedup},
             "access": access, "agreement": agree, "drops": DROP}
    json.dump(panel, open(os.path.join(args.data, "helm_mmlu_pro_panel.json"), "w"), indent=1)
    text = "\n".join(L)
    open(args.out, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
