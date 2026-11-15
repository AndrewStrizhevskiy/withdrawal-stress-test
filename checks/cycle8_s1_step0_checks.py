"""Cycle 8, S1 Step 0 checks on the parsed HELM MMLU outputs (protocol Section 3). No hypothesis is tested here.

Inputs: data/helm_mmlu_parsed.tsv.gz and data/helm_mmlu_runs.json (from cycle8_s1_step0_download.py).
Outputs: data/helm_mmlu_panel.json (panel definition for S1) and checks/cycle8_s1_step0_checks.out.

The deduplication rule and the access classification below were written from model names and model cards
before any answer data were looked at (see protocol Section 16). Accuracy is not computed by this script.

    .venv/bin/python checks/cycle8_s1_step0_checks.py
"""
import collections
import gzip
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

# --- deduplication: one model per lineage and size class (rule in protocol Section 16) -----------------------
# key: dropped model -> kept model of the same lineage and size class, with the reason
DROP = {
    "anthropic_claude-3-5-sonnet-20240620": ("anthropic_claude-3-5-sonnet-20241022", "earlier snapshot"),
    "google_gemini-pro": ("google_gemini-1.0-pro-001", "same model, alias"),
    "google_gemini-1.5-flash-001": ("google_gemini-1.5-flash-002", "earlier snapshot"),
    "google_gemini-1.5-flash-preview-0514": ("google_gemini-1.5-flash-002", "preview snapshot"),
    "google_gemini-1.5-pro-001": ("google_gemini-1.5-pro-002", "earlier snapshot"),
    "google_gemini-1.5-pro-preview-0409": ("google_gemini-1.5-pro-002", "preview snapshot"),
    "meta_llama-3.2-11b-vision-instruct-turbo": ("meta_llama-3.1-8b-instruct-turbo", "text weights documented as those of Llama 3.1 8B [REC]"),
    "meta_llama-3.2-90b-vision-instruct-turbo": ("meta_llama-3.1-70b-instruct-turbo", "text weights documented as those of Llama 3.1 70B [REC]"),
    "openai_gpt-3.5-turbo-0613": ("openai_gpt-3.5-turbo-0125", "earlier snapshot"),
    "openai_gpt-4-1106-preview": ("openai_gpt-4-turbo-2024-04-09", "preview snapshot of GPT-4 Turbo"),
    "openai_gpt-4o-2024-05-13": ("openai_gpt-4o-2024-08-06", "earlier snapshot"),
}

# --- access: open-weight (weights published for local deployment) or API-only -------------------------------
# Open-weight entries were checked on 2026-11-14 against the Hugging Face model API (repository exists);
# the licence tag is recorded because a non-commercial or custom licence limits deployment.
OPEN = {
    "01-ai_yi-34b": "01-ai/Yi-34B (apache-2.0)", "01-ai_yi-6b": "01-ai/Yi-6B (apache-2.0)",
    "ai21_jamba-1.5-large": "ai21labs/AI21-Jamba-1.5-Large (other, gated)", "ai21_jamba-1.5-mini": "ai21labs/AI21-Jamba-1.5-Mini (other, gated)",
    "allenai_olmo-7b": "allenai/OLMo-7B (apache-2.0)", "allenai_olmo-1.7-7b": "allenai/OLMo-1.7-7B (apache-2.0)",
    "cohere_command-r": "CohereForAI/c4ai-command-r-v01 (cc-by-nc-4.0: non-commercial)",
    "cohere_command-r-plus": "CohereForAI/c4ai-command-r-plus (cc-by-nc-4.0: non-commercial)",
    "databricks_dbrx-instruct": "databricks/dbrx-instruct (released 2024; repository returned 401 on 2026-11-14, so access UNVERIFIED)",
    "deepseek-ai_deepseek-llm-67b-chat": "deepseek-ai/deepseek-llm-67b-chat (other)", "deepseek-ai_deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "google_gemma-2-27b": "google/gemma-2-27b (gemma, gated)", "google_gemma-2-9b": "google/gemma-2-9b (gemma, gated)", "google_gemma-7b": "google/gemma-7b (gemma, gated)",
    "meta_llama-2-7b": "meta-llama/Llama-2-7b-hf (llama2, gated)", "meta_llama-2-13b": "meta-llama/Llama-2-13b-hf (llama2, gated)",
    "meta_llama-2-70b": "meta-llama/Llama-2-70b-hf (llama2, gated)", "meta_llama-3-8b": "meta-llama/Meta-Llama-3-8B (llama3, gated)",
    "meta_llama-3-70b": "meta-llama/Meta-Llama-3-70B (llama3, gated)",
    "meta_llama-3.1-8b-instruct-turbo": "meta-llama/Llama-3.1-8B-Instruct (llama3.1, gated)",
    "meta_llama-3.1-70b-instruct-turbo": "meta-llama/Llama-3.1-70B-Instruct (llama3.1, gated)",
    "meta_llama-3.1-405b-instruct-turbo": "meta-llama/Llama-3.1-405B-Instruct (llama3.1, gated)",
    "meta_llama-3.3-70b-instruct-turbo": "meta-llama/Llama-3.3-70B-Instruct (llama3.3, gated)",
    "microsoft_phi-2": "microsoft/phi-2 (mit)", "microsoft_phi-3-medium-4k-instruct": "microsoft/Phi-3-medium-4k-instruct (mit)",
    "microsoft_phi-3-small-8k-instruct": "microsoft/Phi-3-small-8k-instruct (mit)",
    "mistralai_mistral-7b-v0.1": "mistralai/Mistral-7B-v0.1 (apache-2.0)", "mistralai_mistral-7b-instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3 (apache-2.0)",
    "mistralai_mistral-large-2407": "mistralai/Mistral-Large-Instruct-2407 (research licence, gated)",
    "mistralai_mixtral-8x22b": "mistralai/Mixtral-8x22B-v0.1 (apache-2.0)", "mistralai_mixtral-8x7b-32kseqlen": "mistralai/Mixtral-8x7B-v0.1 (apache-2.0)",
    "mistralai_open-mistral-nemo-2407": "mistralai/Mistral-Nemo-Instruct-2407 (apache-2.0)",
    "nvidia_nemotron-4-340b-instruct": "nvidia/Nemotron-4-340B-Instruct (other)",
    "qwen_qwen1.5-7b": "Qwen/Qwen1.5-7B", "qwen_qwen1.5-14b": "Qwen/Qwen1.5-14B", "qwen_qwen1.5-32b": "Qwen/Qwen1.5-32B",
    "qwen_qwen1.5-72b": "Qwen/Qwen1.5-72B", "qwen_qwen1.5-110b-chat": "Qwen/Qwen1.5-110B-Chat", "qwen_qwen2-72b-instruct": "Qwen/Qwen2-72B-Instruct",
    "qwen_qwen2.5-7b-instruct-turbo": "Qwen/Qwen2.5-7B-Instruct (apache-2.0)", "qwen_qwen2.5-72b-instruct-turbo": "Qwen/Qwen2.5-72B-Instruct (other)",
    "snowflake_snowflake-arctic-instruct": "Snowflake/snowflake-arctic-instruct (apache-2.0)",
}
# API-only notes for models whose developer also publishes open weights
API_NOTES = {
    "ai21_jamba-instruct": "instruction-tuned Jamba served by API; only the base Jamba-v0.1 is open [REC]",
    "mistralai_mistral-large-2402": "Mistral Large (first version), API only [REC]",
    "mistralai_mistral-small-2402": "Mistral Small (2402), API only [REC]",
    "upstage_solar-pro-241126": "API release; the open solar-pro-preview-instruct is an earlier preview [REC]",
    "01-ai_yi-large-preview": "API only [REC]",
}

ENGINEERING = ["electrical_engineering", "college_physics", "high_school_physics", "conceptual_physics",
               "college_chemistry", "high_school_chemistry"]
S0_STRUCTURE = {"models": 64, "developers_with_2plus": 13}


def main():
    meta = json.load(open(os.path.join(DATA, "helm_mmlu_runs.json")))
    rows = collections.defaultdict(dict)              # model -> (subject, id) -> (answer, key, prompt)
    with gzip.open(os.path.join(DATA, "helm_mmlu_parsed.tsv.gz"), "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            m, v, s, iid, ans, key, ph, comp = line.rstrip("\n").split("\t")
            rows[m][(s, iid)] = (int(ans), int(key), ph)
    models = sorted(rows)
    all_items = sorted(set().union(*[set(rows[m]) for m in models]))
    lines = [f"models with a complete release: {len(models)} (of {len(meta['models'])} found)",
             f"items in the union: {len(all_items)}"]
    no_release = [m for m, v in meta["models"].items() if v.get("chosen_release") is None]
    lines.append(f"models without a complete release: {no_release}")
    # coverage
    full = [m for m in models if len(rows[m]) == len(all_items)]
    lines.append(f"models covering every item: {len(full)}; incomplete: "
                 + ", ".join(f"{m} ({len(rows[m])})" for m in models if m not in full))
    # keys consistent across models
    keys = collections.defaultdict(set)
    for m in models:
        for it, (_, k, _) in rows[m].items():
            keys[it].add(k)
    lines.append(f"items whose correct key differs between models: {sum(len(v) > 1 for v in keys.values())}")
    # prompt consistency: modal prompt hash per item across models
    hashes = collections.defaultdict(collections.Counter)
    for m in models:
        for it, (_, _, ph) in rows[m].items():
            hashes[it][ph] += 1
    modal = {it: c.most_common(1)[0][0] for it, c in hashes.items()}
    share = {m: sum(rows[m][it][2] == modal[it] for it in rows[m]) / len(rows[m]) for m in models}
    consistent = [m for m in full if share[m] == 1.0]
    lines.append(f"\nprompt consistency (share of items with the modal prompt):")
    for m in models:
        rel = meta["models"][m]["chosen_release"]
        lines.append(f"  {m:48s} {rel:8s} {share[m]:.3f}")
    inval = {m: sum(a == -1 for a, _, _ in rows[m].values()) / len(rows[m]) for m in models}
    lines.append("\nshare of unparseable completions (format check, not accuracy): "
                 + ", ".join(f"{m} {inval[m]:.3f}" for m in models if inval[m] > 0.001))
    # deduplication
    dropped_present = {m: DROP[m] for m in models if m in DROP}
    lines.append(f"\ndeduplication: dropped {len(dropped_present)} of {len(models)}")
    for m, (k, why) in dropped_present.items():
        lines.append(f"  drop {m} -> keep {k} ({why})" + ("" if k in models else "  [kept model missing!]"))
    dedup_all = [m for m in models if m not in DROP]
    primary = [m for m in consistent if m not in DROP]
    def dev(m):
        return m.split("_")[0]
    def structure(ms):
        c = collections.Counter(dev(m) for m in ms)
        return {"models": len(ms), "developers": len(c), "developers_with_2plus": sum(v >= 2 for v in c.values()),
                "api_only": sum(m not in OPEN for m in ms), "by_developer": dict(sorted(c.items()))}
    st_all, st_primary = structure(dedup_all), structure(primary)
    lines.append(f"\ndeduplicated panel, all prompts: {json.dumps({k: v for k, v in st_all.items() if k != 'by_developer'})}")
    lines.append(f"PRIMARY panel (deduplicated, modal prompt on every item, full coverage): "
                 f"{json.dumps({k: v for k, v in st_primary.items() if k != 'by_developer'})}")
    lines.append(f"  by developer: {st_primary['by_developer']}")
    unclassified = [m for m in dedup_all if m not in OPEN and m not in API_NOTES and dev(m) not in
                    ("anthropic", "openai", "amazon", "writer") and not m.startswith("google_gemini") and not m.startswith("google_text")]
    lines.append(f"API-only models with no note (should be empty): {unclassified}")
    # structure test against S0 (Amendment A1: re-run S0 if a difference above 20%)
    d_models = abs(st_primary["models"] - S0_STRUCTURE["models"]) / S0_STRUCTURE["models"]
    d_dev = abs(st_primary["developers_with_2plus"] - S0_STRUCTURE["developers_with_2plus"]) / S0_STRUCTURE["developers_with_2plus"]
    rerun = d_models > 0.20 or d_dev > 0.20
    lines.append(f"\ndifference from the S0 structure: models {100*d_models:.0f}%, developers with >=2 models {100*d_dev:.0f}% "
                 f"-> S0 re-run needed: {'YES' if rerun else 'no'}")
    eng = [it for it in all_items if it[0] in ENGINEERING]
    lines.append(f"engineering-science subset: {len(eng)} items in {len(ENGINEERING)} subjects "
                 + str(dict(collections.Counter(it[0] for it in eng))))
    panel = {"primary": primary, "dedup_all_prompts": dedup_all, "dropped": dropped_present,
             "open_weight": {m: OPEN[m] for m in dedup_all if m in OPEN},
             "api_only": [m for m in dedup_all if m not in OPEN], "api_notes": API_NOTES,
             "engineering_subjects": ENGINEERING, "n_items": len(all_items),
             "releases": {m: meta["models"][m]["chosen_release"] for m in models},
             "prompt_share": share, "structure_primary": st_primary, "s0_rerun_needed": rerun}
    json.dump(panel, open(os.path.join(DATA, "helm_mmlu_panel.json"), "w"), indent=1)
    text = "Cycle 8 S1 Step 0 checks (no hypothesis tested; accuracy not computed)\n" + "\n".join(lines)
    open(os.path.join(HERE, "cycle8_s1_step0_checks.out"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
