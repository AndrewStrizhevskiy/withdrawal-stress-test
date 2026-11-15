"""Cycle 8, Russian-model run (protocol Section 26): extract HELM's exact MMLU prompts for every item.

Standard library only (server). The prompts come from one modal-prompt model's scenario_state.json files; the
prompt SHA-1 is checked against the hash recorded for that model in Step 0. Downloaded JSON is untrusted data:
parsed with json.loads, nothing executed.

    python3 checks/cycle8_ru_prompts.py --model qwen_qwen2.5-7b-instruct-turbo --release v1.9.0 \
        --parsed data/helm_mmlu_parsed.tsv.gz --out data/helm_mmlu_prompts.jsonl.gz
"""
import argparse
import gzip
import hashlib
import json
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import cycle8_s1_step0_download as dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = [p for p in dl.list_prefixes(dl.ROOT + args.release + "/")
            if p.split("/")[-2].startswith("mmlu:subject=") and "method=multiple_choice_joint" in p
            and f"model={args.model}" in p]
    assert len(runs) == dl.N_SUBJECTS, len(runs)
    recorded = {}
    with gzip.open(args.parsed, "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            m, v, s, iid, ans, key, ph, comp = line.rstrip("\n").split("\t")
            if m == args.model:
                recorded[(s, iid)] = ph

    def work(p):
        name = p.split("/")[-2]
        subj = dict(kv.split("=", 1) for kv in name.split(":", 1)[1].split(","))["subject"]
        d = json.loads(dl.get(dl.OBJ + urllib.parse.quote(p) + "scenario_state.json"))
        out = []
        for rs in d["request_states"]:
            if rs.get("train_trial_index", 0) != 0:
                continue
            inst, refs = rs["instance"], rs["instance"]["references"]
            key = [i for i, r in enumerate(refs) if "correct" in r.get("tags", [])]
            if len(key) != 1:
                continue
            text_to_idx = {r["output"]["text"]: i for i, r in enumerate(refs)}
            mapping = {L: text_to_idx[t] for L, t in (rs.get("output_mapping") or {}).items() if t in text_to_idx}
            prompt = rs["request"]["prompt"]
            out.append({"subject": subj, "instance_id": inst["id"], "prompt": prompt, "mapping": mapping, "key": key[0],
                        "sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]})
        req = d["request_states"][0]["request"]
        settings = {k: req.get(k) for k in ("temperature", "max_tokens", "stop_sequences", "num_completions")}
        return out, settings

    rows, settings = [], None
    with ThreadPoolExecutor(12) as ex:
        for out, st in ex.map(work, runs):
            rows += out
            settings = st
    mism = sum(1 for r in rows if recorded.get((r["subject"], r["instance_id"])) != r["sha1"])
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} prompts from {args.model} {args.release}; hash mismatches against Step 0: {mism}; "
          f"HELM request settings: {settings}")
    print("example prompt tail:\n" + rows[0]["prompt"][-400:])


if __name__ == "__main__":
    main()
