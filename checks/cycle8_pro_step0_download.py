"""Cycle 8, MMLU-Pro replication Step 0: download and parse public HELM Capabilities MMLU-Pro outputs (protocol Section 22).

Standard library only (runs on the department server). Downloaded JSON is untrusted data: parsed with json.loads,
only a few fields kept, nothing executed. One run per model (the latest release that has one).

Output (tab-separated, gzip): model, release, instance_id, answer (option index, -1 if unparseable), key, n_options,
parse (1 = "answer is X", 2 = "(X)", 0 = none), helm_correct (HELM chain-of-thought correctness, "" if absent),
tail (last 40 characters of the completion, for audit).

    python3 cycle8_pro_step0_download.py --out helm_mmlu_pro_parsed.tsv.gz --meta helm_mmlu_pro_runs.json --threads 12
"""
import argparse
import gzip
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "Mozilla/5.0 (research-reader)"}
LIST = "https://storage.googleapis.com/storage/v1/b/crfm-helm-public/o?"
OBJ = "https://storage.googleapis.com/crfm-helm-public/"
ROOT = "capabilities/benchmark_output/runs/"
RE1 = re.compile(r"(?i:answer is)\s*\(?\s*([A-J])\b")
RE2 = re.compile(r"\(([A-J])\)")


def get(url, tries=4):
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read()
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(3 * (t + 1))


def list_prefixes(prefix):
    out, tok = [], None
    while True:
        q = {"prefix": prefix, "delimiter": "/", "maxResults": "1000"}
        if tok:
            q["pageToken"] = tok
        d = json.loads(get(LIST + urllib.parse.urlencode(q)))
        out += d.get("prefixes", [])
        tok = d.get("nextPageToken")
        if not tok:
            return out


def release_key(v):
    return tuple(int(x) for x in v.strip("v").split("."))


def parse_letter(text):
    m = RE1.findall(text)
    if m:
        return m[-1], 1
    m = RE2.findall(text)
    if m:
        return m[-1], 2
    return None, 0


def parse_run(prefix):
    base = OBJ + urllib.parse.quote(prefix)
    try:
        ss = json.loads(get(base + "scenario_state.json"))
    except Exception as e:                       # missing file: the model is reported and skipped
        return None, {"error": f"scenario_state.json: {e}"}
    try:
        pis = json.loads(get(base + "per_instance_stats.json"))
    except Exception as e:
        pis, pis_error = [], f"per_instance_stats.json: {e}"
    else:
        pis_error = None
    helm, stat_names = {}, set()
    for e in pis:
        if e.get("train_trial_index", 0) != 0:
            continue
        for s in e.get("stats", []):
            nm = s.get("name", {}).get("name", "")
            if "chain_of_thought_correct" in nm and not s.get("name", {}).get("perturbation"):
                helm[e["instance_id"]] = s.get("mean")
                stat_names.add(nm)
    ad = ss["adapter_spec"]
    settings = {k: ad.get(k) for k in ("method", "temperature", "max_tokens", "num_outputs", "max_eval_instances")}
    settings["prompt_parts_sha1"] = hashlib.sha1(json.dumps(
        [ad.get(k) for k in ("global_prefix", "global_suffix", "instructions", "input_prefix", "input_suffix",
                             "reference_prefix", "reference_suffix", "output_prefix", "output_suffix")]).encode()).hexdigest()[:12]
    rows, n_comp = [], set()
    for rs in ss["request_states"]:
        if rs.get("train_trial_index", 0) != 0:
            continue
        inst = rs["instance"]
        refs = inst["references"]
        key = [i for i, r in enumerate(refs) if "correct" in r.get("tags", [])]
        if len(key) != 1:
            continue
        comps = (rs.get("result") or {}).get("completions") or []
        n_comp.add(len(comps))
        text = comps[0]["text"] if comps else ""
        letter, how = parse_letter(text)
        mapping = rs.get("output_mapping") or {}
        text_to_idx = {}
        for i, r in enumerate(refs):
            text_to_idx.setdefault(r["output"]["text"], i)
        ans = text_to_idx.get(mapping.get(letter), -1) if letter else -1
        hc = helm.get(inst["id"])
        tail = text.strip()[-40:].replace("\t", " ").replace("\n", " ")
        rows.append((inst["id"], ans, key[0], len(refs), how, "" if hc is None else hc, tail))
    settings["completions_per_item"] = sorted(n_comp)
    settings["helm_stat_names"] = sorted(stat_names)
    if pis_error:
        settings["warning"] = pis_error
    return rows, settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    t0 = time.time()
    releases = sorted({p.split("/")[-2] for p in list_prefixes(ROOT)}, key=release_key)
    runs = {}
    for v in releases:
        for p in list_prefixes(ROOT + v + "/"):
            name = p.split("/")[-2]
            if not name.startswith("mmlu_pro:"):
                continue
            model = name.split("model=", 1)[1]
            runs.setdefault(model, []).append((v, name, p))
    chosen = {m: max(r, key=lambda x: release_key(x[0])) for m, r in runs.items()}
    print(f"{len(runs)} models; listing took {time.time() - t0:.0f} s", flush=True)
    meta = {"releases": releases, "models": {m: {"runs": [[v, n] for v, n, _ in r]} for m, r in sorted(runs.items())}}

    def work(item):
        m, (v, name, p) = item
        rows, settings = parse_run(p)
        return m, v, name, rows or [], settings

    with gzip.open(args.out, "wt", encoding="utf-8") as f, ThreadPoolExecutor(args.threads) as ex:
        f.write("model\trelease\tinstance_id\tanswer\tkey\tn_options\tparse\thelm_correct\ttail\n")
        for i, (m, v, name, rows, settings) in enumerate(ex.map(work, sorted(chosen.items())), 1):
            for r in rows:
                f.write(f"{m}\t{v}\t" + "\t".join(str(x) for x in r) + "\n")
            meta["models"][m].update({"chosen_release": v, "run_name": name, "items_parsed": len(rows), **settings})
            print(f"  {i}/{len(chosen)} {m}: {len(rows)} items, {time.time() - t0:.0f} s", flush=True)
    meta["seconds"] = round(time.time() - t0)
    json.dump(meta, open(args.meta, "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
