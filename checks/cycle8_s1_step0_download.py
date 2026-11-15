"""Cycle 8, S1 Step 0: download and parse public HELM MMLU per-question outputs (protocol Section 3).

Standard library only (runs on the department server). Downloaded JSON is treated as untrusted data: it is
parsed with json.loads and only a few fields are kept. Nothing is executed from it.

For every model, the latest HELM MMLU release that has all 57 subjects is used. Output (tab-separated, gzip):
model, release, subject, instance_id, answer (0-3, or -1 if the completion is not a valid option letter),
key (0-3), prompt_sha1 (first 12 hex digits), completion (first 8 characters, for audit).

    python3 cycle8_s1_step0_download.py --out helm_mmlu_parsed.tsv.gz --meta helm_mmlu_runs.json --threads 16
"""
import argparse
import gzip
import hashlib
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "Mozilla/5.0 (research-reader)"}
LIST = "https://storage.googleapis.com/storage/v1/b/crfm-helm-public/o?"
OBJ = "https://storage.googleapis.com/crfm-helm-public/"
ROOT = "mmlu/benchmark_output/runs/"
N_SUBJECTS = 57


def get(url, tries=4):
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(2 * (t + 1))


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


def parse_run(run_prefix):
    d = json.loads(get(OBJ + urllib.parse.quote(run_prefix) + "scenario_state.json"))
    rows = []
    for rs in d["request_states"]:
        if rs.get("train_trial_index", 0) != 0:
            continue
        inst = rs["instance"]
        refs = inst["references"]
        key = [i for i, r in enumerate(refs) if "correct" in r.get("tags", [])]
        if len(key) != 1:
            continue
        mapping = rs.get("output_mapping") or {}
        text_to_idx = {r["output"]["text"]: i for i, r in enumerate(refs)}
        comp = (rs["result"]["completions"][0]["text"] if rs.get("result") and rs["result"].get("completions") else "")
        letter = comp.strip()[:1]
        ans = -1
        if letter in mapping and mapping[letter] in text_to_idx:
            ans = text_to_idx[mapping[letter]]
        ph = hashlib.sha1(rs["request"]["prompt"].encode("utf-8")).hexdigest()[:12]
        rows.append((inst["id"], ans, key[0], ph, comp.strip()[:8].replace("\t", " ").replace("\n", " ")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    t0 = time.time()
    releases = sorted({p.split("/")[-2] for p in list_prefixes(ROOT)}, key=release_key)
    runs = {}                                    # model -> release -> {subject: prefix}
    for v in releases:
        for p in list_prefixes(ROOT + v + "/"):
            name = p.split("/")[-2]
            if not name.startswith("mmlu:subject=") or "method=multiple_choice_joint" not in name:
                continue
            fields = dict(kv.split("=", 1) for kv in name.split(":", 1)[1].split(","))
            runs.setdefault(fields["model"], {}).setdefault(v, {})[fields["subject"]] = p
    chosen, meta = {}, {"releases": releases, "models": {}}
    for m, by_v in sorted(runs.items()):
        complete = [v for v in by_v if len(by_v[v]) == N_SUBJECTS]
        meta["models"][m] = {"releases_with_runs": {v: len(s) for v, s in sorted(by_v.items(), key=lambda x: release_key(x[0]))}}
        if complete:
            v = max(complete, key=release_key)
            chosen[m] = (v, by_v[v])
            meta["models"][m]["chosen_release"] = v
        else:
            meta["models"][m]["chosen_release"] = None
    print(f"{len(runs)} models found, {len(chosen)} with a complete release; listing took {time.time() - t0:.0f} s", flush=True)
    jobs = [(m, v, s, p) for m, (v, subs) in chosen.items() for s, p in sorted(subs.items())]

    def work(job):
        m, v, s, p = job
        return m, v, s, parse_run(p)

    n_rows = 0
    with gzip.open(args.out, "wt", encoding="utf-8") as f, ThreadPoolExecutor(args.threads) as ex:
        f.write("model\trelease\tsubject\tinstance_id\tanswer\tkey\tprompt_sha1\tcompletion\n")
        for i, (m, v, s, rows) in enumerate(ex.map(work, jobs), 1):
            for r in rows:
                f.write(f"{m}\t{v}\t{s}\t{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\n")
            n_rows += len(rows)
            meta["models"][m].setdefault("items_parsed", 0)
            meta["models"][m]["items_parsed"] += len(rows)
            if i % 200 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} runs parsed, {n_rows} rows, {time.time() - t0:.0f} s", flush=True)
    meta["rows"] = n_rows
    meta["seconds"] = round(time.time() - t0)
    with open(args.meta, "w") as f:
        json.dump(meta, f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
