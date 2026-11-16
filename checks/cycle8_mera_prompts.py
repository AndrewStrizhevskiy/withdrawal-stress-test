"""Cycle 8 (protocol Section 45): build prompts for MERA ruMMLU public_test (MIT) in the runner's format.

Each row: subject = meta.domain, instance_id = meta.id, prompt = MERA instruction filled with the item,
mapping = Latin A-D and Cyrillic look-alikes/equivalents (А, В, С, Д) -> 0..3, key = index of the answer letter.
Downloaded rows are untrusted data: parsed with json.loads, nothing executed.

    python3 checks/cycle8_mera_prompts.py --out data/mera_rummlu_prompts.jsonl.gz
"""
import argparse
import gzip
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://datasets-server.huggingface.co/rows?dataset=ai-forever/MERA&config=rummlu&split=public_test&offset={o}&length=100"
MAPPING = {"A": 0, "B": 1, "C": 2, "D": 3, "А": 0, "В": 1, "С": 2, "Д": 3}


def get(o):
    for t in range(5):
        try:
            return json.loads(urllib.request.urlopen(URL.format(o=o), timeout=60).read())["rows"]
        except Exception as e:                                  # noqa: BLE001
            err = e
            time.sleep(3 + 4 * t)
    raise err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10033)
    args = ap.parse_args()
    with ThreadPoolExecutor(6) as ex:
        parts = list(ex.map(get, range(0, args.n, 100)))
    rows = [r["row"] for p in parts for r in p]
    out = []
    for r in rows:
        inp = r["inputs"]
        prompt = r["instruction"].format(**inp)
        out.append({"subject": r["meta"]["domain"], "instance_id": str(r["meta"]["id"]), "prompt": prompt,
                    "mapping": MAPPING, "key": "ABCD".index(r["outputs"].strip())})
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    subj = {r["subject"] for r in out}
    print(f"{len(out)} prompts, {len(subj)} domains; example:\n{out[0]['prompt']}")


if __name__ == "__main__":
    main()
