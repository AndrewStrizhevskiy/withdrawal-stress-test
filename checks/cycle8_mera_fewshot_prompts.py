"""Cycle 8 (protocol Section 56): 5-shot Russian prompts from MERA ruMMLU public_test.
Shots: per domain, the 5 items with the smallest meta.id (excluded from scoring). Reads the zero-shot prompt file for
items and keys, and re-downloads nothing: options and question text are parsed from the stored MERA rows file.
    python3 checks/cycle8_mera_fewshot_prompts.py --rows data/mera_rummlu_rows.json.gz --out data/mera_rummlu_fewshot_prompts.jsonl.gz
"""
import argparse
import collections
import gzip
import json

HEADER = "Ниже приведены вопросы с вариантами ответов по теме {subject}. Запишите букву правильного ответа.\n\n"
ITEM = "{text}\nA {option_a}\nB {option_b}\nC {option_c}\nD {option_d}\nОтвет:"
MAPPING = {"A": 0, "B": 1, "C": 2, "D": 3, "А": 0, "В": 1, "С": 2, "Д": 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = json.load(gzip.open(args.rows, "rt", encoding="utf-8"))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["meta"]["domain"]].append(r)
    out, shots_ids = [], []
    for dom, rs in sorted(by.items()):
        rs.sort(key=lambda r: r["meta"]["id"])
        shots, rest = rs[:5], rs[5:]
        shots_ids += [f"{dom}|{r['meta']['id']}" for r in shots]
        subj = rs[0]["inputs"]["subject"]
        ex = "".join(ITEM.format(**s["inputs"]) + " " + s["outputs"].strip() + "\n\n" for s in shots)
        for r in rest:
            out.append({"subject": dom, "instance_id": str(r["meta"]["id"]),
                        "prompt": HEADER.format(subject=subj) + ex + ITEM.format(**r["inputs"]),
                        "mapping": MAPPING, "key": "ABCD".index(r["outputs"].strip())})
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(out)} prompts, {len(shots_ids)} shot items excluded, {len(by)} domains; example tail:\n{out[0]['prompt'][-400:]}")


if __name__ == "__main__":
    main()
