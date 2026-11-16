"""Cycle 8, quantisation check (protocol Section 51). POST HOC.
Qwen2.5-7B at 8-bit vs 4-bit vs HELM hosted: accuracy, invalid share, answer agreement, and the excess same-wrong
agreement x with the HELM run (N2 fitted on the 70 HELM models plus both local runs).
    .venv/bin/python checks/cycle8_q8_check.py --data data --local data/ru
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cycle8_ru_analysis as ru                      # noqa: E402
import cycle8_s0_synthetic_validation as v1          # noqa: E402
import cycle8_s0b_synthetic_validation as v2         # noqa: E402
import cycle8_s1_analysis as s1                      # noqa: E402


def read(path, order, n):
    a = np.full(n, -2, dtype=np.int8)
    op = gzip.open(path + ".gz", "rt") if not os.path.exists(path) else open(path)
    for line in op:
        r = json.loads(line)
        if r["k"] in order and "ans" in r:
            a[order[r["k"]]] = r["ans"]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--local", required=True)
    args = ap.parse_args()
    g = s1.load_real(args.data)
    items = set()
    with gzip.open(os.path.join(args.data, "helm_mmlu_parsed.tsv.gz"), "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            fs = line.split("\t")
            items.add((fs[2], fs[3]))
    order = {f"{s}|{i}": j for j, (s, i) in enumerate(sorted(items))}
    n = len(order)
    q8 = read(os.path.join(args.local, "qwen2.5-7b-q8.jsonl"), order, n)
    q4 = read(os.path.join(args.local, "qwen2.5-7b.jsonl"), order, n)
    assert (q8 > -2).all() and (q4 > -2).all(), "incomplete runs"
    h = g["models"].index("qwen_qwen2.5-7b-instruct-turbo")
    A = np.vstack([g["A"], q4[None, :], q8[None, :]]).astype(np.int8)
    key = g["key"]
    i4, i8 = A.shape[0] - 2, A.shape[0] - 1
    acc = (A == key[None, :]).mean(1)
    fit = v2.fit_1plg((A == key[None, :]).astype(float))
    X = ru.excess_matrix(A, key, fit[3], v1.pooled_wrong(A, key))
    L = ["Cycle 8 quantisation check: Qwen2.5-7B-Instruct, HELM MMLU 14,038 items (protocol Section 51). POST HOC",
         f"accuracy: 8-bit {acc[i8]:.4f}, 4-bit {acc[i4]:.4f}, HELM hosted {acc[h]:.4f}",
         f"invalid share: 8-bit {np.mean(q8 < 0):.4f}, 4-bit {np.mean(q4 < 0):.4f}",
         f"answer agreement: 8-bit vs 4-bit {np.mean(q8 == q4):.4f}; 8-bit vs HELM {np.mean(A[i8] == A[h]):.4f}; 4-bit vs HELM {np.mean(A[i4] == A[h]):.4f}",
         f"excess same-wrong agreement x with the HELM run: 8-bit {X[i8, h]:.4f}, 4-bit {X[i4, h]:.4f}; 8-bit vs 4-bit {X[i8, i4]:.4f}"]
    text = "\n".join(L)
    open(os.path.join(HERE, "cycle8_q8_check.out"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
