"""Cycle 8, POST HOC (protocol Section 35, R1): like-for-like sovereignty cost on MMLU and MMLU-Pro.

C  = open5 - cap_full;  C' = min(open5, best_single_open) - min(cap_full, greedy_full, best_single_any).
Paired percentile intervals over the existing bootstrap resamples.
    python3 checks/cycle8_sovereignty_cost.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def q(v, a):
    v = sorted(v)
    i = a * (len(v) - 1)
    lo = int(i)
    return v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (i - lo)


def cost(r):
    c = r["S/open5"] - r["H3/cap/full"]
    c2 = min(r["S/open5"], r["S/best_single_open"]) - min(r["H3/cap/full"], r["H3/greedy/full"], r["S/best_single_any"])
    return c, c2


L = ["Cycle 8 POST HOC: like-for-like sovereignty cost (protocol Section 35, R1); points of error"]
out = {}
for name, js, boot in (("MMLU", "cycle8_s1_analysis.json", "cycle8_s1_bootstrap.jsonl"),
                       ("MMLU-Pro", "cycle8_pro_analysis.json", "cycle8_pro_bootstrap.jsonl")):
    p = json.load(open(os.path.join(HERE, js)))["point"]
    bs = [json.loads(l)["res"] for l in open(os.path.join(HERE, boot))]
    pc, pc2 = cost(p)
    cs = [cost(r) for r in bs]
    c1 = [c for c, _ in cs]
    c2 = [c for _, c in cs]
    out[name] = {"C": [pc, q(c1, .025), q(c1, .975)], "C_prime": [pc2, q(c2, .025), q(c2, .975)], "n_boot": len(bs),
                 "parts": {k: p[k] for k in ("S/open5", "H3/cap/full", "S/best_single_open", "H3/greedy/full", "S/best_single_any")}}
    L.append(f"{name} ({len(bs)} resamples): C = {100*pc:.2f} [{100*q(c1,.025):.2f}, {100*q(c1,.975):.2f}]; "
             f"C' = {100*pc2:.2f} [{100*q(c2,.025):.2f}, {100*q(c2,.975):.2f}]; parts: "
             + ", ".join(f"{k} {100*v:.2f}" for k, v in out[name]["parts"].items()))
json.dump(out, open(os.path.join(HERE, "cycle8_sovereignty_cost.json"), "w"), indent=1)
open(os.path.join(HERE, "cycle8_sovereignty_cost.out"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
