"""Cycle 8, protocol Section 61 (C1): apply the calibrated factors to the reported real intervals and combine them with
the developer-jackknife SE (Section 44). POST HOC. Inputs are copied from the stored outputs named below."""
import json
import math

K = {"MMLU": json.load(open("checks/cycle8_ci_calibration_mmlu.out"))["k_median"],
     "MMLU-Pro": json.load(open("checks/cycle8_ci_calibration_pro.out"))["k_median"]}
# (benchmark, quantity, estimate, lo, hi, jackknife SE or None); sources: cycle8_s1_analysis.out, cycle8_sovereignty_cost.out,
# cycle8_dev_jackknife.out
ROWS = [("MMLU", "dE API-only", 2.19, 1.24, 5.28, 5.25),
        ("MMLU", "dE Anthropic", 0.59, 0.26, 1.27, None),
        ("MMLU", "dE DeepSeek", 0.46, 0.02, 0.83, None),
        ("MMLU", "dE Google", 0.38, 0.06, 0.93, None),
        ("MMLU", "dE OpenAI", 0.27, -0.10, 0.69, None),
        ("MMLU", "C (cap vs open5)", 2.86, 1.89, 3.41, 1.23),
        ("MMLU", "C' (cap vs best open)", 2.33, 1.46, 2.96, None),
        ("MMLU-Pro", "C (cap vs open5)", 5.20, 2.90, 7.90, 2.21)]
L = [f"Calibrated intervals (protocol Section 61). k: MMLU {K['MMLU']:.3f}, MMLU-Pro {K['MMLU-Pro']:.3f}",
     "benchmark | quantity | estimate | bootstrap 95% | calibrated (item sampling) | + developer jackknife (model sampling)"]
out = []
for b, q, e, lo, hi, se in ROWS:
    k = K[b]
    clo, chi = e - k * (e - lo), e + k * (hi - e)
    if se is not None:
        hw_lo, hw_hi = math.hypot(e - clo, 1.96 * se), math.hypot(chi - e, 1.96 * se)
        comb = f"[{e - hw_lo:.2f}, {e + hw_hi:.2f}]"
    else:
        comb = "n/a"
    L.append(f"{b} | {q} | {e:.2f} | [{lo:.2f}, {hi:.2f}] | [{clo:.2f}, {chi:.2f}] | {comb}")
    out.append(dict(benchmark=b, quantity=q, est=e, lo=lo, hi=hi, cal_lo=round(clo, 2), cal_hi=round(chi, 2), jk_se=se, combined=comb))
L.append("Note: the synthetic validation covers Delta E; applying k to C and C' is an approximation (same estimator type).")
open("checks/cycle8_ci_apply.out", "w").write("\n".join(L) + "\n")
json.dump(out, open("checks/cycle8_ci_apply.json", "w"), indent=1)
print("\n".join(L))
