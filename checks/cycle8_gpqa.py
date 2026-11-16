"""Cycle 8, GPQA replication (protocol Section 50): reuse the MMLU-Pro scripts unchanged on HELM Capabilities GPQA main.

Run ONLY inside an isolated working folder that holds copies of the scripts (the reused analysis writes and resumes
fixed file names such as cycle8_pro_bootstrap.jsonl, which must not collide with the MMLU-Pro outputs):
    mkdir -p W/checks W/data && cp checks/cycle8_*.py W/checks/ && cd W
    python3 checks/cycle8_gpqa.py download --out data/helm_mmlu_pro_parsed.tsv.gz --meta data/helm_mmlu_pro_runs.json
    python3 checks/cycle8_gpqa.py checks --data data --out checks/cycle8_gpqa_step0_checks.out
    python3 checks/cycle8_gpqa.py analysis --data data --mde 0 --workers 16
The download step executes the MMLU-Pro download script with one change: the scenario prefix "mmlu_pro:" becomes
"gpqa:subset=gpqa_main,". The analysis step sets K = 4 (four options, guessing floor 1/4) before running.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    step, rest = sys.argv[1], sys.argv[2:]
    sys.argv = [sys.argv[0]] + rest
    assert not os.path.exists(os.path.join(HERE, "cycle8_pro_analysis.out")) or step != "analysis", \
        "refusing to run the analysis in a folder that already holds MMLU-Pro outputs"
    if step == "download":
        path = os.path.join(HERE, "cycle8_pro_step0_download.py")
        src = open(path).read()
        assert src.count('"mmlu_pro:"') == 1
        src = src.replace('"mmlu_pro:"', '"gpqa:subset=gpqa_main,"')
        exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
    elif step == "checks":
        import cycle8_pro_step0_checks as c
        c.main()
    elif step == "analysis":
        import cycle8_pro_analysis as pa
        pa.K = 4
        pa.main()
    else:
        raise SystemExit("step: download | checks | analysis")


if __name__ == "__main__":
    main()
