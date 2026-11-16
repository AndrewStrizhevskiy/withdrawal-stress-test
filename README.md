# Withdrawal stress test for LLM-based decision support

Code, protocol and outputs for the study *Operational autonomy of LLM-based decision support under provider withdrawal: shared errors, diversity caps and model families* (A. V. Strizhevskiy, E. D. Verkhovykh, D. M. Paramonov, M. I. Kolosov, Bauman Moscow State Technical University; manuscript under review).

The study treats a portfolio of large language models (LLMs) as a system of components. It measures what the system loses when groups of models are withdrawn, using reliability importance measures: the error increase ΔE, risk achievement worth (RAW) and joint importance (JI). Data:
- public HELM answers of 70 models on MMLU and 58 models on MMLU-Pro;
- 12 local runs of Russian and foreign open models;
- pair statistics for 349 Open LLM Leaderboard models.

## Contents

| Path | What it is |
|---|---|
| `protocol/withdrawal_stress_test_protocol.md` | The internal analysis protocol with its full amendment history (Sections 1–55): hypotheses, synthetic validation, every result, every post-hoc analysis, with dates. Four lines about the administration of a department server are redacted. |
| `checks/cycle8_s0*.py`, `cycle8_s0pro*.py` | Synthetic validation of the tests (size, power, interval coverage). |
| `checks/cycle8_s1_step0_*.py`, `cycle8_pro_step0_*.py` | Download and parsing of the public HELM MMLU and MMLU-Pro outputs; panel construction. |
| `checks/cycle8_s1_analysis.py`, `cycle8_s1b_supplementary.py`, `cycle8_pro_analysis.py` | Pre-specified analyses (H1–H3, ΔE, RAW, JI, null models, admission chart). |
| `checks/cycle8_ru_*.py` | Local runs of 12 open models via Ollama, and their analyses. |
| `checks/cycle8_olb_*.py` | Inheritance analysis on the Open LLM Leaderboard pair table. |
| `checks/cycle8_h1_measures_posthoc.py`, `cycle8_sovereignty_cost.py`, `cycle8_null_panels_100.py`, `cycle8_olb_robustness.py`, `cycle8_olb_cardlineage.py`, `cycle8_dev_jackknife.py`, `cycle8_ru_posthoc_r2.py` | Post-hoc robustness analyses (labelled as such in the protocol). |
| `checks/cycle8_mera_*.py`, `data/mera_rummlu_prompts.jsonl.gz`, `data/mera_rummlu_fewshot_prompts.jsonl.gz`, `data/ru_mera/`, `data/ru_mera_fs/` | Russian-language runs on MERA ruMMLU (ai-forever/MERA, MIT licence), zero-shot and with five worked examples: prompts, answer parser (v1 and the post-hoc variant v2), analyses and the 12 local models' answers (protocol Sections 45–46, 53, 56–57, 59). |
| `checks/cycle8_h1_letterbias.py` | H1 under a null with model-specific answer-letter bias (N3; protocol Section 58). |
| `checks/cycle8_q8_check.py`, `checks/cycle8_tpro_analysis.py`, `data/ru_tpro/` | Quantisation check (8-bit vs 4-bit) and the 32B domestic model T-pro-it-2.1 (a fine-tune of Qwen3-32B) on all 14,038 MMLU items (`--items all`; protocol Sections 52, 55, 60) and on the earlier 2,850-item subset. |
| `checks/cycle8_redux_sensitivity.py` | MMLU-Redux exclusion sensitivity (pre-specified; uses `data/redux/`, MMLU-Redux 2.0, CC BY 4.0). |
| `data/olb/card_base_models.json` | Base models declared in Hugging Face model cards for the leaderboard models (public API). |
| `checks/*.out`, `*.json`, `*.jsonl` | The stored outputs from which every number in the paper is taken. |
| `data/helm_mmlu*_panel.json`, `*_runs.json` | Panel definitions and the list of HELM runs used. |
| `data/ru/`, `data/ru_superseded/` | Local model answers (one JSON line per item), including the superseded runs replaced after template corrections (protocol Section 28.1). |

## Reproducing

Python 3.10+ with NumPy. The scripts use only the standard library and NumPy.

1. **Third-party data.** These files are not redistributed here:
   - `data/helm_mmlu_parsed.tsv.gz`: run `python3 checks/cycle8_s1_step0_download.py`. It downloads the public HELM outputs.
   - `data/helm_mmlu_pro_parsed.tsv.gz`: run `python3 checks/cycle8_pro_step0_download.py`.
   - `data/helm_mmlu_prompts.jsonl.gz` (exact HELM prompts, needed for the local runs and the MMLU-Redux matching): run `python3 checks/cycle8_ru_prompts.py` (command line in its docstring).
   - `data/olb/olb_contents.json.gz`: the public `open-llm-leaderboard/contents` dataset (rows from the Hugging Face datasets-server API), as a gzipped JSON list.
   - `data/olb/kim2025_hf_featured_models.csv.gz`: the file `data/hugging_face/hf_featured_models.csv` from the repository of Kim et al. (2025), `nikhgarg/llm_correlated_errors_public`, gzipped. That repository has no licence, so the file is not included.
2. **Analyses.** Each script's docstring gives its command line. The main analyses ran on a 16-core server, and most take minutes to about an hour. Example: `python3 checks/cycle8_s1_analysis.py --data data --workers 16`.
3. **Local model runs.** These need an Ollama server with the models listed in `checks/ru_models*.txt`; see `checks/cycle8_ru_run.py` and protocol Sections 26–28.

## Notes

- The protocol was written and hash-recorded in a private working repository before each analysis. This public copy is a snapshot; the protocol text records the hashes of the scripts used.
- Correctness of the analysis code was checked on synthetic panels with known ground truth and by reproducing reported statistics from stored outputs.

## Acknowledgements

The authors are grateful to Anthropic for the Claude AI assistant, which supported the preparation of the analysis code and the editing of the text.
- HELM results belong to their authors (Stanford CRFM). Model answers in `data/ru/` were produced by the respective models under their licences.

## Licence

Code: MIT (see `LICENSE`). Our outputs and local model answers: CC BY 4.0.
