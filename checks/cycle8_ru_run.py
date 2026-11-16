"""Cycle 8, Russian-model run (protocol Section 26): answer HELM MMLU prompts with local models via Ollama.

Standard library only (server). Each item: the exact HELM prompt as the user message (chat template applied by
Ollama), greedy decoding, at most 8 tokens, reasoning off where the model supports a switch. The answer is the first
character of the stripped completion mapped through HELM's output mapping (Step 0 rule); -1 if it is not an option.
Resumable: items already in the output file are skipped.

    python3 checks/cycle8_ru_run.py --models models.txt --prompts data/helm_mmlu_prompts.jsonl.gz --outdir data/ru
    (models.txt: one "label<TAB>ollama-name[<TAB>format]" per line; format "qwen3-nothink" sends the prompt raw in the
     official Qwen3 non-thinking chat format, for templates that always open a <think> block; "gigachat" sends it raw
     in the official GigaChat-20B chat format, for a GGUF packaged without a template; "yandexgpt" and "saiga-yandexgpt"
     send it raw in the official formats of those models, whose GGUF templates are broken or wrong)
A leading reasoning block ("...</think>") is removed before the first character is read.
"""
import argparse
import gzip
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HOST = "http://127.0.0.1:11434"
NUM_CTX, NUM_PREDICT = 4096, 8


def post(path, body, timeout=600):
    req = urllib.request.Request(HOST + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


QWEN3_NOTHINK = "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
# official GigaChat-20B-A3B-instruct-v1.5 chat template (tokenizer_config.json), one user turn, generation prompt
GIGACHAT = "user<|role_sep|>{p}<|message_sep|>available functions<|role_sep|>[]<|message_sep|>assistant<|role_sep|>"
# Ollama adds the BOS token itself when the GGUF asks for it (add_bos_token), so raw formats omit the leading <s>
# official YandexGPT-5-Lite-8B-instruct chat template (tokenizer_config.json, rendered with trim_blocks/lstrip_blocks)
YANDEXGPT = " Пользователь: {p}\n\n Ассистент:[SEP]"
# official saiga_yandexgpt_8b chat template (Llama-3-style headers, bos <s>, content trimmed)
SAIGA_YANDEXGPT = "<|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
RAW = {"qwen3-nothink": QWEN3_NOTHINK, "gigachat": GIGACHAT, "gigachat-bos": "<s>" + GIGACHAT,
       "yandexgpt": YANDEXGPT, "saiga-yandexgpt": SAIGA_YANDEXGPT}


def run_model(label, name, prompts, outdir, threads, fmt=""):
    path = os.path.join(outdir, f"{label}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            done.add(json.loads(line)["k"])
    caps = post("/api/show", {"model": name}).get("capabilities", [])
    think = "thinking" in caps and fmt not in RAW
    todo = [p for p in prompts if f"{p['subject']}|{p['instance_id']}" not in done]
    print(f"{label} ({name}): capabilities {caps}; {len(done)} done, {len(todo)} to run", flush=True)
    t0 = time.time()

    def one(p):
        body = {"model": name, "prompt": p["prompt"], "stream": False, "keep_alive": "30m",
                "options": {"temperature": 0, "seed": 0, "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX}}
        if fmt in RAW:
            body["prompt"], body["raw"] = RAW[fmt].format(p=p["prompt"].strip() if fmt == "saiga-yandexgpt" else p["prompt"]), True
        if think:
            body["think"] = False
        for t in range(3):
            try:
                r = post("/api/generate", body)
                break
            except Exception as e:
                if t == 2:
                    return {"k": f"{p['subject']}|{p['instance_id']}", "error": str(e)[:200]}
                time.sleep(5)
        comp = r.get("response", "")
        text = comp.split("</think>", 1)[1] if "</think>" in comp else comp
        c0 = text.strip()[:1]
        ans = p["mapping"].get(c0, -1)
        return {"k": f"{p['subject']}|{p['instance_id']}", "ans": ans, "key": p["key"], "comp": comp[:160],
                "n_prompt": r.get("prompt_eval_count"), "think": bool(r.get("thinking"))}

    with open(path, "a") as f, ThreadPoolExecutor(threads) as ex:
        for i, row in enumerate(ex.map(one, todo), 1):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 1000 == 0 or i == len(todo):
                f.flush()
                print(f"  {label}: {i}/{len(todo)}, {time.time() - t0:.0f} s", flush=True)
    try:
        post("/api/generate", {"model": name, "keep_alive": 0})          # unload
    except Exception:
        pass


def main():
    global NUM_CTX, NUM_PREDICT
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="test: first N items only")
    ap.add_argument("--num-ctx", type=int, default=4096, help="context window (raise for re-runs of truncated items)")
    ap.add_argument("--num-predict", type=int, default=8, help="maximum output tokens")
    args = ap.parse_args()
    NUM_CTX = args.num_ctx
    NUM_PREDICT = args.num_predict
    os.makedirs(args.outdir, exist_ok=True)
    prompts = [json.loads(l) for l in gzip.open(args.prompts, "rt", encoding="utf-8")]
    if args.limit:
        prompts = prompts[: args.limit]
    for line in open(args.models):
        if line.strip() and not line.startswith("#"):
            parts = line.rstrip("\n").split("\t")
            run_model(parts[0], parts[1], prompts, args.outdir, args.threads, parts[2] if len(parts) > 2 else "")
    print("all models done", flush=True)


if __name__ == "__main__":
    main()
