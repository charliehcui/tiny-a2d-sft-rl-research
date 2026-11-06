"""使用官方 dLLM 和 lm-evaluation-harness 评测，不包含训练逻辑。"""

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", "/opt/bluewhale_cache/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", "/opt/bluewhale_cache/datasets")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "cache" / "matplotlib"))
os.environ.setdefault("WANDB_MODE", "disabled")

import torch
import transformers
import dllm
from lm_eval import evaluator
from lm_eval.api.metrics import exact_match_fn
from lm_eval.filters.extraction import RegexFilter
from lm_eval.tasks import TaskManager
from dllm.core.eval.mdlm import MDLMEvalHarness


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def score_response(response, reference, filters, metric):
    scores = {}
    for name, extractor in filters.items():
        extracted = extractor.apply([[response]], [{}])[0][0]
        result = exact_match_fn(predictions=[extracted], references=[reference], regexes_to_ignore=metric["regexes_to_ignore"], ignore_case=metric["ignore_case"], ignore_punctuation=metric["ignore_punctuation"])
        scores[name] = {"extracted_answer": extracted, "correct": bool(result["exact_match"])}
    return scores


def check_scoring(filters, metric):
    cases = [
        ("<answer>42</answer>", "42", True, True),
        ("<answer>41</answer>", "42", False, False),
        ("<answer>-5</answer>", "-5", True, True),
        ("<answer>12,345</answer>", "12345", True, True),
        ("The answer is 42.", "42", False, True),
        ("", "42", False, False),
    ]
    for response, reference, xml, flexible in cases:
        scores = score_response(response, reference, filters, metric)
        if scores["xml-answer"]["correct"] != xml or scores["flexible-extract"]["correct"] != flexible:
            raise ValueError("Official extraction/metric scoring check failed")
    return {"status": "passed", "cases": len(cases), "implementation": "official RegexFilter and exact_match_fn"}


def resource_summary(path):
    rows = []
    with path.open(encoding="utf-8") as source:
        for row in csv.reader(source):
            if len(row) != 5:
                continue
            try:
                rows.append([float(value.strip()) for value in row[1:]])
            except ValueError:
                continue
    if not rows:
        return {"samples": 0}
    return {"samples": len(rows), "mean_gpu_utilization_percent": statistics.mean(row[0] for row in rows), "max_gpu_utilization_percent": max(row[0] for row in rows), "peak_device_memory_mib": max(row[1] for row in rows), "mean_power_watts": statistics.mean(row[2] for row in rows), "max_temperature_celsius": max(row[3] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROJECT / "configs" / "baseline_protocol_v2.json")
    args = parser.parse_args()
    if args.split == "test" and args.limit is not None:
        raise ValueError("Final baseline must use all 1319 fixed test examples")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    protocol_path = args.protocol
    protocol = json.loads(protocol_path.read_text())
    original_task = json.loads((PROJECT / "configs" / "data_tasks" / "gsm8k_bluewhale_xml.yaml").read_text())
    filters = {item["name"]: RegexFilter(**{key: value for key, value in item["filter"][0].items() if key != "function"}) for item in original_task["filter_list"]}
    metric = original_task["metric_list"][0]
    save_json(args.output / "scoring_checks.json", check_scoring(filters, metric))
    prompts = read_jsonl(PROJECT / "data" / "gsm8k" / "processed" / "eval" / f"{args.split}_prompts.jsonl")
    labels = read_jsonl(PROJECT / "data" / "gsm8k" / "processed" / "eval" / f"{args.split}_labels.jsonl")
    if args.limit:
        prompts = prompts[:args.limit]
        labels = labels[:args.limit]
    labels_by_id = {row["id"]: row for row in labels}
    task = json.loads(json.dumps(original_task))
    task["task"] = "gsm8k_bluewhale_xml_" + args.split
    task["test_split"] = args.split
    for key, relative in task["dataset_kwargs"]["data_files"].items():
        task["dataset_kwargs"]["data_files"][key] = str(PROJECT / relative)
    task_folder = args.output / "task"
    task_folder.mkdir()
    save_json(task_folder / "gsm8k.yaml", task)
    settings = {"protocol": protocol, "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(), "split": args.split, "limit": args.limit, "expected_examples": len(prompts), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)), "model_path": str(args.checkpoint.resolve()), "runtime_environment": "/opt/bluewhale_env", "prompt_source": str(PROJECT / "data" / "gsm8k" / "processed" / "eval" / f"{args.split}_prompts.jsonl"), "answers_are_never_used_to_construct_model_inputs": True}
    save_json(args.output / "run_config.json", settings)
    transformers.set_seed(protocol["seed"])
    torch.set_num_threads(4)
    load_started = time.perf_counter()
    model = MDLMEvalHarness(pretrained=settings["model_path"], dtype=protocol["dtype"], device="cuda", attn_implementation=protocol["attn_implementation"], batch_size=protocol["batch_size"], max_new_tokens=protocol["max_new_tokens"], steps=protocol["steps"], block_size=protocol["block_size"], temperature=protocol["temperature"], cfg_scale=protocol["cfg_scale"], remasking=protocol["remasking"], right_shift_logits=protocol["right_shift_logits"])
    model.model.config.use_cache = False
    model.model.eval()
    if hasattr(model.model, "peft_config"):
        raise ValueError("Baseline must not load a PEFT adapter")
    load_seconds = time.perf_counter() - load_started
    token_to_prompt = {}
    for row in prompts:
        context = model.tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        token_to_prompt[token_hash(model.tokenizer(context)["input_ids"])] = row
    if len(token_to_prompt) != len(prompts):
        raise ValueError("Duplicate tokenized prompts")
    warmup_prompts = read_jsonl(PROJECT / "data" / "gsm8k" / "processed" / "eval" / "validation_prompts.jsonl")[:protocol["batch_size"]]
    warmup_inputs = []
    for row in warmup_prompts:
        warmup_text = model.tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        warmup_inputs.append(torch.tensor(model.tokenizer(warmup_text)["input_ids"], device="cuda"))
    warmup_start = time.perf_counter()
    model.sampler.sample(inputs=warmup_inputs, config=model.sampler_config, return_dict=False)
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_start
    print(json.dumps({"event": "model_loaded", "gpu": torch.cuda.get_device_name(0), "model_class": type(model.model).__name__, "load_seconds": load_seconds, "warmup_seconds": warmup_seconds, "examples": len(prompts)}), flush=True)
    records = []
    batch_times = []
    counter = {"calls": 0}

    def count_forward(module, inputs):
        counter["calls"] += 1

    handle = model.model.register_forward_pre_hook(count_forward)
    original_sample = model.sampler.sample
    predictions_file = (args.output / "predictions.jsonl").open("x", encoding="utf-8")

    def timed_sample(*sample_args, **sample_kwargs):
        inputs = sample_kwargs.get("inputs", sample_args[0] if sample_args else None)
        input_lists = [item.tolist() for item in inputs]
        batch_rows = []
        for input_ids in input_lists:
            row = token_to_prompt.get(token_hash(input_ids))
            if row is None:
                raise ValueError("Model input does not match the fixed answer-free prompt file")
            batch_rows.append(row)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        calls_before = counter["calls"]
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        start_event.record()
        output = original_sample(*sample_args, **sample_kwargs)
        end_event.record()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        batch_times.append(seconds)
        output_lists = output.tolist()
        texts = dllm.utils.sample_trim(model.tokenizer, output_lists, input_lists)
        for index, row in enumerate(batch_rows):
            input_ids = input_lists[index]
            generated = output_lists[index][len(input_ids):len(input_ids) + protocol["max_new_tokens"]]
            eos = model.tokenizer.eos_token_id
            response_tokens = generated.index(eos) if eos in generated else len(generated)
            text = texts[index]
            scores = score_response(text, labels_by_id[row["id"]]["answer_value"], filters, metric)
            record = {"id": row["id"], "question": row["question"], "prompt": row["prompt"], "model_input_text": model.tokenizer.decode(input_ids, skip_special_tokens=False), "input_token_sha256": token_hash(input_ids), "input_tokens": len(input_ids), "response": text, "reference_answer": labels_by_id[row["id"]]["answer_value"], "scores": scores, "generation_seconds": seconds / len(inputs), "timing_definition": "batch_wall_seconds_divided_by_batch_size", "batch_generation_seconds": seconds, "batch_size": len(inputs), "batch_index": len(batch_times) - 1, "cuda_elapsed_milliseconds": start_event.elapsed_time(end_event) / len(inputs), "forward_calls": counter["calls"] - calls_before, "output_tokens_before_eos": response_tokens, "generation_canvas_tokens": len(generated), "has_eos": eos in generated, "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            records.append(record)
            predictions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        predictions_file.flush()
        if len(records) == len(inputs) or len(batch_times) % 5 == 0:
            print(json.dumps({"event": "progress", "completed": len(records), "total": len(prompts), "mean_generation_seconds": statistics.mean(item["generation_seconds"] for item in records)}), flush=True)
        return output

    model.sampler.sample = timed_sample
    monitor_file = (args.output / "gpu_resources.csv").open("x", encoding="utf-8")
    monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu", "--format=csv,noheader,nounits", "-l", "1"], stdout=monitor_file, stderr=subprocess.DEVNULL)
    evaluation_started = time.perf_counter()
    try:
        result = evaluator.simple_evaluate(model=model, tasks=[task["task"]], task_manager=TaskManager(include_path=str(task_folder), include_defaults=False), num_fewshot=0, batch_size=protocol["batch_size"], limit=args.limit, log_samples=True, apply_chat_template=True, random_seed=protocol["seed"], numpy_random_seed=protocol["seed"], torch_random_seed=protocol["seed"], fewshot_random_seed=protocol["seed"], bootstrap_iters=0)
    finally:
        predictions_file.close()
        handle.remove()
        monitor.terminate()
        monitor.wait(timeout=10)
        monitor_file.close()
    evaluation_seconds = time.perf_counter() - evaluation_started
    save_json(args.output / "native_results.json", result)
    if len(records) != len(prompts) or {item["id"] for item in records} != {item["id"] for item in prompts}:
        raise ValueError("Evaluated IDs do not match the fixed selected split")
    primary = sum(item["scores"]["flexible-extract"]["correct"] for item in records) / len(records)
    secondary = sum(item["scores"]["xml-answer"]["correct"] for item in records) / len(records)
    native = result["results"][task["task"]]
    if abs(primary - native["exact_match,flexible-extract"]) > 1e-12 or abs(secondary - native["exact_match,xml-answer"]) > 1e-12:
        raise ValueError("Per-question correctness does not match official harness metrics")
    seconds = [item["generation_seconds"] for item in records]
    total_seconds = time.time() - started
    summary = {"status": "complete", "split": args.split, "examples": len(records), "primary_metric": protocol["primary_metric"], "correct": sum(item["scores"]["flexible-extract"]["correct"] for item in records), "accuracy": primary, "xml_answer_accuracy": secondary, "model_load_seconds": load_seconds, "warmup_seconds_excluded_from_generation_mean": warmup_seconds, "mean_generation_seconds": statistics.mean(seconds), "median_generation_seconds": statistics.median(seconds), "p95_generation_seconds": sorted(seconds)[min(len(seconds) - 1, int(0.95 * len(seconds)))], "sum_generation_seconds": sum(seconds), "evaluation_wall_seconds": evaluation_seconds, "total_process_seconds": total_seconds, "estimated_process_gpu_cost_cny": total_seconds / 3600 * protocol["gpu_price_cny_per_hour"], "gpu_name": torch.cuda.get_device_name(0), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory, "peak_torch_allocated_bytes": max(item["peak_allocated_bytes"] for item in records), "peak_torch_reserved_bytes": max(item["peak_reserved_bytes"] for item in records), "forward_calls_per_example": sorted({item["forward_calls"] for item in records}), "no_eos_fraction": sum(not item["has_eos"] for item in records) / len(records), "gpu_monitor": resource_summary(args.output / "gpu_resources.csv"), "protocol_sha256": settings["protocol_sha256"], "input_prompt_hashes_checked": True, "native_metric_agreement_checked": True, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    summary["batch_size"] = protocol["batch_size"]
    summary["batches"] = len(batch_times)
    summary["mean_batch_generation_seconds"] = statistics.mean(batch_times)
    summary["timing_definition"] = "batch wall time divided by actual batch size; single-request latency only when batch_size=1"
    save_json(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
