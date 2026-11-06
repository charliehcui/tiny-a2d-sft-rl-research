"""先评测 SFT，再用现有 diffu-GRPO 训练固定 RL 子集，最后评测并比较 RL。"""

import argparse
import collections
import hashlib
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from run_experiment import read_json, write_json


PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", "/opt/bluewhale_cache/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", "/opt/bluewhale_cache/datasets")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MPLCONFIGDIR", "/opt/bluewhale_cache/matplotlib")


def read_rows(path):
    # 按实际换行读取，保留题目字符串内部合法的 Unicode 行分隔符。
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def append_row(path, value):
    with Path(path).open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def load_data(config):
    if config.get("data_selection") == "static_mixed_v1":
        from prepare_rl_static import load_static_rows

        return load_static_rows(PROJECT, config)
    path = PROJECT / config["data_path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != config["data_sha256"]:
        raise ValueError("Fixed RL subset checksum mismatch")
    rows = read_rows(path)
    manifest = read_json(PROJECT / config["data_manifest"])
    source = PROJECT / manifest["source"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("Original RL split checksum mismatch")
    if rows != read_rows(source)[:512] or [row["id"] for row in rows] != manifest["selected_ids"]:
        raise ValueError("RL subset must be the registered first 512 fixed RL examples")
    if len(rows) != 512 or len({row["id"] for row in rows}) != 512:
        raise ValueError("Expected exactly 512 unique RL questions")
    other = read_rows(PROJECT / "data/gsm8k/processed/sft_train.jsonl") + read_rows(PROJECT / "data/gsm8k/processed/sft_validation.jsonl")
    if {row["id"] for row in rows} & {row["id"] for row in other}:
        raise ValueError("RL questions overlap with SFT or development data")
    return rows


class OutcomeReward:
    __name__ = "outcome_correctness"

    def __init__(self, tokenizer, run_dir, expected_rows=None):
        from evaluate_model import check_scoring
        from lm_eval.filters.extraction import RegexFilter

        task = read_json(PROJECT / "configs/data_tasks/gsm8k_bluewhale_xml.yaml")
        self.filters = {item["name"]: RegexFilter(**{key: value for key, value in item["filter"][0].items() if key != "function"}) for item in task["filter_list"]}
        self.metric = task["metric_list"][0]
        self.tokenizer = tokenizer
        self.run_dir = run_dir
        self.trainer = None
        self.groups = []
        self.expected_rows = expected_rows
        write_json(run_dir / "reward_scoring_checks.json", check_scoring(self.filters, self.metric))

    def __call__(self, prompts, completions, completion_ids, answer, id, **kwargs):
        import dllm
        from evaluate_model import score_response

        if len(id) != 4 or len(set(id)) != 1 or len(set(answer)) != 1:
            raise ValueError("Each native generation batch must contain one question with four candidates")
        metadata = {}
        if self.expected_rows is not None:
            if len(self.groups) >= len(self.expected_rows):
                raise ValueError("Unexpected extra generation beyond the fixed schedule")
            expected = self.expected_rows[len(self.groups)]
            if id != [expected["id"]] * 4 or answer != [expected["answer"]] * 4 or prompts != [expected["prompt"]] * 4:
                raise ValueError("Question, answer or prompt does not match the next scheduled occurrence")
            for key in ["question_id", "occurrence_id", "presentation_index", "pool", "presentation_number", "scheduled_optimizer_step"]:
                if kwargs.get(key) != [expected[key]] * 4:
                    raise ValueError("Four candidates must belong to the same scheduled occurrence: " + key)
                metadata[key] = expected[key]
            if self.trainer is not None and self.trainer.state.global_step != expected["scheduled_optimizer_step"] - 1:
                raise ValueError("The scheduled occurrence reached the wrong optimizer update")
        # 原生文本可能含 EOS 后内容。复用冻结评测器的截断函数，奖励与实际输出范围保持一致。
        texts = dllm.utils.sample_trim(self.tokenizer, completion_ids, [[] for _ in completion_ids])
        values = []
        for index, text in enumerate(texts):
            scored = score_response(text, answer[index], self.filters, self.metric)
            reward = float(scored["flexible-extract"]["correct"])
            values.append(reward)
            append_row(self.run_dir / "rollouts.jsonl", {"group": len(self.groups), "id": id[index], "candidate": index, "optimizer_step_before_update": self.trainer.state.global_step if self.trainer else 0, "prompt": prompts[index], "response": text, "reference_answer": answer[index], "scores": scored, "reward": reward, "completion_tokens_including_eos": len(completion_ids[index]), **metadata})
        group = {"group": len(self.groups), "id": id[0], "correct_candidates": int(sum(values)), "candidates": 4, "distinct_outputs": len(set(texts)), "mean_reward": sum(values) / 4, "advantages": [value - sum(values) / 4 for value in values], **metadata}
        self.groups.append(group)
        append_row(self.run_dir / "reward_groups.jsonl", group)
        return values


def train(config, run_dir, smoke):
    import torch
    import transformers
    import dllm
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig
    from dllm.pipelines.rl import DiffuGRPOConfig, DiffuGRPOTrainer
    from experiment_callback import ExperimentCallback

    class RLCallback(ExperimentCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            super().on_log(args, state, control, logs, **kwargs)
            if any(isinstance(value, (float, int)) and not math.isfinite(value) for value in (logs or {}).values()):
                raise RuntimeError("Non-finite RL training metric; stop without expanding the experiment")

    started = time.perf_counter()
    for name in ["rollouts.jsonl", "reward_groups.jsonl", "metrics.jsonl", "checkpoints", "final_adapter", "final_model"]:
        if (run_dir / name).exists():
            raise FileExistsError("Training outputs already exist; do not overwrite or silently resume: " + name)
    if shutil.disk_usage(run_dir).free < config["minimum_free_gib"] * 2**30:
        raise RuntimeError("Insufficient persistent space for RL checkpoints and final model")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16 CUDA GPU is required")
    if hashlib.sha256((PROJECT / config["evaluation_protocol"]).read_bytes()).hexdigest() != config["evaluation_protocol_sha256"]:
        raise ValueError("The frozen evaluation protocol changed")
    rows = load_data(config)
    static = config.get("data_selection") == "static_mixed_v1"
    model_path = PROJECT / config["model_path"]
    provenance = read_json(model_path / "source_training.json")
    if provenance["adapter_sha256"] != config["source_adapter_sha256"] or not provenance["merged_diffusion_model"]:
        raise ValueError("RL must start from the completed, merged SFT model")
    options = dict(config["training"])
    if smoke:
        options.update(max_steps=1, save_steps=1)
    options["output_dir"] = str(run_dir / "checkpoints")
    training_args = DiffuGRPOConfig(**options)
    transformers.set_seed(training_args.seed)
    torch.set_num_threads(4)
    write_json(run_dir / "rl_config.json", {**config, "smoke": smoke, "actual_training_options": options})
    shutil.copy2(__file__, run_dir / "code/train_rl.py")
    shutil.copy2(PROJECT / config["data_manifest"], run_dir / "data_manifest.json")
    if static:
        shutil.copy2(PROJECT / "scripts/prepare_rl_static.py", run_dir / "code/prepare_rl_static.py")
        shutil.copy2(PROJECT / config["data_path"], run_dir / "sampling_schedule.jsonl")
    model_args = dllm.utils.ModelArguments(model_name_or_path=str(model_path), dtype="bfloat16", attn_implementation="sdpa", lora=False, load_in_4bit=False)
    tokenizer = dllm.utils.get_tokenizer(model_args)
    for row in rows:
        if row["prompt"] != [{"role": "user", "content": "\nRespond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n\n\n" + row["question"]}]:
            raise ValueError("RL prompt differs from the fixed answer-free evaluation template")
    model = dllm.utils.get_model(model_args)
    model.config.use_cache = False
    sampler = MDLMSampler(model=model, tokenizer=tokenizer)
    sampler_config = MDLMSamplerConfig(steps=256, max_new_tokens=256, block_size=256, temperature=1.0, cfg_scale=0.0, remasking="low_confidence")
    expected_rows = rows[:16] if smoke else rows
    reward = OutcomeReward(tokenizer, run_dir, expected_rows=expected_rows if static else None)
    trainer = DiffuGRPOTrainer(model=model, reward_funcs=[reward], args=training_args, train_dataset=Dataset.from_list(rows), processing_class=tokenizer, peft_config=LoraConfig(**config["lora"]), sampler=sampler, sampler_config=sampler_config)
    reward.trainer = trainer
    if static and (trainer.accelerator.num_processes != 1 or trainer.shuffle_dataset or training_args.steps_per_generation != 4 or training_args.generation_batch_size != 4 or training_args.num_iterations != 1):
        raise ValueError("Static schedule requires one process, native ordered G=4 sampling and no rollout replay")
    if trainer.ref_model is not None or not isinstance(trainer.model, PeftModel):
        raise ValueError("Expected frozen merged SFT as the reference through disabling only the new RL adapter")
    # TRL 0.19 的初始化不传递该参数，显式保持与当前 Transformers 的非重入实现一致。
    trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer.model.enable_input_require_grads()
    trainable = {name: parameter for name, parameter in trainer.model.named_parameters() if parameter.requires_grad}
    count = sum(parameter.numel() for parameter in trainable.values())
    if count != config["expected_lora_parameters"] or any("lora_" not in name for name in trainable):
        raise ValueError("Only the new RL LoRA matrices may be updated")
    initial = {name: parameter.detach().cpu().clone() for name, parameter in trainable.items()}
    trainer.model.eval()
    probe = tokenizer("Check the frozen SFT reference.", return_tensors="pt")["input_ids"].to(trainer.accelerator.device)
    with torch.inference_mode():
        enabled = trainer.model(probe).logits[:, -1, :].float()
        with trainer.model.disable_adapter():
            disabled = trainer.model(probe).logits[:, -1, :].float()
        difference = (enabled - disabled).abs().max().item()
    if difference != 0.0:
        raise ValueError("Initial RL adapter changes the SFT reference logits")
    write_json(run_dir / "model_checks.json", {"base_model": str(model_path), "reference_policy": "merged SFT with only the new RL adapter disabled", "reference_logit_max_abs_difference_at_start": difference, "trainable_lora_parameters": count, "num_generations": training_args.num_generations, "generation_batch_size": training_args.generation_batch_size, "steps_per_generation": training_args.steps_per_generation, "gradient_accumulation_steps": training_args.gradient_accumulation_steps, "effective_questions_per_update": 16, "native_trainer": type(trainer).__name__, "trl_model_accepts_loss_kwargs": trainer.model_accepts_loss_kwargs})
    if static:
        model_checks = read_json(run_dir / "model_checks.json")
        model_checks.update(shuffle_dataset=trainer.shuffle_dataset, expected_presentations=len(expected_rows), unique_scheduled_questions=len({row["id"] for row in expected_rows}), fresh_group_per_occurrence=True)
        write_json(run_dir / "model_checks.json", model_checks)
    del enabled, disabled, probe
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    trainer.add_callback(RLCallback())
    monitor_file = (run_dir / "training_gpu_resources.csv").open("x")
    monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu", "--format=csv,noheader,nounits", "-l", "1"], stdout=monitor_file, stderr=subprocess.DEVNULL)
    training_started = time.perf_counter()
    try:
        result = trainer.train()
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        monitor_file.close()
    training_seconds = time.perf_counter() - training_started
    trainer.save_state()
    trainer.save_model(str(run_dir / "final_adapter"))
    tokenizer.save_pretrained(run_dir / "final_adapter")
    groups = reward.groups
    expected_groups = 16 if smoke else 512
    if len(groups) != expected_groups or (not static and len({group["id"] for group in groups}) != expected_groups):
        raise RuntimeError("Native sampler did not produce exactly the planned unique questions")
    if static:
        if [group["occurrence_id"] for group in groups] != [row["occurrence_id"] for row in expected_rows] or [group["id"] for group in groups] != [row["id"] for row in expected_rows]:
            raise RuntimeError("Native generation did not follow every scheduled occurrence exactly once")
        write_json(run_dir / "sampling_checks.json", {"status": "passed", "presentations": len(groups), "unique_questions": len({group["id"] for group in groups}), "unique_occurrences": len({group["occurrence_id"] for group in groups}), "candidates": len(groups) * 4, "optimizer_steps": trainer.state.global_step, "exact_schedule_order": True, "group_size": 4, "num_iterations": 1, "smoke_prefix_only": smoke})
        write_json(run_dir / "sampling_pool_summary.json", summarize_pools(groups))
    changed = [name for name, parameter in trainable.items() if not torch.equal(initial[name], parameter.detach().cpu())]
    mixed = sum(0 < group["correct_candidates"] < 4 for group in groups)
    summary = {"status": "complete", "optimizer_steps": trainer.state.global_step, "planned_optimizer_steps": options["max_steps"], "questions_sampled": len(groups), "candidates_generated": len(groups) * 4, "correct_candidates_histogram": dict(collections.Counter(group["correct_candidates"] for group in groups)), "mixed_reward_groups": mixed, "candidate_mean_reward": sum(group["correct_candidates"] for group in groups) / (4 * len(groups)), "sampled_solution_coverage": sum(group["correct_candidates"] > 0 for group in groups) / len(groups), "mean_distinct_outputs_per_group": sum(group["distinct_outputs"] for group in groups) / len(groups), "trainable_parameters": count, "adapter_tensors_changed": len(changed), "training_seconds": training_seconds, "training_metrics": result.metrics, "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "gpu_name": torch.cuda.get_device_name(0), "source_sft_run": config["source_sft_run"], "smoke": smoke}
    if static:
        summary.update(data_selection="static_mixed_v1", question_presentations=len(groups), unique_questions=len({group["id"] for group in groups}), data_sha256=config["data_sha256"], sampling_pools=summarize_pools(groups))
    if trainer.state.global_step != options["max_steps"] or not changed or (smoke and not mixed):
        write_json(run_dir / "training_summary.json", {**summary, "status": "failed_learning_signal_check"})
        raise RuntimeError("RL smoke/update failed: insufficient mixed rewards, missing updates or incomplete training")
    if not smoke:
        required = (model_path / "model.safetensors").stat().st_size + 0.25 * 2**30
        if shutil.disk_usage(run_dir).free < required:
            raise RuntimeError("Adapter saved but insufficient space to safely export the final RL model")
        merged = trainer.model.merge_and_unload(safe_merge=True)
        if merged.config.model_type != "a2d-qwen3":
            raise ValueError("Final RL model must retain its diffusion architecture")
        merged.save_pretrained(run_dir / "final_model", safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(run_dir / "final_model")
        write_json(run_dir / "final_model/source_training.json", {"source_sft_run": config["source_sft_run"], "base_model": str(model_path), "rl_adapter_sha256": hashlib.sha256((run_dir / "final_adapter/adapter_model.safetensors").read_bytes()).hexdigest(), "algorithm": "diffu-GRPO", "merged_diffusion_model": True})
    summary["total_seconds"] = time.perf_counter() - started
    summary["free_persistent_bytes"] = shutil.disk_usage(run_dir).free
    write_json(run_dir / "training_summary.json", summary)
    if smoke:
        from safetensors.torch import load_file
        import zipfile

        checkpoint = run_dir / "checkpoints/checkpoint-1"
        for name in ["pipeline_complete.json", "trainer_state.json", "adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"]:
            if not (checkpoint / name).is_file():
                raise RuntimeError("Incomplete native RL checkpoint: " + name)
        for name in ["optimizer.pt", "scheduler.pt", "rng_state.pth"]:
            with zipfile.ZipFile(checkpoint / name) as archive:
                if archive.testzip() is not None:
                    raise RuntimeError("Corrupt checkpoint state: " + name)
        saved = load_file(str(run_dir / "final_adapter/adapter_model.safetensors"))
        restored = load_file(str(checkpoint / "adapter_model.safetensors"))
        if saved.keys() != restored.keys() or any(not torch.equal(value, restored[name]) for name, value in saved.items()):
            raise RuntimeError("Final adapter and native checkpoint differ")
        write_json(run_dir / "smoke_checks.json", {"status": "passed", "native_diffu_grpo_used": True, "binary_outcome_reward_only": True, "reward_uses_frozen_pre_eos_scoring": True, "mixed_reward_groups": mixed, "adapter_tensors_changed": len(changed), "checkpoint_optimizer_scheduler_rng_valid": True, "saved_adapter_matches_checkpoint": True, "peak_allocated_bytes": summary["peak_allocated_bytes"], "peak_reserved_bytes": summary["peak_reserved_bytes"], "formal_training_not_started_by_smoke": True})


def summarize_pools(groups):
    result = {}
    for name, selected in [("priority", [row for row in groups if row["pool"] == "priority"]), ("exploration", [row for row in groups if row["pool"] == "exploration"]), ("priority_first", [row for row in groups if row["pool"] == "priority" and row["presentation_number"] == 1]), ("priority_second", [row for row in groups if row["pool"] == "priority" and row["presentation_number"] == 2])]:
        count = len(selected)
        mixed = sum(0 < row["correct_candidates"] < 4 for row in selected)
        result[name] = {"presentations": count, "unique_questions": len({row["id"] for row in selected}), "all_wrong_groups": sum(row["correct_candidates"] == 0 for row in selected), "all_correct_groups": sum(row["correct_candidates"] == 4 for row in selected), "mixed_groups": mixed, "mixed_group_rate": mixed / count if count else None, "candidate_mean_reward": sum(row["correct_candidates"] for row in selected) / (4 * count) if count else None, "correct_candidates_histogram": dict(collections.Counter(row["correct_candidates"] for row in selected))}
    return result


def reuse_reference_evaluations(run_dir, config):
    fixed = read_rows(PROJECT / "data/gsm8k/splits/validation.jsonl")
    protocol = read_json(PROJECT / config["evaluation_protocol"])
    checks = {}
    for name, reference in config["reference_evaluations"].items():
        source = PROJECT / reference["path"]
        for filename, expected in reference["files_sha256"].items():
            if hashlib.sha256((source / filename).read_bytes()).hexdigest() != expected:
                raise ValueError("Frozen development reference changed: " + name + "/" + filename)
        summary = read_json(source / "summary.json")
        run_config = read_json(source / "run_config.json")
        predictions = read_rows(source / "predictions.jsonl")
        if summary.get("status") != "complete" or summary.get("examples") != 500 or len(predictions) != 500 or summary["protocol_sha256"] != config["evaluation_protocol_sha256"]:
            raise ValueError("Incomplete reference evaluation: " + name)
        if run_config["protocol"] != protocol or run_config["split"] != "validation" or run_config["limit"] is not None:
            raise ValueError("Reference development protocol differs")
        correct = sum(row["scores"]["flexible-extract"]["correct"] for row in predictions)
        if correct != reference["expected_correct"] or summary["correct"] != correct or summary["accuracy"] != correct / 500:
            raise ValueError("Reference development counts differ")
        for actual, expected in zip(predictions, fixed):
            if actual["id"] != expected["id"] or actual["question"] != expected["question"] or actual["reference_answer"] != expected["answer_value"]:
                raise ValueError("Reference evaluation does not match the fixed development split")
        destination = run_dir / name
        destination.mkdir(exist_ok=False)
        for filename in reference["files_sha256"]:
            shutil.copy2(source / filename, destination / filename)
        checks[name] = {"reused": True, "correct": correct, "source": reference, "historical_process_seconds_excluded_from_new_cost": summary["total_process_seconds"]}
    write_json(run_dir / "reference_checks.json", {"status": "passed", "protocol_sha256": config["evaluation_protocol_sha256"], "evaluations": checks, "no_models_rerun": True})


def paired_development(before, after):
    if len(before) != 500 or len(after) != 500:
        raise ValueError("Both development results must contain exactly 500 questions")
    transitions = collections.Counter()
    deltas = []
    for left, right in zip(before, after):
        for key in ["id", "question", "prompt", "reference_answer", "model_input_text", "input_token_sha256"]:
            if left[key] != right[key]:
                raise ValueError("Development comparison is not paired: " + key)
        old = bool(left["scores"]["flexible-extract"]["correct"])
        new = bool(right["scores"]["flexible-extract"]["correct"])
        transitions[(old, new)] += 1
        deltas.append(int(new) - int(old))
    improved, regressed = transitions[(False, True)], transitions[(True, False)]
    discordant = improved + regressed
    exact_p = min(1.0, 2 * sum(math.comb(discordant, value) for value in range(min(improved, regressed) + 1)) / 2**discordant) if discordant else 1.0
    rng = random.Random(20261105)
    differences = sorted(sum(rng.choices(deltas, k=500)) / 5 for _ in range(4000))
    return {"before_correct": transitions[(True, True)] + regressed, "after_correct": transitions[(True, True)] + improved, "wrong_to_right": improved, "right_to_wrong": regressed, "both_correct": transitions[(True, True)], "both_wrong": transitions[(False, False)], "accuracy_delta_percentage_points": (improved - regressed) / 5, "paired_exact_p": exact_p, "paired_bootstrap_delta_ci95_percentage_points": [differences[100], differences[3900]], "bootstrap_repeats": 4000, "bootstrap_seed": 20261105}


def compare_static_development(run_dir, config):
    after = read_rows(run_dir / "development_rl/predictions.jsonl")
    latest = read_json(run_dir / "development_rl/summary.json")
    latest_config = read_json(run_dir / "development_rl/run_config.json")
    comparisons = {}
    for name in ["development_sft", "development_previous_rl"]:
        comparisons[name] = paired_development(read_rows(run_dir / name / "predictions.jsonl"), after)
    correct = comparisons["development_previous_rl"]["after_correct"]
    if latest["protocol_sha256"] != config["evaluation_protocol_sha256"] or latest_config["protocol"] != read_json(PROJECT / config["evaluation_protocol"]) or latest_config["split"] != "validation" or latest_config["limit"] is not None or latest["correct"] != correct or latest["accuracy"] != correct / 500:
        raise ValueError("New development summary or protocol differs from the frozen evaluation")
    training = read_json(run_dir / "training_summary.json")
    seconds = training["total_seconds"] + latest["total_process_seconds"]
    write_json(run_dir / "development_comparison.json", {"status": "complete", "development_questions": 500, "sft_correct": comparisons["development_sft"]["before_correct"], "previous_rl_correct": comparisons["development_previous_rl"]["before_correct"], "rl_correct": correct, "rl_accuracy": correct / 500, "comparisons": comparisons, "training_summary": training, "mixed_group_rate": training["mixed_reward_groups"] / training["question_presentations"], "preregistered_targets": {"mixed_group_rate": 0.35, "development_correct": 162, "mixed_group_target_met": training["mixed_reward_groups"] / training["question_presentations"] >= 0.35, "development_target_met": correct >= 162, "targets_do_not_control_stopping_or_selection": True}, "estimated_total_process_seconds": seconds, "estimated_process_cost_cny": seconds / 3600 * config["gpu_price_cny_per_hour"], "historical_evaluations_excluded_from_cost": True, "cost_excludes_instance_idle_time_and_separate_smoke": True, "official_test_evaluated": False, "rl_generation_summary": latest})


def compare_development(run_dir, config):
    if config.get("data_selection") == "static_mixed_v1":
        compare_static_development(run_dir, config)
        return
    before = read_rows(run_dir / "development_sft/predictions.jsonl")
    after = read_rows(run_dir / "development_rl/predictions.jsonl")
    if len(before) != 500 or len(after) != 500:
        raise ValueError("Both development evaluations must contain exactly 500 questions")
    transitions = collections.Counter()
    for left, right in zip(before, after):
        for key in ["id", "question", "reference_answer", "input_token_sha256"]:
            if left[key] != right[key]:
                raise ValueError("Development comparison is not paired under the same prompts")
        transitions[(left["scores"]["flexible-extract"]["correct"], right["scores"]["flexible-extract"]["correct"]) ] += 1
    sft = read_json(run_dir / "development_sft/summary.json")
    rl = read_json(run_dir / "development_rl/summary.json")
    improved, regressed = transitions[(False, True)], transitions[(True, False)]
    discordant = improved + regressed
    exact_p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1)) / 2**discordant) if discordant else 1.0
    training = read_json(run_dir / "training_summary.json")
    seconds = training["total_seconds"] + sft["total_process_seconds"] + rl["total_process_seconds"]
    write_json(run_dir / "development_comparison.json", {"status": "complete", "development_questions": 500, "sft_correct": sft["correct"], "rl_correct": rl["correct"], "sft_accuracy": sft["accuracy"], "rl_accuracy": rl["accuracy"], "accuracy_delta_percentage_points": 100 * (rl["accuracy"] - sft["accuracy"]), "wrong_to_right": improved, "right_to_wrong": regressed, "both_correct": transitions[(True, True)], "both_wrong": transitions[(False, False)], "paired_exact_p": exact_p, "training_summary": training, "estimated_total_process_seconds": seconds, "estimated_process_cost_cny": seconds / 3600 * config["gpu_price_cny_per_hour"], "cost_excludes_instance_idle_time": True, "official_test_evaluated": False, "sft_generation_summary": sft, "rl_generation_summary": rl})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("BLUEWHALE_RUN_DIR") or args.run_dir.resolve() != Path(os.environ["BLUEWHALE_RUN_DIR"]).resolve():
        raise ValueError("Launch using the existing Pipeline")
    config = read_json(args.config)
    if args.train_only:
        train(config, args.run_dir, args.smoke)
        return
    # 每个阶段的子进程退出后释放显存，成功后才启动下一个阶段。
    child = None
    stopping = False

    def remember_stop(number, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(number)

    signal.signal(signal.SIGTERM, remember_stop)
    signal.signal(signal.SIGINT, remember_stop)

    def execute(phase, command):
        nonlocal child
        if stopping:
            raise RuntimeError("Stop requested; do not start another phase")
        write_json(args.run_dir / "phase.json", {"phase": phase, "started_unix_seconds": time.time()})
        print(json.dumps({"phase": phase, "command": command}), flush=True)
        child = subprocess.Popen(command)
        code = child.wait()
        if code or stopping:
            raise RuntimeError("Phase did not complete: " + phase + ", exit=" + str(code))

    training_command = [sys.executable, "-u", str(Path(__file__).resolve()), "--config", str(args.config), "--run-dir", str(args.run_dir), "--train-only"]
    if args.smoke:
        execute("rl_smoke", training_command + ["--smoke"])
        return
    write_json(args.run_dir / "rl_config.json", config)
    shutil.copy2(__file__, args.run_dir / "code/train_rl.py")
    evaluator = PROJECT / "scripts/evaluate_model.py"
    if config.get("data_selection") == "static_mixed_v1":
        load_data(config)
        reuse_reference_evaluations(args.run_dir, config)
        execute("rl_training", training_command)
        execute("development_rl", [sys.executable, "-u", str(evaluator), "--checkpoint", str(args.run_dir / "final_model"), "--split", "validation", "--protocol", str(PROJECT / config["evaluation_protocol"]), "--output", str(args.run_dir / "development_rl")])
        summary = read_json(args.run_dir / "development_rl/summary.json")
        if summary.get("status") != "complete" or summary.get("examples") != 500:
            raise RuntimeError("Incomplete static RL development evaluation")
        compare_development(args.run_dir, config)
        write_json(args.run_dir / "phase.json", {"phase": "complete", "finished_unix_seconds": time.time()})
        return
    for name, checkpoint in [("development_sft", PROJECT / config["model_path"]), ("development_rl", args.run_dir / "final_model")]:
        if name == "development_rl":
            execute("rl_training", training_command)
        execute(name, [sys.executable, "-u", str(evaluator), "--checkpoint", str(checkpoint), "--split", "validation", "--protocol", str(PROJECT / config["evaluation_protocol"]), "--output", str(args.run_dir / name)])
        summary = read_json(args.run_dir / name / "summary.json")
        if summary.get("status") != "complete" or summary.get("examples") != 500:
            raise RuntimeError("Incomplete development evaluation: " + name)
    compare_development(args.run_dir, config)
    write_json(args.run_dir / "phase.json", {"phase": "complete", "finished_unix_seconds": time.time()})


if __name__ == "__main__":
    main()
