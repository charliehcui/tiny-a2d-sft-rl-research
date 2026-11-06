"""候选复用的独立训练入口；默认只允许固定日程前两周期的程序短测。"""

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from prepare_rl_reuse import CONFIG_PATH, PROJECT, load_reuse_rows
from prepare_rl_static import digest, json_bytes, write_new
from run_experiment import read_json, write_json
from train_rl import OutcomeReward
from rl_storage import ADAPTER_LIMIT, CHECKPOINT_LIMIT, check_log_budget, copy_verified_model, files_in, require_space, rotate_checkpoints, seal_checkpoint, verify_checkpoint


def validate_resume(checkpoint, config, schedule_sha256):
    checkpoint = Path(checkpoint).resolve()
    for name in ["pipeline_complete.json", "reuse_state.json", "trainer_state.json", "adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"]:
        if not (checkpoint / name).is_file():
            raise ValueError("Incomplete reuse checkpoint: " + name)
    saved = read_json(checkpoint / "reuse_state.json")
    if config.get("checkpoint_rotation_enabled"):
        marker = read_json(checkpoint / "checkpoint_verified.json")
        verify_checkpoint(checkpoint, marker["family_root"])
    state = read_json(checkpoint / "trainer_state.json")
    if not saved["safe_cycle_boundary"] or saved["global_step"] % 4 or saved["global_step"] != state["global_step"]:
        raise ValueError("Resume requires a completed four-update cycle")
    if saved["config_sha256"] != hashlib.sha256(json_bytes(config)).hexdigest() or saved["schedule_sha256"] != schedule_sha256:
        raise ValueError("Resume configuration or full frozen schedule differs")
    current_code = {name: digest(PROJECT / "scripts" / name) for name in saved["code_sha256"]}
    if current_code != saved["code_sha256"]:
        raise ValueError("Resume implementation changed since checkpoint creation")
    return saved


def require_mode(config, smoke):
    if not smoke and not config.get("formal_training_enabled", False):
        raise PermissionError("Full 1536-question training is disabled; explicit user authorization is still required")


def train_segment(config, run_dir, smoke, resume=None, stop_after_step=None):
    import shutil
    import torch
    import transformers
    import dllm
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig
    from dllm.pipelines.rl import DiffuGRPOConfig
    from experiment_callback import ExperimentCallback
    from rl_reuse_trainer import ReuseTrainer, tensor_digest

    require_mode(config, smoke)
    family_root = Path(os.environ.get("BLUEWHALE_WORKFLOW_DIR", str(run_dir))).resolve()
    if not run_dir.resolve().is_relative_to(family_root):
        raise ValueError("Segment must remain inside its checkpoint family")
    for name in ["checkpoints", "final_adapter", "final_model", "rollouts.jsonl", "metrics.jsonl", "training_summary.json"]:
        if (run_dir / name).exists():
            raise FileExistsError("Use a new segment directory; refusing to overwrite " + name)
    minimum = config["smoke_minimum_free_gib"] if smoke else config["minimum_free_gib"]
    if shutil.disk_usage(run_dir).free < minimum * 2**30:
        raise RuntimeError("Insufficient persistent space for this run")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This entry requires a BF16 CUDA GPU")
    if digest(PROJECT / config["evaluation_protocol"]) != config["evaluation_protocol_sha256"]:
        raise ValueError("Frozen evaluation protocol changed")
    schedule = load_reuse_rows(PROJECT, config)
    rows = schedule[:32] if smoke else schedule
    restored = validate_resume(resume, config, config["data_sha256"]) if resume else None
    initial_step = restored["global_step"] if restored else 0
    committed_questions = initial_step // 4 * 16
    if initial_step >= (8 if smoke else 384):
        raise ValueError("Checkpoint already completed this fixed run budget")
    options = dict(config["training"])
    if smoke:
        options.update(max_steps=8, save_steps=4)
    options["output_dir"] = str(run_dir / "checkpoints")
    arguments = DiffuGRPOConfig(**options)
    transformers.set_seed(arguments.seed)
    torch.set_num_threads(4)
    model_path = PROJECT / config["model_path"]
    provenance = read_json(model_path / "source_training.json")
    if provenance["adapter_sha256"] != config["source_adapter_sha256"] or not provenance["merged_diffusion_model"]:
        raise ValueError("Only the original merged SFT626 starting point is allowed")
    model_arguments = dllm.utils.ModelArguments(model_name_or_path=str(model_path), dtype="bfloat16", attn_implementation="sdpa", lora=False, load_in_4bit=False)
    tokenizer = dllm.utils.get_tokenizer(model_arguments)
    model = dllm.utils.get_model(model_arguments)
    model.config.use_cache = False
    sampler = MDLMSampler(model=model, tokenizer=tokenizer)
    sampler_config = MDLMSamplerConfig(steps=256, max_new_tokens=256, block_size=256, temperature=1.0, cfg_scale=0.0, remasking="low_confidence")
    reward = OutcomeReward(tokenizer, run_dir, expected_rows=rows[committed_questions:])
    trainer = ReuseTrainer(model=model, reward_funcs=[reward], args=arguments, train_dataset=Dataset.from_list(rows), processing_class=tokenizer, peft_config=LoraConfig(**config["lora"]), sampler=sampler, sampler_config=sampler_config, physical_generation_batch_size=config["physical_generation_batch_size"])
    reward.trainer = trainer
    if trainer.ref_model is not None or not isinstance(trainer.model, PeftModel):
        raise ValueError("The only reference must be SFT626 with the new RL adapter disabled")
    trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer.model.enable_input_require_grads()
    trainer.model.eval()
    probe = tokenizer.apply_chat_template(rows[0]["prompt"], tokenize=True, add_generation_prompt=True, return_tensors="pt").to(trainer.accelerator.device)
    with torch.inference_mode():
        enabled = trainer.model(probe).logits[:, -1, :].float()
        with trainer.model.disable_adapter():
            disabled = trainer.model(probe).logits[:, -1, :].float()
        difference = (enabled - disabled).abs().max().item()
    if difference != 0.0:
        raise RuntimeError("Fresh RL adapter changes the frozen SFT reference before training")
    del probe, enabled, disabled
    torch.cuda.empty_cache()
    write_new(run_dir / "model_checks.json", json_bytes({"source_sft_run": config["source_sft_run"], "fresh_adapter_vs_sft_max_abs_logit_difference": difference, "reference_is_merged_sft626": True, "logical_batch": 64, "physical_generation_batch": 4}))
    trainable = {name: parameter for name, parameter in trainer.model.named_parameters() if parameter.requires_grad}
    if sum(parameter.numel() for parameter in trainable.values()) != config["expected_lora_parameters"] or any("lora_" not in name for name in trainable):
        raise ValueError("Only the registered LoRA parameters may train")
    initial = {name: value.detach().cpu().clone() for name, value in trainable.items()}
    base_version = {name: tensor_digest(value) for name, value in trainer.model.named_parameters() if not value.requires_grad}
    code_names = ["train_rl_reuse.py", "rl_reuse_trainer.py", "prepare_rl_reuse.py", "train_rl.py", "prepare_rl_static.py", "rl_storage.py"]
    code_sha256 = {name: digest(PROJECT / "scripts" / name) for name in code_names}
    metadata = {"config_sha256": hashlib.sha256(json_bytes(config)).hexdigest(), "schedule_sha256": config["data_sha256"], "code_sha256": code_sha256, "smoke": smoke}
    write_new(run_dir / "rl_config.json", json_bytes({**config, "actual_training_options": options, "smoke": smoke}))
    write_new(run_dir / "sampling_schedule.jsonl", ("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)).encode("utf-8"))

    class ReuseCallback(ExperimentCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if state.global_step != initial_step:
                raise RuntimeError("Trainer did not restore the expected global optimizer step")
            if resume:
                from safetensors.torch import load_file

                saved = load_file(str(Path(resume) / "adapter_model.safetensors"))
                from peft import get_peft_model_state_dict

                current = get_peft_model_state_dict(trainer.model)
                if saved.keys() != current.keys() or any(not torch.equal(value, current[name].detach().cpu()) for name, value in saved.items()):
                    raise RuntimeError("Resume did not restore the saved adapter before generation")
                optimizer_steps = [int(item["step"]) for item in trainer.optimizer.state.values() if "step" in item]
                if not optimizer_steps or set(optimizer_steps) != {initial_step}:
                    raise RuntimeError("Resume did not restore AdamW step counters")
                write_new(run_dir / "resume_checks.json", json_bytes({"status": "passed", "adapter_equal_before_generation": True, "optimizer_step_counters": sorted(set(optimizer_steps)), "global_step": initial_step, "committed_questions": committed_questions, "native_rng_restore_requested": True}))
            self.previous_adapter_hash = self.adapter_hash()
            self.update_evidence = []
            return super().on_train_begin(args, state, control, **kwargs)

        def adapter_hash(self):
            return hashlib.sha256("".join(tensor_digest(value) for value in trainable.values()).encode()).hexdigest()

        def on_step_end(self, args, state, control, **kwargs):
            if smoke and state.global_step > initial_step:
                current_hash = self.adapter_hash()
                optimizer_steps = {int(item["step"]) for item in trainer.optimizer.state.values() if "step" in item}
                changed_now = current_hash != self.previous_adapter_hash
                if optimizer_steps != {state.global_step} or not changed_now:
                    raise RuntimeError("A smoke optimizer update did not advance AdamW state and adapter parameters")
                self.update_evidence.append({"step": state.global_step, "adapter_sha256": current_hash, "changed_from_previous_update": changed_now, "adamw_step_counters": sorted(optimizer_steps)})
                self.previous_adapter_hash = current_hash
            if stop_after_step is not None and state.global_step >= stop_after_step:
                self.stop_requested = True
            deadline = os.environ.get("BLUEWHALE_TASK_DEADLINE")
            if deadline and time.time() >= float(deadline) - 900:
                self.stop_requested = True
            if not smoke and state.global_step == 32:
                predicted = (time.perf_counter() - started) / max(1, state.global_step - initial_step) * (384 - initial_step)
                write_json(run_dir / "runtime_budget_check.json", {"step": 32, "predicted_training_seconds": predicted, "maximum_training_seconds": 16 * 3600, "passed": predicted <= 16 * 3600})
                if predicted > 16 * 3600:
                    self.stop_requested = True
            if state.global_step > initial_step and state.global_step % 4 == 0:
                check_log_budget(family_root)
                if state.global_step % int(args.save_steps) == 0 or self.stop_requested:
                    require_space(family_root, CHECKPOINT_LIMIT)
            write_json(self.run_dir / "training_progress.json", {"completed": state.global_step, "initial_step": initial_step, "total": state.max_steps, "unit": "step", "epoch": state.epoch, "stop_requested": self.stop_requested, "updated_unix_seconds": time.time()})
            if self.stop_requested and state.global_step > initial_step and state.global_step % 4 == 0:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if any(isinstance(value, (float, int)) and not math.isfinite(value) for value in (logs or {}).values()):
                raise FloatingPointError("Non-finite training metric; stop smoke")
            return super().on_log(args, state, control, logs, **kwargs)

        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / ("checkpoint-" + str(state.global_step))
            write_new(checkpoint / "reuse_state.json", json_bytes({**trainer.reuse_state(), **metadata}))
            seal_checkpoint(checkpoint, family_root)
            if config.get("checkpoint_rotation_enabled"):
                rotate_checkpoints(family_root, checkpoint)
            return control

    callback = ReuseCallback()
    trainer.add_callback(callback)
    (run_dir / "code").mkdir(exist_ok=True)
    for name in code_names:
        write_new(run_dir / "code" / name, (PROJECT / "scripts" / name).read_bytes())
    monitor_output = (run_dir / "training_gpu_resources.csv").open("x")
    monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu", "--format=csv,noheader,nounits", "-l", "1"], stdout=monitor_output, stderr=subprocess.DEVNULL)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        monitor_output.close()
    trainer.save_state()
    state = trainer.reuse_state()
    complete = state["global_step"] == options["max_steps"]
    if complete or smoke:
        trainer.save_model(str(run_dir / "final_adapter"))
        tokenizer.save_pretrained(run_dir / "final_adapter")
        if sum(file.stat().st_size for file in files_in(run_dir / "final_adapter")) > ADAPTER_LIMIT:
            raise RuntimeError("Final adapter exceeds its 48 MiB budget")
    expected_groups = (state["global_step"] - initial_step) // 4 * 16
    if len(reward.groups) != expected_groups or [group["id"] for group in reward.groups] != [row["id"] for row in rows[committed_questions:committed_questions + expected_groups]]:
        raise RuntimeError("Actual new groups differ from the committed schedule segment")
    changed = [name for name, value in trainable.items() if not torch.equal(initial[name], value.detach().cpu())]
    if not changed:
        raise RuntimeError("No LoRA parameters changed")
    if any(tensor_digest(value) != base_version[name] for name, value in trainer.model.named_parameters() if not value.requires_grad):
        raise RuntimeError("The frozen SFT base changed")
    ratios = {str(step): {**item, "mean_abs_log_ratio": item["abs_log_ratio_sum"] / max(1, item["tokens"])} for step, item in trainer.reuse_ratio_sums.items()}
    if any(item["nonfinite"] for item in ratios.values()):
        raise FloatingPointError("Non-finite policy ratio")
    summary = {"status": "complete" if complete else "stopped_at_cycle_boundary", **state, "initial_step": initial_step, "new_questions": expected_groups, "new_candidates": expected_groups * 4, "physical_generation_calls": trainer.reuse_generation_calls, "four_real_updates_per_cycle": True, "old_probability_cache_unchanged": True, "frozen_sft_parameters_unchanged": True, "adapter_tensors_changed": len(changed), "training_seconds": time.perf_counter() - started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "training_metrics": result.metrics, "ratios_by_step_before_update": ratios, "generation_events": trainer.reuse_events, "optimizer_update_evidence": callback.update_evidence, "smoke": smoke, "resume_from": str(resume) if resume else None, "verified_checkpoint_rotation_enabled": config.get("checkpoint_rotation_enabled", False), "evaluation_performed": False}
    write_new(run_dir / "training_summary.json", json_bytes(summary))
    if complete and not smoke:
        require_space(run_dir, 1216 * 2**20)
        verified = [path for path in family_root.rglob("checkpoint-*") if path.is_dir() and (path / "checkpoint_verified.json").is_file()]
        all_checkpoints = [path for path in family_root.rglob("checkpoint-*") if path.is_dir() and path.parent.name == "checkpoints"]
        if len(verified) != 2 or len(all_checkpoints) - len(verified) > 1:
            raise RuntimeError("Checkpoint rotation must finish before final export")
        export_root = Path("/opt/bluewhale_cache/exports") / family_root.name
        export_root.mkdir(parents=True, exist_ok=False)
        if export_root.resolve().is_relative_to(PROJECT.resolve()) or shutil.disk_usage(export_root).free < 4 * 2**30:
            raise RuntimeError("A separate instance disk with 4 GiB free is required for export")
        merged = trainer.model.merge_and_unload(safe_merge=True)
        if merged.config.model_type != "a2d-qwen3":
            raise RuntimeError("Final model lost the registered diffusion architecture")
        merged.save_pretrained(export_root / "final_model", safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(export_root / "final_model")
        write_new(export_root / "final_model/source_training.json", json_bytes({"algorithm": "diffu-GRPO with rollout reuse", "source_sft_run": config["source_sft_run"], "rl_adapter_sha256": digest(run_dir / "final_adapter/adapter_model.safetensors"), "merged_diffusion_model": True}))
        copy_verified_model(export_root / "final_model", run_dir / "final_model")
        if export_root.resolve().parent != Path("/opt/bluewhale_cache/exports").resolve():
            raise RuntimeError("Temporary export path escaped its allowed directory")
        shutil.rmtree(export_root)
    if not complete and stop_after_step is None:
        raise RuntimeError("Training interrupted at a complete cycle; resume the preserved checkpoint in a new segment")
    return summary


def run_smoke(config_path, run_dir):
    code = Path(__file__).resolve()
    first = [sys.executable, "-u", str(code), "--config", str(config_path), "--run-dir", str(run_dir), "--smoke", "--train-only", "--stop-after-step", "4"]
    subprocess.run(first, check=True)
    checkpoint = run_dir / "checkpoints/checkpoint-4"
    resumed_dir = run_dir / "resumed"
    resumed_dir.mkdir(exist_ok=False)
    environment = os.environ.copy()
    environment["BLUEWHALE_RUN_DIR"] = str(resumed_dir)
    second = [sys.executable, "-u", str(code), "--config", str(config_path), "--run-dir", str(resumed_dir), "--smoke", "--train-only", "--resume", str(checkpoint)]
    subprocess.run(second, env=environment, check=True)
    before = read_json(run_dir / "training_summary.json")
    after = read_json(resumed_dir / "training_summary.json")
    restored = read_json(resumed_dir / "resume_checks.json")
    if before["global_step"] != 4 or after["global_step"] != 8 or before["new_candidates"] + after["new_candidates"] != 128 or restored["global_step"] != 4:
        raise RuntimeError("Smoke/resume counts differ from two registered cycles")
    later_ratios = [item["max_abs_ratio_delta"] for segment in [before, after] for step, item in segment["ratios_by_step_before_update"].items() if int(step) % 4 > 0]
    if not later_ratios or max(later_ratios) == 0:
        raise RuntimeError("Later updates show no actual old/current probability-ratio change")
    checks = {"status": "passed", "smoke_only": True, "fresh_questions": 32, "fresh_candidates": 128, "rollout_cycles": 2, "optimizer_updates": 8, "four_real_updates_per_cycle": True, "frozen_old_probabilities": True, "four_correctly_indexed_mask_views": True, "distinct_mask_seeds": True, "resume_actually_continued_from_step_4_to_8": True, "maximum_later_ratio_delta": max(later_ratios), "peak_allocated_bytes": max(before["peak_allocated_bytes"], after["peak_allocated_bytes"]), "peak_reserved_bytes": max(before["peak_reserved_bytes"], after["peak_reserved_bytes"]), "no_model_evaluation": True, "no_deletions": True}
    if checks["peak_reserved_bytes"] >= 24 * 2**30:
        raise RuntimeError("Smoke memory exceeds 24 GiB")
    write_new(run_dir / "reuse_smoke_checks.json", json_bytes(checks))
    print(json.dumps(checks), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / CONFIG_PATH)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    arguments = parser.parse_args()
    if arguments.run_dir.resolve() != Path(os.environ.get("BLUEWHALE_RUN_DIR", "")).resolve():
        raise ValueError("Launch through the existing detached Pipeline")
    config = read_json(arguments.config)
    require_mode(config, arguments.smoke)
    if arguments.train_only:
        train_segment(config, arguments.run_dir, arguments.smoke, arguments.resume, arguments.stop_after_step)
    elif arguments.smoke:
        run_smoke(arguments.config, arguments.run_dir)
    else:
        train_segment(config, arguments.run_dir, False, arguments.resume, arguments.stop_after_step)


if __name__ == "__main__":
    main()
