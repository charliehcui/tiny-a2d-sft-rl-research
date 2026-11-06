"""把固定 GSM8K 数据及 LoRA 配置接入官方 MDLMTrainer；不实现新的训练目标。"""

import argparse
import hashlib
import json
import math
import os
import shutil
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


def planned_steps(config):
    training = config["training"]
    batches = math.ceil(config["data"]["train_rows"] / training["per_device_train_batch_size"])
    if training.get("max_steps", -1) > 0:
        return training["max_steps"]
    return math.ceil(math.ceil(batches / training["gradient_accumulation_steps"]) * training["num_train_epochs"])


def load_fixed_rows(config, split):
    path = PROJECT / config["data"][split]
    if hashlib.sha256(path.read_bytes()).hexdigest() != config["data"][split + "_sha256"]:
        raise ValueError("Fixed dataset checksum mismatch: " + split)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != config["data"][split + "_rows"] or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Fixed dataset count/ID mismatch: " + split)
    return rows


def tokenized_rows(rows, tokenizer, maximum):
    import dllm

    prepared = []
    lengths = []
    for row in rows:
        mapped = dllm.utils.default_sft_map_fn(row, tokenizer=tokenizer, mask_prompt_loss=True)
        prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=True, add_generation_prompt=True)
        if mapped["input_ids"][:len(prompt)] != prompt:
            raise ValueError("Training prompt differs from evaluation template: " + row["id"])
        length = len(mapped["input_ids"])
        if length > maximum:
            raise ValueError("Refusing to truncate or drop a fixed example: " + row["id"])
        if mapped["labels"][:len(prompt)] != [-100] * len(prompt) or all(label == -100 for label in mapped["labels"]):
            raise ValueError("Invalid response supervision: " + row["id"])
        prepared.append({"input_ids": mapped["input_ids"], "labels": mapped["labels"]})
        lengths.append(length)
    return prepared, {"rows": len(rows), "max_tokens": max(lengths), "mean_tokens": sum(lengths) / len(lengths), "truncated": 0}


def validate_resume(path, config):
    path = path.resolve()
    for name in ("pipeline_complete.json", "trainer_state.json", "adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        if not (path / name).is_file():
            raise ValueError("Resume requires a completed native checkpoint; missing " + name)
    if read_json(path.parent.parent / "sft_config.json") != config:
        raise ValueError("Resume configuration must match the original training run")
    return str(path)


def train(config, run_dir, resume_from=None):
    import torch
    import transformers
    import dllm
    from datasets import Dataset
    from dllm.core.trainers import MDLMConfig, MDLMTrainer
    from experiment_callback import ExperimentCallback

    started = time.perf_counter()
    minimum = config["storage"]["minimum_start_free_gib"]
    if shutil.disk_usage(run_dir).free < minimum * 2**30:
        raise RuntimeError("Not enough persistent space for LoRA checkpoints plus the merged final model")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires a BF16-capable CUDA GPU")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This configuration is for one GPU")
    protocol_path = PROJECT / config["evaluation"]["protocol"]
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != config["evaluation"]["protocol_sha256"]:
        raise ValueError("The frozen evaluation protocol changed")
    write_json(run_dir / "sft_config.json", config)
    write_json(run_dir / "evaluation_protocol.json", read_json(protocol_path))
    source_path = Path(__file__).resolve()
    shutil.copy2(source_path, run_dir / "code" / source_path.name)
    transformers.set_seed(config["training"]["seed"])
    torch.set_num_threads(4)
    training_options = dict(config["training"])
    training_options["output_dir"] = str(run_dir / "checkpoints")
    training_args = MDLMConfig(**training_options)
    model_options = dict(config["model"])
    model_options["model_name_or_path"] = str(PROJECT / model_options["model_name_or_path"])
    model_args = dllm.utils.ModelArguments(**model_options)
    data_args = dllm.utils.DataArguments(dataset_args=str(PROJECT / config["data"]["train"]), max_length=config["data"]["max_length"], num_proc=1)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)
    tokenizer = dllm.utils.get_tokenizer(model_args)
    raw_train = load_fixed_rows(config, "train")
    raw_validation = load_fixed_rows(config, "validation")
    if {row["id"] for row in raw_train} & {row["id"] for row in raw_validation}:
        raise ValueError("Training and validation overlap")
    train_rows, train_audit = tokenized_rows(raw_train, tokenizer, config["data"]["max_length"])
    validation_rows, validation_audit = tokenized_rows(raw_validation, tokenizer, config["data"]["max_length"])
    write_json(run_dir / "data_audit.json", {"train": train_audit, "validation": validation_audit, "planned_optimizer_steps": planned_steps(config)})
    train_data = Dataset.from_list(train_rows)
    validation_data = Dataset.from_list(validation_rows)
    model = dllm.utils.get_model(model_args)
    model.config.use_cache = False
    model.enable_input_require_grads()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    count = sum(parameter.numel() for name, parameter in trainable)
    if count != config["storage"]["expected_trainable_parameters"] or any("lora_" not in name for name, parameter in trainable):
        raise ValueError("Expected only the configured LoRA matrices to be trainable")
    write_json(run_dir / "trainable_parameters.json", {"parameters": count, "modules": len(trainable), "base_model_revision": config["model_revision"], "only_lora_trainable": True})
    trainer = MDLMTrainer(model=model, processing_class=tokenizer, train_dataset=train_data, eval_dataset=validation_data, args=training_args, data_collator=dllm.utils.NoAttentionMaskWrapper(transformers.DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True, pad_to_multiple_of=8, label_pad_token_id=tokenizer.pad_token_id)))
    # 官方 MDLM 损失已按本微批次归一化，并未使用 num_items_in_batch。
    # Transformers 4.57 要求此标志为 False，才会正确应用梯度累积缩放。
    trainer.model_accepts_loss_kwargs = False
    trainer.add_callback(ExperimentCallback())
    resume_path = validate_resume(resume_from, config) if resume_from else None
    torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=resume_path)
    training_seconds = time.perf_counter() - training_started
    trainer.save_state()
    trainer.save_model(str(run_dir / "final_adapter"))
    tokenizer.save_pretrained(run_dir / "final_adapter")
    adapter_path = run_dir / "final_adapter/adapter_model.safetensors"
    if adapter_path.stat().st_size > config["storage"]["adapter_file_limit_mib"] * 2**20:
        raise RuntimeError("Unexpected adapter size; refusing to export a full model without checking storage")
    final_loss = trainer.evaluate()
    trainer.save_metrics("train", result.metrics)
    trainer.save_metrics("eval", final_loss)
    complete = trainer.state.global_step == planned_steps(config)
    summary = {"status": "complete" if complete else "stopped_early", "optimizer_steps": trainer.state.global_step, "planned_optimizer_steps": planned_steps(config), "epochs": trainer.state.epoch, "trainable_parameters": count, "training_seconds": training_seconds, "training_metrics": result.metrics, "validation_metrics": final_loss, "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "base_model_revision": config["model_revision"], "resume_from": resume_path, "gpu_name": torch.cuda.get_device_name(0)}
    if complete and config["storage"]["persist_merged_final_model"]:
        base_bytes = (PROJECT / config["model"]["model_name_or_path"] / "model.safetensors").stat().st_size
        floor = config["storage"]["minimum_after_export_free_gib"] * 2**30
        if shutil.disk_usage(run_dir).free < base_bytes + floor:
            raise RuntimeError("Adapter is saved, but there is insufficient space to export a merged model safely")
        merged = model.merge_and_unload(safe_merge=True)
        merged.config.use_cache = False
        if merged.config.model_type != "a2d-qwen3":
            raise ValueError("The exported model must retain the diffusion architecture")
        merged.save_pretrained(run_dir / "final_model", safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(run_dir / "final_model")
        write_json(run_dir / "final_model/source_training.json", {"base_model_id": config["model_id"], "base_model_revision": config["model_revision"], "adapter_sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(), "merged_diffusion_model": True})
    summary["total_seconds"] = time.perf_counter() - started
    summary["free_persistent_bytes"] = shutil.disk_usage(run_dir).free
    write_json(run_dir / "training_summary.json", summary)
    if not complete:
        raise RuntimeError("Training stopped before the planned number of steps; saved state is available")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    expected = os.environ.get("BLUEWHALE_RUN_DIR")
    if not expected or args.run_dir.resolve() != Path(expected).resolve():
        raise ValueError("Use the existing Pipeline to create a fresh persistent run directory")
    train(read_json(args.config), args.run_dir, args.resume_from)


if __name__ == "__main__":
    main()
