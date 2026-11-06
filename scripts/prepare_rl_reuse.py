"""冻结候选复用方案的数据日程；不加载模型，不启动训练。"""

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

from prepare_rl_static import check_leakage, digest, json_bytes, read_rows, write_new
from run_experiment import read_json


PROJECT = Path(__file__).resolve().parents[1]
DATA_PATH = "data/gsm8k/processed/rl_reuse_1536.jsonl"
MANIFEST_PATH = "configs/rl_reuse_data_1536.json"
CONFIG_PATH = "configs/rl_reuse_tiny_a2d.json"


def make_schedule(source):
    if len(source) != 1973 or len({row["id"] for row in source}) != 1973:
        raise ValueError("Expected the original 1973 unique RL questions")
    selected = list(source)
    random.Random(42).shuffle(selected)
    result = []
    for index, row in enumerate(selected[:1536]):
        result.append({**row, "question_id": row["id"], "occurrence_id": "reuse-" + str(index).zfill(4), "presentation_index": index, "pool": "uniform", "presentation_number": 1, "rollout_cycle": index // 16, "scheduled_optimizer_step": 4 * (index // 16) + 1})
    return result


def load_reuse_rows(project, config):
    if digest(project / config["data_manifest"]) != config["data_manifest_sha256"]:
        raise ValueError("Reuse manifest changed")
    manifest = read_json(project / config["data_manifest"])
    for name, expected in manifest["input_sha256"].items():
        if digest(project / name) != expected:
            raise ValueError("Frozen reuse input changed: " + name)
    if digest(project / config["data_path"]) != config["data_sha256"]:
        raise ValueError("Reuse schedule changed")
    rows = read_rows(project / config["data_path"])
    if rows != make_schedule(read_rows(project / manifest["source"])):
        raise ValueError("Reuse schedule differs from Random(42) shuffled source prefix")
    baseline = read_json(project / "configs/rl_tiny_a2d.json")
    for name in ["source_sft_run", "model_path", "source_adapter_sha256", "dllm_revision", "lora", "reward_metric", "reward_values", "evaluation_protocol", "evaluation_protocol_sha256", "expected_lora_parameters"]:
        if config[name] != baseline[name]:
            raise ValueError("An approved fixed setting changed: " + name)
    changes = {"max_steps": 384, "num_iterations": 4, "generation_batch_size": 64, "save_steps": 16, "shuffle_dataset": False}
    if config["training"] != {**baseline["training"], **changes}:
        raise ValueError("Only the approved reuse scheduling settings may change")
    if config["physical_generation_batch_size"] != 4 or config["questions_per_cycle"] != 16:
        raise ValueError("Expected 16 questions per logical cycle and physical generation batch 4")
    if check_leakage(project, rows) != manifest["leakage_checks"]:
        raise ValueError("Frozen split overlap checks changed")
    return rows


def prepare(project):
    source_path = "data/gsm8k/processed/rl_train.jsonl"
    baseline_path = "configs/rl_tiny_a2d.json"
    rows = make_schedule(read_rows(project / source_path))
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    manifest = {"strategy": "uniform_reuse_v1", "source": source_path, "input_sha256": {source_path: digest(project / source_path), baseline_path: digest(project / baseline_path)}, "selection": "Original source order; independent Random(42).shuffle; first 1536", "questions": 1536, "group_size": 4, "new_candidates": 6144, "questions_per_cycle": 16, "rollout_cycles": 96, "updates_per_cycle": 4, "optimizer_steps": 384, "leakage_checks": check_leakage(project, rows), "test_used_only_for_id_and_normalized_question_overlap": True, "historical_rewards_used": False}
    manifest_data = json_bytes(manifest)
    config = copy.deepcopy(read_json(project / baseline_path))
    config.update(name="tiny-a2d-sft626-reuse1536-g4-u4-v1", data_selection="uniform_reuse_v1", data_path=DATA_PATH, data_sha256=hashlib.sha256(data).hexdigest(), data_manifest=MANIFEST_PATH, data_manifest_sha256=hashlib.sha256(manifest_data).hexdigest(), train_questions=1536, expected_total_candidates=6144, questions_per_cycle=16, physical_generation_batch_size=4, formal_training_enabled=False, minimum_free_gib=5.0, smoke_minimum_free_gib=0.35, evaluation_order="No model evaluation in this preparation or smoke workflow", checkpoint_rotation_enabled=False)
    config["training"].update(max_steps=384, num_iterations=4, generation_batch_size=64, save_steps=16, shuffle_dataset=False)
    for name, content in [(DATA_PATH, data), (MANIFEST_PATH, manifest_data), (CONFIG_PATH, json_bytes(config))]:
        write_new(project / name, content)
    load_reuse_rows(project, config)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.project), ensure_ascii=True, indent=2))
