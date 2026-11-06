"""生成并核验方案 A 的固定日程；只使用标准库，不加载模型或连接 GPU。"""

import argparse
import collections
import copy
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

from run_experiment import read_json


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_RUN = "rl-gsm8k-512-g4-20261104-110156"
SOURCE_FOLDER = "data/gsm8k/rl_static_sources"
DATA_PATH = "data/gsm8k/processed/rl_train_static_512.jsonl"
MANIFEST_PATH = "configs/rl_static_data_512.json"
CONFIG_PATH = "configs/rl_static_tiny_a2d.json"


def read_rows(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_new(path, content):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError("Refusing to replace a different prepared artifact: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(content)


def make_schedule(source, history):
    by_id = {row["id"]: row for row in source}
    if len(source) != 1973 or len(by_id) != 1973 or len(history) != 512 or len({row["id"] for row in history}) != 512:
        raise ValueError("Expected 1973 unique RL questions and 512 historical groups")
    if {row["id"] for row in history} != {row["id"] for row in source[:512]}:
        raise ValueError("Historical groups must match the original 512 RL questions")
    for row in history:
        if row["candidates"] != 4 or row["correct_candidates"] not in range(5):
            raise ValueError("Invalid historical four-candidate group")
    priority_ids = sorted(row["id"] for row in history if 0 < row["correct_candidates"] < 4)
    if len(priority_ids) != 127:
        raise ValueError("The registered historical priority pool must contain 127 questions")
    priority_set = set(priority_ids)
    strata = [[], [], []]
    for row in source:
        if row["id"] in priority_set:
            continue
        steps = len(re.findall(r"<<.*?>>", row["original_answer"]))
        index = 0 if steps <= 2 else 1 if steps <= 4 else 2
        strata[index].append(row["id"])
    sizes = [len(rows) for rows in strata]
    quotas = [258 * size // 1846 for size in sizes]
    remainder_order = sorted(range(3), key=lambda index: (-(258 * sizes[index] % 1846), index))
    for index in remainder_order[:258 - sum(quotas)]:
        quotas[index] += 1
    selection_rng = random.Random(43)
    exploration_ids = []
    for rows, count in zip(strata, quotas):
        exploration_ids.extend(selection_rng.sample(rows, count))
    # 选择和槽位分配连续使用独立的 seed=43 RNG，不接触模型生成的全局 RNG。
    selection_rng.shuffle(exploration_ids)
    random.Random(42).shuffle(priority_ids)
    result = []
    exploration_index = 0
    for half in range(2):
        priority_index = 0
        for step_in_half in range(16):
            priority_count = 7 if step_in_half == 15 else 8
            for slot in range(16):
                if slot < priority_count:
                    question_id = priority_ids[priority_index]
                    priority_index += 1
                    pool = "priority"
                    presentation_number = half + 1
                else:
                    question_id = exploration_ids[exploration_index]
                    exploration_index += 1
                    pool = "exploration"
                    presentation_number = 1
                position = len(result)
                result.append({**by_id[question_id], "question_id": question_id, "occurrence_id": "static-" + str(position).zfill(4), "presentation_index": position, "pool": pool, "presentation_number": presentation_number, "scheduled_optimizer_step": half * 16 + step_in_half + 1})
    details = {"selection_seed": 43, "priority_order_seed": 42, "canonical_order": "priority IDs sorted; strata retain source order; ties use 0-2,3-4,5+ order", "exploration_rng": "Random(43): sample each stratum in order, then shuffle the 258 selected IDs", "slot_order": "priority slots precede exploration slots inside each update; priority slots repeat after 16 updates", "strata": [{"name": name, "pool_questions": size, "selected_questions": quota} for name, size, quota in zip(["steps_0_2", "steps_3_4", "steps_5_plus"], sizes, quotas)], "presentations": 512, "unique_questions": 385, "priority_questions": 127, "priority_presentations": 254, "exploration_presentations": 258, "candidates_per_group": 4, "total_candidates": 2048, "optimizer_steps": 32, "presentations_per_update": 16}
    return result, details


def check_leakage(project, rows):
    def normalize(question):
        return " ".join(unicodedata.normalize("NFKC", question).casefold().split())

    selected_ids = {row["id"] for row in rows}
    selected_questions = {normalize(row["question"]) for row in rows}
    result = {}
    for split, count in [("sft_train", 5000), ("validation", 500), ("test", 1319)]:
        path = project / "data/gsm8k/splits" / (split + ".jsonl")
        other = read_rows(path)
        if len(other) != count:
            raise ValueError("Unexpected frozen split size: " + split)
        ids = selected_ids & {row["id"] for row in other}
        questions = selected_questions & {normalize(row["question"]) for row in other}
        if ids or questions:
            raise ValueError("Static RL schedule overlaps frozen split: " + split)
        result[split] = {"rows": count, "id_overlap": len(ids), "normalized_question_overlap": len(questions), "sha256": digest(path)}
    return result


def load_static_rows(project, config):
    manifest_path = project / config["data_manifest"]
    if digest(manifest_path) != config["data_manifest_sha256"]:
        raise ValueError("Static manifest checksum mismatch")
    manifest = read_json(manifest_path)
    for name, expected in manifest["input_sha256"].items():
        if digest(project / name) != expected:
            raise ValueError("Static selection input changed: " + name)
    if digest(project / config["data_path"]) != config["data_sha256"]:
        raise ValueError("Static schedule checksum mismatch")
    baseline = read_json(project / manifest["baseline_config"])
    for key in ["model_path", "source_sft_run", "source_adapter_sha256", "dllm_revision", "lora", "reward_metric", "reward_values", "evaluation_protocol", "evaluation_protocol_sha256", "expected_lora_parameters", "development_examples", "candidates_per_question", "expected_total_candidates"]:
        if config[key] != baseline[key]:
            raise ValueError("A non-data setting changed: " + key)
    if config["training"] != {**baseline["training"], "shuffle_dataset": False}:
        raise ValueError("Only native dataset shuffling may change in training settings")
    expected, details = make_schedule(read_rows(project / manifest["source"]), read_rows(project / manifest["historical_groups"]))
    rows = read_rows(project / config["data_path"])
    if rows != expected or manifest["schedule"] != details:
        raise ValueError("Static schedule differs from the registered selection and ordering rules")
    if config["train_presentations"] != 512 or config["unique_train_questions"] != 385 or config["train_questions"] != 512:
        raise ValueError("Static presentation budget changed")
    if check_leakage(project, rows) != manifest["leakage_checks"]:
        raise ValueError("Frozen leakage-check inputs changed")
    return rows


def prepare(project):
    source_path = "data/gsm8k/processed/rl_train.jsonl"
    baseline_path = "configs/rl_tiny_a2d.json"
    history_source = project / "矩池云" / SOURCE_RUN / "reward_groups.jsonl"
    historical_path = SOURCE_FOLDER + "/reward_groups.jsonl"
    rows, details = make_schedule(read_rows(project / source_path), read_rows(history_source))
    leakage = check_leakage(project, rows)
    baseline = read_json(project / baseline_path)
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    manifest = {"strategy": "static_mixed_v1", "research_report": "reports/rl_training_data_research_20261105.md", "source": source_path, "historical_groups": historical_path, "historical_run": SOURCE_RUN, "baseline_config": baseline_path, "input_sha256": {source_path: digest(project / source_path), historical_path: digest(history_source), baseline_path: digest(project / baseline_path)}, "schedule": details, "leakage_checks": leakage, "semantic_near_duplicates_checked": False, "official_test_used_only_for_overlap_check": True, "old_responses_or_rewards_reused_for_training": False}
    manifest_data = json_bytes(manifest)
    config = copy.deepcopy(baseline)
    config.update(name="tiny-a2d-sft626-diffu-grpo-static512-g4-v1", data_selection="static_mixed_v1", data_path=DATA_PATH, data_sha256=hashlib.sha256(data).hexdigest(), data_manifest=MANIFEST_PATH, data_manifest_sha256=hashlib.sha256(manifest_data).hexdigest(), train_presentations=512, unique_train_questions=385, evaluation_order="reuse frozen SFT and previous RL results, RL training, new RL development evaluation; no official test", reference_evaluations={})
    config["training"]["shuffle_dataset"] = False
    copies = [(history_source, project / historical_path)]
    for destination_name, source_name, correct in [("development_sft", "development_sft", 144), ("development_previous_rl", "development_rl", 147)]:
        source_folder = project / "矩池云" / SOURCE_RUN / source_name
        destination = SOURCE_FOLDER + "/" + destination_name
        config["reference_evaluations"][destination_name] = {"path": destination, "expected_correct": correct, "source_run": SOURCE_RUN, "source_phase": source_name, "files_sha256": {name: digest(source_folder / name) for name in ["predictions.jsonl", "summary.json", "run_config.json"]}}
        copies.extend((source_folder / name, project / destination / name) for name in config["reference_evaluations"][destination_name]["files_sha256"])
    for source, destination in copies:
        write_new(destination, source.read_bytes())
    write_new(project / DATA_PATH, data)
    write_new(project / MANIFEST_PATH, manifest_data)
    write_new(project / CONFIG_PATH, json_bytes(config))
    for smoke in [False, True]:
        job = read_json(project / "configs/jobs" / ("rl_smoke.json" if smoke else "rl_gsm8k_512.json"))
        job["command"][job["command"].index("--config") + 1] = "{project}/" + CONFIG_PATH
        job["max_runtime_minutes"] = 15 if smoke else 285
        job["note"] = "Static selection only; original SFT start, G=4, temperature=1, 512 presentations/385 questions/2048 new candidates/32 updates. Manual GPU stop. No retries or test evaluation."
        if not smoke:
            job["required_outputs"].extend(["sampling_schedule.jsonl", "sampling_checks.json", "development_previous_rl/summary.json", "reference_checks.json", "sampling_pool_summary.json"])
        else:
            job["required_outputs"].extend(["sampling_schedule.jsonl", "sampling_checks.json", "sampling_pool_summary.json"])
        write_new(project / "configs/jobs" / ("rl_static_smoke.json" if smoke else "rl_static_512.json"), json_bytes(job))
    load_static_rows(project, config)
    return {"status": "prepared", "config": CONFIG_PATH, "data_sha256": config["data_sha256"], "manifest_sha256": config["data_manifest_sha256"], "schedule": details, "leakage_checks": leakage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        rows = load_static_rows(PROJECT, read_json(PROJECT / CONFIG_PATH))
        print(json.dumps({"status": "passed", "presentations": len(rows), "unique_questions": len({row["id"] for row in rows})}))
    else:
        print(json.dumps(prepare(PROJECT), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
