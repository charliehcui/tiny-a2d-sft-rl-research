"""方案 A 的本地标准库检查；原生采样方法用 AST 载入，不导入 CUDA 或训练模型。"""

import ast
import collections
import contextlib
import copy
import hashlib
import io
import json
import os
import random
import sys
import tempfile
import types
import typing
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import prepare_rl_static as selection
import train_rl
from run_experiment import read_json, validate_job


def native_node(path, name, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body
    if class_name:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name).body
    node = next(node for node in nodes if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)
    if isinstance(node, ast.FunctionDef):
        node.decorator_list = []
    return node


class StaticSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read_json(PROJECT / selection.CONFIG_PATH)
        cls.rows = train_rl.load_data(cls.config)
        cls.source = selection.read_rows(PROJECT / "data/gsm8k/processed/rl_train.jsonl")
        cls.history = selection.read_rows(PROJECT / selection.SOURCE_FOLDER / "reward_groups.jsonl")

    def setUp(self):
        parent = PROJECT / "work/rl_static_tests"
        parent.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.run = Path(self.temporary.name)
        self.assertTrue(self.run.resolve().is_relative_to(parent.resolve()))

    def tearDown(self):
        self.temporary.cleanup()

    def test_registered_counts_separation_and_stratification(self):
        rows = self.rows
        self.assertEqual(len(rows), 512)
        self.assertEqual(len({row["id"] for row in rows}), 385)
        self.assertEqual(len({row["occurrence_id"] for row in rows}), 512)
        counts = collections.Counter(row["id"] for row in rows)
        self.assertEqual(collections.Counter(counts.values()), {2: 127, 1: 258})
        self.assertEqual(collections.Counter(row["pool"] for row in rows), {"priority": 254, "exploration": 258})
        for index, row in enumerate(rows):
            self.assertEqual(row["presentation_index"], index)
            self.assertEqual(row["scheduled_optimizer_step"], index // 16 + 1)
            if index < 256 and row["pool"] == "priority":
                repeated = rows[index + 256]
                self.assertEqual(row["id"], repeated["id"])
                self.assertEqual(repeated["presentation_number"], 2)
        manifest = read_json(PROJECT / selection.MANIFEST_PATH)
        self.assertEqual([row["selected_questions"] for row in manifest["schedule"]["strata"]], [92, 123, 43])
        self.assertEqual(selection.check_leakage(PROJECT, rows), manifest["leakage_checks"])

    def test_fixed_seeds_reproduce_bytes_without_changing_global_rng(self):
        state = random.getstate()
        rows, details = selection.make_schedule(self.source, self.history)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(rows, self.rows)
        encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), "1054fc57d466c3a93f04c51e98bf095248f822190c566affd4db934a4843e602")
        self.assertEqual(details["selection_seed"], 43)
        self.assertEqual(details["priority_order_seed"], 42)

    def test_only_data_order_flag_changes_training_options(self):
        baseline = read_json(PROJECT / "configs/rl_tiny_a2d.json")
        self.assertEqual(self.config["training"], {**baseline["training"], "shuffle_dataset": False})
        for key in ["model_path", "source_adapter_sha256", "source_sft_run", "lora", "evaluation_protocol_sha256", "reward_metric"]:
            self.assertEqual(self.config[key], baseline[key])
        changed = copy.deepcopy(self.config)
        changed["training"]["temperature"] = 0.7
        with self.assertRaisesRegex(ValueError, "Only native dataset shuffling"):
            train_rl.load_data(changed)
        changed = copy.deepcopy(self.config)
        changed["data_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum"):
            train_rl.load_data(changed)

    def test_reordered_schedule_or_changed_answers_are_rejected(self):
        original = selection.read_rows
        for change in ["order", "answer"]:
            tampered = copy.deepcopy(self.rows)
            if change == "order":
                tampered[0], tampered[1] = tampered[1], tampered[0]
            else:
                tampered[0]["answer"] = "999999"

            def read(path):
                return tampered if Path(path) == PROJECT / self.config["data_path"] else original(path)

            with patch.object(selection, "read_rows", side_effect=read):
                with self.assertRaisesRegex(ValueError, "schedule differs"):
                    train_rl.load_data(self.config)

    def test_original_512_configuration_still_loads(self):
        rows = train_rl.load_data(read_json(PROJECT / "configs/rl_tiny_a2d.json"))
        self.assertEqual(len(rows), 512)
        self.assertEqual(len({row["id"] for row in rows}), 512)
        self.assertNotIn("occurrence_id", rows[0])

    def test_native_sampler_and_buffer_generate_exactly_one_fresh_group_per_occurrence(self):
        native_path = PROJECT / "work/rl_source/trl__grpo_trainer.py"
        diffusion_path = PROJECT / "work/rl_source/dllm__pipelines__rl__grpo__trainer.py"
        nodes = [native_node(native_path, "RepeatSampler"), native_node(native_path, "_get_train_sampler", "GRPOTrainer"), native_node(diffusion_path, "_prepare_inputs", "DiffuGRPOTrainer")]
        namespace = {"Sampler": object, "Sized": typing.Sized, "Optional": typing.Optional, "Dataset": list, "split_tensor_dict": lambda data, count: data}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(native_path), "exec"), namespace)
        generated = []

        def generate(batch):
            self.assertEqual(len(batch), 4)
            self.assertEqual(len({row["occurrence_id"] for row in batch}), 1)
            generated.append(batch[0])
            return batch

        trainer = types.SimpleNamespace(train_dataset=self.rows, args=types.SimpleNamespace(generation_batch_size=4, steps_per_generation=4, seed=42), num_generations=4, num_iterations=1, shuffle_dataset=False, model=types.SimpleNamespace(training=True), _step=0, _buffered_inputs=None, _generate_and_score_completions=generate)
        indices = list(namespace["_get_train_sampler"](trainer))
        self.assertEqual(len(indices), 8192)
        for offset in range(0, len(indices), 4):
            batch = [self.rows[index] for index in indices[offset:offset + 4]]
            namespace["_prepare_inputs"](trainer, batch)
        self.assertEqual(trainer._step, 2048)
        self.assertEqual(trainer._step // self.config["training"]["gradient_accumulation_steps"], 32)
        self.assertEqual([row["occurrence_id"] for row in generated], [row["occurrence_id"] for row in self.rows])
        self.assertEqual(len(generated) * 4, 2048)
        self.assertEqual(len({row["id"] for row in generated}), 385)
        self.assertEqual(generated[0]["id"], generated[256]["id"])
        self.assertNotEqual(generated[0]["occurrence_id"], generated[256]["occurrence_id"])

    def make_reward(self, expected):
        reward = train_rl.OutcomeReward.__new__(train_rl.OutcomeReward)
        reward.tokenizer = object()
        reward.run_dir = self.run
        reward.trainer = types.SimpleNamespace(state=types.SimpleNamespace(global_step=0))
        reward.groups = []
        reward.expected_rows = expected
        reward.filters = {}
        reward.metric = {}
        return reward

    def invoke_reward(self, reward, row, values):
        metadata = {key: [row[key]] * 4 for key in ["question_id", "occurrence_id", "presentation_index", "pool", "presentation_number", "scheduled_optimizer_step"]}
        return reward(prompts=[row["prompt"]] * 4, completions=[None] * 4, completion_ids=values, answer=[row["answer"]] * 4, id=[row["id"]] * 4, **metadata)

    def test_repeated_question_is_scored_again_and_logged_as_a_separate_group(self):
        first, second = self.rows[0], self.rows[256]
        reward = self.make_reward([first, second])
        # 仅模拟评分返回值以验证记录与组边界；冻结真实评分器在 GPU Smoke Test 验证。
        fake_dllm = types.SimpleNamespace(utils=types.SimpleNamespace(sample_trim=lambda tokenizer, ids, stops: [str(row[0]) for row in ids]))
        fake_evaluator = types.SimpleNamespace(score_response=lambda text, answer, filters, metric: {"flexible-extract": {"correct": text == "1"}})
        with patch.dict(sys.modules, {"dllm": fake_dllm, "evaluate_model": fake_evaluator}):
            self.assertEqual(self.invoke_reward(reward, first, [[1], [0], [0], [0]]), [1, 0, 0, 0])
            reward.trainer.state.global_step = 16
            self.assertEqual(self.invoke_reward(reward, second, [[1], [1], [1], [1]]), [1, 1, 1, 1])
            with self.assertRaisesRegex(ValueError, "extra generation"):
                self.invoke_reward(reward, second, [[1]] * 4)
        records = selection.read_rows(self.run / "rollouts.jsonl")
        groups = selection.read_rows(self.run / "reward_groups.jsonl")
        self.assertEqual(len(records), 8)
        self.assertEqual([row["correct_candidates"] for row in groups], [1, 4])
        self.assertEqual(groups[0]["advantages"], [0.75, -0.25, -0.25, -0.25])
        self.assertEqual(groups[1]["advantages"], [0, 0, 0, 0])
        self.assertEqual([row["occurrence_id"] for row in groups], [first["occurrence_id"], second["occurrence_id"]])
        self.assertEqual(train_rl.summarize_pools(groups)["priority"]["mixed_groups"], 1)

    def test_out_of_order_occurrence_fails_before_writing(self):
        reward = self.make_reward(self.rows)
        fake_dllm = types.SimpleNamespace()
        fake_evaluator = types.SimpleNamespace(score_response=lambda *args: None)
        with patch.dict(sys.modules, {"dllm": fake_dllm, "evaluate_model": fake_evaluator}):
            with self.assertRaisesRegex(ValueError, "next scheduled occurrence"):
                self.invoke_reward(reward, self.rows[1], [[1]] * 4)
        self.assertFalse((self.run / "rollouts.jsonl").exists())

    def test_same_question_with_wrong_occurrence_is_not_merged(self):
        reward = self.make_reward(self.rows)
        wrong = {**self.rows[0], "occurrence_id": self.rows[256]["occurrence_id"]}
        with patch.dict(sys.modules, {"dllm": types.SimpleNamespace(), "evaluate_model": types.SimpleNamespace(score_response=lambda *args: None)}):
            with self.assertRaisesRegex(ValueError, "same scheduled occurrence"):
                self.invoke_reward(reward, wrong, [[1]] * 4)
        self.assertFalse((self.run / "rollouts.jsonl").exists())

    def test_wrong_optimizer_update_fails_before_scoring(self):
        reward = self.make_reward(self.rows)
        reward.trainer.state.global_step = 1
        with patch.dict(sys.modules, {"dllm": types.SimpleNamespace(), "evaluate_model": types.SimpleNamespace(score_response=lambda *args: None)}):
            with self.assertRaisesRegex(ValueError, "wrong optimizer update"):
                self.invoke_reward(reward, self.rows[0], [[1]] * 4)
        self.assertFalse((self.run / "rollouts.jsonl").exists())

    def test_references_are_reused_without_evaluation_and_excluded_from_cost(self):
        train_rl.reuse_reference_evaluations(self.run, self.config)
        checks = read_json(self.run / "reference_checks.json")
        self.assertTrue(checks["no_models_rerun"])
        self.assertEqual(checks["evaluations"]["development_sft"]["correct"], 144)
        self.assertEqual(checks["evaluations"]["development_previous_rl"]["correct"], 147)
        import shutil

        shutil.copytree(self.run / "development_previous_rl", self.run / "development_rl")
        summary = read_json(self.run / "development_rl/summary.json")
        summary["total_process_seconds"] = 20
        train_rl.write_json(self.run / "development_rl/summary.json", summary)
        train_rl.write_json(self.run / "training_summary.json", {"total_seconds": 10, "mixed_reward_groups": 127, "question_presentations": 512})
        train_rl.compare_static_development(self.run, self.config)
        result = read_json(self.run / "development_comparison.json")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["estimated_total_process_seconds"], 30)
        self.assertEqual(result["rl_correct"], 147)
        self.assertFalse(result["preregistered_targets"]["development_target_met"])
        self.assertFalse(result["preregistered_targets"]["mixed_group_target_met"])
        self.assertEqual(result["comparisons"]["development_previous_rl"]["wrong_to_right"], 0)
        self.assertEqual(result["comparisons"]["development_sft"]["wrong_to_right"], 21)
        self.assertEqual(result["comparisons"]["development_sft"]["right_to_wrong"], 18)

    def test_pipeline_reuses_references_then_trains_and_evaluates_once_or_stops_on_failure(self):
        for failure in [None, "rl_training"]:
            run = self.run / ("success" if failure is None else "failure")
            (run / "code").mkdir(parents=True)
            calls = []

            class FakeProcess:
                def __init__(process, command):
                    process.phase = read_json(run / "phase.json")["phase"]
                    calls.append((process.phase, command))

                def wait(process):
                    if process.phase == failure:
                        return 9
                    if process.phase == "development_rl":
                        (run / "development_rl").mkdir()
                        train_rl.write_json(run / "development_rl/summary.json", {"status": "complete", "examples": 500})
                    return 0

            argv = ["train_rl.py", "--config", str(PROJECT / selection.CONFIG_PATH), "--run-dir", str(run)]
            with patch.dict(os.environ, {"BLUEWHALE_RUN_DIR": str(run)}), patch.object(sys, "argv", argv), patch.object(train_rl.subprocess, "Popen", FakeProcess), patch.object(train_rl, "compare_development") as compare, contextlib.redirect_stdout(io.StringIO()):
                if failure:
                    with self.assertRaises(RuntimeError):
                        train_rl.main()
                    compare.assert_not_called()
                else:
                    train_rl.main()
                    compare.assert_called_once()
            self.assertEqual([phase for phase, command in calls], ["rl_training"] if failure else ["rl_training", "development_rl"])
            for phase, command in calls:
                if "--split" in command:
                    self.assertEqual(command[command.index("--split") + 1], "validation")
            self.assertTrue((run / "reference_checks.json").exists())

    def test_jobs_keep_detached_pipeline_manual_stop_and_separate_smoke(self):
        for name, smoke in [("rl_static_512.json", False), ("rl_static_smoke.json", True)]:
            job = read_json(PROJECT / "configs/jobs" / name)
            validate_job(job)
            command = job["command"]
            self.assertEqual(command[command.index("--config") + 1], "{project}/" + selection.CONFIG_PATH)
            self.assertEqual(command[command.index("--run-dir") + 1], "{run_dir}")
            self.assertEqual("--smoke" in command, smoke)
        self.assertEqual(read_json(PROJECT / "configs/matpool_manual_stop.json")["mode"], "disabled")
        with self.assertRaises(FileExistsError):
            selection.write_new(PROJECT / selection.CONFIG_PATH, b"different")


if __name__ == "__main__":
    unittest.main(verbosity=2)
