"""检查准备好的 SFT 参数和恢复约束；不加载大模型、不执行训练。"""

import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from train_sft import load_fixed_rows, planned_steps, validate_resume
from run_experiment import validate_job


class SFTConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((PROJECT / "configs/sft_tiny_a2d.json").read_text(encoding="utf-8"))

    def test_steps_and_effective_batch(self):
        training = self.config["training"]
        self.assertEqual(training["per_device_train_batch_size"] * training["gradient_accumulation_steps"], 16)
        self.assertEqual(planned_steps(self.config), 626)
        self.assertEqual(math.ceil(626 * training["warmup_ratio"]), 19)

    def test_fixed_training_and_validation_used_without_overlap(self):
        train = load_fixed_rows(self.config, "train")
        validation = load_fixed_rows(self.config, "validation")
        self.assertEqual((len(train), len(validation)), (5000, 500))
        self.assertFalse({row["id"] for row in train} & {row["id"] for row in validation})

    def test_adapter_only_updates_and_complete_native_checkpoints(self):
        model = self.config["model"]
        self.assertTrue(model["lora"])
        self.assertEqual(model["r"], 8)
        self.assertIsNone(model["modules_to_save"])
        self.assertFalse(model["load_in_4bit"])
        self.assertNotIn("lm_head", model["target_modules"])
        training = self.config["training"]
        self.assertFalse(training["save_only_model"])
        self.assertEqual(training["save_total_limit"], 2)
        self.assertTrue(training["prediction_loss_only"])
        self.assertFalse(training["dataloader_drop_last"])

    def test_prepared_job_can_validate_after_explicit_review(self):
        job = json.loads((PROJECT / "configs/jobs/sft_gsm8k.json").read_text())
        disabled = copy.deepcopy(job)
        disabled["enabled"] = False
        with self.assertRaises(ValueError):
            validate_job(disabled)
        reviewed = copy.deepcopy(job)
        reviewed["enabled"] = True
        validate_job(reviewed)
        self.assertEqual(reviewed["min_free_gib"], self.config["storage"]["minimum_start_free_gib"])

    def test_no_training_sample_requires_truncation(self):
        audit = json.loads((PROJECT / "reports/sft_tokenizer_audit_20261104.json").read_text())
        for split in ("train", "validation"):
            self.assertLessEqual(audit["splits"][split]["max_tokens"], self.config["data"]["max_length"])
            self.assertEqual(audit["splits"][split]["prompt_prefix_mismatch_count"], 0)
        self.assertFalse(self.config["data"]["truncate_or_filter"])

    def test_general_evaluator_retains_the_frozen_scoring_and_sampler(self):
        source = (PROJECT / "scripts/run_baseline.py").read_text(encoding="utf-8")
        source = source.replace('parser.add_argument("--output", type=Path, required=True)', 'parser.add_argument("--output", type=Path, required=True)\n    parser.add_argument("--checkpoint", type=Path, required=True)')
        source = source.replace('PROJECT / "configs" / "baseline_protocol_v1.json"', 'PROJECT / "configs" / "baseline_protocol_v2.json"')
        source = source.replace('str(PROJECT / "models" / "Qwen3-0.6B-diffusion-mdlm-v0.1")', 'str(args.checkpoint.resolve())')
        self.assertEqual(source, (PROJECT / "scripts/evaluate_model.py").read_text(encoding="utf-8"))
        protocol_path = PROJECT / self.config["evaluation"]["protocol"]
        self.assertEqual(hashlib.sha256(protocol_path.read_bytes()).hexdigest(), self.config["evaluation"]["protocol_sha256"])

    def test_resume_requires_marker_optimizer_and_same_configuration(self):
        parent = PROJECT / "work/sft_tests"
        assert parent.resolve().is_relative_to(PROJECT.resolve())
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            run = Path(folder)
            checkpoint = run / "checkpoints/checkpoint-100"
            checkpoint.mkdir(parents=True)
            with self.assertRaises(ValueError):
                validate_resume(checkpoint, self.config)
            for name in ("pipeline_complete.json", "trainer_state.json", "adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
                (checkpoint / name).write_text("{}")
            (run / "sft_config.json").write_text(json.dumps(self.config))
            self.assertEqual(validate_resume(checkpoint, self.config), str(checkpoint.resolve()))
            different = copy.deepcopy(self.config)
            different["training"]["learning_rate"] *= 2
            with self.assertRaises(ValueError):
                validate_resume(checkpoint, different)


if __name__ == "__main__":
    unittest.main(verbosity=2)
