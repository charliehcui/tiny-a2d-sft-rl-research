"""无 GPU 验证日程、原生 TRL 更新、候选复用与实际保存恢复；保留全部测试产物。"""

import ast
import collections
import copy
import hashlib
import json
import os
import random
import sys
import types
import unittest
import unicodedata
import uuid
from contextlib import nullcontext
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "disabled"
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

import torch
import torch.nn.functional as F
from accelerate.utils import gather_object, set_seed
from dataclasses import dataclass
from datasets import Dataset
from peft import LoraConfig, get_peft_model_state_dict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast, TrainerCallback
from trl import GRPOConfig
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_trainer import GRPOTrainer, nanstd, split_tensor_dict

from prepare_rl_reuse import CONFIG_PATH, load_reuse_rows, make_schedule
from prepare_rl_static import digest, json_bytes, read_rows, write_new
from run_experiment import read_json


def load_locked_native_classes():
    # 只装载已锁定训练器的原始类，避免导入与本测试无关的整套 CUDA 工具。
    path = PROJECT / "work/rl_source/dllm__pipelines__rl__grpo__trainer.py"
    module = types.ModuleType("reuse_native_cpu")
    sys.modules[module.__name__] = module
    module.__dict__.update(torch=torch, F=F, dataclass=dataclass, GRPOConfig=GRPOConfig, GRPOTrainer=GRPOTrainer, set_seed=set_seed, gather_object=gather_object, nullcontext=nullcontext, FSDP=torch.distributed.fsdp.FullyShardedDataParallel, maybe_apply_chat_template=maybe_apply_chat_template, is_conversational=is_conversational, unwrap_model_for_generation=unwrap_model_for_generation, nanstd=nanstd, split_tensor_dict=split_tensor_dict, profiling_decorator=lambda function: function)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in ["DiffuGRPOConfig", "DiffuGRPOTrainer"]]
    source = ast.Module(body=ast.parse("from __future__ import annotations").body + classes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(source), str(path), "exec"), module.__dict__)
    package = types.ModuleType("dllm.pipelines.rl")
    package.DiffuGRPOTrainer = module.DiffuGRPOTrainer
    sys.modules["dllm.pipelines.rl"] = package
    return module


native = load_locked_native_classes()
from rl_reuse_trainer import ReuseTrainer, tensor_digest
from train_rl_reuse import require_mode, validate_resume
from rl_storage import seal_checkpoint

torch.set_num_threads(1)


class TinySampler:
    def __init__(self):
        self.calls = []
        self.model = None

    def sample(self, prompts, config):
        if len(prompts) != 4 or len({len(prompt) for prompt in prompts}) != 1:
            raise AssertionError("Physical sampler must receive only one four-candidate question")
        if any((prompt == 0).any() for prompt in prompts):
            raise AssertionError("Padding leaked into the physical sampler")
        self.calls.append(len(prompts))
        completions = torch.randint(4, 16, (4, config.max_new_tokens))
        completions[:, -1] = 3
        return torch.cat([torch.stack(prompts), completions], dim=1)


class TinyReward:
    __name__ = "fixture_outcome"

    def __init__(self):
        self.groups = []

    def __call__(self, prompts, completions, completion_ids, id, **kwargs):
        if len(id) != 4 or len(set(id)) != 1:
            raise AssertionError("Rewards crossed question groups")
        self.groups.append(id[0])
        return [0.0, 1.0, 0.0, 1.0]


def tiny_tokenizer():
    words = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2, "[EOS]": 3, "word": 4, "assistant": 5}
    for index in range(6, 32):
        words["token" + str(index)] = index
    backend = Tokenizer(WordLevel(words, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", mask_token="[MASK]", eos_token="[EOS]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}assistant"
    return tokenizer


def tiny_rows():
    return [{"id": "cpu-" + str(index), "prompt": [{"role": "user", "content": " ".join(["word"] * (2 + index % 16))}]} for index in range(32)]


def build_trainer(folder, stop=None, family_root=None):
    set_seed(42)
    model = BertForMaskedLM(BertConfig(vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0))
    model.config._name_or_path = "cpu_fixture_only"
    arguments = native.DiffuGRPOConfig(output_dir=str(folder / "checkpoints"), max_steps=8, num_train_epochs=1, per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=64, generation_batch_size=64, num_generations=4, num_iterations=4, shuffle_dataset=False, learning_rate=3e-6, lr_scheduler_type="constant", warmup_steps=0, beta=0.04, p_mask_prompt=0.15, epsilon=0.2, scale_rewards=False, loss_type="grpo", max_completion_length=8, steps=8, block_size=8, temperature=1.0, bf16=False, fp16=False, use_cpu=True, optim="adamw_torch", weight_decay=0.0, max_grad_norm=1.0, save_strategy="steps", save_steps=4, save_total_limit=2, save_only_model=False, eval_strategy="no", logging_steps=1, logging_nan_inf_filter=False, report_to="none", disable_tqdm=True, remove_unused_columns=False, dataloader_num_workers=0, dataloader_pin_memory=False, seed=42, data_seed=42)
    sampler = TinySampler()
    sampler_config = types.SimpleNamespace(max_new_tokens=8, cfg_scale=0.0)
    reward = TinyReward()
    trainer = ReuseTrainer(model=model, reward_funcs=[reward], args=arguments, train_dataset=Dataset.from_list(tiny_rows()), processing_class=tiny_tokenizer(), peft_config=LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["query", "value"]), sampler=sampler, sampler_config=sampler_config)

    class Marker(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / ("checkpoint-" + str(state.global_step))
            config = read_json(PROJECT / CONFIG_PATH)
            metadata = {**trainer.reuse_state(), "config_sha256": hashlib.sha256(json_bytes(config)).hexdigest(), "schedule_sha256": config["data_sha256"], "code_sha256": {"rl_reuse_trainer.py": digest(PROJECT / "scripts/rl_reuse_trainer.py")}}
            write_new(checkpoint / "reuse_state.json", json_bytes(metadata))
            write_new(checkpoint / "pipeline_complete.json", json_bytes({"global_step": state.global_step, "native_checkpoint_save_completed": True}))
            seal_checkpoint(checkpoint, family_root or folder)

        def on_step_end(self, args, state, control, **kwargs):
            if stop is not None and state.global_step == stop:
                control.should_save = True
                control.should_training_stop = True
            return control

    trainer.add_callback(Marker())
    return trainer, sampler, reward


class ReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = PROJECT / "work/rl_reuse_tests" / uuid.uuid4().hex
        cls.folder.mkdir(parents=True)
        cls.facts = {"cpu_only": True, "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "test_output": str(cls.folder)}

    def test_01_schedule_counts_reproducibility_and_leakage(self):
        config = read_json(PROJECT / CONFIG_PATH)
        rows = load_reuse_rows(PROJECT, config)
        state = random.getstate()
        expected = make_schedule(read_rows(PROJECT / "data/gsm8k/processed/rl_train.jsonl"))
        self.assertEqual(random.getstate(), state)
        self.assertEqual(rows, expected)
        self.assertEqual(len(rows), 1536)
        self.assertEqual(len({row["id"] for row in rows}), 1536)
        normalized_questions = {" ".join(unicodedata.normalize("NFKC", row["question"]).casefold().split()) for row in rows}
        self.assertEqual(len(normalized_questions), 1536)
        self.assertEqual(collections.Counter(row["rollout_cycle"] for row in rows), {cycle: 16 for cycle in range(96)})
        self.assertEqual(rows[-1]["scheduled_optimizer_step"], 381)
        manifest = read_json(PROJECT / config["data_manifest"])
        self.assertTrue(all(item["id_overlap"] == item["normalized_question_overlap"] == 0 for item in manifest["leakage_checks"].values()))
        self.facts["schedule"] = {"questions": 1536, "unique_normalized_questions": len(normalized_questions), "new_candidates": 6144, "rollout_cycles": 96, "optimizer_updates": 384, "reproduced": True, "leakage_checks": manifest["leakage_checks"], "sha256": config["data_sha256"]}

    def test_02_full_schedule_microstep_boundaries(self):
        config = read_json(PROJECT / CONFIG_PATH)
        rows = load_reuse_rows(PROJECT, config)
        trainer = object.__new__(ReuseTrainer)
        trainer.model = types.SimpleNamespace(training=True)
        trainer.state = types.SimpleNamespace(global_step=0)
        trainer._buffered_inputs = None
        trainer.reuse_usage = None
        generated = []
        consumed = collections.Counter()

        def generate(batch):
            trainer.reuse_cycle = trainer.state.global_step // 4
            generated.extend({row["id"] for row in batch})
            return [{"reuse_candidate": index, "reuse_seeds": (10, 20, 30, 40)} for index in range(64)]

        trainer._generate_and_score_completions = generate
        for cycle in range(96):
            batch = [row for row in rows[cycle * 16:cycle * 16 + 16] for _ in range(4)]
            for view in range(4):
                for candidate in range(64):
                    item = trainer._prepare_inputs(batch)
                    self.assertEqual((item["reuse_candidate"], item["reuse_view"]), (candidate, view))
                    self.assertEqual(item["reuse_seeds"][item["reuse_view"]], (view + 1) * 10)
                    consumed[(cycle, candidate, view)] += 1
                trainer.state.global_step += 1
        self.assertEqual(len(generated), 1536)
        self.assertEqual(len(set(generated)), 1536)
        self.assertEqual(len(consumed), 24576)
        self.assertEqual(set(consumed.values()), {1})
        self.assertEqual(trainer.state.global_step, 384)
        self.facts["full_microstep_simulation"] = {"new_questions": len(generated), "candidate_view_uses": len(consumed), "updates": trainer.state.global_step, "every_candidate_view_used_once": True}

    def test_03_real_cpu_optimizer_and_checkpoint_resume(self):
        continuous, sampler, reward = build_trainer(self.folder / "continuous")
        generate = continuous._generate_and_score_completions
        checked_views = []

        def verify_views(batch):
            candidates = generate(batch)
            with torch.no_grad():
                for item in candidates:
                    ids = torch.cat([item["prompt_ids"], item["completion_ids"]], dim=1)
                    mask = torch.cat([item["prompt_mask"], item["completion_mask"]], dim=1)
                    for view in range(4):
                        actual_logps = continuous._get_per_token_logps(continuous.model, ids, mask, item["completion_ids"].size(1), 1, seed=item["reuse_seeds"][view])
                        self.assertTrue(torch.equal(actual_logps, item["diffu_old_logps_all"][:, view, :]))
                        checked_views.append((item["reuse_cycle"], item["reuse_candidate"], view))
            return candidates

        continuous._generate_and_score_completions = verify_views
        continuous.train()
        self.assertEqual(len(checked_views), 512)
        self.assertEqual(len(set(checked_views)), 512)
        self.assertEqual(continuous.state.global_step, 8)
        self.assertEqual(len(sampler.calls), 32)
        self.assertEqual(set(sampler.calls), {4})
        self.assertEqual(reward.groups, [row["id"] for row in tiny_rows()])
        self.assertEqual(continuous.reuse_state()["completed_cycles"], 2)
        target = {name: value.detach().cpu().clone() for name, value in get_peft_model_state_dict(continuous.model).items()}
        first, _, first_reward = build_trainer(self.folder / "interrupted", stop=4)
        first.train()
        checkpoint = self.folder / "interrupted/checkpoints/checkpoint-4"
        metadata = validate_resume(checkpoint, read_json(PROJECT / CONFIG_PATH), read_json(PROJECT / CONFIG_PATH)["data_sha256"])
        self.assertEqual(metadata["global_step"], 4)
        self.assertEqual(len(first_reward.groups), 16)
        resumed, _, second_reward = build_trainer(self.folder / "resumed")
        resumed.train(resume_from_checkpoint=str(checkpoint))
        self.assertEqual(resumed.state.global_step, 8)
        self.assertEqual(second_reward.groups, [row["id"] for row in tiny_rows()[16:]])
        actual = get_peft_model_state_dict(resumed.model)
        delta = max((actual[name].detach().cpu() - value).abs().max().item() for name, value in target.items())
        self.assertEqual(delta, 0.0)
        self.assertTrue(all(not item["nonfinite"] for item in continuous.reuse_ratio_sums.values()))
        ratio_change = max(item["max_abs_ratio_delta"] for step, item in continuous.reuse_ratio_sums.items() if step % 4 > 0)
        self.assertGreater(ratio_change, 0.0)
        self.facts["real_cpu_training"] = {"optimizer_updates": 8, "new_questions": 32, "physical_batches": len(sampler.calls), "physical_batch_size": 4, "mask_seed_and_logprob_views_compared": len(checked_views), "all_four_view_old_probabilities_exactly_match_frozen_current": True, "native_checkpoint_saved": True, "native_resume_continued_4_to_8": True, "resumed_next_question_prefix_correct": True, "continuous_vs_resumed_adapter_max_abs_difference": delta, "later_probability_ratio_max_delta": ratio_change, "no_nonfinite_ratios": True, "old_cache_hash_checks": True, "fixture_only_not_gsm8k_training": True}
        damaged = dict(resumed._buffered_inputs[0])
        damaged["reuse_view"] = 0
        damaged["diffu_old_logps_all"] = damaged["diffu_old_logps_all"].clone() + 1
        with self.assertRaisesRegex(RuntimeError, "modified"):
            resumed._compute_loss(resumed.model, damaged)

    def test_04_formal_training_guard_and_no_deletions(self):
        disabled_config = read_json(PROJECT / CONFIG_PATH)
        disabled_config["formal_training_enabled"] = False
        with self.assertRaises(PermissionError):
            require_mode(disabled_config, False)
        require_mode(read_json(PROJECT / CONFIG_PATH), True)
        self.assertTrue((self.folder / "continuous/checkpoints/checkpoint-4").exists())
        self.assertTrue((self.folder / "continuous/checkpoints/checkpoint-8").exists())
        self.facts["formal_training_guard_verified"] = True
        self.facts["all_test_artifacts_retained"] = True

    @classmethod
    def tearDownClass(cls):
        write_new(cls.folder / "local_checks.json", json_bytes(cls.facts))
        print("LOCAL_CHECKS=" + str(cls.folder / "local_checks.json"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
