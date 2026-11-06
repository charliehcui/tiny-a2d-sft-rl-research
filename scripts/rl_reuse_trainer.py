"""项目适配层：固定候选跨四次真实更新复用；上游源码保持不变。"""

import hashlib
import torch
from dllm.pipelines.rl import DiffuGRPOTrainer
from trl.trainer.grpo_trainer import GRPOTrainer


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


class ReuseTrainer(DiffuGRPOTrainer):
    def __init__(self, *args, physical_generation_batch_size=4, **kwargs):
        super().__init__(*args, **kwargs)
        settings = (self.args.per_device_train_batch_size, self.args.gradient_accumulation_steps, self.args.generation_batch_size, self.args.steps_per_generation, self.num_generations, self.num_iterations, self.accelerator.num_processes, physical_generation_batch_size)
        if settings != (1, 64, 64, 64, 4, 4, 1, 4) or self.shuffle_dataset:
            raise ValueError("Reuse requires one process, ordered 16x4 logical batch, physical batch 4, and four 64-microstep updates")
        if self.args.use_liger_loss or self.args.use_vllm or self.args.eval_strategy != "no":
            raise ValueError("Only the frozen native loss, sampler and training-only workflow are supported")
        self.reuse_events = []
        self.reuse_cycle = None
        self.reuse_batch_ids = None
        self.reuse_old_hashes = None
        self.reuse_usage = None
        self.reuse_ratio_sums = {}
        self.reuse_generation_calls = 0

    def _generate_and_score_completions(self, inputs):
        if len(inputs) != 64 or self.state.global_step % 4:
            raise ValueError("Generate exactly 16 groups at a completed reuse-cycle boundary")
        candidates = []
        cycle = self.state.global_step // 4
        for start in range(0, 64, 4):
            group = inputs[start:start + 4]
            if len({row["id"] for row in group}) != 1 or any(row["prompt"] != group[0]["prompt"] for row in group):
                raise ValueError("Each physical generation batch must be one question repeated four times")
            # 父实现只收到同题四条输入，既不会生成 64 条，也不会跨题填充或计算奖励。
            prepared = super()._generate_and_score_completions(group)
            self.reuse_generation_calls += 1
            seeds = tuple(self._mask_seeds)
            if len(seeds) != 4 or len(set(seeds)) != 4:
                raise ValueError("Each group needs four distinct prompt-mask seeds")
            if not torch.all(prepared["prompt_mask"] == 1):
                raise ValueError("Same-question physical groups must have no left padding")
            expected_prompt = self.processing_class.apply_chat_template(group[0]["prompt"], tokenize=True, add_generation_prompt=True)
            expected_ids = torch.tensor(expected_prompt, device=prepared["prompt_ids"].device)
            if any(not torch.equal(prompt, expected_ids) for prompt in prepared["prompt_ids"]):
                raise ValueError("Physical generation changed prompt tokens or positions")
            if prepared["diffu_old_logps_all"] is None or prepared["diffu_ref_logps_all"] is None:
                raise ValueError("Old and SFT reference probabilities must be cached before updates")
            for offset in range(4):
                item = {}
                for name, value in prepared.items():
                    item[name] = value[offset:offset + 1].detach().clone() if isinstance(value, torch.Tensor) else value
                item["reuse_seeds"] = seeds
                item["reuse_question_id"] = group[offset]["id"]
                item["reuse_candidate"] = start + offset
                item["reuse_cycle"] = cycle
                candidates.append(item)
        if len({row["id"] for row in inputs}) != 16:
            raise ValueError("A fresh logical batch must contain 16 distinct questions")
        self.reuse_cycle = cycle
        self.reuse_old_hashes = [tensor_digest(item["diffu_old_logps_all"]) for item in candidates]
        self.reuse_usage = [[0] * 4 for _ in candidates]
        self.reuse_events.append({"event": "generated", "cycle": cycle, "step": self.state.global_step, "new_candidates": 64, "physical_batch": 4, "old_probability_hash": hashlib.sha256("".join(self.reuse_old_hashes).encode()).hexdigest()})
        return candidates

    def _prepare_inputs(self, generation_batch):
        if not self.model.training:
            raise RuntimeError("ReuseTrainer does not run model evaluations")
        if self._buffered_inputs is None:
            if self.state.global_step % 4:
                raise ValueError("Resume is supported only after all four updates of a cycle")
            self._step = self.state.global_step * 64
        position = self._step % 256
        if position == 0:
            if self.reuse_usage is not None and any(views != [1, 1, 1, 1] for views in self.reuse_usage):
                raise RuntimeError("Previous candidates did not complete exactly four mask views")
            self.reuse_batch_ids = [row["id"] for row in generation_batch]
            self._buffered_inputs = self._generate_and_score_completions(generation_batch)
        elif [row["id"] for row in generation_batch] != self.reuse_batch_ids:
            raise ValueError("Native dataloader advanced before four optimizer updates finished")
        view = position // 64
        candidate = position % 64
        if self.state.global_step != self.reuse_cycle * 4 + view:
            raise RuntimeError("A full 64-candidate pass must end with a real optimizer update")
        item = dict(self._buffered_inputs[candidate])
        item["reuse_view"] = view
        self._step += 1
        return item

    def _compute_loss(self, model, inputs):
        view = inputs["reuse_view"]
        candidate = inputs["reuse_candidate"]
        if tensor_digest(inputs["diffu_old_logps_all"]) != self.reuse_old_hashes[candidate]:
            raise RuntimeError("Frozen old-policy probabilities were modified")
        if inputs["diffu_old_logps_all"].requires_grad or inputs["diffu_ref_logps_all"].requires_grad:
            raise RuntimeError("Cached old/reference probabilities must be detached")
        prepared = dict(inputs)
        prepared["old_per_token_logps"] = inputs["diffu_old_logps_all"][:, view, :]
        prepared["ref_per_token_logps"] = inputs["diffu_ref_logps_all"][:, view, :]
        self._current_mask_seed = inputs["reuse_seeds"][view]
        self.reuse_usage[candidate][view] += 1
        if self.reuse_usage[candidate][view] != 1:
            raise RuntimeError("Candidate mask view was consumed more than once")
        # 直接调用原 TRL 公式，绕开上游按微批递增的掩码索引。
        self.reuse_current_old = prepared["old_per_token_logps"]
        try:
            loss = GRPOTrainer._compute_loss(self, model, prepared)
        finally:
            self.reuse_current_old = None
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite reuse loss")
        return loss

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, seed=None):
        values = super()._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep, batch_size, seed)
        # 仅在当前策略的反传路径记录概率比，不保留 logits 或计算图。
        if torch.is_grad_enabled() and getattr(self, "reuse_current_old", None) is not None:
            active = attention_mask[:, -logits_to_keep:].bool()
            difference = (values.detach() - self.reuse_current_old)[active]
            ratio = difference.exp()
            record = self.reuse_ratio_sums.setdefault(self.state.global_step, {"tokens": 0, "abs_log_ratio_sum": 0.0, "max_abs_log_ratio": 0.0, "max_abs_ratio_delta": 0.0, "nonfinite": False})
            record["tokens"] += difference.numel()
            record["abs_log_ratio_sum"] += difference.abs().sum().item()
            record["max_abs_log_ratio"] = max(record["max_abs_log_ratio"], difference.abs().max().item())
            record["max_abs_ratio_delta"] = max(record["max_abs_ratio_delta"], (ratio - 1).abs().max().item())
            record["nonfinite"] |= not torch.isfinite(ratio).all().item()
        return values

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None):
        # 原生轮换早于项目校验；实际轮换由 on_save 完整校验后执行。
        return

    def reuse_state(self):
        if self.state.global_step % 4 or self.reuse_usage is None or any(views != [1, 1, 1, 1] for views in self.reuse_usage):
            raise RuntimeError("Checkpoint is not at a fully consumed reuse-cycle boundary")
        return {"global_step": self.state.global_step, "completed_cycles": self.state.global_step // 4, "committed_questions": self.state.global_step // 4 * 16, "committed_candidates": self.state.global_step // 4 * 64, "safe_cycle_boundary": True, "all_candidates_used_once_per_view": True}
