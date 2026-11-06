"""给已有 Hugging Face Trainer / DiffuGRPOTrainer 添加进度记录，不实现训练算法。"""

import os
import json
import signal
import time
from pathlib import Path

from transformers import TrainerCallback
from run_experiment import write_json


class ExperimentCallback(TrainerCallback):
    def __init__(self):
        self.run_dir = Path(os.environ["BLUEWHALE_RUN_DIR"])
        self.stop_requested = False
        self.initial_step = 0
        self.old_handlers = {}

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if args.save_strategy != "steps" or args.save_only_model or args.save_steps <= 0 or args.save_total_limit != 2:
            raise ValueError("Use save_strategy=steps, save_steps>0, save_only_model=False, save_total_limit=2")
        if not Path(args.output_dir).resolve().is_relative_to(self.run_dir.resolve()):
            raise ValueError("Training checkpoints must be inside this persistent run directory")
        self.initial_step = state.global_step
        for number in (signal.SIGTERM, signal.SIGINT):
            self.old_handlers[number] = signal.signal(number, self.request_save)
        self.on_step_end(args, state, control)

    def request_save(self, number, frame):
        self.stop_requested = True

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and (state.global_step % max(1, int(args.logging_steps)) == 0 or state.global_step in (self.initial_step, state.max_steps) or self.stop_requested):
            write_json(self.run_dir / "training_progress.json", {"completed": state.global_step, "initial_step": self.initial_step, "total": state.max_steps, "epoch": state.epoch, "unit": "step", "updated_unix_seconds": time.time(), "stop_requested": self.stop_requested})
        if self.stop_requested:
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps({"step": state.global_step, "metrics": logs or {}}, default=str) + "\n")
                output.flush()
                os.fsync(output.fileno())

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            checkpoint = Path(args.output_dir) / ("checkpoint-" + str(state.global_step))
            # 原生保存成功之后才写标记；恢复时忽略没有该标记的半成品目录。
            write_json(checkpoint / "pipeline_complete.json", {"global_step": state.global_step, "native_checkpoint_save_completed": True})

    def on_train_end(self, args, state, control, **kwargs):
        for number, handler in self.old_handlers.items():
            signal.signal(number, handler)
