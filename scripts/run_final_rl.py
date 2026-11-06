"""一次性后台执行固定 RL、冻结开发评测和一次官方测试；不管理实例生命周期。"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from prepare_rl_reuse import PROJECT, load_reuse_rows
from rl_storage import CHECKPOINT_LIMIT, LOG_LIMIT, check_log_budget, files_in, require_space, sha256, verify_checkpoint
from run_experiment import read_json, write_json


def assert_training_complete(run_dir):
    summary = read_json(run_dir / "training_summary.json")
    expected = {"status": "complete", "global_step": 384, "completed_cycles": 96, "committed_questions": 1536, "committed_candidates": 6144, "new_questions": 1536, "new_candidates": 6144, "physical_generation_calls": 1536}
    if any(summary.get(name) != value for name, value in expected.items()):
        raise RuntimeError("Final evaluation requires all 384 updates and the exact full schedule")
    verify_checkpoint(run_dir / "checkpoints/checkpoint-384", run_dir)
    source = read_json(run_dir / "final_model/source_training.json")
    if source["rl_adapter_sha256"] != sha256(run_dir / "final_adapter/adapter_model.safetensors"):
        raise ValueError("Final adapter does not match merged-model provenance")
    exported = read_json(run_dir / "final_model/export_complete.json")
    for name, item in exported["files"].items():
        path = run_dir / "final_model" / name
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("Frozen final model export changed")
    return summary


def evaluation_complete(folder, split, examples, protocol_sha256):
    summary = read_json(folder / "summary.json")
    if summary.get("status") != "complete" or summary.get("split") != split or summary.get("examples") != examples or summary.get("protocol_sha256") != protocol_sha256:
        raise ValueError("Evaluation did not finish the exact frozen split/protocol")
    rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    expected = [json.loads(line)["id"] for line in (PROJECT / f"data/gsm8k/processed/eval/{split}_prompts.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != examples or len({row["id"] for row in rows}) != examples or {row["id"] for row in rows} != set(expected):
        raise ValueError("Evaluation predictions have missing/duplicate IDs")
    if sum(bool(row["scores"]["flexible-extract"]["correct"]) for row in rows) != summary["correct"]:
        raise ValueError("Evaluation score disagrees with individual predictions")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if run_dir != Path(os.environ.get("BLUEWHALE_RUN_DIR", "")).resolve():
        raise ValueError("Use the detached experiment launcher")
    if (run_dir / "workflow_budget.json").exists():
        raise FileExistsError("Never automatically retry an existing final workflow")
    config = read_json(args.config)
    if not config.get("formal_training_enabled") or not config.get("checkpoint_rotation_enabled"):
        raise PermissionError("Authorized formal training and verified rotation must be enabled")
    load_reuse_rows(PROJECT, config)
    if sha256(PROJECT / config["evaluation_protocol"]) != config["evaluation_protocol_sha256"]:
        raise ValueError("Frozen evaluation protocol changed")
    require_space(run_dir, int((config["minimum_free_gib"] * 2**30) - 512 * 2**20))
    started = time.time()
    deadline = started + 18 * 3600
    write_json(run_dir / "workflow_budget.json", {"started_unix_seconds": started, "deadline_unix_seconds": deadline, "maximum_allocated_seconds": 18 * 3600, "scope": "Entire workflow, including loading, training, export and both evaluations; no retry resets", "soft_stop_reserve_seconds": 900, "hard_stop_reserve_seconds": 60})
    environment = os.environ.copy()
    environment.update(BLUEWHALE_WORKFLOW_DIR=str(run_dir), BLUEWHALE_TASK_DEADLINE=str(deadline), TMPDIR="/opt/bluewhale_cache", PYTHONDONTWRITEBYTECODE="1")
    stop = {"signal": None}

    def receive_signal(number, frame):
        stop["signal"] = number

    for number in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(number, receive_signal)
    phases = []
    final = {"status": "running", "training_complete": False, "official_test_attempted": False}
    worker = None

    def execute(name, command, output=None, expected_examples=None):
        nonlocal worker
        remaining = deadline - time.time()
        required = 3000 if name == "official_test" else 4200 if name == "development" else 900
        if remaining < required or stop["signal"]:
            raise TimeoutError("Insufficient cumulative budget to start " + name)
        if name == "official_test":
            marker = run_dir / "official_test_attempt.json"
            with marker.open("x", encoding="utf-8") as stream:
                json.dump({"started_unix_seconds": time.time(), "protocol_sha256": config["evaluation_protocol_sha256"], "final_model_sha256": sha256(run_dir / "final_model/model.safetensors"), "no_retry": True}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            final["official_test_attempted"] = True
        phase = {"phase": name, "status": "running", "started_unix_seconds": time.time(), "command": command}
        write_json(run_dir / "phase.json", phase)
        print(json.dumps({"event": "phase_started", "phase": name}), flush=True)
        worker = subprocess.Popen(command, cwd=PROJECT, env=environment, stdin=subprocess.DEVNULL)
        phase["pid"] = worker.pid
        write_json(run_dir / "phase.json", phase)
        notified = False
        while worker.poll() is None:
            now = time.time()
            if now >= deadline - 60:
                worker.kill()
                worker.wait()
                raise TimeoutError("18-hour cumulative hard budget; latest verified checkpoints retained")
            if (stop["signal"] or (name == "training" and now >= deadline - 900)) and not notified:
                worker.send_signal(signal.SIGTERM)
                notified = True
            if output is not None:
                predictions = output / "predictions.jsonl"
                completed = sum(1 for line in predictions.open(encoding="utf-8")) if predictions.exists() else 0
                write_json(run_dir / "evaluation_progress.json", {"phase": name, "completed": completed, "total": expected_examples, "updated_unix_seconds": now})
            require_space(run_dir)
            time.sleep(5)
        code = worker.returncode
        phase.update(status="complete" if code == 0 else "failed", exit_code=code, elapsed_seconds=time.time() - phase["started_unix_seconds"])
        phases.append(phase)
        write_json(run_dir / "phase.json", phase)
        if code != 0 or notified:
            raise RuntimeError("Phase stopped without full success: " + name)
        check_log_budget(run_dir)

    try:
        history_path = PROJECT / config["historical_results"]
        if sha256(history_path) != config["historical_results_sha256"]:
            raise ValueError("Historical comparison summaries changed")
        history = read_json(history_path)
        write_json(run_dir / "historical_comparison_data.json", history)
        snapshot = run_dir / "code"
        snapshot.mkdir(exist_ok=True)
        for name in ["run_final_rl.py", "rl_storage.py", "evaluate_model.py"]:
            destination = snapshot / name
            if destination.exists():
                raise FileExistsError(destination)
            destination.write_bytes((PROJECT / "scripts" / name).read_bytes())
        write_json(run_dir / "frozen_execution_manifest.json", {"protocol_sha256": config["evaluation_protocol_sha256"], "data_sha256": config["data_sha256"], "code_sha256": {name: sha256(PROJECT / "scripts" / name) for name in ["run_final_rl.py", "rl_storage.py", "train_rl_reuse.py", "rl_reuse_trainer.py", "evaluate_model.py"]}})
        execute("training", [sys.executable, "-u", str(PROJECT / "scripts/train_rl_reuse.py"), "--config", str(args.config), "--run-dir", str(run_dir), "--train-only"])
        assert_training_complete(run_dir)
        final["training_complete"] = True
        evaluator = PROJECT / "scripts/evaluate_model.py"
        for name, split, count in [("development", "validation", 500), ("official_test", "test", 1319)]:
            require_space(run_dir, 32 * 2**20)
            execute(name, [sys.executable, "-u", str(evaluator), "--split", split, "--output", str(run_dir / name), "--checkpoint", str(run_dir / "final_model"), "--protocol", str(PROJECT / config["evaluation_protocol"])], run_dir / name, count)
            evaluation_complete(run_dir / name, split, count, config["evaluation_protocol_sha256"])
        development = read_json(run_dir / "development/summary.json")
        official = read_json(run_dir / "official_test/summary.json")
        write_json(run_dir / "comparison_summary.json", {"status": "complete", "historical": history, "final_rl": {"development": development, "official_test": official}, "accuracy_is_not_used_for_selection": True, "official_attempts": 1, "final_optimizer_step": 384, "historical_protocol_differences_are_retained": True})
        final["status"] = "complete"
    except BaseException:
        final.update(status="failed_or_stopped", error=traceback.format_exc())
        if worker is not None and worker.poll() is None:
            worker.send_signal(signal.SIGTERM)
            try:
                worker.wait(timeout=max(1, min(900, deadline - time.time() - 60)))
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        raise
    finally:
        # 强制结束计算进程时，清理同一后台进程组内可能遗留的资源监控器。
        for process_path in Path("/proc").iterdir():
            if not process_path.name.isdigit():
                continue
            try:
                pid = int(process_path.name)
                if os.getpgid(pid) == os.getpgrp() and (process_path / "comm").read_text().strip() == "nvidia-smi":
                    os.kill(pid, signal.SIGTERM)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        elapsed = time.time() - started
        final.update(phases=phases, allocated_task_wall_seconds=elapsed, gpu_time_definition="Elapsed allocation time of this workflow, including GPU-associated loading, saving and evaluation; not summed CUDA kernel time", configured_price_cny_per_hour=config["gpu_price_cny_per_hour"], estimated_task_cost_cny=elapsed / 3600 * config["gpu_price_cny_per_hour"], actual_invoice_cost_cny=None, cost_note="Platform invoice is authoritative; estimate excludes preparation and instance idle billing after job completion", maximum_seconds=18 * 3600, finished_unix_seconds=time.time(), instance_lifecycle_untouched=True)
        write_json(run_dir / "workflow_summary.json", final)
        print(json.dumps({"event": "workflow_finished", "status": final["status"], "seconds": elapsed}), flush=True)


if __name__ == "__main__":
    main()
