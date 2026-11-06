"""在云端独立运行已有命令，记录进度和产物；默认由用户手动暂停实例。"""

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
MATPOOL_CLI = Path("/public/script/matncli")
MATPOOL_URL = "https://matpool.com/api/public/node"
ACTIVE_LOCK = Path("/dev/shm/bluewhale-experiment")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def cancel_token():
    token = os.environ.get("MATCLOUD_CANCELTOKEN")
    if token:
        return token
    for item in Path("/proc/1/environ").read_bytes().split(b"\0"):
        if item.startswith(b"MATCLOUD_CANCELTOKEN="):
            return os.fsdecode(item.split(b"=", 1)[1])
    raise RuntimeError("MatPool cancellation token is unavailable; no token was logged")


def check_stop(stop, project):
    mode = stop.get("mode")
    if mode == "disabled":
        return
    if mode != "save_release":
        raise ValueError("Only disabled or the documented save_release action is supported")
    if not stop.get("cloud_validated"):
        raise RuntimeError("Automatic stopping has not been validated on this instance; launch refused")
    if os.name != "posix" or not project.resolve().is_relative_to(Path("/mnt")) or not os.path.ismount("/mnt"):
        raise RuntimeError("Automatic stopping requires a Linux instance and its persistent /mnt mount")
    if not MATPOOL_CLI.is_file() or not os.access(MATPOOL_CLI, os.X_OK):
        raise RuntimeError("The official MatPool CLI is unavailable")
    reserve = stop.get("snapshot_min_free_gib")
    if not isinstance(reserve, (int, float)) or reserve <= 0:
        raise ValueError("Measure and configure snapshot_min_free_gib before unattended use")
    if shutil.disk_usage(project).free < reserve * 2**30:
        raise RuntimeError("Insufficient persistent storage for the configured environment snapshot reserve")
    cancel_token()


def request_stop(stop, project, run_id, state, status_path):
    if stop["mode"] == "disabled":
        state["stop"] = {"status": "disabled", "billing_stopped_confirmed": False}
        write_json(status_path, state)
        return
    try:
        check_stop(stop, project)
        state["stop"] = {"status": "requested", "action": "save_environment_then_release", "requested_utc": utc_now(), "billing_stopped_confirmed": False}
        write_json(status_path, state)
        environment = os.environ.copy()
        environment["MATCLOUD_CANCELTOKEN"] = cancel_token()
        command = [str(MATPOOL_CLI), "node", "cancel", "-url", MATPOOL_URL, "-save", "-name", "bluewhale-" + run_id]
        # 不记录命令输出，避免平台工具将令牌写入日志。没有不保存环境的降级释放。
        result = subprocess.run(command, env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=stop.get("request_timeout_seconds", 120), check=False)
        state["stop"]["status"] = "command_returned" if result.returncode == 0 else "request_failed"
        state["stop"]["exit_code"] = result.returncode
    except Exception as error:
        state["stop"] = {"status": "request_failed", "error_type": type(error).__name__, "reason": str(error), "billing_stopped_confirmed": False}
    write_json(status_path, state)


def validate_job(job):
    if job.get("enabled") is not True:
        raise ValueError("This job is a disabled template; enable it only after user approval")
    if job.get("stage") not in ("baseline", "sft", "rl"):
        raise ValueError("stage must be baseline, sft, or rl")
    command = job.get("command")
    if not isinstance(command, list) or not command or any(not isinstance(item, str) for item in command):
        raise ValueError("command must be a nonempty argument list, not a shell string")
    if not isinstance(job.get("max_runtime_minutes"), (int, float)) or job["max_runtime_minutes"] <= 0:
        raise ValueError("A positive max_runtime_minutes is required")
    for key in ("poll_seconds", "stop_grace_seconds", "min_free_gib"):
        if not isinstance(job.get(key), (int, float)) or job[key] <= 0:
            raise ValueError(key + " must be positive")
    progress = job.get("progress", {})
    if progress.get("kind") not in ("jsonl", "trainer"):
        raise ValueError("progress.kind must be jsonl or trainer")
    for name in [progress.get("path", "")] + job.get("required_outputs", []):
        path = Path(name)
        if not name or path.is_absolute() or ".." in path.parts:
            raise ValueError("Artifact paths must remain inside the new run directory")
    if progress["kind"] == "jsonl" and (not isinstance(progress.get("total"), int) or progress["total"] <= 0):
        raise ValueError("JSONL progress requires a positive total")


def release_lock(run_id):
    owner_path = ACTIVE_LOCK / "owner.json"
    if owner_path.exists() and read_json(owner_path).get("run_id") == run_id:
        owner_path.unlink()
        ACTIVE_LOCK.rmdir()


def progress_value(run_dir, job):
    progress = job["progress"]
    path = run_dir / progress["path"]
    if progress["kind"] == "jsonl":
        completed = 0
        if path.exists():
            with path.open("rb") as source:
                completed = sum(line.endswith(b"\n") for line in source)
        return {"completed": completed, "total": progress["total"], "unit": "question"}
    if path.exists():
        return read_json(path)
    return {"completed": 0, "total": None, "unit": "step"}


def stop_child(process, grace_seconds):
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def run_job(run_dir, project=PROJECT):
    from tqdm import tqdm

    job = read_json(run_dir / "job.json")
    stop = read_json(run_dir / "stop_config.json")
    validate_job(job)
    status_path = run_dir / "status.json"
    state = {"run_id": run_dir.name, "stage": job["stage"], "status": "starting", "started_utc": utc_now(), "pid": os.getpid(), "stop": {"status": "not_requested", "billing_stopped_confirmed": False}}
    process = None
    interrupted = {"signal": None}
    previous_handlers = {}
    started = time.monotonic()
    bar = None

    def remember_signal(number, frame):
        interrupted["signal"] = number

    for number in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[number] = signal.signal(number, remember_signal)
    try:
        check_stop(stop, project)
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["BLUEWHALE_RUN_DIR"] = str(run_dir)
        environment["PYTHONPATH"] = str(project / "scripts") + os.pathsep + environment.get("PYTHONPATH", "")
        with (run_dir / "task.log").open("x", encoding="utf-8") as task_log:
            process = subprocess.Popen(job["command"], cwd=project, env=environment, stdin=subprocess.DEVNULL, stdout=task_log, stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
            state.update(status="running", child_pid=process.pid)
            write_json(status_path, state)
            while True:
                progress = progress_value(run_dir, job)
                if bar is None or progress.get("initial_step", 0) > bar.n:
                    if bar is not None:
                        bar.close()
                    initial = progress.get("initial_step", 0)
                    bar = tqdm(total=progress["total"], initial=initial, unit=progress["unit"], desc=job["stage"], ascii=True, file=sys.stdout, mininterval=job["poll_seconds"])
                bar.total = progress["total"]
                if progress.get("epoch") is not None:
                    bar.set_postfix(epoch=round(progress["epoch"], 3), refresh=False)
                advance = max(0, progress["completed"] - bar.n)
                if advance:
                    bar.update(advance)
                else:
                    bar.refresh()
                elapsed = time.monotonic() - started
                rate = max(0, progress["completed"] - progress.get("initial_step", 0)) / max(elapsed, 0.001)
                total = progress["total"]
                progress.update(percent=100 * progress["completed"] / total if total else None, rate_per_second=rate, eta_seconds=(total - progress["completed"]) / rate if total and rate else None)
                state.update(progress=progress, elapsed_seconds=elapsed, updated_utc=utc_now())
                write_json(status_path, state)
                code = process.poll()
                if code is not None:
                    state.update(exit_code=code, status="succeeded" if code == 0 else "failed")
                    break
                if interrupted["signal"] or elapsed > job["max_runtime_minutes"] * 60:
                    stop_child(process, job["stop_grace_seconds"])
                    state.update(status="interrupted" if interrupted["signal"] else "timed_out", exit_code=process.returncode)
                    break
                time.sleep(job["poll_seconds"])
            task_log.flush()
            os.fsync(task_log.fileno())
        if state["status"] == "succeeded":
            for name in job.get("required_outputs", []):
                if not (run_dir / name).is_file():
                    raise RuntimeError("Required output is missing: " + name)
            if job["stage"] == "baseline":
                summary = read_json(run_dir / "evaluation/summary.json")
                if summary.get("status") != "complete" or summary.get("examples") != job["progress"]["total"]:
                    raise RuntimeError("Baseline did not finish the configured evaluation set")
            elif not list((run_dir / "checkpoints").glob("checkpoint-*/pipeline_complete.json")):
                raise RuntimeError("No completed native training checkpoint was recorded")
        state.setdefault("progress", {}).update(progress_value(run_dir, job))
        state["artifacts"] = [{"path": str(path.relative_to(run_dir)), "bytes": path.stat().st_size} for path in sorted(run_dir.rglob("*")) if path.is_file() and not path.name.endswith(".partial")]
    except Exception:
        if process is not None:
            stop_child(process, job["stop_grace_seconds"])
        state.update(status="failed", error=traceback.format_exc())
    finally:
        if bar is not None:
            bar.close()
        state["finished_utc"] = utc_now()
        # 保存状态失败时不触发释放，以保留容器盘上的环境和故障现场。
        try:
            write_json(status_path, state)
            sys.stdout.flush()
            if hasattr(os, "sync"):
                os.sync()
            request_stop(stop, project, run_dir.name, state, status_path)
        finally:
            for number, handler in previous_handlers.items():
                signal.signal(number, handler)
            release_lock(run_dir.name)
    return 0 if state["status"] == "succeeded" else 1


def launch(job_path, stop_path, run_id=None, project=PROJECT):
    if os.name != "posix":
        raise RuntimeError("Launch on the cloud Linux instance; this command does not resume or connect to it")
    job = read_json(job_path)
    stop = read_json(stop_path)
    validate_job(job)
    check_stop(stop, project)
    reserve = stop.get("snapshot_min_free_gib", 0) if stop["mode"] == "save_release" else 0
    if shutil.disk_usage(project).free < (job["min_free_gib"] + reserve) * 2**30:
        raise RuntimeError("Insufficient persistent storage for job outputs plus snapshot reserve")
    frozen = read_json(project / "configs/baseline_final_manifest.json")
    for name, expected in frozen["files_sha256"].items():
        if hashlib.sha256((project / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("A frozen Baseline file changed: " + name)
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + job["stage"] + "-" + uuid.uuid4().hex[:6]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
        raise ValueError("Invalid run ID")
    run_dir = project / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(run_dir)
    ACTIVE_LOCK.mkdir(exist_ok=False)
    try:
        # 防止两个本项目任务互相释放正在使用的实例；不自动接管残留锁。
        write_json(ACTIVE_LOCK / "owner.json", {"run_id": run_id})
        run_dir.mkdir(parents=True, exist_ok=False)
        values = {"project": str(project), "run_dir": str(run_dir), "python": sys.executable}
        for index, item in enumerate(job["command"]):
            for name, value in values.items():
                item = item.replace("{" + name + "}", value)
            job["command"][index] = item
        write_json(run_dir / "job.json", job)
        write_json(run_dir / "stop_config.json", stop)
        snapshot = run_dir / "code"
        snapshot.mkdir()
        for source in [project / "scripts/run_experiment.py", project / "scripts/run_baseline.py", project / "scripts/experiment_callback.py", project / "configs/baseline_protocol_v2.json", project / "configs/data_tasks/gsm8k_bluewhale_xml.yaml"]:
            shutil.copy2(source, snapshot / source.name)
        with (run_dir / "launcher.log").open("x", encoding="utf-8") as output:
            worker = subprocess.Popen([sys.executable, "-u", str(project / "scripts/run_experiment.py"), "run", str(run_dir)], cwd=project, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    except Exception:
        release_lock(run_id)
        raise
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status_path = run_dir / "status.json"
        if status_path.exists():
            state = read_json(status_path)
            if state["status"] == "running":
                print(json.dumps({"run_id": run_id, "pid": worker.pid, "status": "running", "run_dir": str(run_dir), "auto_stop": stop["mode"]}), flush=True)
                return run_dir
            if state["status"] in ("succeeded", "failed", "interrupted", "timed_out"):
                print(json.dumps(state), flush=True)
                if state["status"] != "succeeded":
                    raise RuntimeError("Job ended during launch; inspect its saved logs")
                return run_dir
        if worker.poll() is not None:
            raise RuntimeError("Worker exited before startup confirmation; inspect " + str(run_dir / "launcher.log"))
        time.sleep(0.2)
    raise RuntimeError("Startup was not confirmed; inspect the run directory before starting another job")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    start = commands.add_parser("launch")
    start.add_argument("job", type=Path)
    start.add_argument("--stop-config", type=Path, default=PROJECT / "configs/matpool_manual_stop.json")
    start.add_argument("--id")
    run = commands.add_parser("run")
    run.add_argument("directory", type=Path)
    status = commands.add_parser("status")
    status.add_argument("id")
    args = parser.parse_args()
    if args.action == "launch":
        launch(args.job, args.stop_config, args.id)
    elif args.action == "run":
        sys.exit(run_job(args.directory))
    else:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.id):
            raise ValueError("Invalid run ID")
        print(json.dumps(read_json(PROJECT / "runs" / args.id / "status.json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
