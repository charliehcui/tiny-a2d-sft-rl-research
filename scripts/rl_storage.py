"""仅处理本次 RL 的校验、两份检查点轮换和单份模型持久化。"""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from run_experiment import read_json, write_json

RESERVE = 512 * 2**20
CHECKPOINT_LIMIT = 96 * 2**20
MODEL_LIMIT = 1216 * 2**20
ADAPTER_LIMIT = 48 * 2**20
LOG_LIMIT = 192 * 2**20
REQUIRED = ["adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "reuse_state.json"]


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def require_space(path, additional=0):
    free = shutil.disk_usage(path).free
    if free < RESERVE + additional:
        raise RuntimeError(f"Persistent space protection: free={free}, required={RESERVE + additional}")
    return free


def checked_path(path, root):
    path = Path(path)
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()) or path.resolve() == Path(root).resolve():
        raise ValueError("Artifact path escaped its new workflow directory")
    return path.resolve()


def files_in(path):
    files = []
    for item in sorted(Path(path).rglob("*")):
        if item.is_symlink():
            raise ValueError("Symlinks are not permitted in saved artifacts")
        if item.is_file():
            files.append(item)
    return files


def verify_checkpoint(path, root):
    path = checked_path(path, root)
    marker = read_json(path / "checkpoint_verified.json")
    complete = read_json(path / "pipeline_complete.json")
    if marker["family_root"] != str(Path(root).resolve()) or complete["verified_manifest_sha256"] != sha256(path / "checkpoint_verified.json"):
        raise ValueError("Checkpoint completion marker mismatch")
    actual = {str(item.relative_to(path)) for item in files_in(path) if item.name not in ("checkpoint_verified.json", "pipeline_complete.json")}
    if actual != set(marker["files"]):
        raise ValueError("Checkpoint file inventory changed")
    for name, item in marker["files"].items():
        file = checked_path(path / name, path)
        if file.stat().st_size != item["bytes"] or sha256(file) != item["sha256"]:
            raise ValueError("Checkpoint checksum failed: " + name)
    return marker


def seal_checkpoint(path, root):
    import torch
    from safetensors.torch import load_file

    path = checked_path(path, root)
    if (path / "checkpoint_verified.json").exists():
        return verify_checkpoint(path, root)
    for name in REQUIRED:
        if not (path / name).is_file():
            raise ValueError("Incomplete native checkpoint: " + name)
    state = read_json(path / "trainer_state.json")
    reuse = read_json(path / "reuse_state.json")
    step = state["global_step"]
    if step % 4 or step != reuse["global_step"] or not reuse["safe_cycle_boundary"]:
        raise ValueError("Checkpoint is not a completed reuse cycle")
    adapter = load_file(str(path / "adapter_model.safetensors"))
    if not adapter or any(not torch.isfinite(value).all() for value in adapter.values()):
        raise ValueError("Nonfinite or empty saved adapter")
    optimizer = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
    counters = [int(item["step"]) for item in optimizer["state"].values() if "step" in item]
    if not counters or set(counters) != {step}:
        raise ValueError("Saved optimizer counters disagree with global step")
    for item in optimizer["state"].values():
        for value in item.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError("Nonfinite optimizer state")
    # 这些文件由本任务原生保存；读取验证结构，不恢复到当前训练器。
    if not isinstance(torch.load(path / "scheduler.pt", map_location="cpu", weights_only=True), dict):
        raise ValueError("Invalid saved scheduler")
    rng = torch.load(path / "rng_state.pth", map_location="cpu", weights_only=False)
    if not {"python", "numpy", "cpu"}.issubset(rng):
        raise ValueError("Incomplete saved random state")
    files = files_in(path)
    if sum(item.stat().st_size for item in files) > CHECKPOINT_LIMIT:
        raise RuntimeError("Checkpoint exceeds the 96 MiB storage budget")
    inventory = {}
    for file in files:
        if file.name in ("checkpoint_verified.json", "pipeline_complete.json"):
            continue
        with file.open("r+b") as source:
            os.fsync(source.fileno())
        inventory[str(file.relative_to(path))] = {"bytes": file.stat().st_size, "sha256": sha256(file)}
    marker = {"global_step": step, "family_root": str(Path(root).resolve()), "config_sha256": reuse["config_sha256"], "schedule_sha256": reuse["schedule_sha256"], "verified_unix_seconds": time.time(), "files": inventory}
    write_json(path / "checkpoint_verified.json", marker)
    write_json(path / "pipeline_complete.json", {"global_step": step, "native_checkpoint_save_completed": True, "verified_manifest_sha256": sha256(path / "checkpoint_verified.json")})
    return verify_checkpoint(path, root)


def rotate_checkpoints(root, current):
    root = Path(root).resolve()
    current = checked_path(current, root)
    latest = verify_checkpoint(current, root)
    complete = []
    incomplete = []
    for path in root.rglob("checkpoint-*"):
        if not path.is_dir() or path.parent.name != "checkpoints":
            continue
        path = checked_path(path, root)
        if not (path / "checkpoint_verified.json").is_file() or not (path / "pipeline_complete.json").is_file():
            incomplete.append(path)
            continue
        marker = verify_checkpoint(path, root)
        if marker["config_sha256"] != latest["config_sha256"] or marker["schedule_sha256"] != latest["schedule_sha256"]:
            raise ValueError("Refusing to rotate another training plan")
        complete.append((marker["global_step"], marker["verified_unix_seconds"], path))
    if len(incomplete) > 1:
        raise RuntimeError("More than one incomplete checkpoint; stop rather than accumulate artifacts")
    complete.sort(key=lambda item: (item[0], item[1]))
    keep = complete[-2:]
    if current not in [item[2] for item in keep]:
        raise ValueError("Current checkpoint is not one of the latest two")
    removed = []
    for _, _, path in complete[:-2]:
        verify_checkpoint(path, root)
        shutil.rmtree(checked_path(path, root))
        removed.append(str(path.relative_to(root)))
    with (root / "checkpoint_rotation.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps({"time": time.time(), "step": latest["global_step"], "kept": [str(item[2].relative_to(root)) for item in keep], "removed": removed}) + "\n")
        log.flush()
        os.fsync(log.fileno())
    require_space(root)
    return [item[2] for item in keep]


def check_log_budget(root):
    excluded = {"checkpoints", "final_adapter", "final_model"}
    total = sum(file.stat().st_size for file in files_in(root) if not excluded.intersection(file.relative_to(root).parts))
    if total > LOG_LIMIT:
        raise RuntimeError("Workflow logs exceed the 192 MiB budget")
    require_space(root, CHECKPOINT_LIMIT)


def copy_verified_model(source, target):
    source = Path(source).resolve()
    target = Path(target)
    files = files_in(source)
    size = sum(file.stat().st_size for file in files)
    if size > MODEL_LIMIT:
        raise RuntimeError("Merged model exceeds the 1216 MiB budget")
    require_space(target.parent, size)
    target.mkdir(exist_ok=False)
    inventory = {}
    for file in files:
        relative = file.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = sha256(file)
        with file.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=2**20)
            writer.flush()
            os.fsync(writer.fileno())
        if destination.stat().st_size != file.stat().st_size or sha256(destination) != expected:
            raise ValueError("Final persistent model copy checksum failed")
        inventory[str(relative)] = {"bytes": file.stat().st_size, "sha256": expected}
    write_json(target / "export_complete.json", {"status": "complete", "files": inventory})
    require_space(target.parent)
    return inventory
