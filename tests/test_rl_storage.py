"""最小真实保存、全局轮换、跨分段恢复与失败文件保护测试。"""

import json
import shutil
import uuid
from pathlib import Path

import test_rl_reuse as fixture
from datasets import Dataset
from transformers import TrainerCallback
from rl_storage import copy_verified_model, rotate_checkpoints, seal_checkpoint, verify_checkpoint
from train_rl_reuse import validate_resume


def main():
    root = fixture.PROJECT / "work/final_storage_tests" / uuid.uuid4().hex
    root.mkdir(parents=True)
    incomplete = root / "checkpoints/checkpoint-999"
    incomplete.mkdir(parents=True)
    (incomplete / "unfinished.bin").write_bytes(b"interrupted native checkpoint")
    rows = [{"id": "storage-" + str(index), "prompt": [{"role": "user", "content": " ".join(["word"] * (2 + index % 16))}]} for index in range(64)]

    class Rotation(fixture.TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            rotate_checkpoints(root, Path(args.output_dir) / ("checkpoint-" + str(state.global_step)))
            return control

    trainer, _, _ = fixture.build_trainer(root, stop=12, family_root=root)
    trainer.args.max_steps = 16
    trainer.train_dataset = Dataset.from_list(rows)
    trainer.add_callback(Rotation())
    trainer.train()
    assert trainer.state.global_step == 12
    assert not (root / "checkpoints/checkpoint-4").exists()
    for step in [8, 12]:
        verify_checkpoint(root / ("checkpoints/checkpoint-" + str(step)), root)
    config = fixture.read_json(fixture.PROJECT / fixture.CONFIG_PATH)
    checkpoint = root / "checkpoints/checkpoint-12"
    assert validate_resume(checkpoint, config, config["data_sha256"])["global_step"] == 12
    resumed, _, rewards = fixture.build_trainer(root / "segments/resumed", family_root=root)
    resumed.args.max_steps = 16
    resumed.train_dataset = Dataset.from_list(rows)
    resumed.add_callback(Rotation())
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 16
    assert rewards.groups == [row["id"] for row in rows[48:]]
    assert not (root / "checkpoints/checkpoint-8").exists()
    latest = root / "segments/resumed/checkpoints/checkpoint-16"
    verify_checkpoint(checkpoint, root)
    verify_checkpoint(latest, root)
    assert (incomplete / "unfinished.bin").is_file()
    verified = [path for path in root.rglob("checkpoint-*") if (path / "checkpoint_verified.json").is_file()]
    assert len(verified) == 2
    bad = root / "bad_candidate"
    shutil.copytree(latest, bad)
    (bad / "checkpoint_verified.json").unlink()
    (bad / "pipeline_complete.json").unlink()
    (bad / "adapter_model.safetensors").write_bytes(b"invalid interrupted weights")
    try:
        seal_checkpoint(bad, root)
    except Exception:
        pass
    else:
        raise AssertionError("Invalid saved model was committed")
    verify_checkpoint(checkpoint, root)
    verify_checkpoint(latest, root)
    export = root / "instance_export_fixture"
    export.mkdir()
    (export / "model.safetensors").write_bytes(b"copy and checksum fixture")
    (export / "config.json").write_text("{}", encoding="utf-8")
    inventory = copy_verified_model(export, root / "persistent_export_fixture")
    assert len(inventory) == 2
    assert (root / "persistent_export_fixture/export_complete.json").is_file()
    facts = {"status": "passed", "real_native_cpu_updates_before_stop": 12, "actual_resume_12_to_16": True, "new_questions_after_resume": rewards.groups, "global_verified_checkpoints": [12, 16], "old_4_and_8_rotated_after_validation": True, "incomplete_checkpoint_preserved": True, "invalid_checkpoint_does_not_delete_good_ones": True, "model_copy_fsync_and_sha256_verified": True, "gpu_used": False, "root": str(root)}
    (root / "storage_checks.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print("STORAGE_CHECKS=" + json.dumps(facts), flush=True)


if __name__ == "__main__":
    main()
