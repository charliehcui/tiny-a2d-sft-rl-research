"""公开仓库不含合并权重；安装依赖后运行 python scripts/merge_adapter.py --stage sft 或 --stage final-rl。"""

import argparse
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['sft', 'final-rl'], required=True)
    args = parser.parse_args()
    sft_model = PROJECT / 'runs/sft-gsm8k-20261104-011915/final_model'
    if args.stage == 'sft':
        base = PROJECT / 'models/Qwen3-0.6B-diffusion-mdlm-v0.1'
        adapter = PROJECT / 'models/sft_adapter'
        output = sft_model
        provenance = PROJECT / 'models/provenance/sft_source_training.json'
        expected = '777eb90450fa3809660f11ce58411f9eceb2081a57a06d06d3917f08a37b412a'
    else:
        base = sft_model
        adapter = PROJECT / 'models/final_rl_adapter'
        output = PROJECT / 'runs/rl-reuse1536-g4-u4-20261105-final01/final_model'
        provenance = PROJECT / 'models/provenance/final_rl_source_training.json'
        expected = 'c1018272ce0182765fdb1907c0684b38dbf4eb1fc17ec70020ea7950e9020f80'
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing model: ' + str(output))
    if not (base / 'model.safetensors').is_file():
        raise FileNotFoundError('Download the fixed base or restore SFT first: ' + str(base))
    import dllm
    from peft import PeftModel
    from transformers import AutoTokenizer

    model = dllm.utils.get_model(model_name_or_path=str(base), dtype='bfloat16', attn_implementation='sdpa')
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model = model.merge_and_unload(safe_merge=True)
    if model.config.model_type != 'a2d-qwen3':
        raise ValueError('The restored model must retain the diffusion architecture')
    model.config.use_cache = False
    output.mkdir(parents=True)
    model.save_pretrained(output, safe_serialization=True, max_shard_size='2GB')
    AutoTokenizer.from_pretrained(adapter, local_files_only=True).save_pretrained(output)
    (output / 'source_training.json').write_bytes(provenance.read_bytes())
    with (output / 'model.safetensors').open('rb') as handle:
        actual = hashlib.file_digest(handle, 'sha256').hexdigest()
    result = {'stage': args.stage, 'sha256': actual, 'original_merged_sha256': expected, 'matches_original_weight_bytes': actual == expected, 'note': 'Reconstruction can differ across devices or library implementations. A different hash is not evidence of reproducing the saved historical metrics; rerun evaluation.'}
    (output / 'restoration.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
