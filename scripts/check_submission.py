"""从提交根目录运行 python scripts/check_submission.py；只读检查，不训练、不推理。"""

import argparse
import collections
import csv
import hashlib
import json
import math
import re
from pathlib import Path

from prepare_rl_reuse import load_reuse_rows
from prepare_rl_static import load_static_rows, read_rows
from run_experiment import read_json

PROJECT = Path(__file__).resolve().parents[1]
RUNS = {'sft': 'sft-gsm8k-20261104-011915', 'rl_pilot1': 'rl-gsm8k-512-g4-20261104-110156', 'rl_pilot2': 'rl-static512-g4-20261105-013032', 'final_rl': 'rl-reuse1536-g4-u4-20261105-final01'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def check_weights(path):
    with path.open('rb') as handle:
        header_length = int.from_bytes(handle.read(8), 'little')
        require(0 < header_length < 16 * 2**20, 'Invalid safetensors header')
        header = json.loads(handle.read(header_length))
    byte_sizes = {'F32': 4, 'BF16': 2, 'F16': 2, 'I64': 8}
    offset = 0
    parameters = 0
    tensors = []
    for name, value in header.items():
        if name != '__metadata__':
            tensors.append((name, value))
    tensors.sort(key=lambda item: item[1]['data_offsets'][0])
    for name, value in tensors:
        start, end = value['data_offsets']
        count = math.prod(value['shape'])
        require(start == offset and end - start == count * byte_sizes[value['dtype']], 'Invalid tensor span: ' + name)
        parameters += count
        offset = end
    require(offset + 8 + header_length == path.stat().st_size, 'Safetensors is truncated or has trailing bytes')
    return {'tensors': len(tensors), 'parameters': parameters, 'bytes': path.stat().st_size, 'sha256': digest(path), 'dtypes': sorted({value['dtype'] for _, value in tensors})}


def extract(response, options):
    matches = re.findall(options['regex_pattern'], response)
    if not matches:
        return '[invalid]'
    value = matches[options.get('group_select', 0)]
    if isinstance(value, tuple):
        value = next((part for part in value if part), '[invalid]')
    return value.strip()


def normalized(value, metric):
    for pattern in metric['regexes_to_ignore']:
        value = re.sub(pattern, '', value)
    return value.lower() if metric['ignore_case'] else value


def check_scores(rows, task):
    metric = task['metric_list'][0]
    counts = collections.Counter()
    for row in rows:
        for item in task['filter_list']:
            name = item['name']
            answer = extract(row['response'], item['filter'][0])
            correct = normalized(answer, metric) == normalized(row['reference_answer'], metric)
            require(row['scores'][name] == {'extracted_answer': answer, 'correct': correct}, 'Stored score differs from response: ' + row['id'])
            counts[name] += int(correct)
            if name == 'xml-answer':
                counts['xml_extractor_success'] += int(answer != '[invalid]')
    return dict(counts)


def verify(adapters_only=False):
    task = read_json(PROJECT / 'configs/data_tasks/gsm8k_bluewhale_xml.yaml')
    protocol_hash = digest(PROJECT / 'configs/baseline_protocol_v2.json')
    result = {'status': 'passed', 'scope': 'Offline files, saved response rescoring and schedule checks; no model inference', 'evaluations': {}, 'training': {}, 'weights': {}, 'manifest': {}}
    for folder in sorted((PROJECT / 'results').glob('*/*')):
        if not (folder / 'summary.json').is_file():
            continue
        summary = read_json(folder / 'summary.json')
        rows = read_rows(folder / 'predictions.jsonl')
        labels = read_rows(PROJECT / 'data/gsm8k/processed/eval' / (summary['split'] + '_labels.jsonl'))
        labels_by_id = {row['id']: row for row in labels}
        require(len(rows) == summary['examples'] and len({row['id'] for row in rows}) == len(rows), 'Evaluation count/ID mismatch')
        if len(rows) > 8:
            require({row['id'] for row in rows} == set(labels_by_id), 'Incomplete evaluation split')
        for row in rows:
            require(row['reference_answer'] == labels_by_id[row['id']]['answer_value'], 'Evaluation reference differs from frozen label')
        scores = check_scores(rows, task)
        require(scores['flexible-extract'] == summary['correct'], 'Correct count differs from summary')
        require(abs(scores['flexible-extract'] / len(rows) - summary['accuracy']) < 1e-12, 'Summary accuracy mismatch')
        require(abs(scores['xml-answer'] / len(rows) - summary['xml_answer_accuracy']) < 1e-12, 'XML accuracy mismatch')
        require(summary['protocol_sha256'] == protocol_hash, 'Frozen evaluation protocol changed')
        native = read_json(folder / 'native_results.json')['results']
        values = next(iter(native.values()))
        require(abs(values['exact_match,flexible-extract'] - summary['accuracy']) < 1e-12, 'Native primary metric mismatch')
        require(abs(values['exact_match,xml-answer'] - summary['xml_answer_accuracy']) < 1e-12, 'Native XML metric mismatch')
        result['evaluations'][folder.relative_to(PROJECT).as_posix()] = {'rows': len(rows), 'correct': summary['correct'], 'accuracy': summary['accuracy'], **scores}
    total_candidates = 0
    total_steps = 0
    for label, name in RUNS.items():
        root = PROJECT / 'runs' / name
        summary = read_json(root / 'training_summary.json')
        state = read_json(root / 'checkpoints/trainer_state.json')
        steps = summary.get('optimizer_steps', summary.get('global_step'))
        require(state['global_step'] == steps, 'Trainer global step mismatch')
        metrics = read_rows(root / 'metrics.jsonl')
        if label == 'sft':
            require(steps == 626, 'Expected SFT 626 steps')
            result['training'][label] = {'optimizer_steps': steps, 'max_logged_step': max(row['step'] for row in metrics)}
            continue
        rollouts = read_rows(root / 'rollouts.jsonl')
        groups = read_rows(root / 'reward_groups.jsonl')
        scores = check_scores(rollouts, task)
        require(len(rollouts) == len(groups) * 4, 'Candidate/group count mismatch')
        require(max(row['step'] for row in metrics) == steps, 'Metrics do not reach final step')
        grouped = collections.defaultdict(list)
        for row in rollouts:
            require(row['reward'] == float(row['scores']['flexible-extract']['correct']), 'Reward differs from correctness')
            grouped[row['group']].append(row)
        for group in groups:
            candidates = grouped[group['group']]
            rewards = [row['reward'] for row in candidates]
            mean = sum(rewards) / 4
            require(len(candidates) == 4 and {row['candidate'] for row in candidates} == set(range(4)), 'Invalid candidate indices')
            require(sum(rewards) == group['correct_candidates'] and mean == group['mean_reward'], 'Reward group mismatch')
            require([reward - mean for reward in rewards] == group['advantages'], 'Group advantage mismatch')
        mixed = sum(0 < group['correct_candidates'] < 4 for group in groups)
        if label == 'final_rl':
            require(len(rollouts) == summary['committed_candidates'] == 6144 and steps == 384, 'Incomplete Final RL')
            updates = summary['optimizer_update_evidence']
            require({row['step'] for row in metrics if 'loss' in row['metrics']} == set(range(1, 385)), 'Missing per-step training metrics')
            require(set(summary['ratios_by_step_before_update']) == {str(step) for step in range(384)}, 'Missing per-step probability statistics')
            require(len(summary['generation_events']) == 96, 'Missing generation cycle events')
            result['training'][label] = {'optimizer_steps': steps, 'new_candidates': len(rollouts), 'groups': len(groups), 'mixed_groups': mixed, 'mean_reward': sum(row['reward'] for row in rollouts) / len(rollouts), 'per_step_adapter_hash_records': len(updates), 'per_step_metrics_and_probability_records': 384, **scores}
        else:
            require(mixed == summary['mixed_reward_groups'] and len(rollouts) == summary['candidates_generated'] == 2048 and steps == 32, 'Incomplete Pilot evidence')
            result['training'][label] = {'optimizer_steps': steps, 'new_candidates': len(rollouts), 'groups': len(groups), 'mixed_groups': mixed, 'mean_reward': sum(row['reward'] for row in rollouts) / len(rollouts), **scores}
        total_candidates += len(rollouts)
        total_steps += steps
    result['training']['rl_totals'] = {'new_candidates': total_candidates, 'optimizer_steps': total_steps}
    for path in (PROJECT / 'models').glob('*_adapter/adapter_model.safetensors'):
        record = check_weights(path)
        require(record['parameters'] == 5046272, 'Unexpected adapter parameter count')
        result['weights'][path.relative_to(PROJECT).as_posix()] = record
    for label in ['sft', 'final_rl']:
        folder = PROJECT / 'runs' / RUNS[label] / 'final_model'
        weights = folder / 'model.safetensors'
        if adapters_only:
            result['weights'][weights.relative_to(PROJECT).as_posix()] = {'status': 'omitted_from_public_repository', 'reason': 'GB-sized merged model; restore from the included adapters before evaluation'}
            continue
        result['weights'][weights.relative_to(PROJECT).as_posix()] = check_weights(weights)
        require(read_json(folder / 'config.json')['model_type'] == 'a2d-qwen3', 'Lost diffusion model type')
        provenance = read_json(folder / 'source_training.json')
        field = 'adapter_sha256' if label == 'sft' else 'rl_adapter_sha256'
        adapter = PROJECT / 'models' / ('sft_adapter' if label == 'sft' else 'final_rl_adapter') / 'adapter_model.safetensors'
        require(provenance[field] == digest(adapter), 'Model/adapter provenance mismatch')
        if (folder / 'export_complete.json').is_file():
            for name, item in read_json(folder / 'export_complete.json')['files'].items():
                require((folder / name).stat().st_size == item['bytes'] and digest(folder / name) == item['sha256'], 'Final export checksum mismatch: ' + name)
    static = load_static_rows(PROJECT, read_json(PROJECT / 'configs/rl_static_tiny_a2d.json'))
    reuse = load_reuse_rows(PROJECT, read_json(PROJECT / 'configs/rl_reuse_tiny_a2d.json'))
    result['schedules'] = {'static_presentations': len(static), 'static_unique_questions': len({row['id'] for row in static}), 'reuse_unique_questions': len({row['id'] for row in reuse})}
    manifest = PROJECT / 'SUBMISSION_MANIFEST.csv'
    if manifest.exists():
        with manifest.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            path = (PROJECT / row['path']).resolve()
            require(path.is_relative_to(PROJECT) and path.is_file(), 'Missing/unsafe manifest path')
            require(path.stat().st_size == int(row['bytes']) and digest(path) == row['sha256'], 'Submission manifest mismatch: ' + row['path'])
        result['manifest'] = {'checked_files': len(rows), 'status': 'passed'}
    else:
        result['manifest'] = {'status': 'not_created_yet'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='可选：将本次检查结果保存到一个新 JSON 文件')
    parser.add_argument('--adapters-only', action='store_true', help='公开仓库未包含 GB 级合并模型时，只核验已提交 Adapter')
    args = parser.parse_args()
    result = verify(args.adapters_only)
    if args.output:
        with args.output.open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
