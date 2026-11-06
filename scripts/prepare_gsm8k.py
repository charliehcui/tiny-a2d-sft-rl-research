"""下载、划分并检查 GSM8K；不加载模型，不执行训练。"""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
import urllib.request
from decimal import Decimal
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
LOCAL_LIBRARIES = PROJECT / "work" / "pythonlibs"
if LOCAL_LIBRARIES.exists():
    sys.path.insert(0, str(LOCAL_LIBRARIES))

import pyarrow as pa
import pyarrow.parquet as pq


DATA = PROJECT / "data" / "gsm8k"
REVISION = "740312add88f781978c0658806c59bc2815b9866"
DLLM_REVISION = "ca176752fbceec49c6b4777a2c18ae88e4eb10ed"
SEED = 42
SFT_COUNT = 5000
VALIDATION_COUNT = 500
SOURCE_HASHES = {
    "train": "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d",
    "test": "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
}
SYSTEM_PROMPT = "\nRespond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_question(question):
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_raw():
    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    result = {}
    source_info = {"dataset": "openai/gsm8k", "config": "main", "revision": REVISION, "license": "MIT", "files": {}}
    for split, expected_hash in SOURCE_HASHES.items():
        filename = f"{split}-00000-of-00001.parquet"
        path = raw / filename
        url = f"https://huggingface.co/datasets/openai/gsm8k/resolve/{REVISION}/main/{filename}"
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            require(hashlib.sha256(content).hexdigest() == expected_hash, f"Source checksum mismatch: {split}")
            with path.open("xb") as output:
                output.write(content)
        require(file_hash(path) == expected_hash, f"Source checksum mismatch: {path}")
        result[split] = pq.read_table(path).to_pylist()
        source_info["files"][split] = {"file": filename, "url": url, "sha256": expected_hash, "bytes": path.stat().st_size, "rows": len(result[split])}
    if not (raw / "source.json").exists():
        write_json(raw / "source.json", source_info)
    readme = raw / "README.upstream.md"
    if not readme.exists():
        with urllib.request.urlopen(f"https://huggingface.co/datasets/openai/gsm8k/resolve/{REVISION}/README.md", timeout=60) as response:
            content = response.read()
        with readme.open("xb") as output:
            output.write(content)
    return result


def canonical_rows(rows, source_split):
    result = []
    for index, row in enumerate(rows):
        require(set(row) == {"question", "answer"}, f"Unexpected source columns: {source_split}/{index}")
        require(isinstance(row["question"], str) and row["question"].strip(), f"Missing question: {source_split}/{index}")
        require(isinstance(row["answer"], str) and row["answer"].strip(), f"Missing answer: {source_split}/{index}")
        require(row["answer"].count("####") == 1, f"Invalid answer delimiter: {source_split}/{index}")
        final_answer = row["answer"].rsplit("####", 1)[1].strip()
        number = Decimal(final_answer.replace(",", ""))
        require(number.is_finite(), f"Invalid numeric answer: {source_split}/{index}")
        answer_value = format(number, "f")
        if "." in answer_value:
            answer_value = answer_value.rstrip("0").rstrip(".")
        if number == 0:
            answer_value = "0"
        result.append({"id": f"gsm8k-main-{source_split}-{index:05d}", "source_split": source_split, "source_index": index, "question": row["question"], "answer": row["answer"], "final_answer": final_answer, "answer_value": answer_value})
    return result


def question_set(rows):
    return {normalize_question(row["question"]) for row in rows}


def prompt_messages(row):
    return [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + row["question"]}]


def sft_rows(rows):
    result = []
    for row in rows:
        reasoning = row["answer"].rsplit("####", 1)[0].strip()
        reasoning = re.sub(r"<<[^<>]*>>", "", reasoning).strip()
        require("<<" not in reasoning and ">>" not in reasoning, f"Unprocessed calculator annotation: {row['id']}")
        completion = f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{row['answer_value']}\n</answer>\n"
        result.append({"id": row["id"], "question": row["question"], "original_answer": row["answer"], "final_answer": row["final_answer"], "answer_value": row["answer_value"], "messages": prompt_messages(row) + [{"role": "assistant", "content": completion}]})
    return result


def rl_rows(rows):
    result = []
    for row in rows:
        result.append({"id": row["id"], "question": row["question"], "original_answer": row["answer"], "final_answer": row["final_answer"], "answer": row["answer_value"], "prompt": prompt_messages(row)})
    return result


def write_eval(rows, name):
    prompts = []
    labels = []
    for row in rows:
        prompts.append({"id": row["id"], "question": row["question"], "prompt": prompt_messages(row)})
        labels.append({"id": row["id"], "answer": row["answer"], "final_answer": row["final_answer"], "answer_value": row["answer_value"]})
    write_jsonl(DATA / "processed" / "eval" / f"{name}_prompts.jsonl", prompts)
    write_jsonl(DATA / "processed" / "eval" / f"{name}_labels.jsonl", labels)


def validate(raw, splits):
    checks = {}
    require(len(raw["train"]) == 7473 and len(raw["test"]) == 1319, "Official split counts changed")
    all_ids = set()
    all_questions = set()
    assigned_train_indices = set()
    for name, rows in splits.items():
        ids = {row["id"] for row in rows}
        questions = question_set(rows)
        require(len(ids) == len(rows), f"Duplicate ID within {name}")
        require(len(questions) == len(rows), f"Duplicate normalized question within {name}")
        require(not ids.intersection(all_ids), f"Cross-split ID leakage: {name}")
        require(not questions.intersection(all_questions), f"Cross-split question leakage: {name}")
        all_ids.update(ids)
        all_questions.update(questions)
        for row in rows:
            source = raw[row["source_split"]][row["source_index"]]
            require(row["question"] == source["question"] and row["answer"] == source["answer"], f"Original text changed: {row['id']}")
            require(Decimal(row["answer_value"]) == Decimal(row["final_answer"].replace(",", "")), f"Numeric normalization changed value: {row['id']}")
            if name != "test":
                require(row["source_split"] == "train", f"Official test entered training/dev: {row['id']}")
                assigned_train_indices.add(row["source_index"])
        checks[name] = {"rows": len(rows), "missing_questions": 0, "missing_answers": 0, "invalid_final_answers": 0, "duplicate_ids": 0, "duplicate_normalized_questions": 0}
    require(assigned_train_indices == set(range(7473)), "Training source not fully and uniquely assigned")
    require([row["source_index"] for row in splits["test"]] == list(range(1319)), "Official test order changed")
    require(len(splits["sft_train"]) == 5000 and len(splits["rl_train"]) == 1973 and len(splits["validation"]) == 500, "Prepared split counts changed")
    checks["cross_split_overlap"] = {"ids": 0, "normalized_questions": 0}
    checks["official_test_question_and_answer_preserved"] = True
    checks["official_test_order_preserved"] = True
    checks["all_official_train_rows_assigned_once"] = True
    checks["normalization"] = "NFKC Unicode normalization, casefold, whitespace collapse; comparison only, no source text changes"
    checks["semantic_near_duplicate_search_performed"] = False
    return checks


def validate_processed(splits):
    for filename, name in [("sft_train.jsonl", "sft_train"), ("sft_validation.jsonl", "validation")]:
        actual = read_jsonl(DATA / "processed" / filename)
        require(actual == sft_rows(splits[name]), f"SFT conversion mismatch: {filename}")
        for row in actual:
            require(len(row["messages"]) == 2 and row["messages"][0]["role"] == "user" and row["messages"][1]["role"] == "assistant", f"Invalid SFT messages: {row['id']}")
            target = re.fullmatch(r"<reasoning>\n(.*?)\n</reasoning>\n<answer>\n(.*?)\n</answer>\n", row["messages"][1]["content"], flags=re.DOTALL)
            require(target is not None and target.group(2) == row["answer_value"], f"SFT target mismatch: {row['id']}")
    require(read_jsonl(DATA / "processed" / "rl_train.jsonl") == rl_rows(splits["rl_train"]), "RL conversion mismatch")
    require(pq.read_table(DATA / "processed" / "sft" / "train.parquet").to_pylist() == sft_rows(splits["sft_train"]), "Native SFT train mismatch")
    require(pq.read_table(DATA / "processed" / "sft" / "test.parquet").to_pylist() == sft_rows(splits["validation"]), "Native SFT dev mismatch")
    for name in ["validation", "test"]:
        prompts = read_jsonl(DATA / "processed" / "eval" / f"{name}_prompts.jsonl")
        labels = read_jsonl(DATA / "processed" / "eval" / f"{name}_labels.jsonl")
        require(len(prompts) == len(labels) == len(splits[name]), f"Eval count mismatch: {name}")
        for source, prompt, label in zip(splits[name], prompts, labels):
            require(set(prompt) == {"id", "question", "prompt"}, f"Unexpected answer column in eval prompts: {name}")
            require(source["id"] == prompt["id"] == label["id"], f"Eval alignment mismatch: {name}")
            require(prompt["question"] == source["question"] and prompt["prompt"] == prompt_messages(source), f"Prompt construction mismatch: {source['id']}")
            require(label == {"id": source["id"], "answer": source["answer"], "final_answer": source["final_answer"], "answer_value": source["answer_value"]}, f"Eval label mismatch: {source['id']}")


def verify_only():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = PROJECT / item["path"]
        require(file_hash(path) == item["sha256"], f"Prepared file checksum mismatch: {path}")
    raw = load_raw()
    splits = {name: read_jsonl(DATA / "splits" / f"{name}.jsonl") for name in ["sft_train", "rl_train", "validation", "test"]}
    validate(raw, splits)
    validate_processed(splits)
    print(json.dumps({"verification": "passed", "counts": {name: len(rows) for name, rows in splits.items()}, "verified_files": len(manifest["files"])}, ensure_ascii=True))


def prepare():
    for path in [DATA / "splits", DATA / "processed", DATA / "manifest.json", PROJECT / "reports" / "gsm8k_data_checks.json"]:
        require(not path.exists(), f"Refusing to overwrite existing output: {path}; use --verify-only for verification")
    raw = load_raw()
    train = canonical_rows(raw["train"], "train")
    test = canonical_rows(raw["test"], "test")
    require(len(question_set(train)) == len(train) and len(question_set(test)) == len(test), "Duplicate normalized source questions require review")
    require(not question_set(train).intersection(question_set(test)), "Source train/test overlap requires review")
    ranked = sorted(train, key=lambda row: (hashlib.sha256(f"{SEED}\n{normalize_question(row['question'])}".encode("utf-8")).hexdigest(), row["source_index"]))
    splits = {
        "validation": ranked[:VALIDATION_COUNT],
        "sft_train": ranked[VALIDATION_COUNT:VALIDATION_COUNT + SFT_COUNT],
        "rl_train": ranked[VALIDATION_COUNT + SFT_COUNT:],
        "test": test,
    }
    for name in splits:
        splits[name] = sorted(splits[name], key=lambda row: row["source_index"])
    checks = validate(raw, splits)
    for name, rows in splits.items():
        write_jsonl(DATA / "splits" / f"{name}.jsonl", rows)
    train_sft = sft_rows(splits["sft_train"])
    dev_sft = sft_rows(splits["validation"])
    write_jsonl(DATA / "processed" / "sft_train.jsonl", train_sft)
    write_jsonl(DATA / "processed" / "sft_validation.jsonl", dev_sft)
    write_jsonl(DATA / "processed" / "rl_train.jsonl", rl_rows(splits["rl_train"]))
    sft_folder = DATA / "processed" / "sft"
    sft_folder.mkdir()
    pq.write_table(pa.Table.from_pylist(train_sft), sft_folder / "train.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(dev_sft), sft_folder / "test.parquet", compression="zstd")
    write_eval(splits["validation"], "validation")
    write_eval(splits["test"], "test")
    validate_processed(splits)
    checks["processed_formats_verified"] = True
    checks["no_gold_answers_in_model_prompt_fields"] = True
    checks["source_numbers_with_thousands_separator"] = {name: sum("," in row["answer"].rsplit("####", 1)[1] for row in rows) for name, rows in raw.items()}
    checks["source_negative_answers"] = {name: sum(Decimal(row["answer"].rsplit("####", 1)[1].strip().replace(",", "")) < 0 for row in rows) for name, rows in raw.items()}
    write_json(PROJECT / "reports" / "gsm8k_data_checks.json", checks)
    files = []
    for path in sorted(DATA.rglob("*")):
        if path.is_file():
            files.append({"path": path.relative_to(PROJECT).as_posix(), "bytes": path.stat().st_size, "sha256": file_hash(path)})
    for path in [Path(__file__), PROJECT / "reports" / "gsm8k_data_checks.json"]:
        files.append({"path": path.relative_to(PROJECT).as_posix(), "bytes": path.stat().st_size, "sha256": file_hash(path)})
    manifest = {"dataset": "openai/gsm8k", "config": "main", "source_revision": REVISION, "dllm_revision_reviewed": DLLM_REVISION, "seed": SEED, "split_method": "Sort official train rows by SHA256(str(seed) + newline + normalized_question); first 500 dev, next 5000 SFT, remainder RL; preserve source order within output splits", "counts": {name: len(rows) for name, rows in splits.items()}, "test_policy": "All 1319 official test rows; original question/answer and order unchanged; never training or tuning", "prompt_template": SYSTEM_PROMPT, "prompt_construction": "SYSTEM_PROMPT + two newline characters + original question, one user message", "answer_processing": "Keep original answer and final_answer; answer_value removes thousands separators using exact Decimal arithmetic; SFT removes only <<calculator annotations>> from the original reasoning", "tokenized": False, "pyarrow_version": pa.__version__, "python_version": sys.version, "files": files}
    write_json(DATA / "manifest.json", manifest)
    verify_only()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="只检查现有数据和校验和，不覆盖文件")
    arguments = parser.parse_args()
    if arguments.verify_only:
        verify_only()
    else:
        prepare()
