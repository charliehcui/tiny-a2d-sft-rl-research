# 蓝鲸作业 GSM8K 数据

数据已准备完成。仅使用公开的 `openai/gsm8k`、`main` 配置；未加入其他数据集或 AI 生成题目。

## 固定划分

| 用途 | 数量 | 来源 |
|---|---:|---|
| 监督微调训练集（SFT Training Set） | 5,000 | 官方 train |
| 强化学习训练集（RL Training Set） | 1,973 | 官方 train |
| 开发集（Validation / Development Set） | 500 | 官方 train |
| 最终测试集（Test Set） | 1,319 | 官方 test，完整保留 |

三个 train 来源的子集互不重合，合计 7,473 条。种子（Seed）为 42：按种子和归一化问题文本的 SHA-256 排序，前 500 条为开发集，接着 5,000 条为 SFT，其余为 RL。每份输出内部恢复官方源行顺序；此划分不依赖 Python 随机数实现。

官方版本固定为 `740312add88f781978c0658806c59bc2815b9866`。原始两个 Parquet 文件已与 Hugging Face 公布的 SHA-256 核对一致。

## 文件用途

以下路径均相对于本项目根目录 `<PROJECT_ROOT>`。

| 路径 | 内容 |
|---|---|
| `data/gsm8k/raw/` | 官方原始 Parquet、上游说明和下载来源 |
| `data/gsm8k/splits/sft_train.jsonl` | 5,000 条 SFT 记录及原始问题、答案 |
| `data/gsm8k/splits/rl_train.jsonl` | 1,973 条 RL 记录及原始问题、答案 |
| `data/gsm8k/splits/validation.jsonl` | 500 条开发记录及原始问题、答案 |
| `data/gsm8k/splits/test.jsonl` | 全部 1,319 条官方测试记录，问题和完整标准答案未修改 |
| `data/gsm8k/processed/sft_train.jsonl` | SFT 的 `messages` 对话格式，可直接检查 |
| `data/gsm8k/processed/sft_validation.jsonl` | 开发集的同款 SFT 格式 |
| `data/gsm8k/processed/sft/` | 兼容 dLLM 本地数据目录加载的 Parquet 版，含 `train.parquet` 和 `test.parquet` |
| `data/gsm8k/processed/rl_train.jsonl` | RL 的 `prompt` 和奖励用 `answer` |
| `data/gsm8k/processed/eval/validation_prompts.jsonl` | 开发集提示，不含答案字段 |
| `data/gsm8k/processed/eval/validation_labels.jsonl` | 开发集答案，按 id 与提示对应 |
| `data/gsm8k/processed/eval/test_prompts.jsonl` | 最终测试提示，不含答案字段 |
| `data/gsm8k/processed/eval/test_labels.jsonl` | 最终测试答案，按 id 与提示对应 |
| `data/gsm8k/manifest.json` | 版本、划分方法、各文件数量和校验和（Checksum） |
| `reports/gsm8k_data_checks.json` | 缺失、重复、交叉重合、原文保留及格式检查结果 |
| `configs/data_tasks/gsm8k_bluewhale_xml.yaml` | 已有评测框架可用的本地数据任务配置；不是新训练框架 |
| `scripts/prepare_gsm8k.py` | 可复现的下载、划分、转换与检查脚本 |

**注意：`processed/sft/test.parquet` 是 500 条开发集，绝不是最终测试集。** 该命名用于适配 dLLM SFT 将 `dataset["test"]` 当作训练期间评估数据的现有约定。

## 原文与转换规则

`splits/` 中每条记录包含：

- `id`、`source_split`、`source_index`：追溯到官方来源和从零开始的行号。
- `question`：原始问题，逐字符保留。
- `answer`：原始完整标准解答，逐字符保留，包括 `####` 和计算标记。
- `final_answer`：从 `####` 后提取的原始数值字符串。
- `answer_value`：用于训练和评分的等值数值字符串，去掉千位分隔逗号，不改变数值。

SFT 仅删除标准推理中的 `<<计算标记>>`，不重写推理或生成新解答；完整原答案同时保存在 `original_answer`。助手回复包为 `<reasoning>...</reasoning>` 和 `<answer>...</answer>`，末尾保留换行。RL 的 `answer` 是独立的数值标签，`original_answer` 保存完整原文。

SFT、RL、开发和测试使用同一用户提示（Prompt）：dLLM 官方 RL 的格式指令，加上原始问题。答案只出现在 SFT 的助手监督消息或独立评分字段中，不拼入模型用户提示。

官方数据有带千位逗号和少量负数的答案。本次保留原文，同时用精确十进制（Decimal）校验派生字段的数值相等。后续奖励应以正确性为主，不能把官方 `isdigit()` 格式奖励等同于对所有数值都适用。

## 后续入口怎么使用

### SFT

在已有 Tiny-A2D SFT 脚本中，把 `dataset_args` 设为 `data/gsm8k/processed/sft`，保持 `load_preprocessed_data=False`。本地目录中的 Parquet 已有 `messages`；脚本会继续应用分词器（Tokenizer）生成训练张量。这里的 False 表示尚未分词，并不表示数据尚未清洗。

也可在未来训练环境中读取便于检查的 JSONL：

```python
from datasets import load_dataset

sft = load_dataset("json", data_files={"train": "data/gsm8k/processed/sft_train.jsonl", "test": "data/gsm8k/processed/sft_validation.jsonl"})
```

### RL

未来在现有 RL 入口中，将数据加载替换为本地文件读取即可；奖励与训练代码本次没有修改。

```python
from datasets import load_dataset

rl = load_dataset("json", data_files="data/gsm8k/processed/rl_train.jsonl", split="train")
```

不要继续使用默认的“加载全部 GSM8K train”入口，否则会把已留出的开发集也带入 RL。

### Baseline / SFT / RL 最终评测

三阶段均固定使用 `processed/eval/test_prompts.jsonl` 和 `test_labels.jsonl`，按 `id` 对齐。生成时只读取 prompts，评分时才读取 labels。用开发集选择配置，不用官方 test 调参。

如果沿用 dLLM 配套的 lm-evaluation-harness，在项目根目录运行后续评测时，可加载 `configs/data_tasks` 中的 `gsm8k_bluewhale_xml` 任务。它直接读取本地 `splits/test.jsonl`，只用 `question` 构造提示，目标从独立字段读取；配置规定零样本（Zero-shot），不从数据抽取示例。

该任务提供 XML 答案提取和与原 GSM8K 任务相同风格的宽松末数字提取，两项均应报告，不可事后只挑较好的一项。它使用官方全量题目和答案，但采用本项目统一 XML 提示；不是论文原始 8-shot `gsm8k_cot` 配置，分数不能直接冒充该协议的复现。

本次只准备了数据与静态任务配置，**未运行 Baseline**。模型分词、对话模板实际应用、生成长度、采样步数及最终评分运行仍属于下一阶段。没有提前截断样本；训练时要审计分词长度，不能静默切掉答案。

## 检查与复现

已通过全部检查：官方行数及哈希、空值与 `####` 格式、数值提取、原始问题与解答保留、源行唯一分配、划分间 ID 和归一化题目零重合、测试顺序、SFT / RL 格式、提示与答案对齐。归一化仅用于重复检测：Unicode NFKC、大小写折叠和空白折叠，不修改原始文本。没有声称完成语义近重复的穷举搜索。

只读复查，不覆盖现有文件：

```powershell
python scripts/prepare_gsm8k.py --verify-only
```

脚本会自动使用项目 `work/pythonlibs` 中的 CPU Parquet 读写库。换到其他机器时，仅复现数据处理需要 `scripts/requirements-data.txt`，不需要 PyTorch 或 GPU。上传训练机器可以只带 `data/`、`scripts/` 和 `configs/`，不必带本地 `work/pythonlibs`。

在一个新的空项目副本中运行 `python scripts/prepare_gsm8k.py`，可以按同一版本和划分方法重建数据。脚本遇到已有处理结果会拒绝覆盖。

## 来源

- 官方数据：[openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)，MIT 许可。
- 数据版本：[740312add88f781978c0658806c59bc2815b9866](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866)。
- dLLM 格式依据：[SFT 映射](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/utils/data.py)、[RL 数据和提示](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/rl/grpo/datasets.py)。
- 评测配置格式依据：[配套 GSM8K 配置](https://github.com/ZHZisZZ/lm-evaluation-harness/blob/dca091dbee3d6ee1dab268e9f649aa64e8db97d8/lm_eval/tasks/gsm8k/gsm8k-cot.yaml)。
