# 数据与划分

保留原始 GSM8K 数据（Dataset）、固定划分（Split）、预处理结果及抽样依据，不需要在运行评测时重新下载。

固定数据版本：`openai/gsm8k` / `main` / `740312add88f781978c0658806c59bc2815b9866`。原始来源、许可证（License）与文件 SHA-256 在 `gsm8k/raw/source.json` 和 `gsm8k/raw/README.upstream.md`。固定文件校验清单为 `gsm8k/manifest.json`。

官方训练集 7473 题，按种子 42 与规范化题面 SHA-256 排序分配：开发集（Development Set）500 题、SFT 5000 题、RL 1973 题；输出保留原始来源顺序。官方测试集（Official Test Set）1319 题保持完整。

- `gsm8k/splits`：原始题目、推理及答案，含四个固定划分。
- `gsm8k/processed/sft_*`：SFT 对话。移除计算器标注 `<<...>>`，保留原推理与最终答案，监督回答部分。
- `gsm8k/processed/eval`：模型提示与独立标签，避免答案进入模型输入。
- `rl_train_512.jsonl`：Pilot 1 的 512 个不同题。
- `rl_train_static_512.jsonl`：Pilot 2 的 512 次呈现，共 385 个不同题；127 个历史混合组题重复两次，加 258 个探索题，每次重新生成四个候选。
- `rl_reuse_1536.jsonl`：Final RL 的 1536 个不同题；每题四个新候选，每个候选四个掩码视图分别参与四次优化。
- `rl_static_sources`：静态抽样所用历史奖励组和已冻结开发评测。少量与 results 重复的数据是现有配置的哈希依赖，不能删除。

在提交根目录执行只读验证：

```bash
python scripts/prepare_gsm8k.py --verify-only
python scripts/prepare_rl_static.py --check
python scripts/check_submission.py --adapters-only
```

`prepare_gsm8k.py` 默认模式会拒绝覆盖现存划分；`prepare_rl_reuse.py` 默认模式是最初准备流程，也会因后续正式配置与初始草案不同而拒绝覆盖。不要为重新准备而改写已经冻结的数据。本包的只读核验脚本调用实际 `load_reuse_rows` 校验完整日程和交集。

原数据清单记录了脚本及历史检查报告的哈希，提交副本保留这两份文件原字节。审计使用物理换行读取 JSONL，避免字符串中的合法 Unicode 行分隔符被误判为新记录。未做语义近重复识别。
