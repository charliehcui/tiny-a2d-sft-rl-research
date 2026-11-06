# Tiny-A2D：GSM8K 上的 SFT 与 diffu-GRPO 实验

本项目比较 Qwen3-0.6B 扩散语言模型（Diffusion Language Model）的基线（Baseline）、监督微调（Supervised Fine-Tuning, SFT）与三轮强化学习（Reinforcement Learning, RL）。重点检查二值结果奖励、静态优先抽样（Priority Sampling）及候选复用（Rollout Reuse）在有限生成预算下的实际效果。

**最终测试准确率（Accuracy）为 22.67%，低于 SFT 的 22.90%。不能据此宣称 RL 提升了泛化能力。** 本包保留真实配置、原始预测、逐候选奖励、训练步数证据及最终权重。本公开仓库发布源码、配置、数据、评测记录与必要 Adapter，不发布公司原始作业题面或本地会话审计。

## 模型与环境

采用 `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`，版本 `c8d24a3f4adaeef46881b450e1bf7d1005203bd7`。它是掩码扩散模型（Masked Diffusion Model），保留 `a2d-qwen3` 架构。没有用自回归 Qwen3 替代。官方 dLLM 源码版本为 `ca176752fbceec49c6b4777a2c18ae88e4eb10ed`，评测框架 lm-evaluation-harness 版本为 `dca091dbee3d6ee1dab268e9f649aa64e8db97d8`。两份官方源码归档和许可证均在 `third_party/`。

实际云端记录为 Linux、Python 3.11.18、PyTorch 2.6.0+cu124、Transformers 4.57.0、TRL 0.19.1、PEFT 0.17.1、Accelerate 1.11.0、Datasets 4.2.0，以及单张 NVIDIA RTX 4090（约 24 GB）。训练需要支持 BF16 的 CUDA GPU。RL 已分配显存（Allocated Memory）峰值约 15.9 GiB，实际设备峰值和预留显存（Reserved Memory）更高；评测批量 8 曾预留约 21.5 GiB。建议使用 24 GB GPU。完整环境记录在 `logs/baseline_environment*`。

公开版本不含 GB 级基础模型或合并模型，只提交两份最终 Adapter。基础模型约 1.50 GB，按固定版本和哈希下载。评测或训练前按下述命令恢复 SFT 与 Final RL 合并模型。完整复现建议至少准备 20 GB 空闲磁盘，并给 `/opt/bluewhale_cache` 所在盘至少 4 GiB 导出空间。

## 目录

```text
scripts/       原核心训练、数据、评测、启动器；保持原导入关系
configs/       冻结评测协议、训练配置和任务配置
data/          原始数据、固定划分、预处理、抽样历史
models/        最终 SFT / Final RL 适配器
runs/          四轮正式训练证据与当时源码；合并模型需本地恢复
results/       baseline、sft、rl_pilot1、rl_pilot2、final_rl 的评测
logs/          实际云端环境及正式基线日志
reports/       数据校验与 SFT 分词审计证据
third_party/   固定提交版本的官方源码归档与许可证
tests/         原配置、抽样、候选复用与恢复测试
audit/         结果复核与脱敏后的优化器步数审计
```

`runs/*/code` 是当时运行的审计快照，不能直接当成可移植入口，因为其中项目根路径依赖目录深度。运行统一从根目录下的 `scripts/` 开始。历史检查点只保留 JSON 状态，不包含优化器二进制状态，不能用于继续历史训练。

## 安装

以下命令从本包根目录执行，目标环境为 Linux。先创建 Python 3.11 虚拟环境（Virtual Environment）：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
bash scripts/install.sh
export HF_HOME=/opt/bluewhale_cache/huggingface
export HF_DATASETS_CACHE=/opt/bluewhale_cache/datasets
export MPLCONFIGDIR=/opt/bluewhale_cache/matplotlib
export WANDB_MODE=disabled
mkdir -p /opt/bluewhale_cache/exports
python scripts/download_baseline_model.py
```

安装脚本从随包归档解压两个固定版本源码，再安装实际运行所需依赖。`requirements.txt` 固定直接依赖版本，完整历史冻结清单在 `logs/baseline_environment_freeze.txt`；后者含云端 editable 路径，仅是环境证据，不能直接 `pip install -r`。DeepSpeed、FlashAttention、vLLM 和 4-bit 量化不在本次固定运行方案中。

下载器验证所有模型文件的大小与 SHA-256 或 Git blob SHA-1，已有正确文件不会重下。下载器根路径从脚本位置推导；没有修改冻结训练和评测参数。公开版本另提供必要的 Adapter 合并入口。

## 从 Adapter 恢复模型

本公开仓库不包含预先合并的 GB 级权重。安装并下载基础模型后，按顺序执行：

```bash
python scripts/merge_adapter.py --stage sft
python scripts/merge_adapter.py --stage final-rl
```

第一条恢复三轮 RL 的 SFT 626 步起点；第二条恢复最终评测模型。Final RL Adapter 的基座是合并后的 SFT，不能直接作用于公开原始基础模型。重建只写新的、被 Git 忽略的模型目录，不覆盖已有目录。脚本提供原始权重哈希对照；设备差异可能产生不同 BF16 权重，历史评测分数不能据重建命令自动认定已复现。此脚本本次只验证语法和 CLI，未执行 GPU 重建。

## 数据准备与核验

数据已经准备好：SFT 5000、开发 500、原始 RL 池 1973、官方测试 1319。划分规则与来源在 `data/README.md`；详细审计在 `data/gsm8k/manifest.json`。

```bash
python scripts/prepare_gsm8k.py --verify-only
python scripts/prepare_rl_static.py --check
python scripts/check_submission.py --adapters-only
```

第三条命令只读核对实际抽样实现、结果、权重和提交校验清单，不启动训练或推理。无需重新运行默认数据准备模式，更不要覆盖冻结配置。

## 基线评测

```bash
python scripts/run_baseline.py --split test --protocol configs/baseline_protocol_v2.json --output results/baseline_reproduction_NEW
```

输出目录必须不存在。此命令评测完整官方测试集，不接受测试子集 `--limit`。开发短测可用 `--split validation --limit 8` 配合新的输出目录；8 题结果不代表完整开发准确率。

## SFT 训练与加载

```bash
python scripts/run_experiment.py start configs/jobs/sft_gsm8k.json --stop-config configs/matpool_manual_stop.json --id sft-reproduction-001
python scripts/run_experiment.py status sft-reproduction-001
```

启动器（Launcher）负责创建新目录、设置 `BLUEWHALE_RUN_DIR`、后台运行和写状态。不能绕过此约束直接调用 `train_sft.py`。使用新运行 ID，禁止覆盖既有证据。真实配置为 LoRA r=8、alpha=16、dropout=0.05、学习率 1e-4、两轮训练、梯度累积 16，共 626 次优化器更新（Optimizer Step）。

完成上述 SFT 模型恢复后，可执行：

```bash
python scripts/evaluate_model.py --split validation --checkpoint runs/sft-gsm8k-20261104-011915/final_model --protocol configs/baseline_protocol_v2.json --output results/sft_dev_reproduction_NEW
python scripts/evaluate_model.py --split test --checkpoint runs/sft-gsm8k-20261104-011915/final_model --protocol configs/baseline_protocol_v2.json --output results/sft_test_reproduction_NEW
```

适配器加载示例及 SFT → RL 基座关系见 `models/README.md`。Final RL Adapter 不能直接叠到原始基础模型上。

## RL 与候选复用

两轮试验（Pilot）与最终训练均从原 SFT 626 步合并模型开始，不是连续把三轮 Adapter 叠加。以下命令各自启动一个新运行：

```bash
python scripts/run_experiment.py start configs/jobs/rl_gsm8k_512.json --stop-config configs/matpool_manual_stop.json --id rl-pilot1-reproduction-001
python scripts/run_experiment.py start configs/jobs/rl_static_512.json --stop-config configs/matpool_manual_stop.json --id rl-pilot2-reproduction-001
python scripts/run_experiment.py start configs/jobs/rl_reuse_1536.json --stop-config configs/matpool_manual_stop.json --id rl-final-reproduction-001
```

以上三项独立运行，不能同时占用同一 GPU。启动器使用项目级锁；当前任务完成后再启动下一项。默认停止配置的 `mode=disabled`，不自动暂停或释放云实例。Final RL 工作流含训练、模型导出、完整开发集和一次官方测试；有 18 小时总预算及禁止自动重试约束。

Pilot 1：512 个不同题 × 4 个新候选，32 次优化。Pilot 2：127 个历史混合奖励组（Mixed Reward Group）题各呈现两次，另选 258 个探索题；512 次呈现、385 个不同题，每次重新生成 4 个候选，32 次优化。

Final RL：1536 个不同题，每周期 16 题 × 4 个候选，先缓存 64 个候选的四组提示掩码及旧/参考策略概率，再以四个视图分别完成四次真实更新。96 个周期，6144 个新候选，384 次更新，24576 次候选视图使用。生成批量为 4，训练微批量为 1，梯度累积为 64。实现见 `scripts/rl_reuse_trainer.py`，训练/保存约束在 `scripts/train_rl_reuse.py`，完整工作流在 `scripts/run_final_rl.py`。

两轮 Pilot 和 Final RL 均采用学习率 3e-6、KL 系数 0.04、裁剪参数 0.2、LoRA r=8/alpha=16/dropout=0、G=4。RL 的随机生成温度为 1.0。评测采用冻结协议中的温度 0、256 个输出位置、256 个去噪步及批量 8。

本包提供原程序短测（Smoke Test）配置及候选复用测试，未打包过去短测的无效模型。原恢复测试可在安装后执行 `python tests/test_rl_reuse.py` 与 `python tests/test_rl_storage.py`；它们使用小模型程序夹具，不代表 GSM8K 效果。本次打包没有执行 GPU 训练。历史运行缺少优化器二进制状态，若需从历史断点继续训练，须另从原目录选取完整检查点并核对代码哈希，不应把本包 JSON 状态当成完整检查点。

## 最终模型评测

```bash
python scripts/evaluate_model.py --split validation --checkpoint runs/rl-reuse1536-g4-u4-20261105-final01/final_model --protocol configs/baseline_protocol_v2.json --output results/final_rl_dev_reproduction_NEW
python scripts/evaluate_model.py --split test --checkpoint runs/rl-reuse1536-g4-u4-20261105-final01/final_model --protocol configs/baseline_protocol_v2.json --output results/final_rl_test_reproduction_NEW
```

所有输出使用新目录，不写回历史结果。实际评测依据官方 RegexFilter 与 Exact Match，逐题记录与原生框架汇总一致。

## 实际结果与局限

| 实验 | 完整 Dev Accuracy | Official Test Accuracy | 正式 RL 新候选 | 优化器步数 |
|---|---:|---:|---:|---:|
| Baseline | 未完整评测 | 267/1319，20.24% | — | — |
| SFT | 144/500，28.80% | 302/1319，22.90% | — | 626 |
| Pilot 1 | 147/500，29.40% | 未完整评测 | 2048 | 32 |
| Pilot 2 | 145/500，29.00% | 未完整评测 | 2048 | 32 |
| Final RL | 146/500，29.20% | 299/1319，22.67% | 6144 | 384 |

三轮正式 RL 总计 10240 个新候选、448 次优化，不包括过去程序短测。完整结果在 `results/`，训练奖励和步数在 `runs/`，公开版本的离线结果复核在 `audit/local_checks.json`，原优化器步数核对在 `audit/optimizer_state_audit.json`。

主准确率使用 `exact_match,flexible-extract`，提取最后一个数值并按冻结规则归一化。奖励（Reward）是训练时每个随机候选的二值数学正确性。XML 答案准确率（XML Answer Accuracy）要求提取后答案正确；XML 答案可提取率（XML Answer Extractability）只要求能提取数值。三者不能混为一谈。独立复核脚本另用 `xml_extractor_success` 明确报告官方 XML 数值正则的成功次数；不会把任意 XML 标签的存在率当作数值可提取率。

本实验使用统一零样本 XML 提示，不是公开榜单的 8-shot GSM8K CoT 协议。只有一次种子，没有可靠显著性结论。Pilot 2 的优先抽样未提升 Dev；Final RL 没有超过 SFT 的官方测试结果。扩散模型的对数概率代理（Likelihood Proxy）与多步生成分布不完全一致，候选复用的工程正确性不能证明算法收益。

## Docker 与交付状态

原项目无 Dockerfile，本次补充的是尚未构建验证的草案：

```bash
docker build -t bluewhale-submission .
docker run --rm --gpus all -it bluewhale-submission
```

本机 Docker 引擎未运行、没有可用 CUDA GPU，也未安装原训练环境，故**没有验证容器构建、依赖完整安装、模型导入和 GPU 端到端复现**。静态参数检查、数据/结果/权重校验和本次可执行测试详见 `VERIFICATION_REPORT.md`，不能用历史成功日志替代本次运行验证。

本公开仓库不发布公司原始题面/PPT、本地完整盘点及会话路径。研究目标和真实结果已在本 README 汇总，公开文件用途与排除项见 `SUBMISSION_MANIFEST.md`。
