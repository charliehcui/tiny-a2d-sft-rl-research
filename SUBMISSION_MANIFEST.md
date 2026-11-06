# 公开仓库文件清单

完整公开文件、字节数、SHA-256 与用途见 `SUBMISSION_MANIFEST.csv`。该清单不包含自身与本说明，避免循环哈希；核验命令为 `python scripts/check_submission.py --adapters-only`。

| 目录/文件 | 用途 |
|---|---|
| README.md | 模型选择、真实命令、恢复步骤、结果和限制 |
| scripts/、tests/ | 核心训练、静态抽样、候选复用、数据、评测与原有测试 |
| configs/ | 真实训练配置与冻结评测协议 |
| data/ | 固定 GSM8K 原始数据、划分、预处理与抽样依据 |
| models/sft_adapter、models/final_rl_adapter | 两份必要最终 LoRA 权重与分词器 |
| models/provenance | 原始模型来源与导出哈希，不含合并权重 |
| runs/ | 四轮正式训练指标、奖励、日志、状态 JSON 与实际运行源码 |
| results/ | Baseline、SFT、Pilot 1、Pilot 2、Final RL 的原始评测 |
| logs/、reports/ | 真实环境版本及必要数据检查证据 |
| audit/ | 公开版本结果核验与原始优化器步数审计 |
| third_party/ | 固定版本依赖源码归档、模型卡、许可证和引用资料 |
| work/rl_source/ | 两份原测试依赖的锁定源码，不是临时日志 |
| requirements.txt、Dockerfile、.dockerignore、.gitignore | 依赖及排除规则；GPU/Docker 未验证 |
| VERIFICATION_REPORT.md | 已验证、未验证与公开版本调整说明 |

## 未上传内容

- 两份正式合并模型，每份权重 1,192,135,096 字节，及其重复分词器。原始哈希已保留，README 提供 Adapter 重建命令。
- 可公开下载的基础模型、任何中间/重复检查点、优化器二进制状态、缓存、本地虚拟环境。
- 本地原项目完整盘点、源目录哈希清单、会话路径和命令调用栈；这些是本地审计材料，不用于公开展示。
- 云端部署、进程、预算状态和依赖安装临时日志；正式训练日志、指标及原始结果保留。
- 公司原始作业题面/PPT 与无关研究草稿。

本地原始项目及最终提交文件夹均保留，没有删除或修改其中任何文件。公开版本只调整说明文档、个人电脑路径、核验范围，并增加必要的 Adapter 恢复入口。训练、抽样、奖励代码及实验原始记录未更改。
