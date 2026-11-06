# 公开仓库模型说明

本仓库提交两份最终低秩适配器（LoRA Adapter），不上传 GB 级基础模型或合并模型（Merged Model）。两份 Adapter 各 20,236,472 字节，均为 392 个 FP32 张量、5,046,272 个参数。

| Adapter | 基座 | SHA-256 |
|---|---|---|
| `sft_adapter` | 固定公开扩散基础模型 | `320d3459f8678878aeaa1d36f421b1b238f374bbb081d03e146e50c17d6f3be2` |
| `final_rl_adapter` | 已合并 SFT 626 步的扩散模型 | `94079715d1e7c7cada16ffd1ed51a2e6078209ca00d4b29dca62df1a06d3bbc0` |

基础模型：`dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`。精确版本（Revision）：`c8d24a3f4adaeef46881b450e1bf7d1005203bd7`。文件清单与来源校验值在 `configs/baseline_model_files.json`，下载器会验证已有文件，不重复下载正确文件。

安装依赖后，从仓库根目录依次执行：

```bash
python scripts/download_baseline_model.py
python scripts/merge_adapter.py --stage sft
python scripts/merge_adapter.py --stage final-rl
```

第一条恢复公开基础权重。第二条生成 `runs/sft-gsm8k-20261104-011915/final_model`，作为后续 RL 的基座。第三条生成 `runs/rl-reuse1536-g4-u4-20261105-final01/final_model`，用于最终评测。Final RL Adapter 不能直接加载到公开原始基础模型上。

本地提交包保留的两份原始合并权重各 1,192,135,096 字节。原始 SFT 权重 SHA-256 为 `777eb90450fa3809660f11ce58411f9eceb2081a57a06d06d3917f08a37b412a`；原始 Final RL 为 `c1018272ce0182765fdb1907c0684b38dbf4eb1fc17ec70020ea7950e9020f80`。`provenance/` 保留原始导出来源及 Final RL 导出清单。

新增合并脚本的语法和 CLI 已检查，**本次未执行模型重建或 GPU 验证**。不同设备的 BF16 运算可能使重建权重与原始文件不同。脚本报告是否匹配原始哈希，重建不能代替重新评测，也不能据此声称复制了历史得分。

原 Adapter 配置中的 `/mnt/bluewhale` 是历史通用云端路径。脚本显式指定本仓库基座路径，不依赖该历史路径。无需修改或覆盖 Adapter 权重。基础模型卡及许可说明见 `third_party/BASE_MODEL_CARD.md`。
