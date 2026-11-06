---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: Qwen/Qwen3-0.6B
---

<center> <div style="text-align: center;"> <img src="https://raw.githubusercontent.com/ZHZisZZ/dllm/main/assets/logo.gif" width="400" />
 </div> </center>

# Qwen3-0.6B-diffusion-mdlm-v0.1

Qwen3-0.6B-diffusion-mdlm-v0.1 is a diffusion-based language model adapted from [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) using [MDLM](https://arxiv.org/abs/2406.07524) (masked diffusion), trained with the [dLLM](https://github.com/ZHZisZZ/dllm) framework.

The model was presented in the paper [dLLM: Simple Diffusion Language Modeling](https://huggingface.co/papers/2602.22661).

## Model Overview

Qwen3-0.6B-diffusion-mdlm-v0.1 has the following features:

- **Method**: [Masked Diffusion Language Modeling (MDLM)](https://arxiv.org/abs/2406.07524)
- **Framework**: [dLLM](https://github.com/ZHZisZZ/dllm)
- **Base Model**: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- **Datasets**: [tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture), [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk), [opc-sft-stage1](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage1) and [opc-sft-stage2](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2) 

For training details, see the [W&B report](https://wandb.ai/asap-zzhou/dllm/reports/dLLM-Tiny-A2D--VmlldzoxNTI2NTEzOA).

## Installation

```shell
pip install torch transformers accelerate
```

## Quick Start

> [!NOTE]
> We recommend setting `enable_thinking=False` when using the model to ensure stable behavior and reproducible results.

```python
import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForMaskedLM


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate(model, prompt, prompt_lens, pad_id, steps=128, max_new_tokens=128, block_size=64, temperature=0.0, cfg_scale=0.0, remasking="random"):
    mask_id = tokenizer.mask_token_id
    batch_size = prompt.size(0)
    total_length = int(prompt_lens.max().item() + max_new_tokens)
    x = torch.full((batch_size, total_length), pad_id, dtype=torch.long, device=model.device)
    for i, length in enumerate(prompt_lens.tolist()):
        x[i, :length] = prompt[i, :length]
        x[i, length : length + max_new_tokens] = mask_id

    prompt_index = torch.arange(total_length, device=x.device).unsqueeze(0) < prompt_lens.unsqueeze(1)
    positions = torch.arange(total_length, device=x.device)

    assert max_new_tokens % block_size == 0
    num_blocks = max_new_tokens // block_size
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt_lens + num_block * block_size
        block_end = block_start + block_size
        init_block_mask = (
            (positions.unsqueeze(0) >= block_start.unsqueeze(1))
            & (positions.unsqueeze(0) < block_end.unsqueeze(1))
            & (x == mask_id)
        )
        num_transfer_tokens = get_num_transfer_tokens(init_block_mask, steps_per_block)

        for i in range(steps_per_block):
            block_mask = (
                (positions.unsqueeze(0) >= block_start.unsqueeze(1))
                & (positions.unsqueeze(0) < block_end.unsqueeze(1))
                & (x == mask_id)
            )

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1.0) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand_like(x0, dtype=torch.float)
            else:
                raise NotImplementedError(remasking)

            confidence = torch.full_like(x0_p, -np.inf)
            confidence = torch.where(block_mask, x0_p, confidence)

            x0 = torch.where(block_mask, x0, x)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                k = int(num_transfer_tokens[j, i].item())
                if k == 0:
                    continue
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForMaskedLM.from_pretrained("dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1", dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained("dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1")
if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or tokenizer.mask_token_id

messages = [
    [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Implement a DFS traversal in Python with clear inline comments."},
    ],
    [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 10 kilometers per hour. How many kilometers can she run in 10 hours?"},
    ],
]

encoded = [tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=True, enable_thinking=False) for m in messages]
prompt_lens = torch.tensor([len(e) for e in encoded], dtype=torch.long)
max_prompt_len = max(prompt_lens).item()
prompt_tensor = torch.full((len(encoded), max_prompt_len), pad_id, dtype=torch.long)
for i, ids in enumerate(encoded):
    prompt_tensor[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

prompt_tensor = prompt_tensor.to(device)
prompt_lens = prompt_lens.to(device)
max_new_tokens = 256

text = generate(
    model, prompt_tensor, prompt_lens, pad_id=pad_id, steps=256, max_new_tokens=max_new_tokens, block_size=64, temperature=0.0, cfg_scale=0.0, remasking="low_confidence"
)

new_tokens = [
    text[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens].tolist() for i in range(text.size(0))
]
for idx, decoded in enumerate(tokenizer.batch_decode(new_tokens, skip_special_tokens=False)):
    print(f"
[Sample {idx}]")
    print(decoded)
```

## Generation Parameters

| Parameter        | Description                                                                                    | Default  |
| ---------------- | ---------------------------------------------------------------------------------------------- | -------- |
| `max_new_tokens` | Number of tokens to generate                                                                   | 256      |
| `steps`          | Number of diffusion denoising iterations                                                       | 256      |
| `temperature`    | Sampling temperature; set to `0.0` for deterministic generation                                | 0.0      |
| `block_size`   | Token block size used during iterative denoising                                               | 64       |
| `cfg_scale`      | Classifier-free guidance scale controlling instruction adherence (higher = more deterministic) | 0.0      |
| `remasking`      | Strategy for re-masking during each denoising step (`random` or `low_confidence`)         | `low_confidence` |

## Command-Line Interface

Follow the Github repo's demo script [examples/a2d/mdlm/chat.py](https://github.com/ZHZisZZ/dllm/blob/main/examples/a2d/mdlm/chat.py) for visualized generation:

```shell
python -u examples/a2d/mdlm/chat.py \
    --model_name_or_path dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 \
    --chat_template True --block_size 64 --remasking low_confidence --steps 256 --max_new_tokens 256
```

## Evaluation

<table style="border-collapse: collapse; width: 100%; text-align: center;">
  <thead>
    <tr style="border-bottom: 3px solid #333;">
      <th style="padding: 8px;">Model                     </th>
      <th style="padding: 8px;">GSM8K</th>
      <th style="padding: 8px;">MATH</th>
      <th style="padding: 8px;">BBH</th>
      <th style="padding: 8px;">MMLU&#8209;Pro</th>
      <th style="padding: 8px;">Hellaswag</th>
      <th style="padding: 8px;">MMLU</th>
      <th style="padding: 8px;">HumanEval</th>
      <th style="padding: 8px;">MBPP</th>
    </tr>
  </thead>

  <tr style="background-color: #e8f2ff">
    <td style="padding: 8px;"><a href="https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"><code>Qwen3-0.6B-diffusion-bd3lm-v0.1</code></a> (evaluated)</td>
    <td>46.6</td><td>13.9</td><td>27.0</td><td>14.1</td><td>40.0</td><td>38.8</td><td>47.6</td><td>32.0</td>
  </tr>

  <tr style="background-color: #e8f2ff">
    <td style="padding: 8px;"><a href="https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"><code>Qwen3-0.6B-diffusion-mdlm-v0.1</code></a> (evaluated)</td>
    <td>29.8</td><td>8.8</td><td>27.0</td><td>17.6</td><td>42.1</td><td>40.0</td><td>30.5</td><td>29.2</td>
  </tr>
  <tr>
    <td colspan="9" style="padding: 0; border-top: 3px double #666;"></td>
  </tr>

<tr>
  <td style="padding: 8px;"><a href="https://huggingface.co/Qwen/Qwen3-0.6B-Base"><code>Qwen3-0.6B-Base</code></a> (reported)</td>
  <td>59.6</td><td>32.4</td><td>41.5</td><td>24.7</td><td>47.4</td><td>52.8</td><td>32.3</td><td>36.6</td>
</tr>

  <tr>
    <td style="padding: 8px;"><a href="https://huggingface.co/Qwen/Qwen2.5-0.5B"><code>Qwen2.5-0.5B</code></a> (reported)</td>
    <td>41.6</td><td>19.5</td><td>20.3</td><td>15.7</td><td>52.1</td><td>47.5</td><td>30.5</td><td>39.3</td>
  </tr>


<!-- https://qwenlm.github.io/blog/qwen2.5-llm/ -->

</table>

To automatically evaluate Qwen3-0.6B-diffusion-mdlm-v0.1 on all benchmarks, run:
```shell
bash examples/a2d/mdlm/eval.sh \
  --model_name_or_path dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1
```


## Citation

If you use Qwen3-0.6B-diffusion-mdlm-v0.1 or dLLM, please cite:

```bibtex
@misc{zhou2026dllm,
      title={dLLM: Simple Diffusion Language Modeling}, 
      author={Zhanhui Zhou and Lingjie Chen and Hanghang Tong and Dawn Song},
      year={2026},
      eprint={2602.22661},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.22661}, 
}
```