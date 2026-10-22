# Evidence Before Action

A Step-Level Self-Verification Framework for Computer-Use Agents: Capabilities ⬆️ Trustworthiness ⬆️ Safety ⬆️

## 📊 Main Results

### Qwen3.5-9B
#### Observation Type: `screenshot`

| Model Version | Successful Tasks | Success Rate | Δ vs. Base |
|---|---:|---:|---:|
| Base | 3/150 | 2.0% | — |
| SFT | 20/150 | 13.3% | +11.3% |
| Evidence | 17/150 | 11.3% | +9.3% |
| Evidence + OPD | 21/150 | 14.0% | +12.0% |

#### Observation Type: `screenshot + a11y-tree`

| Model Version | Successful Tasks | Success Rate | Δ vs. Base |
|---|---:|---:|---:|
| Base | 0/150 | 0.0% | — |
| SFT | 19/150 | 12.7% | +12.7% |
| Evidence | 20/150 | 13.3% | +13.3% |

### Qwen3.5-27B
#### Observation Type: `screenshot`

| Model Version | Successful Tasks | Success Rate | Δ vs. Base |
|---|---:|---:|---:|
| Base | 5/150 | 3.3% | — |
| SFT | 21/150 | 14.0% | +10.7% |
| Evidence | 25/150 | 16.7% | +13.4% |

### Qwen3.8-27B
#### Observation Type: `screenshot`

| Model Version | Successful Tasks | Success Rate | Δ vs. Base |
|---|---:|---:|---:|
| Base | 13/150 | 8.7% | — |
| SFT | 21/150 | 14.0% | +5.3% |
| Evidence | 32/150 | 21.3% | +12.6% |

### LoRA Rank Ablation for Evidence of Qwen3.5-9B 

| lora rank | lora alpha | VRAM | Training Time |Format Compliance | Success Rate | Δ vs. Base |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 45 GB ✖ 2 | 125 min | ✅ | 8.0% | +6.0% |
| 32 | 64 | 48 GB ✖ 2 | 150 min | ✅ | 8.6% | +6.6% |
| 128 | 256 | 50 GB ✖ 2 | 173 min | ✅ | 11.3% | +9.3% |

## 🚀 Deployment
All experiments are deployed with **Qwen3.5-9B-Evidence**.
### Quantization
| Scheme | Model Size | Inference Speed (tokens/s) | Quality Loss |
|---|---:|---:|---:|
| BF16 | 18 GB | 101 | baseline |
| FP8 (offline) | 13 GB | 129 | -0.0% |
| AWQ INT4 | 11 GB | 125 | -3.3% |
| GPTQ INT4 | 8 GB | 134 | -2.0% |

*Note: All results are measured without `flash-linear-attention`.*

### Online Serving Stress Test
Measured with vLLM’s benchmark CLI.
| Concurrency | QPS | E2EL P50 (ms) | E2EL P95 (ms) | E2EL P99 (ms) |
|---|---:|---:|---:|---:|
| 8 | 0.33 | 23964 | 23989 | 24019 |
| 32 | 0.99 | 32343 | 32379 | 32386 |
| 64 | 1.51 | 42353 | 42478 | 42483 |
| 128 | 2.00 | 63819 | 65043 | 65212 |

## 🔬 Evaluation Setup
| Configuration | Value |
|---|---|
| Benchmark | OSWorld Custom Test Split |
| Evaluation Tasks | 150 |
| Max Steps | 15 |
| Max Trajectory Length | 3 |
| Temperature | 1.0 |
| Top-P | 0.9 |
| Max Tokens | 4096 |
| Action Space | pyautogui |
| Provider | docker |
| Inference Framework | vLLM |

## 🧪 Model Versions

| Version | Description |
|---|---|
| Base | Original pretrained model without task-specific fine-tuning |
| SFT | Supervised fine-tuning using successful OSWorld trajectories |
| Evidence | Fine-tuning with explicit visual evidence and action verification |
| OPD | On-policy distillation using a stronger teacher model |

## 🤝 Acknowledgement
Evidence-Before-Action is built on top of [OSWorld](https://github.com/xlang-ai/OSWorld), [LlamaFactory](https://github.com/hiyouga/LlamaFactory) and [ms-swift](https://github.com/modelscope/ms-swift). We thank the authors of those projects.

## 📚 References
The following works were consulted during this project.

[1] Liu, Zhuchenyang, Yao Zhang, and Yu Xiao. "[Evidence Attribution in Visual Document Understanding without Coordinates or Region Labels.](https://arxiv.org/pdf/2607.24651)" arXiv preprint arXiv:2607.24651 (2026).

[2] Li, Weiming, et al. "[How auxiliary reasoning unleashes gui grounding in vlms.](https://arxiv.org/abs/2509.11548)" arXiv preprint arXiv:2509.11548 (2025).

[3] Wang, Suyuchen, et al. "[Improving GUI Grounding with Explicit Position-to-Coordinate Mapping.](https://arxiv.org/abs/2510.03230)" arXiv preprint arXiv:2510.03230 (2025).

[4] Yang, Rui, et al. "[Gui-libra: Training native gui agents to reason and act with action-aware supervision and partially verifiable rl.](https://arxiv.org/abs/2602.22190)" arXiv preprint arXiv:2602.22190 (2026).

[5] El hattami, Amine, et al. "[WebArena Verified: Reliable Evaluation for Web Agents.](https://neurips.cc/virtual/2025/loc/san-diego/124576)" NeurIPS 2025 Workshop on Scaling Environments for Agents, 2025.

[6] Li, Gengsheng, et al. "[Data-free on-policy distillation.](https://arxiv.org/abs/2609.14193)" arXiv preprint arXiv:2609.14193 (2026).

[7] Fu, Zixuan, et al. "[Rethinking on-policy distillation of large language models ii: One training example.](https://arxiv.org/abs/2609.04172)" arXiv preprint arXiv:2609.04172 (2026).
