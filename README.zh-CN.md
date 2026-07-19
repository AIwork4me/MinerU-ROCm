# MinerU-ROCm

> [opendatalab/MinerU](https://github.com/opendatalab/MinerU) 的**评估背书 AMD ROCm 移植** ——
> 在 AMD **gfx1100 (RDNA3)** 上运行 **MinerU 3.4 pipeline** 与 **MinerU2.5-Pro** VLM，
> 跨多个推理后端报告 **OmniDocBench v1.6** 结果。**非**精度对齐移植：不存在同页集 CUDA 对照，
> 且上游 headline 可能用不同引擎。见基准方法学 *（P2 落地）*。

[![OmniDocBench v1.6](https://img.shields.io/badge/OmniDocBench-v1.6-blue)](https://github.com/opendatalab/OmniDocBench)
[![VLM full](https://img.shields.io/badge/MinerU2.5--Pro%20VLM%20(full)-95.56-green)](#结果--mineru25-pro-vlm主-model-cardmineru25)
[![pipeline full](https://img.shields.io/badge/MinerU%203.4%20pipeline%20(full)-86.48-yellowgreen)](#结果--mineru-34-pipeline次要-model-cardmineru-pipeline)
[![status: evaluation-backed](https://img.shields.io/badge/status-evaluation--backed-blue)](reproducibility.lock.yaml)
[![license: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0%20(+MinerU%20terms)-blue)](NOTICE)

> 国内用户优先使用镜像与 ModelScope 拉取模型/数据集，速度更稳定。

## 概览（At a glance）

- **是什么。** 在 AMD ROCm 上运行 opendatalab MinerU（3.4 pipeline + 2.5-Pro VLM）并在 OmniDocBench v1.6 上评分的工具。
- **在哪验证。** AMD **gfx1100 (RDNA3, 48 GB ×4)，ROCm 7.2**，bf16。
- **最可靠结果。** **MinerU2.5-Pro VLM (vLLM-on-ROCm) 全量 1651 = 95.56 Overall**；**MinerU 3.4 pipeline 全量 1651 = 86.48 Overall**。
- **最重要限制。** **非精度对齐。** 无同引擎 CUDA 对照；上游 headline 可能用不同引擎测量。所谓"官方 95.75"锚点正在重新核实（上游指向 ~95.69）—— 见填充后的 `reproducibility.lock.yaml`。
- **上游。** 本仓是 [opendatalab/MinerU](https://github.com/opendatalab/MinerU) 的移植；[omnidocbench-amd](https://github.com/AIwork4me/OmniDocBench-AMD) 引擎只是**可选**消费者（装 `[platform]` extra），不是本仓的定义。

## Install（安装）

核心包不依赖 GPU，也无平台依赖。

```bash
pip install -e ".[dev]"          # 核心 + dev/CI 工具（pytest、ruff、reuse）
# 可选：omnidocbench-amd 引擎集成（adapter/run_adapter.py 路径）
pip install -e ".[platform]"
```

平台环境准备（权重、ROCm 运行时）：运行 `make setup-linux`（或
`make setup-windows`）。GPU 后端还需额外安装 ROCm torch +（VLM 所需）
vLLM-on-ROCm，须从已验证的 ROCm wheel 源单独安装 —— 见
`docs/reproducibility.md`。

## Demo（演示）

`smoke` 后端无需 GPU，会为每张图片写出占位 `.md`，便于端到端验证契约：

```bash
bash examples/run_demo.sh        # Linux/macOS
# .\examples\run_demo.ps1        # Windows
```

或直接调用：

```bash
python adapter/run_adapter.py --img-dir examples --out-dir /tmp/out --platform linux-rocm --backend smoke
```

## Evaluation（评测）

在 `_infer` 接入真实模型后，运行完整 OmniDocBench v1.6 流程（下载 → 推理 → 打分 → 发布）：

```bash
make eval-linux      # linux-rocm
# make eval-windows  # windows-hip（在 Windows 上运行）
```

评测配置：[`eval/configs/omnidocbench_v16.yaml`](eval/configs/omnidocbench_v16.yaml)。

### 结果 —— MinerU2.5-Pro VLM（主 model card，`mineru2.5`）

| Model / Backend | Overall ↑ | Text Edit ↓ | Formula CDM ↑ | Table TEDS ↑ |
|---|---:|---:|---:|---:|
| _official_ MinerU2.5-Pro | 95.75 | 0.036 | 97.45 | 93.42 |
| **ours MinerU2.5-Pro（vlm-vllm，ROCm）** | **95.56** | 0.0359 | 96.73 | 93.54 |
| ours MinerU2.5-Pro（vlm-transformers，ROCm） | _仅采样（质量干净；全量约 44 h）_ | | | |

`vlm-vllm` 行在 linux-rocm **已复现**（自证、conformance 通过，`badge: community`）：1651/1651 页、0 失败、GPU 0（gfx1100）约 7 小时、空页率 0.12%、阅读顺序 EditDist 0.1240。以 +0.31 pp 距官方 95.75 PASS（≤0.5 pp）。`vlm-transformers` 后端是干净但较慢的 fallback（约 100–150 s/页；全量约 44 h 未跑），因此无完整 Overall。`windows-hip` 仍为 `community-wanted`。

### 结果 —— MinerU 3.4 pipeline（次要 model card，`mineru-pipeline`）

| Model / Backend | Overall ↑ | Text Edit ↓ | Formula CDM ↑ | Table TEDS ↑ |
|---|---:|---:|---:|---:|
| _official_ MinerU 3.4 pipeline | 86.47 | — | — | — |
| **ours MinerU 3.4 pipeline（ROCm gfx1100，linux-rocm）** | **86.48** | 0.0566 | 83.07 | 82.04 |
| windows-hip | _待测（同事）_ | | | |

两个 `linux-rocm` 行均**已复现**（自证、conformance 通过，`badge: community`）—— 详见 [`docs/reproducibility.md`](docs/reproducibility.md)。上方主 `mineru2.5` VLM 行填充 `hub/registry.yaml` 中的 `mineru2.5` 条目；pipeline 记录在此处的 `model_card.pipeline.json` 与本表格（无独立 registry 行）。

## License —— 下载权重前必读

本仓为 **Apache-2.0**（原创打包/工具）。MinerU pipeline 遵循 **MinerU Open Source License**（Apache-2.0 + 附加条款：MAU 超 1 亿或月营收超 2000 万美元需另获商业授权；在线服务须标注 MinerU）。`mineru-vl-utils` 与 MinerU2.5-Pro 权重为 Apache-2.0。**PDF-Extract-Kit-1.0** pipeline 权重在 HF 卡片上**未声明** license —— 视为授权不明，请勿再分发。完整分解见 [NOTICE](NOTICE) 与 [LICENSES/](LICENSES)。本仓与 MinerU Team / OpenDataLab 无隶属关系。

## Reproducibility（可复现性）

[`reproducibility.lock.yaml`](reproducibility.lock.yaml) 是唯一事实来源 —— 锁定的 commit、与上游 HF 仓交叉校验的逐字节权重/GT SHA256、环境版本，以及指标公式。*（P0 仅交付骨架；P3 在全量重跑后填充已验证值。）* 详见 [docs/reproducibility.md](docs/reproducibility.md)。

## Issues filed（已提交的 issue）

- **[ROCm/AMDMIGraphX#5078](https://github.com/ROCm/AMDMIGraphX/issues/5078)** —— 影响 ROCm 上 ONNX 表格识别的 loop-subgraph 解析器 bug。
- 上游 `opendatalab/MinerU` AMD.md 贡献 + PDF-Extract-Kit-1.0 license 澄清计划在 P4 进行。

## Known Gaps（已知限制）

- `smoke` 后端输出的是占位文本，并非真实 OCR；`pipeline`（默认，真实 MinerU 3.4 in-process 适配器）是生产路径。CI/conformance 可通过 `BACKEND=smoke` 或 `--backend smoke` 强制使用 `smoke`。
- `mineru-pipeline` 在 linux-rocm 上为 `community`（OmniDocBench v1.6 Overall **86.48**，gfx1100 —— 详见 [`docs/reproducibility.md`](docs/reproducibility.md)）；windows-hip 仍为 `community-wanted`。
- 环境准备脚本（`adapter/setup/`）为桩代码。
- 完整列表见 [`docs/known-gaps.md`](docs/known-gaps.md)。
