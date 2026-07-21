# MinerU-ROCm

> [opendatalab/MinerU](https://github.com/opendatalab/MinerU) 的**评估背书 AMD ROCm 移植** ——
> 在 AMD **gfx1100 (RDNA3)** 上运行 **MinerU 3.4 pipeline** 与 **MinerU2.5-Pro** VLM，
> 跨多个推理后端报告 **OmniDocBench v1.6** 结果。**非**精度对齐移植：不存在同页集 CUDA 对照，
> 且上游 headline 可能用不同引擎。

[![OmniDocBench v1.6](https://img.shields.io/badge/OmniDocBench-v1.6-blue)](https://github.com/opendatalab/OmniDocBench)
[![VLM full](https://img.shields.io/badge/MinerU2.5--Pro%20VLM%20(full)-95.56-green)](#evaluation评测)
[![pipeline full](https://img.shields.io/badge/MinerU%203.4%20pipeline%20(full)-86.48-yellowgreen)](#evaluation评测)
[![status: evaluation-backed](https://img.shields.io/badge/status-evaluation--backed-blue)](reproducibility.lock.yaml)
[![license: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0%20(+MinerU%20terms)-blue)](NOTICE)

> 国内用户优先使用镜像与 ModelScope 拉取模型/数据集，速度更稳定。

## Install（安装）

核心包不依赖 GPU，也无平台依赖。

```bash
pip install -e ".[dev]"          # 核心 + dev/CI 工具（pytest、ruff、reuse）
# 可选：omnidocbench-rocm 引擎集成（adapter/run_adapter.py 路径）
pip install -e ".[platform]"
```

平台环境准备（权重、ROCm 运行时）：运行 `make setup-linux`（或
`make setup-windows`）。GPU 后端还需额外安装 ROCm torch +（VLM 所需）
vLLM-on-ROCm，须从已验证的 ROCm wheel 源单独安装 —— 见
`docs/reproducibility.md`。

**注意：** 环境准备脚本（`adapter/setup/`）为桩代码，记录了手动步骤但未
完全自动化环境安装。详见 `adapter/setup/00-install-deps.sh`。

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

正式 OmniDocBench-ROCm 平台评测使用 `omnidocbench-rocm`：

### MinerU2.5-Pro VLM（主 model card，`mineru2.5`）

```bash
omnidocbench-rocm run \
  --stage all \
  --platform linux-rocm \
  --version v16 \
  --revision 2b161d0 \
  --adapter adapter/run_adapter.py \
  --model-id mineru2.5 \
  --backend vlm-vllm \
  --server-url http://127.0.0.1:8265/v1 \
  --api-model-name mineru-pro \
  --git-commit "$(git rev-parse HEAD)" \
  --results-dir results/omnidocbench/v16/linux-rocm \
  --skip-existing
```

### MinerU 3.4 Pipeline（补充，`mineru-pipeline`）

```bash
omnidocbench-rocm run \
  --stage all \
  --platform linux-rocm \
  --version v16 \
  --revision 2b161d0 \
  --adapter adapter/run_adapter.py \
  --model-id mineru-pipeline \
  --backend pipeline \
  --git-commit "$(git rev-parse HEAD)" \
  --results-dir results/omnidocbench/v16/linux-rocm \
  --skip-existing
```

### 拆阶段执行（长时间 VLM 任务推荐）

```bash
omnidocbench-rocm infer --backend vlm-vllm ...
omnidocbench-rocm score ...
omnidocbench-rocm publish --predictions-dir <真实预测目录> ...
```

独立 `mineru-rocm` CLI（`predict` / `score`）仍可作为开发者调试工具使用，
详见 `docs/reproducibility.md`。

### 结果 —— MinerU2.5-Pro VLM（主 model card，`mineru2.5`）

| Model / Backend | Overall | Text Edit | Formula CDM | Table TEDS |
|---|---:|---:|---:|---:|
| _official_ MinerU2.5-Pro _(上游 README vlm-engine 行；社区验证，非官方支持)_ | 95.30 | — | — | — |
| **ours MinerU2.5-Pro（vlm-vllm，ROCm）** | **95.56** | 0.0359 | 96.73 | 93.54 |
| ours MinerU2.5-Pro（vlm-transformers，ROCm） | _仅采样_ | | | |

`vlm-vllm` 行在 linux-rocm **已复现**（自证，`badge: community`）：1651/1651 页
已尝试，1649 个产生非空预测（2 个空输出），无进程崩溃；单卡（gfx1100）约
7 小时；阅读顺序 EditDist 0.1240。Overall 95.56 与公开发布的上游参考区间**一致**
（vlm-engine 95.30；delta +0.26 pp —— **非**受控 CUDA-vs-ROCm 对照）。上游锚点
取自上游 README "Local Deployment" 表，属**社区验证、非官方支持** —— 见
`reproducibility.lock.yaml`（`benchmark.official_reference`）。`windows-hip`
仍为 `community-wanted`（暂无结果）。

> **历史分数说明：** 此前独立 `mineru-rocm score` 路径在同一份 1651 页预测上得分为
> **95.46**（Formula CDM 96.46）；当前平台 CDM 得分 **95.56**（Formula CDM 96.73）。
> 两者使用同一预测集（commit `b75f788`）与同一评分器（revision `2b161d0`），Δ +0.10 pp
> 完全来自 Formula-CDM 子指标（CDM 评分配置差异），并非重新推理。

### 结果 —— MinerU 3.4 pipeline（补充 model card，`mineru-pipeline`）

| Model / Backend | Overall | Text Edit | Formula CDM | Table TEDS |
|---|---:|---:|---:|---:|
| _official_ MinerU 3.4 pipeline | 86.47 | — | — | — |
| **ours MinerU 3.4 pipeline（ROCm gfx1100，linux-rocm）** | **86.48** | 0.0566 | 83.07 | 82.04 |
| windows-hip | _community-wanted_ | | | |

Pipeline 结果位于 `results/omnidocbench/v1.6/pipeline/`。已知：1 个空输出页面。
主 registry card 为 `mineru2.5`（VLM）；pipeline 为同仓补充 card —— 见
`model_card.pipeline.json`。

## Reproducibility（可复现性）

[`reproducibility.lock.yaml`](reproducibility.lock.yaml) 是唯一事实来源 ——
锁定的 commit、与上游 HF 仓交叉校验的逐字节权重/GT SHA256、环境版本，以及
指标公式。已验证值来自 2026-07-19 完成的全量 1651 页重跑。

硬件：AMD gfx1100（Radeon PRO W7900），48 GB VRAM，ROCm 7.2，bf16。
官方参考（pipeline 86.47，vlm-engine 95.30）取自上 MinerU README
"Local Deployment" 表，作为社区验证锚点，非官方支持。详见
`docs/reproducibility.md`。

## License —— 下载权重前必读

本仓为 **Apache-2.0**（原创打包/工具）。MinerU pipeline 遵循 **MinerU Open Source License**
（Apache-2.0 + 附加条款：MAU 超 1 亿或月营收超 2000 万美元需另获商业授权；
在线服务须标注 MinerU）。`mineru-vl-utils` 与 MinerU2.5-Pro 权重为
Apache-2.0。**PDF-Extract-Kit-1.0** pipeline 权重在 HF 卡片上**未声明**
license —— 视为授权不明，请勿再分发。完整分解见 [NOTICE](NOTICE) 与
[LICENSES/](LICENSES)。本仓与 MinerU Team / OpenDataLab 无隶属关系。

## Issues filed（已提交的 issue）

- **[ROCm/AMDMIGraphX#5078](https://github.com/ROCm/AMDMIGraphX/issues/5078)** —— 影响 ROCm 上 ONNX 表格识别的 loop-subgraph 解析器 bug。
- 上游 `opendatalab/MinerU` AMD.md 贡献 + PDF-Extract-Kit-1.0 license 澄清计划在 P4 进行。

## Known Gaps（已知限制）

- `smoke` 后端输出的是占位文本，并非真实 OCR。CI/conformance 可通过 `--backend smoke` 验证适配器契约而无需 GPU。
- **Windows-HIP** 为 `community-wanted` —— 尚无正式结果。两个 model card 的 `windows-hip` badge 均为 `community-wanted`。
- **环境准备脚本**（`adapter/setup/`）为桩代码，仅记录手动步骤，未完全自动化环境安装。
- **平台标准 artifacts** 已于 2026-07-21 在 `results/omnidocbench/v16/linux-rocm/` 生成（`mineru2.5` 与 `mineru-pipeline` 的自包含 CDM bundle：`run_summary` + `provenance` + `metric_result` + `run_stats` + SHA256 `prediction_manifest` + `dataset_identity`）。`results/omnidocbench/v1.6/` 下的遗留结果保留用于历史对比与预测来源 provenance。可用 `omnidocbench-rocm validate-bundle results/omnidocbench/v16/linux-rocm` 校验任意 bundle。
- **VLM 空输出：** 1651 个 VLM 页面中 2 个产生空预测（已记录为失败）。
- **Pipeline 空输出：** 1651 个 pipeline 页面中 1 个产生空预测。
- **Conformance** 通过所有结构性检查；完整 `CONFORMANT` 状态需通过 `omnidocbench-rocm publish` 生成平台标准 artifacts。
- 完整列表见 [`docs/known-gaps.md`](docs/known-gaps.md)。
