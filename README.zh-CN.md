# MinerU-ROCm

[omnidocbench-amd](https://github.com/AIwork4me/OmniDocBench-AMD) 文档解析评测平台的单模型适配仓库。由官方 cookiecutter 模板生成；自带无需 GPU 的 `smoke` 后端，开箱即用。

- 模型：`mineru2.5`（VLM checkpoint 2605）
- 平台：linux-rocm、windows-hip
- 徽章：linux-rocm = `community`（OmniDocBench v1.6 Overall **95.56**，已复现）；windows-hip = `community-wanted`。`verified` 需维护者 Docker 复现。

> 国内用户优先使用镜像与 ModelScope 拉取模型/数据集，速度更稳定。

## Install（安装）

```bash
pip install -e ".[dev]"
pip install omnidocbench-amd        # 引擎（提供 `omnidocbench-amd` CLI 与类型）
```

平台环境准备（权重、ROCm/DirectML 运行时）：

```bash
make setup-linux     # 或：make setup-windows
```

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

## Reproducibility（可复现性）

结果位于 `results/omnidocbench/v16/<platform>/`。每次运行产出经 schema 校验的 `run_summary.json` + `provenance.json`（引擎版本、git commit、数据集版本、适配器命令），确保在声明的硬件上凭已提交的适配器与配置即可独立复现该分数。详见 [`docs/reproducibility.md`](docs/reproducibility.md)。

## Known Gaps（已知限制）

- `smoke` 后端输出的是占位文本，并非真实 OCR；`pipeline`（默认，真实 MinerU 3.4 in-process 适配器）是生产路径。CI/conformance 可通过 `BACKEND=smoke` 或 `--backend smoke` 强制使用 `smoke`。
- `mineru-pipeline` 在 linux-rocm 上为 `community`（OmniDocBench v1.6 Overall **86.48**，gfx1100 —— 详见 [`docs/reproducibility.md`](docs/reproducibility.md)）；windows-hip 仍为 `community-wanted`。
- 环境准备脚本（`adapter/setup/`）为桩代码。
- 完整列表见 [`docs/known-gaps.md`](docs/known-gaps.md)。
