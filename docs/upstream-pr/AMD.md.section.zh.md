<!-- Append this as a new top-level section ABOVE the existing community content in
     docs/zh/usage/acceleration_cards/AMD.md. Do not alter the existing content below. -->

## gfx1100（RDNA3）— Radeon PRO W7900 / ROCm 7.2：社区验证（非官方支持）

> 以下为社区验证结果（[AIwork4me/MinerU-ROCm](https://github.com/AIwork4me/MinerU-ROCm)），非 MinerU 官方支持。
> 上游 README 已声明"非主线环境不保证 100% 可用、欢迎社区反馈"——本节即此类反馈。

MinerU 3.4 流水线与 MinerU2.5-Pro VLM（经 vLLM）在 gfx1100 上经全量 OmniDocBench v1.6（1651 页）
验证可**正确**运行，**无需修改任何 MinerU 源码**（仅环境变量）。

### 环境
GPU：gfx1100（Radeon PRO W7900，48 GB）｜ROCm 7.2，bf16，torch 2.9.1+rocm7.2｜
mineru 3.4.4（pipeline）；mineru_vl_utils 1.0.5 + vLLM-on-ROCm 0.16.1（VLM）

### 关键配置：HSA_OVERRIDE_GFX_VERSION（gfx1100/1101/1102）
- **pipeline 后端**（进程内 PyTorch）：**无需** override —— PyTorch-ROCm 自动识别 RDNA3。
- **VLM 后端经 vLLM**：**必须** `export HSA_OVERRIDE_GFX_VERSION=11.0.0`（vLLM 预编译内核需要）。
- Windows 原生 ROCm 可能不识别此 override（windows-hip 未验证）。

### 性能：重要
- **pipeline**：无需补丁，~3–6 s/页，速度正常。
- **VLM（vLLM）**：**无需补丁即可正确运行，但未打补丁时 ~15–16 s/页（偏慢）**。
  原因同上文：vLLM 的 `qwen2_vl.py` 视觉编码器 `Conv3d(bf16)` 在 RDNA3 缺优化内核而回退。
  **追求速度请沿用上文社区 Triton/矩阵乘补丁**（可降至 ~1.3–1.8 s/it）。本节"无需补丁"仅指**正确性**。

### OmniDocBench v1.6 全量结果（1651 页）
| 模型 / 后端 | Overall | Text EditDist ↓ | Formula CDM ↑ | Table TEDS ↑ |
|---|---:|---:|---:|---:|
| MinerU 3.4 pipeline（ROCm） | 86.48 | 0.0566 | 83.07 | 82.04 |
| MinerU2.5-Pro VLM（vLLM-on-ROCm） | 95.46 | 0.0360 | 96.46 | 93.54 |

与上游 README 官方锚点对齐（容差内）：pipeline 86.47（Δ+0.01pp）、vlm-engine 95.30（Δ+0.16pp，vLLM 非确定性范围内）。
完整可复现锁定（代码 commit、权重 SHA256、评分器 commit、环境）见
[reproducibility.lock.yaml](https://github.com/AIwork4me/MinerU-ROCm/blob/main/reproducibility.lock.yaml)。
