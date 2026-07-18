# Spike: vLLM-on-ROCm serving MinerU2.5-Pro-2605-1.2B (VLM port de-risk)

Status: **DONE** — vLLM-on-ROCm serves MinerU2.5-Pro on GPU 0; two-step +
`MinerULogitsProcessor` produces sane Markdown against real OmniDocBench pages.
vLLM is **viable as the VLM primary backend** for Plan 2.

Spike date: 2026-07-18. GPU 0 only (`HIP_VISIBLE_DEVICES=0`); GPU 3 untouched
(1651-page pipeline eval running concurrently).

## 0. Environment (one-time setup)

- **Python / vLLM**: `/opt/venv/bin/python` (system venv).
  - `vllm 0.16.1.dev0+g89a77b108.d20260317` (ROCm rocm721 nightly).
  - `torch 2.9.1+gitff65f5b` (rocm), `torch.cuda.is_available()==True`, device count 4.
  - `transformers 4.57.6`.
  - `is_torch_equal("2.9.1")` returns **True** for `2.9.1+gitff65f5b` (matters for Conv3d — see §4).
- **mineru-vl-utils**: installed `mineru-vl-utils==1.0.5` into `/opt/venv` via
  `HF_ENDPOINT=http://134.199.133.77 /opt/venv/bin/pip install mineru-vl-utils`.
  Lightweight (aiofiles, httpx-retries). No `mineru` (full) needed for the VLM-only path.
- **Weights**: `opendatalab/MinerU2.5-Pro-2605-1.2B`, downloaded via hf-mirror
  (`HF_ENDPOINT=http://134.199.133.77`, `snapshot_download`). Landed at the
  default cache `/root/.cache/huggingface/models--opendatalab--MinerU2.5-Pro-2605-1.2B/`
  (NOT under `hub/` — this cache uses the flat layout). Snapshot:
  `bff20d4ae2bf202df9f45284b4d43681555a97ed`. Total ~2.31 GB (one BF16 safetensors blob).
- **Model config**: `architectures: ["Qwen2VLForConditionalGeneration"]`,
  `model_type: qwen2_vl`, `hidden_size=896`, `num_hidden_layers=24` (1.156B params BF16).
  `torch_dtype` is unset in config; vLLM `--dtype bfloat16` forces it.

## 1. Boot: does vLLM-on-ROCm boot + serve OpenAI-compatible? — **YES**

### Exact launch command (works as-is)
Script: `/tmp/vlm-spike/launch_vllm.sh`
```bash
export HIP_VISIBLE_DEVICES=0          # GPU 0 ONLY
export HF_ENDPOINT=http://134.199.133.77
export VLLM_USE_V1=1                   # V1 engine required for v1 logits-processor API
export HSA_OVERRIDE_GFX_VERSION=11.0.0 # gfx1100 / W7900 RDNA3
MODEL=/root/.cache/huggingface/models--opendatalab--MinerU2.5-Pro-2605-1.2B/snapshots/bff20d4ae2bf202df9f45284b4d43681555a97ed
exec /opt/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name mineru-pro \
  --trust-remote-code \
  --dtype bfloat16 \
  --chat-template /tmp/vlm-spike/qwen2vl_chat_template.jinja \
  --logits-processors mineru_vl_utils:MinerULogitsProcessor \
  --host 127.0.0.1 --port 8265 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 1}' \
  --enforce-eager
```
Launch detached (foreground gets killed by harness):
```bash
cd /tmp/vlm-spike && nohup bash launch_vllm.sh > vllm.log 2>&1 &
echo $! > launcher.pid
```

### Two launch gotchas (both resolved, documented for Plan 2)
1. **`--limit-mm-per-prompt` JSON**: this vLLM build rejects `image=1`; must pass
   `'{"image": 1}'` (JSON object). Error otherwise: `Value image=1 cannot be
   converted to <function loads ...>`.
2. **`--chat-template` is REQUIRED**: `opendatalab/MinerU2.5-Pro-2605-1.2B` ships
   `tokenizer_config.json` with **no `chat_template` field** (verified). Without
   `--chat-template`, every request returns HTTP 400:
   `As of transformers v4.44, default chat template is no longer allowed`.
   Fix: extract the standard Qwen2-VL chat template from any Qwen2-VL-Instruct
   tokenizer and pass it as a `.jinja` file. We wrote it to
   `/tmp/vlm-spike/qwen2vl_chat_template.jinja` via:
   ```python
   from transformers import AutoTokenizer
   t = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
   open("/tmp/vlm-spike/qwen2vl_chat_template.jinja","w").write(t.chat_template)
   ```
   **Plan-2 task**: vendor this jinja into the repo (`eval/` or `adapter/`) so the
   server doesn't depend on `/tmp`. Or write a `chat_template` field into a
   sidecar tokenizer config and pass `--tokenizer` to it.

### GPU-memory-utilization
First attempt with `--gpu-memory-utilization 0.85` failed:
`ValueError: Free memory on device cuda:0 (38.28/47.98 GiB) on startup is less
than desired GPU memory utilization (0.85, 40.79 GiB).`
GPU 0 already had ~10 GB used by another process. Lowering to **0.70** (33.6 GB
target) booted cleanly. Plan 2: if GPU 0 is shared, use ≤0.70; on a clean GPU,
0.85–0.90 is fine.

### `/v1/models` response (proof of boot)
```bash
$ curl -s http://127.0.0.1:8265/v1/models
{"object":"list","data":[{"id":"mineru-pro","object":"model","created":1784343396,
"owned_by":"vllm","root":".../snapshots/bff20d4ae2bf202df9f45284b4d43681555a97ed",
"max_model_len":8192,...}]}
```
Health: `GET /health` → 200 OK. `Application startup complete` in log.
Engine config log line confirms the logits processor is wired:
```
non-default args: {... 'logits_processors': ['mineru_vl_utils:MinerULogitsProcessor'], ...}
```

### Platform detection — NOT a blocker here
The brief's "platform detection" blocker (MinerU issue #4655,
`UnspecifiedPlatform`/`Device string must not be empty`) **did not occur**. That
class of issue was solved for Unlimited-OCR (see
`/workspace/Unlimited-OCR-ROCm/patches/vllm/` — but those patches register a
brand-new model class absent from the old wheel; they are not platform patches).
Qwen2-VL is natively supported in vLLM 0.16.1, and the installed ROCm wheel boots
on gfx1100 with just `HSA_OVERRIDE_GFX_VERSION=11.0.0`. **No vLLM rebuild
needed.**

## 2. Two-step + MinerULogitsProcessor → sane Markdown — **YES**

### Test harness
`/tmp/vlm-spike/test_twostep.py` — uses `mineru_vl_utils.MinerUClient` with
`backend="http-client"`, `server_url="http://127.0.0.1:8265/v1"`,
`model_name="mineru-pro"`. Calls `client.two_step_extract(PIL.Image)` (the full
layout→per-block-extract chain), then `json2md(result)` to render Markdown.

### 3 real OmniDocBench pages (from `/root/ocr-eval/OmniDocBench_data/images/`)

**Page 1: `PPT_1001115_eng_page_003.png`** (English PPT, text-heavy) — 10.58s, 7 blocks:
```
Who Am I?

• Min-Te Sun (Peter) Sun

- An associate professor of Computer Science & Information Engineering, National Central University

- Studied in US for a long time (from 1993 ~ 2002)

- Worked as a CS professor at Auburn University, Alabama between 2002 and 2008 (before coming back to Taiwan)

- Have taught CS courses in English for more than 10 years
```

**Page 2: `exam_paper_2004-2019上海高考英语听力原文和答案_page_010.png`** (bilingual EN+ZH exam) — 32.67s, 38 blocks. Sample:
```
Section B
(A) 56. A 57. D 58. B 59. D
(B) 60. D 61. C 62. C
...
74. During the past three years, he has committed himself studying the relationship/ connection between the transmission speed information and the pace of human life.
...
F: We will stay there for one night. That means we will leave the camp at Aug 7 \( ^{th} \) .
...
2016年全国普通高等学校招生统一考试
上海 英语试卷
听力文字
```
Formulas rendered as LaTeX `\( ^{th} \)`. Bilingual text (English + Chinese)
preserved. Answer grids as multi-line text.

**Page 3: `color_textbook_..._page_001.png`** (textbook cover) — 1.13s, 1 block.
Empty Markdown — sane (cover page has no extractable body text).

### Layout detection format (correct)
Raw layout output:
```
<|box_start|>000 000 999 999<|box_end|><|ref_start|>image<|ref_end|><|rotate_up|>
```
Parsed by `MinerUClient.parse_layout_output` into typed `ContentBlock`s with
bbox + rotation + type (`text`/`image`/etc.). Then per-block extracts run
concurrently over cropped block images. OTSL→HTML table post-processing is in
the `post_process/` chain (not triggered on these pages — no table blocks).

### `MinerULogitsProcessor` confirmed applied
- Launch flag: `--logits-processors mineru_vl_utils:MinerULogitsProcessor`
  (echoed in the engine `non-default args` log).
- `MinerULogitsProcessor` is an alias for
  `VllmV1NoRepeatNGramLogitsProcessor` (no-repeat-100-gram; defined in
  `mineru_vl_utils/logits_processor/vllm_v1_no_repeat_ngram.py`). It is a **v1
  engine** processor (hence `VLLM_USE_V1=1` is mandatory).
- Every request from the http-client carries
  `"vllm_xargs": {"no_repeat_ngram_size": 100}` (seen in DEBUG request bodies),
  which the processor reads via `extra_args`. Confirmed end-to-end plumbing.

## 3. effort/high → two-step VLM path — **CONFIRMED**

`effort` is a **`mineru` Hybrid-backend** concept (`"medium"`/`"high"`), defined
in `mineru/backend/hybrid/hybrid_analyze.py:81` (`HYBRID_ANALYZE_EFFORTS`), NOT
in `mineru-vl-utils` (which only does the 2-step VLM calls; it has no "effort"
notion). At `effort=="high"` the Hybrid backend calls (line 1005):
```python
window_model_list = predictor.batch_two_step_extract(
    images=images_pil_list, image_analysis=effective_image_analysis, ...)
```
where `predictor` is a `MinerUClient`. **So `effort=high` IS the VLM two-step
path** validated in §2 — the same path that produced the 95.75 reference score.
Plan 2 wires `MinerUClient(backend="http-client", server_url=<our vLLM>)` into
the Hybrid backend's `predictor` slot; effort=high then runs against our ROCm
vLLM server unchanged.

## 4. Throughput / Conv3d — **Conv3d is NOT the bottleneck; full ViT is**

### Direct micro-benchmark (gfx1100, bf16)
`/tmp/vlm-spike/time_vit.py` — timed the Qwen2VL patch-embed Conv3d directly
(`F.conv3d` vs the 5D-GEMM `_forward_mulmat`) at realistic input
`(B=1, C=3, T=2, H=1008, W=1008)`, kernel/stride `(2,14,14)`:
```
conv3d(MIOpen)      :     0.3 ms/fwd
mulmat(linear)      :     0.3 ms/fwd
```
**Both paths are 0.3 ms.** At the patch-embed grid size (72×72 tokens), Conv3d
is negligible on ROCm. This **contradicts the brief's "Conv3d-BF16 ~12s/forward"
assumption** for this specific model/wheel — that diagnosis applied to older
vLLM/PyTorch combos on RDNA3.

### Why Conv3d is already fast here (no patch needed)
`vllm/model_executor/layers/conv.py:77` sets `enable_linear=True` when
`kernel_size == stride and not any(padding) and groups == 1`. Qwen2VL constructs
`Conv3dLayer(kernel_size=(2,14,14), stride=(2,14,14), bias=False)` with **no
padding** (`vllm/model_executor/models/qwen2_vl.py:461`) → `enable_linear=True`.
`forward_cuda` then takes `_forward_mulmat` (the 5D-GEMM fast path) iff
`is_torch_equal("2.9.0") or is_torch_equal("2.9.1")`. Our wheel reports
`2.9.1+gitff65f5b`, and `is_torch_equal("2.9.1")` returns **True** — so the
fast mulmat path is **already active** in this vLLM/PyTorch combo. The upstream
fix for vLLM issue #27406 is included.

### Where the 7 s/TTFT actually comes from
vLLM metrics over 45 requests (3 pages × ~15 blocks each):
- `time_to_first_token_seconds_sum / count = 325.2 / 45 =` **avg 7.23 s/TTFT**.
- TTFT includes base64 decode, the **full Qwen2VL ViT** (~675M-param vision
  encoder, deep attention stack — NOT Conv3d), and the first LM prefill step.
- The full `two_step_extract` for the 38-block exam page: 32.67s end-to-end.
- Warm layout-pass only (1 image, ~22 output tokens): **2.7s** (measured via
  `/tmp/vlm-spike/time_layout.py`).

### Implication for Plan 2
- **Do NOT spend time on the Conv3d Triton/GEMM patch from
  `docs/zh/usage/acceleration_cards/AMD.md`** — vLLM 0.16.1 on torch 2.9.1
  already routes Conv3d through the linear path; a patch would be a no-op.
- The real throughput cost is the **vision encoder + per-block-extract fan-out**
  (one ViT forward per block image). For the full 1651-page OmniDocBench eval
  at ~7s TTFT × N blocks/page × 1651 pages, this is the Plan-2 throughput
  problem to attack — via (a) vLLM continuous batching of block extracts
  (`mineru-vl-utils` already batches via `batch_extract_with_layout`), (b)
  caching layout-detect embeddings, (c) disabling `enforce-eager` once CUDA
  graphs are validated for this model on gfx1100 (we ran eager for the spike;
  graph capture may cut decode-side overhead).

## 5. Decision for Plan 2 — **vLLM-on-ROCm is the VLM primary backend**

**Recommendation**: use vLLM-on-ROCm as the VLM primary backend. It boots on GPU
0 with the installed wheel (no from-source rebuild), serves OpenAI-compatible,
honours `MinerULogitsProcessor` (v1 engine), and produces sane Markdown on real
OmniDocBench pages via the `mineru-vl-utils` two-step path that the `mineru`
Hybrid backend uses for `effort=high`.

**Plan-2 action items** (in priority order):
1. **Vendor the Qwen2-VL chat template** (`/tmp/vlm-spike/qwen2vl_chat_template.jinja`)
   into the repo and reference it via `--chat-template`. Without it the server is
   unusable (HTTP 400 on every request). The model's own `tokenizer_config.json`
   lacks the field.
2. **Pin the launch flags**: `VLLM_USE_V1=1` (mandatory for the v1 logits
   processor), `HSA_OVERRIDE_GFX_VERSION=11.0.0`, `--dtype bfloat16`,
   `--logits-processors mineru_vl_utils:MinerULogitsProcessor`,
   `--limit-mm-per-prompt '{"image": 1}'`, `--gpu-memory-utilization 0.70` if GPU
   0 is shared (≤0.90 otherwise), `--max-model-len 8192`.
3. **Wire `MinerUClient(backend="http-client")` into the Hybrid backend's
   `predictor`** for `effort=high`. No code change to `mineru-vl-utils` needed;
   the Hybrid backend already calls `batch_two_step_extract`.
4. **Throughput**: budget ~7 s/TTFT × N blocks/page for the full eval; rely on
   vLLM continuous batching + `batch_extract_with_layout`. Revisit
   `--enforce-eager` off (CUDA graphs) once a single-block correctness check
   passes. **Do NOT apply the Conv3d patch** — it is a no-op on this wheel.
5. **Fallback**: if a future vLLM ROCm wheel regression breaks boot, the
   `mineru-vl-utils` `transformers` backend (`Qwen2VLForConditionalGeneration`) is
   the fallback — same two-step, same logits-processor semantics, ~2× slower but
   no platform/boot risk. Not needed at current wheel.

### Blockers
**None** for vLLM-on-ROCm as primary. The only operational gotcha is the
chat-template file, which is a config step, not a code blocker.
