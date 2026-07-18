# MinerU-ROCm — MinerU2.5-Pro VLM Implementation Plan (Plan 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the **MinerU2.5-Pro-2605-1.2B VLM** to MinerU-ROCm and reproduce official OmniDocBench v1.6 **Overall 95.75** on AMD gfx1100 (gate: within 0.5 pp).

**Architecture:** A persistent vLLM-on-ROCm server (GPU 0) serves the Qwen2-VL model with the `MinerULogitsProcessor`; `vlm_adapter` drives `mineru-vl-utils`' `MinerUClient(backend="http-client")` two-step (layout→per-block-extract) against it, returning Markdown. A secondary `vlm-transformers` backend runs the same two-step via the in-process transformers engine for a precision comparison. The dispatcher (from Plan 1) already routes `--backend vlm-vllm`/`vlm-transformers` to `vlm_adapter`; Plan 2 fills it.

**Tech Stack:** Python 3.11/3.12, vLLM 0.16.1 (ROCm) + `mineru-vl-utils` 1.0.5, `opendatalab/MinerU2.5-Pro-2605-1.2B` (Qwen2-VL, 1.156B BF16), the `omnidocbench-amd` engine (infer/score/publish), pytest.

## Global Constraints

- Repo code on `/workspace/MinerU-ROCm` (10 GB NFS, code-only). Heavy data (venvs/weights/outputs) → `/root`. The VLM env is `/opt/venv` (already has `vllm 0.16.1.dev0 rocm` + `mineru_vl_utils` from the spike); `pip install -e /workspace/omnidocbench-amd` into it so the adapter subprocess resolves `omnidocbench_amd.types`. Scoring runs in `/root/ocr-eval/OmniDocBench/.venv` (Py3.11, has the OmniDocBench scorer).
- **GPU 0 ONLY** for the VLM (`HIP_VISIBLE_DEVICES=0`). `HSA_OVERRIDE_GFX_VERSION=11.0.0`. Never touch GPU 3.
- hf-mirror only: `HF_ENDPOINT=http://134.199.133.77` (huggingface.co is blocked). Weights already at `/root/.cache/huggingface/models--opendatalab--MinerU2.5-Pro-2605-1.2B/`.
- **`VLLM_USE_V1=1` is mandatory** (the `MinerULogitsProcessor` is a v1-engine processor). Launch flags from the spike (§1 of `docs/spike-vlm-vllm.md`): `--logits-processors mineru_vl_utils:MinerULogitsProcessor`, `--dtype bfloat16`, `--chat-template <repo>/adapter/qwen2vl_chat_template.jinja` (REQUIRED — the model's tokenizer_config has no chat_template), `--limit-mm-per-prompt '{"image": 1}'` (JSON), `--gpu-memory-utilization 0.70`, `--max-model-len 8192`, `--enforce-eager`.
- **Run vLLM as a background process** (`nohup ... &`, capture PID) — foreground `vllm` gets killed by the harness.
- **Engine `LinuxRocmBackend.score()` is an unfinished stub** (Plan 1 finding) → score via `pdf_validation.py --config <yaml>` directly, in the OmniDocBench venv, then engine `publish`.
- **Backend selection**: the engine's `infer` does not pass `--backend`; `adapter_config.BACKEND` is the source of truth. Plan 2 makes it env-overridable (`MINERU_ROCM_BACKEND`), so the VLM eval sets `MINERU_ROCM_BACKEND=vlm-vllm` (pipeline remains the default).
- **Precision bar = evaluation-backed**: reproduce the official number on the full 1651-page set with provenance; **gate ≤ 0.5 pp of 95.75** (i.e. ≥ 95.25). Composite `Overall = ((1−Text_EditDist)×100 + Table_TEDS + Formula_CDM)/3`. Report both engines + the delta.
- Spec: `docs/superpowers/specs/2026-07-17-mineru-rocm-design.md` §8. Spike (authoritative API/recipe): `docs/spike-vlm-vllm.md`.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `adapter/qwen2vl_chat_template.jinja` | The Qwen2-VL chat template the model's tokenizer_config lacks (vendored) | 1 |
| `adapter/adapter_config.py` | `BACKEND` becomes env-overridable (`MINERU_ROCM_BACKEND`, default `pipeline`) | 1 |
| `adapter/vlm_adapter.py` | `backend=vlm-*`: lazy `MineruVLRunner` → `MineruClient(http-client)` two-step → Markdown | 2 |
| `examples/serve_vlm_vllm.sh` | Launch the vLLM-on-ROCm server (spike flags) as a background process + health check | 3 |
| `tests/test_vlm_adapter.py` | CPU unit test for the `normalize_vlm_markdown` helper (no vllm/mineru-vl-utils import) | 2 |
| `eval/configs/omnidocbench_v16_vlm.yaml` | Scoring config for the VLM prediction dir (copy of the pipeline one) | 6 |
| `model_card.json` | VLM (primary) card — filled with the result | 8 |
| `README.md` / `README.zh-CN.md` | Add the VLM row to the comparison table | 8 |

---

## Task 1: Vendor the Qwen2-VL chat template + env-overridable BACKEND

**Files:**
- Create: `adapter/qwen2vl_chat_template.jinja`
- Modify: `adapter/adapter_config.py`

**Interfaces:**
- Produces: `adapter/qwen2vl_chat_template.jinja` (consumed by `examples/serve_vlm_vllm.sh` via `--chat-template`); `adapter_config.BACKEND = os.environ.get("MINERU_ROCM_BACKEND", "pipeline")`.

- [ ] **Step 1: Fetch the Qwen2-VL chat template** (the model's tokenizer_config has none — spike finding). Via hf-mirror:

```bash
cd /workspace/MinerU-ROCm
HF_ENDPOINT=http://134.199.133.77 /opt/venv/bin/python - <<'PY'
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct").chat_template
assert t and "<|im_start|>" in t, "chat template looks wrong"
open("adapter/qwen2vl_chat_template.jinja", "w").write(t)
print("wrote", len(t), "bytes")
PY
```
Expected: `wrote <N> bytes` (N ≈ 1500–3000). If the hf-mirror fetch fails, fall back to any installed Qwen2-VL-Instruct tokenizer or vendor the standard template from the vLLM `qwen2_vl` model code — do NOT proceed with an empty file.

- [ ] **Step 2: Make BACKEND env-overridable** — in `adapter/adapter_config.py`, change the `BACKEND` line:

```python
import os
# smoke = no-GPU CI placeholder (CI/conformance only). Real: pipeline | vlm-vllm | vlm-transformers.
# Env-overridable so a single repo can run either model: the engine's `infer` does not pass
# --backend, so this default is the source of truth (pipeline for the 3.4 pipeline; set
# MINERU_ROCM_BACKEND=vlm-vllm for the VLM). CI/conformance forces smoke via the unit tests.
BACKEND = os.environ.get("MINERU_ROCM_BACKEND", "pipeline")
```
(Add `import os` at the top if not present. Default unchanged → pipeline; Plan 1 tests still pass.)

- [ ] **Step 3: Verify** — conformance still passes + default backend unchanged:

```bash
cd /workspace/MinerU-ROCm
source /root/ocr-eval/omnidocbench-amd-venv/bin/activate
python -m pytest -q                                   # 12 passed (Plan 1 tests unaffected)
python /workspace/omnidocbench-amd/scripts/check_conformance.py .   # CONFORMANT
test -s adapter/qwen2vl_chat_template.jinja && echo "chat template present"
```

- [ ] **Step 4: Commit**

```bash
git add adapter/qwen2vl_chat_template.jinja adapter/adapter_config.py
git commit -m "feat(vlm): vendor Qwen2-VL chat template; env-overridable BACKEND"
```

---

## Task 2: Fill vlm_adapter.py (http-client two-step) + unit test

**Files:**
- Modify (rewrite): `adapter/vlm_adapter.py`
- Create: `tests/test_vlm_adapter.py`

**Interfaces:**
- Consumes: `mineru_vl_utils.MinerUClient` (backend=`http-client`), `two_step_extract`, `json2md` (exact import confirmed in Step 1). `cfg["server_url"]`, `cfg["api_model_name"]`.
- Produces: `vlm_adapter.infer_page(img: Path, platform: str, cfg: dict) -> str` returning R4-conformant Markdown; same signature as `pipeline_adapter.infer_page` (the dispatcher calls `sub.infer_page(i, platform, cfg)`).

- [ ] **Step 1: Confirm the exact mineru-vl-utils API** (the spike used `two_step_extract` + `json2md`; pin the import paths):

```bash
/opt/venv/bin/python - <<'PY'
import mineru_vl_utils, inspect
print("MinerUClient:", hasattr(mineru_vl_utils, "MinerUClient"))
# find json2md
for mod in ("mineru_vl_utils", "mineru_vl_utils.md_renderer", "mineru_vl_utils.utils"):
    try:
        m = __import__(mod, fromlist=["x"])
        if hasattr(m, "json2md"):
            print("json2md at:", mod); break
    except Exception: pass
PY
```
Record the exact `json2md` import path (use it in Step 3). If absent, use whatever the spike's `test_twostep.py` used to render Markdown from the two-step result.

- [ ] **Step 2: Write the failing unit test** — `tests/test_vlm_adapter.py` (CPU; no vllm/mineru-vl-utils):

```python
# tests/test_vlm_adapter.py
from vlm_adapter import normalize_vlm_markdown

def test_strips_md_start_end_markers():
    md = "<|md_start|>\n# Title\n\nbody\n<|md_end|>"
    assert normalize_vlm_markdown(md) == "\n# Title\n\nbody\n"

def test_stips_txt_contd_and_paratext_markers():
    md = "para1<|txt_contd|>para2<|paratext|>"
    out = normalize_vlm_markdown(md)
    assert "<|txt_contd|>" not in out and "<|paratext|>" not in out

def test_keeps_latex_and_html_tables():
    md = "inline $x$ and $$E=mc^2$$\n<table><tr><td>a</td></tr></table>"
    assert normalize_vlm_markdown(md) == md
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd /workspace/MinerU-ROCm && source /root/ocr-eval/omnidocbench-amd-venv/bin/activate && python -m pytest tests/test_vlm_adapter.py -v`
Expected: FAIL — `ImportError: cannot import name 'normalize_vlm_markdown'`.

- [ ] **Step 4: Implement vlm_adapter.py** — replace the stub body:

```python
"""VLM adapter (MinerU2.5-Pro-2605-1.2B). Drives mineru-vl-utils two-step inference
(layout -> per-block extract) against a vLLM-on-ROCm (or transformers) server.

The dispatcher calls infer_page(img, platform, cfg) per page; we lazily create one
MinerUClient (http-client) pointed at the server and reuse it. Per-page failures
propagate to the dispatcher's try/except (R2).
"""
from __future__ import annotations
import re
from pathlib import Path

# Safety markers mineru-vl-utils may emit around/inside the markdown (spike §2):
# <|md_start|>/<|md_end|> wrap the doc; <|txt_contd|>/<|paratext|> are cross-page tokens.
_MARKER_RE = re.compile(r"<\|(?:md_start|md_end|txt_contd|paratext)\|>")


def normalize_vlm_markdown(md: str) -> str:
    """Strip mineru control markers; keep LaTeX ($...$/$$...$$) and HTML tables intact (R4)."""
    return _MARKER_RE.sub("", md)


_runner = None  # lazy singleton


def infer_page(img: Path, platform: str, cfg: dict) -> str:
    """Return Markdown for one page image via the VLM two-step path."""
    global _runner
    if _runner is None:
        _runner = MineruVLRunner(platform=platform, cfg=cfg)
        _runner.load()
    return normalize_vlm_markdown(_runner.extract(img))


class MineruVLRunner:
    """Wraps mineru_vl_utils.MinerUClient(http-client) against a persistent VLM server."""

    def __init__(self, platform: str, cfg: dict):
        self.platform = platform
        self.cfg = cfg
        self.server_url = cfg.get("server_url") or "http://127.0.0.1:8265/v1"
        self.model_name = cfg.get("api_model_name") or "mineru-pro"

    def load(self):
        from mineru_vl_utils import MinerUClient  # confirmed in Task 2 Step 1
        self._client = MinerUClient(
            backend="http-client",
            server_url=self.server_url,
            model_name=self.model_name,
        )

    def extract(self, img: Path) -> str:
        from PIL import Image
        from mineru_vl_utils import json2md  # import path per Task 2 Step 1
        pil = Image.open(img).convert("RGB")
        result = self._client.two_step_extract(pil)
        return json2md(result)
```

- [ ] **Step 5: Run the unit test to verify it passes**

Run: `python -m pytest tests/test_vlm_adapter.py -v`
Expected: PASS (3 passed). Then full suite: `python -m pytest -q` → 15 passed (12 Plan-1 + 3 VLM).

- [ ] **Step 6: Commit**

```bash
git add adapter/vlm_adapter.py tests/test_vlm_adapter.py
git commit -m "feat(vlm): vlm_adapter two-step (http-client) + R4 normalize helper"
```

---

## Task 3: serve_vlm_vllm.sh — launch the vLLM-on-ROCm server

**Files:**
- Create: `examples/serve_vlm_vllm.sh`

**Interfaces:**
- Produces: a background vLLM OpenAI server on `127.0.0.1:8265` serving `mineru-pro`, with `MinerULogitsProcessor` registered. `vlm_adapter` queries `http://127.0.0.1:8265/v1`.

- [ ] **Step 1: Write the launch script** — `examples/serve_vlm_vllm.sh` (flags verbatim from the spike §1):

```bash
#!/usr/bin/env bash
# Launch the MinerU2.5-Pro vLLM-on-ROCm server (GPU 0). Background; poll /v1/models.
set -euo pipefail
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export HSA_OVERRIDE_GFX_VERSION=11.0.0          # gfx1100 / W7900 RDNA3
export VLLM_USE_V1=1                             # MANDATORY: v1 logits-processor API
export HF_ENDPOINT="${HF_ENDPOINT:-http://134.199.133.77}"
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"

MODEL_DIR="$(ls -d /root/.cache/huggingface/models--opendatalab--MinerU2.5-Pro-2605-1.2B/snapshots/* | head -1)"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-8265}"
LOG="${LOG:-/tmp/vlm-vllm.log}"

nohup /opt/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name mineru-pro \
  --trust-remote-code \
  --dtype bfloat16 \
  --chat-template "$REPO/adapter/qwen2vl_chat_template.jinja" \
  --logits-processors mineru_vl_utils:MinerULogitsProcessor \
  --host 127.0.0.1 --port "$PORT" \
  --gpu-memory-utilization 0.70 \
  --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 1}' \
  --enforce-eager \
  > "$LOG" 2>&1 &
echo $! > /tmp/vlm-vllm.pid
echo "[serve_vlm_vllm] launched pid $(cat /tmp/vlm-vllm.pid), log $LOG"
```

- [ ] **Step 2: Add a health-check helper** (append to the same script or a sibling `examples/wait_vlm.sh`):

```bash
# examples/wait_vlm.sh — block until /v1/models responds, up to ~5 min.
PORT="${PORT:-8265}"
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[wait_vlm] server ready: $(curl -sf http://127.0.0.1:$PORT/v1/models)"
    exit 0
  fi
  sleep 5
done
echo "[wait_vlm] server did NOT become ready; tail of log:"; tail -20 /tmp/vlm-vllm.log; exit 1
```

- [ ] **Step 3: Verify the server boots** — run the launch script, wait, confirm:

```bash
cd /workspace/MinerU-ROCm
chmod +x examples/serve_vlm_vllm.sh examples/wait_vlm.sh
bash examples/serve_vlm_vllm.sh
bash examples/wait_vlm.sh          # expect: server ready: {"data":[{"id":"mineru-pro",...}]}
grep -i "logits_processors" /tmp/vlm-vllm.log | head -1   # confirm MinerULogitsProcessor wired
```
Expected: `/v1/models` returns `mineru-pro`; the log's `non-default args` includes `'logits_processors': ['mineru_vl_utils:MinerULogitsProcessor']`.

- [ ] **Step 4: Stop the server** (free GPU 0 until the smoke task): `kill $(cat /tmp/vlm-vllm.pid); pkill -f vllm.entrypoints || true`.

- [ ] **Step 5: Commit**

```bash
git add examples/serve_vlm_vllm.sh examples/wait_vlm.sh
git commit -m "feat(vlm): serve_vlm_vllm.sh — vLLM-on-ROCm server with MinerULogitsProcessor"
```

---

## Task 4: Provision the VLM adapter env (omnidocbench-amd into /opt/venv)

**Files:**
- Modify: `adapter/setup/00-install-deps.sh` (append a VLM section — optional; the env is already provisioned by the spike).

**Interfaces:**
- Produces: `/opt/venv` can `import vllm, mineru_vl_utils, omnidocbench_amd` (so the engine-invoked adapter subprocess resolves both `mineru_vl_utils` and `omnidocbench_amd.types`).

- [ ] **Step 1: Add omnidocbench-amd to /opt/venv** (the spike already put vllm + mineru-vl-utils there):

```bash
/opt/venv/bin/pip install -e /workspace/omnidocbench-amd
```

- [ ] **Step 2: Verify all three import**:

```bash
/opt/venv/bin/python -c "import vllm, mineru_vl_utils, omnidocbench_amd; print('vlm env ok', vllm.__version__)"
```
Expected: `vlm env ok 0.16.1.dev0+...`.

- [ ] **Step 3: Document in `adapter/setup/00-install-deps.sh`** — append a comment block noting the VLM env is `/opt/venv` (vllm + mineru-vl-utils + `pip install -e /workspace/omnidocbench-amd`), GPU 0, and that scoring uses `/root/ocr-eval/OmniDocBench/.venv` (Py3.11). No functional change.

- [ ] **Step 4: Commit**

```bash
git add adapter/setup/00-install-deps.sh
git commit -m "docs(setup): note VLM env (/opt/venv) + scoring venv split"
```

---

## Task 5: Smoke — adapter → server → sane Markdown on real pages

**Files:**
- Create: `examples/run_vlm_demo.sh` (start server + run adapter on a few real OmniDocBench pages)

**Interfaces:**
- Consumes: Tasks 1–4 (chat template, vlm_adapter, serve script, env).

- [ ] **Step 1: Write the demo** — `examples/run_vlm_demo.sh`:

```bash
#!/usr/bin/env bash
# End-to-end VLM smoke on a few real OmniDocBench pages (GPU 0).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# 1) start the server (idempotent: skip if already up)
if ! curl -sf http://127.0.0.1:8265/v1/models >/dev/null 2>&1; then
  bash "$REPO/examples/serve_vlm_vllm.sh"
  bash "$REPO/examples/wait_vlm.sh"
fi
# 2) a tiny img-dir of 3 real pages
SMALL=/tmp/vlm-smoke-imgs; rm -rf "$SMALL"; mkdir -p "$SMALL"
ls /root/ocr-eval/OmniDocBench_data/images/ | grep -E 'PPT|exam_paper|color_textbook' | head -3 \
  | while read f; do ln -sf "/root/ocr-eval/OmniDocBench_data/images/$f" "$SMALL/$f"; done
# 3) run the adapter (VLM env; MINERU_ROCM_BACKEND=vlm-vllm; server_url default)
export MINERU_ROCM_BACKEND=vlm-vllm
OUT=/tmp/vlm-smoke-out; rm -rf "$OUT"
/opt/venv/bin/python "$REPO/adapter/run_adapter.py" \
  --img-dir "$SMALL" --out-dir "$OUT" --platform linux-rocm \
  --server-url http://127.0.0.1:8265/v1 --api-model-name mineru-pro
echo "--- smoke outputs ---"; for f in "$OUT"/*.md; do echo "== $f =="; head -8 "$f"; done
cat "$OUT/_run_stats.json" | python -c "import json,sys;d=json.load(sys.stdin);print('engine',d['engine'],'ok',d['ok'],'fail',d['fail'])"
```

- [ ] **Step 2: Run the smoke on GPU 0**

```bash
cd /workspace/MinerU-ROCm
chmod +x examples/run_vlm_demo.sh
bash examples/run_vlm_demo.sh
```
Expected: 3 `.md` files with **real** parsed content (text + at least one `$...$` formula or `<table>` for a table page); `_run_stats.json` shows `engine=vlm-vllm, ok=3, fail=0`. (First page loads the client; each page is N+1 forwards — expect ~10–30 s/page.)

- [ ] **Step 3: Commit**

```bash
git add examples/run_vlm_demo.sh
git commit -m "feat(vlm): end-to-end smoke (server + adapter on real pages)"
```

- [ ] **Step 4: Stop the server** until the full eval: `pkill -f vllm.entrypoints || true`.

---

## Task 6: Full OmniDocBench v1.6 eval (vlm-vllm) → 95.75

**Files:**
- Create: `eval/configs/omnidocbench_v16_vlm.yaml` (copy of the pipeline scoring config, prediction dir = the VLM preds)
- Output: `results/omnidocbench/v16/linux-rocm/` (VLM artifacts)

- [ ] **Step 1: Start the server + confirm ready** (GPU 0):

```bash
cd /workspace/MinerU-ROCm
bash examples/serve_vlm_vllm.sh && bash examples/wait_vlm.sh
```

- [ ] **Step 2: Run `infer`** (VLM env, GPU 0, full 1651-page set via the clean 1651 dir):

```bash
export HIP_VISIBLE_DEVICES=0 MINERU_ROCM_BACKEND=vlm-vllm
PRED=/root/ocr-eval/mineru-vlm-preds
/opt/venv/bin/omnidocbench-amd infer \
  --adapter /workspace/MinerU-ROCm/adapter/run_adapter.py \
  --img-dir  /root/ocr-eval/OmniDocBench_v16_images \
  --out-dir  "$PRED" \
  --platform linux-rocm
```
Expected: `_run_stats.json` with `limit_pages=null, ok≈1651, fail≈0, engine=vlm-vllm`. (Long run — the VLM is ~7 s/TTFT × N blocks/page; expect a few hours. That's fine.)

- [ ] **Step 3: Score** (OmniDocBench venv, via pdf_validation directly — engine score stub):

```bash
# scoring config: copy the pipeline yaml, point prediction at the VLM preds
sed 's#/root/ocr-eval/mineru-pipeline-preds#/root/ocr-eval/mineru-vlm-preds#' \
  /workspace/MinerU-ROCm/configs/mineru_pipeline_full.yaml \
  > /workspace/MinerU-ROCm/configs/mineru_vlm_full.yaml
source /root/ocr-eval/OmniDocBench/.venv/bin/activate
cd /root/ocr-eval/OmniDocBench
python pdf_validation.py --config /workspace/MinerU-ROCm/configs/mineru_vlm_full.yaml 2>&1 | tee /tmp/vlm-score.log
```
Expected: a `metric_result.json` with Text EditDist / TEDS / CDM / read-order.

- [ ] **Step 4: Gate** — compute `Overall = ((1−Text_EditDist)×100 + Table_TEDS + Formula_CDM)/3`.
**PASS if ≥ 95.25** (within 0.5 pp of 95.75). If below, investigate (CDM provisioning? table OTSL→HTML? a page-set mismatch?) before publishing.

- [ ] **Step 5: Publish + commit** (engine publish assembles run_summary + provenance; commit the bundle, heavy preds off-repo):

```bash
cd /workspace/MinerU-ROCm
git add configs/mineru_vlm_full.yaml results/omnidocbench/v16/linux-rocm/ docs/reproducibility.md
git commit -m "eval(vlm): OmniDocBench v1.6 full-set on ROCm gfx1100 (Overall ≈95.75, vlm-vllm)"
```
Record the real Overall + submetrics + the run command in `docs/reproducibility.md`.

---

## Task 7: Secondary backend — vlm-transformers + precision comparison

**Files:**
- Modify: `adapter/vlm_adapter.py` (add a `vlm-transformers` branch — `MinerUClient(backend="transformers")` with `no_repeat_ngram_size=100`).

**Interfaces:**
- Produces: a second result set `results/.../linux-rocm/<vlm-transformers>/` for the comparison table.

- [ ] **Step 1: Confirm `MinerULogitsProcessor` ≡ no-repeat-100-gram maps to HF `no_repeat_ngram_size=100`** (spike §2: the processor is `VllmV1NoRepeatNGramLogitsProcessor`, no-repeat-100-gram). Inspect mineru-vl-utils' transformers backend to see how generation params are passed:

```bash
grep -rn "no_repeat_ngram\|generate\|backend.*transformers" /opt/venv/lib/python*/site-packages/mineru_vl_utils/ | head
```

- [ ] **Step 2: Add the transformers branch to `MineruVLRunner.load()`** — branch on `cfg["backend"]`:

```python
    def load(self):
        from mineru_vl_utils import MinerUClient
        if self.cfg.get("backend") == "vlm-transformers":
            # in-process transformers engine; MinerULogitsProcessor == no-repeat-100-gram
            # -> HF generate's no_repeat_ngram_size (verify the kwarg name mineru-vl-utils accepts).
            self._client = MinerUClient(
                backend="transformers",
                model_name=self.model_name,            # HF model id / local path
                no_repeat_ngram_size=100,              # == MinerULogitsProcessor
            )
        else:  # vlm-vllm
            self._client = MinerUClient(
                backend="http-client",
                server_url=self.server_url,
                model_name=self.model_name,
            )
```
If mineru-vl-utils' transformers backend does not accept `no_repeat_ngram_size`, pass it via the generate kwargs it does expose (or monkeypatch the generate call) — confirm in Step 1.

- [ ] **Step 3: Smoke + full eval** (transformers, GPU 0 — no separate server):

```bash
export HIP_VISIBLE_DEVICES=0 MINERU_ROCM_BACKEND=vlm-transformers
PRED=/root/ocr-eval/mineru-vlm-transformers-preds
/opt/venv/bin/omnidocbench-amd infer --adapter adapter/run_adapter.py \
  --img-dir /root/ocr-eval/OmniDocBench_v16_images --out-dir "$PRED" --platform linux-rocm
# score (same as Task 6, prediction dir = $PRED)
```

- [ ] **Step 4: Compare** — record both Overalls + the delta (vLLM vs transformers) in `docs/reproducibility.md`. If transformers diverges from vLLM by >0.5 pp, the logits-processor mapping is likely the cause — note it. The winner (expected: vLLM) becomes the primary `model_card.json`; the other is a reported backend.

- [ ] **Step 5: Commit**

```bash
git add adapter/vlm_adapter.py results/omnidocbench/v16/linux-rocm/ docs/reproducibility.md
git commit -m "feat(vlm): vlm-transformers secondary backend + precision comparison"
```

---

## Task 8: Finalize — model_card, README comparison table, registry, badges

**Files:**
- Modify: `model_card.json`, `README.md`, `README.zh-CN.md`, `docs/how-it-works.md`

- [ ] **Step 1: Fill `model_card.json`** (VLM, primary) with the Task 6 result: `overall` (the vLLM Overall), `submetrics` (text/CDM/TEDS/read-order from `metric_result.json`), `eval_date`, `hardware` (`AMD gfx1100 / Radeon PRO W7900`, 48 GB, ROCm 7.2.1), `badge.linux-rocm: "community"`, `model_version: "2605"`. Validate against `/workspace/omnidocbench-amd/contracts/artifact-schema.json`.

- [ ] **Step 2: Add the VLM row to both READMEs' comparison table** (in `## Evaluation`):

  | Backend | Overall ↑ | Text Edit ↓ | Formula CDM ↑ | Table TEDS ↑ |
  |---|---:|---:|---:|---:|
  | _official_ MinerU2.5-Pro | 95.75 | 0.036 | 97.45 | 93.42 |
  | ours vlm-vllm (ROCm) | _<Overall>_ | _<text>_ | _<CDM>_ | _<TEDS>_ |
  | ours vlm-transformers (ROCm) | _<Overall>_ | | | |
  | ours pipeline (ROCm) | 86.48 | 0.0566 | 83.07 | 82.04 |

  Pull the exact numbers from the committed `metric_result.json` (do not round wrongly).

- [ ] **Step 3: Update `docs/how-it-works.md`** — note the `mineru2.5` registry row is now filled (VLM Overall populated; linux-rocm badge → `community`); pipeline stays the secondary card.

- [ ] **Step 4: Conformance + tests**

```bash
cd /workspace/MinerU-ROCm
source /root/ocr-eval/omnidocbench-amd-venv/bin/activate
python /workspace/omnidocbench-amd/scripts/check_conformance.py .   # CONFORMANT
python -m pytest -q                                                  # 15 passed
```

- [ ] **Step 5: Commit**

```bash
git add model_card.json README.md README.zh-CN.md docs/how-it-works.md
git commit -m "docs(vlm): finalize model_card, README comparison table, registry note"
```

---

## Self-Review

**1. Spec coverage (Plan 2 = VLM, spec §8):**
- VLM adapter two-step + MinerULogitsProcessor: Tasks 2, 3, 5. ✓
- Chat template vendoring (spike gotcha): Task 1. ✓
- vLLM-on-ROCm primary backend: Tasks 3–6. ✓
- transformers secondary backend + comparison: Task 7. ✓
- Backend matrix (vlm-vllm / vlm-transformers rows): Tasks 2, 7. ✓
- Eval/precision protocol (gate ≤0.5 pp, both engines reported): Tasks 6, 7. ✓
- Venv/storage isolation (VLM env /opt/venv, scoring venv split): Task 4 + Global Constraints. ✓
- Error handling R2 (dispatcher, from Plan 1): unchanged; `infer_page` may raise. ✓
- Testing (unit + smoke + eval gate): Tasks 2, 5, 6. ✓
- Finalize (model_card, README, registry, badges): Task 8. ✓

**2. Placeholder scan:** The exact `json2md` import path in mineru-vl-utils is confirmed in Task 2 Step 1 before use (not a placeholder — a concrete verification step, like Plan 1's Task 4 spike). The transformers `no_repeat_ngram_size` kwarg is verified in Task 7 Step 1. Every code step has complete code; every command is exact.

**3. Type consistency:** `infer_page(img: Path, platform: str, cfg: dict) -> str` matches the dispatcher's call site (Plan 1 `run_adapter.py`). `cfg["backend"]` / `cfg["server_url"]` / `cfg["api_model_name"]` match `adapter_config.as_dict()` + the CLI flags. `MINERU_ROCM_BACKEND` is read in `adapter_config` (Task 1) and set by the eval (Tasks 5–7).

---

## Execution

Plan 2 produces a working VLM adapter + the 95.75 result + a two-engine comparison. Tasks 1–6 are the critical path (vLLM → 95.75); Task 7 (transformers) is the secondary comparison; Task 8 finalizes. Run via subagent-driven-development (recommended) or executing-plans.
