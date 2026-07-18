# MIGraphX rejects valid ONNX `Loop` subgraph whose body-input names overlap the parent graph (Paddle2ONNX GRU export)

> **Filed (2026-07-18, account AIwork4me):** the crash is in MIGraphX's own source
> (`src/AMDMIGraphX/src/onnx/onnx_parser.cpp:498`), invoked via the ONNX Runtime MIGraphX EP.
> - **Variant A (primary, parser fix) → ROCm/AMDMIGraphX #5078**: https://github.com/ROCm/AMDMIGraphX/issues/5078
> - **Variant B (secondary, ORT graceful-fallback ask) → microsoft/onnxruntime #29778**: https://github.com/microsoft/onnxruntime/issues/29778
>
> The two are cross-linked (B's body cites A as root cause; A's References cite B).
> (Note: the repo is `ROCm/AMDMIGraphX`, not `ROCm/MIGraphX` — it was renamed.)

---

## Variant A — for https://github.com/ROCm/MIGraphX/issues (PRIMARY)

### Title
`[onnx_parser] check_sorted rejects a valid ONNX Loop subgraph whose body-input names overlap parent-graph names (Paddle2ONNX GRU)`

### Summary
MIGraphX fails to compile a **valid** ONNX model that contains a `Loop` (control-flow) op
whose body subgraph declares input parameters whose names also appear in the enclosing
graph. This name overlap is permitted by the ONNX specification — subgraph inputs are
independently scoped — but `check_sorted` in
`src/AMDMIGraphX/src/onnx/onnx_parser.cpp:498` treats it as a conflict and aborts parsing:

```
check_sorted: subgraph "PaddlePaddle Graph 1" has parameter name "gru_cell_0.w_1" existing in parent graph!
```

The pattern is produced routinely by **Paddle2ONNX when exporting GRU / recurrent layers**:
weight names such as `gru_cell_0.w_1` are emitted once in the parent graph and passed into
the `Loop` body, so the same name legitimately appears in both scopes. As a result MIGraphX
cannot run a whole class of Paddle-derived models (table/OCR/recurrent).

### Environment
| Component | Version |
|---|---|
| MIGraphX | `2.15.0.20250912-17-200-gde19b73ad` (`migraphx-driver --version`) |
| ROCm | 7.2.1 |
| GPU | AMD gfx1100 (Radeon PRO W7900, RDNA3) |
| ONNX Runtime | 1.23.2, via the MIGraphX EP (AMD wheel `onnxruntime-migraphx`, `repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/`) |
| Model | `slanet-plus.onnx` (SLANet-Plus wireless table-structure recognition); ONNX **opset 14**, **IR version 7**, exported by **Paddle2ONNX** |

The model is public: [`opendatalab/PDF-Extract-Kit-1.0`](https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0)
→ `models/TabRec/SlanetPlus/slanet-plus.onnx` (~7.4 MB).

### Reproducer
Download the model:
```bash
huggingface-cli download opendatalab/PDF-Extract-Kit-1.0 \
  models/TabRec/SlanetPlus/slanet-plus.onnx --local-dir .
```
Run (the input shape is not load-bearing — parsing fails before input validation):
```python
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession(
    "models/TabRec/SlanetPlus/slanet-plus.onnx",
    providers=["MIGraphXExecutionProvider", "CPUExecutionProvider"],
)
x = np.random.randn(1, 3, 488, 488).astype(np.float32)
out = sess.run(None, {sess.get_inputs()[0].name: x})   # raises RuntimeException
```

### Expected behavior
MIGraphX compiles and runs the model. The model is valid ONNX:
- `onnx.checker.check_model(model)` **passes**.
- The same model runs correctly under the CPU EP —
  `providers=["CPUExecutionProvider"]` returns outputs with shapes `(1, 10, 8)` and `(1, 10, 50)`.

### Actual behavior
`RuntimeException` raised from `session.run()`:
```
migraphx_parse_onnx_buffer: Error: /.../src/AMDMIGraphX/src/onnx/onnx_parser.cpp:498:
  check_sorted: subgraph "PaddlePaddle Graph 1" has parameter name "gru_cell_0.w_1"
  existing in parent graph!
[E:onnxruntime:, sequential_executor.cc:572 ExecuteKernel] Non-zero status code returned
  while running MGXKernel_graph_PaddlePaddle Graph 0_8504660414787792506_0 node.
  Name:'MIGraphXExecutionProvider_MGXKernel_graph_PaddlePaddle Graph 0_8504660414787792506_0_0'
  Status Message: Failed to call function
onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException: [ONNXRuntimeError] : 6 :
  RUNTIME_EXCEPTION ... Status Message: Failed to call function
```

### Root-cause analysis
- The model contains exactly **one** ONNX `Loop` op (opset 14); its body subgraph is named
  `"PaddlePaddle Graph 1"` (Paddle2ONNX export of a GRU). The body declares **32** inputs.
- **16 of those 32 body inputs share their names with node outputs in the parent graph** —
  e.g. `gru_cell_0.w_1`, `gru_cell_0.w_0`, `gru_cell_0.b_0`, `gru_cell_0.b_1`,
  `linear_*.w_0`, `linear_*.b_0`. These are the GRU/linear weight names, emitted once in the
  parent graph and passed into the `Loop` body.
- **This overlap is legal ONNX.** A subgraph's (Loop body's) inputs are independently scoped;
  their names are local to the body and need not be unique with respect to the enclosing
  graph — outer values are bound to body inputs **by position** at the `Loop` node, not by
  name uniqueness. Concrete evidence that the model is not malformed:
  - `onnx.checker.check_model` → **PASS**.
  - The CPU EP runs it and produces correct output shapes.
- `check_sorted` (`onnx_parser.cpp:498`) nevertheless walks the subgraph's parameter names
  and rejects any that also occur in the parent graph — an invariant **not required by the
  ONNX spec** — so it rejects a valid model.
- **The control-flow subgraph is the trigger.** Two sibling table models from the same
  package — `unet.onnx` (UNet table line recovery) and `PP-LCNet_x1_0_table_cls.onnx`
  (table wired/wireless classifier) — contain **no** `Loop` and compile/run on MIGraphX with
  **100% node coverage and bit-exact outputs vs CPU** (UNet ~20× faster than CPU, PP-LCNet
  ~12× faster). Only the `Loop`-containing `slanet-plus.onnx` fails.

### Suggested fix
Honor ONNX subgraph scoping in the parser: a `Loop`/`If`/`Scan` body-input name that
coincides with an enclosing-graph name is **not** a conflict and must not abort parsing
(`check_sorted` / the subgraph parameter handling in `onnx_parser.cpp`). If some internal
MIGraphX representation genuinely cannot tolerate the overlap, rename-on-import rather than
rejecting the model.

### Workaround
Force the affected model onto the CPU EP: `providers=["CPUExecutionProvider"]` (correct;
forfeits the GPU speedup). Models without control-flow ops are unaffected.

### References
- Model: https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/tree/main/models/TabRec/SlanetPlus
- Related ORT-side report (graceful-fallback ask): https://github.com/microsoft/onnxruntime/issues/29778

---

## Variant B — for https://github.com/microsoft/onnxruntime/issues (SECONDARY)

### Title
`[MIGraphX EP] Hard RuntimeException (no CPU fallback) when MIGraphX fails to compile an assigned subgraph`

### Summary
When the MIGraphX execution provider accepts a subgraph (its capability check passes) but
then **fails to compile it at run time** — here, MIGraphX's ONNX parser rejects a valid
`Loop` subgraph (upstream root cause: https://github.com/ROCm/AMDMIGraphX/issues/5078) — ONNX Runtime propagates a
hard `RuntimeException` out of `session.run()` instead of falling back to the CPU EP for the
failing subgraph, even though `CPUExecutionProvider` is listed as a fallback in the session
providers. A graceful fallback would let valid models complete (on CPU for the affected
subgraph) instead of aborting the whole run.

### Reproducer
Identical to the upstream MIGraphX issue (Variant A): `slanet-plus.onnx` with
`providers=["MIGraphXExecutionProvider", "CPUExecutionProvider"]`; `session.run(...)` raises
`RuntimeException` with `Status Message: Failed to call function` originating from
`migraphx_parse_onnx_buffer` (`onnx_parser.cpp:498`). The model passes `onnx.checker` and
runs correctly under `providers=["CPUExecutionProvider"]`.

### Ask (ORT side)
When a provider's compile/parse step fails for a subgraph it had accepted, fall back to the
next provider in the session list (here, CPU) for that subgraph — or at least for the nodes
the provider could not compile — so the run can complete, rather than throwing a hard
`RuntimeException`. (Cross-linking the root cause: https://github.com/ROCm/AMDMIGraphX/issues/5078.)

### Note
This is a resilience enhancement on the ORT side. The underlying parser defect is MIGraphX's
and is tracked in Variant A.
