"""VLM adapter (MinerU2.5-Pro-2605-1.2B). Drives mineru-vl-utils two-step inference
(layout -> per-block extract) against a vLLM-on-ROCm server OR an in-process
transformers engine.

The dispatcher calls infer_page(img, platform, cfg) per page; we lazily create one
MinerUClient and reuse it. Per-page failures propagate to the dispatcher's
try/except (R2).

Two backends, same two-step (MinerUClient.two_step_extract → json2md):
  * vlm-vllm         → MinerUClient(backend="http-client", server_url, model_name)
  * vlm-transformers → MinerUClient(backend="transformers", model_path=<snapshot|HF id>)

no-repeat-100-gram: the vLLM path uses MinerULogitsProcessor (server-side); the
transformers path gets it for free via the default DEFAULT_SAMPLING_PARAMS, whose
MinerUSamplingParams(no_repeat_ngram_size=100, ...) the transformers backend maps
to HF generate(no_repeat_ngram_size=100) (verified Task 6 Step 1 — see load()).
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
    """Wraps mineru_vl_utils.MinerUClient against either a vLLM server (http-client)
    or an in-process transformers engine. Backend selected by cfg["backend"]."""

    # Local snapshot of opendatalab/MinerU2.5-Pro-2605-1.2B (HF_HOME uses the flat
    # cache layout, not hub/). This is the transformers model_path default; the HF
    # repo id also works (slower first call). Override via cfg["vlm_model_path"].
    _DEFAULT_VLM_MODEL_PATH = (
        "/root/.cache/huggingface/models--opendatalab--MinerU2.5-Pro-2605-1.2B/"
        "snapshots/bff20d4ae2bf202df9f45284b4d43681555a97ed"
    )
    _DEFAULT_VLM_MODEL_ID = "opendatalab/MinerU2.5-Pro-2605-1.2B"

    def __init__(self, platform: str, cfg: dict):
        self.platform = platform
        self.cfg = cfg
        # cfg["backend"] ∈ {"vlm-vllm", "vlm-transformers"} (dispatcher routes both
        # here). Default to vlm-vllm for back-compat if invoked directly.
        self.backend = cfg.get("backend", "vlm-vllm")
        self.server_url = cfg.get("server_url") or "http://127.0.0.1:8265/v1"
        # http-client: served model name (NOT the HF id). Default "mineru-pro" is the
        # serve_vlm_vllm.sh --served-model-name.
        self.model_name = cfg.get("api_model_name") or "mineru-pro"

    def _resolve_transformers_model_path(self) -> str:
        """Pick the transformers model_path: cfg override > local snapshot (if
        present) > HF repo id (will download if offline cache missing)."""
        from pathlib import Path
        override = self.cfg.get("vlm_model_path")
        if override:
            return override
        snap = Path(self._DEFAULT_VLM_MODEL_PATH)
        if (snap / "model.safetensors").is_file():
            return str(snap)
        return self._DEFAULT_VLM_MODEL_ID

    def load(self):
        # Lazy: mineru_vl_utils + transformers only import when this backend runs
        # (keeps the unit-test/CI env — which has neither installed — clean).
        from mineru_vl_utils import MinerUClient  # noqa: F401 (re-exported symbol)

        if self.backend == "vlm-transformers":
            # transformers path loads Qwen2VLForConditionalGeneration in-process on
            # the ROCm GPU. The default DEFAULT_SAMPLING_PARAMS carries
            # MinerUSamplingParams(no_repeat_ngram_size=100), which the transformers
            # backend maps to HF generate(no_repeat_ngram_size=100)
            # (mineru_vl_utils/vlm_client/transformers_client.py build_generate_kwargs).
            #
            # The MinerU2.5-Pro snapshot ships NO chat_template in its processor/
            # tokenizer config (verified: tokenizer_config.json chat_template count = 0),
            # so apply_chat_template raises. We load the processor ourselves and inject
            # the vendored Qwen2-VL chat template (Task 1's qwen2vl_chat_template.jinja)
            # before handing the processor to MinerUClient (which then loads only the
            # model). Model loading is left to MinerUClient (device_map="auto" → GPU).
            model_path = self._resolve_transformers_model_path()
            processor = self._load_processor_with_template(model_path)
            self._client = MinerUClient(
                backend="transformers",
                model_path=model_path,
                processor=processor,
            )
        else:  # vlm-vllm → http-client against a persistent vLLM server
            self._client = MinerUClient(
                backend="http-client",
                server_url=self.server_url,
                model_name=self.model_name,
            )

    @staticmethod
    def _load_processor_with_template(model_path: str):
        """AutoProcessor.from_pretrained + inject the vendored Qwen2-VL chat
        template (the snapshot ships none). Returns the processor; raises a clear
        error if the vendored template file is missing."""
        from pathlib import Path
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        template_path = Path(__file__).resolve().parent / "qwen2vl_chat_template.jinja"
        if not template_path.is_file():
            raise FileNotFoundError(
                f"vlm-transformers needs the vendored Qwen2-VL chat template at "
                f"{template_path} (absent)."
            )
        processor.chat_template = template_path.read_text(encoding="utf-8")
        return processor

    def extract(self, img: Path) -> str:
        from PIL import Image
        # json2md lives in the post_process submodule, NOT at mineru_vl_utils top-level
        # (verified Task 2 Step 1: top-level __all__ = MinerUClient, MinerUSamplingParams,
        #  MinerULogitsProcessor, __version__; the spike harness test_twostep.py uses
        #  `from mineru_vl_utils.post_process.json2markdown import json2md`).
        from mineru_vl_utils.post_process.json2markdown import json2md
        pil = Image.open(img).convert("RGB")
        result = self._client.two_step_extract(pil)
        return json2md(result)
