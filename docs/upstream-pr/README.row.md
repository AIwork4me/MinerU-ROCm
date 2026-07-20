<!-- In opendatalab/MinerU README.md "Local Deployment" table, replace ONLY the
     "GPU Acceleration" row's first cell. Do NOT touch the Accuracy row. -->

Before:
| GPU Acceleration | Volta and later architecture GPUs or Apple Silicon | … |

After:
| GPU Acceleration | Volta+ / Apple Silicon / AMD ROCm (gfx1100/RDNA3; see [AMD guide](usage/acceleration_cards/AMD.md))¹ | … |

Footnote (add near the table footnotes):
¹ VLM/vLLM path requires `HSA_OVERRIDE_GFX_VERSION=11.0.0` on gfx1100 (only gfx1100 was tested); the pipeline backend does not. Community-verified (see AMD guide).
