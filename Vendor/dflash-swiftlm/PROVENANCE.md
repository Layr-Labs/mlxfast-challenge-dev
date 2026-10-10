# Vendored: SwiftLM DFlash library

- Source: https://github.com/SharpAI/SwiftLM  (`Sources/DFlash/`)
- License: MIT — see `LICENSE.SwiftLM` (Copyright 2026 SwiftLM Contributors)
- Technique: DFlash speculative decoding (arXiv:2602.06036, z-lab/dflash)
- Vendored for: Qwen3.6-27B (qwen3_5 arch) draft-model speculative decode with
  GatedDeltaNet recurrent-cache rollback (RecurrentRollbackCache).
- Draft weights: z-lab/Qwen3.6-27B-DFlash (MIT).
- Integration TODO: (1) add SwiftPM target; (2) graft `recordTape(...)` call into
  the Qwen35 GatedDeltaNet forward; (3) wire Metal kernels into tools/build-mlx-metallib.sh.
