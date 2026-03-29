# Change 2: QKV Movement-First Bake-In

Code change:

- `qwen_with_nki.py:371` sets `self.neuron_config.fused_qkv = True`
- `qwen_with_nki.py:372` sets `self.neuron_config.qkv_kernel_enabled = True`
- `qwen_with_nki.py:373` sets `self.neuron_config.qkv_nki_kernel_enabled = True`
- `qwen_with_nki.py:374-376` keeps attention TKG kernels disabled so the first attention-side change stays scoped to QKV movement

Why this change was made:

- After Change 1, the next profiling pass no longer showed named MoE selective-loading hotspots in token generation.
- The remaining named compile hotspots were attention/QKV movement oriented:
  - `prep_qkv_tensors`
  - `move_heads_front`
  - `GroupQueryAttention_O`
- Hardware profiling still showed token generation dominating model-attributed device execution, so the next safe one-area change was the fused QKV path.

What changed in the graph:

- The submission path now converts the checkpoint to fused QKV weights and enables the QKV NKI projection path during compile.
- Activation proof is in `analysis/stage2_qkv_candidate/compile_prompt0.log`, which includes:
  - `nki_qkv_projection_isa_kernel`
  - `nki_qkv_projection_cte_impl`
  - `_sharded_qkv_cte_kernel`

Observed result:

- Prompt 0 improved relative to the MoE-only candidate:
  - `e2e_model.latency_ms_p99`: `14932.06 -> 14643.20`
  - `context_encoding_model.latency_ms_avg`: `410.69 -> 410.20`
  - `token_generation_model.latency_ms_avg`: `11.45 -> 10.99`
  - `score`: `0.69516 -> 0.72733`

Key retained artifacts:

- `analysis/stage2_next_bottleneck/summary.md`
- `analysis/stage2_next_bottleneck/compile/compile_prompt0.log`
- `analysis/stage2_next_bottleneck/prompt0/report.json`
- `analysis/stage2_next_bottleneck/prompt4/report.json`
- `analysis/stage2_qkv_candidate/summary.md`
- `analysis/stage2_qkv_candidate/compile_prompt0.log`
- `analysis/stage2_qkv_candidate/prompt0_report.json`

Conclusion:

- Change 2 is the first movement-first attention-side optimization.
- It is a scoped submission-style bake-in that improves the prompt 0 token-generation path and gives the next concrete basis for broader validation on longer prompts.
