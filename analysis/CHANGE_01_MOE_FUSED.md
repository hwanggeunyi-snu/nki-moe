# Change 1: MoE Fused TKG Bake-In

Code change:

- `qwen_with_nki.py:370` sets `self.neuron_config.moe_fused_nki_kernel_enabled = True`
- `qwen_with_nki.py:388-391` aligns `router_config.dtype` with the model dtype when the fused MoE path is enabled
- `qwen_with_nki.py:140` keeps `variance_epsilon` on `NKIRMSNorm` for fused-path compatibility

Why this change was made:

- Baseline token-generation compile profiling showed repeated named DMA hotspots on the MoE selective-loading path.
- The repeated names were `forward_selective_loading`, `ExpertFusedColumnParallelLinear`, and `aten.index_gather`.
- That made MoE selective-loading the first submission-safe removable hotspot candidate.

What changed in the graph:

- Token generation switched from the flat MoE path to the fused MoE token-generation kernel path when `seq_len == 1`.
- The activation proof is in `analysis/deep_profile_candidate_v2/compile_and_benchmark.log`, which includes:
  - `Running MoE Fused NKI kernel`
  - `_moe_token_gen_selective_load_kernel_nki_call`

Observed result:

- The named MoE selective-loading hotspots disappeared from token-generation compile output.
- The remaining visible named hotspots shifted toward attention/QKV movement.
- Representative results from `analysis/deep_profile_candidate_v2`:
  - prompt 0: `token_generation_model.latency_ms_avg = 11.47`, mixed overall result
  - prompt 4: `token_generation_model.latency_ms_avg = 11.42`, mixed overall result

Key retained artifacts:

- `analysis/deep_profile_candidate_v2/summary.md`
- `analysis/deep_profile_candidate_v2/compile_and_benchmark.log`
- `analysis/deep_profile_candidate_v2/prompt0/report.json`
- `analysis/deep_profile_candidate_v2/prompt4/report.json`

Conclusion:

- Change 1 was useful as a hotspot-removal step.
- It proved MoE selective-loading was removable, but it did not produce a clean token-generation win on its own.
- That is why the next profiling step moved to attention/QKV movement.
