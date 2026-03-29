# Trn2 MoE Fused One-Kernel Result

## Decision

- Baseline token-generation compile profiling pointed to MoE selective-loading, not attention.
- In `token_generation_model/_tp0_bk0/log-neuron-cc.txt`, the named DMAProfiler hotspots were repeated `forward_selective_loading -> ExpertFusedColumnParallelLinear -> aten.index_gather` dynamic loads.
- Aggregated named hotspot sums from the baseline compile log:
  - `token_generation`: `moe=4.024`, `attn=0.000`
  - `context_encoding`: `moe=32.931`, `attn=2.537`
- Based on that, the first submission-style one-kernel change was to bake in the fused MoE token-generation kernel in [qwen_with_nki.py](/home/ubuntu/nki-moe/qwen_with_nki.py#L259).

## Code Changes

- Enabled the baked-in fused MoE kernel in [qwen_with_nki.py](/home/ubuntu/nki-moe/qwen_with_nki.py#L259).
- Aligned router matmul dtype with model dtype when the fused path is enabled in [qwen_with_nki.py](/home/ubuntu/nki-moe/qwen_with_nki.py#L270).
- Added `variance_epsilon` compatibility on the custom RMSNorm module in [nki_custom_rmsnorm.py](/home/ubuntu/nki-moe/nki_custom_rmsnorm.py#L117).

## Activation Proof

- Candidate compile log shows repeated:
  - `Neuron: Running MoE Fused NKI kernel`
  - `Neuron: Run seletive loading kernel: _moe_token_gen_selective_load_kernel_nki_call`
- That confirms the submission path is actually using the fused selective-loading MoE kernel during token generation.

## Hotspot Shift

- After the one-kernel change, the named token-generation DMAProfiler hotspots changed materially.
- Aggregated named hotspot sums from the candidate compile log:
  - `token_generation`: `moe=0.000`, `attn=2.912`
- In other words, the named MoE selective-loading DMA hotspots disappeared from token generation, and the remaining named hotspots shifted to attention/QKV movement such as:
  - `prep_qkv_tensors`
  - `move_heads_front`
  - `GroupQueryAttention_O`

## Representative Benchmarks

Prompt 0 (`18 -> 64`):

- Score: `0.7478 -> 0.6852` (`-8.37%`)
- `e2e p99`: `14519.31 ms -> 15074.45 ms` (`+3.82%`, worse)
- `context_encoding avg`: `653.71 ms -> 410.62 ms` (`-37.19%`, better)
- `token_generation avg`: `10.56 ms -> 11.47 ms` (`+8.61%`, worse)

Prompt 4 (`402 -> 640`):

- Score: `0.6138 -> 0.7579` (`+23.47%`)
- `e2e p99`: `4194.23 ms -> 3447.30 ms` (`-17.81%`, better)
- `context_encoding avg`: `695.84 ms -> 455.59 ms` (`-34.53%`, better)
- `token_generation avg`: `11.14 ms -> 11.42 ms` (`+2.54%`, worse)

Across just these two representative prompts, average score improved by about `+5.98%`, but the behavior is mixed by prompt length.

## Interpretation

- The one-kernel change succeeded as a profiling-guided hotspot removal:
  - baseline token-generation MoE selective-loading DMA hotspots are gone
  - token-generation named hotspots now skew toward attention/QKV movement
- But it did **not** translate into lower `token_generation_model.latency_ms_avg` on the two representative prompts.
- The main measured gain showed up in `context_encoding_model`, especially on the long prompt.
- So this patch is useful as a first bottleneck-shift step, but it is not yet a clean “TKG win everywhere”.

## Next Step

- The next single area to investigate should be attention/QKV movement, not more MoE dispatch.
- The current data says:
  - MoE selective-loading was the first removable named token-generation hotspot
  - after removing it, attention/QKV movement becomes the next visible token-generation hotspot

## Artifacts

- Baseline reports:
  - [prompt0/report.json](/home/ubuntu/nki-moe/analysis/deep_profile_baseline/prompt0/report.json)
  - [prompt4/report.json](/home/ubuntu/nki-moe/analysis/deep_profile_baseline/prompt4/report.json)
- Candidate reports:
  - [prompt0/report.json](/home/ubuntu/nki-moe/analysis/deep_profile_candidate_v2/prompt0/report.json)
  - [prompt4/report.json](/home/ubuntu/nki-moe/analysis/deep_profile_candidate_v2/prompt4/report.json)
- Candidate run log:
  - [compile_and_benchmark.log](/home/ubuntu/nki-moe/analysis/deep_profile_candidate_v2/compile_and_benchmark.log)
- Local-only compiler snapshots under `analysis/deep_profile_baseline/nxd_model_snapshot` and `analysis/deep_profile_candidate_v2/nxd_model_snapshot` were used during profiling but are intentionally not part of the committed result set.
