# Stage2 Next Bottleneck Confirmation

## Scope

- Target path: `main.py --enable-nki --platform-target trn2 --tp-degree 4`
- Model file under test: `qwen_with_nki.py` at HEAD `05a0c78`
- Compile artifact used for all runs: `/home/ubuntu/qwen-30b-a3b/traced_model_stage2_post_moe_profile_seq640_tp4`
- Goal: confirm the next custom-kernel target after the baked-in `moe_fused_nki_kernel_enabled=True` change

## What Was Run

1. Clean rebuild with current submission-style `qwen_with_nki.py`
2. `evaluate_single` benchmark for prompt 0 during compile
3. `neuron-profile inspect` for prompt 0 with `--skip-compile`
4. `neuron-profile inspect` for prompt 4 with `--skip-compile`
5. `neuron-profile view --output-format json` export for both profiles
6. Runtime event aggregation from `trace_event`
7. Compile artifact review from `log-neuron-cc.txt`

## Current Build Check

- Clean rebuild completed successfully.
- Build log confirms the fused MoE token-generation path is active.
  - `Running MoE Fused NKI kernel`
  - `Run seletive loading kernel: _moe_token_gen_selective_load_kernel_nki_call`
- Rebuilt prompt 0 benchmark:
  - `e2e_model.latency_ms_p99 = 14932.06`
  - `context_encoding_model.latency_ms_avg = 410.69`
  - `token_generation_model.latency_ms_avg = 11.45`

## Representative Prompt Results

- Prompt 0 report:
  - `e2e_model.latency_ms_p99 = 14856.34`
  - `context_encoding_model.latency_ms_avg = 410.52`
  - `token_generation_model.latency_ms_avg = 11.40`
- Prompt 4 report:
  - `e2e_model.latency_ms_p99 = 3447.80`
  - `context_encoding_model.latency_ms_avg = 455.84`
  - `token_generation_model.latency_ms_avg = 11.42`

These are consistent with the prior MoE-fused candidate and show no regression from rebuilding the current single-file submission form.

## Hardware Profile Findings

### Prompt 0

- Model-attributed `nc_exec_running` time:
  - token generation total: `175.28 s`
  - context encoding total: `9.82 s`
  - token generation share: `94.69%`
- Dominant token-generation NEFF:
  - `_tp0_bk2`: `117.03 s` of `nc_exec_running`

### Prompt 4

- Model-attributed `nc_exec_running` time:
  - token generation total: `175.17 s`
  - context encoding total: `57.27 s`
  - token generation share: `75.36%`
- Dominant token-generation NEFF:
  - `_tp0_bk2`: `175.17 s` of `nc_exec_running`

### Event-Type Shape

Both prompt 0 and prompt 4 show the same runtime pattern:

- Largest HW-side execution event: `nc_exec_running`
- Next large runtime wrapper events: `nrt_execute`, `kmgr_exec_core`, `kbl_exec_wait`
- Explicit copy events are much smaller:
  - prompt 0: `dmem_buf_copyin + dmem_buf_copyout ~= 0.66 s`
  - prompt 4: `dmem_buf_copyin + dmem_buf_copyout ~= 0.66 s`

Interpretation:

- The remaining bottleneck is still inside token-generation execution on the device.
- The runtime trace does not support a pure host-orchestration bottleneck as the next target.
- The runtime trace also does not directly expose attention sub-op names, so it cannot by itself separate `QKV movement` from `attention core compute`.

## Compile Artifact Findings

### Baseline Before MoE Fused

In `analysis/deep_profile_baseline/nxd_model_snapshot/token_generation_model/_tp0_bk0/log-neuron-cc.txt`, the top named DMAProfiler entries after the generic loads were MoE selective-loading entries such as:

- `forward_selective_loading`
- `ExpertFusedColumnParallelLinear`
- `aten.index_gather`

Example pattern:

- `Est. DMA time: 58.827us ... tensor_op_name: ... /forward_selective_loading/.../ExpertFusedColumnParallelLinear/.../aten__index_gather`

### Current Build After MoE Fused

In `analysis/stage2_next_bottleneck/compile/nxd_model_snapshot/token_generation_model/_tp0_bk0/log-neuron-cc.txt`:

- `forward_selective_loading: 0`
- `ExpertFusedColumnParallelLinear: 0`
- `aten.index_gather: 0`
- `prep_qkv_tensors: 4`
- `move_heads_front: 4`
- `GroupQueryAttention_O: 4`

The current top named DMAProfiler entries after the generic loads are now attention-path layout / output-projection operations, for example:

- `prep_qkv_tensors[...]/move_heads_front/...`
- `GroupQueryAttention_O[...]/RowParallelLinear[...]`

Example pattern:

- `Est. DMA time: 19.507us ... tensor_op_name: ... /prep_qkv_tensors/.../move_heads_front/...`
- `Est. DMA time: 19.507us ... tensor_op_name: ... /GroupQueryAttention_O/.../RowParallelLinear/...`

Interpretation:

- The explicit named MoE selective-loading hotspot class is gone.
- The first remaining named hotspot class in token generation is now the attention/QKV path.
- The names are layout-heavy and movement-heavy, not router or expert MLP names.

## Decision

The next custom-kernel target is:

- `attention/QKV movement`

More precisely:

- next scope should be `QKV/layout movement first`
- not another MoE kernel
- not router / expert MLP split kernels
- not host runtime loop work as the first follow-up

## Why This Is The Right Next Target

1. Runtime hardware profiles for both representative prompts still show token generation dominating device execution.
2. After the MoE fused change, compile artifacts no longer show MoE selective-loading as the top named hotspot class.
3. The remaining named token-generation hotspots are now exclusively attention/QKV path names:
   - `prep_qkv_tensors`
   - `move_heads_front`
   - `GroupQueryAttention_O`
4. The hardware profile does not show standalone DMA events overtaking total device execution, so the safest interpretation is:
   - the next bottleneck lives inside the attention path
   - the first kernelization target inside that path should be movement/layout work before trying a larger attention-core rewrite

## Final Recommendation

For the next one-kernel optimization step, target:

- `fused_qkv + qkv kernel path`

Rationale:

- It matches the first remaining named hotspot class in the current token-generation compile artifact.
- It is submission-compatible as a single baked-in change in `qwen_with_nki.py`.
- It is a cleaner second step than trying another MoE change, because the MoE selective-loading hotspot class has already been removed.

## Files Produced

- `analysis/stage2_next_bottleneck/compile/compile_prompt0.log`
- `analysis/stage2_next_bottleneck/compile/prompt0_compile_report.json`
- `analysis/stage2_next_bottleneck/compile/nxd_model_snapshot/...`
- `analysis/stage2_next_bottleneck/prompt0/report.json`
- `analysis/stage2_next_bottleneck/prompt0/view.json`
- `analysis/stage2_next_bottleneck/prompt0/runtime_event_totals.tsv`
- `analysis/stage2_next_bottleneck/prompt0/model_event_totals.tsv`
- `analysis/stage2_next_bottleneck/prompt4/report.json`
- `analysis/stage2_next_bottleneck/prompt4/view.json`
- `analysis/stage2_next_bottleneck/prompt4/runtime_event_totals.tsv`
- `analysis/stage2_next_bottleneck/prompt4/model_event_totals.tsv`
