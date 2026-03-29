# Stage2 QKV Movement Candidate

## Change

- File: `qwen_with_nki.py`
- Bake-in flags:
  - `fused_qkv = True`
  - `qkv_kernel_enabled = True`
  - `qkv_nki_kernel_enabled = True`
- Existing `moe_fused_nki_kernel_enabled = True` retained
- Attention TKG kernels left disabled:
  - `attn_tkg_nki_kernel_enabled = False`
  - `attn_tkg_builtin_kernel_enabled = False`
  - `attn_block_tkg_nki_kernel_enabled = False`

## Validation

- `python3 -m py_compile qwen_with_nki.py` passed
- Fresh compile on Trn2 succeeded
- Compile log shows QKV NKI path is active:
  - `nki_qkv_projection_isa_kernel`
  - `nki_qkv_projection_cte_impl`
  - `_sharded_qkv_cte_kernel`

## Prompt 0 Result

- Current QKV candidate:
  - `e2e_model.latency_ms_p99 = 14643.20`
  - `context_encoding_model.latency_ms_avg = 410.20`
  - `token_generation_model.latency_ms_avg = 10.99`
  - `score = 0.72733`
- Previous MoE-fused-only candidate:
  - `e2e_model.latency_ms_p99 = 14932.06`
  - `context_encoding_model.latency_ms_avg = 410.69`
  - `token_generation_model.latency_ms_avg = 11.45`
  - `score = 0.69516`

## Immediate Takeaway

- Prompt 0 improved on all key metrics.
- The largest visible gain is in token generation average latency:
  - `11.45 ms -> 10.99 ms`
- This is consistent with the profiling-based decision to target attention/QKV movement next.
