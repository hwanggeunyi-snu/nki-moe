# Analysis Artifacts

This directory keeps only the compact artifacts needed to understand the two submission-path changes made in `qwen_with_nki.py`.

Retained artifacts:

- `CHANGE_01_MOE_FUSED.md`
- `CHANGE_02_QKV_MOVEMENT.md`
- `deep_profile_candidate_v2/summary.md`
- `deep_profile_candidate_v2/compile_and_benchmark.log`
- `deep_profile_candidate_v2/prompt0/report.json`
- `deep_profile_candidate_v2/prompt4/report.json`
- `stage2_next_bottleneck/summary.md`
- `stage2_next_bottleneck/compile/compile_prompt0.log`
- `stage2_next_bottleneck/compile/prompt0_compile_report.json`
- `stage2_next_bottleneck/prompt0/report.json`
- `stage2_next_bottleneck/prompt0/model_event_totals.tsv`
- `stage2_next_bottleneck/prompt0/runtime_event_totals.tsv`
- `stage2_next_bottleneck/prompt4/report.json`
- `stage2_next_bottleneck/prompt4/model_event_totals.tsv`
- `stage2_next_bottleneck/prompt4/runtime_event_totals.tsv`
- `stage2_qkv_candidate/summary.md`
- `stage2_qkv_candidate/compile_prompt0.log`
- `stage2_qkv_candidate/prompt0_report.json`

Deleted artifacts:

- large `nxd_model_snapshot/` trees
- `inspect/` capture directories
- exploratory one-off directories that are no longer referenced
