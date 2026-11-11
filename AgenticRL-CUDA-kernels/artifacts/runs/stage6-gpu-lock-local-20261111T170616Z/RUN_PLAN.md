# Stage 6 prepared GPU runs

No run is launched. Claude reviews/commits and launches acceptance, reports,
then launches stratified only after the next review/approval. Each request
uses a separate frozen source and manifest hash.

## Selection and limits

Stratified is a deterministic six-case covering subset per task. It includes
fp16/bf16, mixed slot ranks, none/mixed/single/distinct patterns, rank 8/256
and one/three slices. The full catalog and smoke/acceptance selections are
unchanged. The widest shapes run in acceptance.

The paired catalog already includes 256-token BGMV at hidden=4096. The last/widest acceptance row is 64 tokens at hidden=14336. No catalog change; combined 256-token/widest shape is not claimed.

## Expected outcomes

- Honest: verification passes (relative float bounds, exact skipped y bytes;
  metadata exact), authored provenance >= 0.9 and zero library share.
- Hack: verifies but fails provenance.
- Wrong: fails verification.
- Wrong-padding: fails verification when rank padding exists; otherwise
  unavailable with `control_not_applicable_no_rank_padding`.
- Wrong-skip: fails verification when idx=-1 exists; otherwise unavailable
  with `control_not_applicable_no_skipped_tokens`.
- fp16-intermediate: accuracy diagnostic attached to honest cases only;
  never an expected-failure row or reward gate.
- Report benchmark/capture/replay separately: every honest row should have
  benchmark availability and successful capture/replay for all implementations.
  Core reward success alone does not imply graph acceptance.
- `gpu_busy` is an environment-unavailable row, never a candidate failure;
  report any occurrence. Stop and report after each run; do not auto-rerun.

## Resource evidence

Full physical padding, input/builder copies, reference working storage, graph
pools and full-array float64 verifier/quantile arrays are budgeted. Largest
budget: 106.799 GB; headroom against 141 GB: 34.201 GB. A CPU meta-tensor probe
checks actual full-size reference/control storage shapes without allocating
payloads; it is not a GPU allocator measurement. Native kernels are a storage
proxy in that probe. Compiler/private allocator peaks and latency remain GPU
checks. See `symbolic-sizing.json`, `run-plan.json` and exact commands.

Time estimates use archived smoke rows matched by task/backend/fixture, with
wrong-skip proxied by wrong-output, multiplied by 1.5 plus explicit shape-scaled
extra budgets and 300 s startup. They are estimates, not new measurements.
600 s was only demonstrated at smoke sizes; request 1200 s per new case.
The 20700 s suite and 21600 s provider caps still count wall time, including
any lock wait. Each task deadline separately extends by its lock wait.

## Acceptance

6 cases; 56 rows, 56 applicable. Estimated wall: 8561.914 s (2.378 h).

| Task | Case ID | Skipped tokens | Memory budget GB |
|---|---|---:|---:|
| `lora_shrink_bgmv` | `tokens64_hidden14336_out14336_rank256_slots16_slices3_patternmixed_mixedtrue_bfloat16` | 16 | 29.630 |
| `lora_expand_bgmv` | `tokens64_hidden14336_out14336_rank256_slots16_slices3_patternmixed_mixedtrue_bfloat16` | 16 | 30.656 |
| `lora_segment_metadata` | `tokens4096_hidden14336_out14336_rank256_slots16_slices3_patternmixed_mixedtrue_bfloat16` | 1024 | 17.234 |
| `lora_shrink_sgmv` | `tokens4096_hidden14336_out14336_rank256_slots16_slices3_patternmixed_mixedtrue_bfloat16` | 1024 | 37.098 |
| `lora_expand_sgmv` | `tokens4096_hidden14336_out14336_rank256_slots16_slices3_patternmixed_mixedtrue_bfloat16` | 1024 | 102.773 |
| `lora_fused_sgmv` | `tokens4096_hidden14336_out14336_rank256_slots16_slices3_patternmixed_mixedtrue_bfloat16` | 1024 | 106.799 |

## Stratified

36 cases; 336 rows, 312 applicable. Estimated wall: 14092.250 s (3.915 h).

| Task | Case ID | Skipped tokens | Memory budget GB |
|---|---|---:|---:|
| `lora_shrink_bgmv` | `tokens1_hidden2048_out2048_rank8_slots1_slices1_patternsingle_mixedfalse_float16` | 0 | 17.185 |
| `lora_shrink_bgmv` | `tokens3_hidden2048_out2048_rank8_slots4_slices1_patternmixed_mixedfalse_bfloat16` | 1 | 17.192 |
| `lora_shrink_bgmv` | `tokens3_hidden8192_out8192_rank64_slots4_slices2_patterndistinct_mixedfalse_float16` | 0 | 17.551 |
| `lora_shrink_bgmv` | `tokens64_hidden2048_out2048_rank128_slots16_slices1_patternnone_mixedtrue_bfloat16` | 64 | 17.487 |
| `lora_shrink_bgmv` | `tokens256_hidden4096_out4096_rank256_slots16_slices3_patternmixed_mixedtrue_float16` | 64 | 20.884 |
| `lora_shrink_bgmv` | `tokens256_hidden4096_out4096_rank256_slots16_slices3_patternmixed_mixedtrue_bfloat16` | 64 | 20.884 |
| `lora_expand_bgmv` | `tokens1_hidden2048_out2048_rank8_slots1_slices1_patternsingle_mixedfalse_float16` | 0 | 17.187 |
| `lora_expand_bgmv` | `tokens3_hidden2048_out2048_rank8_slots4_slices1_patternmixed_mixedfalse_bfloat16` | 1 | 17.196 |
| `lora_expand_bgmv` | `tokens3_hidden8192_out8192_rank64_slots4_slices2_patterndistinct_mixedfalse_float16` | 0 | 17.573 |
| `lora_expand_bgmv` | `tokens64_hidden2048_out2048_rank128_slots16_slices1_patternnone_mixedtrue_bfloat16` | 64 | 17.564 |
| `lora_expand_bgmv` | `tokens256_hidden4096_out4096_rank256_slots16_slices3_patternmixed_mixedtrue_float16` | 64 | 22.031 |
| `lora_expand_bgmv` | `tokens256_hidden4096_out4096_rank256_slots16_slices3_patternmixed_mixedtrue_bfloat16` | 64 | 22.031 |
| `lora_segment_metadata` | `tokens1_hidden2048_out2048_rank8_slots1_slices1_patternsingle_mixedfalse_float16` | 0 | 17.180 |
| `lora_segment_metadata` | `tokens3_hidden2048_out2048_rank8_slots4_slices1_patternmixed_mixedfalse_bfloat16` | 1 | 17.180 |
| `lora_segment_metadata` | `tokens3_hidden8192_out8192_rank64_slots4_slices2_patterndistinct_mixedfalse_float16` | 0 | 17.180 |
| `lora_segment_metadata` | `tokens64_hidden2048_out2048_rank128_slots16_slices1_patternnone_mixedtrue_bfloat16` | 64 | 17.181 |
| `lora_segment_metadata` | `tokens512_hidden4096_out4096_rank256_slots16_slices3_patternmixed_mixedtrue_float16` | 128 | 17.187 |
| `lora_segment_metadata` | `tokens512_hidden4096_out4096_rank256_slots16_slices3_patternmixed_mixedtrue_bfloat16` | 128 | 17.187 |
| `lora_shrink_sgmv` | `tokens512_hidden2048_out2048_rank8_slots1_slices1_patternsingle_mixedfalse_float16` | 0 | 17.272 |
| `lora_shrink_sgmv` | `tokens512_hidden2048_out2048_rank8_slots4_slices1_patternmixed_mixedfalse_bfloat16` | 128 | 17.278 |
| `lora_shrink_sgmv` | `tokens512_hidden8192_out8192_rank64_slots4_slices2_patterndistinct_mixedfalse_float16` | 508 | 17.973 |
| `lora_shrink_sgmv` | `tokens512_hidden2048_out2048_rank128_slots16_slices1_patternnone_mixedtrue_bfloat16` | 512 | 17.578 |
| `lora_shrink_sgmv` | `tokens512_hidden4096_out4096_rank256_slots16_slices3_patternmixed_mixedtrue_float16` | 128 | 21.064 |
| `lora_shrink_sgmv` | `tokens512_hidden4096_out4096_rank256_slots16_slices3_patternmixed_mixedtrue_bfloat16` | 128 | 21.064 |
| `lora_expand_sgmv` | `tokens512_hidden2048_out2048_rank8_slots1_slices1_patternsingle_mixedfalse_float16` | 0 | 17.901 |
| `lora_expand_sgmv` | `tokens512_hidden2048_out2048_rank8_slots4_slices1_patternmixed_mixedfalse_bfloat16` | 128 | 17.907 |
| `lora_expand_sgmv` | `tokens512_hidden8192_out8192_rank64_slots4_slices2_patterndistinct_mixedfalse_float16` | 508 | 21.588 |
| `lora_expand_sgmv` | `tokens512_hidden2048_out2048_rank128_slots16_slices1_patternnone_mixedtrue_bfloat16` | 512 | 18.197 |
| `lora_expand_sgmv` | `tokens512_hidden4096_out4096_rank256_slots16_slices3_patternmixed_mixedtrue_float16` | 128 | 23.360 |
| `lora_expand_sgmv` | `tokens512_hidden4096_out4096_rank256_slots16_slices3_patternmixed_mixedtrue_bfloat16` | 128 | 23.360 |
| `lora_fused_sgmv` | `tokens512_hidden2048_out2048_rank8_slots1_slices1_patternsingle_mixedfalse_float16` | 0 | 17.955 |
| `lora_fused_sgmv` | `tokens512_hidden2048_out2048_rank8_slots4_slices1_patternmixed_mixedfalse_bfloat16` | 128 | 17.961 |
| `lora_fused_sgmv` | `tokens512_hidden8192_out8192_rank64_slots4_slices2_patterndistinct_mixedfalse_float16` | 508 | 21.893 |
| `lora_fused_sgmv` | `tokens512_hidden2048_out2048_rank128_slots16_slices1_patternnone_mixedtrue_bfloat16` | 512 | 18.275 |
| `lora_fused_sgmv` | `tokens512_hidden4096_out4096_rank256_slots16_slices3_patternmixed_mixedtrue_float16` | 128 | 23.781 |
| `lora_fused_sgmv` | `tokens512_hidden4096_out4096_rank256_slots16_slices3_patternmixed_mixedtrue_bfloat16` | 128 | 23.781 |

