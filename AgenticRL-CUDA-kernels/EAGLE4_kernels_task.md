# Task: add EAGLE-4 speculative-decoding kernels as Stage 5 workloads

You are working in AgenticRL-CUDA-kernels/ (WSL2). Read AGENTS.md, the Ground
rules in Benchmark_diagnostics_task.md (binding), README.md "Kernel tasks the
agent works on", and an existing task such as tasks/grpo_group_norm/ before
writing anything. Match the existing task format exactly.

> Note: this brief was written from a paraphrased reading of the EAGLE-4
> kernels page. The shapes, k and tree sizes below are chosen for this
> repository, not taken from EAGLE-4. Check every kernel description against
> the original page before implementing it.

## Source and licensing (do this first)

- Reference: https://github.com/eagle-hq/EAGLE-4/blob/main/docs/kernels.md
- EAGLE-4 is released under the "Eagle 4 Source-Available License"
  (LICENSES/ in that repo). Read it before writing anything. If it forbids
  derived benchmarks or evaluation use, STOP and report.
- Write every kernel from the documented behaviour. Do not copy, paraphrase or
  port code from eagle_harness/ (candidate.cu, gdn_tree*, tree_step,
  accept_tail, ...). Cite the doc, not the code.
- Add an "eagle4" entry to SOURCES in utils/task_catalog.py with the exact
  license name and URL. Mark every new task eval_only=True until the user
  clears training use.

## Workloads (stage 5; one tasks/<name>/ folder each)

Engine-free, single-GPU contracts. Each needs a trusted PyTorch reference in
model.py (inputs come from utils/task_inputs.py via the catalog), an honest
Triton solution and a CUDA solution where practical, plus model_new_hack.py
(Torch-op substitution) and model_new_wrong.py (wrong output).

1. **eagle_candidate_topk:** hidden [B, H] bf16 and lm_head [V, H] bf16.
   Return the top-k token ids per row from a sharded vocabulary top-k, then
   exact fp32 re-scoring of those k candidates (gather-dot of hidden with the
   selected lm_head rows). Outputs: ids int64 [B, k], scores fp32 [B, k].
   Ties go to the lower id. Cases: B in {1, 8}, H = 4096,
   V in {32000, 128256}, k in {8, 16}.
2. **eagle_tree_step:** one draft-tree expansion. Inputs: parent cumulative
   log-probs [B, W], child log-probs [B, W, k], parent ids. Outputs: next
   frontier (top-W of parent + child), parent index per new node, and the
   ancestor table [B, nodes, depth] (int32, -1 padded). Integer outputs must
   match exactly.
3. **eagle_tree_row_attention:** draft attention where each tree row attends
   to a sliding window of the committed KV plus its own tree ancestors (mask
   derived from the parent table), split over keys. bf16 inputs, fp32
   softmax. Cases: heads 32/8 (GQA), head_dim 128, window 512–4096, tree
   nodes 16–64.
4. **eagle_tree_verify_attention:** the verify pass. Tree-node queries attend
   to the full committed context plus tree-causal ancestors, split over the
   committed context.
5. **eagle_gdn_tree_conv:** builds the parent table and each node's width-4
   conv-window ancestors once, then a tree causal conv1d over the nodes. The
   parent table and windows must match exactly; conv output uses the standard
   tolerance.
6. **eagle_gdn_tree_recurrence:** the tree gated delta rule. Each node's state
   is its parent's state updated by the node's key/value/gates. Return the
   per-node outputs and final states. Default form only: a node-ordered walk
   over value-dimension tiles. No Gluon.
7. **eagle_accept_tail:** greedy acceptance. Inputs: draft tree tokens, parent
   table and the target model's argmax per node. Walk the tree and output the
   accept length, accepted path indices, bonus token and compacted KV slot
   indices, all integer and exact. Batch size 1 and 8.

Out of scope: L2 prefetch, programmatic dependent launch, FlashInfer and
SGLang/vLLM integration, the Gluon backend, Mamba2 selective state update.

## Rules

- Register each task in utils/task_catalog.py (stage 5, cases, sources,
  backends) with a correct workload() FLOP/byte convention or "unavailable".
- Integer outputs use exact comparison via
  RolloutConfig.exact_integer_outputs / utils/task_verification.py. That mode
  is still awaiting the user's approval, so say so in your report. Do NOT
  change floating-point tolerances (atol/rtol 1e-2); ASK if a kernel cannot
  meet them.
- Kernel names must be authored and creditable (>= 4 characters, no generic
  names) so the authored-only-no-library-v1 gate can attribute them. No
  cuBLAS/cuDNN/FlashInfer calls in honest solutions.
- Keep pure modules Torch-free. Add no Python dependencies.
- Tests: extend tests/test_task_catalog.py with reference edge cases (single
  node trees, depth-1 trees, ties, full window). Explicitly skip-mark GPU
  tests.
- Verify every honest solution under the CPU interpreter
  (CUDA_AGENT_DEVICE=cpu TRITON_INTERPRET=1, utils.verification in a
  scaffolded workdir via utils.task_suite.scaffold) and save the logs plus a
  manifest under artifacts/task-stage5-<UTC timestamp>/. If the interpreter
  cannot evaluate a kernel (e.g. bf16 tl.dot), record it as unsupported and
  do not work around it.
- Run both suites: `python3 -m unittest discover` and
  `.venv/bin/python -m unittest discover`. Both must be green.
- Do not launch Modal/H200 runs (paid; the user approves them). Do not
  commit: .git is read-only in your sandbox, and Claude commits from the
  Windows checkout.
- Update README.md (the per-task table and the stage table: add stage 5),
  VALIDATION.md and KNOWN_ISSUES.md.

## Report back

Files added, the license finding, the raw test summary lines, per-task CPU
verification results, anything unsupported or unverified and why, and open
decisions (license clearance, exact-integer mode, any tolerance question).
