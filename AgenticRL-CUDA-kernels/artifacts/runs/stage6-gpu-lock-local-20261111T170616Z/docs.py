from pathlib import Path
import json
bundle=Path('/tmp/stage6-gpu-lock/bundle-path').read_text()
plan=json.loads(Path('/tmp/stage6-gpu-lock/run-plan.json').read_text())

def update(name,fn):
 p=Path(name);b=p.read_bytes();nl='\r\n' if b'\r\n' in b else '\n';s=b.decode().replace('\r\n','\n');out=fn(s);assert out!=s;p.write_bytes(out.replace('\n',nl).encode())
lock='''

## GPU contention lock

Measurement stages in `run_task` share a per-GPU `fcntl.flock`: profiling,
benchmark and diagnostics hold one lock until all three finish. Compilation
and verification remain concurrent. UUID/name discovery uses `nvidia-smi`
with the child's logical ordinal mapped through `CUDA_VISIBLE_DEVICES`;
a failed query records a reason and uses the selected visibility token as
a fallback identity. CPU evaluation takes no lock. Workers sharing a GPU
must share `CUDA_AGENT_GPU_LOCK_DIR` (default `/tmp/agenticrl-gpu-locks`).
The OS releases the lock on normal exit, stage failure, timeout or SIGKILL.

`gpu_lock=True` is the default; `--no-gpu-lock` disables acquisition.
`--gpu-lock-timeout-s` defaults to 600. Lock wait extends the task deadline;
`env/wall_s` still includes the wait. A capped wait returns reward −1 with
reason `gpu_busy`, excluded within its original group before advantages
and tied-group filtering. `adv/gpu_busy_excluded` counts excluded rows;
other failures and all submitted-row reward statistics retain their meaning.
`meta/gpu_*` fields are diagnostic. Auxiliary GEMM ceiling timing reacquires
the same lock; numerical-only LoRA checks remain verification work.

## Prepared Stage 6 GPU run plan (2026-11-11)

The H200 fp16 smoke rerun passed 46 applicable rows with all 12 honest graph
captures/replays passing (`stage6-smoke-rerun-20261111T031847Z`). That evidence
predates this contention-lock implementation. The next runs are prepared for
Claude to review and launch in order, reporting after each; none is launched here.

| Run | Cases | Rows (applicable) | Estimated wall | Case cap |
|---|---:|---:|---:|---:|
| Acceptance | 6 | 56 (56) | 2.4 h | 1200 s |
| Stratified covering subset | 36 | 336 (312) | 3.9 h | 1200 s |

Stratified selects six stable cases per task covering both dtypes, rank 8/256,
slices 1/3, mixed ranks, and none/mixed/single/distinct patterns. Widest shapes
run in acceptance. The paired catalog already tests 256-token decode at hidden
4096; its widest 14336-hidden BGMV acceptance row uses 64 tokens. No catalog
shape was added. Every acceptance case includes skipped tokens.

These are estimates from archived smoke wall times and explicit sizing budgets,
not new GPU measurements. Largest-case planning budget: 106.8 GB of 141 GB,
including full float64 verification and graph-pool allowances. Allocator peaks,
cold compilation and largest-case correctness remain hardware checks.
[Case IDs, resource assumptions, test evidence and prepared requests](BUNDLE/README.md)
are retained together. Both requests use detached H200 execution, all fixtures,
both backends and CUDA graphs. The fp16-intermediate comparison stays diagnostic.
'''.replace('BUNDLE',bundle)
update('README.md',lambda s:s+lock)
update('Benchmark_diagnostics_task.md',lambda s:s.replace('## Step 5: Representative workloads\n','## Step 5: Representative workloads (implemented)\n\nThe Stage 0–6 task suite (`utils/task_suite.py`, `tasks/`, `scripts.validate`)\nimplements this step, including per-task honest/hack/wrong controls and the\napproved task-only exact/relative verification modes. Hardware coverage and\nremaining larger-case checks are recorded in VALIDATION.md.\n',1).replace('## Step 6: GPU contention control\n','## Step 6: GPU contention control (implemented; local process tests)\n',1).replace('**ASK** how lock wait interacts with the task deadline. A candidate must not get `reason: "timeout"` because another rollout held the GPU.', '''User decision (2026-11-11): lock wait extends the task deadline. A separate
`gpu_lock_timeout_s` cap (default 600 s) returns `gpu_busy`, reward −1, and
excludes only those rows from group advantages. Other failure reasons and
`compute_reward` inputs are unchanged. `gpu_lock=True` by default; the CLI
supports `--no-gpu-lock`. `utils/gpu_lock.py` polls nonblocking `flock` in
`CUDA_AGENT_GPU_LOCK_DIR` (default shared temporary directory), resolves the
child's visible device without Torch, records an explicit fallback when the
UUID cannot be read, and relies on OS close/process-death cleanup. Compile and
verification run outside the lock. `run_task` holds one lock across profiling,
benchmark and diagnostics, then releases it. Auxiliary GEMM ceiling measurements
reacquire that same lock; numerical-only LoRA diagnostics are verification work.
The result records UUID/name, identity reason/key, status and wait; those fields
never enter `compute_reward`. Tests cover contention/deadline extension, capped
wait, SIGKILL, failure/timeout cleanup, visibility mapping, CPU, opt-out and the
narrow advantage exclusion. See VALIDATION.md for raw summaries. Real-GPU
contention and shared-directory deployment remain unverified.''',1))
update('KNOWN_ISSUES.md',lambda s:s+'''

- **GPU contention lock: locally verified, hardware concurrency unverified.**
  Measurement stages now use an OS lock keyed by the child's GPU UUID (or an
  explicitly labelled visibility-token fallback). Cooperating processes must
  share the lock directory. Unknown UUID aliases cannot be reconciled across
  different visibility conventions; metadata records the fallback and reason.
  The lock is advisory and does not serialize external GPU users. Compile and
  verification intentionally remain concurrent. Lock wait extends the deadline;
  the separate cap produces `gpu_busy`, excluded only from advantages. Real
  process contention, SIGKILL and timeout cleanup are covered without a GPU.
- **Stage 6 wider GPU evidence is prepared, not run.** Acceptance and a six-case
  per-task stratified subset cover bf16, mixed ranks and skipped tokens. All
  acceptance cases have skipped tokens. The catalog already includes 256-token
  BGMV at hidden 4096; its widest acceptance case has 64 tokens, and no combined
  256-token/14336-hidden acceptance is claimed. Resource budgets are symbolic
  estimates; hidden compiler/allocator workspaces, full gate sorts, cold compile
  time and wider-case honest bounds still require hardware evidence. Request
  1200 s per case rather than extrapolating the smoke's demonstrated 600 s cap.
  fp16-intermediate remains diagnostic. No Modal run or live W&B write was made.
''')
update('VALIDATION.md',lambda s:s+f'''

## Stage 6 GPU preparation and contention lock (2026-11-11)

Local evidence: `{bundle}/`. No Modal run, live W&B write or commit is performed.
The separate fp16 H200 rerun above remains historical GPU evidence; these new
sources have not run on a GPU. Required suite summaries will be appended when
both discovery processes complete.

`utils/gpu_lock.py` resolves UUID/name with `nvidia-smi` and maps logical CUDA
ordinals through visibility order. CPU and disabled paths do not acquire a
lock. A failed UUID query records its reason and a visibility-token identity;
GPU driver nodes also detect GPU hosts without `nvidia-smi`. Polling uses
nonblocking `flock`, a 600 s default wait cap and a shared overridable lock
directory. File descriptors close on exec and are released by the OS at death;
lock files are never unlinked. One handle spans profiling, benchmark and
profile-artifact diagnostics. Auxiliary GEMM ceiling timing is serialized too;
numerical-only LoRA checks remain verification work. Lock wait extends the
rollout and case diagnostic deadline, but remains visible in total wall time.
`gpu_busy` is the only added reward reason and advantage exclusion. No inputs
to `compute_reward` or other failure gates changed. Exclusion happens within
original complete submitted groups before normalization/tied filtering; rows
never move between groups. Aggregate reward statistics still cover all rows.

Focused synthetic protocol/process tests passed: 71 tests / 44.261 s / OK
(skipped=6: Torch-dependent cases under system Python), then 29 measurement and
policy tests / 15.417 s / OK, then the final 9 lock/exclusion tests / 3.339 s / OK.
The final pass adds default-CUDA driver-node discovery without importing Torch.
Tests cover two contending processes, compile/verify outside the lock, waiting
beyond the original deadline with exact extension, SIGKILL releasing the lock,
cap expiry as `gpu_busy`, holding across all measured stages, release on failed
or timed-out profiling, UUID visibility mapping/fallback, CPU/opt-out and only
`gpu_busy` exclusion under all three presets. Protocol GPU timings are synthetic.

`run-plan.json` records every case/fixture/backend row and the sizing assumptions.
The stratified grid is six deterministic cases per task (36 total), rather than
all 112 cases. Coverage: both dtypes; genuine mixed slot ranks; none, mixed,
single, distinct; ranks 8 and 256; slices 1 and 3. Widest cases run separately in
acceptance. Acceptance: 56/56 applicable; stratified: 312/336 applicable (14
wrong-skip controls have no skipped tokens, 10 wrong-padding controls have no
rank padding). fp16-intermediate never becomes an expected-failure row.

Decode finding: existing catalog rows already reach 256 tokens at hidden 4096;
the paired grid's last/widest row is 64 tokens at hidden 14336. No shape change
was made. Each acceptance case has idx=-1 tokens (1024 at T4096, 16 at T64).
Memory budgets include physical NaN padding, input/builder copies, reference
workspace, graph pools and full-array float64 error/quantile workspace, with a
16 GiB runtime reserve. The largest budget is 106.799303296 GB, leaving
34.200696704 GB against the supplied 141 GB H200 capacity. These are conservative
planning allowances, not measured peaks or a bound on arbitrary compiler memory.
Every fixture shares the applicable task budget. `scripts.size_stage6` probes
full-size CPU meta tensors, including the Torch controls and a reference-storage
proxy for authored kernels; authored GPU execution and hidden workspaces remain
unmeasured. No large payload tensor was allocated by that sizing probe.

Wall estimates use actual archived smoke per-task/backend/fixture wall times
from `stage6-smoke-rerun-20261111T031847Z`, proxying wrong-skip with wrong-output,
then multiply by 1.5, add explicit shape-scaled budgets (300 s for widest honest/
hack, 120 s for verification-only controls), and 300 s startup. They predict
{plan['runs']['acceptance']['estimated_wall_s']:.3f} s (2.38 h) for acceptance and
{plan['runs']['stratified']['estimated_wall_s']:.3f} s (3.91 h) for stratified,
under the 20700 s suite/21600 s function limits. 600 s per new case is not
established by the smoke. Prepared requests use 1200 s; cold specialization,
full-size sort cost and actual allocator peaks remain hardware checks. These
estimates do not promise runtime or establish GPU performance/acceptance.
''')
print(bundle)
