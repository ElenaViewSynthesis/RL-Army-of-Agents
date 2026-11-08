# Triton installation & hacking reference

Reference copy of Triton's installation, build, hacking and compatibility
sections, kept next to `CUDA_on_Modal.md`, `GPU-acceler.md` and the sweep guide.
The repository mapping comes first; the upstream sections below the divider
retain their original wording, code blocks, relative links and environment
variable names. Relative links in the copy refer to the upstream repository;
use the commit-pinned README link to follow them.

## How this maps to AgenticRL-CUDA-kernels

Agent policy: [SKILL_TRITON.md](SKILL_TRITON.md). Shared-host agents: [WANDB_parallel_sweep_agents.md](WANDB_parallel_sweep_agents.md).

- **Resolved version vs upstream main.** `requirements-gpu.txt` specifies
  `triton>=3.0`, a lower bound rather than an exact version pin. H200 preflight
  run `20261106T185236Z` recorded **Triton 3.8.0** in
  `artifacts/preflight.stdout-20261106T185236Z.log` and the corrected
  `artifacts/environment-20261106T185236Z.json`. The copied upstream README
  describes latest main at the commit identified below, not that measured
  wheel. Knobs, defaults and supported configurations may differ by version.
  The mapping knobs below were checked against the
  [v3.8.0 README](https://github.com/triton-lang/triton/blob/v3.8.0/README.md).
  In particular, main's copied `TRITON_ALLOW_FP_FUSION` entry is **not documented
  in the v3.8.0 README**; its availability is not asserted for this repo's wheel.
- **`TRITON_INTERPRET=1` is this repo's CPU path.**
  `scripts/validate.py --device cpu` sets both `CUDA_AGENT_DEVICE=cpu` and
  `TRITON_INTERPRET=1`; `tests/test_triton_kernel.py` sets them in its child
  process too. `kernels/triton/axpby.py` supplies only block size **128** to
  autotuning in interpreter mode, avoiding GPU autotune timers. This path
  checks correctness only. `utils/profiling.py` returns unmeasured/null timing
  and provenance on CPU, and `tests/test_provenance.py` explicitly checks that
  behavior; `utils/benchmark.py` reports CPU benchmarking unavailable.
- **Task-local cache.** `utils/backend.py` registers
  `TRITON_CACHE_DIR` with subdirectory `triton`, and
  `utils/rollout_env.py::run_task` defaults it to
  `<workdir>/.cache/triton` via `env.setdefault`. Separate task workdirs thus
  get separate caches **unless the caller already supplied a cache override**.
  This is the runtime-cache mapping for upstream's `TRITON_HOME` discussion;
  it does not relocate upstream's build/download directory. The override is
  confirmed in [v3.8.0 knobs.py](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/knobs.py).
  For a developer run investigating a suspected stale cache hit,
  `TRITON_ALWAYS_COMPILE=1` is the documented force-compilation knob.
- **Autotuning visibility.** `kernels/triton/axpby.py` tries block sizes
  **128/256/512** on GPU, each with four warps; interpreter mode uses only 128.
  `TRITON_PRINT_AUTOTUNING=1` reports the winning config and total tuning time
  in the v3.8.0 README. Use it to check tuning before interpreting `bench/*`:
  `utils/benchmark.py` runs an initial forward in the separately recorded
  `prepare_s` phase, then warmup and measurement rounds. JIT/autotuning must
  finish in prepare; prepare time is excluded from steady-state samples.
  `utils/profiling.py` separately warms up JIT backends before its profiler
  pass. Turn off forced recompilation and noisy dump/debug settings for
  ordinary performance measurements; these are developer diagnostics.
- **Kernel-symbol inspection for provenance.**
  `tests/test_provenance.py` exercises the honest and Torch-op reward-hack
  fixtures. `utils/provenance.py` and the backend registry implement anchored
  authored-name matching; see `VALIDATION.md`, **Approved symbol rules**.
  Triton's rule is `^{name}(_\d\w*)?$`, with the authored name regex-escaped;
  generic names and Torch/Inductor namespaces cannot earn authored credit.
  H200 run `20261106T185236Z` observed the bare **`axpby_kernel`** symbol.
  On a GPU developer host, `TRITON_KERNEL_DUMP=1` with `TRITON_DUMP_DIR`,
  `MLIR_ENABLE_DUMP=1` and `LLVM_IR_ENABLE_DUMP=1` expose compiled entry points
  and IR stages when investigating a mismatch. All are documented in v3.8.0.
  They help explain an `unmatched_agent_names` entry in
  `artifacts/kernel_names-*.json`; the actual profiler event recorded by
  `scripts/record_kernel_names.py` remains the provenance evidence.
  Upstream warns that caching can suppress dumps and suggests
  `rm -r ~/.triton/cache/*`. Here the relevant default is the particular task's
  `.cache/triton`; resolve any inherited `TRITON_CACHE_DIR` first, and clear
  only the intended inactive task cache on a shared host. No cache deletion
  is performed by this guide. Unknown profiler events still count as Torch.
- **Hardware and Python fit.** The copied README and v3.8.0 README list Linux,
  NVIDIA **CC 8.0+**, AMD **ROCm 6.2+**, and wheels for **CPython 3.10–3.14**.
  The recorded H200 is **CC 9.0**. This repo's `README.md` requires Python
  **3.11+**, consistent with the cuTile floor stated in
  `requirements-cutile.txt`. Local `.venv/pyvenv.cfg` records **3.13.9**;
  the Modal definition in `cloud/modal_runtime.py` uses Python **3.12** and
  that H200 preflight recorded **3.12.1**. Both fit the documented wheel range.
- **ptxas.** `cloud/modal_runtime.py` selects the CUDA **13.3.1 devel** image,
  `requirements-gpu.txt` pins `nvidia-cuda-nvcc==13.3.73`, and the preflight
  records system nvcc **13.3**. This supplies the CUDA 13.3 system toolchain,
  including ptxas, but **the run did not record `ptxas --version`**.
  Do not infer the assembler actually used by Triton from nvcc alone:
  [v3.8.0 compiler.py](https://github.com/triton-lang/triton/blob/v3.8.0/third_party/nvidia/backend/compiler.py)
  selects a Triton tool, and its
  [knobs.py](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/knobs.py)
  has a wheel-bundled default tool path. **`PTXAS_OPTIONS`** is documented in
  v3.8.0 and read by that version's NVIDIA knobs as extra assembler options.
  No claim is made here that the wheel's selected assembler was version 13.3.
- **What this repo does not use.** The cloud runtime consumes a pip wheel; it
  does not build Triton from source or use a custom LLVM. Upstream's
  `make dev-install` / `make test` commands below are upstream build/test
  procedures. This repo's own suite uses `python -m unittest discover`.
  `TRITON_ENABLE_ASAN` is documented as **AMD-only** in both READMEs, so it
  is not an H200/NVIDIA validation feature used here. The source-build,
  override and compiler-inspection sections are developer references;
  they do not change this repo's agent policy or reward measurement contract.

---

Source: <https://github.com/triton-lang/triton> (README, fetched 2026-11-08)

Copied from main commit [`e8dda3400e82957bf3408c8251619790c78d8d5c`](https://github.com/triton-lang/triton/blob/e8dda3400e82957bf3408c8251619790c78d8d5c/README.md) (commit dated 2026-11-07).
The retrieval date is 2026-11-08; 2026-11-07 is the upstream commit date.
Checked against release v3.8.0, commit `c01b6774b1865984607d89d89d3a10833de92037`.
The seven selected sections below are verbatim excerpts; the upstream MIT notice follows them.

# Quick Installation

You can install the latest stable release of Triton from pip:

```shell
pip install triton
```

Binary wheels are available for CPython 3.10-3.14.

# Install from source

```shell
git clone https://github.com/triton-lang/triton.git
cd triton

pip install -r python/requirements.txt # build-time dependencies
pip install -e .
```

Or with a virtualenv:

```shell
git clone https://github.com/triton-lang/triton.git
cd triton

python -m venv .venv --prompt triton
source .venv/bin/activate

pip install -r python/requirements.txt # build-time dependencies
pip install -e .
```

# Building with a custom LLVM

Triton uses LLVM to generate code for GPUs and CPUs.  Normally, the Triton build
downloads a prebuilt LLVM, but you can also build and use LLVM from source.

LLVM does not have a stable API, so the Triton build will not work at an
arbitrary LLVM version.

For convenience, use the following command to build LLVM and install Triton with the custom LLVM:

```shell
make dev-install-llvm                 # Core LLVM (default)
make dev-install-llvm LLVM_BUILD=amd  # AMD LLVM only
make dev-install-llvm LLVM_BUILD=all
```

<details>
<summary>
Alternatively, follow these steps to build LLVM from source manually.
</summary>

1. Find the version of LLVM that Triton builds against. Check the `llvm_hash`
field in `cmake/llvm-info.json` to see the current version. For example, if it
says:
       "llvm_hash": "49af6502c6dcb4a7f7520178bd14df396f78240c"

   This means that the version of Triton you have builds against
   [LLVM](https://github.com/llvm/llvm-project) 49af6502.

2. `git checkout` LLVM at this revision.  Optionally, make additional
   modifications to LLVM.

3. [Build LLVM](https://llvm.org/docs/CMake.html).  For example, you might run:

       $ cd $HOME/llvm-project  # your clone of LLVM.
       $ mkdir build
       $ cd build
       $ cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld;clang" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"
       $ ninja

4. Grab a snack, this will take a while.

5. Build Triton as above, but set the following environment variables:

       # Modify as appropriate to point to your LLVM build.
       $ export LLVM_BUILD_DIR=$HOME/llvm-project/build

       $ cd <triton install>
       $ LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
         LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
         LLVM_SYSPATH=$LLVM_BUILD_DIR \
         pip install -e .

</details>

# Tips for building

- Set `TRITON_BUILD_WITH_CLANG_LLD=true` as an environment variable to use clang
  and lld.  lld in particular results in faster builds.

- Set `TRITON_BUILD_WITH_CCACHE=true` to build with ccache.

- Set `TRITON_HOME=/some/path` to change the location of the `.triton`
  directory where Triton's cache is located and downloads are stored
  during the build. By default, this is the user's home directory. It
  can be changed anytime.

- If you're running out of memory when building Triton, specify the `MAX_JOBS`
  environment variable (to the `pip install -e .` command) to limit the
  number of jobs.

- Pass `--no-build-isolation` to `pip install` to make nop builds faster.
  Without this, every invocation of `pip install` uses a different symlink to
  cmake, and this forces ninja to rebuild most of the `.a` files.

- The build system creates a `compile_commands.json` file under the Triton repo
  directory. This file is used by VSCode IntelliSense and clangd to provide
  code completion and other features for C++ code.

  If IntelliSense does not work, you can try the following steps:

    - Do a local build. Run command `pip install -e .`.
    - Get the full path to the `compile_commands.json` file produced by the build:
      `find ./build -name 'compile_commands.json' | xargs readlink -f`.
      You might get a full path similar to `/Users/{username}/triton/build/cmake.macosx-11.1-arm64-cpython-3.12/compile_commands.json`.
    - In VSCode, install the
      [C/C++
      extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools),
      then open the command palette (`Shift + Command + P` on Mac, or `Shift +
      Ctrl + P` on Windows/Linux) and open `C/C++: Edit Configurations (UI)`.
    - Open "Advanced Settings" and paste the full path to
      `compile_commands.json` into the "Compile Commands" textbox.

# Running tests

There currently isn't a turnkey way to run all the Triton tests, but you can
follow the following recipe:

```shell
# One-time setup.  Note this will reinstall local Triton because torch
# overwrites it with the public version.
$ make dev-install

# To run all tests (requires a GPU)
$ make test

# Or, to run tests without a gpu
$ make test-nogpu
```

# Tips for hacking

For detailed instructions on how to debug Triton's frontend, please refer to this [tutorial](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html). The following includes additional tips for hacking on Triton's backend.

**Configuration knobs**

See [`python/triton/knobs.py`](python/triton/knobs.py) for the full list of configuration knobs. You can set those knobs directly in python or use environment variables to control them. Below are some of the environment variables you can specify (see `knobs.py` for the full list):

- `MLIR_ENABLE_DUMP=1` dumps the IR before every MLIR pass Triton runs, for all
   kernels. Use `MLIR_ENABLE_DUMP=kernelName` to dump for a specific kernel only.
  - Triton cache can interfere with the dump. In cases where `MLIR_ENABLE_DUMP=1` does not work, try cleaning your triton cache: `rm -r ~/.triton/cache/*`.
- `MLIR_DUMP_PATH` specifies where `MLIR_ENABLE_DUMP` will dump to. If unset will dump to stderr.
- `LLVM_IR_ENABLE_DUMP=1` dumps the IR before every pass run over the LLVM IR.
- `TRITON_REPRODUCER_PATH=<reproducer_path>` will generate an MLIR reproducer file
  at `<reproducer_path>` before each MLIR compiler stage. If any of the stages fail,
  `<reproducer_path>` will be a local MLIR reproducer captured right before the failing pass.
- `TRITON_INTERPRET=1` uses the Triton interpreter instead of running on the
  GPU.  You can insert Python breakpoints in your kernel code!
- `TRITON_ENABLE_LLVM_DEBUG=1` passes `-debug` to LLVM, printing a lot of
  debugging information to stdout.  If this is too noisy, run with just
  `TRITON_LLVM_DEBUG_ONLY` instead to limit the output.
  - An alternative way to reduce output noisiness is running with
  `LLVM_IR_ENABLE_DUMP=1`, extract the IR before the LLVM pass of interest, and
  then run LLVM's `opt` standalone, perhaps passing `-debug-only=foo` on the
  command line.

- `TRITON_LLVM_DEBUG_ONLY=<comma-separated>` is the equivalent of LLVM's
  `-debug-only` command-line option. This limits the LLVM debug output to
  specific pass or component names (which are specified using `#define
  DEBUG_TYPE` throughout LLVM and Triton) in order to allow the debug output to
  be less noisy. `TRITON_LLVM_DEBUG_ONLY` allows for one or more comma
  separated values to be specified (eg
  `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions"` or
  `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions,regalloc"`).
- `TRITON_ENABLE_ASAN=1` invokes the LLVM address sanitizer for
  memory leak and out of bounds access detection. Currently only supported on the AMD
  backend. This must be run using the ASAN libraries documented [here](https://rocm.docs.amd.com/projects/llvm-project/en/latest/conceptual/using-gpu-sanitizer.html).
  - When enabling the address sanitizer it is recommended to disable various memory caching strategies
  both within the ROCm stack and PyTorch. This will give the address sanitizer the best chance at finding the
  memory fault where it originates. See this [test](https://github.com/triton-lang/triton/blob/main/third_party/amd/python/test/test_address_sanitizer.py) for more details.

- `USE_IR_LOC={ttir,ttgir}` reparses the IR such that the location information
  will be the line number of the IR file with that particular extension,
  instead of line number of the python file. This can provide a direct mapping
  from the IR to llir/ptx. When used with performance tools, it can provide a
  breakdown on IR instructions.
- `TRITON_PRINT_AUTOTUNING=1` prints out the best autotuning config and total time
  spent for each kernel after autotuning is complete.
- `DISABLE_LLVM_OPT` will disable llvm optimizations for make_llir and make_ptx
  if its value is true when parsing as Bool. Otherwise, it will be parsed as a list
  of flags to disable llvm optimizations. One usage case is
  `DISABLE_LLVM_OPT="disable-lsr"`
  Loop strength reduction is known to cause up to 10% performance changes for
  certain kernels with register pressure.
- `TRITON_ALWAYS_COMPILE=1` forces to compile kernels regardless of cache hit.
- `MLIR_ENABLE_TIMING` dumps the timing information for each MLIR pass.
- `LLVM_ENABLE_TIMING` dumps the timing information for each LLVM pass.
- `TRITON_DEFAULT_FP_FUSION` overrides the default behavior of disabling fp fusion (mul+add->fma).
- `TRITON_ALLOW_FP_FUSION=1` allows kernels to control implicit fp fusion through
  `enable_fp_fusion`. Without this opt-in, implicit fusion is disabled. Explicit
  fused operations such as `tl.fma` are unchanged.
- `MLIR_ENABLE_DIAGNOSTICS=<comma-separated>` controls diagnostic emission in MLIR.
  Options are: `warnings`, `remarks`, `stacktraces`, `operations`.
  Use comma-separated values to customize output. For example,
  `MLIR_ENABLE_DIAGNOSTICS=remarks,operations` enables remarks and IR operations,
  while `MLIR_ENABLE_DIAGNOSTICS=warnings,stacktraces` enables warnings with
  stacktraces. By default, only errors are shown. Setting `warnings` includes
  errors and warnings; `remarks` includes errors, warnings, and remarks.
- `MLIR_ENABLE_REMARK` is deprecated. Please use `MLIR_ENABLE_DIAGNOSTICS=remarks`.
- `TRITON_KERNEL_DUMP` enables the dumping of the IR from each compilation stage and the final ptx/amdgcn.
- `TRITON_DUMP_DIR` specifies the directory to save the dumped IR and ptx/amdgcn when `TRITON_KERNEL_DUMP` is set to 1.
- `TRITON_KERNEL_OVERRIDE` enables the override of the compiled kernel with a user-specified IR/ptx/amdgcn at the beginning of each compilation stage.
- `TRITON_OVERRIDE_DIR` specifies the directory from which to load the IR/ptx/amdgcn files when `TRITON_KERNEL_OVERRIDE` is set to 1.
- `TRITON_F32_DEFAULT` sets the default input precision of `tl.dot` when using 32-bit floats, which can be either `ieee`, `tf32`, or `tf32x3`.
- `TRITON_FRONT_END_DEBUGGING=1` disables exception wrapping when an error occurs in the compiler frontend, allowing the full stack trace to be seen.
- `TRITON_DISABLE_LINE_INFO=1` removes all line information from the module.
- `PTXAS_OPTIONS` passes additional command-line options to the PTX assembler `ptxas` (only on NVIDIA).
- `LLVM_EXTRACT_DI_LOCAL_VARIABLES` emit full debug info, allowing for eval of values in gpu debuggers (ie cuda-gdb, rocm-gdb etc)
- `TRITON_DEFAULT_BACKEND=<backend>` optionally sets the default backend used by Triton when
  constructing the active driver (i.e., `triton.runtime.driver.active`).

> [!NOTE]
> Some of these environment variables don't have a knob in `knobs.py`-- those are only relevant to the C++ layer(s), hence they don't exist in the python layer.

**Kernel Override Steps**

```bash
export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=<dump_dir>
export TRITON_KERNEL_OVERRIDE=1
export TRITON_OVERRIDE_DIR=<override_dir>
# Step 1: Run the kernel once to dump kernel's IRs and ptx/amdgcn in $TRITON_DUMP_DIR
# Step 2: Copy $TRITON_DUMP_DIR/<kernel_hash> to $TRITON_OVERRIDE_DIR
# Step 3: Delete the stages that you do not want to override and modify the stage you do want to override
# Step 4: Run the kernel again to see the overridden result
```

**Compiler Pipeline Inspection Steps**
To introspect the pipeline `add_stages`, before running your kernels, simply set
the add_stages_inspection_hook like so:

```python
def inspect_stages(_self, stages, options, language, capability):
    # inspect or modify add_stages here
triton.knobs.runtime.add_stages_inspection_hook = inspect_stages
```
Examples of how to use this for out of tree plugin passes is [here](examples/plugins/README.md)
# Compatibility

Supported Platforms:

- Linux

Supported Hardware:

- NVIDIA GPUs (Compute Capability 8.0+)
- AMD GPUs (ROCm 6.2+)
- Under development: CPUs


---

## Upstream license notice

```text
/*
* Copyright 2018-2020 Philippe Tillet
* Copyright 2020-2022 OpenAI
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
```
