#!/usr/bin/env python3
"""Simplified compile script: force-compile root and kernels CUDA/C++ sources."""

import argparse
import importlib
import shutil
import sys
import time
from pathlib import Path

from utils.backend import BACKEND_CHOICES, DEVICE, detect_backend, get_spec


def find_sources() -> list[str]:
    root = Path('.')
    kernels_dir = Path('kernels')

    root_sources = [str(p) for p in root.glob('*.cu')] + [str(p) for p in root.glob('*.cpp')]
    kernel_sources = []
    if kernels_dir.is_dir():
        kernel_sources = [str(p) for p in kernels_dir.glob('*.cu')] + [str(p) for p in kernels_dir.glob('*.cpp')]

    return sorted(set(root_sources + kernel_sources))


def compile_kernels() -> int:
    build_dir = Path('build/forced_compile')
    output_so = Path('cuda_extension.so')
    sources = find_sources()

    if not sources:
        print('Error: no source files found (*.cu, *.cpp in root or kernels/)')
        return 1

    print(f'Compiling {len(sources)} files: {", ".join(sources)}')

    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    if output_so.exists():
        output_so.unlink()

    try:
        import torch.utils.cpp_extension as cpp_ext

        cpp_ext.load(
            name='cuda_extension',
            sources=sources,
            build_directory=str(build_dir),
            verbose=False,
            with_cuda=True,
            # Let Torch select its required standard (C++20 in Torch 2.14).
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
        )
    except Exception as exc:
        print('Compilation failed.')
        print(str(exc))
        return 1

    built_so = build_dir / 'cuda_extension.so'
    if built_so.exists():
        shutil.copy2(built_so, output_so)
        print(f'Compile success: {output_so}')
        return 0

    print('Compilation finished but cuda_extension.so was not generated.')
    return 1


def compile_jit(spec) -> int:
    """JIT backends have no .so: import the kernel package and run one forward
    so compilation and autotuning happen here, not inside verification."""
    import torch
    from utils.verification import build_inputs, initialize_models

    importlib.import_module(spec.kernel_package)
    _, model = initialize_models()
    _, inputs = build_inputs()
    with torch.no_grad():
        model(*inputs)
    if torch.device(DEVICE).type == 'cuda':
        torch.cuda.synchronize()
    print(f'Compile success: {spec.display} JIT/autotune (no .so)')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=BACKEND_CHOICES, default='auto')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    start = time.monotonic()
    try:
        import torch
        torch.manual_seed(args.seed)
        spec = get_spec(detect_backend(Path.cwd()) if args.backend == 'auto' else args.backend)
        return compile_jit(spec) if spec.compile_mode == 'jit' else compile_kernels()
    except Exception as exc:
        print('Compilation failed.')
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    finally:
        print(f'[TIME] Compilation took {time.monotonic() - start:.2f}s')


if __name__ == '__main__':
    sys.exit(main())
