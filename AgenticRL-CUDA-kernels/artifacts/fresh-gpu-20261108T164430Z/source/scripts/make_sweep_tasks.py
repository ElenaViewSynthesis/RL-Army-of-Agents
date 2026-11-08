"""Create one isolated task workdir per sweep agent (one per GPU).

    python -m scripts.make_sweep_tasks --backend triton --gpus 0 1 [--root sweep_tasks]

Writes <root>/gpu<N>/<backend>/ from the honest template for that backend via
tests.support.make_task, so concurrent agents never share build or JIT caches.
Prints one `export` line per agent to paste into that agent's terminal.
"""

import argparse
import shutil
from pathlib import Path

from tests.support import ROOT, make_task
from utils.backend import BACKENDS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="triton")
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--root", default="sweep_tasks")
    parser.add_argument("--force", action="store_true", help="Replace existing workdirs")
    args = parser.parse_args(argv)
    root = (ROOT / args.root).resolve()
    for gpu in args.gpus:
        parent = root / f"gpu{gpu}"
        if parent.exists():
            if not args.force:
                raise SystemExit(f"{parent} exists; pass --force to replace it")
            shutil.rmtree(parent)
        parent.mkdir(parents=True)
        task = make_task(parent, args.backend)
        print(f"export CUDA_VISIBLE_DEVICES={gpu} SWEEP_WORKDIR={task}")


if __name__ == "__main__":
    main()
