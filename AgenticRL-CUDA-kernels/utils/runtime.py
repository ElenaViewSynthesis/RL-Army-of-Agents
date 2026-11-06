"""Torch-free GPU runtime preflight. Reports versions and availability; never fabricates."""

import argparse
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys


def version_tuple(s):
    """Leading dotted integers of a version string; () when unparsable."""
    match = re.match(r"\d+(?:\.\d+)*", s or "")
    return tuple(int(x) for x in match.group(0).split(".")) if match else ()


def parse_nvidia_smi(text):
    """Rows of the csv,noheader,nounits query into flat dicts; malformed lines skipped."""
    gpus = []
    for line in (text or "").splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 5:
            continue
        try:
            index = int(fields[0])
        except ValueError:
            continue
        gpus.append({"index": index, "name": fields[1], "driver_version": fields[2],
                     "compute_cap": fields[3], "uuid": fields[4]})
    return gpus


def parse_nvcc_version(text):
    match = re.search(r"release (\d+\.\d+)", text or "")
    return match.group(1) if match else None


def tool_output(cmd, timeout_s=20):
    """stdout of cmd on a zero exit; None when the tool is absent or fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def package_version(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _at_least(version, minimum):
    found = version_tuple(version)
    return bool(found) and found >= minimum


def check(run=tool_output, which=shutil.which, pkg=package_version,
          python_version=sys.version_info):
    """Flat availability record; every value is a scalar or None."""
    gpus = parse_nvidia_smi(run(["nvidia-smi",
                                 "--query-gpu=index,name,driver_version,compute_cap,gpu_uuid",
                                 "--format=csv,noheader,nounits"]))
    gpu0 = gpus[0] if gpus else {}
    driver_version = gpu0.get("driver_version")
    driver_ok = _at_least(driver_version, (580,))
    cc = version_tuple(gpu0.get("compute_cap"))
    cc_major = cc[0] if cc else 0
    cutile_supported = 8 <= cc_major <= 12
    nvcc_version = parse_nvcc_version(run(["nvcc", "--version"]))
    nvcc_ok = _at_least(nvcc_version, (13, 1))
    tileiras = pkg("nvidia-cuda-tileiras") or ("on PATH" if which("tileiras") else None)
    if tileiras is None:
        tileiras_ok, tileiras_reason = False, "tileiras missing"
    elif version_tuple(tileiras):
        minimum = (13, 3) if cc_major == 9 else (13, 1)
        tileiras_ok = _at_least(tileiras, minimum)
        tileiras_reason = (None if tileiras_ok
                           else f"tileiras {tileiras} < {'.'.join(map(str, minimum))}")
    else:
        tileiras_ok, tileiras_reason = False, "tileiras version unknown"
    ninja = which("ninja")
    torch_version = pkg("torch")
    triton_version = pkg("triton")
    cuda_tile_version = pkg("cuda-tile")
    tilelang_version = pkg("tilelang")
    cutlass_dsl_version = pkg("nvidia-cutlass-dsl")
    cuda_bindings_version = pkg("cuda-bindings")
    python_ok = tuple(python_version[:2]) >= (3, 11)
    ok_cuda = driver_ok and nvcc_ok and ninja is not None and torch_version is not None
    ok_triton = driver_ok and torch_version is not None and triton_version is not None
    ok_cutile = (python_ok and driver_ok and tileiras_ok
                 and cuda_tile_version is not None and cutile_supported)
    ok_tilelang = driver_ok and torch_version is not None and tilelang_version is not None
    ok_cutedsl = (driver_ok and torch_version is not None
                  and cutlass_dsl_version is not None
                  and cuda_bindings_version is not None and cc_major >= 8)
    reasons = []
    if not gpus:
        reasons.append("no GPU (nvidia-smi unavailable)")
    elif not driver_ok:
        reasons.append(f"driver {driver_version} < 580")
    if not python_ok:
        reasons.append(f"python {python_version.major}.{python_version.minor} < 3.11")
    if not nvcc_ok:
        reasons.append("nvcc missing" if nvcc_version is None else f"nvcc {nvcc_version} < 13.1")
    if ninja is None:
        reasons.append("ninja missing")
    if torch_version is None:
        reasons.append("torch missing")
    if triton_version is None:
        reasons.append("triton missing")
    if tilelang_version is None:
        reasons.append("tilelang missing")
    if not tileiras_ok:
        reasons.append(tileiras_reason)
    if cuda_tile_version is None:
        reasons.append("cuda-tile missing")
    if cutlass_dsl_version is None:
        reasons.append("nvidia-cutlass-dsl missing")
    if cuda_bindings_version is None:
        reasons.append("cuda-bindings missing")
    if gpus and not cutile_supported:
        reasons.append(f"gpu0 compute capability {gpu0.get('compute_cap')} outside 8.x-12.x")
    return {"runtime/python": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            "runtime/python_ok": python_ok,
            "runtime/gpu_count": len(gpus), "runtime/gpu0_name": gpu0.get("name"),
            "runtime/gpu0_compute_cap": gpu0.get("compute_cap"),
            "runtime/gpu0_uuid": gpu0.get("uuid"), "runtime/driver_version": driver_version,
            "runtime/driver_ok": driver_ok, "runtime/gpu0_cutile_supported": cutile_supported,
            "runtime/nvcc_version": nvcc_version, "runtime/nvcc_ok": nvcc_ok,
            "runtime/ninja": ninja, "runtime/tileiras_version": tileiras,
            "runtime/tileiras_ok": tileiras_ok, "runtime/tileiras_reason": tileiras_reason,
            "runtime/cuda_tile_version": cuda_tile_version,
            "runtime/torch_version": torch_version,
            "runtime/torch_cuda_build": "+cu" in (torch_version or ""),
            "runtime/triton_version": triton_version,
            "runtime/tilelang_version": tilelang_version,
            "runtime/cutlass_dsl_version": cutlass_dsl_version,
            "runtime/cuda_bindings_version": cuda_bindings_version,
            "runtime/nsys": which("nsys"), "runtime/ncu": which("ncu"),
            "runtime/compute_sanitizer": which("compute-sanitizer"),
            "runtime/ok_cuda_backend": ok_cuda, "runtime/ok_triton_backend": ok_triton,
            "runtime/ok_cutile": ok_cutile, "runtime/ok_tilelang": ok_tilelang,
            "runtime/ok_cutedsl": ok_cutedsl,
            "runtime/unavailable_reason": None if all(
                (ok_cuda, ok_triton, ok_cutile, ok_tilelang, ok_cutedsl))
            else "; ".join(reasons)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the flat JSON record")
    parser.add_argument("--require", type=str,
                        help="Comma-separated profiles to require: cuda,triton,cutile,tilelang,cutedsl")
    args = parser.parse_args()
    record = check()
    profiles = {"cuda": "runtime/ok_cuda_backend", "triton": "runtime/ok_triton_backend",
                "cutile": "runtime/ok_cutile", "tilelang": "runtime/ok_tilelang",
                "cutedsl": "runtime/ok_cutedsl"}
    if args.require:
        missing = [name for name in args.require.split(",") if name not in profiles]
        if missing:
            parser.error(f"unknown profiles: {missing}")
        if any(not record[profiles[name]] for name in args.require.split(",")):
            print(record["runtime/unavailable_reason"] or "unavailable", file=sys.stderr)
            sys.exit(1)
    if args.json:
        print(json.dumps(record, allow_nan=False))
    else:
        for key, value in record.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
