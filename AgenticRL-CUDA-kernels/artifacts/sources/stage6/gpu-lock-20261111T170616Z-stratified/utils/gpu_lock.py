"""Torch-free, process-owned advisory locks for GPU measurement stages.

All cooperating workers must share CUDA_AGENT_GPU_LOCK_DIR (default: a shared
temporary directory). Files are never unlinked: replacing an inode would let
two workers lock the same device independently. flock is released by the OS
on close or process death; child stage executables do not inherit the fd.
"""

import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def _has_cuda_driver():
    # GPU containers and WSL can expose CUDA without shipping nvidia-smi.
    return any(Path(name).exists() for name in ('/dev/nvidiactl', '/dev/nvidia0', '/dev/dxg'))


def device_identity(env):
    """Resolve the child's logical CUDA ordinal through CUDA_VISIBLE_DEVICES."""
    device = env.get("CUDA_AGENT_DEVICE", "cuda")
    visible = env.get("CUDA_VISIBLE_DEVICES")
    info = {"meta/gpu_uuid": None, "meta/gpu_name": None,
            "meta/gpu_uuid_reason": None, "meta/gpu_lock_key": None,
            "meta/gpu_lock": "none_cpu", "meta/gpu_lock_wait_s": 0.0}
    if device.startswith("cpu") or visible is not None and visible.strip() in ("", "-1"):
        info["meta/gpu_uuid_reason"] = "CPU requested or CUDA devices hidden"
        return info
    ordinal = int(device.split(":", 1)[1]) if ":" in device else 0
    tokens = [token.strip() for token in visible.split(",")] if visible is not None else None
    selected = tokens[ordinal] if tokens is not None and 0 <= ordinal < len(tokens) else str(ordinal)
    binary = shutil.which("nvidia-smi", path=env.get("PATH"))
    if binary is None and "CUDA_AGENT_DEVICE" not in env and visible is None and not _has_cuda_driver():
        info["meta/gpu_uuid_reason"] = "No CUDA driver device, nvidia-smi or explicit CUDA selection"
        return info
    try:
        if tokens is not None and (ordinal >= len(tokens) or ordinal < 0):
            raise ValueError("CUDA ordinal is outside CUDA_VISIBLE_DEVICES")
        if binary is None:
            raise FileNotFoundError("nvidia-smi unavailable")
        query = subprocess.run(
            [binary, "--query-gpu=uuid,name", "--format=csv,noheader", "-i", selected],
            env=env, capture_output=True, text=True, timeout=5, check=True)
        rows = [row for row in query.stdout.splitlines() if row.strip()]
        if len(rows) != 1:
            raise ValueError("nvidia-smi did not return exactly one device")
        gpu_uuid, name = (value.strip() for value in rows[0].split(",", 1))
        if not gpu_uuid.startswith(("GPU-", "MIG-")) or not name:
            raise ValueError("nvidia-smi returned an invalid UUID/name")
        info.update({"meta/gpu_uuid": gpu_uuid, "meta/gpu_name": name,
                     "meta/gpu_lock_key": gpu_uuid, "meta/gpu_lock": "held"})
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        # A visible UUID is already a usable identity even when the utility
        # fails, but it remains explicitly unverified in gpu_uuid metadata.
        key = selected if selected.startswith(("GPU-", "MIG-")) else "visible-" + hashlib.sha256(
            selected.encode()).hexdigest()
        info.update({"meta/gpu_lock_key": key, "meta/gpu_lock": "unknown_uuid",
                     "meta/gpu_uuid_reason": f"{type(exc).__name__}: {exc}"[:512]})
    return info


class MeasurementLock:
    """Nonblocking polling, a separate wait cap, and exactly zero free-lock wait."""

    def __init__(self, env, enabled=True, timeout_s=600.0):
        self.env, self.enabled, self.timeout_s = env, enabled, timeout_s
        self.metadata = None
        self.acquired = False
        self.file = None

    def __enter__(self):
        self.metadata = device_identity(self.env)
        if not self.enabled:
            self.metadata["meta/gpu_lock"] = "disabled"
            return self
        if self.metadata["meta/gpu_lock_key"] is None:
            return self
        directory = Path(self.env.get("CUDA_AGENT_GPU_LOCK_DIR") or
                         str(Path(tempfile.gettempdir()) / "agenticrl-gpu-locks"))
        directory.mkdir(parents=True, exist_ok=True)
        key = self.metadata["meta/gpu_lock_key"]
        filename = key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else hashlib.sha256(key.encode()).hexdigest()
        self.file = (directory / (filename + ".lock")).open("a+")
        started = time.monotonic()
        blocked = False
        try:
            while True:
                if blocked and time.monotonic() - started >= self.timeout_s:
                    self.metadata["meta/gpu_lock_wait_s"] = time.monotonic() - started
                    self.metadata["meta/gpu_lock"] = "gpu_busy"
                    return self
                try:
                    fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.acquired = True
                    if blocked:
                        self.metadata["meta/gpu_lock_wait_s"] = time.monotonic() - started
                    return self
                except BlockingIOError:
                    blocked = True
                    elapsed = time.monotonic() - started
                    self.metadata["meta/gpu_lock_wait_s"] = elapsed
                    if elapsed >= self.timeout_s:
                        self.metadata["meta/gpu_lock"] = "gpu_busy"
                        return self
                    time.sleep(min(.05, self.timeout_s - elapsed))
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        if self.file is not None:
            self.file.close()
            self.file = None
        self.acquired = False

    @property
    def busy(self):
        return self.metadata["meta/gpu_lock"] == "gpu_busy"
