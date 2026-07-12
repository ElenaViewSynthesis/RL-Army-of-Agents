"""Sandbox environments for TrajectoryOS agents."""

from trajectoryos.environments.base import ExecResult, Sandbox, SandboxError
from trajectoryos.environments.docker import DockerSandbox
from trajectoryos.environments.local import LocalProcessSandbox

__all__ = [
    "DockerSandbox",
    "ExecResult",
    "LocalProcessSandbox",
    "Sandbox",
    "SandboxError",
]
