"""Canonical typed schemas for TrajectoryOS.

These models are the single source of truth for data exchanged between
environments, agents, verifiers, rewards, the trajectory store and the slime
integration. All models forbid unknown fields so config/data typos fail loudly.
"""

from trajectoryos.schemas.budget import BudgetSpec
from trajectoryos.schemas.events import EventType, Role, TrajectoryEvent
from trajectoryos.schemas.model import ModelSpec
from trajectoryos.schemas.reward import RewardResult
from trajectoryos.schemas.task import EnvironmentRef, TaskSpec, ToolSpec, VerifierRef
from trajectoryos.schemas.trajectory import (
    CostSummary,
    TerminalState,
    Trajectory,
    VerifierResult,
)

__all__ = [
    "BudgetSpec",
    "CostSummary",
    "EnvironmentRef",
    "EventType",
    "ModelSpec",
    "RewardResult",
    "Role",
    "TaskSpec",
    "TerminalState",
    "ToolSpec",
    "Trajectory",
    "TrajectoryEvent",
    "VerifierRef",
    "VerifierResult",
]
