"""Task specification: what an agent is asked to do and under which constraints."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from trajectoryos.schemas.budget import BudgetSpec


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="JSON-schema of the tool arguments."
    )


class EnvironmentRef(BaseModel):
    """Reference to a registered environment adapter plus its instantiation config."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Registry key, e.g. 'docker_repo_sandbox'.")
    config: dict[str, Any] = Field(default_factory=dict)


class VerifierRef(BaseModel):
    """Reference to a registered verifier plus its instantiation config."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Registry key, e.g. 'clean_sandbox_unit_tests'.")
    config: dict[str, Any] = Field(default_factory=dict)


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    prompt: str
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Task inputs beyond the prompt: issue text, repo ref, file hints, ...",
    )
    tools: list[ToolSpec] = Field(default_factory=list)
    environment: EnvironmentRef
    verifier: VerifierRef
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    metadata: dict[str, Any] = Field(default_factory=dict)
