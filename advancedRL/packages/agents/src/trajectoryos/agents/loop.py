"""Episode loop: policy ↔ tools ↔ sandbox, with every action recorded.

Records each step as ``TrajectoryEvent``s (policy output → tool call → tool
result), enforces token / per-tool / wallclock budgets before and after every
step, and accumulates the patch (modified files) for the clean-sandbox
verifier. Nothing is discarded on failure: budget violations and tool errors
terminate the episode with an explicit state and reason.
"""

import time
import uuid

from pydantic import BaseModel, ConfigDict, Field
from trajectoryos.agents.budget import BudgetTracker
from trajectoryos.agents.policy import Policy
from trajectoryos.agents.tools import execute_tool
from trajectoryos.core import get_logger
from trajectoryos.environments import Sandbox
from trajectoryos.schemas import (
    EventType,
    Role,
    TaskSpec,
    TerminalState,
    Trajectory,
    TrajectoryEvent,
)

logger = get_logger(__name__)


class EpisodeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trajectory: Trajectory
    patch: dict[str, str] = Field(
        default_factory=dict,
        description="path -> full new content for every file the agent modified",
    )


def run_episode(
    task: TaskSpec,
    policy: Policy,
    sandbox: Sandbox,
    *,
    run_id: str,
    policy_version: str,
    rollout_id: str | None = None,
    max_steps: int = 50,
) -> EpisodeResult:
    rollout_id = rollout_id or str(uuid.uuid4())
    tracker = BudgetTracker(task.budget)
    events: list[TrajectoryEvent] = [
        TrajectoryEvent(
            event_type=EventType.USER_MESSAGE,
            role=Role.USER,
            content=task.prompt,
        )
    ]
    patch: dict[str, str] = {}
    terminal_state = TerminalState.TRUNCATED
    termination_reason: str | None = f"max_steps ({max_steps}) reached"

    log = logger.bind(run_id=run_id, rollout_id=rollout_id, task_id=task.task_id)

    for _step in range(max_steps):
        turn = policy.next_turn(events)
        tracker.add_tokens(
            input_tokens=turn.input_tokens_used, output_tokens=turn.output_tokens_used
        )
        policy_event = TrajectoryEvent(
            event_type=EventType.POLICY_OUTPUT,
            role=Role.ASSISTANT,
            content=turn.content,
            token_ids=turn.token_ids,
            rollout_logprobs=turn.rollout_logprobs,
            loss_mask=turn.loss_mask,
        )
        events.append(policy_event)

        if (reason := tracker.violation()) is not None:
            events.append(_budget_event(reason, policy_event.event_id))
            terminal_state, termination_reason = TerminalState.BUDGET_EXCEEDED, reason
            break

        if turn.tool_call is None:
            terminal_state, termination_reason = TerminalState.COMPLETED, None
            break

        call = turn.tool_call
        tracker.record_tool_call(call.tool_name)
        call_event = TrajectoryEvent(
            event_type=EventType.TOOL_CALL,
            role=Role.ASSISTANT,
            parent_event_id=policy_event.event_id,
            tool_name=call.tool_name,
            tool_arguments=call.arguments,
        )
        events.append(call_event)

        if (reason := tracker.violation()) is not None:
            events.append(_budget_event(reason, call_event.event_id))
            terminal_state, termination_reason = TerminalState.BUDGET_EXCEEDED, reason
            break

        started = time.monotonic()
        result = execute_tool(sandbox, call.tool_name, call.arguments)
        latency_ms = (time.monotonic() - started) * 1000
        if result.ok and result.modified_path is not None and result.modified_content is not None:
            patch[result.modified_path] = result.modified_content
        events.append(
            TrajectoryEvent(
                event_type=EventType.TOOL_RESULT,
                role=Role.TOOL,
                parent_event_id=call_event.event_id,
                tool_name=call.tool_name,
                tool_result=result.output,
                latency_ms=latency_ms,
                metadata={"ok": result.ok},
            )
        )
        log.debug("tool_executed", tool=call.tool_name, ok=result.ok, latency_ms=latency_ms)

        if (reason := tracker.violation()) is not None:
            events.append(_budget_event(reason, call_event.event_id))
            terminal_state, termination_reason = TerminalState.BUDGET_EXCEEDED, reason
            break

    trajectory = Trajectory(
        run_id=run_id,
        rollout_id=rollout_id,
        task_id=task.task_id,
        policy_version=policy_version,
        events=events,
        terminal_state=terminal_state,
        termination_reason=termination_reason,
        cost_summary=tracker.cost_summary(),
    )
    log.info(
        "episode_finished",
        terminal_state=terminal_state.value,
        termination_reason=termination_reason,
        events=len(events),
        modified_files=len(patch),
    )
    return EpisodeResult(trajectory=trajectory, patch=patch)


def _budget_event(reason: str, parent_event_id: str) -> TrajectoryEvent:
    return TrajectoryEvent(
        event_type=EventType.BUDGET_VIOLATION,
        role=Role.ENVIRONMENT,
        parent_event_id=parent_event_id,
        content=reason,
    )
