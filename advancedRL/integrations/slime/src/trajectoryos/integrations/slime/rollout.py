"""Agentic rollout: task → sandboxed episode → verified, reward-scored sample.

This is the engine behind the slime custom generate function. It is pure
composition of tested components (episode loop, clean verifier, composite
reward, sample conversion), so it can be smoke-tested end-to-end with mocked
inference and exercised unchanged on a GPU cluster.
"""

from collections.abc import Callable

from trajectoryos.agents import Policy, run_episode
from trajectoryos.environments import Sandbox
from trajectoryos.integrations.slime.sample import SlimeSampleData, trajectory_to_sample_data
from trajectoryos.rewards import RewardInputs, RewardWeights, compute_reward, normalize_costs
from trajectoryos.schemas import TaskSpec
from trajectoryos.verifiers import CleanSandboxTestVerifier


def run_agentic_rollout(
    task: TaskSpec,
    policy: Policy,
    *,
    sandbox_factory: Callable[[], Sandbox],
    test_command: str,
    reward_weights: RewardWeights,
    run_id: str,
    policy_version: str,
    rollout_id: str | None = None,
    max_steps: int = 50,
) -> SlimeSampleData:
    """Run one episode, verify in a clean sandbox, compose the reward, emit a sample."""
    work_sandbox = sandbox_factory()
    try:
        episode = run_episode(
            task,
            policy,
            work_sandbox,
            run_id=run_id,
            policy_version=policy_version,
            rollout_id=rollout_id,
            max_steps=max_steps,
        )
    finally:
        work_sandbox.close()

    verifier = CleanSandboxTestVerifier(
        sandbox_factory=sandbox_factory,
        test_command=test_command,
    )
    verifier_result = verifier.verify(episode.patch)

    trajectory = episode.trajectory
    costs = normalize_costs(trajectory.cost_summary, task.budget)
    attempted_test_tampering = bool(verifier_result.details.get("rejected_paths"))
    reward = compute_reward(
        RewardInputs(
            task_success=1.0 if verifier_result.passed else 0.0,
            format_compliance=1.0,  # refined per-step in Milestone 4 metric logging
            token_cost=costs.token_cost,
            tool_cost=costs.tool_cost,
            latency_cost=costs.latency_cost,
            gpu_cost=costs.gpu_cost,
            safety_penalty=1.0 if attempted_test_tampering else 0.0,
        ),
        reward_weights,
    )

    scored = trajectory.model_copy(
        update={
            "verifier_result": verifier_result,
            "reward_components": reward.model_dump(exclude={"total_reward"}),
            "total_reward": reward.total_reward,
        }
    )
    sample = trajectory_to_sample_data(scored)
    sample.reward = reward.total_reward
    sample.metadata["reward_components"] = reward.model_dump(exclude={"total_reward"})
    sample.metadata["verifier"] = verifier_result.model_dump(exclude={"logs"})
    return sample
