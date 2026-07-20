"""slime entry point: --custom-rm-path

Pass to slime as:
    --custom-rm-path trajectoryos.integrations.slime.reward.reward_func

The agentic generate function already verifies and scores each rollout
(clean-sandbox verification + composite cost-aware reward) and stores the
result on the sample; this RM simply surfaces that scalar to slime. Keeping
reward computation inside the rollout keeps verification and scoring atomic —
slime never re-scores text.
"""

from typing import Any


async def reward_func(args: Any, sample: Any, **kwargs: Any) -> float:
    if sample.reward is None:
        raise ValueError(
            f"sample {getattr(sample, 'index', '?')} reached the RM without a reward; "
            "agentic rollouts must be scored by run_agentic_rollout"
        )
    return float(sample.reward)
