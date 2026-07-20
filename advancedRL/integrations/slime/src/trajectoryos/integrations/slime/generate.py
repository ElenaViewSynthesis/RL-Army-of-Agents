"""slime entry point: --custom-generate-function-path

Pass to slime as:
    --custom-generate-function-path trajectoryos.integrations.slime.generate.generate

Runs the full agentic rollout (sandbox episode → clean verification → composite
reward) against the SGLang router slime manages, and returns a Sample carrying
the exact sampled token IDs, rollout logprobs and loss mask.

This module imports slime lazily: it is only importable *as a slime plugin*
inside a slime training environment. Its logic lives in tested, slime-free
components (`run_agentic_rollout`); this file is a thin adapter.
"""

import json
from typing import Any

from trajectoryos.environments import DockerSandbox
from trajectoryos.integrations.sglang import SGLangClient, SGLangToolPolicy
from trajectoryos.integrations.slime.rollout import run_agentic_rollout
from trajectoryos.integrations.slime.sample import to_slime_sample
from trajectoryos.rewards import RewardWeights
from trajectoryos.schemas import TaskSpec


async def generate(args: Any, sample: Any, sampling_params: dict[str, Any]) -> Any:
    """slime custom generate function (agentic rollout).

    Expects each dataset row's ``metadata`` to contain a serialized TaskSpec
    under ``task_spec`` plus ``source_dir`` / ``test_command`` for the sandbox
    and verifier (produced by our data source; see Milestone 4 configs).
    """
    from transformers import AutoTokenizer  # available in slime environments

    metadata = sample.metadata or {}
    task = TaskSpec.model_validate(
        metadata["task_spec"]
        if isinstance(metadata["task_spec"], dict)
        else json.loads(metadata["task_spec"])
    )
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    client = SGLangClient(f"http://{args.sglang_router_ip}:{args.sglang_router_port}")
    env_config = task.environment.config
    policy = SGLangToolPolicy(client, tokenizer, task, sampling_params=sampling_params)

    sample_data = run_agentic_rollout(
        task,
        policy,
        sandbox_factory=lambda: DockerSandbox(
            metadata["source_dir"],
            image=str(env_config.get("image", "python:3.12-slim")),
            network=str(env_config.get("network", "none")),
        ),
        test_command=str(metadata["test_command"]),
        reward_weights=RewardWeights.model_validate(metadata.get("reward_weights", {})),
        run_id=str(getattr(args, "run_id", "slime-run")),
        policy_version=str(getattr(args, "policy_version", args.hf_checkpoint)),
    )

    result = to_slime_sample(sample_data)
    result.index = sample.index
    return result
