"""End-to-end agentic rollout against mocked inference (Milestone 3 smoke test).

Drives the real slime custom-generate path — ``SGLangToolPolicy`` →
``run_episode`` (real ``LocalProcessSandbox`` + tools) → clean-sandbox
verification → composite reward → ``SlimeSampleData`` — with the only mock
being the SGLang engine itself (an ``httpx.MockTransport`` returning scripted
completions). No GPU, no server, no slime install required.

What it pins:

- The exact token IDs the mock engine "sampled" survive verbatim into
  ``sample.tokens`` with ``loss_mask=1``; ingested context (template + tool
  observations) lands in the response region with ``loss_mask=0``.
- Every trainable token keeps its real rollout logprob (conversion is a hard
  error otherwise), and the arrays stay aligned 1:1.
- A rollout that actually fixes the bug verifies in a *clean* sandbox and earns
  reward 1.0 with status COMPLETED — scoring is atomic with generation.
"""

import json
import sys
from pathlib import Path

import httpx
from trajectoryos.integrations.sglang import SGLangClient, SGLangToolPolicy
from trajectoryos.integrations.slime import run_agentic_rollout
from trajectoryos.rewards import RewardWeights
from trajectoryos.schemas import (
    BudgetSpec,
    EnvironmentRef,
    TaskSpec,
    VerifierRef,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "buggy_calculator"
TEST_COMMAND = f'"{sys.executable}" test_calculator.py'


def _tool_call(name: str, arguments: dict[str, object]) -> str:
    return f"<tool_call>{json.dumps({'tool_name': name, 'arguments': arguments})}</tool_call>"


# The scripted "sampled" completions, in order: read → fix → run tests → submit.
# Each carries a distinct block of token IDs so we can assert they survive verbatim.
_SCRIPT: list[tuple[str, list[int]]] = [
    (_tool_call("read_file", {"path": "calculator.py"}), [1001, 1002]),
    (
        _tool_call(
            "edit_file",
            {
                "path": "calculator.py",
                "old_string": "return a + b  # BUG: should be a - b",
                "new_string": "return a - b",
            },
        ),
        [2001, 2002, 2003],
    ),
    (_tool_call("run_shell", {"command": TEST_COMMAND}), [3001]),
    ("The subtract bug is fixed and tests pass; submitting.", [4001, 4002]),
]


class _FakeTokenizer:
    """Deterministic byte-level tokenizer — no HF/transformers dependency.

    ``decode`` is unused by the policy (it parses the engine's decoded ``text``),
    but is provided to satisfy the ``Tokenizer`` protocol.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        return bytes(t % 256 for t in token_ids).decode("utf-8", errors="replace")


def _scripted_sglang_client() -> SGLangClient:
    """Mock SGLang engine that replays ``_SCRIPT`` one completion per request."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        text, token_ids = _SCRIPT[min(calls["n"], len(_SCRIPT) - 1)]
        calls["n"] += 1
        return httpx.Response(
            200,
            json={
                "text": text,
                "meta_info": {
                    # descending, distinct logprobs so alignment is checkable
                    "output_token_logprobs": [
                        [-0.1 * (i + 1), tid] for i, tid in enumerate(token_ids)
                    ],
                    "finish_reason": {"type": "stop"},
                    "e2e_latency": 0.01,
                },
            },
        )

    http_client = httpx.Client(
        base_url="http://mock-sglang", transport=httpx.MockTransport(handler)
    )
    return SGLangClient("http://mock-sglang", http_client=http_client)


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="bugfix-calculator-sglang-smoke",
        prompt="subtract(5, 3) returns 8 instead of 2. Fix the bug and make the tests pass.",
        environment=EnvironmentRef(name="local_process", config={"source_dir": str(FIXTURE)}),
        verifier=VerifierRef(name="clean_sandbox_unit_tests"),
        budget=BudgetSpec(max_tool_calls={"read_file": 5, "edit_file": 5, "run_shell": 5}),
    )


class TestAgenticRolloutSmoke:
    def test_mocked_rollout_produces_trainable_sample(self) -> None:
        from trajectoryos.environments import LocalProcessSandbox

        task = _task()
        client = _scripted_sglang_client()
        policy = SGLangToolPolicy(
            client,
            _FakeTokenizer(),
            task,
            sampling_params={"temperature": 0.7, "max_new_tokens": 64},
        )

        sample = run_agentic_rollout(
            task,
            policy,
            sandbox_factory=lambda: LocalProcessSandbox(FIXTURE),
            test_command=TEST_COMMAND,
            reward_weights=RewardWeights(),
            run_id="smoke-run",
            policy_version="mock-sglang@m3",
        )
        client.close()

        # The bug was really fixed → clean-sandbox verification passed → reward 1.0.
        assert sample.status == "COMPLETED"
        assert sample.reward == 1.0
        assert sample.metadata["verifier"]["passed"] is True

        # Every "sampled" block survives verbatim, and only those tokens are trainable.
        sampled = [tid for _text, ids in _SCRIPT for tid in ids]
        trainable = [
            tid
            for tid, m in zip(
                sample.tokens[-sample.response_length :], sample.loss_mask, strict=True
            )
            if m == 1
        ]
        assert trainable == sampled

        # Ingested context (template + tool observations) sits in the response with mask 0.
        assert 0 in sample.loss_mask and 1 in sample.loss_mask

        # Alignment invariants (also enforced by SlimeSampleData, asserted for clarity).
        assert len(sample.loss_mask) == sample.response_length
        assert len(sample.rollout_log_probs) == sample.response_length
        # Trainable positions keep their exact (non-zero) sampled logprobs.
        for m, lp in zip(sample.loss_mask, sample.rollout_log_probs, strict=True):
            if m == 1:
                assert lp < 0.0

    def test_prompt_region_excluded_from_training(self) -> None:
        """No prompt-region token is ever marked trainable."""
        from trajectoryos.environments import LocalProcessSandbox

        task = _task()
        client = _scripted_sglang_client()
        policy = SGLangToolPolicy(client, _FakeTokenizer(), task)
        sample = run_agentic_rollout(
            task,
            policy,
            sandbox_factory=lambda: LocalProcessSandbox(FIXTURE),
            test_command=TEST_COMMAND,
            reward_weights=RewardWeights(),
            run_id="smoke-run-2",
            policy_version="mock-sglang@m3",
        )
        client.close()
        prompt_length = len(sample.tokens) - sample.response_length
        assert prompt_length > 0  # the system template + issue were ingested first
        # loss_mask only covers the response region, so prompt tokens are structurally
        # untrainable; confirm the response's first tokens are the first sampled block.
        first_sampled = _SCRIPT[0][1]
        response_tokens = sample.tokens[prompt_length:]
        assert response_tokens[: len(first_sampled)] == first_sampled
