"""SGLang client: token-in / token-out against mocked inference.

Uses ``httpx.MockTransport`` (explicitly labelled mock inference — no server)
to pin the wire contract: requests carry ``input_ids`` and ``return_logprob``,
and the exact sampled token IDs and rollout log-probs are read back from
``meta_info.output_token_logprobs`` without any re-tokenization.
"""

import json
from collections.abc import Callable

import httpx
import pytest
from trajectoryos.integrations.sglang import GenerateOutput, SGLangClient


def _mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> SGLangClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(base_url="http://mock-sglang", transport=transport)
    return SGLangClient("http://mock-sglang", http_client=http_client)


class TestSGLangClient:
    def test_sends_input_ids_and_returns_exact_tokens(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            captured["url"] = str(request.url)
            captured["input_ids"] = body["input_ids"]
            captured["return_logprob"] = body["return_logprob"]
            captured["sampling_params"] = body["sampling_params"]
            return httpx.Response(
                200,
                json={
                    "text": "<tool_call>{}</tool_call>",
                    "meta_info": {
                        # [logprob, token_id, (optional decoded piece)]
                        "output_token_logprobs": [
                            [-0.10, 501, "a"],
                            [-0.25, 502, "b"],
                            [-0.05, 503, "c"],
                        ],
                        "finish_reason": {"type": "stop"},
                        "e2e_latency": 0.5,
                    },
                },
            )

        client = _mock_client(handler)
        out = client.generate([1, 2, 3], {"temperature": 0.7, "max_new_tokens": 8})

        # The request is token-in and asks for logprobs.
        assert captured["url"] == "http://mock-sglang/generate"
        assert captured["input_ids"] == [1, 2, 3]
        assert captured["return_logprob"] is True
        assert captured["sampling_params"] == {"temperature": 0.7, "max_new_tokens": 8}

        # Sampled IDs and logprobs are read verbatim, aligned 1:1.
        assert isinstance(out, GenerateOutput)
        assert out.token_ids == [501, 502, 503]
        assert out.logprobs == [-0.10, -0.25, -0.05]
        assert out.text == "<tool_call>{}</tool_call>"
        assert out.finish_reason == "stop"
        assert out.latency_ms == pytest.approx(500.0)
        client.close()

    def test_empty_completion_is_valid(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"text": "", "meta_info": {"output_token_logprobs": []}},
            )

        out = _mock_client(handler).generate([7], {})
        assert out.token_ids == []
        assert out.logprobs == []
        assert out.finish_reason == "unknown"

    def test_http_error_propagates(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "engine oom"})

        with pytest.raises(httpx.HTTPStatusError):
            _mock_client(handler).generate([1], {})

    def test_token_logprob_length_mismatch_is_impossible(self) -> None:
        """token_ids and logprobs are decoded from the same pairs, so they align."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "text": "hi",
                    "meta_info": {"output_token_logprobs": [[-0.3, 42]]},
                },
            )

        out = _mock_client(handler).generate([1], {})
        assert len(out.token_ids) == len(out.logprobs) == 1
