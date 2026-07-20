"""SGLang native-API client, token-in/token-out.

The single place where sampled token IDs enter TrajectoryOS. Requests use
``input_ids`` (never text) and ``return_logprob=True``; outputs are read from
``meta_info.output_token_logprobs`` — ``[[logprob, token_id, ...], ...]`` — so
the exact sampled IDs and their rollout log-probs are preserved. Decoded text
is carried along for tool-call parsing and logging only; it is never
re-tokenized into training targets.
"""

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, model_validator


class GenerateOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_ids: list[int]
    logprobs: list[float]
    text: str
    finish_reason: str
    latency_ms: float

    @model_validator(mode="after")
    def _aligned(self) -> "GenerateOutput":
        if len(self.token_ids) != len(self.logprobs):
            raise ValueError(
                f"token/logprob length mismatch: {len(self.token_ids)} != {len(self.logprobs)}"
            )
        return self


class SGLangClient:
    """Client for an SGLang server's native ``/generate`` endpoint.

    ``http_client`` is injectable so tests can use ``httpx.MockTransport``
    (explicitly labelled mock inference) without a running server.
    """

    def __init__(
        self,
        base_url: str,
        *,
        http_client: httpx.Client | None = None,
        timeout_seconds: float = 600.0,
    ) -> None:
        self._client = http_client or httpx.Client(base_url=base_url, timeout=timeout_seconds)

    def generate(self, input_ids: list[int], sampling_params: dict[str, Any]) -> GenerateOutput:
        response = self._client.post(
            "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": sampling_params,
                "return_logprob": True,
            },
        )
        response.raise_for_status()
        payload = response.json()
        meta = payload["meta_info"]
        token_logprobs: list[list[Any]] = meta["output_token_logprobs"]
        finish = meta.get("finish_reason") or {}
        return GenerateOutput(
            token_ids=[int(entry[1]) for entry in token_logprobs],
            logprobs=[float(entry[0]) for entry in token_logprobs],
            text=payload.get("text", ""),
            finish_reason=str(finish.get("type", "unknown")),
            latency_ms=float(meta.get("e2e_latency", 0.0)) * 1000,
        )

    def close(self) -> None:
        self._client.close()
