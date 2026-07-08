"""An ADK model backed by the OpenRouter **client SDK** (`openrouter` package).

This replaces the LiteLLM path for OpenRouter agents: instead of routing through
litellm, it calls `OpenRouter().chat.send_async(...)` directly and maps between
ADK's genai request/response types and OpenRouter's OpenAI-compatible schema.

Only non-streaming generation is implemented (one `LlmResponse` per turn) —
enough for the agentic tool loop and A2A. Tool calls are supported: genai
function declarations -> OpenAI `tools`, and OpenRouter `tool_calls` -> genai
`function_call` parts, so ADK's flow engine drives the loop normally.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.lite_llm import _schema_to_dict  # reuse ADK's schema converter
from google.genai import types
from openrouter import OpenRouter


def _content_text(content: types.Content) -> str:
    return " ".join(p.text for p in (content.parts or []) if getattr(p, "text", None))


def _tools_from_request(llm_request: LlmRequest) -> list[dict] | None:
    """genai function declarations -> OpenAI-style `tools`."""
    cfg = llm_request.config
    if not cfg or not cfg.tools:
        return None
    out: list[dict] = []
    for tool in cfg.tools:
        for fd in getattr(tool, "function_declarations", None) or []:
            params = _schema_to_dict(fd.parameters) if fd.parameters else {"type": "object", "properties": {}}
            out.append({
                "type": "function",
                "function": {"name": fd.name, "description": fd.description or "", "parameters": params},
            })
    return out or None


def _messages_from_request(llm_request: LlmRequest) -> list[dict]:
    """genai contents (+ system instruction) -> OpenAI-style messages."""
    messages: list[dict] = []
    cfg = llm_request.config
    sys_inst = getattr(cfg, "system_instruction", None) if cfg else None
    if sys_inst:
        text = sys_inst if isinstance(sys_inst, str) else _content_text(sys_inst)
        if text:
            messages.append({"role": "system", "content": text})

    for content in llm_request.contents or []:
        role = "assistant" if content.role == "model" else "user"
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_msgs: list[dict] = []  # function_response -> separate 'tool' messages
        for part in content.parts or []:
            if getattr(part, "thought", None):
                # Reasoning traces are not resent as content on later turns.
                continue
            if getattr(part, "text", None):
                text_parts.append(part.text)
            elif getattr(part, "function_call", None):
                fc = part.function_call
                tool_calls.append({
                    "id": fc.id or fc.name,
                    "type": "function",
                    "function": {"name": fc.name, "arguments": json.dumps(fc.args or {})},
                })
            elif getattr(part, "function_response", None):
                fr = part.function_response
                body = fr.response if isinstance(fr.response, str) else json.dumps(fr.response)
                tool_msgs.append({"role": "tool", "tool_call_id": fr.id or fr.name, "content": body})

        if tool_calls:
            messages.append({"role": "assistant", "content": " ".join(text_parts) or None, "tool_calls": tool_calls})
        elif text_parts:
            messages.append({"role": role, "content": " ".join(text_parts)})
        messages.extend(tool_msgs)
    return messages


class OpenRouterLlm(BaseLlm):
    """ADK model that talks to OpenRouter via its official client SDK."""

    max_tokens: int = 2000
    # When True, request reasoning and surface it as a separate genai *thought*
    # part (thought=True) instead of letting it bleed into the answer text.
    reasoning: bool = True

    @classmethod
    def supported_models(cls) -> list[str]:
        # Instantiated directly (model=<openrouter id>), not resolved by registry.
        return []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))
        cfg = llm_request.config
        max_tokens = getattr(cfg, "max_output_tokens", None) or self.max_tokens

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_from_request(llm_request),
            "max_tokens": max_tokens,
        }
        tools = _tools_from_request(llm_request)
        if tools:
            kwargs["tools"] = tools
        if self.reasoning:
            # Ask OpenRouter to return the reasoning trace in a dedicated field.
            kwargs["reasoning"] = {"enabled": True}

        resp = await client.chat.send_async(**kwargs)
        msg = resp.choices[0].message

        parts: list[types.Part] = []
        # Reasoning first, as a genai *thought* part — kept out of the answer text
        # so the flow engine / callers can distinguish thinking from the response.
        reasoning_text = getattr(msg, "reasoning", None)
        if reasoning_text:
            parts.append(types.Part(text=reasoning_text, thought=True))
        if getattr(msg, "content", None):
            parts.append(types.Part(text=msg.content))
        for tc in getattr(msg, "tool_calls", None) or []:
            raw = tc.function.arguments
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except (ValueError, TypeError):
                args = {}
            parts.append(types.Part(function_call=types.FunctionCall(id=tc.id, name=tc.function.name, args=args)))

        usage = getattr(resp, "usage", None)
        usage_md = None
        if usage is not None:
            usage_md = types.GenerateContentResponseUsageMetadata(
                prompt_token_count=getattr(usage, "prompt_tokens", None),
                candidates_token_count=getattr(usage, "completion_tokens", None),
                total_token_count=getattr(usage, "total_tokens", None),
            )

        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            usage_metadata=usage_md,
            turn_complete=True,
        )
