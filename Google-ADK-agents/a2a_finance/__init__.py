"""A2A finance layer.

Tier-A A2A: specialist agents run as independent A2A **services** (`to_a2a`),
and a coordinator delegates to them over HTTP via `RemoteA2aAgent` — instead of
in-process `sub_agents`. The valuation service's model is powered by the
OpenRouter **client SDK** (`OpenRouterLlm`). Future agents get added the same
way: define an agent, wrap it with `to_a2a`, and register a `RemoteA2aAgent` on
the coordinator.
"""
