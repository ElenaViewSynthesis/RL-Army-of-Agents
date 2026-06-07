# Autonomous Coding Agent Harness

This project is built in small, verifiable slices. Each slice should add one
clear capability, include tests where appropriate, and preserve a readable
commit history.

## Working Rules

- Keep the implementation grounded in the architecture in `SPEC.md`.
- Prefer typed inputs and outputs at module boundaries.
- Build depth-first: one working path first, then widen the tool surface.
- Do not add broad abstractions until at least two concrete call sites need
  them.
- Keep model behavior grounded in observed tool results.
- Treat retries, rate limits, structured logging, and tests as part of the
  product, not as polish.

## Slice Order

1. Architecture and build plan.
2. Minimal agent spine with one filesystem tool.
3. Filesystem and git tool namespaces.
4. Tool retrieval over a larger registry.
5. Additional MCP namespaces for code analysis, tests, dependencies, and CI.
6. Isolated subagent with scoped tools and typed return values.
7. Long-horizon execution with context compaction.
8. Production hardening, documentation, Docker, and eval artifacts.

## Quality Bar

A slice is complete only when the new behavior can be explained, run, and
tested independently. If a later slice reveals a design issue, fix it in the
smallest sensible scope and document the reason in the commit message.
