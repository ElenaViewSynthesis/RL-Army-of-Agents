# SPEC: Autonomous Coding Agent Harness

## 1. Purpose

This project is a production-shaped autonomous coding agent harness. The goal
is not to build a better language model; the goal is to build the engineering
layer around a model so it can operate on a real repository through typed
tools, explicit planning, scoped subagents, durable context management,
observability, retries, rate limiting, tests, and evals.

The project should demonstrate how an AI coding system can remain coherent as
its tool surface grows beyond a trivial demo.

## 2. Core Requirements

The harness should eventually satisfy five properties:

1. **Large tool registry:** 50 or more tools across at least four namespaces,
   with model-driven tool selection.
2. **Real subagent:** an isolated subagent with fresh context, scoped tools,
   a separate budget, and a typed structured return.
3. **Long-horizon execution:** a 20+ tool-call task with context management
   implemented in code.
4. **Production scaffolding:** retries, rate limiting, typed errors,
   structured logs, tests, evals, configuration, and deployment affordances.
5. **Composable tool I/O:** structured output from one tool should feed
   structured input for another.

## 3. Planned Stack

| Layer | Choice | Reason |
| --- | --- | --- |
| Orchestration | LangGraph | Explicit state graph for plan, tool retrieval, acting, tool execution, and context management. |
| Tool transport | MCP over stdio | Keeps tools behind a protocol and allows namespace-based tool servers. |
| Tool retrieval | Embeddings + vector store | Avoids binding every tool schema on every model call. |
| Models | Configurable provider | Start with a single provider, but keep provider choice behind configuration. |
| Data validation | Pydantic | Typed tool inputs, outputs, subagent results, and errors. |
| Tests | pytest | Unit tests first, integration tests for live agent runs. |

## 4. Target Architecture

```text
autonomous-coding-agent-harness/
├── README.md
├── SPEC.md
├── CLAUDE.md
├── .env.example
├── Makefile
├── pyproject.toml
├── Dockerfile
├── src/
│   └── agent/
│       ├── graph/
│       ├── mcp_client/
│       ├── models/
│       ├── servers/
│       ├── retrieval/
│       ├── subagent/
│       ├── resilience/
│       ├── observability/
│       └── main.py
├── evals/
│   ├── retrieval/
│   └── e2e/
├── fixture_repo/
├── tests/
│   ├── unit/
│   └── integration/
└── tools/
```

## 5. Tool Namespaces

The planned tool registry should grow in stages:

- `fs`: read, write, search, grep, stat, copy, move, delete, and directory
  operations.
- `git`: status, diff, log, branch, checkout, commit, changed files, tags,
  blame, and stash.
- `ast`: parse modules, list symbols, find definitions, find references,
  inspect imports, and compute complexity.
- `test`: discover tests, run suites, run nodes, rerun failures, report
  coverage, and summarize failures.
- `deps`: list dependencies, detect unused dependencies, resolve imports,
  check outdated packages, and scan vulnerabilities.
- `ci`: run lint, format checks, type checks, build checks, pre-commit hooks,
  and quality summaries.

## 6. Retrieval Strategy

Once the registry is larger than the first few tools, the agent should not bind
every tool schema on each step. Instead:

1. Discover tools from MCP servers.
2. Build a registry with namespace, name, description, input schema, and
   output schema.
3. Embed tool descriptions and signatures.
4. Store embeddings in a local vector store.
5. Retrieve the top-k relevant tools for the current goal.
6. Bind only the retrieved subset to the model.
7. If the model requests a missing tool, widen retrieval and retry.

## 7. Subagent Contract

The first subagent should be a test-triage subagent. It should:

- Receive a task brief, not the parent transcript.
- Be restricted to test tools plus read-only file access.
- Have its own step and token budget.
- Return a typed result containing status, findings, artifacts, steps used,
  tokens used, and errors if any.

## 8. Context Management

The long-horizon path should include a context manager node that:

- Estimates message history size.
- Keeps the persistent plan outside the message list.
- Compacts older tool-call pairs into a deterministic progress ledger.
- Preserves unresolved work and recent tool outputs.

## 9. Production Scaffolding

Production hardening should include:

- Typed error hierarchy.
- Exponential backoff with jitter.
- Token-bucket rate limiting.
- Structured JSON logging with correlation IDs.
- Optional tracing through environment configuration.
- Unit tests for pure logic.
- Integration tests for MCP and live agent runs.
- Retrieval recall evals.
- E2E task-success evals.
- Dockerfile and Makefile targets.

## 10. Build Order

1. Create architecture docs and project contract.
2. Add minimal Python package, config, and one filesystem tool.
3. Add LangGraph loop and a smoke task.
4. Expand `fs` and `git` tools.
5. Add retrieval registry and recall eval.
6. Add remaining tool namespaces.
7. Add isolated subagent.
8. Add context compaction and a long-horizon fixture.
9. Add resilience, logging, Docker, and final documentation.

## 11. Initial Done Definition

This first slice is complete when the repository contains only the project
contract and contributor guidance for building the autonomous coding agent
harness in clean, incremental commits.
