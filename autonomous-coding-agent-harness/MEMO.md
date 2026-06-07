# MEMO: Autonomous Coding Agent Harness

## What Was Built

This project is a production-shaped autonomous coding agent harness. It is not
a custom model; it is the engineering layer around a model: graph
orchestration, MCP tool servers, typed tool contracts, semantic retrieval over
a large tool registry, isolated subagents, context compaction, resilience,
evaluation scaffolds, and deployment support.

The current implementation includes six MCP namespaces:

- `fs`: 11 filesystem tools
- `git`: 12 version-control tools
- `ast`: 9 Python static-analysis tools
- `test`: 8 pytest-oriented tools
- `deps`: 7 dependency-inspection tools
- `ci`: 7 local quality/check tools

That creates a 54-tool registry, plus the parent-facing `spawn_subagent` tool.

## Architecture

The parent agent follows this graph shape:

```text
plan -> retrieve -> act -> tools -> manage_context -> act
                 \-> widen -> retrieve
```

The graph keeps the task plan in state, not only in chat history. Tool calls
are executed through LangGraph `ToolNode`, and long-running sessions pass
through `manage_context` after every tool execution.

## Retrieval Design

The retrieval layer exists so the model does not need every tool schema on
every step. The agent discovers tools from MCP servers, builds
`ToolRegistryEntry` records, embeds registry text, stores vectors in either
`InMemoryVectorStore` or `PgVectorStore`, retrieves a top-k subset for the
current goal, and widens retrieval if the model asks for a tool outside the
subset.

The default local path uses deterministic hash embeddings and an in-memory
store for repeatable development. When `DATABASE_URL` is set, the agent uses
LangChain's `langchain-postgres` path: `PGEngine` creates the connection pool,
`init_vectorstore_table` initializes the table, `PGVectorStore.create_sync`
opens the store, `add_texts` upserts registry text, and
`similarity_search_by_vector` retrieves persisted tool embeddings.

## Subagent Design

The project includes a real isolated test-triage subagent implemented with
LangChain `create_agent`. It receives a fresh task brief instead of the parent
transcript, sees only tools allowed by `NamespaceScope`, has its own
`SubagentBudget`, and returns a typed `SubagentResult`.

The default scope is intentionally narrow: the `test` namespace plus
`fs.read_file`. This allows failure diagnosis without write or git access.

## Context Strategy

The `manage_context` node estimates token use by message content length. Once
the estimate crosses `CONTEXT_COMPACT_THRESHOLD`, it compacts older completed
tool-call pairs into a deterministic progress ledger and preserves recent tool
outputs verbatim.

This keeps long-horizon runs coherent without making another model call just to
summarize context.

## Production Scaffolding

Implemented scaffolding includes:

- Typed runtime errors
- Retry with exponential backoff
- Provider-error normalization for rate-limit and request-size failures
- Token-bucket rate limiting
- Structured JSON logging
- Unit tests
- Integration-test scaffolds
- Retrieval recall eval
- Dockerfile and Docker-oriented Makefile targets

## What Was Deferred

Deferred work includes:

- Replacing the deterministic hash embedder with sentence-transformer
  embeddings in the live path.
- Running and recording LangChain PGVectorStore recall metrics against a real
  database.
- Adding a fully live long-horizon integration test that invokes the model and
  commits changes.
- Adding active LangSmith or OpenTelemetry spans around graph nodes.
- Adding parallel subagents and aggregate budget management.
- Hardening filesystem and git tools with stricter workspace sandboxing.

## Main Design Decision

The key decision is semantic retrieval over a flat all-tools prompt. Passing
54+ schemas to the model on every turn is simple, but it scales poorly: it
burns tokens, worsens attention over similar tools, and does not demonstrate a
coherent tool-selection layer. Retrieval makes the registry explicit,
measurable, and replaceable.

The current implementation keeps the contract stable:

```text
goal -> retriever -> tool names -> model-bound subset
```

That allows the backing store to evolve from local memory to pgvector without
rewriting the graph.

## How To Verify

Local checks:

```bash
make test
make eval
```

Integration checks:

```bash
make test-integration
```

Docker:

```bash
make docker-build
make docker-run
```

Live agent runs use Gemini by default and require `GOOGLE_API_KEY`. Groq remains
available as a secondary provider with `AGENT_MODEL_PROVIDER=groq` and
`GROQ_API_KEY`. pgvector retrieval requires `DATABASE_URL` pointing at
PostgreSQL with the pgvector extension available; plain PostgreSQL URLs are
normalized to the asyncpg driver required by LangChain's `PGEngine`.
