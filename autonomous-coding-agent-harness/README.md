# Autonomous Coding Agent Harness

A production-oriented autonomous coding agent harness built in incremental
milestones. The project focuses on the engineering layer around an LLM:
orchestration, typed tools, retrieval over a growing tool registry, evals, and
eventual production scaffolding.

This folder is one independent project inside the larger repository.

## Current Status

Implemented so far:

- Architecture and build contract in `SPEC.md`
- Codex workflow guide in `CODEX.md`
- Minimal LangGraph-style agent spine
- Groq-backed model call path through `langchain-groq`
- MCP tool discovery over stdio
- Filesystem MCP namespace with 11 tools
- Git MCP namespace with 12 tools
- AST analysis MCP namespace with 9 tools
- Test runner MCP namespace with 8 tools
- Dependency MCP namespace with 7 tools
- CI and quality MCP namespace with 7 tools
- Typed pydantic input/output models for all six namespaces
- Semantic tool retrieval layer
- Deterministic local text embeddings for retrieval tests
- In-memory vector store abstraction
- Retrieval-miss widening path in the graph
- Retrieval recall eval scaffold
- Unit tests for model contracts, registry building, and retrieval logic

Planned next:

- Add `ast`, `test`, `deps`, and `ci` MCP namespaces
- Add isolated subagent execution
- Add long-horizon context compaction
- Add typed errors, retries, rate limiting, and structured logging
- Replace or extend local retrieval with sentence-transformers and pgvector
- Add Docker, full documentation, and e2e eval artifacts

## Project Layout

```text
autonomous-coding-agent-harness/
├── README.md
├── SPEC.md
├── CODEX.md
├── CLAUDE.md
├── .env.example
├── .gitignore
├── Makefile
├── pyproject.toml
├── requirements.txt
├── evals/
│   └── retrieval/
├── src/
│   └── agent/
│       ├── graph/
│       ├── mcp_client/
│       ├── models/
│       ├── retrieval/
│       ├── servers/
│       └── main.py
└── tests/
    └── unit/
```

## Implemented Tool Namespaces

### `fs`

Filesystem tools:

- `read_file`
- `read_file_range`
- `write_file`
- `list_dir`
- `search_files`
- `grep`
- `file_stat`
- `make_dir`
- `move`
- `delete`
- `copy`

### `git`

Git tools:

- `git_status`
- `git_diff`
- `git_log`
- `git_blame`
- `git_branch_create`
- `git_branch_list`
- `git_checkout`
- `git_commit`
- `git_stash`
- `git_show_commit`
- `git_list_changed_files`
- `git_tag`

### `ast`

Static-analysis tools:

- `parse_module`
- `list_symbols`
- `find_definition`
- `find_references`
- `list_imports`
- `compute_complexity`
- `detect_dead_code`
- `extract_function_signature`
- `find_unused_imports`

### `test`

Test-runner tools:

- `discover_tests`
- `run_test_file`
- `run_test_node`
- `run_suite`
- `coverage_report`
- `coverage_diff`
- `last_failures`
- `rerun_failed`

### `deps`

Dependency tools:

- `list_dependencies`
- `check_outdated`
- `resolve_import`
- `find_unused_deps`
- `dependency_graph`
- `vulnerability_scan`
- `add_dependency`

### `ci`

CI and quality tools:

- `run_linter`
- `run_formatter`
- `run_type_check`
- `build_check`
- `pre_commit_run`
- `run_security_scan`
- `summarize_quality`

## Retrieval Layer

The agent now builds a registry from discovered MCP tools and retrieves a
task-relevant subset before each model action. This prevents the graph from
binding every tool schema on every turn as the registry grows.

Current retrieval implementation:

- `ToolRegistryEntry` captures namespace, name, description, and input schema.
- `Embedder` creates deterministic local hash embeddings.
- `InMemoryVectorStore` stores local vectors for top-k search.
- `ToolRetriever` returns top-k tool names and always includes core tools.
- The graph widens retrieval if the model requests a tool outside the current
  subset.

The local embedding/vector-store path is intentionally lightweight so the
milestone can run without a database or model download. Dependencies for
`sentence-transformers`, `psycopg`, and `pgvector` are already declared for the
later production vector-store implementation.

## Setup

From this project folder:

```bash
pip install -e ".[dev]"
cp .env.example .env
```

Set `GROQ_API_KEY` in `.env` before running the live agent.

## Commands

```bash
make test
make eval
make run
```

Equivalent direct commands:

```bash
pytest tests/unit/ -v
python -m evals.retrieval.eval_recall
python -m agent.main
```

## Environment Variables

| Variable | Description |
| --- | --- |
| `AGENT_MODEL` | Groq model name. Defaults to `llama-3.1-8b-instant`. |
| `GROQ_API_KEY` | Required for live model calls. |

## Testing

Current tests cover:

- Filesystem pydantic contracts
- Git pydantic contracts
- AST, test, dependency, and CI pydantic contracts
- Tool registry construction
- Local retrieval behavior

The tests are designed to run without live model calls. The live agent path
requires dependencies plus a valid Groq API key.

## Development Notes

This project intentionally follows a commit-by-milestone history. Each
milestone should explain:

- What capability it adds
- Which files or folders changed
- How it can be tested
- What remains deferred

Use `CODEX.md` as the working guide for Codex-driven implementation.
