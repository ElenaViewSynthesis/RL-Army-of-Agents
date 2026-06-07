# Tavily Extract Skill Template

## Purpose

Use this skill when the research agent must extract page content from a known list of source URLs. The URLs below are placeholders and should be replaced with project-specific website URLs before running the extraction workflow.

This skill is for targeted extraction from specific pages, not broad web search. Use Tavily Search first only when the source URLs are unknown.

## Required Inputs

- `TAVILY_API_KEY` must be configured in `.env`.
- `urls` must contain one or more absolute website URLs.
- Each URL should point to a source that is expected to contain relevant page content for downstream research, GraphRAG ingestion, source tracing, or structured retrieval.

## Instantiation

The tool accepts various parameters during instantiation:

- `extract_depth` (optional, `str`): The depth of the extraction, either `basic` or `advanced`. Default is `basic`.
- `include_images` (optional, `bool`): Whether to include images in the extraction. Default is `False`.

For a comprehensive overview of the available parameters, refer to the Tavily Extract API documentation.

```python
from langchain_tavily import TavilyExtract

tool = TavilyExtract(
    extract_depth="basic",
    include_images=False,
)
```

## Placeholder URLs

Replace these placeholders with real project URLs:

```text
https://example.com/source-page-1
https://example.com/source-page-2
https://example.com/source-page-3
```

## Tool Call Pattern

Call the Tavily extract tool with the known URLs:

```json
{
  "urls": [
    "https://example.com/source-page-1",
    "https://example.com/source-page-2",
    "https://example.com/source-page-3"
  ],
  "extract_depth": "advanced",
  "include_images": false
}
```

If the tool interface only accepts the URL list, use:

```json
{
  "urls": [
    "https://example.com/source-page-1",
    "https://example.com/source-page-2",
    "https://example.com/source-page-3"
  ]
}
```

## Agent Instructions

1. Read the target URL list from the task input or project configuration.
2. Remove duplicate URLs before calling Tavily Extract.
3. Keep the original URL with every extracted result.
4. Extract the main page content, title, and available metadata.
5. Do not invent missing content. If extraction fails for a URL, record the failure and continue with the remaining URLs.
6. Return structured records that are ready for graph ingestion.

## Expected Output Shape

Return a list of extracted source records:

```json
[
  {
    "url": "https://example.com/source-page-1",
    "title": "Page title",
    "content": "Extracted page content...",
    "metadata": {
      "source_type": "web_page",
      "provider": "tavily_extract"
    },
    "extraction_status": "success"
  },
  {
    "url": "https://example.com/source-page-2",
    "title": null,
    "content": null,
    "metadata": {
      "source_type": "web_page",
      "provider": "tavily_extract"
    },
    "extraction_status": "failed",
    "error": "Extraction error or unavailable page"
  }
]
```

## Graph Ingestion Mapping

For each successful extraction, create or update graph data using this mapping:

- `Source` node:
  - `url`
  - `title`
  - `provider = "tavily_extract"`
  - `retrieved_at`
- `DocumentChunk` nodes:
  - `chunk_id`
  - `source_url`
  - `text`
  - `chunk_index`
- Relationships:
  - `(Source)-[:HAS_CHUNK]->(DocumentChunk)`
  - `(DocumentChunk)-[:MENTIONS]->(Entity)` when entity extraction is available

## Quality Checks

- Every extracted chunk must retain its source URL.
- Failed URLs must be included in the final report.
- Do not persist placeholder URLs to production graph storage.
- Do not treat extracted content as verified truth; preserve attribution and source metadata.

## Example Research Agent Prompt

```text
Use Tavily Extract on the provided URLs. Preserve each source URL, extract page titles and main content, and return structured records suitable for Neo4j graph ingestion. If a URL fails, include a failed extraction record with the URL and error.
```
