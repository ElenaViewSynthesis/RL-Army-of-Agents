"""Application settings using pydantic-settings."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: str | None = None
    github_personal_access_token: str | None = None
    tavily_api_key: str | None = None
    neo4j_uri: str | None = None
    neo4j_username: str | None = None
    neo4j_password: str | None = None
    neo4j_database: str | None = None
    neo4j_query_api_url: str | None = None
    database_url: str | None = None
    repo_path: str | None = None

    main_model: str = "google_genai:gemini-3.5-flash"
    research_model: str = "google_genai:gemini-3.5-flash"
    web_graph_model: str = "google_genai:gemini-3.5-flash"
    code_model: str = "google_genai:gemini-3.5-flash"
    db_model: str = "google_genai:gemini-3.5-flash"

    log_level: str = "INFO"
