"""Application settings using pydantic-settings."""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str | None = Field(default=None, env="ANTHROPIC_API_KEY")
    google_api_key: str | None = Field(default=None, env="GOOGLE_API_KEY")
    github_personal_access_token: str | None = Field(default=None, env="GITHUB_PERSONAL_ACCESS_TOKEN")
    database_url: str | None = Field(default=None, env="DATABASE_URL")
    repo_path: str | None = Field(default=None, env="REPO_PATH")

    main_model: str = Field(default="claude-sonnet-4-6", env="MAIN_MODEL")
    research_model: str = Field(default="google_genai:gemini-3.5-flash", env="RESEARCH_MODEL")
    code_model: str = Field(default="claude-sonnet-4-6", env="CODE_MODEL")
    db_model: str = Field(default="google_genai:gemini-3.5-flash", env="DB_MODEL")

    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
