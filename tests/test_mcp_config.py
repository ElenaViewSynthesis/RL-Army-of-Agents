import pytest

from app.config.settings import Settings
from app.mcp.server_configs import build_mcp_server_configs


def test_build_mcp_server_configs_success(tmp_path):
    s = Settings(repo_path=str(tmp_path), github_personal_access_token="token", database_url="postgresql://x")
    configs = build_mcp_server_configs(s)
    assert "github" in configs
    assert "filesystem" in configs
    assert "postgres" in configs


def test_build_mcp_server_configs_missing(monkeypatch):
    monkeypatch.delenv("REPO_PATH", raising=False)
    monkeypatch.delenv("GITHUB_PERSONAL_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    s = Settings(_env_file=None)
    with pytest.raises(ValueError):
        build_mcp_server_configs(s)
