import pytest

from app.config.settings import Settings
from app.mcp.server_configs import build_mcp_server_configs


def test_build_mcp_server_configs_success(tmp_path):
    s = Settings(repo_path=str(tmp_path), github_personal_access_token="token", database_url="postgresql://x")
    configs = build_mcp_server_configs(s)
    assert "github" in configs
    assert "filesystem" in configs
    assert "postgres" in configs


def test_build_mcp_server_configs_missing():
    s = Settings()
    with pytest.raises(ValueError):
        build_mcp_server_configs(s)
