import os

from app.config.settings import Settings


def test_settings_load(tmp_path, monkeypatch):
    monkeypatch.setenv("REPO_PATH", str(tmp_path))
    monkeypatch.setenv("GITHUB_PERSONAL_ACCESS_TOKEN", "ghp_test")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")

    s = Settings()
    assert s.repo_path == str(tmp_path)
    assert s.github_personal_access_token == "ghp_test"
    assert s.database_url.startswith("postgresql://")
