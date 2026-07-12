"""Config loading, deep-merge, env overrides and seeding tests."""

import random
from pathlib import Path

import pytest
from pydantic import ValidationError
from trajectoryos.core import (
    apply_env_overrides,
    deep_merge,
    derive_seed,
    load_config,
    load_yaml,
    seed_everything,
)
from trajectoryos.schemas import ModelSpec

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestLoadYaml:
    def test_loads_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("a: 1\nb:\n  c: true\n", encoding="utf-8")
        assert load_yaml(path) == {"a": 1, "b": {"c": True}}

    def test_empty_file_is_empty_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert load_yaml(path) == {}

    def test_non_mapping_top_level_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(TypeError, match="mapping"):
            load_yaml(path)


class TestDeepMerge:
    def test_nested_override_wins(self) -> None:
        base = {"a": {"x": 1, "y": 2}, "b": 1}
        override = {"a": {"y": 3}, "c": 4}
        assert deep_merge(base, override) == {"a": {"x": 1, "y": 3}, "b": 1, "c": 4}

    def test_non_mutating(self) -> None:
        base = {"a": {"x": 1}}
        deep_merge(base, {"a": {"x": 2}})
        assert base == {"a": {"x": 1}}


class TestEnvOverrides:
    def test_nested_path_and_yaml_typing(self) -> None:
        cfg = {"model": {"hf_repo": "Qwen/Qwen3-4B-Base", "is_moe": False}, "seed": 1}
        env = {
            "TRAJECTORYOS__model__is_moe": "true",
            "TRAJECTORYOS__seed": "42",
            "UNRELATED": "ignored",
        }
        result = apply_env_overrides(cfg, env=env)
        assert result["model"]["is_moe"] is True
        assert result["seed"] == 42
        assert result["model"]["hf_repo"] == "Qwen/Qwen3-4B-Base"

    def test_custom_prefix(self) -> None:
        result = apply_env_overrides({}, prefix="TOS", env={"TOS__a__b": "1.5"})
        assert result == {"a": {"b": 1.5}}


class TestLoadConfig:
    def test_loads_shipped_model_config(self) -> None:
        spec = load_config(
            REPO_ROOT / "configs" / "models" / "qwen3-4b-base.yaml", ModelSpec, env={}
        )
        assert spec.hf_repo == "Qwen/Qwen3-4B-Base"
        assert spec.is_moe is False
        assert spec.max_context_tokens == 32768

    def test_env_override_swaps_model(self, tmp_path: Path) -> None:
        """The model identifier stays configurable (MoE swap for GSPO, Milestone 5)."""
        spec = load_config(
            REPO_ROOT / "configs" / "models" / "qwen3-4b-base.yaml",
            ModelSpec,
            env={
                "TRAJECTORYOS__hf_repo": "Qwen/Qwen3-30B-A3B-Base",
                "TRAJECTORYOS__is_moe": "true",
            },
        )
        assert spec.hf_repo == "Qwen/Qwen3-30B-A3B-Base"
        assert spec.is_moe is True

    def test_unknown_key_fails_loudly(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("hf_repo: x\nmax_context_tokns: 1024\n", encoding="utf-8")
        with pytest.raises(ValidationError):
            load_config(path, ModelSpec, env={})

    def test_explicit_overrides_beat_env(self, tmp_path: Path) -> None:
        path = tmp_path / "m.yaml"
        path.write_text("hf_repo: from-file\n", encoding="utf-8")
        spec = load_config(
            path,
            ModelSpec,
            env={"TRAJECTORYOS__hf_repo": "from-env"},
            overrides={"hf_repo": "from-arg"},
        )
        assert spec.hf_repo == "from-arg"


class TestSeeding:
    def test_seed_everything_is_deterministic(self) -> None:
        seed_everything(1234)
        first = [random.random() for _ in range(5)]
        seed_everything(1234)
        second = [random.random() for _ in range(5)]
        assert first == second

    def test_derive_seed_stable_and_distinct(self) -> None:
        a1 = derive_seed(42, "task-001")
        a2 = derive_seed(42, "task-001")
        b = derive_seed(42, "task-002")
        assert a1 == a2
        assert a1 != b
