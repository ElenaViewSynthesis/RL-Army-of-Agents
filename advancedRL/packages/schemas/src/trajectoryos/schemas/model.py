"""Model specification: the only way code refers to a concrete model.

Keeping the identifier in config (``configs/models/*.yaml``) lets the dense
Qwen3-4B-Base baseline be swapped for a Qwen3 MoE checkpoint (GSPO experiments)
without code changes.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ModelFamily(StrEnum):
    QWEN3 = "qwen3"
    OTHER = "other"


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    hf_repo: str = Field(min_length=1, description="Hugging Face repo id or local path.")
    family: ModelFamily = ModelFamily.QWEN3
    is_moe: bool = False
    max_context_tokens: int = Field(default=32768, ge=1024)
    dtype: str = "bfloat16"
    chat_template: str | None = Field(
        default=None,
        description="Override chat template name; None uses the tokenizer's built-in template.",
    )
