"""Shared, Torch-free environment and future trainer configuration."""

import argparse
import math
from dataclasses import asdict, dataclass, field, fields


@dataclass
class RolloutConfig:
    backend: str = "auto"
    iters: int = 10
    timeout_s: float = 180
    provenance_threshold: float = 0.90
    speedup_margin: float = 0.05
    max_turns: int = 20
    max_context_tokens: int = 32768
    algo: str = "grpo"
    group_size: int = 8
    adv_norm: str = "std"
    filter_tied_groups: bool = False
    overlong_buffer_turns: int = 0
    clip_low: float = 0.2
    clip_high: float = 0.2
    loss_agg: str = "seq"
    kl_coef: float = 0.0
    seed: int = 0
    run_name: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        for name, choices in {"backend": ("auto", "cuda", "triton"),
                              "algo": ("grpo", "dapo"), "adv_norm": ("std", "none"),
                              "loss_agg": ("seq", "token")}.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {choices}")
        for name in ("iters", "max_turns", "max_context_tokens", "group_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("timeout_s", "provenance_threshold", "speedup_margin",
                     "clip_low", "clip_high", "kl_coef"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if not 0 <= self.provenance_threshold <= 1 or not 0 <= self.speedup_margin < 1:
            raise ValueError("Invalid provenance_threshold or speedup_margin")
        if (not isinstance(self.overlong_buffer_turns, int)
                or not 0 <= self.overlong_buffer_turns <= self.max_turns):
            raise ValueError("overlong_buffer_turns must be an integer in [0, max_turns]")
        if not 0 <= self.clip_low < 1 or self.clip_high < 0 or self.kl_coef < 0:
            raise ValueError("Invalid clip range or kl_coef")
        if not isinstance(self.filter_tied_groups, bool):
            raise ValueError("filter_tied_groups must be bool")
        if not isinstance(self.seed, int) or not isinstance(self.run_name, str):
            raise ValueError("seed must be int and run_name must be str")
        if not isinstance(self.tags, list) or not all(isinstance(t, str) for t in self.tags):
            raise ValueError("tags must be a list of strings")

    @classmethod
    def preset(cls, algo="grpo"):
        if algo == "grpo":
            return cls(algo=algo, kl_coef=0.001)
        if algo == "dapo":
            return cls(algo=algo, clip_high=0.28, filter_tied_groups=True,
                       loss_agg="token", overlong_buffer_turns=4)
        raise ValueError(f"Unknown algorithm: {algo}")

    @staticmethod
    def add_args(parser):
        """Suppress absent flags so from_args applies the preset before overrides."""
        choices = {"backend": ("auto", "cuda", "triton"), "algo": ("grpo", "dapo"),
                   "adv_norm": ("std", "none"), "loss_agg": ("seq", "token")}
        defaults = RolloutConfig()
        for f in fields(defaults):
            kwargs = {"default": argparse.SUPPRESS}
            if f.name == "filter_tied_groups":
                kwargs["action"] = argparse.BooleanOptionalAction
            elif f.name == "tags":
                kwargs.update(nargs="*", type=str)
            else:
                kwargs["type"] = type(getattr(defaults, f.name))
                if f.name == "timeout_s":
                    kwargs["type"] = float
                if f.name in choices:
                    kwargs["choices"] = choices[f.name]
            parser.add_argument("--" + f.name.replace("_", "-"), **kwargs)
        return parser

    @classmethod
    def from_args(cls, args):
        args = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
        values = cls.preset(args.get("algo", "grpo")).to_dict()
        values.update({f.name: args[f.name] for f in fields(cls) if f.name in args})
        return cls(**values)

    def to_dict(self):
        """Flat field map in declaration order; tags remain a JSON-safe list."""
        return asdict(self)


add_args = RolloutConfig.add_args
from_args = RolloutConfig.from_args
