"""slime integration: the only place slime's extension interfaces are touched."""

from trajectoryos.integrations.slime.rollout import run_agentic_rollout
from trajectoryos.integrations.slime.sample import (
    SampleConversionError,
    SlimeSampleData,
    to_slime_sample,
    trajectory_to_sample_data,
)

__all__ = [
    "SampleConversionError",
    "SlimeSampleData",
    "run_agentic_rollout",
    "to_slime_sample",
    "trajectory_to_sample_data",
]
