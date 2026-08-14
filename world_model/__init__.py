"""World-model components.

Currently the persistent transition log only. The log is written now, before any
model exists, because real interaction at 30 Hz is the project's scarce resource
and data not captured today cannot be recovered later.
"""

from world_model.replay_log import (
    observation_interval,
    SCHEMA_VERSION,
    TERMINAL_ACTION,
    WorldReplayRecorder,
    WorldReplayWriter,
    iter_transitions,
    read_session,
)

__all__ = [
    "SCHEMA_VERSION",
    "TERMINAL_ACTION",
    "WorldReplayRecorder",
    "WorldReplayWriter",
    "iter_transitions",
    "observation_interval",
    "read_session",
]
