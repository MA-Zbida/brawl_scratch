"""World-model components.

Currently the persistent transition log only. Entry points attach it explicitly
for world-model data sessions because real interaction at 30 Hz is scarce, while
ordinary policy runs avoid its compression and disk-write cost.
"""

from brawl.world_model.replay_log import (
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
