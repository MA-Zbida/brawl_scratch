"""A recurrent reader of what one fighter is doing, from its crop over time (the agent's vision).

Trained on play where the fighter's key presses are known (the scripted player, brawl/scripted.py),
so the labels are free and exact in time:

- ``attacking``: light or heavy held in any of the 8 decisions (0.4 s) up to the one whose effect is now on
  the screen (a charged heavy stays "attacking" while held);
- ``dodging``: a dodge pressed in the 6 such decisions; ``jumping``: a jump pressed in the 4 such decisions;
- ``since_dodge``: decisions since the last dodge press, up to CLIP (4 s), scaled to 0-1;
- ``jumps``: jump presses since the fighter last stood on the ground, up to 4, scaled to 0-1.

What is on the screen at decision i is the effect of the actions chosen up to i - DELAY: measured on
jumps from the ground, the fighter is seen rising 3 decisions after the live agent chose the jump, and 2
after a human's recorded press (2026-11-13). Labels are shifted by that delay; without the shift they
describe the 0.2 s before a move shows.
A press is an attempt: pressed while stunned it shows nothing, so no reader can reach 1.0.
The network: a small CNN on the 96 px crop, with the fighter's own kinematics, into a GRU. Its five
outputs and its hidden state (HIDDEN numbers) are what a policy can be given.
"""
from __future__ import annotations

import numpy as np

from brawl import actions, observation

STATES = ("attacking", "dodging", "jumping")
COUNTS = ("since_dodge", "jumps")
OUTPUTS = (*STATES, *COUNTS)
KINEMATICS = ("x", "y", "vx", "vy", "present", "unarmed", "weapon_1", "weapon_2", "grounded", "offstage", "below",
              "to_left_ledge", "to_right_ledge")
HIDDEN = 128
DELAY = {"scripted": 3, "human": 2}  # decisions from choosing an action to seeing its effect, by who pressed
SIDES = {"wu_shang": ("agent", 0), "mordex": ("opponent", 1)}  # the fighter's entries in the vector, its crop's index


def kinematics(obs: np.ndarray, who: str) -> np.ndarray:
    side = SIDES[who][0]
    return obs[:, [observation.FEATURES.index(f"{side}_{n}") for n in KINEMATICS]].astype(np.float32)


def targets(action: np.ndarray, grounded: np.ndarray, delay: int) -> np.ndarray:
    """[N, 5] for the fighter whose actions these are; row i describes what is visible at decision i."""
    button = action[:, 1]
    names = actions.BUTTON_NAMES
    onset = lambda n: (button == names.index(n)) & (np.concatenate([[0], button[:-1]]) != names.index(n))  # noqa: E731
    before = lambda flag, back: np.array([flag[max(0, i - delay - back + 1):max(0, i - delay + 1)].any()  # noqa: E731
                                          for i in range(len(flag))])
    attack = np.isin(button, (names.index("light"), names.index("heavy")))
    since, jumps = np.zeros(len(button), np.float32), np.zeros(len(button), np.float32)
    last, n, dodge, jump = -observation.CLIP, 0, onset("dodge"), onset("jump")
    for i in range(len(button)):
        if grounded[i]:
            n = 0
        since[i], jumps[i] = min(observation.CLIP, i - last) / observation.CLIP, min(n, 4) / 4
        if dodge[i]:
            last = i
        n += int(jump[i])
    return np.stack([before(attack, 8), before(dodge, 6), before(jump, 4), since, jumps], 1).astype(np.float32)


def build(crop: bool = True, device: str = "cpu"):
    import torch
    from torch import nn

    class Reader(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(nn.Conv2d(1, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(), nn.Linear(64 * 8 * 8, 256),
                                     nn.ReLU()) if crop else None
            self.kin = nn.Sequential(nn.Linear(len(KINEMATICS), 64), nn.ReLU())
            self.gru = nn.GRU(64 + (256 if crop else 0), HIDDEN, batch_first=True)
            self.head = nn.Linear(HIDDEN, len(OUTPUTS))

        def forward(self, img, kin, h=None):
            """img: float [B, T, 96, 96] in 0-1 (ignored without crops); kin [B, T, 13]. Returns the outputs
            (logits for the three states, values for the two counts), the hidden states [B, T, HIDDEN], and h."""
            parts = [self.kin(kin)]
            if self.cnn is not None:
                b, t = img.shape[:2]
                parts.append(self.cnn(img.reshape(b * t, 1, *img.shape[-2:])).reshape(b, t, -1))
            z, h = self.gru(torch.cat(parts, -1), h)
            return self.head(z), z, h

    return Reader().to(device)


def read(model, crops, kin: np.ndarray, device: str, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Run a reader over one whole session, memory carried from its start. ``crops``: uint8 [N, 96, 96]
    (an array or a memory map). Returns the outputs [N, 5] (states as probabilities) and the hidden states."""
    import torch

    model.eval()
    out, hidden, h = [], [], None
    with torch.no_grad():
        for a in range(0, len(kin), chunk):
            img = torch.as_tensor(np.ascontiguousarray(crops[a:a + chunk]), device=device).float().div_(255)[None]
            o, z, h = model(img, torch.as_tensor(kin[a:a + chunk], device=device)[None], h)
            o = o[0]
            o[:, :len(STATES)] = torch.sigmoid(o[:, :len(STATES)])
            out.append(o.cpu().numpy())
            hidden.append(z[0].cpu().numpy())
    return np.concatenate(out), np.concatenate(hidden)
