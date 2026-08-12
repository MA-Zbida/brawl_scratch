"""Stocks and damage, decoded from fixed UI pixel coordinates.

Neither quantity is available from the detector, so both are read from calibrated
screen positions. Stock changes are debounced with a confirmation window, a
cooldown and a per-side lock, because the loss flash persists across frames and
would otherwise be counted several times.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from reward.extract_rgb import get_rgb
from reward.rgb_to_dmg import get_dmg
from reward.stock import get_stock


class PixelStocksHealthProvider:
    def __init__(
        self,
        ui_regions: dict,
        max_health: float = 351.0,
        self_stocks: float = 3.0,
        op_stocks: float = 3.0,
        stock_confirm_frames: int = 2,
        stock_event_cooldown_sec: float = 0.8,
        stock_event_lock_sec: float = 4.7,
    ):
        self.ui_regions = ui_regions
        self.max_health = max_health
        self._initial_self_stocks = self_stocks
        self._initial_op_stocks = op_stocks
        self.self_stocks_left = self_stocks
        self.op_stocks_left = op_stocks
        self._last_stock_signal = 0
        self._stable_stock_signal = 0
        self._stable_stock_frames = 0
        self._stock_confirm_frames = stock_confirm_frames
        self._stock_event_cooldown_sec = stock_event_cooldown_sec
        self._stock_event_lock_sec = float(max(0.0, stock_event_lock_sec))
        self._last_stock_event_time = 0.0
        self._neutral_frames = 0
        self._armed_for_event = True
        self._self_event_lock_until = 0.0
        self._op_event_lock_until = 0.0

    def reset(self, preserve_match_state: bool = True) -> None:
        if not bool(preserve_match_state):
            self.self_stocks_left = float(self._initial_self_stocks)
            self.op_stocks_left = float(self._initial_op_stocks)
        self._last_stock_signal = 0
        self._stable_stock_signal = 0
        self._stable_stock_frames = 0
        self._last_stock_event_time = 0.0
        self._neutral_frames = 0
        self._armed_for_event = True
        # Keep per-side stock event locks across env resets to avoid
        # re-counting the same red/cyan flash when episodes restart quickly.

    def _read_pixel(self, frame, coord: Tuple[int, int]) -> Optional[np.ndarray]:
        if frame is None:
            return None
        x, y = coord
        if y < 0 or x < 0 or y >= frame.shape[0] or x >= frame.shape[1]:
            return None
        return frame[y, x]

    def __call__(self, frame, detections):
        stock_coord = self.ui_regions.get("stock")
        op_coord = self.ui_regions.get("op")
        agent_coord = self.ui_regions.get("agent")

        if stock_coord is not None:
            stock_pixel = self._read_pixel(frame, stock_coord)
            if stock_pixel is not None:
                stock_rgb = np.asarray(get_rgb(stock_pixel), dtype=np.float32)
                stock_signal = int(get_stock(stock_rgb))
                if stock_signal == self._stable_stock_signal:
                    self._stable_stock_frames += 1
                else:
                    self._stable_stock_signal = stock_signal
                    self._stable_stock_frames = 1

                if stock_signal == 0:
                    self._neutral_frames += 1
                else:
                    self._neutral_frames = 0

                if self._neutral_frames >= max(1, int(self._stock_confirm_frames)):
                    self._armed_for_event = True

                now = time.perf_counter()
                stable_confirmed = self._stable_stock_frames >= max(1, int(self._stock_confirm_frames))
                cooldown_ready = (now - self._last_stock_event_time) >= float(self._stock_event_cooldown_sec)

                if stable_confirmed and stock_signal != 0 and stock_signal != self._last_stock_signal and cooldown_ready and self._armed_for_event:
                    accepted = False
                    if stock_signal < 0:
                        if now >= self._self_event_lock_until:
                            self.self_stocks_left = max(0.0, self.self_stocks_left - 1.0)
                            self._self_event_lock_until = now + self._stock_event_lock_sec
                            accepted = True
                    else:
                        if now >= self._op_event_lock_until:
                            self.op_stocks_left = max(0.0, self.op_stocks_left - 1.0)
                            self._op_event_lock_until = now + self._stock_event_lock_sec
                            accepted = True
                    if accepted:
                        self._last_stock_event_time = now
                    self._last_stock_signal = stock_signal
                    self._armed_for_event = False
                if stock_signal == 0:
                    self._last_stock_signal = 0
            else:
                self._last_stock_signal = 0
                self._stable_stock_signal = 0
                self._stable_stock_frames = 0
                self._neutral_frames = 0

        self_health = None
        op_health = None

        if agent_coord is not None:
            agent_pixel = self._read_pixel(frame, agent_coord)
            if agent_pixel is not None:
                agent_rgb = np.asarray(get_rgb(agent_pixel), dtype=np.float32)
                dmg = float(get_dmg(agent_rgb))
                self_health = max(0.0, self.max_health - dmg)

        if op_coord is not None:
            op_pixel = self._read_pixel(frame, op_coord)
            if op_pixel is not None:
                op_rgb = np.asarray(get_rgb(op_pixel), dtype=np.float32)
                dmg = float(get_dmg(op_rgb))
                op_health = max(0.0, self.max_health - dmg)

        return self.self_stocks_left, self.op_stocks_left, self_health, op_health
