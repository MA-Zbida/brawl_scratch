"""Bounded, opt-in evaluation clips; timestamps, not movie FPS, define time."""
from __future__ import annotations

import json


class EvaluationClip:
    def __init__(self, directory, name, *, seconds, fps):
        self.directory, self.name, self.seconds, self.fps = directory, name, seconds, fps
        self.writer = self.stream = None
        self.count = 0

    def write(self, frame, *, elapsed, action, info):
        if elapsed > self.seconds:
            return
        import cv2
        from action_space import Action
        width = min(960, frame.shape[1])
        height = int(frame.shape[0] * width / frame.shape[1]) // 2 * 2
        width = width // 2 * 2
        image = cv2.resize(frame, (width, height))
        cv2.putText(image, f'{elapsed:.2f}s requested {Action(action).name}', (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        if self.writer is None:
            self.writer = cv2.VideoWriter(str(self.directory / f'{self.name}.mp4'),
                cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height))
            if not self.writer.isOpened():
                self.close()
                raise RuntimeError('Cannot open evaluation video encoder')
            self.stream = (self.directory / f'{self.name}.frames.jsonl').open('x', encoding='utf-8')
        self.writer.write(image)
        self.stream.write(json.dumps(dict(frame=self.count, elapsed_s=elapsed, requested_action=action,
            effective_action=info.get('effective_action', action),
            damage_dealt=info.get('op_delta_damage', 0.), damage_received=info.get('self_delta_damage', 0.))) + '\n')
        self.count += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.stream is not None:
            self.stream.close()
            self.stream = None
