from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


def _xywh_to_xyxy(bbox_xywh: List[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in bbox_xywh[:4]]
    return x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0


def _iou(a_xywh: List[float], b_xywh: List[float]) -> float:
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(a_xywh)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(b_xywh)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return float(inter / union)


@dataclass
class _Track:
    track_id: int
    class_name: str
    bbox: List[float]
    confidence: float
    vx: float = 0.0
    vy: float = 0.0
    missing: int = 0


class SortLikeTracker:
    def __init__(
        self,
        max_missing: int = 8,
        iou_threshold: float = 0.1,
        smooth_alpha: float = 0.6,
    ):
        self.max_missing = max(1, int(max_missing))
        self.iou_threshold = float(iou_threshold)
        self.smooth_alpha = float(smooth_alpha)
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def _spawn(self, det: dict) -> int:
        track_id = self._next_id
        self._next_id += 1
        self._tracks[track_id] = _Track(
            track_id=track_id,
            class_name=str(det.get("class_name", "unknown")),
            bbox=[float(v) for v in det.get("bbox", [0.0, 0.0, 0.0, 0.0])[:4]],
            confidence=float(det.get("confidence", 0.0)),
        )
        return track_id

    def update(self, detections: List[dict]) -> List[dict]:
        detections = list(detections or [])
        unmatched_tracks = set(self._tracks.keys())
        unmatched_dets = set(range(len(detections)))
        assignments: List[Tuple[int, int]] = []

        candidate_pairs: List[Tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for det_idx, det in enumerate(detections):
                if str(det.get("class_name")) != track.class_name:
                    continue
                iou = _iou(track.bbox, det.get("bbox", [0.0, 0.0, 0.0, 0.0]))
                if iou >= self.iou_threshold:
                    candidate_pairs.append((iou, track_id, det_idx))

        candidate_pairs.sort(key=lambda t: t[0], reverse=True)
        for _, track_id, det_idx in candidate_pairs:
            if track_id not in unmatched_tracks or det_idx not in unmatched_dets:
                continue
            assignments.append((track_id, det_idx))
            unmatched_tracks.remove(track_id)
            unmatched_dets.remove(det_idx)

        for track_id, det_idx in assignments:
            track = self._tracks[track_id]
            det = detections[det_idx]
            det_bbox = [float(v) for v in det.get("bbox", [0.0, 0.0, 0.0, 0.0])[:4]]

            prev_x, prev_y = track.bbox[0], track.bbox[1]
            new_x = self.smooth_alpha * det_bbox[0] + (1.0 - self.smooth_alpha) * track.bbox[0]
            new_y = self.smooth_alpha * det_bbox[1] + (1.0 - self.smooth_alpha) * track.bbox[1]
            new_w = self.smooth_alpha * det_bbox[2] + (1.0 - self.smooth_alpha) * track.bbox[2]
            new_h = self.smooth_alpha * det_bbox[3] + (1.0 - self.smooth_alpha) * track.bbox[3]

            track.vx = float(new_x - prev_x)
            track.vy = float(new_y - prev_y)
            track.bbox = [float(new_x), float(new_y), float(new_w), float(new_h)]
            track.confidence = float(det.get("confidence", track.confidence))
            track.missing = 0

        for track_id in list(unmatched_tracks):
            track = self._tracks[track_id]
            track.missing += 1
            track.bbox[0] += track.vx
            track.bbox[1] += track.vy
            if track.missing > self.max_missing:
                del self._tracks[track_id]

        for det_idx in unmatched_dets:
            self._spawn(detections[det_idx])

        tracked = []
        for track_id, track in self._tracks.items():
            tracked.append(
                {
                    "class_name": track.class_name,
                    "bbox": [float(v) for v in track.bbox],
                    "confidence": float(track.confidence),
                    "track_id": int(track_id),
                    "vx": float(track.vx),
                    "vy": float(track.vy),
                }
            )
        return tracked
