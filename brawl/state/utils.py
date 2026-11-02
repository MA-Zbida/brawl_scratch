import math

from typing import Tuple

def clamp(v, lo, hi) -> float:
    return max(lo, min(v, hi))

def euclidian(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def bbox_center(det) -> Tuple[float, float]:
    x, y, w, h = det['bbox']

    return (float(x), float(y))

def bbox_size(det) -> Tuple[float, float]:
    """Normalised (width, height) of a detection box.

    The detector already produces these; using only the centre discards the
    silhouette, which is the closest thing to animation state available without
    reading game memory.
    """
    x, y, w, h = det['bbox']

    return (float(w), float(h))

def euclidean(a, b) -> float:
    return euclidian(a, b)

def closest(detections, state):
    min_dist = math.inf
    min_x, min_y = math.inf, math.inf

    for det in detections:
        x, y = bbox_center(det)
        dist = euclidian((x, y), (state.x, state.y))
        if dist < min_dist:
            min_dist = dist
            min_x, min_y = x, y
    
    return min_x, min_y, min_dist

def _nearest_ledge(px: float, y_stage: float, x_min: float, x_max: float) -> Tuple[float, float]:
    """Returns coordinates of current nearest ledge"""
    left: Tuple[float, float] = (x_min, y_stage)
    right: Tuple[float, float] = (x_max, y_stage)

    if abs(px - x_min) < abs(px - x_max):
        return left
    return right

