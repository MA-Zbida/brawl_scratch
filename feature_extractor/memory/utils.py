import numpy as np
import math

from typing import Tuple

def clamp(v, lo, hi) -> float:
    return max(lo, min(v, hi))

def euclidian(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def bbox_center(det) -> Tuple[int, int]:
    x, y, w, h = det['bbox']
    # YOLO `xywhn` already provides center coordinates.
    # Do not add width/height offsets again.
    return (x, y)

def closest(detections, state):
    min_dist = math.inf
    min_x, min_y = 100, 100
    for det in detections:
        x, y = bbox_center(det)
        dist = euclidian((x, y), (state.x, state.y))
        if dist < min_dist:
            min_dist = dist
            min_x, min_y = x, y
    
    return min_x, min_y, min_dist




