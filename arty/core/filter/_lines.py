import numpy as np
from math import radians, cos, sin
import cv2


def lines(image: np.ndarray, angles: np.ndarray, edges: np.ndarray, thickness: int = 3,
          max_length: int = 10) -> np.ndarray:
    priority = [(edges[y, x], (x, y)) for y in range(len(edges)) for x in range(len(edges[y]))]
    priority.sort(reverse=True)
    new = np.zeros_like(image)
    for _, (x, y) in priority:
        if new[y, x].max() != 0: continue
        angle = radians(angles[y, x])
        dy, dx = int(cos(angle) * max_length / 2), int(sin(angle) * max_length / 2)
        cv2.line(new, (x - dx, y - dy), (x + dx, y + dy), image[y, x].tolist(), thickness)
    return new
